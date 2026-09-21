"""Durable drafts and immutable published revisions on the existing data volume.

SQLite transactions serialize edits across processes; a book version prevents
stale admin tabs from overwriting a newer draft or publication.
"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4
from contextlib import closing

from src.agent.profiles import AnswerProfile

PROFILE_PATH = Path("data/answer_profiles.sqlite3")


class ProfileConflict(ValueError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def default_book():
    presets = {
        "general": AnswerProfile(name="General guidance", description="Questions grounded in team documentation.",
            instructions="Answer directly and explain technical terms when needed."),
        "deployment": AnswerProfile(name="Deployment guidance", description="Documented SD-WAN deployment procedures.",
            instructions="Use the team's documented deployment procedure. Organize supported steps into prerequisites, execution, verification, and rollback. Preserve exact commands. Never invent an undocumented step.",
            retrieval_depth="thorough", insufficient_evidence="abstain"),
        "troubleshooting": AnswerProfile(name="Technical troubleshooting", description="Diagnose SD-WAN issues using documented evidence.",
            instructions="Separate observed symptoms, documented possible causes, diagnostic checks, and corrective actions. Explain what each documented check establishes. Do not present a possible cause as a confirmed diagnosis.",
            retrieval_depth="thorough"),
    }
    return {"version": 0, "active": {"profile_id": "general", "revision": 1},
            "profiles": {key: {"draft": value.model_dump(), "revisions": [
                {"revision": 1, "config": value.model_dump(), "created_at": "Built in", "author": "Sauron"}
            ]} for key, value in presets.items()}, "activations": []}


def read_book():
    if not PROFILE_PATH.exists():
        return default_book()
    # Read-only requests do not create files or migrate a store.
    with closing(sqlite3.connect(PROFILE_PATH.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT body FROM profile_state WHERE id = 1").fetchone()
    if row is None:
        raise ValueError("The answer profile store has no active configuration.")
    return json.loads(row[0])


def _edit(expected_version, operation):
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    creating = not PROFILE_PATH.exists()
    with closing(sqlite3.connect(PROFILE_PATH, timeout=10)) as db, db:
        if creating:
            PROFILE_PATH.chmod(0o600)
        db.execute("CREATE TABLE IF NOT EXISTS profile_state (id INTEGER PRIMARY KEY CHECK(id = 1), body TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO profile_state VALUES (1, ?)", (json.dumps(default_book()),))
        db.commit()
        db.execute("BEGIN IMMEDIATE")
        book = json.loads(db.execute("SELECT body FROM profile_state WHERE id = 1").fetchone()[0])
        if book["version"] != expected_version:
            raise ProfileConflict("Profiles changed in another session. Reload before saving or publishing.")
        operation(book)
        book["version"] += 1
        db.execute("UPDATE profile_state SET body = ? WHERE id = 1", (json.dumps(book),))
    return book


def _profile(book, profile_id):
    if profile_id not in book["profiles"]:
        raise KeyError("Profile not found")
    return book["profiles"][profile_id]


def create_profile(config, expected_version):
    value = AnswerProfile.model_validate(config).model_dump()
    profile_id = uuid4().hex
    def create(book):
        book["profiles"][profile_id] = {"draft": value, "revisions": []}
    return _edit(expected_version, create), profile_id


def save_draft(profile_id, config, expected_version):
    value = AnswerProfile.model_validate(config).model_dump()
    def save(book):
        _profile(book, profile_id)["draft"] = value
    return _edit(expected_version, save)


def publish(profile_id, expected_version, author):
    def apply(book):
        profile = _profile(book, profile_id)
        value = AnswerProfile.model_validate(profile["draft"]).model_dump()
        revision = max((r["revision"] for r in profile["revisions"]), default=0) + 1
        profile["revisions"].append({"revision": revision, "config": value, "created_at": _now(), "author": author})
        book["active"] = {"profile_id": profile_id, "revision": revision}
        book["activations"].append({**book["active"], "action": "publish", "created_at": _now(), "author": author})
    return _edit(expected_version, apply)


def activate_revision(profile_id, revision, expected_version, author):
    def apply(book):
        profile = _profile(book, profile_id)
        if not any(r["revision"] == revision for r in profile["revisions"]):
            raise KeyError("Published revision not found")
        book["active"] = {"profile_id": profile_id, "revision": revision}
        book["activations"].append({**book["active"], "action": "activate", "created_at": _now(), "author": author})
    return _edit(expected_version, apply)
