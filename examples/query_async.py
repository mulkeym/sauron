#!/usr/bin/env python3
"""Query the SAURON API asynchronously and watch the processing progression.

This example walks the full async query flow:

  1. POST /api/v1/auth/token        -> mint a JWT for a username + ACL groups
  2. POST /api/v1/query/async       -> submit a question, get back a poll token
  3. GET  /api/v1/query/async/{tok} -> poll until done, printing each step as the
                                       agent moves through its pipeline
  4. print the final answer + citations

Every protected endpoint needs BOTH credentials:
  - X-API-Key:      a configured API key (default deployment uses "dev-key-1")
  - Authorization:  "Bearer <JWT>" minted from /auth/token

The JWT's `groups` control document-level access (ACL). A question only sees
documents whose ACL groups intersect the caller's groups, so pass --groups to
match the datasets you want to search.

Usage:
  python examples/query_async.py "What is the GS pay rate in Florida?"
  python examples/query_async.py "..." --groups executives admin
  python examples/query_async.py "..." --base-url http://localhost:8880 --api-key dev-key-1

Requires: requests  (already a project dependency)
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

# The agent emits these node names as it runs; the API maps them to the
# human-readable labels shown below. Listed here only to document the expected
# progression — the script prints whatever the server reports.
EXPECTED_STEPS = [
    "queued",
    "checking cache",
    "classifying question",
    "retrieving documents",
    "searching knowledge graph",
    "merging results",
    "synthesizing answer",
    "complete",
]


def login(base_url: str, username: str, groups: list[str], timeout: float) -> str:
    """Mint a JWT. The dev login accepts any password and encodes username+groups."""
    resp = requests.post(
        f"{base_url}/api/v1/auth/token",
        json={"username": username, "password": "example", "groups": groups},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def submit_async(base_url: str, headers: dict, question: str, timeout: float) -> str:
    """Submit the question for async processing; return the poll token."""
    resp = requests.post(
        f"{base_url}/api/v1/query/async",
        headers=headers,
        json={"question": question},
        timeout=timeout,
    )
    if resp.status_code == 503:
        sys.exit("Query queue is full; try again shortly.")
    resp.raise_for_status()
    data = resp.json()
    return data["token"]


def poll_until_done(
    base_url: str,
    headers: dict,
    token: str,
    interval: float,
    timeout: float,
    overall_deadline: float,
) -> dict:
    """Poll the status endpoint, printing each new step, until complete/failed."""
    start = time.monotonic()
    printed = 0          # how many timeline entries we've already shown
    last_step = None
    classification_shown = False

    while True:
        resp = requests.get(
            f"{base_url}/api/v1/query/async/{token}",
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        job = resp.json()

        # Prefer the server-side step timeline (each entry carries its own elapsed
        # time, and it captures sub-steps that flicker by between polls). Fall back
        # to the single 'step' label if an older server doesn't send a timeline.
        steps = job.get("steps") or []
        if steps:
            for entry in steps[printed:]:
                print(f"  [{entry.get('at', 0):5.1f}s] {entry.get('step', '')}")
            printed = len(steps)
        else:
            step = job.get("step", "")
            if step != last_step:
                print(f"  [{time.monotonic() - start:5.1f}s] {step}")
                last_step = step

        # Print the classify decision once, as soon as it's available.
        if not classification_shown and job.get("classification"):
            _print_classification(job["classification"])
            classification_shown = True

        status = job.get("status")
        if status in ("complete", "failed"):
            return job

        if time.monotonic() - start > overall_deadline:
            sys.exit(f"Timed out after {overall_deadline:.0f}s (last status: {status}).")

        time.sleep(interval)


def _print_classification(c: dict) -> None:
    """Render the classify step's decision: query type + reason, sub-tasks, and
    the strategy-memory routing decision (the work that happens during the
    otherwise-opaque 'classifying question' step)."""
    qt = c.get("query_type", "?")
    reason = c.get("reason") or ""
    print(f"     ↳ classified as {qt}" + (f' — "{reason}"' if reason else ""))
    sub_tasks = [t for t in (c.get("sub_tasks") or [])]
    if sub_tasks:
        print(f"       sub-tasks: {sub_tasks}")
    sm = c.get("strategy_memory") or {}
    if sm and sm.get("reason") not in (None, "disabled"):
        if sm.get("overrode"):
            print(f"       strategy memory: OVERRODE {sm.get('llm_pick')} → {sm.get('memory_best')} "
                  f"(n={sm.get('count')}, margin={float(sm.get('margin', 0)) * 100:.0f}%)")
        else:
            print(f"       strategy memory: kept {sm.get('llm_pick')} ({sm.get('reason')})")


def print_result(job: dict) -> None:
    """Render the final answer and citations."""
    print("\n" + "=" * 70)
    if job.get("status") == "failed":
        print("QUERY FAILED")
        print(f"  error: {job.get('error')}")
        return

    if job.get("cached"):
        cq = job.get("cached_query")
        print(f"(served from cache{f' — matched: {cq!r}' if cq else ''})\n")

    print("ANSWER\n")
    print(job.get("answer") or "(no answer returned)")

    citations = job.get("citations") or []
    if citations:
        print(f"\nCITATIONS ({len(citations)})\n")
        for i, c in enumerate(citations, 1):
            page = f" p.{c['page']}" if c.get("page") is not None else ""
            print(f"  [{i}] {c.get('filename')}{page}  (relevance {c.get('relevance', 0):.3f})")
            snippet = (c.get("snippet") or "").strip().replace("\n", " ")
            if snippet:
                print(f"      {snippet[:160]}{'…' if len(snippet) > 160 else ''}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the SAURON API asynchronously and show processing progression.",
    )
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument("--base-url", default="http://localhost:8880", help="API base URL.")
    parser.add_argument("--api-key", default="dev-key-1", help="X-API-Key value.")
    parser.add_argument("--username", default="example-user", help="Username for the JWT.")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=[],
        help="ACL groups for the JWT (controls which documents are searchable).",
    )
    parser.add_argument("--poll-interval", type=float, default=0.5, help="Seconds between polls.")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="Per-request timeout (s).")
    parser.add_argument("--max-wait", type=float, default=300.0, help="Overall wait deadline (s).")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    print(f"→ authenticating as {args.username!r} (groups: {args.groups or 'none'})")
    jwt = login(base_url, args.username, args.groups, args.request_timeout)
    headers = {"X-API-Key": args.api_key, "Authorization": f"Bearer {jwt}"}

    print(f"→ submitting query: {args.question!r}")
    token = submit_async(base_url, headers, args.question, args.request_timeout)
    print(f"  token: {token}\n→ processing:")

    job = poll_until_done(
        base_url, headers, token,
        interval=args.poll_interval,
        timeout=args.request_timeout,
        overall_deadline=args.max_wait,
    )
    print_result(job)


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        sys.exit(f"HTTP error: {e}  body={e.response.text if e.response is not None else ''}")
    except requests.RequestException as e:
        sys.exit(f"Request failed: {e}")
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
