# tests/test_audit/test_logger.py
import json
import pytest
from src.audit.logger import AuditLogger, AuditEntry

def test_log_entry(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=str(log_file))
    logger.log(AuditEntry(agent_id="hr-agent", username="mike", tool="ask", query="What is the PTO policy?", documents_returned=["doc-1", "doc-2"], retrieval_strategy="lookup"))
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["agent_id"] == "hr-agent"
    assert entry["username"] == "mike"
    assert "timestamp" in entry

def test_multiple_entries(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=str(log_file))
    for i in range(3):
        logger.log(AuditEntry(agent_id="agent", username="user", tool="search", query=f"query {i}"))
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 3
