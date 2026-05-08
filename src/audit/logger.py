# src/audit/logger.py
import json
import time
from dataclasses import dataclass, field, asdict

@dataclass
class AuditEntry:
    agent_id: str = ""
    username: str = ""
    tool: str = ""
    query: str = ""
    documents_returned: list[str] = field(default_factory=list)
    retrieval_strategy: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

class AuditLogger:
    def __init__(self, log_path: str = "data/audit.jsonl"):
        self._log_path = log_path

    def log(self, entry: AuditEntry) -> None:
        with open(self._log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
