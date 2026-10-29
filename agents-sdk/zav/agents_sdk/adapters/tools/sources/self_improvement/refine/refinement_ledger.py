import json
from pathlib import Path
from typing import Any, Dict, List


class RefinementLedger:
    """Append-only record of every refinement edit: what triggered it, what it
    was meant to change and whether it was applied. One JSON object per line."""

    def __init__(self, path: Path) -> None:
        self.__path = path

    def append(self, record: Dict[str, Any]) -> None:
        self.__path.parent.mkdir(parents=True, exist_ok=True)
        with self.__path.open("a") as ledger:
            ledger.write(json.dumps(record, default=str) + "\n")

    def read(self) -> List[Dict[str, Any]]:
        if not self.__path.exists():
            return []
        return [
            json.loads(line)
            for line in self.__path.read_text().splitlines()
            if line.strip()
        ]
