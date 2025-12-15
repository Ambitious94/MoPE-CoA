from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    store_path: Path = Path(".mope_store.json")


def load_config(path: Optional[Path]) -> AppConfig:
    if path is None:
        return AppConfig()
    if not path.exists():
        return AppConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AppConfig()
    store_path = Path(data.get("store_path", ".mope_store.json"))
    return AppConfig(store_path=store_path)
