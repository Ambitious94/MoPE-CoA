from __future__ import annotations

import json
from typing import Any, Dict, List

import importlib
from .retrieval import DocumentStore
from .task_engine import SearchQAFactCheckingSystem


class MoPEService:
    def __init__(self) -> None:
        self.store = DocumentStore()

    def index(self, docs: List[Dict[str, str]]) -> int:
        pairs = [(d["id"], d["text"]) for d in docs if d.get("id") and d.get("text")]
        self.store.add_many(pairs)
        return len(pairs)

    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        system = SearchQAFactCheckingSystem(self.store)
        return system.answer(question, top_k=top_k)


def build_app() -> Any:
    spec = importlib.util.find_spec("fastapi")
    if spec is None:
        raise RuntimeError("fastapi is not installed; install with 'pip install mope[server]'")
    fastapi_mod = importlib.import_module("fastapi")
    FastAPI = fastapi_mod.FastAPI
    HTTPException = fastapi_mod.HTTPException

    svc = MoPEService()
    app = FastAPI(title="MoPE Service", version="0.1.0")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/index")
    def index(payload: List[Dict[str, Any]]) -> Dict[str, int]:
        count = svc.index(payload)
        return {"indexed": count}

    @app.post("/ask")
    def ask(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            question = str(payload.get("question", ""))
            top_k = int(payload.get("top_k", 3))
            res = svc.ask(question, top_k=top_k)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return res

    return app
