from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from .retrieval import DocumentStore
from .task_engine import SearchQAFactCheckingSystem, build_task_pipelines

DEFAULT_STORE_PATH = Path(".mope_store.json")


def _load_pairs_from_txt_dir(dir_path: Path) -> Iterable[Tuple[str, str]]:
    for p in dir_path.rglob("*.txt"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            text = p.read_text(errors="ignore")
        yield (p.stem, text)


def _load_pairs_from_jsonl(file_path: Path) -> Iterable[Tuple[str, str]]:
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("id") or obj.get("doc_id"))
            text = str(obj.get("text") or obj.get("content") or "")
            if not doc_id or not text:
                continue
            yield (doc_id, text)


def _persist_store_snapshot(store: DocumentStore, out_path: Path) -> None:
    out: List[dict] = []
    # Convert to public snapshot by re-searching doc ids
    for doc_id, doc in store._documents.items():  # type: ignore[attr-defined]
        out.append({"id": doc_id, "text": doc.text})
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _restore_store_from_snapshot(in_path: Path) -> DocumentStore:
    store = DocumentStore()
    if not in_path.exists():
        return store
    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Try JSONL fallback
        data = []
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    pairs = []
    for obj in data:
        doc_id = str(obj.get("id") or obj.get("doc_id"))
        text = str(obj.get("text") or obj.get("content") or "")
        if doc_id and text:
            pairs.append((doc_id, text))
    store.add_many(pairs)
    return store


def cmd_index(args: argparse.Namespace) -> int:
    store_path = Path(args.store or DEFAULT_STORE_PATH)
    store = _restore_store_from_snapshot(store_path)

    # Ingest sources
    if args.dir:
        store.add_many(_load_pairs_from_txt_dir(Path(args.dir)))
    if args.jsonl:
        store.add_many(_load_pairs_from_jsonl(Path(args.jsonl)))

    _persist_store_snapshot(store, store_path)
    print(f"Indexed documents. Snapshot saved to {store_path}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    store_path = Path(args.store or DEFAULT_STORE_PATH)
    store = _restore_store_from_snapshot(store_path)
    if not store._documents:  # type: ignore[attr-defined]
        print("No documents indexed yet. Use 'mope index' first.", file=sys.stderr)
        return 2

    system = SearchQAFactCheckingSystem(store)
    pipelines = build_task_pipelines(system)

    result = system.answer(args.question, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.trace:
        # Also run through a pipeline to produce a trace update
        expert = list(pipelines.values())[0]
        trace = expert.run(args.question)
        print("\nPipeline trace:\n" + "\n".join(trace.trace))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mope", description="MoPE engineering CLI")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Index documents from a folder or JSONL")
    p_index.add_argument("--dir", type=str, help="Directory of .txt files to index")
    p_index.add_argument("--jsonl", type=str, help="JSONL with {id,text}")
    p_index.add_argument("--store", type=str, help="Path to snapshot store JSON/JSONL")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a question with retrieval+reading+fact-checking")
    p_ask.add_argument("question", type=str, help="The question to ask")
    p_ask.add_argument("--top-k", type=int, default=3, dest="top_k")
    p_ask.add_argument("--store", type=str, help="Path to snapshot store JSON/JSONL")
    p_ask.add_argument("--trace", action="store_true", help="Print pipeline trace as well")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
