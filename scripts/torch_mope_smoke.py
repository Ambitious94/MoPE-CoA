"""Smoke test for TorchMoPE integration.

Runs a tiny forward pass replacing a dummy block's FFN with TorchMoPELayer,
optionally loading gate.json. Verifies shape consistency and prints summary.
"""
from __future__ import annotations















"""Deprecated: use scripts.qwen_mope_smoke for HF models.

This legacy nanoGPT torch smoke file is kept to avoid breaking imports.
"""
def main() -> int:
    print("[deprecated] Use 'python -m scripts.qwen_mope_smoke' for modern HF models.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
