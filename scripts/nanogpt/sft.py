"""Wrapper for unified nanoGPT MoPE SFT script.

Usage:
  python -m scripts.nanogpt.sft --help
  # Example flags are defined in scripts.nanogpt_mope_sft.py
"""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    try:
        from scripts.nanogpt_mope_sft import main as legacy_main  # type: ignore
    except Exception as e:
        print("Failed to import legacy SFT script: scripts.nanogpt_mope_sft.py")
        print("Error:", e)
        return 2
    print("[scripts.nanogpt.sft] Delegating to legacy unified SFT script.")
    return legacy_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
