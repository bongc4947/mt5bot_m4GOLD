"""
scripts/version.py — print the repo's current state.

Works anywhere — Windows, Linux, Kaggle, Colab — does not depend on git
hooks being installed. Useful when you want to confirm:
   - Did my last `git pull` actually advance HEAD?
   - Does this Kaggle session have the commit I just pushed?
   - Are train and EA boxes on the same revision?

Usage:
    python scripts/version.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except Exception:
        return ""


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    short    = _git("rev-parse", "--short", "HEAD", cwd=repo) or "?"
    branch   = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo) or "?"
    date_iso = _git("show", "-s", "--format=%cI", "HEAD", cwd=repo) or "?"
    date_rel = _git("show", "-s", "--format=%cr", "HEAD", cwd=repo) or "?"
    subject  = _git("show", "-s", "--format=%s",  "HEAD", cwd=repo) or "?"
    remote   = _git("remote", "get-url", "origin", cwd=repo) or "?"
    n_files  = _git("ls-files", cwd=repo)
    n_files  = str(len([l for l in n_files.splitlines() if l])) if n_files else "?"
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Distinguish from the remote URL — pull out user/repo if it's github.
    if remote.startswith("https://github.com/") or "github.com:" in remote:
        slug = remote.split("github.com")[-1].lstrip("/:").removesuffix(".git")
    else:
        slug = remote

    print()
    print("=" * 64)
    print("  HYDRA mk4 — repo state (verified at runtime)")
    print("=" * 64)
    print(f"  remote   : {slug}")
    print(f"  branch   : {branch}")
    print(f"  HEAD     : {short}")
    print(f"  commit   : {date_iso}  ({date_rel})")
    print(f"  subject  : {subject}")
    print(f"  tracked  : {n_files} files")
    print(f"  checked  : {now}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
