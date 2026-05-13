"""
bundle_data.py — package training-ready parquet caches into one .zip
for upload to Google Drive (Colab handoff).

USAGE
-----
On the box where MT5 is installed (after running extract_data.py):

    python bundle_data.py                  # zip everything in data/parquet/
                                           # to data/HYDRA4_data_bundle.zip
    python bundle_data.py --out path.zip   # explicit output path
    python bundle_data.py --include cal    # also include economic_calendar +
                                           #   fae_cache parquet
    python bundle_data.py --list           # show what would be bundled

On Colab (after mounting Drive and uploading the .zip):

    python bundle_data.py --restore /content/drive/MyDrive/HYDRA4_data_bundle.zip

Restore unpacks into data/parquet/ relative to the project root, ready
for `train.py all --skip-extract`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import BASE_DIR, DATA_DIR, PARQUET_DIR

log = logging.getLogger(__name__)
DEFAULT_OUT = DATA_DIR / "HYDRA4_data_bundle.zip"


def _gather() -> list[tuple[Path, str]]:
    """Return [(absolute_path, archive_name), ...] for everything in the bundle."""
    items: list[tuple[Path, str]] = []
    if PARQUET_DIR.exists():
        for p in sorted(PARQUET_DIR.rglob("*.parquet")):
            arcname = p.relative_to(BASE_DIR).as_posix()
            items.append((p, arcname))
    return items


def cmd_bundle(args: argparse.Namespace) -> int:
    items = _gather()
    if not items:
        print(f"No parquet files found under {PARQUET_DIR}. "
              "Run `python extract_data.py` first.")
        return 1

    out = Path(args.out) if args.out else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in items:
            z.write(src, arcname=arc)
            total += src.stat().st_size
    out_size = out.stat().st_size
    print(f"\nWrote {out}")
    print(f"  files       : {len(items)}")
    print(f"  uncompressed: {total/1e6:.1f} MB")
    print(f"  compressed  : {out_size/1e6:.1f} MB  (ratio {out_size/max(total,1):.2f})")
    print(f"\nNext: upload to Drive, then on Colab run:")
    print(f"  python python/bundle_data.py --restore <drive-path-to-this-zip>")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    items = _gather()
    if not items:
        print(f"Empty: {PARQUET_DIR}")
        return 1
    total = 0
    for src, arc in items:
        sz = src.stat().st_size
        total += sz
        print(f"  {sz/1e6:>8.1f} MB  {arc}")
    print(f"  --")
    print(f"  {total/1e6:>8.1f} MB  ({len(items)} files)")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    src = Path(args.restore)
    if not src.exists():
        print(f"Bundle not found: {src}")
        return 1
    print(f"Restoring {src} ({src.stat().st_size/1e6:.1f} MB) → {BASE_DIR}")
    with zipfile.ZipFile(src) as z:
        members = z.namelist()
        z.extractall(BASE_DIR)
    print(f"Extracted {len(members)} files.")
    # Sanity check: list how many parquet rows landed under PARQUET_DIR
    cached = sorted(PARQUET_DIR.rglob("*.parquet"))
    print(f"Parquet cache now contains {len(cached)} files under {PARQUET_DIR}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", dest="list_only", action="store_true",
                   help="Show files that would be bundled and exit.")
    g.add_argument("--restore", metavar="ZIP",
                   help="Unpack a previously-created bundle into the project.")
    p.add_argument("--out", help=f"Output zip path (default: {DEFAULT_OUT}).")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_parser().parse_args(argv)
    if args.list_only:
        return cmd_list(args)
    if args.restore:
        return cmd_restore(args)
    return cmd_bundle(args)


if __name__ == "__main__":
    sys.exit(main())
