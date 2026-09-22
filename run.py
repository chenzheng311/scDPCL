"""Stable entry point for the organized scDPCL experiment package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parent
REPRODUCE_SCRIPT = RELEASE_ROOT / "config" / "reproduce.py"
DATASETS = ("PBMC-10k", "PBMC-3k", "BMNC")


def parse_args():
    parser = argparse.ArgumentParser(description="Run one organized scDPCL experiment.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--profile",
        choices=("two_group", "unified", "legacy_tuned"),
        default="two_group",
    )
    parser.add_argument("--mode", choices=("optimized", "original"), default="optimized")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-original-pretrain", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    command = [
        sys.executable,
        str(REPRODUCE_SCRIPT),
        "--datasets",
        args.dataset,
        "--mode",
        args.mode,
        "--profile",
        args.profile,
        "--python",
        sys.executable,
    ]
    if args.output_root:
        command.extend(["--output_root", args.output_root])
    if args.dry_run:
        command.append("--dry_run")
    if args.reuse_original_pretrain:
        command.append("--reuse_original_pretrain")

    return subprocess.call(command, cwd=str(RELEASE_ROOT.parent))


if __name__ == "__main__":
    raise SystemExit(main())
