"""Validate the built docs-site artifact."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_RELATIVE_PATHS = (
    "index.html",
    "math/index.html",
    "math/svd.html",
    "math/registry.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-root", type=Path, required=True)
    return parser.parse_args(argv)


def validate_site_root(site_root: Path) -> None:
    missing = [relative for relative in REQUIRED_RELATIVE_PATHS if not (site_root / relative).exists()]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"docs site missing required files: {joined}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_site_root(args.site_root)
    print(f"docs_site_ok=1 site_root={args.site_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
