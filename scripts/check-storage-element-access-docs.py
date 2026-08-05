#!/usr/bin/env python3
"""Validate the executable concepts in the views/slicing guide."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "# Views, slicing, and explicit copies",
    "as_view()",
    "as_view_mut()",
    "TypedTensorView",
    "TypedTensorViewMut",
    "duplicate()",
    "Prepared element access",
    "provider lookup",
    "synchronize",
    "upload_tensor",
    "download_tensor",
)
FORBIDDEN = (
    "Buffer<T>",
    "BackendBuffer<T>",
    "ArcTensor",
    "to_contiguous_tensor",
    "cuda_interop",
    "webgpu_interop::allocate_raw",
    "webgpu_interop::finish_",
)


class CheckError(ValueError):
    pass


def validate(root: Path, requested: str) -> None:
    relative = Path(requested)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise CheckError("guide path must be repository-relative")
    if relative != Path("docs/guides/views-and-slicing.md"):
        raise CheckError("the canonical element-access guide is docs/guides/views-and-slicing.md")
    path = root / relative
    if not path.is_file():
        raise CheckError(f"missing guide: {relative}")
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in REQUIRED if marker not in text]
    if missing:
        raise CheckError("guide is missing markers: " + ", ".join(missing))
    present = [marker for marker in FORBIDDEN if marker in text]
    if present:
        raise CheckError("removed API/language remains: " + ", ".join(present))
    if "../storage-ownership.md" not in text:
        raise CheckError("guide must link the storage ownership contract")
    if "as_ptr()" not in text:
        raise CheckError("guide must demonstrate that duplicate creates fresh storage")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide")
    args = parser.parse_args(argv)
    try:
        validate(ROOT, args.guide)
    except (CheckError, OSError, UnicodeError) as error:
        print(f"storage-element-access-docs: {error}", file=sys.stderr)
        return 1
    print("storage-element-access-docs-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
