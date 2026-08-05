#!/usr/bin/env python3
"""Check the rendered/user-facing storage ownership documentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
GUIDE = Path("docs/storage-ownership.md")
USER_DOCS = (
    GUIDE,
    Path("docs/guides/views-and-slicing.md"),
    Path("docs/getting-started/core-concepts.md"),
    Path("docs/guides/devices-and-gpu.md"),
    Path("README.md"),
)
REQUIRED = (
    "# Storage ownership and access",
    "## The capability triad",
    "## Copies are named",
    "## Explicit device movement",
    "## Prepared access and loops",
    "## Detached and scoped execution",
    "TypedTensorView",
    "TypedTensorViewMut",
    "TensorRead<'_>",
    "duplicate()",
    "upload",
    "download",
    "synchronization",
    "completion-unproven",
)
FORBIDDEN = (
    "Buffer<T>",
    "BackendBuffer<T>",
    "ArcTensor",
    "to_contiguous_tensor",
    "cuda_interop",
    "webgpu_interop::allocate_raw",
    "webgpu_interop::finish_",
    "shallow clone",
    "hidden transfer",
)


class CheckError(ValueError):
    pass


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise CheckError(f"cannot read {path}: {error}") from error


def validate(root: Path, include_rendered: bool) -> None:
    guide = root / GUIDE
    if not guide.is_file():
        raise CheckError(f"missing guide: {GUIDE}")
    guide_text = read(guide)
    missing = [marker for marker in REQUIRED if marker not in guide_text]
    if missing:
        raise CheckError("storage guide is missing markers: " + ", ".join(missing))

    for relative in USER_DOCS:
        path = root / relative
        if not path.is_file():
            raise CheckError(f"missing user documentation: {relative}")
        text = read(path)
        for marker in FORBIDDEN:
            if marker in text:
                raise CheckError(f"removed API/language {marker!r} remains in {relative}")

    if "views-and-slicing.md" not in guide_text:
        raise CheckError("storage guide must link the views and slicing guide")
    if include_rendered:
        rendered = root / "target/docs-site"
        expected = (rendered / "storage-ownership.html", rendered / "guides/views-and-slicing.html")
        missing_rendered = [str(path.relative_to(root)) for path in expected if not path.is_file()]
        if missing_rendered:
            raise CheckError("rendered documentation is missing: " + ", ".join(missing_rendered))
        for path in expected:
            text = read(path)
            for marker in FORBIDDEN:
                if marker in text:
                    raise CheckError(f"removed API/language {marker!r} remains in {path.relative_to(root)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-rendered", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate(ROOT, args.include_rendered)
    except CheckError as error:
        print(f"storage-docs: {error}", file=sys.stderr)
        return 1
    print("storage-docs-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
