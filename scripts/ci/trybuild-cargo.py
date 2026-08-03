#!/usr/bin/env python3
"""Append CI link flags to rustflags arrays supplied by nested trybuild Cargo."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys


def augment_rustflags_config(argument: str, extra: list[str]) -> str:
    """Append flags to one Cargo build/target rustflags array."""
    if not argument.startswith("--config="):
        return argument
    assignment = argument[len("--config=") :]
    key, separator, value = assignment.partition("=")
    is_rustflags = key == "build.rustflags" or (
        key.startswith("target.") and key.endswith(".rustflags")
    )
    if not separator or not is_rustflags:
        return argument

    rustflags = json.loads(value)
    if not isinstance(rustflags, list) or not all(
        isinstance(flag, str) for flag in rustflags
    ):
        raise ValueError(f"{key} must be an array of strings")
    encoded = json.dumps([*rustflags, *extra], separators=(",", ":"))
    return f"--config={key}={encoded}"


def main() -> int:
    extra = shlex.split(os.environ.get("TENFERRO_TRYBUILD_RUSTFLAGS", ""))
    if not extra:
        return subprocess.call(["cargo", *sys.argv[1:]])

    args = [augment_rustflags_config(argument, extra) for argument in sys.argv[1:]]

    environment = os.environ.copy()
    environment.pop("CARGO", None)
    return subprocess.call(["cargo", *args], env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
