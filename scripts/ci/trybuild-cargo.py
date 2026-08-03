#!/usr/bin/env python3
"""Add the CI link flags to trybuild's nested Cargo rustflags."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys


def main() -> int:
    extra = shlex.split(os.environ.get("TENFERRO_TRYBUILD_RUSTFLAGS", ""))
    if not extra:
        return subprocess.call(["cargo", *sys.argv[1:]])

    rustflags = ["--cfg", "trybuild", "--verbose", "-A", "dead_code", *extra]
    encoded = json.dumps(rustflags, separators=(",", ":"))
    args: list[str] = []
    replaced = False
    for argument in sys.argv[1:]:
        if argument.startswith("--config=build.rustflags="):
            args.append(f"--config=build.rustflags={encoded}")
            replaced = True
        elif argument.startswith("--config=target.") and ".rustflags=" in argument:
            key = argument[len("--config=") :].split("=", 1)[0]
            args.append(f"--config={key}={encoded}")
            replaced = True
        else:
            args.append(argument)
    if not replaced:
        args.append(f"--config=build.rustflags={encoded}")

    environment = os.environ.copy()
    environment.pop("CARGO", None)
    return subprocess.call(["cargo", *args], env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
