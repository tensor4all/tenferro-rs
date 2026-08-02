#!/usr/bin/env python3
"""Execute active schema-v2 commands and write their execution log."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType


CHECKER_PATH = Path(__file__).with_name("check-storage-ownership-contracts.py")
CONTRACT_PROBE = "--contract-schema"
CLI_SCHEMA = "tenferro.storage-ownership-cli-contract.v1"
MANIFEST_SCHEMA = "tenferro.storage-ownership-contracts.v2"
RECEIPT_SCHEMA = "tenferro.storage-ownership-receipt.v1"
DIAGNOSTICS_SCHEMA = "tenferro.storage-ownership-diagnostics.v1"

CLI_CONTRACT = {
    "schema": CLI_SCHEMA,
    "tool": "run-storage-ownership-contracts",
    "role": "runner",
    "manifest_schema": MANIFEST_SCHEMA,
    "probe": CONTRACT_PROBE,
    "options": [
        "--root",
        "--manifest",
        "--base-commit",
        "--receipt-out",
        "--diagnostics-json",
    ],
}


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("storage_ownership_checker", CHECKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load checker from {CHECKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _run(arguments: argparse.Namespace) -> int:
    root = arguments.root.resolve()
    if not root.is_dir():
        raise ValueError(f"repository root '{root}' is not a directory")
    manifest_relative, manifest_path = CHECKER._manifest_relative(root, arguments.manifest)
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read manifest '{manifest_relative}': {error}") from error
    data = CHECKER._load_toml_bytes(manifest_bytes)
    base_commit = CHECKER._canonical_commit(root, arguments.base_commit)
    CHECKER._promotion_check(
        root,
        data,
        base_commit=base_commit,
        manifest_relative=manifest_relative,
    )
    rows, _ = CHECKER._validate_manifest(root, data)
    CHECKER._tracked_tree_clean(root, rows, manifest_relative)
    candidate_commit = CHECKER._git(root, "rev-parse", "HEAD")
    environment = os.environ.copy()
    executions: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: item["id"]):
        if row["state"]["kind"] != CHECKER.ACTIVE_STATE:
            continue
        command = row["command"]
        cwd = CHECKER._validate_command(root, row)
        try:
            result = subprocess.run(
                list(command["argv"]),
                cwd=cwd,
                env=environment,
                text=True,
                capture_output=True,
                shell=False,
                check=False,
            )
        except OSError as error:
            CHECKER._fail(
                "E_COMMAND_FAILED",
                {"command_id": command["id"], "exit_code": 127},
                f"command could not be started: {error}",
            )
        if result.returncode != 0:
            CHECKER._fail(
                "E_COMMAND_FAILED",
                {"command_id": command["id"], "exit_code": result.returncode},
                "active command returned a non-zero exit code",
            )
        executions.append(
            {
                "obligation_id": row["id"],
                "argv": list(command["argv"]),
                "cwd": command["cwd"],
                "artifact_path": row["artifact"]["path"],
                "exit_code": result.returncode,
            }
        )
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "candidate_commit": candidate_commit,
        "base_commit": base_commit,
        "executions": executions,
    }
    CHECKER._validate_receipt(
        root,
        rows,
        receipt,
        base_commit=base_commit,
        manifest_relative=manifest_relative,
    )
    receipt_path = arguments.receipt_out.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("storage ownership runner: OK")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CHECKER_PATH.parents[1])
    parser.add_argument("--manifest", default="scripts/storage-ownership-contracts.toml")
    parser.add_argument("--base-commit")
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--diagnostics-json", action="store_true")
    return parser.parse_args()


def main() -> int:
    if sys.argv[1:] == [CONTRACT_PROBE]:
        print(json.dumps(CLI_CONTRACT, separators=(",", ":")))
        return 0
    try:
        return _run(_parse_args())
    except CHECKER.LedgerFailure as error:
        if "--diagnostics-json" in sys.argv[1:]:
            print(json.dumps({"schema": DIAGNOSTICS_SCHEMA, "diagnostics": [CHECKER._diagnostic(error)]}, sort_keys=True))
        else:
            print(f"error: {error.message}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        if "--diagnostics-json" in sys.argv[1:]:
            diagnostic = CHECKER.LedgerFailure("E_MANIFEST_INPUT", {"actual": str(error)}, "unable to execute storage ownership commands")
            print(json.dumps({"schema": DIAGNOSTICS_SCHEMA, "diagnostics": [CHECKER._diagnostic(diagnostic)]}, sort_keys=True))
        else:
            print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
