#!/usr/bin/env python3
"""Test-only adapter for the public Phase 2E subprocess lifecycle.

The adapter replaces expensive measurement and external network boundaries in
an isolated temporary copy of the orchestrator.  Lifecycle state remains owned
by the production implementation.  The retained semantic-evidence builder
deliberately reuses ``OuterOrchestratorTests.make_complete_root`` because a
second independent copy of the large Task 8A artifact schema would create a
worse producer/validator drift risk; lifecycle-owned context, ledger, journal,
children, progress, locks, aggregate, index, and preservation transitions are
excluded from that reuse.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
from typing import Any, MutableMapping


ENTRYPOINT_MARKER = '\nif __name__ == "__main__":\n'
ADAPTER_MARKER = "# PHASE2E_TEMP_COPY_ADAPTER"


def instrument_orchestrator_copy(
    source_path: pathlib.Path, destination: pathlib.Path
) -> dict[str, Any]:
    """Inject one provenance-bound adapter call at the sole entrypoint marker."""
    source_path = pathlib.Path(source_path).resolve(strict=True)
    source = source_path.read_text(encoding="utf-8")
    marker_count = source.count(ENTRYPOINT_MARKER)
    if marker_count != 1:
        raise AssertionError(
            "orchestrator source must contain exactly one entrypoint marker"
        )
    source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    adapter = (
        f"\n{ADAPTER_MARKER} source_sha256={source_sha256}\n"
        "from scripts.phase2e_public_lifecycle_fixture import "
        "install_temp_copy_adapters as _phase2e_install_temp_copy_adapters\n"
        "_phase2e_install_temp_copy_adapters(globals())\n"
        "del _phase2e_install_temp_copy_adapters\n"
    )
    prefix, marker, suffix = source.partition(ENTRYPOINT_MARKER)
    if marker != ENTRYPOINT_MARKER:
        raise AssertionError("orchestrator entrypoint partition failed")
    destination = pathlib.Path(destination)
    destination.write_text(
        prefix + adapter + marker + suffix,
        encoding="utf-8",
    )
    return {
        "version": 1,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "entrypoint_marker_count": marker_count,
        "adapter_sha256": hashlib.sha256(adapter.encode()).hexdigest(),
    }


def _write_fixture_journal(namespace: MutableMapping[str, Any], root: pathlib.Path):
    namespace["protocol"].atomic_write_json(
        root / namespace["PROCESS_JOURNAL"],
        {
            "version": 1,
            "entries": [
                {
                    "ordinal": 1,
                    "stage": namespace["STAGE_ORDER"][0],
                    "argv": ["fixture-stage"],
                    "pid": 999999,
                    "pgid": 999999,
                    "start_ticks": 1,
                    "state": "EXITED",
                    "exit_code": 0,
                    "signals": [],
                    "reaped": True,
                }
            ],
        },
    )


def _install_fixture_evidence(
    namespace: MutableMapping[str, Any],
    root: pathlib.Path,
    context_path: pathlib.Path,
) -> None:
    from unittest import mock

    from scripts import run_phase2e_gates as fixture_gates
    from scripts import test_run_phase2e as fixture_tests

    context = namespace["load_stage_context"](
        context_path,
        hashlib.sha256(context_path.read_bytes()).hexdigest(),
        require_fresh_scratch=False,
    )
    staging = root.parent / "fixture-evidence-source"
    staging.mkdir()
    case = fixture_tests.OuterOrchestratorTests()
    case.REPOSITORY = pathlib.Path(context["repository"])
    case.CANDIDATE = context["candidate_sha"]
    with mock.patch.object(
        fixture_gates, "validate_source_contract", return_value={}
    ), mock.patch.object(
        fixture_gates, "validate_terminal_evidence", return_value=None
    ):
        case.make_complete_root(staging)
    excluded = {
        namespace["AGGREGATE_MANIFEST"],
        namespace["PROGRESS_MANIFEST"],
        namespace["STAGE_CONTEXT"],
        namespace["PROCESS_JOURNAL"],
        "evidence-ledger.json",
        ".orchestrator.lock",
        "children",
    }
    for child in staging.iterdir():
        if child.name in excluded:
            continue
        target = root / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    shutil.rmtree(root / "attempts" / "allocation")


def _run_fixture_lane(
    namespace: MutableMapping[str, Any],
    root: pathlib.Path,
    stage: str,
    validity_state: str,
) -> None:
    protocol = namespace["protocol"]
    stage_name, lane = stage.split("/", 1)
    ledger_path = root / "evidence-ledger.json"
    ledger = namespace["_read_json"](ledger_path, "fixture ledger")
    attempt = namespace["_next_attempt"](
        {"evidence_root": str(root)}, stage_name, lane
    )
    artifact_root = (
        f"/phase2e-fixture-missing/{stage_name}/{lane}/{attempt}"
        if stage_name == "allocation"
        else None
    )
    ledger = protocol.open_attempt(
        ledger,
        stage_name,
        lane,
        attempt,
        artifact_root=artifact_root,
    )
    if stage_name == "allocation":
        ledger = protocol.bind_attempt_artifact(
            ledger,
            stage_name,
            lane,
            attempt,
            artifact_root=artifact_root,
            artifact_device=1,
            artifact_inode=1,
        )
    ledger = protocol.close_attempt(
        ledger,
        stage_name,
        lane,
        attempt,
        None if validity_state == "INCONCLUSIVE" else "PASS",
        validity_state=validity_state,
    )
    protocol.atomic_write_json(ledger_path, ledger)


def install_temp_copy_adapters(namespace: MutableMapping[str, Any]) -> None:
    """Install adapters into one already-loaded temporary orchestrator copy."""
    fixture_marker = (
        pathlib.Path(namespace["__file__"]).resolve().parent.parent
        / ".phase2e-real-handoff-fixture"
    )
    mode = os.environ.get("PHASE2E_FIXTURE_MODE")
    if fixture_marker.is_file():
        mode = "real-handoff"
    if mode is None:
        raise AssertionError("Phase 2E lifecycle fixture mode is missing")
    if mode == "real-handoff":
        for ordinal, stage in enumerate(namespace["STAGE_ORDER"]):
            exit_code = 2 if ordinal == 1 else 0
            namespace["STAGE_HANDLERS"][stage] = (
                lambda _context, code=exit_code: code
            )
        namespace["_preflight_offline_feature_queries"] = lambda _context: None
        namespace["require_remote_index"] = lambda *_args, **_kwargs: None
        return

    def stage_runner(
        context_path,
        context_sha256,
        repository,
        *,
        root,
        root_identity,
    ):
        def run(stage, _environment):
            fixture_root = pathlib.Path(root)
            _write_fixture_journal(namespace, fixture_root)
            if mode == "start" and stage == "allocation/direct-current-main":
                _run_fixture_lane(
                    namespace, fixture_root, stage, "INCONCLUSIVE"
                )
                return 2
            if stage.startswith(("allocation/", "timing/")):
                _run_fixture_lane(namespace, fixture_root, stage, "COMPLETE")
            if stage == "aggregate-validation":
                _install_fixture_evidence(
                    namespace, fixture_root, pathlib.Path(context_path)
                )
            return 0

        return run

    def fetch_comment(issue_url):
        commit = os.environ["PHASE2E_FIXTURE_PRESERVATION_COMMIT"]
        return {
            "id": 1,
            "html_url": issue_url,
            "issue_url": (
                "https://api.github.com/repos/tensor4all/tenferro-rs/issues/1436"
            ),
            "body": (
                f"preservation_commit: {commit}\n"
                f"evidence_root: {os.environ['PHASE2E_FIXTURE_ROOT']}\n"
                f"candidate: {os.environ['PHASE2E_FIXTURE_CANDIDATE']}\n"
                "terminal_status: PASS\n"
            ),
        }

    namespace["_preflight_offline_feature_queries"] = lambda _context: None
    namespace["require_remote_index"] = lambda *_args, **_kwargs: None
    namespace["_subprocess_stage_runner"] = stage_runner
    namespace["record_preserved"].__kwdefaults__["remote_validator"] = (
        lambda _repository, _commit: None
    )
    namespace["record_preserved"].__kwdefaults__["comment_fetcher"] = (
        fetch_comment
    )
