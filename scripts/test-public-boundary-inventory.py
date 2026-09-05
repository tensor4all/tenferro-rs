#!/usr/bin/env python3
"""Focused mutation tests for the public-boundary inventory checker."""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_boundary_inventory", ROOT / "scripts/check-public-boundary-inventory.py")
assert spec and spec.loader
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)


def inventory() -> tuple[dict, dict, list[dict]]:
    auth = checker.authorities()
    data = json.loads(checker.OVERLAY.read_text(encoding="utf-8"))
    rows = checker.validate_overlay(data, auth)
    return data, auth, rows


def rejects(mutator, phrase: str) -> None:
    data, auth, _ = inventory()
    mutator(data)
    try:
        checker.validate_overlay(data, auth)
    except ValueError as error:
        assert phrase in str(error), (phrase, error)
    else:
        raise AssertionError("mutation was accepted")


def test_current_inventory_is_exhaustive() -> None:
    _, auth, rows = inventory()
    assert len(rows) == 184
    assert [row["operation"] for row in rows if row["id"].endswith("ordinary.eager")] == ["add", "einsum"]
    new_routes = {
        row["id"]: (row["surface"], row["phase"])
        for row in rows
        if row["id"] in {
            "einsum.einsum.ordinary.eager",
            "einsum.einsum.prepare.concrete",
            "einsum.einsum.prepared.concrete",
        }
    }
    assert new_routes == {
        "einsum.einsum.ordinary.eager": ("eager", "execution"),
        "einsum.einsum.prepare.concrete": ("concrete", "setup"),
        "einsum.einsum.prepared.concrete": ("concrete", "execution"),
    }


def test_case_filters_reject_malformed_uncovered_and_duplicate_contracts() -> None:
    rejects(
        lambda value: value["families"]["core"]["selectors"][0]["cases"][0].update(operations=[]),
        "nonempty list",
    )
    rejects(
        lambda value: value["families"]["core"]["selectors"][0]["cases"][0].update(operations="add"),
        "nonempty list",
    )
    rejects(
        lambda value: value["families"]["core"]["selectors"][0]["cases"][0].update(operations=[1]),
        "contain strings",
    )
    rejects(
        lambda value: value["families"]["core"]["selectors"][0]["cases"][0].update(operations=["abs", "abs"]),
        "must be unique",
    )
    rejects(
        lambda value: value["families"]["core"]["selectors"][0]["cases"][0].update(operations=["external"]),
        "outside selector",
    )
    def remove_coverage(value: dict) -> None:
        elementwise = next(item for item in value["families"]["core"]["selectors"] if item["id"] == "core.elementwise")
        for case in elementwise["cases"]:
            case["operations"] = ["add"]
    rejects(remove_coverage, "uncovered operations")
    def duplicate_id(value: dict) -> None:
        elementwise = next(item for item in value["families"]["core"]["selectors"] if item["id"] == "core.elementwise")
        eager = next(case for case in elementwise["cases"] if case["id"].endswith("ordinary.eager"))
        eager["id"] = "{family}.{operation}.ordinary.concrete"
    rejects(duplicate_id, "duplicate expanded case id")


def test_operation_addition_removal_and_rename_fail() -> None:
    data, auth, _ = inventory()
    selector = data["families"]["core"]["selectors"][0]
    selector["operations"].append("invented")
    try:
        checker.validate_overlay(data, auth)
    except ValueError as error:
        assert "unknown operation selector" in str(error)
    else:
        raise AssertionError("added operation accepted")
    rejects(lambda value: value["families"]["core"]["selectors"].pop(), "uncovered operations")
    rejects(lambda value: value["families"]["core"]["selectors"][0].update(operations=["renamed"]), "unknown operation selector")


def test_family_and_selector_coverage_mutations_fail() -> None:
    rejects(lambda value: value["families"].pop("sparse"), "families must exactly match")
    rejects(lambda value: value["families"]["core"]["selectors"].append(copy.deepcopy(value["families"]["core"]["selectors"][0])), "overlapping")
    rejects(lambda value: value["families"]["core"]["selectors"][0].update(operations=["unknown"]), "unknown operation selector")


def test_category_drift_is_rejected() -> None:
    rejects(lambda value: value["families"]["core"]["selectors"][0].update(category="wrong"), "category mismatch")


def test_disposition_evidence_owner_and_snapshot_mutations_fail() -> None:
    rejects(lambda value: value["families"]["core"]["selectors"][0]["surfaces"]["concrete"].update(disposition="measured"), "needs evidence")
    rejects(lambda value: value["families"]["core"]["selectors"][0]["surfaces"]["concrete"]["owners"].pop("validation"), "incomplete owners")
    rejects(lambda value: value["families"]["core"]["selectors"][0]["surfaces"]["concrete"]["owners"]["planning"].update(status="wrong"), "invalid not-applicable owner")
    data, auth, rows = inventory()
    with tempfile.TemporaryDirectory() as directory:
        stale_path = pathlib.Path(directory) / "snapshot.md"
        stale_path.write_text(checker.render_snapshot(data, auth, rows) + "drift\n", encoding="utf-8")
        original = checker.SNAPSHOT
        checker.SNAPSHOT = stale_path
        try:
            assert checker.main([]) == 1
        finally:
            checker.SNAPSHOT = original


def test_source_operation_addition_and_rename_require_review() -> None:
    source = (ROOT / "crates/tenferro-core-ops/src/catalog.rs").read_text(encoding="utf-8")
    added = source.replace('            Constant, "constant", Host, Constant, 0, 0, true;', '            Added, "added", Host, Constant, 0, 0, true;\n            Constant, "constant", Host, Constant, 0, 0, true;')
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "catalog.rs"
        path.write_text(added, encoding="utf-8")
        operations, categories = checker._core(path, pathlib.Path(directory))
        assert "added" in operations
        data, auth, _ = inventory()
        mutated = dict(auth)
        mutated["core"] = replace(auth["core"], operations=tuple(sorted(operations)), categories=categories)
        try:
            checker.validate_overlay(data, mutated)
        except ValueError as error:
            assert "uncovered operations" in str(error)
        else:
            raise AssertionError("added source operation accepted")
        renamed = source.replace('Add, "add"', 'Add, "renamed_add"')
        path.write_text(renamed, encoding="utf-8")
        renamed_ops, _ = checker._core(path, pathlib.Path(directory))
        assert "add" not in renamed_ops and "renamed_add" in renamed_ops


def test_family_discovery_addition_and_rename_require_review() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        extension = root / "ext/new/src/extension.rs"
        extension.parent.mkdir(parents=True)
        extension.write_text('const NEW_FAMILY_ID: &str = "new.family.v1";\\n', encoding="utf-8")
        try:
            checker._discover_extension_files(root)
        except ValueError as error:
            assert "unreviewed extension family" in str(error)
        else:
            raise AssertionError("new extension family accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        extension = root / "ext/sparse/src/extension.rs"
        extension.parent.mkdir(parents=True)
        extension.write_text('const FAMILY_ID: &str = "a"; const JVP_FAMILY_ID: &str = "b"; const VJP_FAMILY_ID: &str = "c"; const EXTRA_FAMILY_ID: &str = "d";\\n', encoding="utf-8")
        try:
            checker._discover_extension_files(root)
        except ValueError as error:
            assert "declarations drift" in str(error)
        else:
            raise AssertionError("added family declaration accepted")
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "extension.rs"
        path.write_text('const RENAMED_ID: &str = "tenferro-ext-sparse.matmul.v1";\\n', encoding="utf-8")
        try:
            checker._family_ids(path, "FAMILY_ID", pathlib.Path(directory))
        except ValueError as error:
            assert "family identifier" in str(error) or "no family identifiers" in str(error)
        else:
            raise AssertionError("renamed family identifier accepted")


def test_malformed_authority_syntax_is_rejected() -> None:
    source = "macro_rules! primitive_ops { ($macro:ident) => { $macro! { Broken } }; }\n"
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "catalog.rs"
        path.write_text(source, encoding="utf-8")
        try:
            checker._core(path.relative_to(directory), pathlib.Path(directory))
        except ValueError as error:
            assert "unsupported core catalog" in str(error)
        else:
            raise AssertionError("malformed catalog accepted")


def test_source_refs_require_declarations() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        path = root / "source.rs"
        path.write_text(
            "use crate::execute;\nmacro_rules! wrapper { ($execute:ident) => {} }\n",
            encoding="utf-8",
        )
        for ref in (
            {"path": "source.rs", "symbol": "execute"},
            {"path": "source.rs", "symbol": "ident"},
        ):
            try:
                checker._source_ref(ref, root)
            except ValueError as error:
                assert "symbol not found" in str(error)
            else:
                raise AssertionError("phantom source declaration accepted")

        path.write_text("pub(crate) fn execute() {}\n", encoding="utf-8")
        checker._source_ref({"path": "source.rs", "symbol": "execute"}, root)
        checker._source_ref({"path": "source.rs", "symbol": "execute"}, root, "test")


def test_enum_tuple_variants_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "ops.rs"
        path.write_text("enum Ops {\n    Good,\n    Broken(String),\n}\n", encoding="utf-8")
        try:
            checker._enum(path, "Ops", pathlib.Path(directory))
        except ValueError as error:
            assert "unsupported Ops variant syntax" in str(error)
        else:
            raise AssertionError("unsupported tuple enum variant accepted")

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "ops.rs"
        path.write_text("enum Ops {\n    Good { value: String },\n}\n", encoding="utf-8")
        try:
            checker._enum(path, "Ops", pathlib.Path(directory))
        except ValueError as error:
            assert "unsupported Ops variant syntax" in str(error)
        else:
            raise AssertionError("single-line struct enum variant accepted")


def test_each_category_has_distinct_regression_refs() -> None:
    _, _, rows = inventory()
    selectors = {}
    for row in rows:
        selectors.setdefault(row["selector"], set()).update(
            (test["path"], test["symbol"])
            for case in [row]
            for test in case["tests"]
        )
    assert len(selectors) == 13
    assert all(refs for refs in selectors.values())
    assert len(set(tuple(sorted(refs)) for refs in selectors.values())) == len(selectors)


def test_benchmark_cases_are_unique_and_contract_complete() -> None:
    data, _, rows = inventory()
    payload = checker.benchmark_payload(data, rows)
    cases = payload["cases"]
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["status"] == "pending" for case in cases)
    required = {"id", "operation", "family", "surface", "phase", "setup_boundary", "workflow"}
    assert all(required <= set(case) for case in cases)
    assert payload["benchmarks"][0]["status"] == "pending"


def test_case_selection_is_conservative_and_aliases_stay_distinct() -> None:
    _, _, rows = inventory()
    all_ids = {row["id"] for row in rows}
    core_ids = {row["id"] for row in rows if row["family"] == "core"}
    assert checker.select_case_ids(rows, ["crates/tenferro-tensor/src/backend.rs"]) == sorted(core_ids)
    assert checker.select_case_ids(rows, ["crates/tenferro-core-ops/src/new-owner.rs"]) == sorted(all_ids)
    assert checker.select_case_ids(rows, ["docs/README.md"]) == []
    alias_ids = {row["id"] for row in rows if row["family"] == "sparse"}
    assert len(alias_ids) == 6


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
