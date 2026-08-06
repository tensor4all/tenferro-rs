#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_doc_snippets", ROOT / "scripts/check-doc-snippets.py")
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def write_fixture(root: pathlib.Path, source: str, reference: str) -> pathlib.Path:
    (root / "docs").mkdir(parents=True)
    (root / "source.rs").write_text(source, encoding="utf-8")
    doc = root / "docs" / "guide.md"
    doc.write_text(reference, encoding="utf-8")
    return doc


def test_successful_region_extraction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        doc = write_fixture(
            root,
            "// snippet-start:demo\nlet x = 1;\n// snippet-end:demo\n",
            "<!-- snippet-source: source.rs#demo -->\nold\n<!-- end-snippet-source -->\n",
        )
        rewritten, changed = CHECKER.rewrite_doc(root, doc)
        assert changed
        assert "let x = 1;" in rewritten


def test_region_marker_without_final_newline_preserves_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        doc = write_fixture(
            root,
            "// snippet-start:run\nlet x = 1;\n// snippet-end:run",
            "<!-- snippet-source: source.rs#run -->\nold\n<!-- end-snippet-source -->\n",
        )
        rewritten, changed = CHECKER.rewrite_doc(root, doc)
        assert changed
        assert "let x = 1;" in rewritten


def test_unknown_region_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        doc = write_fixture(root, "// snippet-start:demo\nx\n// snippet-end:demo\n", "<!-- snippet-source: source.rs#missing -->\n<!-- end-snippet-source -->\n")
        try:
            CHECKER.rewrite_doc(root, doc)
        except ValueError as exc:
            assert "unknown snippet region" in str(exc)
        else:
            raise AssertionError("unknown region should fail")


def test_duplicate_region_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        doc = write_fixture(root, "// snippet-start:demo\nx\n// snippet-end:demo\n// snippet-start:demo\ny\n// snippet-end:demo\n", "<!-- snippet-source: source.rs#demo -->\n<!-- end-snippet-source -->\n")
        try:
            CHECKER.rewrite_doc(root, doc)
        except ValueError as exc:
            assert "duplicate snippet region" in str(exc)
        else:
            raise AssertionError("duplicate region should fail")


def test_empty_region_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        doc = write_fixture(root, "// snippet-start:demo\n\n// snippet-end:demo\n", "<!-- snippet-source: source.rs#demo -->\n<!-- end-snippet-source -->\n")
        try:
            CHECKER.rewrite_doc(root, doc)
        except ValueError as exc:
            assert "empty snippet region" in str(exc)
        else:
            raise AssertionError("empty region should fail")


def test_nested_region_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        doc = write_fixture(root, "// snippet-start:outer\n// snippet-start:inner\nx\n// snippet-end:inner\n// snippet-end:outer\n", "<!-- snippet-source: source.rs#outer -->\n<!-- end-snippet-source -->\n")
        try:
            CHECKER.rewrite_doc(root, doc)
        except ValueError as exc:
            assert "nested" in str(exc)
        else:
            raise AssertionError("nested regions should fail")


def test_reversed_or_overlapping_region_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        doc = write_fixture(root, "// snippet-end:demo\n// snippet-start:demo\nx\n// snippet-end:demo\n", "<!-- snippet-source: source.rs#demo -->\n<!-- end-snippet-source -->\n")
        try:
            CHECKER.rewrite_doc(root, doc)
        except ValueError as exc:
            assert "without matching" in str(exc) or "reversed" in str(exc)
        else:
            raise AssertionError("reversed regions should fail")


def test_no_unmarked_plain_rust_fences_remain() -> None:
    inventory = CHECKER.unmarked_rust_fences(ROOT)
    assert inventory == [], "unmarked Rust fences remain: " + ", ".join(inventory)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
    print("check-doc-snippets-tests-ok")
