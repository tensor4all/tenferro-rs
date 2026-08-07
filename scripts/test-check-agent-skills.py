#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_agent_skills", ROOT / "scripts" / "check-agent-skills.py"
)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_skill_root(root: pathlib.Path) -> None:
    canonical = root / ".agents" / "skills" / "tenferro-compute"
    required = ["SKILL.md", *CHECKER.REFERENCE_FILES]
    for relative in required:
        write(canonical / relative, f"canonical {relative}\n")
    write(canonical / "agents" / "openai.yaml", "interface:\n  display_name: compute\n")

    for mirror in (root / ".claude" / "skills", root / ".kimi" / "skills"):
        for relative in required:
            write(mirror / "tenferro-compute" / relative, f"canonical {relative}\n")

    opencode = root / ".opencode" / "commands" / "tenferro-compute.md"
    write(
        opencode,
        "\n".join(
            f"@.agents/skills/tenferro-compute/{relative}"
            for relative in CHECKER.REFERENCE_FILES
        )
        + "\n",
    )


def test_complete_skill_layout_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        make_skill_root(root)
        assert CHECKER.check(root) == []


def test_missing_mirror_file_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        make_skill_root(root)
        (root / ".kimi" / "skills" / "tenferro-compute" / "SKILL.md").unlink()
        errors = CHECKER.check(root)
        assert any("missing mirror file" in error for error in errors)


def test_mirror_content_drift_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        make_skill_root(root)
        write(
            root / ".claude" / "skills" / "tenferro-compute" / "SKILL.md",
            "stale\n",
        )
        errors = CHECKER.check(root)
        assert any("does not match canonical" in error for error in errors)


def test_opencode_missing_reference_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        make_skill_root(root)
        opencode = root / ".opencode" / "commands" / "tenferro-compute.md"
        opencode.write_text("@.agents/skills/tenferro-compute/SKILL.md\n", encoding="utf-8")
        errors = CHECKER.check(root)
        assert any("OpenCode entry is missing" in error for error in errors)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
    print("check-agent-skills-tests-ok")
