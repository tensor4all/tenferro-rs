#!/usr/bin/env python3
"""Source gate for the build-time ISA rule (AGENTS.md, tenferro-rs#2762).

A kernel must not change with the build machine's ISA. This check fails on:

* `cfg!(target_feature ...)` / `cfg(target_feature ...)` in crates/**/*.rs,
  except in a `#![no_std]` file that is listed in NO_STD_ALLOWLIST with a reason;
* `-C target-cpu` / `-C target-feature` in committed build configuration
  (.cargo/, .github/ workflows, scripts/**) unless listed in EXPERIMENT_ALLOWLIST;
* `target-cpu` / `target-feature` in RUSTFLAGS / CARGO_ENCODED_RUSTFLAGS of the
  environment (the same entry point CI calls).

It warns (exit code unaffected unless --deny-warnings) on bare
`#[target_feature(enable = ...)]` functions with no visible runtime-detection
guard. That one is a heuristic: it is a same-file textual check and cannot see
guards in other files or through macros.

Source check only; no disassembly. `--self-test` proves each rule rejects.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Exact repo-relative paths; every entry needs a reason.
NO_STD_ALLOWLIST = {}
# Files that mention the flags on purpose. Only files that are clearly
# experiments belong here; when in doubt, leave the hit reported.
EXPERIMENT_ALLOWLIST = {}
# This checker and its tests contain the forbidden strings as fixtures.
SELF_FILES = {"scripts/check-isa-rules.py", "scripts/ci/tests/test_isa_rules.py"}

ENV_VARS = (
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_BUILD_RUSTFLAGS",
)
ISA_FLAG = re.compile(r"target[-_]?(?:cpu|feature)", re.IGNORECASE)
CONFIG_FLAG = re.compile(
    r"(?:-C|--codegen)[\"',\s=]*target-(?:cpu|feature)|\btarget-(?:cpu|feature)="
)
CFG_OPEN = re.compile(r"\bcfg(?:_attr)?\s*!?\s*\(")
NO_STD = re.compile(r"#!\[[^\]]*\bno_std\b")
TF_ATTR = re.compile(r"#\[\s*target_feature\s*\(\s*enable")
FN_NAME = re.compile(r"\bfn\s+(\w+)")
DETECT = re.compile(r"\bis_\w+_feature_detected\s*!|\bArch::new\b|\bpulp::")
NON_CODE = re.compile(
    r'//[^\n]*|/\*|(?:br|cr|r)(?P<hashes>\#*)"|"(?:\\.|[^"\\])*"'
    r"|'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\])'",
    re.DOTALL,
)
CONFIG_SUFFIXES = {".sh", ".bash", ".py", ".toml", ".yml", ".yaml", ".json", ".js", ".cfg", ""}


def mask_non_code(source):
    """Blank comments and string literals, preserving offsets and newlines."""
    masked = list(source)
    offset = 0
    while match := NON_CODE.search(source, offset):
        end = match.end()
        if match.group() == "/*":
            depth = 1
            while depth and (marker := re.search(r"/\*|\*/", source[end:])):
                depth += 1 if marker.group() == "/*" else -1
                end += marker.end()
            if depth:
                end = len(source)
        elif match.group("hashes") is not None:
            terminator = '"' + match.group("hashes")
            close = source.find(terminator, end)
            end = len(source) if close < 0 else close + len(terminator)
        masked[match.start():end] = ["\n" if c == "\n" else " " for c in source[match.start():end]]
        offset = end
    return "".join(masked)


def line_of(text, index):
    return text.count("\n", 0, index) + 1


def cfg_target_feature_lines(masked):
    lines = []
    for match in CFG_OPEN.finditer(masked):
        depth, index = 1, match.end()
        while depth and index < len(masked):
            depth += {"(": 1, ")": -1}.get(masked[index], 0)
            index += 1
        if re.search(r"\btarget_feature\b", masked[match.end():index]):
            lines.append(line_of(masked, match.start()))
    return lines


def check_env(env):
    errors = []
    for name in ENV_VARS:
        value = env.get(name, "")
        if ISA_FLAG.search(value):
            shown = value.replace("\x1f", " ")
            errors.append(
                f"environment: {name} contains target-cpu/target-feature ({shown!r}); "
                "unset it, e.g. `env -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS ...`"
            )
    return errors


def rust_files(root):
    return sorted(p for p in (root / "crates").rglob("*.rs") if p.is_file()) if (root / "crates").is_dir() else []


def check_cfg(root):
    errors, warnings = [], []
    for path in rust_files(root):
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8", errors="replace")
        masked = mask_non_code(source)
        for line in cfg_target_feature_lines(masked):
            if NO_STD.search(masked):
                if rel in NO_STD_ALLOWLIST:
                    continue
                errors.append(
                    f"{rel}:{line}: cfg(target_feature) in a #![no_std] file that is not in "
                    "NO_STD_ALLOWLIST (scripts/check-isa-rules.py); add it with a reason or remove it"
                )
            else:
                errors.append(f"{rel}:{line}: cfg(target_feature) is forbidden; select SIMD at runtime")
        for tf in TF_ATTR.finditer(masked):
            fn = FN_NAME.search(masked, tf.end())
            name = fn.group(1) if fn else None
            guarded = bool(
                name
                and DETECT.search(masked)
                and re.search(rf"(?<!fn )\b{re.escape(name)}\s*[(:]", masked[: fn.start()] + masked[fn.end():])
            )
            if not guarded:
                warnings.append(
                    f"{rel}:{line_of(masked, tf.start())}: #[target_feature] fn `{name}` has no "
                    "runtime-detection guard visible in this file (heuristic)"
                )
    return errors, warnings


def config_files(root):
    found = []
    for sub in (".cargo", ".github", "scripts"):
        base = root / sub
        if base.is_dir():
            found += [p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for name in ("rust-toolchain.toml", "Makefile", "justfile"):
        if (root / name).is_file():
            found.append(root / name)
    return sorted(p for p in found if p.suffix in CONFIG_SUFFIXES)


def check_configs(root):
    errors = []
    for path in config_files(root):
        rel = path.relative_to(root).as_posix()
        if rel in SELF_FILES or rel in EXPERIMENT_ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in CONFIG_FLAG.finditer(text):
            errors.append(
                f"{rel}:{line_of(text, match.start())}: `{match.group().strip()}` in committed build "
                "configuration; report it, or list a clear experiment in EXPERIMENT_ALLOWLIST with a reason"
            )
    return errors


def run_checks(root, env):
    errors = check_env(env)
    cfg_errors, warnings = check_cfg(root)
    return errors + cfg_errors + check_configs(root), warnings


def self_test():
    failures = []

    def expect(label, files, env, want_fail, want_warn=False, text=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, content in files.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            errors, warnings = run_checks(root, env)
            ok = bool(errors) == want_fail and bool(warnings) == want_warn
            if ok and text:
                ok = any(text in e for e in errors + warnings)
            print(f"  [{'ok' if ok else 'WRONG'}] {label}: {'FAIL' if errors else 'PASS'}"
                  f"{' +warn' if warnings else ''}")
            if not ok:
                failures.append(label)

    lib = "crates/x/src/lib.rs"
    print("self-test (each forbidden pattern must be rejected):")
    expect("cfg! target_feature", {lib: 'fn f() -> bool { cfg!(target_feature = "avx512f") }'}, {}, True, text="lib.rs:1")
    expect("cfg attr target_feature (nested)", {lib: '#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]\nfn f() {}'}, {}, True)
    expect("cfg_attr target_feature", {lib: '#[cfg_attr(target_feature = "avx2", inline)]\nfn f() {}'}, {}, True)
    expect("no_std file not allow-listed", {lib: '#![no_std]\n#[cfg(target_feature = "avx2")]\nfn f() {}'}, {}, True, text="NO_STD_ALLOWLIST")
    expect("-C target-cpu in .cargo/config.toml", {".cargo/config.toml": '[build]\nrustflags = ["-C", "target-cpu=native"]\n'}, {}, True, text="config.toml:2")
    expect("-Ctarget-feature in CI workflow", {".github/workflows/ci.yml": "      - run: RUSTFLAGS='-Ctarget-feature=+avx2' cargo build\n"}, {}, True)
    expect("-C target-cpu in scripts shell", {"scripts/bench.sh": "RUSTFLAGS='-C target-cpu=znver5' cargo bench\n"}, {}, True)
    expect("RUSTFLAGS target-cpu", {}, {"RUSTFLAGS": "-C target-cpu=znver5"}, True, text="RUSTFLAGS")
    expect("CARGO_ENCODED_RUSTFLAGS target-feature", {}, {"CARGO_ENCODED_RUSTFLAGS": "-C\x1ftarget-feature=+avx2"}, True)
    expect("bare #[target_feature] warns", {lib: '#[target_feature(enable = "avx2")]\nunsafe fn k() {}'}, {}, False, want_warn=True, text="lib.rs:1")
    print("self-test (these must be accepted):")
    expect("mention in comment/string", {lib: '// cfg!(target_feature = "avx2")\nconst S: &str = "cfg!(target_feature = \\"x\\")";'}, {}, False)
    expect("guarded #[target_feature]", {lib: '#[target_feature(enable = "avx2")]\nunsafe fn k() {}\nfn g() { if is_x86_feature_detected!("avx2") { unsafe { k() } } }'}, {}, False)
    expect("clean env and tree", {lib: "fn f() {}"}, {"RUSTFLAGS": "-D warnings"}, False)

    # End to end: the real CLI must exit non-zero on a bad tree.
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / lib
        bad.parent.mkdir(parents=True)
        bad.write_text('fn f() -> bool { cfg!(target_feature = "avx512f") }\n')
        env = {k: v for k, v in os.environ.items() if k not in ENV_VARS}
        proc = subprocess.run([sys.executable, __file__, "--root", tmp], env=env, text=True, capture_output=True)
        ok = proc.returncode != 0 and "cfg(target_feature) is forbidden" in proc.stdout + proc.stderr
        print(f"  [{'ok' if ok else 'WRONG'}] CLI exits non-zero on bad tree (rc={proc.returncode})")
        if not ok:
            failures.append("cli")
    if failures:
        print(f"self-test FAILED: {', '.join(failures)}")
        return 1
    print("self-test PASSED")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--self-test", action="store_true", help="prove each rule rejects its pattern")
    parser.add_argument("--root", type=Path, default=ROOT, help="tree to check (default: this repo)")
    parser.add_argument("--deny-warnings", action="store_true", help="treat heuristic warnings as errors")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    errors, warnings = run_checks(args.root, os.environ)
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")
    if errors or (warnings and args.deny_warnings):
        print(f"check-isa-rules: FAIL ({len(errors)} errors, {len(warnings)} warnings)")
        return 1
    print(f"check-isa-rules: PASS ({len(rust_files(args.root))} Rust files, {len(warnings)} warnings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
