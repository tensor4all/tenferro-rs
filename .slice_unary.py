#!/usr/bin/env python3
"""Step-2 slice for a unary analytic operation (issue #1926).

Usage: python3 .slice_unary.py expm1

Deletes `TensorAnalytic::<op>`, moves the CUDA dispatch into that backend's read
half, removes the one-shot implementations and the macro declaration/delegation
lines, absorbs the removed body into the read halves that were delegating to it,
switches the CPU helper macro to its `read_only:` form, and migrates the call
sites. Residuals are reported by the next `cargo check`.

Every step asserts its expected occurrence count, so a surprise in the source
stops the script instead of writing a wrong edit.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OP = sys.argv[1]
CAP = OP[0].upper() + OP[1:]
ELEM = re.search(r"(\w+)_elem", (ROOT / "crates/tenferro-cpu/src/analytic.rs").read_text()).group(1)


def sanitize(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if text.startswith('r"', i) or text.startswith("r#", i):
            h, k = 0, i + 1
            while k < n and text[k] == "#":
                h += 1
                k += 1
            if k < n and text[k] == '"':
                c = '"' + "#" * h
                j = text.find(c, k + 1)
                i = n if j == -1 else j + len(c)
                continue
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            i = j
            continue
        if ch == "'":
            if i + 1 < n and text[i + 1] == "\\":
                j = i + 2
                while j < n and text[j] != "'":
                    j += 1
                i = j + 1
                continue
            if i + 2 < n and text[i + 2] == "'":
                i += 3
                continue
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def fn_span(lines: list[str], start: int) -> tuple[int, int]:
    depth, opened = 0, False
    for i in range(start, len(lines)):
        for ch in sanitize(lines[i]):
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    return start, i
    raise SystemExit(f"fn at line {start + 1} never closes")


def sub(path: str, old: str, new: str, count: int, label: str) -> None:
    p = ROOT / path
    s = p.read_text()
    found = s.count(old)
    assert found == count, f"{label}: expected {count}, found {found}"
    p.write_text(s.replace(old, new))
    print(f"  {label}")


def drop_fns(path: str, signature: str, label: str) -> int:
    """Remove every `fn <op>(...)` whose first parameter line matches signature."""
    p = ROOT / path
    lines = p.read_text().split("\n")
    removed = 0
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith(signature) and "&mut self" in lines[i + 1]:
            first, last = fn_span(lines, i)
            del lines[first : last + 1]
            removed += 1
            i = first
            continue
        i += 1
    p.write_text("\n".join(lines))
    print(f"  {label}: removed {removed}")
    return removed


# --- 1. trait item -----------------------------------------------------------
sub("crates/tenferro-tensor/src/backend.rs",
    f"    fn {OP}(&mut self, input: &Tensor) -> crate::Result<Tensor>;\n", "",
    1, "trait item")

# --- 2. CUDA: capture the one-shot body, inline it into the read half --------
cuda = ROOT / "crates/tenferro-gpu/src/cubecl/mod.rs"
lines = cuda.read_text().split("\n")
idx = [i for i, l in enumerate(lines) if l == f"    fn {OP}(&mut self, input: &Tensor) -> crate::Result<Tensor> {{"]
assert len(idx) == 1, idx
first, last = fn_span(lines, idx[0])
body = lines[first + 1 : last]
assert len(body) == 1, body
cuda_body = body[0].strip()
read_call = f"        self.{OP}(input.as_tensor())\n"
text = "\n".join(lines)
assert text.count(read_call) == 1, "CUDA read-half fallback not found"
text = text.replace(read_call, f"        {cuda_body}\n")
lines = text.split("\n")
first, last = fn_span(lines, [i for i, l in enumerate(lines) if l == f"    fn {OP}(&mut self, input: &Tensor) -> crate::Result<Tensor> {{"][0])
del lines[first : last + 1]
cuda.write_text("\n".join(lines))
print(f"  CUDA read-half fallback inlined: {cuda_body}")

# --- 3. declaration and delegation lines ------------------------------------
DECL = re.compile(rf"^ *{OP}\(input: &Tensor\) -> [^;\n]+;\n", re.M)
DELEG = re.compile(rf"^ *fn {OP}\(input: &Tensor\) -> [^;\n]+;\n", re.M)
for path in [
    "crates/tenferro-ad/src/eager_backend.rs",
    "crates/tenferro-gpu/src/cubecl/exec_session.rs",
    "crates/tenferro-gpu/src/webgpu/exec_session.rs",
    "crates/tenferro-linalg/tests/integration/backend_errors.rs",
    "crates/tenferro-einsum/tests/integration/session_plan.rs",
    "crates/tenferro-einsum/src/typed_eager_tests.rs",
    "crates/tenferro-cpu/src/tests/cpu_tests/backend_misc.rs",
    "crates/tenferro-fft/tests/backend_capability.rs",
    "crates/tenferro-runtime/tests/integration/session_ops.rs",
]:
    p = ROOT / path
    s = p.read_text()
    s2, n1 = DECL.subn("", s)
    s3, n2 = DELEG.subn("", s2)
    if n1 or n2:
        p.write_text(s3)
        print(f"  {path}: decl {n1}, delegate {n2}")

sub("crates/tenferro-cpu/src/exec_session.rs",
    f"    delegate_with_pool!({OP}(input: &Tensor) => analytic::{OP}_with_pool);\n", "",
    1, "CpuExecSession delegation")

# --- 4. remaining one-shot implementations ----------------------------------
for path in [
    "crates/tenferro-gpu/src/webgpu/mod.rs",
    "crates/tenferro-cpu/src/backend.rs",
    "crates/tenferro-einsum/tests/integration/session_plan.rs",
    "crates/tenferro-tensor/src/tests/backend_default_read_tests.rs",
    "crates/tenferro-einsum/src/eager/tests.rs",
]:
    drop_fns(path, f"fn {OP}(", path.split("/")[-1])

# --- 5. absorb the delegating read halves -----------------------------------
R = r"(?:crate|tenferro_tensor)::backend::read_owned_tensor"
CALL = re.compile(rf"self\.{OP}\(\s*{R}\(\"{OP}\", input\)\?\s*,?\s*\)")


def absorb(path: str, count: int, body_lines: list[str], bind: bool = False) -> None:
    p = ROOT / path
    s = p.read_text()

    def repl(m: re.Match[str]) -> str:
        line_start = s.rfind("\n", 0, m.start()) + 1
        ind = re.match(r" *", s[line_start : m.start()]).group(0)
        head = f'{"let input = " if bind else "let _ = "}tenferro_tensor::backend::read_owned_tensor("{OP}", input)?;\n'
        return head + "".join(f"{ind}{l}\n" for l in body_lines).rstrip("\n")

    s2, n = CALL.subn(repl, s)
    assert n == count, f"{path}: {n} != {count}"
    p.write_text(s2)
    print(f"  absorbed x{n}: {path}")


absorb("crates/tenferro-tensor/src/tests/backend_default_read_tests.rs", 1,
       [f'self.calls.push("{OP}");', "Ok(marker())"])
absorb("crates/tenferro-einsum/tests/integration/session_plan.rs", 2,
       [f'panic!("{OP} should not be called in this test")'])
absorb("crates/tenferro-einsum/src/typed_eager_tests.rs", 1,
       [f'panic!("{OP} should not be called in this test")'])
absorb("crates/tenferro-einsum/src/eager/tests.rs", 1, [f'Err(unexpected("{OP}"))'])
absorb("crates/tenferro-runtime/tests/integration/session_ops.rs", 2,
       [f'panic!("{OP} should not be called in this test")'])
absorb("crates/tenferro-fft/tests/backend_capability.rs", 1,
       [f'panic!("{OP} should not be called by this test")'])
absorb("crates/tenferro-linalg/tests/integration/backend_errors.rs", 1,
       [f'panic!("{OP} should not be called by this test")'])
absorb("crates/tenferro-cpu/src/tests/cpu_tests/backend_misc.rs", 2,
       ["let mut backend = CpuBackend::new();",
        "tenferro_tensor::BackendSessionHost::with_backend_session(&mut backend, |__s| {",
        f"    __s.{OP}_read(TensorRead::from_tensor(input))",
        "})"], bind=True)
absorb("crates/tenferro-ad/src/eager_backend.rs", 1,
       [f"self.inner.{OP}_read(TensorRead::from_tensor(input))"], bind=True)

# --- 6. CPU helper macro ----------------------------------------------------
sub("crates/tenferro-cpu/src/analytic.rs",
    f"define_unary_analytic_dispatch!(\n    {OP},\n    {OP}_with_pool,\n    {OP}_read_with_pool,\n    {CAP},\n    {OP}_elem\n);",
    f"define_unary_analytic_dispatch!(read_only: {OP}, {OP}_read_with_pool, {CAP}, {OP}_elem);",
    1, "analytic.rs read_only invocation")

# --- 7. call sites ----------------------------------------------------------
sub("crates/tenferro-runtime/src/tensor.rs",
    f"        session.{OP}(self)\n",
    f"        session.{OP}_read(TensorRead::from_tensor(self))\n", 1, "runtime TensorSessionOpsExt")
sub("crates/tenferro-ad/src/eager_exec.rs",
    f"StdTensorOp::{CAP} => vec![exec.{OP}(inputs[0])?],",
    f"StdTensorOp::{CAP} => vec![exec.{OP}_read(TensorRead::from_tensor(inputs[0]))?],",
    1, "eager dispatcher")

print("done; run cargo fmt and cargo check --workspace --all-targets next")
