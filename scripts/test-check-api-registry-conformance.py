#!/usr/bin/env python3
"""Self-test of scripts/check-api-registry-conformance.py (#2602).

Each case copies the committed snapshots (the compiled surface), reintroduces
one forbidden shape exactly as `cargo public-api` would render it after a
regeneration, and requires the conformance check to fail. The unmodified copy
must pass.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_api_registry_conformance", ROOT / "scripts/check-api-registry-conformance.py"
)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)

R = "tenferro_runtime::error::Result"
ET = "tenferro_ad::EagerTensor"
T = "tenferro_tensor::types::Tensor"
EXEC = "&mut dyn tenferro_runtime::runtime::engine::ExecutionEngine"
ENGINE = "&mut dyn tenferro_tensor::backend::Engine"
SESSION = "pub fn tenferro_ad::EagerSession<'_>::"
GRAD_SIG = f"{SESSION}grad(&mut self, &{ET}, &{ET}) -> {R}<{ET}>"
BACKWARD_SIG = f"{SESSION}backward(&mut self, &{ET}) -> {R}<tenferro_ad::Gradients>"

# Lines that must each fail the check when added to a regenerated snapshot.
REINTRODUCED = {
    "legacy engine-taking backward": f"pub fn tenferro_ad::EagerTensor::backward(&self, {EXEC}) -> {R}<tenferro_ad::Gradients>",
    "legacy engine-taking runtime grad": f"pub fn tenferro_ad::EagerRuntime::grad(&alloc::sync::Arc<Self>, {EXEC}, &{ET}, &{ET}) -> {R}<{ET}>",
    "legacy tensor detach": f"pub fn tenferro_ad::EagerTensor::detach(&self, {ENGINE}) -> {R}<Self>",
    "duplicate copy alias on the tensor": f"pub fn tenferro_ad::EagerTensor::to_tensor(&self, {ENGINE}) -> {R}<{T}>",
    "duplicate copy alias on the session": f"{SESSION}to_tensor(&mut self, &{ET}) -> {R}<{T}>",
    "re-exported duplicate copy alias": f"pub fn tenferro_ad::eager::EagerTensor::duplicate_value(&self, {ENGINE}) -> {R}<{T}>",
    "second engine accessor": f"{SESSION}backend_session(&mut self) -> {ENGINE}",
    "engine-free tensor leaf alias": f"pub fn tenferro_ad::EagerTensor::from_tensor_in({T}, alloc::sync::Arc<tenferro_ad::EagerRuntime>) -> {R}<Self>",
    "engine argument on a session method": f"{SESSION}exp_on(&mut self, &{ET}, {ENGINE}) -> {R}<{ET}>",
    "engine-taking free function over eager tensors": f"pub fn tenferro_ad::eager::run_eager({EXEC}, &[&{ET}]) -> {R}<{ET}>",
    # Shapes from the independent review: an excepted name with an engine
    # argument, a changed accessor type, generic/where/concrete engines, other
    # engine traits, copy aliases outside the vocabulary, `pub unsafe fn`.
    "engine argument under an excepted name": f"pub fn tenferro_ad::EagerRuntime::variable_from(&alloc::sync::Arc<Self>, {EXEC}, {T}) -> {R}<{ET}>",
    "engine argument on an _optional form": f"{SESSION}grad_optional(&mut self, {EXEC}, &{ET}, &{ET}) -> {R}<core::option::Option<{ET}>>",
    "accessor returning the other engine type": f"{SESSION}engine(&mut self) -> {ENGINE}",
    "engine as a generic bound": f"pub fn tenferro_ad::EagerTensor::reduce<E: tenferro_runtime::runtime::engine::ExecutionEngine + ?Sized>(&self, &mut E) -> {R}<{ET}>",
    "engine in a where clause": f"pub fn tenferro_ad::EagerTensor::reduce<E>(&self, &mut E) -> {R}<{ET}> where E: tenferro_runtime::runtime::engine::ExecutionEngine",
    "concrete engine type": f"pub fn tenferro_ad::EagerTensor::sum_on(&self, &mut tenferro_cpu::CpuEngine) -> {R}<{ET}>",
    "other engine trait": f"pub fn tenferro_ad::EagerTensor::sum_on(&self, &mut dyn tenferro_runtime::ExtensionEngine) -> {R}<{ET}>",
    "copy alias outside the vocabulary": f"{SESSION}copy(&mut self, &{ET}) -> {R}<{T}>",
    "copy alias named clone_value": f"{SESSION}clone_value(&mut self, &{ET}) -> {R}<{T}>",
    "unsafe copy alias": f"pub unsafe fn tenferro_ad::EagerTensor::to_tensor(&self, {ENGINE}) -> {R}<{T}>",
}


def copy_surface(tmp: pathlib.Path) -> pathlib.Path:
    surface = tmp / "api"
    shutil.copytree(ROOT / "api", surface)
    return surface


def edit(path: pathlib.Path, add: list[str] = (), remove: list[str] = ()) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for r in remove:
        assert r in lines, f"fixture line not in {path.name}: {r}"
        lines = [line for line in lines if line != r]
    lines.extend(add)
    path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")


def errors_after(add: list[str] = (), remove: list[str] = (), file: str = "tenferro-ad.txt") -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        surface = copy_surface(pathlib.Path(tmp))
        edit(surface / file, add=add, remove=remove)
        return CHECKER.check(surface, ROOT)


def test_committed_surface_conforms() -> None:
    assert CHECKER.check(ROOT / "api", ROOT) == []


def test_reintroduced_shapes_fail_in_every_profile() -> None:
    for profile in ("tenferro-ad.txt", "tenferro-ad+native-cuda.txt"):
        for label, line in REINTRODUCED.items():
            errors = errors_after(add=[line], file=profile)
            assert errors, f"{label} in {profile} passed the conformance check:\n  {line}"


def test_extension_trait_method_on_the_session_is_in_scope() -> None:
    trait = "tenferro_linalg::EagerLinalgOps"
    errors = errors_after(
        add=[
            f"impl {trait} for tenferro_ad::EagerSession<'_>",
            f"pub fn {trait}::to_tensor(&mut self, &{ET}) -> {R}<{T}>",
        ],
        file="tenferro-linalg.txt",
    )
    assert any("undeclared eager entry point" in e for e in errors), errors


def test_trait_impl_with_where_clause_is_attributed() -> None:
    trait = "tenferro_ad::SomeExt"
    errors = errors_after(
        add=[
            f"impl<T> {trait} for tenferro_ad::EagerTensor where T: core::marker::Send",
            f"pub fn {trait}::to_tensor(&self, {ENGINE}) -> {R}<{T}>",
        ]
    )
    assert any("undeclared eager entry point" in e for e in errors), errors


def test_missing_declared_signature_fails() -> None:
    errors = errors_after(remove=[GRAD_SIG])
    assert any("missing declared signature of `grad`" in e for e in errors), errors


def test_changed_declared_signature_fails() -> None:
    changed = BACKWARD_SIG.replace("(&mut self, ", f"(&mut self, {EXEC}, ")
    errors = errors_after(add=[changed], remove=[BACKWARD_SIG], file="tenferro-ad+native-cuda.txt")
    assert any("missing declared signature of `backward`" in e for e in errors), errors
    assert any("names an engine" in e for e in errors), errors


def test_stale_exception_symbol_fails() -> None:
    line = f"pub fn tenferro_ad::GradientValue::to_tensor(&self, {ENGINE}) -> {R}<{T}>"
    errors = errors_after(remove=[line])
    assert any("exception E12 lists a signature that is not public" in e for e in errors), errors


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
