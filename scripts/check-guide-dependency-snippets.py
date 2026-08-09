#!/usr/bin/env python3
"""Compile smoke tests from guide dependency snippets."""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@dataclass(frozen=True)
class GuideCase:
    name: str
    guide_path: str
    required_deps: tuple[str, ...]
    required_features: dict[str, tuple[str, ...]]
    source: str


EINSUM_SOURCE = r"""
use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_einsum::{EagerEinsumExt, TraceContextEinsumExt};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{GraphCompiler, Runtime, TraceContext};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_vec_col_major(
        vec![2, 3],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?;
    let b = Tensor::from_vec_col_major(
        vec![3, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )?;

    let mut trace = TraceContext::new();
    let a_value = trace.input(ProgramInputSpec::new(a.dtype(), [2.into(), 3.into()]))?;
    let b_value = trace.input(ProgramInputSpec::new(b.dtype(), [3.into(), 2.into()]))?;
    let c = trace.einsum(&[a_value, b_value], "ij,jk->ik")?;
    let graph = trace.finish(&[c])?;
    let program = GraphCompiler::new().compile_traced_graph(&graph)?;
    let backend = CpuBackend::new();
    let engine_id = runtime_engine_id()?;
    let mut runtime_builder = Runtime::builder();
    runtime_builder.register_engine(runtime_engine_registration(&backend)?)?;
    runtime_builder.install_extension_module(tenferro_einsum::extension_module::<CpuBackend>(engine_id)?)?;
    let runtime = runtime_builder.build()?;
    let mut outputs = runtime.run_compiled(&program, &[&a, &b])?;
    assert_eq!(outputs.remove(0).shape(), &[2, 2]);

    let ctx = EagerRuntime::new()?;
    let u = ctx.variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
    let v = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0])?)?;
    let outer = [&u, &v].einsum("i,j->ij")?;
    assert_eq!(outer.shape(), &[2, 3]);

    Ok(())
}
"""


LINALG_SOURCE = r"""
use tenferro_ad::AdContext;
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_linalg::LinalgBackend;
use tenferro_runtime::{BackendSessionHost, Tensor, TensorOpsExt};

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> f64 {
    lhs.as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(rhs.as_slice::<f64>().unwrap())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
        .build()?;

    let mut backend = CpuBackend::new();
    let a = Tensor::from_vec_col_major(
        vec![2, 2],
        vec![0.0_f64, 2.0, 1.0, 3.0],
    )?;
    let outputs = backend.with_backend_session(|session| {
        with_cpu_exec_session(session, |exec_session| {
            LinalgBackend::full_piv_lu(exec_session, &a)
        })
        .expect("CpuBackend must expose a CPU execution session")
    })?;
    let p = &outputs[0];
    let l = &outputs[1];
    let u = &outputs[2];
    let q = &outputs[3];
    let parity = &outputs[4];

    let pt = p.transpose(&[1, 0], &mut backend)?;
    let pt_l = pt.matmul(l, &mut backend)?;
    let pt_lu = pt_l.matmul(u, &mut backend)?;
    let reconstructed = pt_lu.matmul(q, &mut backend)?;
    assert!(max_abs_diff(&reconstructed, &a) < 1.0e-12);

    assert_eq!(parity.shape(), &[] as &[usize]);
    let parity_value = parity.as_slice::<f64>().unwrap()[0];
    assert!(parity_value == 1.0 || parity_value == -1.0);

    let b = Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0])?;
    let x = backend.with_backend_session(|session| {
        with_cpu_exec_session(session, |exec_session| {
            LinalgBackend::full_piv_lu_solve(exec_session, &a, &b, false)
        })
        .expect("CpuBackend must expose a CPU execution session")
    })?;
    assert_eq!(x.shape(), &[2, 1]);

    Ok(())
}
"""


TENSOR_SOURCE = r"""
use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{CompareDir, TypedTensorOpsExt};
use tenferro_tensor::{Rank, TypedTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut x = TypedTensor::<f64>::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    )?;
    *x.get_mut(&[0, 1])? = 5.0;
    assert_eq!(*x.get(&[1, 1])?, 4.0);

    let static_rank = TypedTensor::<f64, Rank<2>>::from_vec_col_major(
        [2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    )?;
    assert_eq!(static_rank.rank(), 2);

    let mut backend = CpuBackend::new();
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0])?;
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![4.0, 5.0, 6.0])?;
    let sum = lhs.add(&rhs, &mut backend)?;
    let product = lhs.mul(&rhs, &mut backend)?;
    let mask = sum.compare(&product, CompareDir::Lt, &mut backend)?;
    assert_eq!(mask.as_slice()?, &[false, true, true]);

    let ctx = EagerRuntime::new()?;
    let tracked = ctx.variable_from(Tensor::from_vec_col_major(vec![1], vec![1.0_f64])?)?;
    assert_eq!(tracked.shape(), &[1]);

    Ok(())
}
"""


FFT_SOURCE = r"""
use num_complex::Complex64;
use tenferro_ad::AdContext;
use tenferro_cpu::{runtime_engine_id, runtime_engine_registration, CpuBackend};
use tenferro_fft::{FftNorm, TracedTensorFftExt};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ad = AdContext::builder().build()?;
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )?;
    let y = x.fft(None, -1, FftNorm::Backward)?;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y)?;
    let backend = CpuBackend::new();
    let engine_id = runtime_engine_id()?;
    let mut runtime_builder = Runtime::builder();
    runtime_builder.register_engine(runtime_engine_registration(&backend)?)?;
    runtime_builder.install_extension_module(tenferro_fft::extension_module::<CpuBackend>(engine_id)?)?;
    let runtime = runtime_builder.build()?;
    let mut outputs = runtime.run_compiled(&program, &[])?;
    assert_eq!(outputs.remove(0).shape(), &[4]);

    Ok(())
}
"""


CASES = (
    GuideCase(
        name="einsum",
        guide_path="docs/guides/einsum.md",
        required_deps=(
            "tenferro-runtime",
            "tenferro-cpu",
            "tenferro-ad",
            "tenferro-einsum",
        ),
        required_features={"tenferro-einsum": ("autodiff",)},
        source=EINSUM_SOURCE,
    ),
    GuideCase(
        name="linear-algebra",
        guide_path="docs/guides/linear-algebra.md",
        required_deps=("tenferro-runtime", "tenferro-cpu", "tenferro-ad", "tenferro-linalg"),
        required_features={"tenferro-linalg": ("autodiff",)},
        source=LINALG_SOURCE,
    ),
    GuideCase(
        name="tensor-operations",
        guide_path="docs/guides/tensor-operations.md",
        required_deps=("tenferro-runtime", "tenferro-cpu", "tenferro-tensor", "tenferro-ad"),
        required_features={},
        source=TENSOR_SOURCE,
    ),
    GuideCase(
        name="tenferro-fft",
        guide_path="docs/guides/tenferro-fft.md",
        required_deps=("num-complex", "tenferro-runtime", "tenferro-cpu", "tenferro-ad", "tenferro-fft"),
        required_features={"tenferro-fft": ("autodiff",)},
        source=FFT_SOURCE,
    ),
)


_COMMIT_CHECKOUT_HASH = re.compile(
    r"\bgit\b[^\n]*?\bcheckout\b[^\n]*?"
    r"(?P<commit>(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f]))",
    flags=re.IGNORECASE,
)


def guide_commit_checkout_hashes(root: pathlib.Path) -> list[tuple[pathlib.Path, int, str]]:
    """Return commit-hash checkout commands found in every user guide."""

    guides = sorted((root / "docs" / "guides").rglob("*.md"))
    findings: list[tuple[pathlib.Path, int, str]] = []
    for path in guides:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = _COMMIT_CHECKOUT_HASH.search(line)
            if match:
                findings.append((path, line_number, match.group("commit")))
    return findings


def validate_no_guide_commit_checkout_hashes(root: pathlib.Path) -> None:
    """Reject rebase-sensitive commit pins in all user-facing guides."""

    findings = guide_commit_checkout_hashes(root)
    if findings:
        details = "; ".join(
            f"{path.relative_to(root)}:{line}: git checkout {commit}"
            for path, line, commit in findings
        )
        raise RuntimeError(f"guide commit checkout hash pin found: {details}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default=".", help="Repository root")
    return parser.parse_args()


def extract_dependency_block(root: pathlib.Path, case: GuideCase) -> str:
    path = root / case.guide_path
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r"```toml\n(.*?)\n```", text, flags=re.DOTALL):
        block = match.group(1).strip()
        if "[dependencies]" in block:
            return block
    raise RuntimeError(f"{case.guide_path}: no TOML [dependencies] block found")


def load_dependency_table(block: str, case: GuideCase) -> dict:
    if tomllib is None:
        raise RuntimeError("Python 3.11+ is required to parse guide dependency snippets")
    parsed = tomllib.loads(block)
    dependencies = parsed.get("dependencies")
    if not isinstance(dependencies, dict):
        raise RuntimeError(f"{case.guide_path}: dependency block does not parse as TOML")
    return dependencies


def dependency_features(spec: object) -> set[str]:
    if isinstance(spec, dict):
        features = spec.get("features", [])
        if isinstance(features, list):
            return {str(feature) for feature in features}
    return set()


def validate_dependency_contract(dependencies: dict, case: GuideCase) -> None:
    missing = [dep for dep in case.required_deps if dep not in dependencies]
    if missing:
        raise RuntimeError(f"{case.guide_path}: missing dependencies: {', '.join(missing)}")

    for dep, features in case.required_features.items():
        enabled = dependency_features(dependencies[dep])
        missing_features = [feature for feature in features if feature not in enabled]
        if missing_features:
            joined = ", ".join(missing_features)
            raise RuntimeError(f"{case.guide_path}: {dep} missing features: {joined}")


def absolutize_repo_paths(block: str, root: pathlib.Path) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        path = pathlib.PurePosixPath(raw)
        parts = path.parts
        if "crates" in parts:
            crate_index = parts.index("crates")
            absolute = root.joinpath(*parts[crate_index:]).resolve()
            return f'path = "{absolute}"'
        return match.group(0)

    return re.sub(r'path\s*=\s*"([^"]+)"', replace, block)


def cargo_environment(target_dir: pathlib.Path) -> dict[str, str]:
    """Return the environment for an independent guide Cargo workspace."""

    env = os.environ.copy()
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "0")
    env["CARGO_TARGET_DIR"] = str(target_dir)
    return env


def run_case(root: pathlib.Path, target_dir: pathlib.Path, case: GuideCase) -> None:
    block = extract_dependency_block(root, case)
    dependencies = load_dependency_table(block, case)
    validate_dependency_contract(dependencies, case)

    with tempfile.TemporaryDirectory(prefix=f"tenferro-guide-{case.name}-") as tmp:
        tmp_path = pathlib.Path(tmp)
        (tmp_path / "src").mkdir()
        (tmp_path / "Cargo.toml").write_text(
            textwrap.dedent(
                f"""
                [package]
                name = "guide-dependency-{case.name}"
                version = "0.0.0"
                edition = "2021"
                publish = false

                [workspace]

                {absolutize_repo_paths(block, root)}
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (tmp_path / "src/main.rs").write_text(
            textwrap.dedent(case.source).lstrip(),
            encoding="utf-8",
        )
        result = subprocess.run(
            ["cargo", "run", "--quiet", "--manifest-path", str(tmp_path / "Cargo.toml")],
            cwd=root,
            env=cargo_environment(target_dir),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{case.guide_path}: dependency smoke test failed")


def main() -> int:
    args = parse_args()
    root = pathlib.Path(args.root_dir).resolve()
    target_dir = root / "target" / "guide-snippet-check"

    try:
        validate_no_guide_commit_checkout_hashes(root)
        for case in CASES:
            run_case(root, target_dir, case)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 1

    print(f"guide-dependency-snippets-ok: {len(CASES)} guides verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
