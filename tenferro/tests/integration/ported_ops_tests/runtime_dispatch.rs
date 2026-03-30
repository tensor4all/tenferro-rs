use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(path)).unwrap()
}

fn repo_files(paths: &[&str]) -> String {
    paths
        .iter()
        .map(|path| repo_file(path))
        .collect::<Vec<_>>()
        .join("\n")
}

// IMPORTANT: Do not delete or weaken these runtime-dispatch structure tests.
// They guard the generic/capability-driven architecture that keeps tenferro
// extensible as more scalar, analytic, and linalg families are added.

#[test]
fn unary_and_reduction_entrypoints_route_through_runtime_dispatch() {
    let scalar_builders = [
        repo_file("../internal/tenferro-internal-ad-ops/src/ops/scalar/primal.rs"),
        repo_file("../internal/tenferro-internal-ad-ops/src/ops/scalar/ad/common.rs"),
        repo_file("../internal/tenferro-internal-ad-ops/src/ops/scalar/ad/unary.rs"),
        repo_file("../internal/tenferro-internal-ad-ops/src/ops/scalar/ad/binary.rs"),
        repo_file("../internal/tenferro-internal-ad-ops/src/ops/scalar/ad/reduction.rs"),
    ]
    .join("\n");
    let ad_api = repo_files(&[
        "src/ops/ad/mod.rs",
        "../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/eager.rs",
        "../internal/tenferro-internal-ad-ops/src/ops/ad/pullback.rs",
        "../internal/tenferro-internal-ad-ops/src/ops/ad/scalar_eager.rs",
    ]);
    let ad_builders = [
        repo_file("../internal/tenferro-internal-ad-ops/src/ops/einsum/ad.rs"),
        repo_file("../internal/tenferro-internal-ad-ops/src/ops/reduction/ad.rs"),
        repo_file("../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/common.rs"),
        repo_file("../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/single.rs"),
        repo_file("../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/slogdet.rs"),
        repo_file("../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/svd_qr.rs"),
        repo_file("../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/lu_lstsq.rs"),
        repo_file("../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/spectral.rs"),
    ]
    .join("\n");

    assert!(
        !scalar_builders.contains("with_cpu_runtime("),
        "scalar generic builders should not depend on with_cpu_runtime(...) once runtime dispatch is centralized"
    );
    assert!(
        !ad_api.contains("with_cpu_runtime("),
        "AD entrypoints should route through runtime dispatch instead of with_cpu_runtime(...)"
    );
    assert!(
        !ad_api.contains("with_runtime_cpu_only(\"einsum"),
        "einsum AD entrypoints should dispatch through runtime-aware helpers instead of with_runtime_cpu_only(...)"
    );
    assert!(
        !ad_api.contains("with_runtime_cpu_only(\"solve_triangular_rrule"),
        "solve_triangular_rrule should dispatch through runtime-aware linalg helpers instead of with_runtime_cpu_only(...)"
    );
    assert!(
        !ad_builders.contains("with_cpu_runtime("),
        "AD builders should route through runtime dispatch instead of with_cpu_runtime(...)"
    );
    assert!(
        !ad_builders.contains("with_runtime_cpu_only("),
        "AD builders should not hard-code CPU-only runtime dispatch once shared runtime-aware helpers exist"
    );
}

#[test]
fn linalg_entrypoints_report_runtime_capability_failures() {
    let linalg_builders = repo_files(&[
        "src/ops/linalg/mod.rs",
        "src/ops/linalg/common.rs",
        "src/ops/linalg/primal/mod.rs",
        "src/ops/linalg/primal/factorizations.rs",
        "src/ops/linalg/primal/solve.rs",
        "src/ops/linalg/primal/spectral.rs",
        "src/ops/linalg/primal/tensorized.rs",
    ]);
    assert!(
        !linalg_builders.contains("with_cpu_runtime("),
        "linalg builders should use shared runtime dispatch instead of with_cpu_runtime(...)"
    );
    assert!(
        !linalg_builders.contains("with_runtime_cpu_only("),
        "linalg builders should use shared runtime/capability dispatch instead of with_runtime_cpu_only(...)"
    );
}

#[test]
fn eager_dyn_extra_entrypoints_use_matching_linalg_capabilities() {
    let eager = repo_files(&[
        "../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/eager.rs",
        "../internal/tenferro-internal-ad-linalg/src/ops/linalg/ad/eager_impl.rs",
    ]);
    let start = eager
        .find("fn lu_factor_ex_dyn_impl")
        .expect("lu_factor_ex_dyn_impl should exist");
    let end = eager[start..]
        .find("fn lu_solve_dyn_impl")
        .map(|offset| start + offset)
        .expect("lu_solve_dyn_impl should delimit lu_factor_ex_dyn_impl");
    let body = &eager[start..end];

    assert!(
        body.contains("LinalgCapabilityOp::LuFactorEx"),
        "lu_factor_ex_dyn_impl must dispatch with the LuFactorEx capability bit"
    );
}

#[test]
fn structured_einsum_uses_shared_runtime_dispatch_path() {
    let structured_einsum = repo_file("src/structured/einsum.rs");

    assert!(
        !structured_einsum.contains("with_runtime_cpu_only("),
        "structured einsum should share the same runtime-dispatch path as the rest of the tenferro frontend"
    );
    assert!(
        !structured_einsum.contains("CpuBackend")
            && !structured_einsum.contains("CudaBackend")
            && !structured_einsum.contains("RocmBackend"),
        "structured einsum should dispatch through shared generic helpers rather than spelling out backend triples"
    );
    assert!(
        !structured_einsum.contains("CpuContext")
            && !structured_einsum.contains("CudaContext")
            && !structured_einsum.contains("RocmContext"),
        "structured einsum should not hard-code runtime context triples once shared dispatch exists"
    );
}

#[test]
fn runtime_helpers_stay_capability_driven() {
    let runtime_helpers = repo_file("src/ops/common.rs");
    let runtime_dispatch = repo_file("../internal/tenferro-internal-runtime/src/dispatch.rs");
    let contracts = repo_file("../internal/tenferro-internal-runtime/src/contracts.rs");
    let reduction_builders =
        repo_file("../internal/tenferro-internal-ad-ops/src/ops/reduction/ad.rs");

    assert!(
        !runtime_helpers.contains("TypeId::of::<Backend>()"),
        "runtime helpers should dispatch through capability-aware generic helpers rather than backend type checks"
    );
    assert!(
        !runtime_helpers.contains("compress_pullback_like_in_backend::<CpuBackend"),
        "runtime helpers should not hard-code CPU-only compression paths once einsum runtime dispatch exists"
    );
    assert!(
        !reduction_builders.contains("unsupported_runtime_capability(\"sum_ad_pullback\""),
        "sum_ad pullback should rely on transfer-aware runtime helpers rather than hard-coded CPU-only runtime failures"
    );
    assert!(
        !contracts.contains("CpuBackend")
            && !contracts.contains("CudaBackend")
            && !contracts.contains("RocmBackend"),
        "semantic runtime value traits should not hard-code the backend matrix"
    );
    assert!(
        !contracts.contains("CpuContext")
            && !contracts.contains("CudaContext")
            && !contracts.contains("RocmContext"),
        "semantic runtime value traits should not hard-code runtime context triples"
    );
    assert!(
        !contracts.contains("CpuLinalgScalar"),
        "semantic runtime value traits should not leak CPU-specific scalar contracts into the generic tenferro surface"
    );
    assert!(
        runtime_dispatch.contains("trait RuntimeSlot"),
        "runtime dispatch should centralize concrete runtime slot metadata instead of repeating ad hoc backend/context wiring"
    );
}
