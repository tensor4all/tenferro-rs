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
// They guard the generic/capability-driven architecture that keeps dyadtensor
// extensible as more scalar, analytic, and linalg families are added.

#[test]
fn unary_and_reduction_entrypoints_route_through_runtime_dispatch() {
    let scalar_builders = [
        repo_file("src/api/scalar_ad_builders/common.rs"),
        repo_file("src/api/scalar_ad_builders/unary.rs"),
        repo_file("src/api/scalar_ad_builders/binary.rs"),
        repo_file("src/api/scalar_ad_builders/reduction.rs"),
    ]
    .join("\n");
    let ad_api = repo_files(&[
        "src/api/ad/mod.rs",
        "src/api/ad/eager_linalg.rs",
        "src/api/ad/pullback.rs",
        "src/api/ad/scalar_eager.rs",
    ]);
    let ad_builders = [
        repo_file("src/api/ad_builders/common.rs"),
        repo_file("src/api/ad_builders/einsum.rs"),
        repo_file("src/api/ad_builders/reduction.rs"),
        repo_file("src/api/ad_builders/linalg_single.rs"),
        repo_file("src/api/ad_builders/linalg_multi.rs"),
        repo_file("src/api/ad_builders/linalg_multi/svd_qr.rs"),
        repo_file("src/api/ad_builders/linalg_multi/lu_lstsq.rs"),
        repo_file("src/api/ad_builders/linalg_multi/spectral.rs"),
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
        "src/api/linalg_builders/mod.rs",
        "src/api/linalg_builders/common.rs",
        "src/api/linalg_builders/factorizations.rs",
        "src/api/linalg_builders/solve.rs",
        "src/api/linalg_builders/spectral.rs",
        "src/api/linalg_builders/tensorized.rs",
    ]);
    let primal_builders = repo_file("src/api/primal_builders.rs");

    assert!(
        !linalg_builders.contains("with_cpu_runtime("),
        "linalg builders should use shared runtime dispatch instead of with_cpu_runtime(...)"
    );
    assert!(
        !linalg_builders.contains("with_runtime_cpu_only("),
        "linalg builders should use shared runtime/capability dispatch instead of with_runtime_cpu_only(...)"
    );
    assert!(
        !primal_builders.contains("with_runtime_cpu_only("),
        "primal builders should dispatch through runtime-aware helpers instead of with_runtime_cpu_only(...)"
    );
}

#[test]
fn structured_einsum_uses_shared_runtime_dispatch_path() {
    let structured_einsum = repo_file("src/structured/einsum.rs");

    assert!(
        !structured_einsum.contains("with_runtime_cpu_only("),
        "structured einsum should share the same runtime-dispatch path as the rest of dyadtensor"
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
    let runtime_helpers = repo_file("src/api/runtime.rs");
    let runtime_dispatch = repo_file("src/api/runtime_dispatch.rs");
    let contracts = repo_file("src/api/contracts.rs");
    let reduction_builders = repo_file("src/api/ad_builders/reduction.rs");

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
        "semantic runtime value traits should not leak CPU-specific scalar contracts into the generic dyadtensor surface"
    );
    assert!(
        runtime_dispatch.contains("trait RuntimeSlot"),
        "runtime dispatch should centralize concrete runtime slot metadata instead of repeating ad hoc backend/context wiring"
    );
}
