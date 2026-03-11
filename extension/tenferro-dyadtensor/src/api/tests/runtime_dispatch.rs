use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(path)).unwrap()
}

#[test]
fn unary_and_reduction_entrypoints_route_through_runtime_dispatch() {
    let scalar_builders = [
        repo_file("src/api/scalar_ad_builders/common.rs"),
        repo_file("src/api/scalar_ad_builders/unary.rs"),
        repo_file("src/api/scalar_ad_builders/binary.rs"),
        repo_file("src/api/scalar_ad_builders/reduction.rs"),
    ]
    .join("\n");
    let ad_api = [
        repo_file("src/api/ad.rs"),
        repo_file("src/api/ad/scalar_eager.rs"),
    ]
    .join("\n");
    let ad_builders = repo_file("src/api/ad_builders.rs");

    assert!(
        !scalar_builders.contains("with_cpu_runtime("),
        "scalar generic builders should not depend on with_cpu_runtime(...) once runtime dispatch is centralized"
    );
    assert!(
        !ad_api.contains("with_cpu_runtime("),
        "AD entrypoints should route through runtime dispatch instead of with_cpu_runtime(...)"
    );
    assert!(
        !ad_builders.contains("with_cpu_runtime("),
        "AD builders should route through runtime dispatch instead of with_cpu_runtime(...)"
    );
}

#[test]
fn linalg_entrypoints_report_runtime_capability_failures() {
    let linalg_builders = repo_file("src/api/linalg_builders.rs");
    let primal_builders = repo_file("src/api/primal_builders.rs");

    assert!(
        !linalg_builders.contains("with_cpu_runtime("),
        "linalg builders should use shared runtime dispatch instead of with_cpu_runtime(...)"
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
}
