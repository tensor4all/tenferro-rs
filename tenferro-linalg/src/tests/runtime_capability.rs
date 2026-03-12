use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(path)).unwrap()
}

// IMPORTANT: Do not delete or weaken these tests.
// They are the regression guard that keeps linalg on capability-based runtime
// checks instead of slipping back to CPU/backend-name special cases.

#[test]
fn capability_checked_composite_paths_do_not_require_cpu_type_checks() {
    let frules = repo_file("src/frules/mod.rs");
    let rrules = repo_file("src/rrules/mod.rs");

    assert!(
        !frules.contains("ensure_cpu_backend::<"),
        "forward rules should gate on capability-driven backend contracts rather than ensure_cpu_backend(...)"
    );
    assert!(
        !rrules.contains("ensure_cpu_backend::<"),
        "reverse rules should gate on capability-driven backend contracts rather than ensure_cpu_backend(...)"
    );
}

#[test]
fn cpu_only_kernel_paths_fail_through_capability_not_backend_name() {
    let primal = repo_file("src/primal/mod.rs");

    assert!(
        !primal.contains("TypeId::of::<C::Backend>()"),
        "primal linalg paths should not identify unsupported runtimes through direct backend type checks"
    );
}

#[test]
fn public_linalg_layers_do_not_spell_cpu_scalar_contracts() {
    let public_layers = [
        "src/lib.rs",
        "src/primal/mod.rs",
        "src/frules/mod.rs",
        "src/rrules/mod.rs",
        "src/ad_helpers/mod.rs",
        "src/backend/tensor_api.rs",
        "src/backend/tensor_context.rs",
    ]
    .into_iter()
    .map(repo_file)
    .collect::<Vec<_>>()
    .join("\n");

    assert!(
        !public_layers.contains("CpuLinalgScalar"),
        "public linalg layers should depend on backend-generic kernel scalar contracts rather than CPU-specific scalar names"
    );
}
