use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(path)).unwrap()
}

#[test]
fn capability_checked_composite_paths_do_not_require_cpu_type_checks() {
    let frules = repo_file("src/frules.rs");
    let rrules = repo_file("src/rrules.rs");

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
    let primal = repo_file("src/primal.rs");

    assert!(
        !primal.contains("TypeId::of::<C::Backend>()"),
        "primal linalg paths should not identify unsupported runtimes through direct backend type checks"
    );
}
