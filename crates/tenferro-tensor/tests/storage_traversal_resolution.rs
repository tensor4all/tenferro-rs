#[test]
fn p4_traversal_resolution_artifact_runs_real_proofs() {
    let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO for tests");
    let status = std::process::Command::new(cargo)
        .args([
            "test",
            "-p",
            "tenferro-tensor",
            "--lib",
            "storage::tests::prepared_access::provider_resolution",
            "--quiet",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("launch private P4 traversal proof tests");
    assert!(status.success(), "private P4 traversal proof tests failed");
}
