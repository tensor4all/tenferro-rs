#[test]
fn p4_prepared_validation_artifact_runs_real_proofs() {
    let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO for tests");
    let status = std::process::Command::new(cargo)
        .args([
            "test",
            "-p",
            "tenferro-tensor",
            "--lib",
            "storage::tests::prepared_access::checked_layout",
            "--quiet",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("launch private P4 validation proof tests");
    assert!(status.success(), "private P4 validation proof tests failed");
}
