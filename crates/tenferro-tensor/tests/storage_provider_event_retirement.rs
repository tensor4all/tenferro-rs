#[test]
fn p4_provider_release_artifact_runs_real_proofs() {
    let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO for tests");
    let status = std::process::Command::new(cargo)
        .args([
            "test",
            "-p",
            "tenferro-tensor",
            "--lib",
            "storage::tests::retirement",
            "--quiet",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("launch private P4 retirement proof tests");
    assert!(status.success(), "private P4 retirement proof tests failed");
}
