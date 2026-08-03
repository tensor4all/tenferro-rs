#[test]
fn p2_root_claims_artifact_runs_private_runtime_proofs() {
    // The storage module is intentionally crate-private, so the ledger's
    // integration artifact delegates to the real unit-test proof surface
    // instead of exporting a test-only owner API.
    let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO for tests");
    let status = std::process::Command::new(cargo)
        .args([
            "test",
            "-p",
            "tenferro-tensor",
            "--lib",
            "storage::tests::root_claims",
            "--quiet",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("launch private root-claims proof tests");
    assert!(status.success(), "private root-claims proof tests failed");
}
