#[test]
fn as_view_paths_do_not_allocate_or_clone_storage() {
    let cargo = std::env::var_os("CARGO").expect("cargo sets CARGO for tests");
    let status = std::process::Command::new(cargo)
        .args([
            "test",
            "-p",
            "tenferro-tensor",
            "--lib",
            "tests::types_tests::as_view_paths_do_not_allocate_or_clone_storage",
            "--quiet",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("launch private P3 allocation proof");
    assert!(status.success(), "private P3 allocation proof failed");
}
