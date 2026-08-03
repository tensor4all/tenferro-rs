#[test]
fn eager_backend_capability_boundary() {
    if std::env::var_os("TENFERRO_TRYBUILD_RUSTFLAGS").is_some() {
        let cargo_wrapper = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/ci/trybuild-cargo.py");
        std::env::set_var("CARGO", cargo_wrapper);
    }
    let tests = trybuild::TestCases::new();

    tests.compile_fail("tests/ui/eager_backend_owner_private.rs");
    tests.compile_fail("tests/ui/eager_backend_mutation_removed.rs");
    tests.compile_fail("tests/ui/eager_session_no_owner_projection.rs");
    tests.compile_fail("tests/ui/eager_session_mutation_removed.rs");
    tests.pass("tests/ui/eager_session_positive_contract.rs");
}
