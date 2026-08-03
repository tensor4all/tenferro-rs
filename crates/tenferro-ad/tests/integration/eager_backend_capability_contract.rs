#[test]
fn eager_backend_capability_boundary() {
    let tests = trybuild::TestCases::new();

    tests.compile_fail("tests/ui/eager_backend_owner_private.rs");
    tests.compile_fail("tests/ui/eager_backend_mutation_removed.rs");
    tests.compile_fail("tests/ui/eager_session_no_owner_projection.rs");
    tests.compile_fail("tests/ui/eager_session_mutation_removed.rs");
    tests.pass("tests/ui/eager_session_positive_contract.rs");
}
