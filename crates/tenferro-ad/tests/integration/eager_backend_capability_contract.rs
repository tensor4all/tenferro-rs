#[test]
fn eager_backend_capability_boundary() {
    if std::env::var_os("NEXTEST").is_some()
        && std::env::var("CARGO_NET_OFFLINE").is_ok_and(|value| value == "true" || value == "1")
    {
        eprintln!("skipping compile-only trybuild contract in an offline nextest archive");
        return;
    }

    let tests = trybuild::TestCases::new();

    tests.compile_fail("tests/ui/eager_backend_owner_private.rs");
    tests.compile_fail("tests/ui/eager_backend_mutation_removed.rs");
    tests.compile_fail("tests/ui/eager_session_no_owner_projection.rs");
    tests.compile_fail("tests/ui/eager_session_mutation_removed.rs");
    tests.pass("tests/ui/eager_session_positive_contract.rs");
}
