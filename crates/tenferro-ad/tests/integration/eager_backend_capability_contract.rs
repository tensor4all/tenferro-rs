#[test]
fn eager_backend_capability_boundary() {
    if std::env::var_os("NEXTEST").is_some() {
        eprintln!("skipping compile-only trybuild contract under nextest");
        return;
    }

    let tests = trybuild::TestCases::new();

    tests.compile_fail("tests/ui/eager_session_no_owner_projection.rs");
    tests.compile_fail("tests/ui/eager_session_mutation_removed.rs");
    tests.pass("tests/ui/eager_session_positive_contract.rs");
    // #2602: the session spellings compile; the legacy engine-taking receivers
    // and duplicate copy aliases stay compile errors.
    tests.pass("tests/ui/eager_session_ad_positive_contract.rs");
    tests.compile_fail("tests/ui/eager_legacy_engine_receivers_removed.rs");
}
