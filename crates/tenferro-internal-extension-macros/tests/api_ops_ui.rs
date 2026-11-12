//! Compile-fail diagnostics of `api_ops!`.

#[test]
fn api_ops_diagnostics() {
    if std::env::var_os("NEXTEST").is_some() {
        eprintln!("skipping compile-only trybuild contract under nextest");
        return;
    }

    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/api_ops_unknown_key.rs");
    tests.compile_fail("tests/ui/api_ops_undeclared_layer.rs");
    tests.compile_fail("tests/ui/api_ops_bad_into_mode.rs");
    tests.compile_fail("tests/ui/api_ops_missing_example.rs");
    tests.compile_fail("tests/ui/api_ops_signature_unlisted_layer.rs");
}
