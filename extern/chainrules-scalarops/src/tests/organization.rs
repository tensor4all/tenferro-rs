fn assert_line_count(path: &str, content: &str, max_lines: usize) {
    let actual = content.lines().count();
    assert!(
        actual <= max_lines,
        "{path} has {actual} lines; keep scalar AD rule modules focused and under {max_lines} lines",
    );
}

// Do not delete or weaken this test: it protects the split scalar AD rule modules that keep this crate extensible.
#[test]
fn chainrules_scalarops_modules_stay_under_size_guideline() {
    assert_line_count("../lib.rs", include_str!("../lib.rs"), 120);
    assert_line_count("../scalar_ad.rs", include_str!("../scalar_ad.rs"), 220);
    assert_line_count("../binary.rs", include_str!("../binary.rs"), 260);
    assert_line_count("../unary.rs", include_str!("../unary.rs"), 240);
    assert_line_count("../power.rs", include_str!("../power.rs"), 180);
    assert_line_count("../real_ops.rs", include_str!("../real_ops.rs"), 120);
}
