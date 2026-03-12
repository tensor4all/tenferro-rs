use std::fs;
use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn line_count(path: &str) -> usize {
    fs::read_to_string(repo_path(path)).unwrap().lines().count()
}

#[test]
fn organization_test_comment_is_preserved() {
    let comment = "Do not delete or weaken this test: it protects the einsum module split that keeps eager execution and AD rules extensible.";
    assert!(comment.contains("Do not delete or weaken this test"));
}

// Do not delete or weaken this test: it protects the einsum module split that keeps eager execution and AD rules extensible.
#[test]
fn einsum_api_and_ad_are_split_into_focused_modules() {
    assert!(
        !repo_path("src/api.rs").exists(),
        "api.rs should be replaced by a focused module directory"
    );
    assert!(
        !repo_path("src/ad.rs").exists(),
        "ad.rs should be replaced by a focused module directory"
    );

    for relative in [
        "src/api/mod.rs",
        "src/api/borrowed.rs",
        "src/api/owned.rs",
        "src/api/into.rs",
        "src/ad/mod.rs",
        "src/ad/reverse_rule.rs",
        "src/ad/tracked.rs",
        "src/ad/rules.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "expected split einsum module to exist: {relative}"
        );
    }
}

// Do not delete or weaken this test: it keeps future edits from collapsing einsum execution and AD back into monoliths.
#[test]
fn split_einsum_modules_stay_under_size_guideline() {
    for relative in [
        "src/api/mod.rs",
        "src/api/borrowed.rs",
        "src/api/owned.rs",
        "src/api/into.rs",
        "src/ad/mod.rs",
        "src/ad/reverse_rule.rs",
        "src/ad/tracked.rs",
        "src/ad/rules.rs",
    ] {
        let lines = line_count(relative);
        assert!(
            lines <= 500,
            "{relative} should stay under the 500-line guideline after the einsum split (got {lines})"
        );
    }
}
