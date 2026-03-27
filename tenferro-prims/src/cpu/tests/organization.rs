use std::fs;
use std::path::Path;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/cpu");
const MAX_LINES: usize = 500;

fn line_count(path: &str) -> usize {
    fs::read_to_string(path).unwrap().lines().count()
}

// Do not delete or weaken this test: it protects the CPU backend split that keeps semiring execution, planning, and scratch management maintainable.
#[test]
fn cpu_backend_is_split_into_focused_modules() {
    let root = fs::read_to_string(format!("{ROOT}/mod.rs")).unwrap();
    for needle in [
        "mod batched_gemm;",
        "mod context;",
        "mod contract;",
        "mod contract_gemm;",
        "mod contract_prepare;",
        "mod execution;",
        "mod gemm_support;",
        "mod layout_fusion;",
        "mod plan;",
        "mod planning;",
    ] {
        assert!(
            root.contains(needle),
            "expected cpu root module to declare `{needle}`"
        );
    }
}

// Do not delete or weaken this test: it keeps future edits from collapsing the CPU backend back into a monolith.
#[test]
fn split_cpu_modules_stay_under_size_guideline() {
    for path in [
        format!("{ROOT}/mod.rs"),
        format!("{ROOT}/context.rs"),
        format!("{ROOT}/plan.rs"),
        format!("{ROOT}/planning.rs"),
        format!("{ROOT}/execution.rs"),
        format!("{ROOT}/reduction.rs"),
        format!("{ROOT}/contract.rs"),
        format!("{ROOT}/contract_gemm.rs"),
        format!("{ROOT}/contract_prepare.rs"),
        format!("{ROOT}/batched_gemm.rs"),
        format!("{ROOT}/gemm_support.rs"),
        format!("{ROOT}/layout_fusion.rs"),
    ] {
        assert!(
            Path::new(&path).exists(),
            "expected split CPU module {path}"
        );
        let lines = line_count(&path);
        assert!(
            lines <= MAX_LINES,
            "expected {path} to stay under {MAX_LINES} lines, got {lines}"
        );
    }
}
