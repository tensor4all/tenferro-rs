use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .to_path_buf()
}

#[test]
fn small_append_benchmark_pins_issue_1750_workflow() {
    let root = workspace_root();
    let benchmark = fs::read_to_string(
        root.join("crates/tenferro-linalg/benches/householder_qr_small_appends.rs"),
    )
    .unwrap();
    let manifest = fs::read_to_string(root.join("crates/tenferro-linalg/Cargo.toml")).unwrap();
    let design = fs::read_to_string(
        root.join("docs/design/incremental-householder-qr-small-append-performance.md"),
    )
    .unwrap();

    for required in [
        "const INITIAL_RANK: usize = 5",
        "const BLOCK_WIDTH: usize = 3",
        "const APPENDS: usize = 9",
        "const SAMPLE_BATCH: usize = 64",
        "const CPU_CLOCK_WARMUP_MS: u64 = 50",
        "[64, 128, 256]",
        "BenchDType::F64",
        "BenchDType::C64",
        "errors_c64",
        ".conj()",
        "tenferro.householder-qr-small-appends.v1",
        "Lane::Append",
        "Lane::R",
        "Lane::QColumns",
        "Lane::Complete",
        "Lane::FreshSessionComplete",
        "Lane::EagerComplete",
        "QrGauge::Raw",
        "reconstruction_relative_error",
        "orthogonality_relative_error",
        "cpu_affinity",
        "cpu_frequency_mhz",
        "thread_environment",
        "TENFERRO_BENCH_GIT_COMMIT is required",
        "warm_cpu_clock();",
        "attributed_concrete",
        "elapsed += start.elapsed();",
    ] {
        assert!(benchmark.contains(required), "benchmark drift: {required}");
    }
    assert!(manifest.contains("name = \"householder_qr_small_appends\""));
    assert!(design.contains("complete ~= append + R + selected-Q"));
}
