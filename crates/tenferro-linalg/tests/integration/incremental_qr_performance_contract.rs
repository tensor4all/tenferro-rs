use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("tenferro-linalg should live in the workspace")
        .to_path_buf()
}

fn read(path: &str) -> String {
    fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|error| panic!("performance contract source {path} unreadable: {error}"))
}

#[test]
fn incremental_qr_performance_gate_contract() {
    let benchmark = read("crates/tenferro-linalg/benches/incremental_householder_qr.rs");
    let script = read("scripts/incremental-householder-qr-performance.py");
    let protocol = read("docs/design/incremental-householder-qr-performance-gate.md");
    let ledger = read("docs/performance/incremental-householder-qr-bcgs2-ledger.md");
    let manifest = read("crates/tenferro-linalg/Cargo.toml");
    let commit = "da0775a208006352f6e5eab18bc6bb09ca39a1f6";

    for source in [&benchmark, &script, &protocol, &ledger] {
        assert!(
            source.contains(commit),
            "missing pinned tensor4all-rs#694 commit"
        );
    }
    assert_eq!(benchmark.matches("matmul(session, &qh").count(), 2);
    assert!(benchmark.contains("session.qr(&residual)"));
    assert!(benchmark.contains("let new_q = session"));
    assert!(benchmark.contains("let new_r = session"));
    assert!(benchmark.contains("Prepared::Compact"));
    assert!(benchmark.contains("append_columns(black_box(block), session)"));

    for required in [
        "Case(\"bond32\", 2 * 32 * 32, 2, 3, 32, 5",
        "Case(\"bond64\", 2 * 64 * 64, 2, 3, 32, 5",
        "Case(\"bond128\", 2 * 128 * 128, 2, 3, 32, 3",
        "Case(\"scaling-rank29\", 32768, 29, 3, 32, 3",
        "CYCLES = 7",
        "BOOTSTRAP_SAMPLES = 10_000",
        "compact exceeded {limit:.2f}x BCGS2",
        "normalized append scaling exceeded 35%",
        "def validate_record(record: dict)",
        "performance gates missing summaries",
        "paired_ratio_ci(compact_values, bcgs2_values)",
        "invalid or incomplete artifact",
        "benchmark_sha256",
        "checker_sha256",
        "ledger_sha256",
        "SAMPLE_BATCH = 4",
        "cpu_reference_mhz",
        "runner_affinity",
        "target = inconclusive if environment_issues else findings",
        "not (backend == \"cuda\" and bond == 32)",
        "backend != \"cuda\"",
        "except Exception as error",
        "environment.get(\"schema\") != SCHEMA",
    ] {
        assert!(script.contains(required), "checker drift: {required}");
    }
    assert!(benchmark.contains("sample_batch: SAMPLE_BATCH"));
    assert!(benchmark.contains("CPU_CLOCK_WARMUP_MS: u64 = 50"));
    assert!(benchmark.contains("warm_cpu_clock();"));
    assert!(benchmark.contains("cpu0_frequency_mhz()"));
    assert!(benchmark.contains("process_affinity()"));
    assert!(protocol.contains("A batched process median below 1 ms is"));
    assert!(protocol.contains("`INCONCLUSIVE`"));
    assert!(protocol.contains("same batch"));
    assert!(protocol.contains("reproducible numeric or source-contract failure is `FAIL`"));
    assert!(protocol.contains("max(pre-run load, 0.1)"));
    assert!(ledger.contains("Omitting those costs favors BCGS2"));
    assert!(manifest.contains("name = \"incremental_householder_qr\""));
}
