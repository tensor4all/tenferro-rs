use std::collections::BTreeMap;
use std::env;

#[path = "oracle_replay/db.rs"]
mod db;
#[path = "oracle_replay/replay.rs"]
mod replay;
#[path = "oracle_replay/support.rs"]
mod support;

#[derive(Debug, Default, Eq, PartialEq)]
struct ExpectedReplayCounts {
    total_records: usize,
    supported_records: usize,
    supported_hvp_records: usize,
    unsupported_records: usize,
    expected_error_case_ids: Vec<String>,
    supported_by_key: BTreeMap<(String, String, String), usize>,
    supported_hvp_by_key: BTreeMap<(String, String, String), usize>,
    unsupported_by_key: BTreeMap<(String, String, String, String), usize>,
}

fn expected_replay_counts() -> ExpectedReplayCounts {
    let root = db::default_oracle_db_root().expect("vendored tensor-ad-oracles root not found");
    let files = db::case_files(&root).expect("case files should load");
    let mut counts = ExpectedReplayCounts::default();

    for path in files {
        let records = db::load_case_records(&path).expect("case records should parse");
        for record in records {
            counts.total_records += 1;
            match support::classify_record(&record) {
                support::RecordSupport::Supported(_) => {
                    counts.supported_records += 1;
                    let key = (
                        record.op.clone(),
                        record.family.clone(),
                        record.observable.kind.clone(),
                    );
                    *counts.supported_by_key.entry(key.clone()).or_default() += 1;
                    if record
                        .probes
                        .iter()
                        .any(|probe| probe.pytorch_ref.hvp.is_some())
                    {
                        counts.supported_hvp_records += 1;
                        *counts.supported_hvp_by_key.entry(key).or_default() += 1;
                    }
                }
                support::RecordSupport::ExpectedError(_) => {
                    counts.expected_error_case_ids.push(record.case_id);
                }
                support::RecordSupport::Unsupported { reason } => {
                    counts.unsupported_records += 1;
                    let key = (
                        record.op.clone(),
                        record.family.clone(),
                        record.observable.kind.clone(),
                        reason.to_string(),
                    );
                    *counts.unsupported_by_key.entry(key).or_default() += 1;
                }
            }
        }
    }

    counts.expected_error_case_ids.sort();
    counts
}

#[test]
fn oracle_support_snapshot_counts() {
    let counts = expected_replay_counts();
    if env::var("DUMP_ORACLE_SUPPORT_MARKDOWN").is_ok() {
        dump_oracle_support_markdown(&counts);
    }
    assert_eq!(counts.total_records, 9585);
    assert_eq!(counts.supported_records, 2090);
    assert_eq!(counts.unsupported_records, 7493);
    if counts.supported_hvp_records != 1339 {
        eprintln!("supported_by_key = {:#?}", counts.supported_by_key);
        eprintln!("supported_hvp_by_key = {:#?}", counts.supported_hvp_by_key);
    }
    assert_eq!(counts.supported_hvp_records, 1339);
    assert_eq!(
        counts.expected_error_case_ids,
        [
            "eigh_c128_gauge_ill_defined_001".to_string(),
            "svd_c128_gauge_ill_defined_001".to_string(),
        ]
    );
}

fn dump_oracle_support_markdown(counts: &ExpectedReplayCounts) {
    eprintln!("## Supported");
    eprintln!("| op | family | observable | sample count |");
    eprintln!("| --- | --- | --- | ---: |");
    for ((op, family, observable), count) in &counts.supported_by_key {
        eprintln!("| {op} | {family} | {observable} | {count} |");
    }
    eprintln!();
    eprintln!("## Unsupported");
    eprintln!("| op | family | observable | sample count | reason |");
    eprintln!("| --- | --- | --- | ---: | --- |");
    for ((op, family, observable, reason), count) in &counts.unsupported_by_key {
        eprintln!("| {op} | {family} | {observable} | {count} | {reason} |");
    }
}

#[test]
fn oracle_db_root_resolves_existing_cases_tree() {
    let root = db::default_oracle_db_root().expect("vendored tensor-ad-oracles root not found");
    assert!(root.join("cases").is_dir());

    let files = db::case_files(&root).unwrap();
    assert!(!files.is_empty());
    assert!(files
        .iter()
        .any(|path| path.file_name().unwrap() == "identity.jsonl"));
}

#[test]
fn oracle_db_every_record_is_classified() {
    let root = db::default_oracle_db_root().expect("vendored tensor-ad-oracles root not found");
    let files = db::case_files(&root).expect("case files should load");

    for path in files {
        let records = db::load_case_records(&path).expect("case records should parse");
        for record in records {
            let _ = support::classify_record(&record);
        }
    }
}

#[test]
fn oracle_replays_solve_jvp_vjp_hvp() {
    replay::replay_case_id("solve", "identity", "solve_f64_identity_001")
        .expect("solve oracle replay should pass");
}

#[test]
fn oracle_replay_accepts_expected_error_records() {
    replay::replay_expected_error_case_id(
        "eigh",
        "gauge_ill_defined",
        "eigh_c128_gauge_ill_defined_001",
    )
    .expect("eigh gauge expected-error record should pass");
    replay::replay_expected_error_case_id(
        "svd",
        "gauge_ill_defined",
        "svd_c128_gauge_ill_defined_001",
    )
    .expect("svd gauge expected-error record should pass");
}

#[test]
fn oracle_replays_supported_db_cases_when_requested() {
    let Some(summary) =
        replay::replay_supported_cases_from_env().expect("oracle replay runner should not fail")
    else {
        eprintln!("set RUN_ORACLE_REPLAY=1 to replay supported tensor-ad-oracles records");
        return;
    };
    eprintln!("oracle replay summary: {summary:?}");
    assert!(
        summary.replayed_success_records + summary.replayed_expected_error_records > 0,
        "RUN_ORACLE_REPLAY=1 should replay at least one supported or expected-error record"
    );
}
