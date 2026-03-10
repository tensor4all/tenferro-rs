mod db;
mod decode;
mod replay;

use serde_json::json;

#[test]
fn oracle_db_root_resolves_vendored_subtree() {
    let root = db::default_oracle_db_root().expect("vendored tensor-ad-oracles root not found");
    assert!(root.ends_with("third_party/tensor-ad-oracles"));

    let files = db::case_files(&root).unwrap();
    assert!(!files.is_empty());
    assert!(files
        .iter()
        .any(|path| path.ends_with("cases/solve/identity.jsonl")));
}

#[test]
fn oracle_db_decode_moves_pytorch_matrix_dims_to_tenferro_front() {
    let encoded = db::DbTensor {
        dtype: "float64".to_string(),
        shape: vec![2, 3, 4],
        order: "row_major".to_string(),
        data: (0..24).map(|value| json!(value as f64)).collect(),
    };
    let tensor = decode::decode_f64_tensor_with_core_rank(&encoded, 2).unwrap();
    assert_eq!(tensor.dims(), &[3, 4, 2]);
}

#[test]
fn oracle_db_replay_against_tensor_ad_oracles() {
    let summary = replay::run_database_replay();

    assert_eq!(
        summary.validated_records, 348,
        "unexpected replay summary: validated={}, expected_error={:?}, failures={:?}",
        summary.validated_records, summary.expected_error_case_ids, summary.failures
    );
    assert_eq!(
        summary.expected_error_case_ids,
        vec![
            "eigh_c128_gauge_ill_defined_001".to_string(),
            "svd_c128_gauge_ill_defined_001".to_string(),
        ]
    );
    assert!(
        summary.failures.is_empty(),
        "oracle replay failures: {:?}",
        summary.failures
    );
}
