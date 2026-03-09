mod db;
mod decode;
mod replay;

use serde_json::json;

#[test]
fn oracle_db_replay_against_tensor_ad_oracles() {
    let root = db::default_oracle_db_root().expect("tensor-ad-oracles checkout not found");
    let summary = replay::run_database_replay(&root);

    assert_eq!(
        summary.validated_records, 348,
        "unexpected replay summary: validated={}, unsupported={:?}, failures={:?}",
        summary.validated_records, summary.unsupported_case_ids, summary.failures
    );
    assert_eq!(
        summary.unsupported_case_ids,
        vec![
            "eigh_c128_gauge_ill_defined_001".to_string(),
            "svd_c128_gauge_ill_defined_001".to_string(),
        ]
    );
    assert!(
        summary.failures.is_empty(),
        "database replay failures: {:?}",
        summary.failures
    );
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
