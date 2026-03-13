use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;

use crate::db;

fn temp_oracle_db_root(test_name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("tenferro-oracle-db-{test_name}-{nanos}"));
    fs::create_dir_all(path.join("cases/svd"))
        .expect("temporary oracle db cases directory should be creatable");
    path
}

#[test]
fn oracle_db_parser_accepts_success_without_second_order_and_hvp() {
    let record = json!({
        "case_id": "svd_c64_s_schema_minimal",
        "op": "svd",
        "dtype": "complex64",
        "family": "s",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "complex64",
                "shape": [0, 0],
                "order": "row_major",
                "data": []
            }
        },
        "observable": { "kind": "svd_s" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e4, "atol": 10.0 }
        },
        "probes": [{
            "probe_id": "p0",
            "direction": {
                "a": {
                    "dtype": "complex64",
                    "shape": [0, 0],
                    "order": "row_major",
                    "data": []
                }
            },
            "cotangent": {
                "s": {
                    "dtype": "float32",
                    "shape": [0],
                    "order": "row_major",
                    "data": []
                }
            },
            "pytorch_ref": {
                "jvp": {
                    "s": {
                        "dtype": "float32",
                        "shape": [0],
                        "order": "row_major",
                        "data": []
                    }
                },
                "vjp": {
                    "a": {
                        "dtype": "complex64",
                        "shape": [0, 0],
                        "order": "row_major",
                        "data": []
                    }
                }
            },
            "fd_ref": {
                "method": "central_difference",
                "stencil_order": 2,
                "step": 1e-4,
                "jvp": {
                    "s": {
                        "dtype": "float32",
                        "shape": [0],
                        "order": "row_major",
                        "data": []
                    }
                }
            }
        }]
    });

    let parsed = db::parse_case_record_value(record)
        .expect("success case without second-order or hvp payloads should parse");
    assert_eq!(parsed.dtype, "complex64");
    assert!(parsed.comparison.first_order().is_some());
    assert!(parsed.comparison.second_order().is_none());
    assert!(parsed.probes[0].pytorch_ref.hvp.is_none());
    assert!(parsed.probes[0].fd_ref.hvp.is_none());
}

#[test]
fn oracle_db_parser_accepts_complex_payload_entries() {
    let record = json!({
        "case_id": "svd_c64_s_complex_payload",
        "op": "svd",
        "dtype": "complex64",
        "family": "s",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "complex64",
                "shape": [1, 1],
                "order": "row_major",
                "data": [[1.25, -0.5]]
            }
        },
        "observable": { "kind": "svd_s" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e4, "atol": 10.0 }
        },
        "probes": [{
            "probe_id": "p0",
            "direction": {
                "a": {
                    "dtype": "complex64",
                    "shape": [1, 1],
                    "order": "row_major",
                    "data": [[0.25, 0.125]]
                }
            },
            "cotangent": {
                "s": {
                    "dtype": "float32",
                    "shape": [1],
                    "order": "row_major",
                    "data": [1.0]
                }
            },
            "pytorch_ref": {
                "jvp": {
                    "s": {
                        "dtype": "float32",
                        "shape": [1],
                        "order": "row_major",
                        "data": [0.0]
                    }
                },
                "vjp": {
                    "a": {
                        "dtype": "complex64",
                        "shape": [1, 1],
                        "order": "row_major",
                        "data": [[1.0, 0.0]]
                    }
                }
            },
            "fd_ref": {
                "method": "central_difference",
                "stencil_order": 2,
                "step": 1e-4,
                "jvp": {
                    "s": {
                        "dtype": "float32",
                        "shape": [1],
                        "order": "row_major",
                        "data": [0.0]
                    }
                }
            }
        }]
    });

    let parsed =
        db::parse_case_record_value(record).expect("complex payload record should parse cleanly");
    assert_eq!(parsed.inputs["a"].data.len(), 1);
}

#[test]
fn oracle_db_load_case_records_accepts_nan_payloads() {
    let root = temp_oracle_db_root("nan-payload");
    let jsonl_path = root.join("cases/svd/s.jsonl");
    fs::write(
        &jsonl_path,
        r#"{"case_id":"nanmean_f64_identity_025","comparison":{"first_order":{"atol":1e-7,"kind":"allclose","rtol":1e-4}},"dtype":"float64","expected_behavior":"success","family":"identity","inputs":{"a":{"data":[2.0,NaN,-1.0],"dtype":"float64","order":"row_major","shape":[3]}},"observable":{"kind":"identity"},"op":"nanmean","probes":[{"cotangent":{"value":{"data":[1.0],"dtype":"float64","order":"row_major","shape":[]}},"direction":{"a":{"data":[0.1,0.2,0.3],"dtype":"float64","order":"row_major","shape":[3]}},"fd_ref":{"method":"central_difference","stencil_order":2,"step":1e-6,"jvp":{"value":{"data":[0.2],"dtype":"float64","order":"row_major","shape":[]}}},"probe_id":"p0","pytorch_ref":{"jvp":{"value":{"data":[0.2],"dtype":"float64","order":"row_major","shape":[]}},"vjp":{"a":{"data":[0.5,0.0,0.5],"dtype":"float64","order":"row_major","shape":[3]}}}}],"schema_version":1}"#,
    )
    .expect("oracle db fixture should be writable");

    let records =
        db::load_case_records(&jsonl_path).expect("NaN payload line should be parsed leniently");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].case_id, "nanmean_f64_identity_025");
}
