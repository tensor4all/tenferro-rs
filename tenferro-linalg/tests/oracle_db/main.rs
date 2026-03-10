mod db;
mod decode;
mod hvp;
mod replay;
mod report;
mod support;

use serde_json::json;

fn oracle_support_record(
    case_id: &str,
    op: &str,
    family: &str,
    observable_kind: &str,
) -> db::CaseRecord {
    db::parse_case_record_value(json!({
        "case_id": case_id,
        "op": op,
        "dtype": "float64",
        "family": family,
        "expected_behavior": "success",
        "inputs": {},
        "observable": { "kind": observable_kind },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": []
    }))
    .expect("test oracle support record should parse")
}

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
fn oracle_db_parser_handles_current_schema() {
    let record = json!({
        "case_id": "solve_f64_identity_hvp_001",
        "op": "solve",
        "dtype": "float64",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "float64",
                "shape": [2, 2],
                "order": "row_major",
                "data": [1.0, 0.0, 0.0, 1.0]
            },
            "b": {
                "dtype": "float64",
                "shape": [2],
                "order": "row_major",
                "data": [1.0, 2.0]
            }
        },
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": [{
            "probe_id": "p0",
            "direction": {
                "a": {
                    "dtype": "float64",
                    "shape": [2, 2],
                    "order": "row_major",
                    "data": [0.0, 0.0, 0.0, 0.0]
                },
                "b": {
                    "dtype": "float64",
                    "shape": [2],
                    "order": "row_major",
                    "data": [0.0, 0.0]
                }
            },
            "cotangent": {
                "value": {
                    "dtype": "float64",
                    "shape": [2],
                    "order": "row_major",
                    "data": [1.0, 1.0]
                }
            },
            "pytorch_ref": {
                "jvp": {
                    "value": {
                        "dtype": "float64",
                        "shape": [2],
                        "order": "row_major",
                        "data": [0.0, 0.0]
                    }
                },
                "hvp": {
                    "a": {
                        "dtype": "float64",
                        "shape": [2, 2],
                        "order": "row_major",
                        "data": [0.0, 0.0, 0.0, 0.0]
                    },
                    "b": {
                        "dtype": "float64",
                        "shape": [2],
                        "order": "row_major",
                        "data": [0.0, 0.0]
                    }
                },
                "vjp": {
                    "a": {
                        "dtype": "float64",
                        "shape": [2, 2],
                        "order": "row_major",
                        "data": [1.0, 0.0, 0.0, 1.0]
                    },
                    "b": {
                        "dtype": "float64",
                        "shape": [2],
                        "order": "row_major",
                        "data": [1.0, 1.0]
                    }
                }
            },
            "fd_ref": {
                "method": "central_difference",
                "stencil_order": 2,
                "step": 1e-6,
                "jvp": {
                    "value": {
                        "dtype": "float64",
                        "shape": [2],
                        "order": "row_major",
                        "data": [0.0, 0.0]
                    }
                },
                "hvp": {
                    "a": {
                        "dtype": "float64",
                        "shape": [2, 2],
                        "order": "row_major",
                        "data": [0.0, 0.0, 0.0, 0.0]
                    },
                    "b": {
                        "dtype": "float64",
                        "shape": [2],
                        "order": "row_major",
                        "data": [0.0, 0.0]
                    }
                }
            }
        }]
    });

    let parsed = db::parse_case_record_value(record).expect("current schema should parse");
    assert_eq!(parsed.comparison.first_order().unwrap().kind, "allclose");
    assert_eq!(parsed.comparison.second_order().unwrap().kind, "allclose");
    assert!(parsed.probes[0].pytorch_ref.hvp.is_some());
    assert!(parsed.probes[0].fd_ref.hvp.is_some());
}

#[test]
fn oracle_db_parser_rejects_half_present_hvp_payloads() {
    let record = json!({
        "case_id": "solve_f64_identity_hvp_half_present",
        "op": "solve",
        "dtype": "float64",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "float64",
                "shape": [1, 1],
                "order": "row_major",
                "data": [1.0]
            },
            "b": {
                "dtype": "float64",
                "shape": [1],
                "order": "row_major",
                "data": [1.0]
            }
        },
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": [{
            "probe_id": "p0",
            "direction": {
                "a": {
                    "dtype": "float64",
                    "shape": [1, 1],
                    "order": "row_major",
                    "data": [0.0]
                },
                "b": {
                    "dtype": "float64",
                    "shape": [1],
                    "order": "row_major",
                    "data": [0.0]
                }
            },
            "cotangent": {
                "value": {
                    "dtype": "float64",
                    "shape": [1],
                    "order": "row_major",
                    "data": [1.0]
                }
            },
            "pytorch_ref": {
                "jvp": {
                    "value": {
                        "dtype": "float64",
                        "shape": [1],
                        "order": "row_major",
                        "data": [0.0]
                    }
                },
                "hvp": {
                    "a": {
                        "dtype": "float64",
                        "shape": [1, 1],
                        "order": "row_major",
                        "data": [0.0]
                    },
                    "b": {
                        "dtype": "float64",
                        "shape": [1],
                        "order": "row_major",
                        "data": [0.0]
                    }
                },
                "vjp": {
                    "a": {
                        "dtype": "float64",
                        "shape": [1, 1],
                        "order": "row_major",
                        "data": [1.0]
                    },
                    "b": {
                        "dtype": "float64",
                        "shape": [1],
                        "order": "row_major",
                        "data": [1.0]
                    }
                }
            },
            "fd_ref": {
                "method": "central_difference",
                "stencil_order": 2,
                "step": 1e-6,
                "jvp": {
                    "value": {
                        "dtype": "float64",
                        "shape": [1],
                        "order": "row_major",
                        "data": [0.0]
                    }
                }
            }
        }]
    });

    let err = db::parse_case_record_value(record).expect_err("half-present HVP should fail");
    assert!(err.contains("half-present HVP"));
}

#[test]
fn oracle_db_every_record_is_classified() {
    let root = db::default_oracle_db_root().expect("vendored tensor-ad-oracles root not found");
    let files = db::case_files(&root).expect("case files should load");
    let mut unknown = Vec::new();

    for path in files {
        let records = db::load_case_records(&path).expect("case records should parse");
        for record in records {
            if matches!(
                support::classify_record(&record),
                support::RecordSupport::Unknown
            ) {
                unknown.push(format!(
                    "{}/{}/{} ({})",
                    record.op, record.family, record.observable.kind, record.expected_behavior
                ));
            }
        }
    }

    assert!(
        unknown.is_empty(),
        "unclassified oracle families: {:?}",
        unknown
    );
}

#[test]
fn oracle_db_marks_batch_a_oracles_supported_for_replay() {
    let cases = [
        (
            "cholesky_ex_f64_identity_test",
            "cholesky_ex",
            "identity",
            "identity",
        ),
        (
            "solve_ex_f64_identity_test",
            "solve_ex",
            "identity",
            "identity",
        ),
        ("inv_ex_f64_identity_test", "inv_ex", "identity", "identity"),
        (
            "lu_factor_f64_identity_test",
            "lu_factor",
            "identity",
            "identity",
        ),
        (
            "lu_factor_ex_f64_identity_test",
            "lu_factor_ex",
            "identity",
            "identity",
        ),
        (
            "lu_solve_f64_identity_test",
            "lu_solve",
            "identity",
            "identity",
        ),
        ("cond_f64_identity_test", "cond", "identity", "identity"),
        (
            "matrix_power_f64_identity_test",
            "matrix_power",
            "identity",
            "identity",
        ),
    ];

    for (case_id, op, family, observable) in cases {
        let record = oracle_support_record(case_id, op, family, observable);
        assert!(
            matches!(
                support::classify_record(&record),
                support::RecordSupport::Supported(_)
            ),
            "{op}/{family}/{observable} should be supported"
        );
    }
}

#[test]
fn oracle_db_replay_against_tensor_ad_oracles() {
    let summary = replay::run_database_replay();

    assert_eq!(
        summary.validated_records, 867,
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

#[test]
fn oracle_db_replays_supported_hvp_cases() {
    let summary = replay::run_database_replay();

    assert_eq!(
        summary.validated_hvp_records, 867,
        "unexpected HVP replay summary: validated_hvp={}, unsupported={}, failures={:?}",
        summary.validated_hvp_records, summary.unsupported_records, summary.failures
    );
}

#[test]
fn oracle_db_support_report_matches_checked_in_markdown() {
    let root = db::default_oracle_db_root().expect("vendored tensor-ad-oracles root not found");
    let generated = report::generate_support_report(&root).expect("support report should render");
    let checked_in = std::fs::read_to_string(report::checked_in_report_path())
        .expect("checked-in support report should exist");
    assert_eq!(generated, checked_in);
}
