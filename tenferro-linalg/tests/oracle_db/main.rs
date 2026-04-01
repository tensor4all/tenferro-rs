mod db;
mod decode;
mod hvp;
mod replay;
mod report;
mod schema_tests;
mod support;

use serde_json::json;

#[derive(Debug, Default, Eq, PartialEq)]
struct ExpectedReplayCounts {
    supported_records: usize,
    supported_hvp_records: usize,
    expected_error_case_ids: Vec<String>,
}

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

fn expected_replay_counts() -> ExpectedReplayCounts {
    let root = db::default_oracle_db_root().expect("vendored tensor-ad-oracles root not found");
    let files = db::case_files(&root).expect("case files should load");
    let mut counts = ExpectedReplayCounts::default();

    for path in files {
        let records = db::load_case_records(&path).expect("case records should parse");
        for record in records {
            match support::classify_record(&record) {
                support::RecordSupport::Supported(_) => {
                    counts.supported_records += 1;
                    if record
                        .probes
                        .iter()
                        .any(|probe| probe.pytorch_ref.hvp.is_some())
                    {
                        counts.supported_hvp_records += 1;
                    }
                }
                support::RecordSupport::ExpectedError(_) => {
                    counts.expected_error_case_ids.push(record.case_id);
                }
                support::RecordSupport::Unsupported { .. } => {}
            }
        }
    }

    counts.expected_error_case_ids.sort();
    counts
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
fn oracle_db_parser_preserves_op_args() {
    let record = db::parse_case_record_value(json!({
        "case_id": "norm_f64_identity_args_001",
        "op": "norm",
        "dtype": "float64",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "float64",
                "shape": [5],
                "order": "row_major",
                "data": [1.0, 2.0, 3.0, 4.0, 5.0]
            }
        },
        "op_args": [null],
        "op_kwargs": { "keepdim": true },
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": []
    }))
    .expect("norm oracle support record should parse");

    assert_eq!(record.op_args.len(), 1);
    assert!(record.op_args[0].is_null());
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

    for path in files {
        let records = db::load_case_records(&path).expect("case records should parse");
        for record in records {
            let _ = support::classify_record(&record);
        }
    }
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
fn oracle_db_marks_solve_triangular_oracles_supported_for_replay() {
    let record = oracle_support_record(
        "solve_triangular_f64_identity_001",
        "solve_triangular",
        "identity",
        "identity",
    );
    assert!(
        matches!(
            support::classify_record(&record),
            support::RecordSupport::Supported(_)
        ),
        "solve_triangular/identity/identity should be supported"
    );
}

#[test]
fn oracle_db_marks_lu_oracles_supported_for_replay() {
    let record = oracle_support_record("lu_f64_identity_001", "lu", "identity", "identity");
    assert!(matches!(
        support::classify_record(&record),
        support::RecordSupport::Supported(support::ReplayKind::LuIdentity)
    ));
}

#[test]
fn oracle_db_marks_supported_norm_subsets_for_replay() {
    let vector_norm = db::parse_case_record_value(json!({
        "case_id": "norm_c128_identity_001",
        "op": "norm",
        "dtype": "complex128",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "complex128",
                "shape": [5],
                "order": "row_major",
                "data": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
            }
        },
        "op_args": [null],
        "op_kwargs": { "keepdim": true },
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": []
    }))
    .expect("vector norm support record should parse");
    assert!(matches!(
        support::classify_record(&vector_norm),
        support::RecordSupport::Supported(support::ReplayKind::NormIdentity)
    ));

    let matrix_norm = db::parse_case_record_value(json!({
        "case_id": "matrix_norm_f64_identity_020",
        "op": "matrix_norm",
        "dtype": "float64",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "float64",
                "shape": [2, 2],
                "order": "row_major",
                "data": [1.0, 2.0, 3.0, 4.0]
            }
        },
        "op_args": [1, [-1, 0], false],
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": []
    }))
    .expect("matrix norm support record should parse");
    assert!(matches!(
        support::classify_record(&matrix_norm),
        support::RecordSupport::Supported(support::ReplayKind::NormIdentity)
    ));
}

#[test]
fn oracle_db_marks_supported_lstsq_subset_for_replay() {
    let supported = db::parse_case_record_value(json!({
        "case_id": "lstsq_grad_oriented_f64_identity_square",
        "op": "lstsq_grad_oriented",
        "dtype": "float64",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "float64",
                "shape": [3, 3],
                "order": "row_major",
                "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            },
            "b": {
                "dtype": "float64",
                "shape": [3, 2],
                "order": "row_major",
                "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            }
        },
        "op_kwargs": { "driver": "gels" },
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": [{
            "probe_id": "p0",
            "direction": {},
            "cotangent": {},
            "pytorch_ref": {
                "jvp": {
                    "output_0": { "dtype": "float64", "shape": [3, 2], "order": "row_major", "data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
                    "output_1": { "dtype": "float64", "shape": [0], "order": "row_major", "data": [] }
                },
                "vjp": {}
            },
            "fd_ref": {
                "method": "central_difference",
                "stencil_order": 2,
                "step": 1e-4,
                "jvp": {
                    "output_0": { "dtype": "float64", "shape": [3, 2], "order": "row_major", "data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
                    "output_1": { "dtype": "float64", "shape": [0], "order": "row_major", "data": [] }
                }
            }
        }]
    }))
    .expect("supported lstsq record should parse");
    assert!(matches!(
        support::classify_record(&supported),
        support::RecordSupport::Supported(support::ReplayKind::LstsqIdentity)
    ));

    let unsupported = db::parse_case_record_value(json!({
        "case_id": "lstsq_grad_oriented_f64_identity_overdetermined",
        "op": "lstsq_grad_oriented",
        "dtype": "float64",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "float64",
                "shape": [4, 3],
                "order": "row_major",
                "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            },
            "b": {
                "dtype": "float64",
                "shape": [4, 2],
                "order": "row_major",
                "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            }
        },
        "op_kwargs": { "driver": "gels" },
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": [{
            "probe_id": "p0",
            "direction": {},
            "cotangent": {},
            "pytorch_ref": {
                "jvp": {
                    "output_0": { "dtype": "float64", "shape": [3, 2], "order": "row_major", "data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
                    "output_1": { "dtype": "float64", "shape": [2], "order": "row_major", "data": [1.0, 1.0] }
                },
                "vjp": {}
            },
            "fd_ref": {
                "method": "central_difference",
                "stencil_order": 2,
                "step": 1e-4,
                "jvp": {
                    "output_0": { "dtype": "float64", "shape": [3, 2], "order": "row_major", "data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
                    "output_1": { "dtype": "float64", "shape": [2], "order": "row_major", "data": [1.0, 1.0] }
                }
            }
        }]
    }))
    .expect("unsupported lstsq record should parse");
    assert!(matches!(
        support::classify_record(&unsupported),
        support::RecordSupport::Unsupported { .. }
    ));
}

#[test]
fn oracle_db_marks_supported_vector_norm_subset_for_replay() {
    let record = db::parse_case_record_value(json!({
        "case_id": "vector_norm_f64_identity_001",
        "op": "vector_norm",
        "dtype": "float64",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "float64",
                "shape": [2],
                "order": "row_major",
                "data": [1.0, 2.0]
            }
        },
        "op_kwargs": { "ord": 2, "dim": [0], "keepdim": true },
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": []
    }))
    .expect("supported vector norm record should parse");
    assert!(matches!(
        support::classify_record(&record),
        support::RecordSupport::Supported(support::ReplayKind::NormIdentity)
    ));
}

#[test]
fn oracle_db_leaves_unsupported_vector_norm_subset_rejected() {
    let record = db::parse_case_record_value(json!({
        "case_id": "vector_norm_f64_identity_scalar",
        "op": "vector_norm",
        "dtype": "float64",
        "family": "identity",
        "expected_behavior": "success",
        "inputs": {
            "a": {
                "dtype": "float64",
                "shape": [],
                "order": "row_major",
                "data": [1.0]
            }
        },
        "op_kwargs": { "ord": 2 },
        "observable": { "kind": "identity" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": []
    }))
    .expect("unsupported scalar vector norm record should parse");
    assert!(matches!(
        support::classify_record(&record),
        support::RecordSupport::Unsupported { .. }
    ));
}

#[test]
fn oracle_db_marks_supported_eig_subset_for_replay() {
    let record = oracle_support_record(
        "eig_f64_values_vectors_abs_001",
        "eig",
        "values_vectors_abs",
        "eig_values_vectors_abs",
    );
    assert!(matches!(
        support::classify_record(&record),
        support::RecordSupport::Supported(support::ReplayKind::EigValuesVectorsAbs)
    ));
}

#[test]
fn oracle_db_leaves_complex_eig_subset_rejected_for_now() {
    let record = db::parse_case_record_value(json!({
        "case_id": "eig_c64_values_vectors_abs_001",
        "op": "eig",
        "dtype": "complex64",
        "family": "values_vectors_abs",
        "expected_behavior": "success",
        "inputs": {},
        "observable": { "kind": "eig_values_vectors_abs" },
        "comparison": {
            "first_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-6 },
            "second_order": { "kind": "allclose", "rtol": 1e-4, "atol": 1e-5 }
        },
        "probes": []
    }))
    .expect("complex eig support record should parse");
    assert!(matches!(
        support::classify_record(&record),
        support::RecordSupport::Unsupported { .. }
    ));
}

#[test]
fn oracle_db_replay_against_tensor_ad_oracles() {
    let summary = replay::run_database_replay();
    let expected = expected_replay_counts();

    assert_eq!(
        summary.validated_records, expected.supported_records,
        "unexpected replay summary: validated={}, expected_error={:?}, failures={:?}",
        summary.validated_records, summary.expected_error_case_ids, summary.failures
    );
    assert_eq!(
        summary.expected_error_case_ids,
        expected.expected_error_case_ids
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
    let expected = expected_replay_counts();

    assert_eq!(
        summary.validated_hvp_records, expected.supported_hvp_records,
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

#[test]
#[ignore = "manual helper to refresh checked-in oracle support report after DB or support changes"]
fn oracle_db_regenerates_checked_in_support_report() {
    let root = db::default_oracle_db_root().expect("vendored tensor-ad-oracles root not found");
    report::write_checked_in_report(&root).expect("support report should write");
}
