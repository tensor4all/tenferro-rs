use serde_json::Value;

use super::db::CaseRecord;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayKind {
    SolveIdentity,
    SolveTriangularIdentity,
    LstsqIdentity,
    CholeskyIdentity,
    InvIdentity,
    DetIdentity,
    SlogdetIdentity,
    LuFactorIdentity,
    LuIdentity,
    NormIdentity,
    PinvIdentity,
    NumericalIdentity,
    QrIdentity,
    SvdUAbs,
    SvdS,
    SvdVhAbs,
    SvdUvhProduct,
    EighValuesVectorsAbs,
    PinvSingularIdentity,
    IncrementalHouseholderQr,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpectedErrorKind {
    GaugeIllDefined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecordSupport {
    Supported(ReplayKind),
    ExpectedError(ExpectedErrorKind),
    Unsupported { reason: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReplayNormKind {
    Fro,
    L1,
    Lp(f64),
    Inf,
    Spectral,
}

fn float64_only(dtype: &str) -> bool {
    dtype == "float64"
}

fn batch_a_replay_dtype(dtype: &str) -> bool {
    matches!(dtype, "float64" | "complex64" | "complex128")
}

fn svd_replay_dtype(dtype: &str) -> bool {
    matches!(dtype, "float32" | "float64" | "complex64" | "complex128")
}

fn is_complex_dtype(dtype: &str) -> bool {
    matches!(dtype, "complex64" | "complex128")
}

fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) if text == "Infinity" => Some(f64::INFINITY),
        Value::String(text) if text == "-Infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

fn normalized_axis(value: &Value, rank: usize) -> Option<usize> {
    let axis = value.as_i64()?;
    let rank = i64::try_from(rank).ok()?;
    let normalized = if axis < 0 { rank + axis } else { axis };
    if (0..rank).contains(&normalized) {
        usize::try_from(normalized).ok()
    } else {
        None
    }
}

fn matrix_axes_order(value: &Value, rank: usize) -> Option<Vec<usize>> {
    let Value::Array(axes) = value else {
        return None;
    };
    if axes.len() != rank {
        return None;
    }
    let mut normalized = axes
        .iter()
        .filter_map(|axis| normalized_axis(axis, rank))
        .collect::<Vec<_>>();
    let mut sorted = normalized.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted == (0..rank).collect::<Vec<_>>() {
        Some(std::mem::take(&mut normalized))
    } else {
        None
    }
}

fn matrix_axes_cover_rank(value: &Value, rank: usize) -> bool {
    matrix_axes_order(value, rank).is_some()
}

fn replayable_lstsq_subset(record: &CaseRecord) -> bool {
    if !matches!(record.dtype.as_str(), "float32" | "float64") {
        return false;
    }
    let Some(a) = record.inputs.get("a") else {
        return false;
    };
    if a.shape.len() < 2 {
        return false;
    }
    let m = a.shape[a.shape.len() - 2];
    let n = a.shape[a.shape.len() - 1];
    if m < n {
        return false;
    }
    if m > n
        && matches!(
            record.op_kwargs.get("driver").and_then(Value::as_str),
            Some("gelsy")
        )
    {
        return false;
    }
    !record.probes.is_empty()
}

fn square_matrix_input(record: &CaseRecord) -> bool {
    let Some(a) = record.inputs.get("a") else {
        return false;
    };
    a.shape.len() >= 2 && a.shape[a.shape.len() - 2] == a.shape[a.shape.len() - 1]
}

fn replayable_norm_kind(record: &CaseRecord) -> Option<ReplayNormKind> {
    let rank = record.inputs.get("a")?.shape.len();
    let kind = match record.op.as_str() {
        "norm" => replayable_norm_kind_from_norm(record, rank)?,
        "matrix_norm" => replayable_norm_kind_from_matrix_norm(record, rank)?,
        "vector_norm" => replayable_norm_kind_from_vector_norm(record, rank)?,
        _ => return None,
    };
    if is_complex_dtype(&record.dtype) {
        match kind {
            ReplayNormKind::Fro => Some(kind),
            ReplayNormKind::Lp(2.0) => Some(kind),
            _ => None,
        }
    } else {
        Some(kind)
    }
}

const NORM_UNSUPPORTED_REASON: &str = "tenferro replay currently supports only the whole-tensor torch.linalg.norm subset expressible by current NormKind AD rules; remaining dim-aware and unsupported ord/rank cases are not replayed yet";
const VECTOR_NORM_UNSUPPORTED_REASON: &str = "tenferro replay currently supports only the rank-1 scalar-output vector_norm slice accepted by the current NormKind adapter; complex inputs are further restricted to ord=P(2)";
const MATRIX_NORM_UNSUPPORTED_REASON: &str = "tenferro replay currently supports only the rank-2 scalar-output matrix_norm slice accepted by the current NormKind adapter; complex inputs are further restricted to ord=Fro";

fn replayable_vector_ord(value: Option<&Value>) -> Option<ReplayNormKind> {
    match value {
        None | Some(Value::Null) => Some(ReplayNormKind::Lp(2.0)),
        Some(Value::Number(_) | Value::String(_)) => {
            let ord = value_as_f64(value?)?;
            if ord == 1.0 {
                Some(ReplayNormKind::L1)
            } else if ord == 2.0 {
                Some(ReplayNormKind::Lp(2.0))
            } else if ord.is_infinite() && ord.is_sign_positive() {
                Some(ReplayNormKind::Inf)
            } else if ord >= 1.0 {
                Some(ReplayNormKind::Lp(ord))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn replayable_norm_kind_from_norm(record: &CaseRecord, rank: usize) -> Option<ReplayNormKind> {
    if rank == 0 || rank > 2 {
        return None;
    }

    if rank == 1 {
        return replayable_vector_ord(record.op_args.first());
    }

    if let Some(dim) = record.op_kwargs.get("dim") {
        if !matrix_axes_cover_rank(dim, rank) {
            return None;
        }
        return match record.op_kwargs.get("ord") {
            Some(Value::String(text)) if text == "fro" => Some(ReplayNormKind::Fro),
            Some(Value::String(text)) if text == "nuc" => None,
            _ => None,
        };
    }

    let arg = record.op_args.first();
    match arg {
        None | Some(Value::Null) => Some(ReplayNormKind::Fro),
        Some(Value::String(text)) if text == "fro" => Some(ReplayNormKind::Fro),
        Some(Value::String(text)) if text == "nuc" => None,
        Some(Value::Number(_) | Value::String(_)) => {
            let ord = value_as_f64(arg?)?;
            if ord == 1.0 {
                Some(ReplayNormKind::L1)
            } else if ord == 2.0 {
                Some(ReplayNormKind::Spectral)
            } else if ord.is_infinite() && ord.is_sign_positive() {
                Some(ReplayNormKind::Inf)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn replayable_norm_kind_from_vector_norm(
    record: &CaseRecord,
    rank: usize,
) -> Option<ReplayNormKind> {
    if rank != 1 {
        return None;
    }

    match record.op_kwargs.get("dim") {
        None => {}
        Some(Value::Number(_)) => {
            if normalized_axis(record.op_kwargs.get("dim")?, rank)? != 0 {
                return None;
            }
        }
        Some(Value::Array(values)) if values.len() == 1 => {
            if normalized_axis(values.first()?, rank)? != 0 {
                return None;
            }
        }
        _ => return None,
    }

    replayable_vector_ord(record.op_kwargs.get("ord"))
}

fn replayable_norm_kind_from_matrix_norm(
    record: &CaseRecord,
    rank: usize,
) -> Option<ReplayNormKind> {
    if rank != 2 || record.op_args.len() != 3 {
        return None;
    }
    let axes = matrix_axes_order(&record.op_args[1], rank)?;
    let reversed_axes = axes == vec![1, 0];
    match &record.op_args[0] {
        Value::String(text) if text == "fro" => Some(ReplayNormKind::Fro),
        Value::String(text) if text == "nuc" => None,
        Value::Number(_) | Value::String(_) => {
            let ord = value_as_f64(&record.op_args[0])?;
            if ord == 1.0 {
                Some(if reversed_axes {
                    ReplayNormKind::Inf
                } else {
                    ReplayNormKind::L1
                })
            } else if ord == 2.0 {
                Some(ReplayNormKind::Spectral)
            } else if ord.is_infinite() && ord.is_sign_positive() {
                Some(if reversed_axes {
                    ReplayNormKind::L1
                } else {
                    ReplayNormKind::Inf
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

fn supported_if(dtype_ok: bool, kind: ReplayKind, reason: &'static str) -> RecordSupport {
    if dtype_ok {
        RecordSupport::Supported(kind)
    } else {
        RecordSupport::Unsupported { reason }
    }
}

pub fn classify_record(record: &CaseRecord) -> RecordSupport {
    match (
        record.op.as_str(),
        record.family.as_str(),
        record.observable.kind.as_str(),
        record.expected_behavior.as_str(),
    ) {
        ("solve", "identity", "identity", "success")
        | ("solve_ex", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::SolveIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("lu_solve", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro has no public traced lu_solve replay adapter for this oracle family yet",
        },
        ("solve_triangular", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::SolveTriangularIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("cholesky", "identity", "identity", "success")
        | ("cholesky_ex", "identity", "identity", "success") => supported_if(
            batch_a_replay_dtype(&record.dtype),
            ReplayKind::CholeskyIdentity,
            "tenferro replay currently supports this family only for float64/complex64/complex128",
        ),
        ("inv", "identity", "identity", "success")
        | ("inv_ex", "identity", "identity", "success") => supported_if(
            batch_a_replay_dtype(&record.dtype),
            ReplayKind::InvIdentity,
            "tenferro replay currently supports this family only for float64/complex64/complex128",
        ),
        ("lu_factor", "identity", "identity", "success")
        | ("lu_factor_ex", "identity", "identity", "success") => supported_if(
            false,
            ReplayKind::LuFactorIdentity,
            "tenferro lu_factor is a prepared factor carrier whose AD rule is intentionally unsupported",
        ),
        ("lu", "identity", "identity", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::LuIdentity,
            "tenferro replay currently supports this family only for float32/float64/complex64/complex128",
        ),
        ("cond", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro has no public traced condition-number API to replay this oracle family yet",
        },
        ("matrix_power", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro has no public traced matrix_power API to replay this oracle family yet",
        },
        ("matrix_exp", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro has no public traced matrix_exp API to replay this oracle family yet",
        },
        ("qr", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::QrIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("svd", "u_abs", "svd_u_abs", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::SvdUAbs,
            "tenferro replay currently supports SVD only for float32/float64/complex64/complex128",
        ),
        ("svd", "s", "svd_s", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::SvdS,
            "tenferro replay currently supports SVD only for float32/float64/complex64/complex128",
        ),
        ("svd", "vh_abs", "svd_vh_abs", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::SvdVhAbs,
            "tenferro replay currently supports SVD only for float32/float64/complex64/complex128",
        ),
        ("svd", "uvh_product", "svd_uvh_product", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::SvdUvhProduct,
            "tenferro replay currently supports SVD only for float32/float64/complex64/complex128",
        ),
        ("eigh", "values_vectors_abs", "eigh_values_vectors_abs", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::EighValuesVectorsAbs,
            "tenferro replay currently supports this family only for float32/float64/complex64/complex128",
        ),
        ("pinv_singular", "identity", "identity", "success") => supported_if(
            false,
            ReplayKind::PinvSingularIdentity,
            "tenferro replay does not expose the two-input pinv_singular oracle family",
        ),
        ("svd", "gauge_ill_defined", "svd_uvh_product", "error")
        | ("eigh", "gauge_ill_defined", "eigh_values_vectors_abs", "error") => {
            RecordSupport::ExpectedError(ExpectedErrorKind::GaugeIllDefined)
        }
        (
            "incremental_householder_qr",
            "factor_qr" | "append_qr" | "from_factors_qr" | "selected_q_columns" | "r",
            "identity",
            "success",
        ) => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::IncrementalHouseholderQr,
            "incremental Householder QR replay supports float32/float64/complex64/complex128",
        ),
        ("full_pivot_lu", "identity", "identity", "success") => supported_if(
            svd_replay_dtype(&record.dtype) && square_matrix_input(record),
            ReplayKind::NumericalIdentity,
            "tenferro replay currently supports this family only for square float32/float64/complex64/complex128 inputs",
        ),
        ("cross", "identity", "identity", "success")
        | ("householder_product", "identity", "identity", "success")
        | ("multi_dot", "identity", "identity", "success")
        | ("pinv_hermitian", "identity", "identity", "success")
        | ("tensorinv", "identity", "identity", "success")
        | ("tensorsolve", "identity", "identity", "success")
        | ("vander", "identity", "identity", "success")
        | ("vecdot", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro has no matching public traced linalg API to replay this oracle family yet",
        },
        ("det", "identity", "identity", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::DetIdentity,
            "tenferro replay currently supports this family only for float32/float64/complex64/complex128",
        ),
        ("slogdet", "identity", "identity", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::SlogdetIdentity,
            "tenferro replay currently supports this family only for float32/float64/complex64/complex128",
        ),
        ("norm", "identity", "identity", "success") => {
            if replayable_norm_kind(record).is_some() {
                RecordSupport::Supported(ReplayKind::NormIdentity)
            } else {
                RecordSupport::Unsupported {
                    reason: NORM_UNSUPPORTED_REASON,
                }
            }
        }
        ("vector_norm", "identity", "identity", "success") => {
            if replayable_norm_kind(record).is_some() {
                RecordSupport::Supported(ReplayKind::NormIdentity)
            } else {
                RecordSupport::Unsupported {
                    reason: VECTOR_NORM_UNSUPPORTED_REASON,
                }
            }
        }
        ("matrix_norm", "identity", "identity", "success") => {
            if replayable_norm_kind(record).is_some() {
                RecordSupport::Supported(ReplayKind::NormIdentity)
            } else {
                RecordSupport::Unsupported {
                    reason: MATRIX_NORM_UNSUPPORTED_REASON,
                }
            }
        }
        ("eigvals", "identity", "identity", "success")
        | ("eigvalsh", "identity", "identity", "success")
        | ("svdvals", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this scalar-output oracle family yet",
        },
        ("diagonal", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this tensor-construction oracle family yet",
        },
        ("eig", "values_vectors_abs", "eig_values_vectors_abs", "success") => {
            RecordSupport::Unsupported {
                reason: "tenferro Eig AD support is values-only; eigenvector observables are unsupported by the manifest",
            }
        }
        ("pinv", "identity", "identity", "success") => supported_if(
            batch_a_replay_dtype(&record.dtype),
            ReplayKind::PinvIdentity,
            "tenferro replay currently supports this family only for float64/complex64/complex128",
        ),
        ("lstsq_grad_oriented", "identity", "identity", "success") => {
            if replayable_lstsq_subset(record) {
                RecordSupport::Supported(ReplayKind::LstsqIdentity)
            } else {
                RecordSupport::Unsupported {
                    reason: "tenferro replay currently supports only the real-valued m>=n least-squares subset excluding unsupported driver-specific oracle cases",
                }
            }
        }
        _ => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this oracle family yet",
        },
    }
}
