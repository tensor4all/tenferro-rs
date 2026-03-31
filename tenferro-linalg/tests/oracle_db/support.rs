use serde_json::Value;
use tenferro_linalg::NormKind;

use crate::db::CaseRecord;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayKind {
    SolveIdentity,
    SolveTriangularIdentity,
    CholeskyIdentity,
    InvIdentity,
    DetIdentity,
    SlogdetIdentity,
    LuFactorIdentity,
    LuIdentity,
    NormIdentity,
    CondIdentity,
    MatrixPowerIdentity,
    MatrixExpIdentity,
    PinvIdentity,
    NumericalIdentity,
    QrIdentity,
    SvdUAbs,
    SvdS,
    SvdVhAbs,
    SvdUvhProduct,
    EighValuesVectorsAbs,
    PinvSingularIdentity,
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

pub fn replayable_norm_kind(record: &CaseRecord) -> Option<NormKind> {
    let rank = record.inputs.get("a")?.shape.len();
    let kind = match record.op.as_str() {
        "norm" => replayable_norm_kind_from_norm(record, rank)?,
        "matrix_norm" => replayable_norm_kind_from_matrix_norm(record, rank)?,
        _ => return None,
    };
    if is_complex_dtype(&record.dtype) {
        match kind {
            NormKind::Fro => Some(kind),
            NormKind::Lp(p) if p == 2.0 => Some(kind),
            _ => None,
        }
    } else {
        Some(kind)
    }
}

fn replayable_norm_kind_from_norm(record: &CaseRecord, rank: usize) -> Option<NormKind> {
    if rank == 0 || rank > 2 {
        return None;
    }

    if rank == 1 {
        let arg = record.op_args.first();
        return match arg {
            None | Some(Value::Null) => Some(NormKind::Lp(2.0)),
            Some(Value::Number(_) | Value::String(_)) => {
                let ord = value_as_f64(arg?)?;
                if ord == 1.0 {
                    Some(NormKind::L1)
                } else if ord == 2.0 {
                    Some(NormKind::Lp(2.0))
                } else if ord.is_infinite() && ord.is_sign_positive() {
                    Some(NormKind::Inf)
                } else if ord >= 1.0 {
                    Some(NormKind::Lp(ord))
                } else {
                    None
                }
            }
            _ => None,
        };
    }

    if let Some(dim) = record.op_kwargs.get("dim") {
        if !matrix_axes_cover_rank(dim, rank) {
            return None;
        }
        return match record.op_kwargs.get("ord") {
            Some(Value::String(text)) if text == "fro" => Some(NormKind::Fro),
            Some(Value::String(text)) if text == "nuc" => Some(NormKind::Nuclear),
            _ => None,
        };
    }

    let arg = record.op_args.first();
    match arg {
        None | Some(Value::Null) => Some(NormKind::Fro),
        Some(Value::String(text)) if text == "fro" => Some(NormKind::Fro),
        Some(Value::String(text)) if text == "nuc" => Some(NormKind::Nuclear),
        Some(Value::Number(_) | Value::String(_)) => {
            let ord = value_as_f64(arg?)?;
            if ord == 1.0 {
                Some(NormKind::L1)
            } else if ord == 2.0 {
                Some(NormKind::Spectral)
            } else if ord.is_infinite() && ord.is_sign_positive() {
                Some(NormKind::Inf)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn replayable_norm_kind_from_matrix_norm(record: &CaseRecord, rank: usize) -> Option<NormKind> {
    if rank != 2 || record.op_args.len() != 3 {
        return None;
    }
    let axes = matrix_axes_order(&record.op_args[1], rank)?;
    let reversed_axes = axes == vec![1, 0];
    match &record.op_args[0] {
        Value::String(text) if text == "fro" => Some(NormKind::Fro),
        Value::String(text) if text == "nuc" => Some(NormKind::Nuclear),
        Value::Number(_) | Value::String(_) => {
            let ord = value_as_f64(&record.op_args[0])?;
            if ord == 1.0 {
                Some(if reversed_axes {
                    NormKind::Inf
                } else {
                    NormKind::L1
                })
            } else if ord == 2.0 {
                Some(NormKind::Spectral)
            } else if ord.is_infinite() && ord.is_sign_positive() {
                Some(if reversed_axes {
                    NormKind::L1
                } else {
                    NormKind::Inf
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
        | ("solve_ex", "identity", "identity", "success")
        | ("lu_solve", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::SolveIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
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
        ("inv", "identity", "identity", "success") => supported_if(
            batch_a_replay_dtype(&record.dtype),
            ReplayKind::InvIdentity,
            "tenferro replay currently supports this family only for float64/complex64/complex128",
        ),
        ("inv_ex", "identity", "identity", "success") => supported_if(
            batch_a_replay_dtype(&record.dtype),
            ReplayKind::InvIdentity,
            "tenferro replay currently supports this family only for float64/complex64/complex128",
        ),
        ("lu_factor", "identity", "identity", "success")
        | ("lu_factor_ex", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::LuFactorIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("lu", "identity", "identity", "success") => supported_if(
            svd_replay_dtype(&record.dtype),
            ReplayKind::LuIdentity,
            "tenferro replay currently supports this family only for float32/float64/complex64/complex128",
        ),
        ("cond", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::CondIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("matrix_power", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::MatrixPowerIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("matrix_exp", "identity", "identity", "success") => supported_if(
            batch_a_replay_dtype(&record.dtype),
            ReplayKind::MatrixExpIdentity,
            "tenferro replay currently supports this family only for float64/complex64/complex128",
        ),
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
            float64_only(&record.dtype),
            ReplayKind::PinvSingularIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("svd", "gauge_ill_defined", "svd_uvh_product", "error")
        | ("eigh", "gauge_ill_defined", "eigh_values_vectors_abs", "error") => {
            RecordSupport::ExpectedError(ExpectedErrorKind::GaugeIllDefined)
        }
        ("cross", "identity", "identity", "success")
        | ("householder_product", "identity", "identity", "success")
        | ("multi_dot", "identity", "identity", "success")
        | ("pinv_hermitian", "identity", "identity", "success")
        | ("tensorinv", "identity", "identity", "success")
        | ("tensorsolve", "identity", "identity", "success")
        | ("vander", "identity", "identity", "success")
        | ("vecdot", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::NumericalIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
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
        ("norm", "identity", "identity", "success")
        | ("matrix_norm", "identity", "identity", "success") => {
            if replayable_norm_kind(record).is_some() {
                RecordSupport::Supported(ReplayKind::NormIdentity)
            } else {
                RecordSupport::Unsupported {
                    reason: "tenferro replay currently supports only the scalar-output norm subset covered by current NormKind AD rules",
                }
            }
        }
        ("eigvals", "identity", "identity", "success")
        | ("eigvalsh", "identity", "identity", "success")
        | ("svdvals", "identity", "identity", "success")
        | ("vector_norm", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this scalar-output oracle family yet",
        },
        ("diagonal", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this tensor-construction oracle family yet",
        },
        ("eig", "values_vectors_abs", "eig_values_vectors_abs", "success") => {
            RecordSupport::Unsupported {
                reason: "tenferro replay does not implement this spectral/inverse family yet",
            }
        }
        ("pinv", "identity", "identity", "success") => supported_if(
            batch_a_replay_dtype(&record.dtype),
            ReplayKind::PinvIdentity,
            "tenferro replay currently supports this family only for float64/complex64/complex128",
        ),
        ("lstsq_grad_oriented", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this solver/decomposition family yet",
        },
        _ => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this oracle family yet",
        },
    }
}
