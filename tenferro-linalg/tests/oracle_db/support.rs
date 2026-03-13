use crate::db::CaseRecord;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayKind {
    SolveIdentity,
    CholeskyIdentity,
    InvIdentity,
    LuFactorIdentity,
    CondIdentity,
    MatrixPowerIdentity,
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

fn svd_replay_dtype(dtype: &str) -> bool {
    matches!(dtype, "float32" | "float64" | "complex64" | "complex128")
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
        ("cholesky", "identity", "identity", "success")
        | ("cholesky_ex", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::CholeskyIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("inv_ex", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::InvIdentity,
            "tenferro replay currently supports this family only for float64",
        ),
        ("lu_factor", "identity", "identity", "success")
        | ("lu_factor_ex", "identity", "identity", "success") => supported_if(
            float64_only(&record.dtype),
            ReplayKind::LuFactorIdentity,
            "tenferro replay currently supports this family only for float64",
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
            float64_only(&record.dtype),
            ReplayKind::EighValuesVectorsAbs,
            "tenferro replay currently supports this family only for float64",
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
        ("det", "identity", "identity", "success")
        | ("eigvals", "identity", "identity", "success")
        | ("eigvalsh", "identity", "identity", "success")
        | ("matrix_norm", "identity", "identity", "success")
        | ("norm", "identity", "identity", "success")
        | ("slogdet", "identity", "identity", "success")
        | ("svdvals", "identity", "identity", "success")
        | ("vector_norm", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this scalar-output oracle family yet",
        },
        ("diagonal", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this tensor-construction oracle family yet",
        },
        ("eig", "values_vectors_abs", "eig_values_vectors_abs", "success")
        | ("inv", "identity", "identity", "success")
        | ("pinv", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this spectral/inverse family yet",
        },
        ("lstsq_grad_oriented", "identity", "identity", "success")
        | ("lu", "identity", "identity", "success")
        | ("solve_triangular", "identity", "identity", "success") => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this solver/decomposition family yet",
        },
        _ => RecordSupport::Unsupported {
            reason: "tenferro replay does not implement this oracle family yet",
        },
    }
}
