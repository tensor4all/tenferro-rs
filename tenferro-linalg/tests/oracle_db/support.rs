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
    Unknown,
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
        | ("lu_solve", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::SolveIdentity)
        }
        ("cholesky", "identity", "identity", "success")
        | ("cholesky_ex", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::CholeskyIdentity)
        }
        ("inv_ex", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::InvIdentity)
        }
        ("lu_factor", "identity", "identity", "success")
        | ("lu_factor_ex", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::LuFactorIdentity)
        }
        ("cond", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::CondIdentity)
        }
        ("matrix_power", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::MatrixPowerIdentity)
        }
        ("qr", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::QrIdentity)
        }
        ("svd", "u_abs", "svd_u_abs", "success") => RecordSupport::Supported(ReplayKind::SvdUAbs),
        ("svd", "s", "svd_s", "success") => RecordSupport::Supported(ReplayKind::SvdS),
        ("svd", "vh_abs", "svd_vh_abs", "success") => {
            RecordSupport::Supported(ReplayKind::SvdVhAbs)
        }
        ("svd", "uvh_product", "svd_uvh_product", "success") => {
            RecordSupport::Supported(ReplayKind::SvdUvhProduct)
        }
        ("eigh", "values_vectors_abs", "eigh_values_vectors_abs", "success") => {
            RecordSupport::Supported(ReplayKind::EighValuesVectorsAbs)
        }
        ("pinv_singular", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::PinvSingularIdentity)
        }
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
        | ("vecdot", "identity", "identity", "success") => {
            RecordSupport::Supported(ReplayKind::NumericalIdentity)
        }
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
        _ => RecordSupport::Unknown,
    }
}
