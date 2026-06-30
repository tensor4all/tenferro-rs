#[cfg(test)]
use crate::extension::LinalgOp;

/// AD rule support status for a linalg operation or output.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{linalg_ad_support, LinalgAdOpKind, LinalgAdRuleSupport};
///
/// let svd = linalg_ad_support(LinalgAdOpKind::Svd);
/// assert_eq!(svd.linearize, LinalgAdRuleSupport::SupportedViaLinearize);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinalgAdRuleSupport {
    Supported,
    SupportedViaLinearize,
    PartiallySupported,
    NonDifferentiable,
    Unsupported,
    PendingOracle,
}

/// Operation keys covered by the linalg AD support manifest.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::LinalgAdOpKind;
///
/// assert!(LinalgAdOpKind::Svd.as_index() < LinalgAdOpKind::COUNT);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinalgAdOpKind {
    Cholesky,
    Lu,
    LuFactor,
    LuSolvePrepared,
    FullPivLu,
    FullPivLuSolve,
    Svd,
    SvdVals,
    Qr,
    Eigh,
    EighVals,
    Eig,
    EigVals,
    TriangularSolve,
}

impl LinalgAdOpKind {
    pub const COUNT: usize = 14;

    /// Return the manifest index for this operation kind.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::LinalgAdOpKind;
    ///
    /// assert_eq!(LinalgAdOpKind::Cholesky.as_index(), 0);
    /// ```
    pub const fn as_index(self) -> usize {
        match self {
            Self::Cholesky => 0,
            Self::Lu => 1,
            Self::LuFactor => 2,
            Self::LuSolvePrepared => 3,
            Self::FullPivLu => 4,
            Self::FullPivLuSolve => 5,
            Self::Svd => 6,
            Self::SvdVals => 7,
            Self::Qr => 8,
            Self::Eigh => 9,
            Self::EighVals => 10,
            Self::Eig => 11,
            Self::EigVals => 12,
            Self::TriangularSolve => 13,
        }
    }

    #[cfg(test)]
    pub(crate) const fn from_linalg_op(op: LinalgOp) -> Self {
        match op {
            LinalgOp::Cholesky => Self::Cholesky,
            LinalgOp::Lu => Self::Lu,
            LinalgOp::LuFactor => Self::LuFactor,
            LinalgOp::LuSolvePrepared { .. } => Self::LuSolvePrepared,
            LinalgOp::FullPivLu => Self::FullPivLu,
            LinalgOp::FullPivLuSolve { .. } => Self::FullPivLuSolve,
            LinalgOp::Svd { .. } => Self::Svd,
            LinalgOp::SvdVals { .. } => Self::SvdVals,
            LinalgOp::Qr => Self::Qr,
            LinalgOp::Eigh { .. } => Self::Eigh,
            LinalgOp::EighVals { .. } => Self::EighVals,
            LinalgOp::Eig { .. } => Self::Eig,
            LinalgOp::EigVals { .. } => Self::EigVals,
            LinalgOp::TriangularSolve { .. } => Self::TriangularSolve,
        }
    }
}

/// AD support status for one output of a linalg operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{linalg_ad_support, LinalgAdOpKind, LinalgAdRuleSupport};
///
/// let full_piv_lu = linalg_ad_support(LinalgAdOpKind::FullPivLu);
/// let l_output = full_piv_lu.outputs.iter().find(|output| output.name == "l").unwrap();
/// assert_eq!(l_output.status, LinalgAdRuleSupport::SupportedViaLinearize);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinalgAdOutputSupport {
    /// Output position in the linalg operation result tuple.
    pub index: usize,
    /// Stable output name used by tests and support dashboards.
    pub name: &'static str,
    /// AD support status for this specific output.
    pub status: LinalgAdRuleSupport,
}

/// AD support manifest entry for one linalg operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{linalg_ad_support, LinalgAdOpKind, LinalgAdRuleSupport};
///
/// let solve = linalg_ad_support(LinalgAdOpKind::TriangularSolve);
/// assert_eq!(solve.transpose, LinalgAdRuleSupport::Supported);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinalgAdSupport {
    /// Operation kind described by this manifest entry.
    pub kind: LinalgAdOpKind,
    /// Forward-mode graph emission support.
    pub linearize: LinalgAdRuleSupport,
    /// Transposed-linear graph emission support.
    pub transpose: LinalgAdRuleSupport,
    /// Per-output support status for multi-output operations.
    pub outputs: &'static [LinalgAdOutputSupport],
}

const fn output(
    index: usize,
    name: &'static str,
    status: LinalgAdRuleSupport,
) -> LinalgAdOutputSupport {
    LinalgAdOutputSupport {
        index,
        name,
        status,
    }
}

static CHOLESKY_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "factor",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];
static LU_OUTPUTS: [LinalgAdOutputSupport; 4] = [
    output(0, "p", LinalgAdRuleSupport::NonDifferentiable),
    output(1, "l", LinalgAdRuleSupport::SupportedViaLinearize),
    output(2, "u", LinalgAdRuleSupport::SupportedViaLinearize),
    output(3, "parity", LinalgAdRuleSupport::NonDifferentiable),
];
static LU_FACTOR_OUTPUTS: [LinalgAdOutputSupport; 3] = [
    output(0, "packed_lu", LinalgAdRuleSupport::Unsupported),
    output(1, "pivots", LinalgAdRuleSupport::NonDifferentiable),
    output(2, "parity", LinalgAdRuleSupport::NonDifferentiable),
];
static SOLUTION_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "solution",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];
static FULL_PIV_LU_OUTPUTS: [LinalgAdOutputSupport; 5] = [
    output(0, "p", LinalgAdRuleSupport::NonDifferentiable),
    output(1, "l", LinalgAdRuleSupport::SupportedViaLinearize),
    output(2, "u", LinalgAdRuleSupport::SupportedViaLinearize),
    output(3, "q", LinalgAdRuleSupport::NonDifferentiable),
    output(4, "parity", LinalgAdRuleSupport::NonDifferentiable),
];
static FULL_PIV_LU_SOLVE_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "solution",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];
static SVD_OUTPUTS: [LinalgAdOutputSupport; 3] = [
    output(0, "u", LinalgAdRuleSupport::SupportedViaLinearize),
    output(
        1,
        "singular_values",
        LinalgAdRuleSupport::SupportedViaLinearize,
    ),
    output(2, "vt", LinalgAdRuleSupport::SupportedViaLinearize),
];
static SVD_VALS_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "singular_values",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];
static QR_OUTPUTS: [LinalgAdOutputSupport; 2] = [
    output(0, "q", LinalgAdRuleSupport::SupportedViaLinearize),
    output(1, "r", LinalgAdRuleSupport::SupportedViaLinearize),
];
static EIGH_OUTPUTS: [LinalgAdOutputSupport; 2] = [
    output(0, "eigenvalues", LinalgAdRuleSupport::SupportedViaLinearize),
    output(
        1,
        "eigenvectors",
        LinalgAdRuleSupport::SupportedViaLinearize,
    ),
];
static EIGH_VALS_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "eigenvalues",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];
static EIG_OUTPUTS: [LinalgAdOutputSupport; 2] = [
    output(0, "eigenvalues", LinalgAdRuleSupport::SupportedViaLinearize),
    output(1, "eigenvectors", LinalgAdRuleSupport::Unsupported),
];
static EIG_VALS_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "eigenvalues",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];

static LINALG_AD_SUPPORT: [LinalgAdSupport; LinalgAdOpKind::COUNT] = [
    LinalgAdSupport {
        kind: LinalgAdOpKind::Cholesky,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &CHOLESKY_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::Lu,
        linearize: LinalgAdRuleSupport::PartiallySupported,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &LU_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::LuFactor,
        linearize: LinalgAdRuleSupport::Unsupported,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &LU_FACTOR_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::LuSolvePrepared,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::PartiallySupported,
        outputs: &SOLUTION_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::FullPivLu,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &FULL_PIV_LU_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::FullPivLuSolve,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Supported,
        outputs: &FULL_PIV_LU_SOLVE_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::Svd,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &SVD_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::SvdVals,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &SVD_VALS_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::Qr,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &QR_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::Eigh,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Supported,
        outputs: &EIGH_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::EighVals,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Supported,
        outputs: &EIGH_VALS_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::Eig,
        linearize: LinalgAdRuleSupport::PartiallySupported,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &EIG_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::EigVals,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Unsupported,
        outputs: &EIG_VALS_OUTPUTS,
    },
    LinalgAdSupport {
        kind: LinalgAdOpKind::TriangularSolve,
        linearize: LinalgAdRuleSupport::SupportedViaLinearize,
        transpose: LinalgAdRuleSupport::Supported,
        outputs: &SOLUTION_OUTPUTS,
    },
];

/// Return the complete linalg AD support manifest.
///
/// # Examples
///
/// ```rust
/// let manifest = tenferro_linalg::all_linalg_ad_support();
/// assert_eq!(manifest.len(), tenferro_linalg::LinalgAdOpKind::COUNT);
/// ```
pub fn all_linalg_ad_support() -> &'static [LinalgAdSupport; LinalgAdOpKind::COUNT] {
    &LINALG_AD_SUPPORT
}

/// Return the support manifest entry for one linalg operation kind.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{linalg_ad_support, LinalgAdOpKind};
///
/// let entry = linalg_ad_support(LinalgAdOpKind::Eigh);
/// assert_eq!(entry.kind, LinalgAdOpKind::Eigh);
/// ```
pub fn linalg_ad_support(kind: LinalgAdOpKind) -> &'static LinalgAdSupport {
    &LINALG_AD_SUPPORT[kind.as_index()]
}

#[cfg(test)]
pub(crate) fn linalg_ad_support_for_op(op: LinalgOp) -> &'static LinalgAdSupport {
    linalg_ad_support(LinalgAdOpKind::from_linalg_op(op))
}

#[cfg(test)]
mod tests;
