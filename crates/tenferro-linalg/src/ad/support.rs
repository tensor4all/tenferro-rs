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

/// Implementation route used for a user-visible AD mode.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{linalg_ad_support, LinalgAdOpKind, LinalgAdRoute};
///
/// let svd = linalg_ad_support(LinalgAdOpKind::Svd);
/// assert_eq!(svd.vjp.route, LinalgAdRoute::LinearizeThenTranspose);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinalgAdRoute {
    /// No supported route exists.
    Unsupported,
    /// The mode is emitted directly by the operation's linearize rule.
    Linearize,
    /// Reverse mode is supported by linearizing the primal graph and then
    /// transposing that linear graph.
    LinearizeThenTranspose,
    /// Reverse mode is supported by linearizing the primal graph and using a
    /// custom transposed-linear rule for the emitted linear operation.
    LinearizeThenCustomLinearTranspose,
    /// Reverse mode is emitted by a direct operation-specific primal VJP rule.
    CustomVjp,
    /// A custom VJP is preferred, with the canonical linearize-then-transpose
    /// route retained as an intentional not-applicable fallback.
    CustomPreferredWithLinearizeFallback,
}

/// User-visible AD support for one mode.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::{linalg_ad_support, LinalgAdOpKind, LinalgAdRoute};
///
/// let solve = linalg_ad_support(LinalgAdOpKind::TriangularSolve);
/// assert_eq!(solve.vjp.route, LinalgAdRoute::LinearizeThenCustomLinearTranspose);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinalgAdModeSupport {
    /// Whether the user-visible mode is supported.
    pub status: LinalgAdRuleSupport,
    /// How the mode is implemented.
    pub route: LinalgAdRoute,
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
    SignDetFromLuFactor,
    LogAbsDetFromLuFactor,
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
    SvdFull,
    HouseholderQrFactor,
    HouseholderQrFromFactors,
    HouseholderQrAppend,
    HouseholderQrR,
    HouseholderQrQColumns,
    HouseholderQrThinQ,
    HouseholderQrAppendTangent,
    HouseholderQrSplitTangent,
}

impl LinalgAdOpKind {
    pub const COUNT: usize = 25;

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
            Self::SignDetFromLuFactor => 3,
            Self::LogAbsDetFromLuFactor => 4,
            Self::LuSolvePrepared => 5,
            Self::FullPivLu => 6,
            Self::FullPivLuSolve => 7,
            Self::Svd => 8,
            Self::SvdVals => 9,
            Self::Qr => 10,
            Self::Eigh => 11,
            Self::EighVals => 12,
            Self::Eig => 13,
            Self::EigVals => 14,
            Self::TriangularSolve => 15,
            Self::SvdFull => 16,
            Self::HouseholderQrFactor => 17,
            Self::HouseholderQrFromFactors => 18,
            Self::HouseholderQrAppend => 19,
            Self::HouseholderQrR => 20,
            Self::HouseholderQrQColumns => 21,
            Self::HouseholderQrThinQ => 22,
            Self::HouseholderQrAppendTangent => 23,
            Self::HouseholderQrSplitTangent => 24,
        }
    }

    #[cfg(test)]
    pub(crate) const fn from_linalg_op(op: LinalgOp) -> Self {
        match op {
            LinalgOp::Cholesky => Self::Cholesky,
            LinalgOp::Lu => Self::Lu,
            LinalgOp::LuFactor => Self::LuFactor,
            LinalgOp::SignDetFromLuFactor => Self::SignDetFromLuFactor,
            LinalgOp::LogAbsDetFromLuFactor => Self::LogAbsDetFromLuFactor,
            LinalgOp::LuSolvePrepared { .. } => Self::LuSolvePrepared,
            LinalgOp::FullPivLu => Self::FullPivLu,
            LinalgOp::FullPivLuSolve { .. } => Self::FullPivLuSolve,
            // The partial-pivot single-op solve shares the solve-family AD
            // route (linearize + custom linear transpose) with FullPivLuSolve
            // and is not separately exposed in the public manifest.
            LinalgOp::Solve => Self::FullPivLuSolve,
            LinalgOp::Svd { .. } => Self::Svd,
            LinalgOp::SvdFull => Self::SvdFull,
            LinalgOp::SvdVals { .. } => Self::SvdVals,
            LinalgOp::Qr { .. } => Self::Qr,
            LinalgOp::HouseholderQrFactor => Self::HouseholderQrFactor,
            LinalgOp::HouseholderQrFromFactors => Self::HouseholderQrFromFactors,
            LinalgOp::HouseholderQrAppend => Self::HouseholderQrAppend,
            LinalgOp::HouseholderQrR { .. } => Self::HouseholderQrR,
            LinalgOp::HouseholderQrQColumns { .. } => Self::HouseholderQrQColumns,
            LinalgOp::HouseholderQrThinQ { .. } => Self::HouseholderQrThinQ,
            LinalgOp::HouseholderQrAppendTangent => Self::HouseholderQrAppendTangent,
            LinalgOp::HouseholderQrSplitTangent { .. } => Self::HouseholderQrSplitTangent,
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
/// assert_eq!(solve.vjp.route, tenferro_linalg::LinalgAdRoute::LinearizeThenCustomLinearTranspose);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinalgAdSupport {
    /// Operation kind described by this manifest entry.
    pub kind: LinalgAdOpKind,
    /// User-visible JVP support and route.
    pub jvp: LinalgAdModeSupport,
    /// User-visible VJP support and route.
    pub vjp: LinalgAdModeSupport,
    /// Definitional linearize rule implementation status.
    pub linearize_rule: LinalgAdRuleSupport,
    /// Direct primal VJP rule implementation status.
    pub custom_vjp_rule: LinalgAdRuleSupport,
    /// Custom transposed-linear rule implementation status.
    pub custom_linear_transpose_rule: LinalgAdRuleSupport,
    /// Forward-mode graph emission support.
    pub linearize: LinalgAdRuleSupport,
    /// Transposed-linear graph emission support.
    pub transpose: LinalgAdRuleSupport,
    /// Per-output support status for multi-output operations.
    pub outputs: &'static [LinalgAdOutputSupport],
    /// Numerical or semantic caveats for this operation family.
    pub caveats: &'static [&'static str],
}

const fn mode(status: LinalgAdRuleSupport, route: LinalgAdRoute) -> LinalgAdModeSupport {
    LinalgAdModeSupport { status, route }
}

const fn jvp_route(status: LinalgAdRuleSupport) -> LinalgAdRoute {
    match status {
        LinalgAdRuleSupport::Unsupported
        | LinalgAdRuleSupport::NonDifferentiable
        | LinalgAdRuleSupport::PendingOracle => LinalgAdRoute::Unsupported,
        LinalgAdRuleSupport::Supported
        | LinalgAdRuleSupport::SupportedViaLinearize
        | LinalgAdRuleSupport::PartiallySupported => LinalgAdRoute::Linearize,
    }
}

const fn support_entry(
    kind: LinalgAdOpKind,
    linearize: LinalgAdRuleSupport,
    transpose: LinalgAdRuleSupport,
    vjp: LinalgAdModeSupport,
    custom_linear_transpose_rule: LinalgAdRuleSupport,
    outputs: &'static [LinalgAdOutputSupport],
    caveats: &'static [&'static str],
) -> LinalgAdSupport {
    LinalgAdSupport {
        kind,
        jvp: mode(linearize, jvp_route(linearize)),
        vjp,
        linearize_rule: linearize,
        custom_vjp_rule: LinalgAdRuleSupport::Unsupported,
        custom_linear_transpose_rule,
        linearize,
        transpose,
        outputs,
        caveats,
    }
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
static SIGNDET_FROM_LU_FACTOR_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "sign",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];
static LOGABSDET_FROM_LU_FACTOR_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "logabsdet",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];
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
static SVD_FULL_OUTPUTS: [LinalgAdOutputSupport; 3] = [
    output(0, "u", LinalgAdRuleSupport::Unsupported),
    output(1, "singular_values", LinalgAdRuleSupport::Unsupported),
    output(2, "vt", LinalgAdRuleSupport::Unsupported),
];
static QR_OUTPUTS: [LinalgAdOutputSupport; 2] = [
    output(0, "q", LinalgAdRuleSupport::SupportedViaLinearize),
    output(1, "r", LinalgAdRuleSupport::SupportedViaLinearize),
];
static HOUSEHOLDER_QR_STATE_OUTPUTS: [LinalgAdOutputSupport; 2] = [
    output(0, "packed", LinalgAdRuleSupport::SupportedViaLinearize),
    output(1, "coeff", LinalgAdRuleSupport::NonDifferentiable),
];
static HOUSEHOLDER_QR_VALUE_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "value",
    LinalgAdRuleSupport::SupportedViaLinearize,
)];
static HOUSEHOLDER_QR_RESIDUAL_OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
    0,
    "internal_value",
    LinalgAdRuleSupport::Unsupported,
)];
static HOUSEHOLDER_QR_CAVEATS: [&str; 1] =
    ["Rank-deficient states are outside the differentiable domain."];
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

static DECOMPOSITION_CAVEATS: [&str; 1] = [
    "Derivative regularization handles near-degenerate spectra but does not make exact degeneracies smoothly differentiable.",
];

static LINALG_AD_SUPPORT: [LinalgAdSupport; LinalgAdOpKind::COUNT] = [
    support_entry(
        LinalgAdOpKind::Cholesky,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &CHOLESKY_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::Lu,
        LinalgAdRuleSupport::PartiallySupported,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::PartiallySupported,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &LU_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::LuFactor,
        LinalgAdRuleSupport::Unsupported,
        LinalgAdRuleSupport::Unsupported,
        mode(LinalgAdRuleSupport::Unsupported, LinalgAdRoute::Unsupported),
        LinalgAdRuleSupport::Unsupported,
        &LU_FACTOR_OUTPUTS,
        &[],
    ),
    support_entry(
        LinalgAdOpKind::SignDetFromLuFactor,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &SIGNDET_FROM_LU_FACTOR_OUTPUTS,
        &[],
    ),
    support_entry(
        LinalgAdOpKind::LogAbsDetFromLuFactor,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &LOGABSDET_FROM_LU_FACTOR_OUTPUTS,
        &[],
    ),
    support_entry(
        LinalgAdOpKind::LuSolvePrepared,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::PartiallySupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenCustomLinearTranspose,
        ),
        LinalgAdRuleSupport::PartiallySupported,
        &SOLUTION_OUTPUTS,
        &[],
    ),
    support_entry(
        LinalgAdOpKind::FullPivLu,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &FULL_PIV_LU_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::FullPivLuSolve,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Supported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenCustomLinearTranspose,
        ),
        LinalgAdRuleSupport::Supported,
        &FULL_PIV_LU_SOLVE_OUTPUTS,
        &[],
    ),
    support_entry(
        LinalgAdOpKind::Svd,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &SVD_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::SvdVals,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &SVD_VALS_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::Qr,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &QR_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::Eigh,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &EIGH_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::EighVals,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &EIGH_VALS_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::Eig,
        LinalgAdRuleSupport::PartiallySupported,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::PartiallySupported,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &EIG_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::EigVals,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &EIG_VALS_OUTPUTS,
        &DECOMPOSITION_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::TriangularSolve,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Supported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenCustomLinearTranspose,
        ),
        LinalgAdRuleSupport::Supported,
        &SOLUTION_OUTPUTS,
        &[],
    ),
    // Full-matrices SVD is a value-only route: nullspace/kernel extraction does
    // not require derivatives, and the thin-SVD linearize rule does not extend
    // to the square factors. AD is intentionally unsupported for every output.
    support_entry(
        LinalgAdOpKind::SvdFull,
        LinalgAdRuleSupport::Unsupported,
        LinalgAdRuleSupport::Unsupported,
        mode(LinalgAdRuleSupport::Unsupported, LinalgAdRoute::Unsupported),
        LinalgAdRuleSupport::Unsupported,
        &SVD_FULL_OUTPUTS,
        &[],
    ),
    support_entry(
        LinalgAdOpKind::HouseholderQrFactor,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &HOUSEHOLDER_QR_STATE_OUTPUTS,
        &HOUSEHOLDER_QR_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::HouseholderQrFromFactors,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &HOUSEHOLDER_QR_STATE_OUTPUTS,
        &HOUSEHOLDER_QR_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::HouseholderQrAppend,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &HOUSEHOLDER_QR_STATE_OUTPUTS,
        &HOUSEHOLDER_QR_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::HouseholderQrR,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &HOUSEHOLDER_QR_VALUE_OUTPUTS,
        &HOUSEHOLDER_QR_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::HouseholderQrQColumns,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Unsupported,
        mode(
            LinalgAdRuleSupport::SupportedViaLinearize,
            LinalgAdRoute::LinearizeThenTranspose,
        ),
        LinalgAdRuleSupport::Unsupported,
        &HOUSEHOLDER_QR_VALUE_OUTPUTS,
        &HOUSEHOLDER_QR_CAVEATS,
    ),
    support_entry(
        LinalgAdOpKind::HouseholderQrThinQ,
        LinalgAdRuleSupport::Unsupported,
        LinalgAdRuleSupport::Unsupported,
        mode(LinalgAdRuleSupport::Unsupported, LinalgAdRoute::Unsupported),
        LinalgAdRuleSupport::Unsupported,
        &HOUSEHOLDER_QR_RESIDUAL_OUTPUTS,
        &["Internal fixed residual; no public differentiable surface."],
    ),
    support_entry(
        LinalgAdOpKind::HouseholderQrAppendTangent,
        LinalgAdRuleSupport::Unsupported,
        LinalgAdRuleSupport::Supported,
        mode(LinalgAdRuleSupport::Unsupported, LinalgAdRoute::Unsupported),
        LinalgAdRuleSupport::Supported,
        &HOUSEHOLDER_QR_RESIDUAL_OUTPUTS,
        &["Internal linear append operation; no public differentiable surface."],
    ),
    support_entry(
        LinalgAdOpKind::HouseholderQrSplitTangent,
        LinalgAdRuleSupport::Unsupported,
        LinalgAdRuleSupport::Unsupported,
        mode(LinalgAdRuleSupport::Unsupported, LinalgAdRoute::Unsupported),
        LinalgAdRuleSupport::Unsupported,
        &HOUSEHOLDER_QR_RESIDUAL_OUTPUTS,
        &["Internal transpose residual; no public differentiable surface."],
    ),
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
