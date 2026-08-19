//! Semantic-program automatic-differentiation rules for extension operations.
//!
//! Rules in this module are owned explicitly by [`crate::AdContext`]. They
//! operate on opaque semantic [`ProgramValue`] tokens and never expose
//! computegraph node keys or execution-program slots.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

pub use tenferro_ops::ad::ResidualSpec;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_runtime::program::{
    ProgramBuildError, ProgramValue, ProgramValueMetadata, SemanticOpRef, SemanticOperationView,
    SemanticProgramBuilder, SemanticProvenanceView,
};

/// One optional semantic AD value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdValue {
    /// This primal, tangent, or cotangent is inactive.
    Absent,
    /// Active value owned by the destination semantic-program builder.
    Value(ProgramValue),
}

impl AdValue {
    /// Return the active semantic value, if present.
    #[must_use]
    pub const fn value(self) -> Option<ProgramValue> {
        match self {
            Self::Absent => None,
            Self::Value(value) => Some(value),
        }
    }
}

/// Semantic extension AD rule role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SemanticAdRuleRole {
    /// Definitional forward linearization.
    Linearize,
    /// Transpose of a linearized extension operation.
    LinearTranspose,
    /// Direct reverse rule expressed against primal values.
    PrimalVjp,
}

/// Registration failures for semantic extension AD rules.
#[derive(Debug, thiserror::Error)]
pub enum SemanticExtensionRegistryError {
    /// A rule with the same family and role is already present.
    #[error("semantic extension AD {role:?} rule for family {family_id:?} is already registered")]
    DuplicateRule {
        /// Duplicate extension family identifier.
        family_id: &'static str,
        /// Duplicate semantic AD role.
        role: SemanticAdRuleRole,
    },
    /// A rule family is not a namespaced, versioned identifier.
    #[error("semantic extension AD family {family_id:?} is not namespaced and versioned")]
    MalformedFamilyId {
        /// Invalid extension family identifier.
        family_id: &'static str,
    },
}

/// Whether a checked semantic AD request refers to a primal input or output.
///
/// # Examples
///
/// ```
/// use tenferro_ad::semantic_extension::PrimalValueKind;
///
/// assert_eq!(format!("{:?}", PrimalValueKind::Input), "Input");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PrimalValueKind {
    /// An ordered primal input.
    Input,
    /// An ordered primal output.
    Output,
}

/// Failures while dispatching or building semantic extension AD.
#[derive(Debug, thiserror::Error)]
pub enum SemanticAdError {
    /// The supplied operation is not an extension operation.
    #[error("semantic AD extension dispatch received a core operation")]
    CoreOperation,
    /// Observable effects make implicit differentiation unsafe.
    #[error("semantic extension family {family_id:?} has observable effects")]
    EffectfulExtension {
        /// Effectful extension family.
        family_id: &'static str,
    },
    /// No rule is registered for the requested family and role.
    #[error("semantic extension family {family_id:?} has no {role:?} AD rule")]
    MissingRule {
        /// Extension family without a rule.
        family_id: &'static str,
        /// Missing semantic AD role.
        role: SemanticAdRuleRole,
    },
    /// An ordered request or result field has the wrong length.
    #[error("semantic AD field {field} expects {expected} values, got {actual}")]
    Arity {
        /// Name of the invalid ordered field.
        field: &'static str,
        /// Required field length.
        expected: usize,
        /// Supplied field length.
        actual: usize,
    },
    /// A checked primal value index is outside the request's ordered values.
    #[error(
        "semantic extension family {family_id:?} {kind:?} index {index} is out of bounds for length {len}"
    )]
    PrimalIndexOutOfBounds {
        /// Extension family that owns the request.
        family_id: &'static str,
        /// Ordered primal collection being accessed.
        kind: PrimalValueKind,
        /// Requested index.
        index: usize,
        /// Number of values in the collection.
        len: usize,
    },
    /// A checked primal value was not declared as a tensor residual by the rule.
    #[error(
        "semantic extension family {family_id:?} may not access undeclared {kind:?} residual value {index}"
    )]
    UndeclaredResidualValue {
        /// Extension family that owns the request.
        family_id: &'static str,
        /// Ordered primal collection being accessed.
        kind: PrimalValueKind,
        /// Requested index.
        index: usize,
    },
    /// A request or result value belongs to another builder.
    #[error("semantic AD field {field}[{index}] does not belong to the destination builder")]
    ForeignValue {
        /// Name of the invalid ordered field.
        field: &'static str,
        /// Index of the foreign value.
        index: usize,
    },
    /// A family-specific rule deliberately rejects this payload.
    #[error("semantic extension family {family_id:?} does not support {role:?}: {message}")]
    Unsupported {
        /// Extension family that rejected the transform.
        family_id: &'static str,
        /// Rejected semantic AD role.
        role: SemanticAdRuleRole,
        /// Bounded family-specific diagnostic.
        message: String,
    },
    /// A family-specific rule failed with a typed source error.
    #[error("semantic extension family {family_id:?} {role:?} rule failed: {source}")]
    Rule {
        /// Extension family whose rule failed.
        family_id: &'static str,
        /// Semantic AD role being evaluated.
        role: SemanticAdRuleRole,
        /// Original typed rule failure.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// A family-specific semantic rule invariant was violated.
    #[error("semantic extension family {family_id:?} {role:?} invariant failed: {message}")]
    Invariant {
        /// Extension family whose invariant failed.
        family_id: &'static str,
        /// Semantic AD role being evaluated.
        role: SemanticAdRuleRole,
        /// Bounded invariant diagnostic.
        message: String,
    },
    /// Semantic-program construction failed inside a rule.
    #[error("semantic extension AD program construction failed: {0}")]
    Build(#[from] ProgramBuildError),
}

/// Ordered inputs for one semantic extension linearization rule.
#[derive(Clone, Copy)]
pub struct SemanticLinearizeRequest<'a> {
    op: &'a dyn ExtensionOp,
    primal_inputs: &'a [ProgramValue],
    primal_outputs: &'a [ProgramValue],
    tangent_inputs: &'a [AdValue],
    active_outputs: &'a [bool],
    provenance: SemanticProvenanceView<'a>,
}

impl<'a> SemanticLinearizeRequest<'a> {
    /// Borrow the extension payload.
    pub const fn op(self) -> &'a dyn ExtensionOp {
        self.op
    }

    /// Borrow ordered destination-local primal inputs.
    pub const fn primal_inputs(self) -> &'a [ProgramValue] {
        self.primal_inputs
    }

    /// Borrow ordered destination-local primal outputs.
    pub const fn primal_outputs(self) -> &'a [ProgramValue] {
        self.primal_outputs
    }

    /// Borrow ordered optional tangent inputs.
    pub const fn tangent_inputs(self) -> &'a [AdValue] {
        self.tangent_inputs
    }

    /// Borrow the ordered active-output mask.
    pub const fn active_outputs(self) -> &'a [bool] {
        self.active_outputs
    }

    /// Return bounded operation provenance.
    pub const fn provenance(self) -> SemanticProvenanceView<'a> {
        self.provenance
    }
}

/// Output of one semantic extension linearization rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticLinearizeResult {
    tangent_outputs: Box<[AdValue]>,
    residuals: Box<[ProgramValue]>,
}

impl SemanticLinearizeResult {
    /// Construct ordered tangent outputs and residuals.
    #[must_use]
    pub fn new(
        tangent_outputs: impl IntoIterator<Item = AdValue>,
        residuals: impl IntoIterator<Item = ProgramValue>,
    ) -> Self {
        Self {
            tangent_outputs: tangent_outputs.into_iter().collect(),
            residuals: residuals.into_iter().collect(),
        }
    }

    /// Borrow ordered optional tangent outputs.
    pub fn tangent_outputs(&self) -> &[AdValue] {
        &self.tangent_outputs
    }

    /// Borrow ordered residual values saved for transpose.
    pub fn residuals(&self) -> &[ProgramValue] {
        &self.residuals
    }
}

/// Ordered inputs for one semantic linear-transpose rule.
pub struct SemanticLinearTransposeRequest<'a> {
    op: &'a dyn ExtensionOp,
    primal_inputs: &'a [ProgramValue],
    primal_outputs: &'a [ProgramValue],
    primal_input_metadata: Box<[ProgramValueMetadata]>,
    primal_output_metadata: Box<[ProgramValueMetadata]>,
    cotangent_outputs: &'a [AdValue],
    active_inputs: &'a [bool],
    residuals: &'a [ProgramValue],
    residual_mask: ResidualSpec,
    provenance: SemanticProvenanceView<'a>,
}

impl<'a> SemanticLinearTransposeRequest<'a> {
    /// Borrow the extension payload.
    pub fn op(&self) -> &dyn ExtensionOp {
        self.op
    }

    /// Return one declared primal input tensor value.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::PrimalIndexOutOfBounds`] before checking the
    /// residual mask, or [`SemanticAdError::UndeclaredResidualValue`] when the
    /// rule did not declare this input as a tensor residual.
    pub fn primal_input_value(&self, index: usize) -> Result<ProgramValue, SemanticAdError> {
        checked_primal_value(
            self.op.family_id(),
            PrimalValueKind::Input,
            index,
            self.primal_inputs,
            self.residual_mask,
        )
    }

    /// Return one declared primal output tensor value.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::PrimalIndexOutOfBounds`] before checking the
    /// residual mask, or [`SemanticAdError::UndeclaredResidualValue`] when the
    /// rule did not declare this output as a tensor residual.
    pub fn primal_output_value(&self, index: usize) -> Result<ProgramValue, SemanticAdError> {
        checked_primal_value(
            self.op.family_id(),
            PrimalValueKind::Output,
            index,
            self.primal_outputs,
            self.residual_mask,
        )
    }

    /// Borrow metadata for one primal input without exposing its value token.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::PrimalIndexOutOfBounds`] when `index` is not
    /// an ordered primal input.
    pub fn primal_input_meta(
        &self,
        index: usize,
    ) -> Result<&ProgramValueMetadata, SemanticAdError> {
        checked_primal_metadata(
            self.op.family_id(),
            PrimalValueKind::Input,
            index,
            &self.primal_input_metadata,
        )
    }

    /// Borrow metadata for one primal output without exposing its value token.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::PrimalIndexOutOfBounds`] when `index` is not
    /// an ordered primal output.
    pub fn primal_output_meta(
        &self,
        index: usize,
    ) -> Result<&ProgramValueMetadata, SemanticAdError> {
        checked_primal_metadata(
            self.op.family_id(),
            PrimalValueKind::Output,
            index,
            &self.primal_output_metadata,
        )
    }

    /// Return the number of ordered primal inputs.
    #[must_use]
    pub const fn primal_input_count(&self) -> usize {
        self.primal_input_metadata.len()
    }

    /// Return the number of ordered primal outputs.
    #[must_use]
    pub const fn primal_output_count(&self) -> usize {
        self.primal_output_metadata.len()
    }

    /// Borrow ordered optional output cotangents.
    pub fn cotangent_outputs(&self) -> &[AdValue] {
        self.cotangent_outputs
    }

    /// Borrow the ordered active-input mask.
    pub fn active_inputs(&self) -> &[bool] {
        self.active_inputs
    }

    /// Borrow ordered residuals produced by linearization.
    pub fn residuals(&self) -> &[ProgramValue] {
        self.residuals
    }

    /// Return this rule's declared residual mask: which primal input/output
    /// indices may be read as tensor values. Accesses outside the mask must
    /// only use shape/dtype metadata.
    pub const fn residual_mask(&self) -> ResidualSpec {
        self.residual_mask
    }

    /// Return bounded operation provenance.
    pub const fn provenance(&self) -> SemanticProvenanceView<'a> {
        self.provenance
    }
}

/// Ordered inputs for one direct semantic primal-VJP rule.
pub struct SemanticPrimalVjpRequest<'a> {
    op: &'a dyn ExtensionOp,
    primal_inputs: &'a [ProgramValue],
    primal_outputs: &'a [ProgramValue],
    primal_input_metadata: Box<[ProgramValueMetadata]>,
    primal_output_metadata: Box<[ProgramValueMetadata]>,
    cotangent_outputs: &'a [AdValue],
    active_inputs: &'a [bool],
    residual_mask: ResidualSpec,
    provenance: SemanticProvenanceView<'a>,
}

impl<'a> SemanticPrimalVjpRequest<'a> {
    /// Borrow the extension payload.
    pub fn op(&self) -> &dyn ExtensionOp {
        self.op
    }

    /// Return one declared primal input tensor value.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::PrimalIndexOutOfBounds`] before checking the
    /// residual mask, or [`SemanticAdError::UndeclaredResidualValue`] when the
    /// rule did not declare this input as a tensor residual.
    pub fn primal_input_value(&self, index: usize) -> Result<ProgramValue, SemanticAdError> {
        checked_primal_value(
            self.op.family_id(),
            PrimalValueKind::Input,
            index,
            self.primal_inputs,
            self.residual_mask,
        )
    }

    /// Return one declared primal output tensor value.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::PrimalIndexOutOfBounds`] before checking the
    /// residual mask, or [`SemanticAdError::UndeclaredResidualValue`] when the
    /// rule did not declare this output as a tensor residual.
    pub fn primal_output_value(&self, index: usize) -> Result<ProgramValue, SemanticAdError> {
        checked_primal_value(
            self.op.family_id(),
            PrimalValueKind::Output,
            index,
            self.primal_outputs,
            self.residual_mask,
        )
    }

    /// Borrow metadata for one primal input without exposing its value token.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::PrimalIndexOutOfBounds`] when `index` is not
    /// an ordered primal input.
    pub fn primal_input_meta(
        &self,
        index: usize,
    ) -> Result<&ProgramValueMetadata, SemanticAdError> {
        checked_primal_metadata(
            self.op.family_id(),
            PrimalValueKind::Input,
            index,
            &self.primal_input_metadata,
        )
    }

    /// Borrow metadata for one primal output without exposing its value token.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::PrimalIndexOutOfBounds`] when `index` is not
    /// an ordered primal output.
    pub fn primal_output_meta(
        &self,
        index: usize,
    ) -> Result<&ProgramValueMetadata, SemanticAdError> {
        checked_primal_metadata(
            self.op.family_id(),
            PrimalValueKind::Output,
            index,
            &self.primal_output_metadata,
        )
    }

    /// Return the number of ordered primal inputs.
    #[must_use]
    pub const fn primal_input_count(&self) -> usize {
        self.primal_input_metadata.len()
    }

    /// Return the number of ordered primal outputs.
    #[must_use]
    pub const fn primal_output_count(&self) -> usize {
        self.primal_output_metadata.len()
    }

    /// Borrow ordered optional output cotangents.
    pub fn cotangent_outputs(&self) -> &[AdValue] {
        self.cotangent_outputs
    }

    /// Borrow the ordered active-input mask.
    pub fn active_inputs(&self) -> &[bool] {
        self.active_inputs
    }

    /// Return this rule's declared residual mask: which primal input/output
    /// indices may be read as tensor values. Accesses outside the mask must
    /// only use shape/dtype metadata.
    pub const fn residual_mask(&self) -> ResidualSpec {
        self.residual_mask
    }

    /// Return bounded operation provenance.
    pub const fn provenance(&self) -> SemanticProvenanceView<'a> {
        self.provenance
    }
}

/// Definitional JVP rule for one extension family.
pub trait SemanticLinearizeRule: Debug + Send + Sync + 'static {
    /// Return the versioned extension family handled by this rule.
    fn family_id(&self) -> &'static str;

    /// Emit ordered tangent outputs and residuals into `builder`.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::Unsupported`] when the payload is outside
    /// the rule's supported domain, or [`SemanticAdError::Build`] when emitted
    /// semantic operations fail validation.
    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError>;
}

/// Transpose rule for an extension viewed as a linear map.
pub trait SemanticLinearTransposeRule: Debug + Send + Sync + 'static {
    /// Return the versioned extension family handled by this rule.
    fn family_id(&self) -> &'static str;

    /// Declare which primal input/output indices this rule reads as tensor
    /// residuals. Indices not declared may only be accessed through metadata.
    fn residual_mask(&self) -> ResidualSpec;

    /// Emit ordered optional input cotangents into `builder`.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::Unsupported`] when the payload is outside
    /// the rule's supported domain, or [`SemanticAdError::Build`] when emitted
    /// semantic operations fail validation.
    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError>;
}

/// Optional direct VJP rule expressed against primal semantic values.
pub trait SemanticPrimalVjpRule: Debug + Send + Sync + 'static {
    /// Return the versioned extension family handled by this rule.
    fn family_id(&self) -> &'static str;

    /// Declare which primal input/output indices this rule reads as tensor
    /// residuals. Indices not declared may only be accessed through metadata.
    fn residual_mask(&self) -> ResidualSpec;

    /// Emit ordered optional input cotangents into `builder`.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::Unsupported`] when the payload is outside
    /// the rule's supported domain, or [`SemanticAdError::Build`] when emitted
    /// semantic operations fail validation.
    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError>;
}

type LinearizeMap = HashMap<&'static str, Arc<dyn SemanticLinearizeRule>>;
type LinearTransposeMap = HashMap<&'static str, Arc<dyn SemanticLinearTransposeRule>>;
type PrimalVjpMap = HashMap<&'static str, Arc<dyn SemanticPrimalVjpRule>>;

/// Explicit clone-on-write set of semantic extension AD rules.
#[derive(Clone, Default)]
pub struct SemanticExtensionRuleSet {
    linearize: Arc<LinearizeMap>,
    linear_transpose: Arc<LinearTransposeMap>,
    primal_vjp: Arc<PrimalVjpMap>,
}

impl Debug for SemanticExtensionRuleSet {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut linearize: Vec<_> = self.linearize.keys().copied().collect();
        let mut linear_transpose: Vec<_> = self.linear_transpose.keys().copied().collect();
        let mut primal_vjp: Vec<_> = self.primal_vjp.keys().copied().collect();
        linearize.sort_unstable();
        linear_transpose.sort_unstable();
        primal_vjp.sort_unstable();
        formatter
            .debug_struct("SemanticExtensionRuleSet")
            .field("linearize", &linearize)
            .field("linear_transpose", &linear_transpose)
            .field("primal_vjp", &primal_vjp)
            .finish()
    }
}

impl SemanticExtensionRuleSet {
    /// Construct an empty rule set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one semantic linearize rule.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] for an
    /// invalid family or [`SemanticExtensionRegistryError::DuplicateRule`] for
    /// an existing family in this role.
    pub fn register_linearize(
        &mut self,
        rule: Arc<dyn SemanticLinearizeRule>,
    ) -> Result<(), SemanticExtensionRegistryError> {
        validate_insert(
            &self.linearize,
            rule.family_id(),
            SemanticAdRuleRole::Linearize,
        )?;
        Arc::make_mut(&mut self.linearize).insert(rule.family_id(), rule);
        Ok(())
    }

    /// Register one semantic linear-transpose rule.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] for an
    /// invalid family or [`SemanticExtensionRegistryError::DuplicateRule`] for
    /// an existing family in this role.
    pub fn register_linear_transpose(
        &mut self,
        rule: Arc<dyn SemanticLinearTransposeRule>,
    ) -> Result<(), SemanticExtensionRegistryError> {
        validate_insert(
            &self.linear_transpose,
            rule.family_id(),
            SemanticAdRuleRole::LinearTranspose,
        )?;
        Arc::make_mut(&mut self.linear_transpose).insert(rule.family_id(), rule);
        Ok(())
    }

    /// Register one direct semantic primal-VJP rule.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] for an
    /// invalid family or [`SemanticExtensionRegistryError::DuplicateRule`] for
    /// an existing family in this role.
    pub fn register_primal_vjp(
        &mut self,
        rule: Arc<dyn SemanticPrimalVjpRule>,
    ) -> Result<(), SemanticExtensionRegistryError> {
        validate_insert(
            &self.primal_vjp,
            rule.family_id(),
            SemanticAdRuleRole::PrimalVjp,
        )?;
        Arc::make_mut(&mut self.primal_vjp).insert(rule.family_id(), rule);
        Ok(())
    }

    /// Return a rule set containing one semantic linearize rule.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] for an
    /// invalid family or [`SemanticExtensionRegistryError::DuplicateRule`] for
    /// an existing linearize rule.
    pub fn with_linearize(
        mut self,
        rule: Arc<dyn SemanticLinearizeRule>,
    ) -> Result<Self, SemanticExtensionRegistryError> {
        self.register_linearize(rule)?;
        Ok(self)
    }

    /// Return a rule set containing one semantic linear-transpose rule.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] for an
    /// invalid family or [`SemanticExtensionRegistryError::DuplicateRule`] for
    /// an existing linear-transpose rule.
    pub fn with_linear_transpose(
        mut self,
        rule: Arc<dyn SemanticLinearTransposeRule>,
    ) -> Result<Self, SemanticExtensionRegistryError> {
        self.register_linear_transpose(rule)?;
        Ok(self)
    }

    /// Return a rule set containing one direct semantic primal-VJP rule.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] for an
    /// invalid family or [`SemanticExtensionRegistryError::DuplicateRule`] for
    /// an existing primal-VJP rule.
    pub fn with_primal_vjp(
        mut self,
        rule: Arc<dyn SemanticPrimalVjpRule>,
    ) -> Result<Self, SemanticExtensionRegistryError> {
        self.register_primal_vjp(rule)?;
        Ok(self)
    }

    /// Merge another rule set atomically.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] for an
    /// invalid family or [`SemanticExtensionRegistryError::DuplicateRule`] for
    /// a role-equivalent duplicate. The receiver is unchanged on failure.
    pub fn merge(&mut self, other: Self) -> Result<(), SemanticExtensionRegistryError> {
        let mut candidate = self.clone();
        for rule in other.linearize.values() {
            candidate.register_linearize(Arc::clone(rule))?;
        }
        for rule in other.linear_transpose.values() {
            candidate.register_linear_transpose(Arc::clone(rule))?;
        }
        for rule in other.primal_vjp.values() {
            candidate.register_primal_vjp(Arc::clone(rule))?;
        }
        *self = candidate;
        Ok(())
    }

    /// Look up a semantic linearize rule by extension family.
    #[must_use]
    pub fn lookup_linearize(&self, family_id: &str) -> Option<Arc<dyn SemanticLinearizeRule>> {
        self.linearize.get(family_id).cloned()
    }

    /// Look up a semantic linear-transpose rule by extension family.
    #[must_use]
    pub fn lookup_linear_transpose(
        &self,
        family_id: &str,
    ) -> Option<Arc<dyn SemanticLinearTransposeRule>> {
        self.linear_transpose.get(family_id).cloned()
    }

    /// Look up a direct semantic primal-VJP rule by extension family.
    #[must_use]
    pub fn lookup_primal_vjp(&self, family_id: &str) -> Option<Arc<dyn SemanticPrimalVjpRule>> {
        self.primal_vjp.get(family_id).cloned()
    }

    /// Validate and dispatch one semantic extension linearization.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::CoreOperation`] for a core operation,
    /// [`SemanticAdError::EffectfulExtension`] before rule dispatch for an
    /// effectful extension, or typed rule/arity/ownership/build failures.
    #[allow(clippy::too_many_arguments)]
    pub fn linearize_operation(
        &self,
        operation: SemanticOperationView<'_>,
        primal_inputs: &[ProgramValue],
        primal_outputs: &[ProgramValue],
        tangent_inputs: &[AdValue],
        active_outputs: &[bool],
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        let op = extension_for_dispatch(operation)?;
        validate_operation_inputs(operation, primal_inputs, primal_outputs, builder)?;
        validate_len("tangent_inputs", op.input_count(), tangent_inputs.len())?;
        validate_len("active_outputs", op.output_count(), active_outputs.len())?;
        validate_ad_values("tangent_inputs", tangent_inputs, builder)?;
        let rule = self
            .lookup_linearize(op.family_id())
            .ok_or(SemanticAdError::MissingRule {
                family_id: op.family_id(),
                role: SemanticAdRuleRole::Linearize,
            })?;
        let result = rule.linearize(
            SemanticLinearizeRequest {
                op,
                primal_inputs,
                primal_outputs,
                tangent_inputs,
                active_outputs,
                provenance: operation.provenance(),
            },
            builder,
        )?;
        validate_len(
            "tangent_outputs",
            op.output_count(),
            result.tangent_outputs.len(),
        )?;
        validate_ad_values("tangent_outputs", &result.tangent_outputs, builder)?;
        validate_values("residuals", &result.residuals, builder)?;
        Ok(result)
    }

    /// Validate and dispatch one semantic extension linear transpose.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::EffectfulExtension`] before rule dispatch,
    /// [`SemanticAdError::MissingRule`] when no transpose rule exists,
    /// [`SemanticAdError::Arity`] / [`SemanticAdError::ForeignValue`] for an
    /// invalid request or result, or a typed family-rule failure.
    #[allow(clippy::too_many_arguments)]
    pub fn linear_transpose_operation(
        &self,
        operation: SemanticOperationView<'_>,
        primal_inputs: &[ProgramValue],
        primal_outputs: &[ProgramValue],
        cotangent_outputs: &[AdValue],
        active_inputs: &[bool],
        residuals: &[ProgramValue],
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let op = extension_for_dispatch(operation)?;
        validate_operation_inputs(operation, primal_inputs, primal_outputs, builder)?;
        validate_len(
            "cotangent_outputs",
            op.output_count(),
            cotangent_outputs.len(),
        )?;
        validate_len("active_inputs", op.input_count(), active_inputs.len())?;
        validate_ad_values("cotangent_outputs", cotangent_outputs, builder)?;
        validate_values("residuals", residuals, builder)?;
        let primal_input_metadata = snapshot_metadata(primal_inputs, builder)?;
        let primal_output_metadata = snapshot_metadata(primal_outputs, builder)?;
        let rule =
            self.lookup_linear_transpose(op.family_id())
                .ok_or(SemanticAdError::MissingRule {
                    family_id: op.family_id(),
                    role: SemanticAdRuleRole::LinearTranspose,
                })?;
        let result = rule.linear_transpose(
            SemanticLinearTransposeRequest {
                op,
                primal_inputs,
                primal_outputs,
                primal_input_metadata,
                primal_output_metadata,
                cotangent_outputs,
                active_inputs,
                residuals,
                residual_mask: rule.residual_mask(),
                provenance: operation.provenance(),
            },
            builder,
        )?;
        validate_len("cotangent_inputs", op.input_count(), result.len())?;
        validate_ad_values("cotangent_inputs", &result, builder)?;
        Ok(result)
    }

    /// Validate and dispatch one direct semantic primal VJP.
    ///
    /// # Errors
    ///
    /// Returns [`SemanticAdError::EffectfulExtension`] before rule dispatch,
    /// [`SemanticAdError::MissingRule`] when no primal-VJP rule exists,
    /// [`SemanticAdError::Arity`] / [`SemanticAdError::ForeignValue`] for an
    /// invalid request or result, or a typed family-rule failure.
    #[allow(clippy::too_many_arguments)]
    pub fn primal_vjp_operation(
        &self,
        operation: SemanticOperationView<'_>,
        primal_inputs: &[ProgramValue],
        primal_outputs: &[ProgramValue],
        cotangent_outputs: &[AdValue],
        active_inputs: &[bool],
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let op = extension_for_dispatch(operation)?;
        validate_operation_inputs(operation, primal_inputs, primal_outputs, builder)?;
        validate_len(
            "cotangent_outputs",
            op.output_count(),
            cotangent_outputs.len(),
        )?;
        validate_len("active_inputs", op.input_count(), active_inputs.len())?;
        validate_ad_values("cotangent_outputs", cotangent_outputs, builder)?;
        let primal_input_metadata = snapshot_metadata(primal_inputs, builder)?;
        let primal_output_metadata = snapshot_metadata(primal_outputs, builder)?;
        let rule = self
            .lookup_primal_vjp(op.family_id())
            .ok_or(SemanticAdError::MissingRule {
                family_id: op.family_id(),
                role: SemanticAdRuleRole::PrimalVjp,
            })?;
        let result = rule.primal_vjp(
            SemanticPrimalVjpRequest {
                op,
                primal_inputs,
                primal_outputs,
                primal_input_metadata,
                primal_output_metadata,
                cotangent_outputs,
                active_inputs,
                residual_mask: rule.residual_mask(),
                provenance: operation.provenance(),
            },
            builder,
        )?;
        validate_len("cotangent_inputs", op.input_count(), result.len())?;
        validate_ad_values("cotangent_inputs", &result, builder)?;
        Ok(result)
    }
}

fn extension_for_dispatch(
    operation: SemanticOperationView<'_>,
) -> Result<&dyn ExtensionOp, SemanticAdError> {
    let SemanticOpRef::Extension(op) = operation.op() else {
        return Err(SemanticAdError::CoreOperation);
    };
    if !operation.effects().is_empty() {
        return Err(SemanticAdError::EffectfulExtension {
            family_id: op.family_id(),
        });
    }
    Ok(op)
}

fn validate_operation_inputs(
    operation: SemanticOperationView<'_>,
    primal_inputs: &[ProgramValue],
    primal_outputs: &[ProgramValue],
    builder: &SemanticProgramBuilder,
) -> Result<(), SemanticAdError> {
    validate_len(
        "primal_inputs",
        operation.inputs().len(),
        primal_inputs.len(),
    )?;
    validate_len(
        "primal_outputs",
        operation.outputs().len(),
        primal_outputs.len(),
    )?;
    validate_values("primal_inputs", primal_inputs, builder)?;
    validate_values("primal_outputs", primal_outputs, builder)
}

fn checked_primal_value(
    family_id: &'static str,
    kind: PrimalValueKind,
    index: usize,
    values: &[ProgramValue],
    residual_mask: ResidualSpec,
) -> Result<ProgramValue, SemanticAdError> {
    if index >= values.len() {
        return Err(SemanticAdError::PrimalIndexOutOfBounds {
            family_id,
            kind,
            index,
            len: values.len(),
        });
    }
    let declared = match kind {
        PrimalValueKind::Input => residual_mask.declares_input(index),
        PrimalValueKind::Output => residual_mask.declares_output(index),
    };
    if !declared {
        return Err(SemanticAdError::UndeclaredResidualValue {
            family_id,
            kind,
            index,
        });
    }
    Ok(values[index])
}

fn checked_primal_metadata<'a>(
    family_id: &'static str,
    kind: PrimalValueKind,
    index: usize,
    metadata: &'a [ProgramValueMetadata],
) -> Result<&'a ProgramValueMetadata, SemanticAdError> {
    metadata
        .get(index)
        .ok_or(SemanticAdError::PrimalIndexOutOfBounds {
            family_id,
            kind,
            index,
            len: metadata.len(),
        })
}

fn snapshot_metadata(
    values: &[ProgramValue],
    builder: &SemanticProgramBuilder,
) -> Result<Box<[ProgramValueMetadata]>, SemanticAdError> {
    values
        .iter()
        .copied()
        .map(|value| Ok(builder.value_metadata(value)?.clone()))
        .collect()
}

fn validate_values(
    field: &'static str,
    values: &[ProgramValue],
    builder: &SemanticProgramBuilder,
) -> Result<(), SemanticAdError> {
    for (index, value) in values.iter().copied().enumerate() {
        if builder.validate_value(value).is_err() {
            return Err(SemanticAdError::ForeignValue { field, index });
        }
    }
    Ok(())
}

fn validate_ad_values(
    field: &'static str,
    values: &[AdValue],
    builder: &SemanticProgramBuilder,
) -> Result<(), SemanticAdError> {
    for (index, value) in values.iter().copied().enumerate() {
        if let AdValue::Value(value) = value {
            if builder.validate_value(value).is_err() {
                return Err(SemanticAdError::ForeignValue { field, index });
            }
        }
    }
    Ok(())
}

fn validate_len(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), SemanticAdError> {
    if expected != actual {
        return Err(SemanticAdError::Arity {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_insert<T>(
    map: &HashMap<&'static str, T>,
    family_id: &'static str,
    role: SemanticAdRuleRole,
) -> Result<(), SemanticExtensionRegistryError> {
    if !is_valid_family_id(family_id) {
        return Err(SemanticExtensionRegistryError::MalformedFamilyId { family_id });
    }
    if map.contains_key(family_id) {
        return Err(SemanticExtensionRegistryError::DuplicateRule { family_id, role });
    }
    Ok(())
}

fn is_valid_family_id(family_id: &str) -> bool {
    let Some((prefix, version)) = family_id.rsplit_once('.') else {
        return false;
    };
    let Some(version) = version.strip_prefix('v') else {
        return false;
    };
    let Some((crate_name, op_name)) = prefix.split_once('.') else {
        return false;
    };
    !crate_name.is_empty()
        && !op_name.is_empty()
        && !version.is_empty()
        && version.bytes().all(|byte| byte.is_ascii_digit())
        && crate_name.is_ascii()
        && op_name.is_ascii()
        && !crate_name.chars().any(char::is_whitespace)
        && !op_name.chars().any(char::is_whitespace)
}
