//! Shared primitive-rule contracts; no core differentiation dispatcher.

#[cfg(feature = "autodiff")]
use crate::ad::context;
#[cfg(feature = "autodiff")]
use crate::ext_op::ExtensionOp;
#[cfg(feature = "autodiff")]
use crate::std_tensor_op::StdTensorOp;
#[cfg(feature = "autodiff")]
use computegraph::graph::GraphBuilder;
#[cfg(feature = "autodiff")]
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};

#[cfg(feature = "autodiff")]
mod residual_spec;
#[cfg(feature = "autodiff")]
pub use residual_spec::primitive_residual_spec;

#[cfg(feature = "autodiff")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ADRuleKind {
    Jvp,
    Transpose,
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ADRuleError {
    #[error("unsupported AD rule {rule:?} for {op}")]
    Unsupported { op: String, rule: ADRuleKind },
    #[error("invalid AD rule input for {op} ({rule:?}): {message}")]
    InvalidInput {
        op: String,
        rule: ADRuleKind,
        message: String,
    },
}

#[cfg(feature = "autodiff")]
impl ADRuleError {
    pub fn unsupported(op: impl Into<String>, rule: ADRuleKind) -> Self {
        Self::Unsupported {
            op: op.into(),
            rule,
        }
    }

    pub fn invalid_input(
        op: impl Into<String>,
        rule: ADRuleKind,
        message: impl Into<String>,
    ) -> Self {
        Self::InvalidInput {
            op: op.into(),
            rule,
            message: message.into(),
        }
    }

    pub const fn rule(&self) -> ADRuleKind {
        match self {
            Self::Unsupported { rule, .. } | Self::InvalidInput { rule, .. } => *rule,
        }
    }
}

#[cfg(feature = "autodiff")]
pub type ADRuleResult<T> = std::result::Result<T, ADRuleError>;

/// Per-rule declaration of which primal inputs/outputs a transpose (VJP) rule
/// reads as full tensor residuals versus which need only shape/dtype metadata.
///
/// The AD engine uses this to bound residual retention: an index not declared
/// here may only be accessed through its metadata, never as a tensor operand.
/// Indices are counted in primal-input order (input mask) and primal-output
/// order (output mask).
///
/// # Examples
///
/// ```
/// use tenferro_ops::ad::ResidualSpec;
///
/// // `add` needs no tensor residuals; `mul` keeps both operands.
/// let add = ResidualSpec::none();
/// let mul = ResidualSpec::input(0).with_input(1);
/// assert!(add.is_empty());
/// assert!(mul.declares_input(0) && mul.declares_input(1));
/// // Unary ops that reuse the forward output (e.g. exp) declare it.
/// let exp = ResidualSpec::output(0);
/// assert!(exp.declares_output(0));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidualSpec {
    input_mask: u64,
    output_mask: u64,
}

impl ResidualSpec {
    /// A mask declaring no tensor residuals (metadata-only rule).
    pub const fn none() -> Self {
        Self {
            input_mask: 0,
            output_mask: 0,
        }
    }

    /// A mask declaring one input index as a tensor residual.
    pub const fn input(index: usize) -> Self {
        Self {
            input_mask: 1 << index,
            output_mask: 0,
        }
    }

    /// A mask declaring one output index as a tensor residual.
    pub const fn output(index: usize) -> Self {
        Self {
            input_mask: 0,
            output_mask: 1 << index,
        }
    }

    /// Add one input index to this mask.
    pub const fn with_input(mut self, index: usize) -> Self {
        self.input_mask |= 1 << index;
        self
    }

    /// Add one output index to this mask.
    pub const fn with_output(mut self, index: usize) -> Self {
        self.output_mask |= 1 << index;
        self
    }

    /// Add every input index to this mask.
    pub const fn with_all_inputs(mut self) -> Self {
        self.input_mask = u64::MAX;
        self
    }

    /// Add every output index to this mask.
    pub const fn with_all_outputs(mut self) -> Self {
        self.output_mask = u64::MAX;
        self
    }

    /// A mask declaring every input index as a tensor residual.
    ///
    /// Used by rules whose required operand set depends on the active-input
    /// configuration (e.g. `mul`, concatenate, einsum), and by multi-op
    /// families whose individual ops collectively read any operand.
    pub const fn all_inputs() -> Self {
        Self {
            input_mask: u64::MAX,
            output_mask: 0,
        }
    }

    /// A mask declaring every output index as a tensor residual.
    pub const fn all_outputs() -> Self {
        Self {
            input_mask: 0,
            output_mask: u64::MAX,
        }
    }

    /// Whether input `index` is declared as a tensor residual.
    pub const fn declares_input(&self, index: usize) -> bool {
        index < 64 && (self.input_mask >> index) & 1 == 1
    }

    /// Whether output `index` is declared as a tensor residual.
    pub const fn declares_output(&self, index: usize) -> bool {
        index < 64 && (self.output_mask >> index) & 1 == 1
    }

    /// Whether this mask declares no tensor residuals.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.input_mask == 0 && self.output_mask == 0
    }
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Debug)]
pub enum PrimitiveTransposeInput<Op: computegraph::GraphOperation> {
    Residual(ValueKey<Op>),
    Linear {
        key: ValueKey<Op>,
        primal: Option<ValueKey<Op>>,
    },
}

#[cfg(feature = "autodiff")]
impl<Op: computegraph::GraphOperation> PrimitiveTransposeInput<Op> {
    pub fn key(&self) -> &ValueKey<Op> {
        match self {
            Self::Residual(key) | Self::Linear { key, .. } => key,
        }
    }
}

/// Builder interface used by tenferro AD rules.
///
/// # Examples
///
/// ```
/// use computegraph::graph::GraphBuilder;
/// use computegraph::{OperationRole, ValueRef};
/// use tenferro_ops::ad::PrimitiveRuleBuilder;
/// use tenferro_ops::input_key::TensorInputKey;
/// use tenferro_ops::std_tensor_op::StdTensorOp;
///
/// let mut builder = GraphBuilder::<StdTensorOp>::new();
/// let x = builder.add_input(TensorInputKey::User { id: 1 });
/// let out = PrimitiveRuleBuilder::add_operation(
///     &mut builder,
///     StdTensorOp::Neg,
///     vec![ValueRef::Local(x)],
///     OperationRole::Primary,
/// );
/// assert_eq!(out.len(), 1);
/// ```
#[cfg(feature = "autodiff")]
pub trait PrimitiveRuleBuilder {
    /// Add one primitive graph operation and return local ids for its outputs.
    fn add_operation(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<ValueRef<StdTensorOp>>,
        role: OperationRole,
    ) -> Vec<LocalValueId>;
}

#[cfg(feature = "autodiff")]
impl PrimitiveRuleBuilder for GraphBuilder<StdTensorOp> {
    fn add_operation(
        &mut self,
        operation: StdTensorOp,
        inputs: Vec<ValueRef<StdTensorOp>>,
        role: OperationRole,
    ) -> Vec<LocalValueId> {
        GraphBuilder::add_operation(self, operation, inputs, role)
    }
}

/// Extension-rule callback contract stored by shape-guard contexts.
///
/// This vocabulary remains shared; core AD transforms are owned by `tenferro-ad`.
#[doc(hidden)]
#[cfg(feature = "autodiff")]
pub trait ExtensionAdDispatcher: std::fmt::Debug + Send + Sync + 'static {
    fn has_primal_vjp(&self, family_id: &str) -> bool;

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut context::ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;

    fn transpose(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[PrimitiveTransposeInput<StdTensorOp>],
        mode: &OperationRole,
        ctx: &mut context::ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}

#[cfg(test)]
mod tests;
