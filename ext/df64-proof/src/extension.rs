//! A minimal extension-owned operation for the external scalar.
//!
//! The operation is an `ExtensionOp` family whose payload, numerical body, and
//! output construction all live in this crate. The runtime reaches it through a
//! registered `ExtensionModule` and prepared execution, and the input travels as
//! a runtime `Tensor` that carries a caller-owned payload.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ad::extension::{apply_eager_with_extension_session, ExtensionOp};
use tenferro_ad::EagerTensor;
use tenferro_cpu::{scalar_fold, CpuBackend};
use tenferro_ops::{ExtensionShapeContext, SymDim};
use tenferro_runtime::{
    EngineId, ErasedExecutionContext, ExecutionContextIdentity, ExtensionCacheKey,
    ExtensionCacheStore, ExtensionEngine, ExtensionModule, ExtensionModuleError, ExtensionModuleId,
    ExtensionModuleRegistrar, ExtensionPlanningConfig, ExtensionPrepareRequest, PrepareCapability,
    PrepareError, PreparedOperation, PreparedOperationBinding, PreparedOperationExecutor,
    PreparedOperationExecutorHandle, PreparedOperationHandle, PreparedOperationPlan,
    SpecializationProjection,
};
use tenferro_tensor::{DType, Tensor, TensorRead, TensorView};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor, Scalar};

use crate::{Df64, Df64Add};

/// Family identifier of the contribution's externally defined operations.
///
/// One family holds both the total sum and its adjoint broadcast, because the runtime
/// keys one planning config per engine and a contribution owns one numerical engine.
pub const DF64_OPS_FAMILY: &str = "tenferro-df64-proof.df64_ops.v1";

/// Canonical identity of the externally defined `Df64` scalar.
///
/// A semantic program's identity must be reproducible across processes, so the
/// contribution that owns a scalar declares its stable name. This is the name a
/// program carrying `Df64` values reports instead of a process-local `TypeId`.
pub const DF64_SCALAR_IDENTITY: &str = "tenferro-df64-proof.df64.v1";

/// Implement the parts of `ExtensionOp` that every payload-free operation shares.
///
/// The operation supplies its arity and its own output metadata; the family identity,
/// the contribution's scalar identity, the empty payload, and the pure, fresh-output
/// declarations are the same for all of them, so they are declared once here.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::ExtensionOp;
/// use tenferro_df64_proof::extension::{Df64Total, DF64_OPS_FAMILY, DF64_SCALAR_IDENTITY};
///
/// assert_eq!(<Df64Total as ExtensionOp>::family_id(&Df64Total), DF64_OPS_FAMILY);
/// assert_eq!(<Df64Total as ExtensionOp>::input_count(&Df64Total), 1);
/// assert_eq!(<Df64Total as ExtensionOp>::output_count(&Df64Total), 1);
/// assert_eq!(
///     <Df64Total as ExtensionOp>::scalar_identity(&Df64Total),
///     Some(DF64_SCALAR_IDENTITY)
/// );
/// ```
macro_rules! df64_operation {
    ($operation:ty, inputs = $inputs:expr, outputs = $outputs:expr, infer = |$ctx:ident| $infer:block) => {
        impl ExtensionOp for $operation {
            fn family_id(&self) -> &'static str {
                DF64_OPS_FAMILY
            }

            fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

            fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
                other.as_any().downcast_ref::<Self>().is_some()
            }

            fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
                Arc::new(self.clone())
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn input_count(&self) -> usize {
                $inputs
            }

            fn output_count(&self) -> usize {
                $outputs
            }

            fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
                tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
            }

            fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
                tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
            }

            fn scalar_identity(&self) -> Option<&'static str> {
                Some(DF64_SCALAR_IDENTITY)
            }

            fn infer_output_meta(
                &self,
                $ctx: &mut ExtensionShapeContext<'_>,
            ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
                $infer
            }
        }
    };
}

/// Family identifier of the extension-owned scalar broadcast.
/// Total sum of an externally defined scalar tensor.
///
/// The payload carries no parameters, so every instance is equal to every other.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::Df64Total;
/// use tenferro_ad::extension::ExtensionOp;
///
/// assert_eq!(<Df64Total as ExtensionOp>::input_count(&Df64Total), 1);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Df64Total;

df64_operation!(
    Df64Total,
    inputs = 1,
    outputs = 1,
    infer = |ctx| {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            // The body is only defined for the external scalar, so anything else
            // fails explicitly instead of being coerced.
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_total",
                dtype,
                "df64_total takes an externally defined scalar",
            ));
        }
        // A total sum has rank zero.
        Ok(vec![(dtype, Vec::new())])
    }
);

/// Broadcast a scalar external value to a declared shape.
///
/// The total sum's adjoint needs to place the output cotangent back into the input's
/// shape, and a preset broadcast is not available for a scalar tenferro does not
/// declare, so the contribution owns this operation too.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::Df64Expand;
/// use tenferro_ad::extension::ExtensionOp;
///
/// let expand = Df64Expand::new(vec![2, 3]);
/// assert_eq!(<Df64Expand as ExtensionOp>::family_id(&expand), "tenferro-df64-proof.df64_ops.v1");
/// assert_eq!(&*expand.shape, &[2, 3]);
/// ```
#[derive(Clone, Debug)]
pub struct Df64Expand {
    /// Shape the scalar is broadcast to.
    pub shape: Box<[usize]>,
}

impl Df64Expand {
    /// Construct the operation for one output shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64Expand;
    ///
    /// let expand = Df64Expand::new(vec![2, 3]);
    /// assert_eq!(&*expand.shape, &[2, 3]);
    /// ```
    #[must_use]
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape: shape.into_boxed_slice(),
        }
    }
}

impl ExtensionOp for Df64Expand {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        for extent in self.shape.iter() {
            hasher.write_usize(*extent);
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.shape == self.shape)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn scalar_identity(&self) -> Option<&'static str> {
        Some(DF64_SCALAR_IDENTITY)
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_expand",
                dtype,
                "df64_expand takes an externally defined scalar",
            ));
        }
        Ok(vec![(
            dtype,
            self.shape
                .iter()
                .map(|extent| SymDim::from(*extent))
                .collect(),
        )])
    }
}

/// Widen a preset `f64` tensor into the externally defined scalar.
///
/// The widening is exact, so the result's low component is zero. A conversion is a
/// separate operation from the factorization, which is what lets a connected program
/// start from ordinary `f64` values.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::Df64FromF64;
/// use tenferro_ad::extension::ExtensionOp;
///
/// assert_eq!(<Df64FromF64 as ExtensionOp>::input_count(&Df64FromF64), 1);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Df64FromF64;

df64_operation!(
    Df64FromF64,
    inputs = 1,
    outputs = 1,
    infer = |ctx| {
        let dtype = ctx.input_dtype(0)?;
        if dtype != DType::F64 {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_from_f64",
                dtype,
                "df64_from_f64 takes a preset f64 tensor",
            ));
        }
        Ok(vec![(
            DType::External(DF64_SCALAR),
            ctx.input_shape(0)?.to_vec(),
        )])
    }
);

/// Narrow the externally defined scalar into a preset `f64` tensor.
///
/// The low component participates and the result is rounded to nearest with ties to
/// even, so this is not the truncation to the high component that reinterpreting the
/// payload would give. Information the narrowing discards is not recovered later.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::Df64ToF64;
/// use tenferro_ad::extension::ExtensionOp;
///
/// assert_eq!(<Df64ToF64 as ExtensionOp>::output_count(&Df64ToF64), 1);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Df64ToF64;

df64_operation!(
    Df64ToF64,
    inputs = 1,
    outputs = 1,
    infer = |ctx| {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_to_f64",
                dtype,
                "df64_to_f64 takes an externally defined scalar",
            ));
        }
        Ok(vec![(DType::F64, ctx.input_shape(0)?.to_vec())])
    }
);

/// Reverse-mode adjoint of the reduced QR factorization.
///
/// The adjoint is a numerical body of its own: it needs a triangular solve against the
/// primal factor, so it is an operation rather than a graph of preset operations, which
/// a scalar tenferro does not declare could not execute anyway.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::Df64QrVjp;
/// use tenferro_ad::extension::ExtensionOp;
///
/// // A loss that depends on the triangular factor alone supplies one cotangent.
/// let adjoint = Df64QrVjp::of(false, true);
/// assert_eq!(<Df64QrVjp as ExtensionOp>::input_count(&adjoint), 3);
/// assert_eq!(<Df64QrVjp as ExtensionOp>::output_count(&adjoint), 1);
/// assert_eq!(
///     <Df64QrVjp as ExtensionOp>::input_count(&Df64QrVjp::of(true, true)),
///     4
/// );
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Df64QrVjp {
    /// Whether the caller supplied the factor's cotangent.
    pub has_q: bool,
    /// Whether the caller supplied the triangular factor's cotangent.
    pub has_r: bool,
}

impl Df64QrVjp {
    /// Construct the adjoint for one cotangent availability.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64QrVjp;
    ///
    /// // Only the triangular factor carries a cotangent here.
    /// let adjoint = Df64QrVjp::of(false, true);
    /// assert!(!adjoint.has_q);
    /// assert!(adjoint.has_r);
    /// ```
    #[must_use]
    pub const fn of(has_q: bool, has_r: bool) -> Self {
        Self { has_q, has_r }
    }
}

impl ExtensionOp for Df64QrVjp {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(u8::from(self.has_q) | (u8::from(self.has_r) << 1));
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.has_q == self.has_q && other.has_r == self.has_r)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        // The primal factors, plus one cotangent per available output.
        2 + usize::from(self.has_q) + usize::from(self.has_r)
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn scalar_identity(&self) -> Option<&'static str> {
        Some(DF64_SCALAR_IDENTITY)
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_qr_vjp",
                dtype,
                "df64_qr_vjp takes an externally defined scalar",
            ));
        }
        Ok(vec![(dtype, ctx.input_shape(0)?.to_vec())])
    }
}

/// A matrix contraction in the external scalar, written in ordinary einsum notation.
///
/// #1793's example is `einsum("ik,kj->ij", A, B)` evaluated in the external scalar, where the
/// contraction of `[1, 1]` with `[1, 2^-80]` has to keep the low component an `f64` accumulator
/// would drop. The operation accepts exactly that pattern: two rank-2 inputs that share one
/// contracted label, and an output of the two free labels. Any other pattern is refused with a
/// typed error rather than approximated, because the general label cases need the diagonal,
/// reduction, and permutation stages the ordinary lowering plans and this body does not execute.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::ExtensionOp;
/// use tenferro_df64_proof::extension::Df64Einsum;
///
/// let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a matrix contraction");
/// assert_eq!(<Df64Einsum as ExtensionOp>::input_count(&op), 2);
/// assert_eq!(<Df64Einsum as ExtensionOp>::output_count(&op), 1);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Df64Einsum {
    inputs: Vec<Vec<u32>>,
    out: Vec<u32>,
}

impl Df64Einsum {
    /// Build the matrix-contraction pattern `lhs,rhs->out`.
    ///
    /// The labels are the ones an ordinary einsum subscript string names, in order, so
    /// `"ik,kj->ij"` is `(&[0, 1], &[1, 2], &[0, 2])`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::InvalidArgument`] when either input is not rank two, when
    /// a label repeats within one input, when the inputs do not share exactly one contracted label
    /// as the second and first label respectively, or when the output is not the two free labels in
    /// that order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64Einsum;
    ///
    /// assert!(Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).is_ok());
    /// // A repeated label inside one input is a trace, which the body evaluates.
    /// assert!(Df64Einsum::new(&[0, 0], &[0, 2], &[0, 2]).is_ok());
    /// ```
    pub fn new(lhs: &[u32], rhs: &[u32], out: &[u32]) -> tenferro_runtime::Result<Self> {
        Self::new_nary(&[lhs, rhs], out)
    }

    /// Build the pattern for any number of operands.
    ///
    /// A label that two operands share and the output omits is contracted; a label the output omits
    /// is summed; a label that repeats inside one operand is a trace or a diagonal extraction. The
    /// operands are contracted from the left in the order given, and an intermediate keeps exactly
    /// the labels the remaining operands or the output still need.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than two operands are given, when one carries no label, or when
    /// an output label appears in no operand.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64Einsum;
    ///
    /// assert!(Df64Einsum::new_nary(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]).is_ok());
    /// assert!(Df64Einsum::new_nary(&[&[0, 1]], &[0]).is_err());
    /// ```
    pub fn new_nary(inputs: &[&[u32]], out: &[u32]) -> tenferro_runtime::Result<Self> {
        let invalid = |message: &str| {
            tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
                "df64_einsum",
                "pattern",
                message,
            ))
        };
        if inputs.len() < 2 {
            return Err(invalid("a contraction takes at least two operands"));
        }
        if inputs.iter().any(|labels| labels.is_empty()) {
            return Err(invalid("an operand must carry at least one label"));
        }
        for label in out {
            if !inputs.iter().any(|labels| labels.contains(label)) {
                return Err(invalid(
                    "an output label must appear in at least one operand",
                ));
            }
        }
        let mut seen = out.to_vec();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != out.len() {
            return Err(invalid("an output label repeats"));
        }
        Ok(Self {
            inputs: inputs.iter().map(|labels| labels.to_vec()).collect(),
            out: out.to_vec(),
        })
    }

    /// The pattern's label lists, when it has exactly two operands.
    ///
    /// The adjoint and tangent helpers are defined for the pairwise case, so they ask for this and
    /// refuse anything wider rather than guessing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64Einsum;
    ///
    /// let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    /// assert!(op.labels().is_some());
    /// let wide = Df64Einsum::new_nary(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]).expect("a contraction");
    /// assert!(wide.labels().is_none());
    /// ```
    #[must_use]
    pub fn labels(&self) -> Option<(&[u32], &[u32], &[u32])> {
        match self.inputs.as_slice() {
            [lhs, rhs] => Some((lhs, rhs, &self.out)),
            _ => None,
        }
    }

    /// Every operand's labels, in operand order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64Einsum;
    ///
    /// let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    /// assert_eq!(op.input_labels(), &[vec![0, 1], vec![1, 2]]);
    /// ```
    #[must_use]
    pub fn input_labels(&self) -> &[Vec<u32>] {
        &self.inputs
    }

    /// The output's labels, in the output's axis order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64Einsum;
    ///
    /// let op = Df64Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    /// assert_eq!(op.out_labels(), &[0, 2]);
    /// ```
    #[must_use]
    pub fn out_labels(&self) -> &[u32] {
        &self.out
    }
}

impl ExtensionOp for Df64Einsum {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.inputs.len());
        for labels in self.inputs.iter().chain(core::iter::once(&self.out)) {
            hasher.write_usize(labels.len());
            for label in labels {
                hasher.write_u32(*label);
            }
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other == self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        self.inputs.len()
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn scalar_identity(&self) -> Option<&'static str> {
        Some(DF64_SCALAR_IDENTITY)
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_einsum",
                dtype,
                "df64_einsum takes an externally defined scalar",
            ));
        }
        // The output's extent for a label is the extent the first operand that names it declares.
        // Whether operands agree on a shared label is a value-level question, so the body checks it
        // at execution rather than the metadata layer guessing.
        let mut out_shape = Vec::with_capacity(self.out.len());
        for label in &self.out {
            let mut extent = None;
            for (operand, labels) in self.inputs.iter().enumerate() {
                if ctx.input_dtype(operand)? != dtype {
                    return Err(tenferro_tensor::Error::invalid_argument(
                        "df64_einsum",
                        "inputs",
                        "every operand must carry the same scalar",
                    ));
                }
                if let Some(axis) = labels.iter().position(|candidate| candidate == label) {
                    let shape = ctx.input_shape(operand)?;
                    if shape.len() != labels.len() {
                        return Err(tenferro_tensor::Error::rank_mismatch(
                            "df64_einsum",
                            labels.len(),
                            shape.len(),
                        ));
                    }
                    extent = Some(shape[axis].clone());
                    break;
                }
            }
            out_shape.push(extent.ok_or_else(|| {
                tenferro_tensor::Error::invalid_argument(
                    "df64_einsum",
                    "pattern",
                    "an output label must appear in at least one operand",
                )
            })?);
        }
        Ok(vec![(dtype, out_shape)])
    }
}

/// The adjoint of a two-input contraction.
///
/// The adjoint of `out = einsum(lhs, rhs)` contracts the output cotangent with the other operand,
/// which is the same operation with the labels rotated: `lhs_bar = einsum(out, rhs -> lhs)` and
/// `rhs_bar = einsum(lhs, out -> rhs)`. It carries the pattern so the adjoint uses exactly the
/// labels the primal used, and it has two outputs because the contraction has two inputs.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::ExtensionOp;
/// use tenferro_df64_proof::extension::Df64EinsumVjp;
///
/// let adjoint = Df64EinsumVjp::of(&[&[0, 1], &[1, 2]], &[0, 2]).expect("a contraction");
/// assert_eq!(<Df64EinsumVjp as ExtensionOp>::input_count(&adjoint), 3);
/// assert_eq!(<Df64EinsumVjp as ExtensionOp>::output_count(&adjoint), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Df64EinsumVjp {
    inputs: Vec<Vec<u32>>,
    out: Vec<u32>,
}

impl Df64EinsumVjp {
    /// Build the adjoint for the same labels as the primal contraction.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::InvalidArgument`] when the pattern is not a pairwise contraction: an operand
    /// carries no label, a label repeats inside one operand, the operands share no contracted
    /// label, or an output label appears in no operand.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64EinsumVjp;
    ///
    /// assert!(Df64EinsumVjp::of(&[&[0, 1], &[1, 2]], &[0, 2]).is_ok());
    /// assert!(Df64EinsumVjp::of(&[&[], &[1, 2]], &[0, 2]).is_err());
    /// ```
    pub fn of(inputs: &[&[u32]], out: &[u32]) -> tenferro_runtime::Result<Self> {
        Df64Einsum::new_nary(inputs, out)?;
        Ok(Self {
            inputs: inputs.iter().map(|labels| labels.to_vec()).collect(),
            out: out.to_vec(),
        })
    }

    /// The primal pattern's label lists.
    ///
    /// Every operand's labels, in operand order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64EinsumVjp;
    ///
    /// let adjoint = Df64EinsumVjp::of(&[&[0, 1], &[1, 2]], &[0, 2]).expect("a contraction");
    /// assert_eq!(adjoint.input_labels(), &[vec![0, 1], vec![1, 2]]);
    /// ```
    #[must_use]
    pub fn input_labels(&self) -> &[Vec<u32>] {
        &self.inputs
    }

    /// The output's labels, in the output's axis order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64EinsumVjp;
    ///
    /// let adjoint = Df64EinsumVjp::of(&[&[0, 1], &[1, 2]], &[0, 2]).expect("a contraction");
    /// assert_eq!(adjoint.out_labels(), &[0, 2]);
    /// ```
    #[must_use]
    pub fn out_labels(&self) -> &[u32] {
        &self.out
    }
}

impl ExtensionOp for Df64EinsumVjp {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.inputs.len());
        for labels in self.inputs.iter().chain(core::iter::once(&self.out)) {
            hasher.write_usize(labels.len());
            for label in labels {
                hasher.write_u32(*label);
            }
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other == self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        // Every operand and the output cotangent.
        self.inputs.len() + 1
    }

    fn output_count(&self) -> usize {
        // One cotangent per operand.
        self.inputs.len()
    }

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn scalar_identity(&self) -> Option<&'static str> {
        Some(DF64_SCALAR_IDENTITY)
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_einsum_vjp",
                dtype,
                "df64_einsum_vjp takes an externally defined scalar",
            ));
        }
        let mut shapes = Vec::with_capacity(self.inputs.len());
        for operand in 0..self.inputs.len() {
            shapes.push((dtype, ctx.input_shape(operand)?.to_vec()));
        }
        Ok(shapes)
    }
}

/// The forward tangent of a two-input contraction.
///
/// The tangent of `out = einsum(lhs, rhs)` is `einsum(lhs_dot, rhs) + einsum(lhs, rhs_dot)`, so the
/// helper contracts each tangent with the other operand and adds the two results in the extended
/// scalar. Its tangent availability is a payload field, because a linearization need not have a
/// tangent for both operands, and the rule must not materialise a zero tangent for one that is
/// absent.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::ExtensionOp;
/// use tenferro_df64_proof::extension::Df64EinsumJvp;
///
/// let tangent = Df64EinsumJvp::of(&[&[0, 1], &[1, 2]], &[0, 2], &[true, false]).expect("a contraction");
/// assert_eq!(<Df64EinsumJvp as ExtensionOp>::input_count(&tangent), 3);
/// assert_eq!(<Df64EinsumJvp as ExtensionOp>::output_count(&tangent), 1);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Df64EinsumJvp {
    inputs: Vec<Vec<u32>>,
    out: Vec<u32>,
    tangents: Vec<bool>,
}

impl Df64EinsumJvp {
    /// Build the tangent for a pattern and one tangent availability.
    ///
    /// # Errors
    ///
    /// Returns an error when the operand list is not a valid pattern (an operand carries no label, or
    /// an output label appears in no operand), when the tangent mask does not have one entry per
    /// operand, or when no operand carries a tangent, because then there is nothing to differentiate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64EinsumJvp;
    ///
    /// assert!(Df64EinsumJvp::of(&[&[0, 1], &[1, 2]], &[0, 2], &[true, true]).is_ok());
    /// assert!(Df64EinsumJvp::of(&[&[0, 1], &[1, 2]], &[0, 2], &[false, false]).is_err());
    /// ```
    pub fn of(inputs: &[&[u32]], out: &[u32], tangents: &[bool]) -> tenferro_runtime::Result<Self> {
        Df64Einsum::new_nary(inputs, out)?;
        if tangents.len() != inputs.len() {
            return Err(tenferro_runtime::Error::from(
                tenferro_tensor::Error::invalid_argument(
                    "df64_einsum_jvp",
                    "tangents",
                    "the tangent mask needs one entry per operand",
                ),
            ));
        }
        if !tangents.iter().any(|present| *present) {
            return Err(tenferro_runtime::Error::from(
                tenferro_tensor::Error::invalid_argument(
                    "df64_einsum_jvp",
                    "tangents",
                    "at least one operand must carry a tangent",
                ),
            ));
        }
        Ok(Self {
            inputs: inputs.iter().map(|labels| labels.to_vec()).collect(),
            out: out.to_vec(),
            tangents: tangents.to_vec(),
        })
    }

    /// The tangent's pattern and the availability of each operand's tangent.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64EinsumJvp;
    ///
    /// let tangent = Df64EinsumJvp::of(&[&[0, 1], &[1, 2]], &[0, 2], &[true, true]).expect("a tangent");
    /// assert_eq!(tangent.tangents(), &[true, true]);
    /// ```
    #[must_use]
    pub fn input_labels(&self) -> &[Vec<u32>] {
        &self.inputs
    }

    /// The output's labels, in the output's axis order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64EinsumJvp;
    ///
    /// let tangent = Df64EinsumJvp::of(&[&[0, 1], &[1, 2]], &[0, 2], &[true, false])
    ///     .expect("a tangent");
    /// assert_eq!(tangent.out_labels(), &[0, 2]);
    /// ```
    #[must_use]
    pub fn out_labels(&self) -> &[u32] {
        &self.out
    }

    /// Which operands carry a tangent, in operand order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_df64_proof::extension::Df64EinsumJvp;
    ///
    /// let tangent = Df64EinsumJvp::of(&[&[0, 1], &[1, 2]], &[0, 2], &[true, false])
    ///     .expect("a tangent");
    /// assert_eq!(tangent.tangents(), &[true, false]);
    /// ```
    #[must_use]
    pub fn tangents(&self) -> &[bool] {
        &self.tangents
    }
}

impl ExtensionOp for Df64EinsumJvp {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.inputs.len());
        for labels in self.inputs.iter().chain(core::iter::once(&self.out)) {
            hasher.write_usize(labels.len());
            for label in labels {
                hasher.write_u32(*label);
            }
        }
        hasher.write_usize(self.tangents.len());
        for present in &self.tangents {
            hasher.write_u8(u8::from(*present));
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other == self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        // Every operand, plus one tangent per operand that carries one.
        self.inputs.len() + self.tangents.iter().filter(|present| **present).count()
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn scalar_identity(&self) -> Option<&'static str> {
        Some(DF64_SCALAR_IDENTITY)
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_einsum_jvp",
                dtype,
                "df64_einsum_jvp takes an externally defined scalar",
            ));
        }
        let shapes: Vec<Vec<SymDim>> = (0..self.inputs.len())
            .map(|operand| {
                let shape = ctx.input_shape(operand)?;
                if shape.len() != self.inputs[operand].len() {
                    return Err(tenferro_tensor::Error::rank_mismatch(
                        "df64_einsum_jvp",
                        self.inputs[operand].len(),
                        shape.len(),
                    ));
                }
                Ok(shape.to_vec())
            })
            .collect::<tenferro_tensor::Result<Vec<_>>>()?;
        let mut out_shape = Vec::with_capacity(self.out.len());
        for label in &self.out {
            let mut extent = None;
            for (operand, labels) in self.inputs.iter().enumerate() {
                if let Some(axis) = labels.iter().position(|candidate| candidate == label) {
                    extent = Some(shapes[operand][axis].clone());
                    break;
                }
            }
            out_shape.push(extent.ok_or_else(|| {
                tenferro_tensor::Error::invalid_argument(
                    "df64_einsum_jvp",
                    "pattern",
                    "an output label must appear in at least one operand",
                )
            })?);
        }
        Ok(vec![(dtype, out_shape)])
    }
}

/// Forward-mode tangent of the reduced QR factorization.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::Df64QrJvp;
/// use tenferro_ad::extension::ExtensionOp;
///
/// assert_eq!(<Df64QrJvp as ExtensionOp>::input_count(&Df64QrJvp), 3);
/// assert_eq!(<Df64QrJvp as ExtensionOp>::output_count(&Df64QrJvp), 2);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Df64QrJvp;

df64_operation!(
    Df64QrJvp,
    inputs = 3,
    outputs = 2,
    infer = |ctx| {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_qr_jvp",
                dtype,
                "df64_qr_jvp takes an externally defined scalar",
            ));
        }
        Ok(vec![
            (dtype, ctx.input_shape(0)?.to_vec()),
            (dtype, ctx.input_shape(1)?.to_vec()),
        ])
    }
);

/// The externally defined element type of the contribution's scalar.
///
/// This is what the runtime tag reports, next to the canonical identity.
pub const DF64_SCALAR: std::any::TypeId = std::any::TypeId::of::<Df64>();

/// Reduced QR factorization of a real square or tall matrix.
///
/// The factorization is the contribution's own numerical body: modified
/// Gram-Schmidt with one re-orthogonalization pass, computed in the external scalar
/// so the factors keep its precision. `R` has a positive diagonal and `Q` has
/// orthonormal columns, returned as two externally defined tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::Df64Qr;
/// use tenferro_ad::extension::ExtensionOp;
///
/// assert_eq!(<Df64Qr as ExtensionOp>::input_count(&Df64Qr), 1);
/// assert_eq!(<Df64Qr as ExtensionOp>::output_count(&Df64Qr), 2);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Df64Qr;

df64_operation!(
    Df64Qr,
    inputs = 1,
    outputs = 2,
    infer = |ctx| {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_qr",
                dtype,
                "df64_qr takes an externally defined scalar",
            ));
        }
        // The factors have the input's extents, so the inference forwards them however
        // they are expressed. Whether the matrix is tall enough is a property of the
        // concrete input and is checked when the body runs.
        let shape = ctx.input_shape(0)?.to_vec();
        let [rows, columns] = match shape.as_slice() {
            [rows, columns] => [rows.clone(), columns.clone()],
            _ => {
                return Err(tenferro_tensor::Error::invalid_argument(
                    "df64_qr",
                    "input",
                    "df64_qr takes a rank-2 matrix",
                ));
            }
        };
        Ok(vec![
            (dtype, vec![rows.clone(), columns.clone()]),
            (dtype, vec![columns.clone(), columns]),
        ])
    }
);

/// Reduced QR factorization of a column-major dense matrix in the external scalar.
///
/// # Errors
///
/// Returns a typed error when the input is not a rank-2 externally defined matrix, or
/// when a column is zero and the factorization has no unit vector for it.
fn qr_of(
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let input = sole_input("df64_qr", session, inputs)?;
    let tensor = input.tensor();
    let payload =
        external_payload::<Df64>("df64_qr", tensor).map_err(tenferro_runtime::Error::from)?;
    let shape = tensor.shape();
    let invalid = |message: &'static str| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
            "df64_qr", "input", message,
        ))
    };
    let [rows, columns] = match shape {
        [rows, columns] if *rows >= *columns => [*rows, *columns],
        _ => {
            return Err(invalid(
                "df64_qr takes a rank-2 matrix with at least as many rows as columns",
            ));
        }
    };
    let values = payload.as_slice();
    if values.len() != rows * columns {
        return Err(invalid("df64_qr takes a dense column-major matrix"));
    }

    // Column-major access into the source matrix.
    let source = |row: usize, column: usize| values[row + column * rows];
    let mut q = vec![Df64::zero(); rows * columns];
    let mut r = vec![Df64::zero(); columns * columns];
    for column in 0..columns {
        for row in 0..rows {
            q[row + column * rows] = source(row, column);
        }
        // One re-orthogonalization pass after the first projection, so the columns stay
        // orthogonal to working precision even when the input is close to rank
        // deficient.
        for _pass in 0..2 {
            for previous in 0..column {
                let mut projection = Df64::zero();
                for row in 0..rows {
                    projection = projection + q[row + previous * rows] * q[row + column * rows];
                }
                for row in 0..rows {
                    q[row + column * rows] =
                        q[row + column * rows] - projection * q[row + previous * rows];
                }
                r[previous + column * columns] = r[previous + column * columns] + projection;
            }
        }
        let mut squares = Df64::zero();
        for row in 0..rows {
            squares = squares + q[row + column * rows] * q[row + column * rows];
        }
        let norm = squares.sqrt();
        if norm.hi == 0.0 {
            return Err(invalid("df64_qr takes a matrix with no zero column"));
        }
        // A positive diagonal is part of the factorization's contract, so a negative
        // norm flips the column and the entry together.
        let sign = if norm.hi < 0.0 {
            Df64::from_f64(-1.0)
        } else {
            Df64::from_f64(1.0)
        };
        let scale = sign * Df64::from_f64(1.0).ratio(norm);
        for row in 0..rows {
            q[row + column * rows] = q[row + column * rows] * scale;
        }
        r[column + column * columns] = sign * norm;
    }

    let q = HostTensor::from_vec_col_major(vec![rows, columns], q)
        .map_err(|source| tenferro_tensor::Error::validation("df64_qr", source))
        .map_err(tenferro_runtime::Error::from)?;
    let r = HostTensor::from_vec_col_major(vec![columns, columns], r)
        .map_err(|source| tenferro_tensor::Error::validation("df64_qr", source))
        .map_err(tenferro_runtime::Error::from)?;
    Ok(vec![
        Tensor::external(ErasedHostTensor::new(q)),
        Tensor::external(ErasedHostTensor::new(r)),
    ])
}

/// Narrow the external scalar to `f64` through the contribution's own conversion.
fn to_f64_of(
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let input = sole_input("df64_to_f64", session, inputs)?;
    let tensor = input.tensor();
    crate::conversion::to_f64(tensor)
        .map(|tensor| vec![tensor])
        .map_err(tenferro_runtime::Error::from)
}

/// Widen a preset `f64` tensor into the external scalar.
fn from_f64_of(
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let input = match inputs.first() {
        // A borrowed read of the preset input is gathered by its own layout rather than
        // requiring a session the executor may not have.
        Some(read @ TensorRead::View(_)) if session.is_none() => {
            Input::Materialized(Box::new(owned_f64_read(read)?))
        }
        _ => sole_input("df64_from_f64", session, inputs)?,
    };
    let tensor = input.tensor();
    crate::conversion::to_df64(tensor)
        .map(|tensor| vec![tensor])
        .map_err(tenferro_runtime::Error::from)
}

fn external_payload<'a, T: Scalar>(
    op: &'static str,
    tensor: &'a Tensor,
) -> tenferro_tensor::Result<&'a HostTensor<T>> {
    match tensor.external_payload() {
        Some(payload) => payload.downcast_ref::<T>().ok_or_else(|| {
            tenferro_tensor::Error::unsupported_dtype(
                op,
                tensor.dtype(),
                "the external payload does not hold the expected scalar",
            )
        }),
        None => Err(tenferro_tensor::Error::unsupported_dtype(
            op,
            tensor.dtype(),
            "the operation takes an externally defined payload",
        )),
    }
}

/// Reusable scratch the numerical bodies own between executions.
///
/// The entry lives in the runtime's accounted extension cache, so its retained bytes are
/// reported rather than hidden: this is the contribution's acquisition and return path for a
/// buffer of its own element type, and the cache's statistics are the evidence for it. The
/// buffers are named fields rather than a list so one can be written while another is read,
/// which the adjoint's chain of products needs.
#[derive(Debug, Default)]
struct Scratch {
    /// The transposed factor cotangent.
    q_bar_t: Vec<Df64>,
    /// The transposed triangular cotangent.
    r_bar_t: Vec<Df64>,
    /// The `Q_bar^T Q` product.
    q_bar_t_q: Vec<Df64>,
    /// The `R R_bar^T` product.
    r_r_bar: Vec<Df64>,
    /// The `R R_bar^T - Q_bar^T Q` difference.
    m: Vec<Df64>,
    /// `copyltu(M)`, built in place.
    s: Vec<Df64>,
    /// The `Q S` product.
    product: Vec<Df64>,
    /// The accumulator `Q_bar + Q S`.
    b: Vec<Df64>,
}

impl Scratch {
    /// The cache namespace these buffers live under.
    const CACHE_NAME: &'static str = "scratch";

    /// Every buffer this entry retains.
    fn buffers(&self) -> [&Vec<Df64>; 8] {
        [
            &self.q_bar_t,
            &self.r_bar_t,
            &self.q_bar_t_q,
            &self.r_r_bar,
            &self.m,
            &self.s,
            &self.product,
            &self.b,
        ]
    }

    /// Resize one buffer and hand it back for writing, zero-filled.
    fn slot(buffer: &mut Vec<Df64>, length: usize) -> &mut [Df64] {
        buffer.clear();
        buffer.resize(length, Df64::zero());
        buffer.as_mut_slice()
    }

    /// Bytes this entry retains, which the cache reports.
    fn retained_bytes(&self) -> usize {
        self.buffers()
            .iter()
            .map(|buffer| buffer.capacity() * std::mem::size_of::<Df64>())
            .sum()
    }
}

/// Acquire the scratch for one operation shape, reusing what the cache already holds.
fn acquire_scratch(caches: &mut ExtensionCacheStore, shape: usize) -> Scratch {
    let key = ExtensionCacheKey::new(DF64_OPS_FAMILY, Scratch::CACHE_NAME, shape as u64);
    caches
        .get_mut::<Scratch>(&key)
        .map_or_else(Scratch::default, |scratch| Scratch {
            q_bar_t: std::mem::take(&mut scratch.q_bar_t),
            r_bar_t: std::mem::take(&mut scratch.r_bar_t),
            q_bar_t_q: std::mem::take(&mut scratch.q_bar_t_q),
            r_r_bar: std::mem::take(&mut scratch.r_r_bar),
            m: std::mem::take(&mut scratch.m),
            s: std::mem::take(&mut scratch.s),
            product: std::mem::take(&mut scratch.product),
            b: std::mem::take(&mut scratch.b),
        })
}

/// Return the scratch to the cache with its retained bytes reported.
fn release_scratch(caches: &mut ExtensionCacheStore, shape: usize, scratch: Scratch) {
    let key = ExtensionCacheKey::new(DF64_OPS_FAMILY, Scratch::CACHE_NAME, shape as u64);
    let retained = scratch.retained_bytes();
    caches.put(key, scratch, retained);
}

/// One operation input, borrowed when the runtime already owns it.
enum Input<'a> {
    Borrowed(&'a Tensor),
    // Boxed because a tensor value is much larger than a borrow, and every body only
    // reads it through `tensor`.
    Materialized(Box<Tensor>),
}

impl Input<'_> {
    fn tensor(&self) -> &Tensor {
        match self {
            Self::Borrowed(tensor) => tensor,
            Self::Materialized(tensor) => tensor,
        }
    }
}

/// Materialize a borrowed `f64` read without a session.
///
/// A reverse pass can hand the widening operation a strided view, such as the
/// broadcast of a seeding cotangent, and the widening operation's input is always a
/// preset `f64` tensor, so the elements are gathered by their own layout.
///
/// # Errors
///
/// Returns a typed error when the read is not a host `f64` tensor or its layout names
/// storage outside the borrowed buffer.
fn owned_f64_read(read: &TensorRead<'_>) -> tenferro_runtime::Result<Tensor> {
    let invalid = |message: &'static str| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
            "df64_from_f64",
            "input",
            message,
        ))
    };
    let TensorView::F64(view) = read.clone().tensor_view() else {
        return Err(invalid("df64_from_f64 takes a preset f64 tensor"));
    };
    let storage = view.host_storage().map_err(|source| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::runtime_state_source(
            "df64_from_f64",
            source,
        ))
    })?;
    let shape = view.shape().to_vec();
    let mut values = Vec::with_capacity(storage.len().min(view.n_elements()));
    let mut index = vec![0usize; shape.len()];
    for _ in 0..view.n_elements() {
        let offset = view
            .linear_offset(&index)
            .ok_or_else(|| invalid("the borrowed f64 view is outside its buffer"))?;
        let value = storage
            .get(offset)
            .ok_or_else(|| invalid("the borrowed f64 view is outside its buffer"))?;
        values.push(*value);
        for (position, current) in index.iter_mut().enumerate() {
            *current += 1;
            if *current < shape[position] {
                break;
            }
            *current = 0;
        }
    }
    Tensor::from_vec_col_major(shape, values).map_err(|source| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::runtime_state_source(
            "df64_from_f64",
            source,
        ))
    })
}

/// Resolve every input of an operation.
///
/// A borrowed read is materialized in one pass first, so the session borrow does not
/// have to outlive the resolved inputs.
fn inputs_of<'a>(
    op: &'static str,
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &'a [TensorRead<'a>],
) -> tenferro_runtime::Result<Vec<Input<'a>>> {
    let borrowed = inputs
        .iter()
        .any(|read| matches!(read, TensorRead::View(_)));
    if borrowed && session.is_none() {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                op,
                "input",
                "the operation needs a session to read a borrowed input",
            ),
        ));
    }
    let mut materialized: Vec<Option<Tensor>> = (0..inputs.len()).map(|_| None).collect();
    if let Some(session) = session {
        for (index, read) in inputs.iter().enumerate() {
            if matches!(read, TensorRead::View(_)) {
                materialized[index] = Some(
                    session
                        .to_contiguous_read(read.clone())
                        .map_err(tenferro_runtime::Error::from)?,
                );
            }
        }
    }
    let mut resolved = Vec::with_capacity(inputs.len());
    for (read, owned) in inputs.iter().zip(materialized) {
        resolved.push(match (read, owned) {
            (TensorRead::Tensor(tensor), _) => Input::Borrowed(tensor),
            (TensorRead::View(_), Some(tensor)) => Input::Materialized(Box::new(tensor)),
            (TensorRead::View(_), None) => {
                return Err(tenferro_runtime::Error::from(
                    tenferro_tensor::Error::invalid_argument(
                        op,
                        "input",
                        "the operation needs a session to read a borrowed input",
                    ),
                ));
            }
        });
    }
    Ok(resolved)
}

/// Resolve one input to a borrow or to materialized storage.
fn resolve_input<'a>(
    op: &'static str,
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    read: &TensorRead<'a>,
) -> tenferro_runtime::Result<Input<'a>> {
    match read {
        TensorRead::Tensor(tensor) => Ok(Input::Borrowed(tensor)),
        view @ TensorRead::View(_) => match session {
            Some(session) => Ok(Input::Materialized(Box::new(
                session
                    .to_contiguous_read(view.clone())
                    .map_err(tenferro_runtime::Error::from)?,
            ))),
            None => Err(tenferro_runtime::Error::from(
                tenferro_tensor::Error::invalid_argument(
                    op,
                    "input",
                    "the operation needs a session to read a borrowed input",
                ),
            )),
        },
    }
}

/// Resolve the single input every operation in this crate takes.
///
/// The reverse pass can hand an operation a borrowed view of another value, so a
/// session materializes it into storage the operation owns. Without a session a view
/// is rejected explicitly rather than read through.
fn sole_input<'a>(
    op: &'static str,
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &'a [TensorRead<'a>],
) -> tenferro_runtime::Result<Input<'a>> {
    match inputs.first() {
        Some(read) => resolve_input(op, session, read),
        None => Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(op, "input", "the operation takes one input"),
        )),
    }
}

/// Read one externally defined dense matrix from a tensor.
fn matrix_of<'a>(
    op: &'static str,
    tensor: &'a Tensor,
) -> tenferro_runtime::Result<crate::dense::Matrix<'a>> {
    let invalid = |message: &'static str| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
            op, "input", message,
        ))
    };
    let payload = external_payload::<Df64>(op, tensor).map_err(tenferro_runtime::Error::from)?;
    let [rows, columns] = match tensor.shape() {
        [rows, columns] => [*rows, *columns],
        _ => return Err(invalid("the operation takes a rank-2 matrix")),
    };
    if payload.as_slice().len() != rows * columns {
        return Err(invalid("the operation takes a dense column-major matrix"));
    }
    // The payload is caller-owned and borrowed for the length of the body, so no
    // tensor-sized copy is made on the way in.
    Ok(crate::dense::Matrix::borrowed(rows, payload.as_slice()))
}

/// Wrap one dense matrix as an externally defined tensor.
fn tensor_of(
    op: &'static str,
    matrix: crate::dense::Matrix<'_>,
) -> tenferro_runtime::Result<Tensor> {
    let columns = matrix.columns();
    let host = HostTensor::from_vec_col_major(vec![matrix.rows, columns], matrix.data.into_owned())
        .map_err(|source| {
            tenferro_runtime::Error::from(tenferro_tensor::Error::runtime_state_source(op, source))
        })?;
    Ok(Tensor::external(ErasedHostTensor::new(host)))
}

/// Reverse-mode adjoint of the reduced QR factorization.
///
/// The adjoint of `A = Q R` for full column rank `A` is
/// `A_bar = (Q_bar + Q copyltu(R R_bar^T - Q_bar^T Q)) R^{-T}`, evaluated in the
/// external scalar, so the derivative keeps the factorization's precision.
fn qr_vjp_of(
    mask: (bool, bool),
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    caches: &mut ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let op = "df64_qr_vjp";
    let (has_q, has_r) = mask;
    let resolved = inputs_of(op, session, inputs)?;
    if resolved.len() != 2 + usize::from(has_q) + usize::from(has_r) {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                op,
                "input",
                "the adjoint takes Q, R, and both cotangents",
            ),
        ));
    }
    let q = matrix_of(op, resolved[0].tensor())?;
    let r = matrix_of(op, resolved[1].tensor())?;
    let mut next = 2;
    // An absent cotangent is the zero cotangent, which is what an inactive derivative
    // means.
    let q_bar = if has_q {
        let matrix = matrix_of(op, resolved[next].tensor())?;
        next += 1;
        matrix
    } else {
        crate::dense::zeros(q.rows, q.columns())
    };
    let r_bar = if has_r {
        matrix_of(op, resolved[next].tensor())?
    } else {
        crate::dense::zeros(r.rows, r.columns())
    };

    // Every intermediate comes from the accounted scratch, so an execution after the first
    // allocates only the factors it returns. The buffers are named fields, so one can be
    // written while another is read.
    let rows = q.rows;
    let columns = q.columns();
    let length = rows * columns;
    let square = r.rows * r.columns();
    let mut scratch = acquire_scratch(caches, square);

    crate::dense::transpose_into(Scratch::slot(&mut scratch.q_bar_t, length), &q_bar);
    let q_bar_t = crate::dense::Matrix::borrowed(columns, scratch.q_bar_t.as_slice());
    crate::dense::transpose_into(Scratch::slot(&mut scratch.r_bar_t, square), &r_bar);
    {
        let r_bar_t = crate::dense::Matrix::borrowed(r.columns(), scratch.r_bar_t.as_slice());
        crate::dense::multiply_into(Scratch::slot(&mut scratch.r_r_bar, square), &r, &r_bar_t);
    }
    {
        let r = crate::dense::Matrix::borrowed(r.rows, scratch.r_r_bar.as_slice());
        crate::dense::multiply_into(Scratch::slot(&mut scratch.q_bar_t_q, square), &q_bar_t, &q);
        let q_bar_t_q = crate::dense::Matrix::borrowed(columns, scratch.q_bar_t_q.as_slice());
        // `M = R R_bar^T - Q_bar^T Q`.
        crate::dense::subtract_into(Scratch::slot(&mut scratch.m, square), &r, &q_bar_t_q);
    }
    {
        let m = crate::dense::Matrix::borrowed(r.rows, scratch.m.as_slice());
        // `copyltu(M)` is the lower triangle plus the strict lower triangle transposed.
        crate::dense::lower_triangle_into(Scratch::slot(&mut scratch.s, square), &m);
        // The second step adds to what the first wrote, so it must not clear the buffer.
        // Asking `slot` for it again would zero it and silently drop the lower triangle.
        crate::dense::add_strictly_lower_transposed_into(scratch.s.as_mut_slice(), &m);
    }
    {
        let s = crate::dense::Matrix::borrowed(r.rows, scratch.s.as_slice());
        crate::dense::multiply_into(Scratch::slot(&mut scratch.product, length), &q, &s);
        let product = crate::dense::Matrix::borrowed(rows, scratch.product.as_slice());
        let accumulated = Scratch::slot(&mut scratch.b, length);
        for (index, slot) in accumulated.iter_mut().enumerate() {
            let cotangent = q_bar.data.get(index).copied().unwrap_or_else(Df64::zero);
            *slot = cotangent + product.data[index];
        }
    }
    let b = crate::dense::Matrix::borrowed(rows, scratch.b.as_slice());
    let a_bar = crate::dense::solve_upper_from_the_right(&r, &b).ok_or_else(|| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
            op,
            "input",
            "the adjoint needs an invertible triangular factor",
        ))
    })?;
    release_scratch(caches, square, scratch);
    Ok(vec![tensor_of(op, a_bar)?])
}

/// Forward-mode tangent of the reduced QR factorization.
///
/// With `A = Q R`, the tangent satisfies
/// `R_dot = triu(Q^T A_dot) R` and `Q_dot = (A_dot - Q R_dot) R^{-1}`, so the forward
/// rule emits both tangent outputs from the primal factors.
fn qr_jvp_of(
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let op = "df64_qr_jvp";
    let resolved = inputs_of(op, session, inputs)?;
    if resolved.len() != 3 {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                op,
                "input",
                "the tangent takes Q, R, and the input tangent",
            ),
        ));
    }
    let q = matrix_of(op, resolved[0].tensor())?;
    let r = matrix_of(op, resolved[1].tensor())?;
    let a_dot = matrix_of(op, resolved[2].tensor())?;

    // With A = Q R, the differential identity W = Q^T A_dot = S R + R_dot holds, where
    // S = Q^T Q_dot is skew and R_dot is upper triangular. The strictly lower part of W
    // therefore determines S by forward substitution, and R_dot = W - S R follows. Taking
    // the upper triangle of W directly would drop the S R term, which is zero only for a
    // single column.
    let w = crate::dense::multiply(&crate::dense::transpose(&q), &a_dot);
    let n = r.columns();
    let mut s = crate::dense::zeros(n, n);
    for column in 0..n {
        for row in (column + 1)..n {
            let mut value = w.at(row, column);
            for earlier in 0..column {
                value = value - s.at(row, earlier) * r.at(earlier, column);
            }
            let skew = value / r.at(column, column);
            s.set(row, column, skew);
            s.set(column, row, -skew);
        }
    }
    let r_dot = crate::dense::subtract(&w, &crate::dense::multiply(&s, &r));
    let residual = crate::dense::subtract(&a_dot, &crate::dense::multiply(&q, &r_dot));
    let q_dot = crate::dense::solve_upper_from_the_right(&r, &residual).ok_or_else(|| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
            op,
            "input",
            "the tangent needs an invertible triangular factor",
        ))
    })?;
    Ok(vec![tensor_of(op, q_dot)?, tensor_of(op, r_dot)?])
}

/// The output shape a label list describes, given the extents the operands declare.
fn shape_of_labels(labels: &[Box<[u32]>], shapes: &[Vec<usize>], out_labels: &[u32]) -> Vec<usize> {
    let mut extents: Vec<(u32, usize)> = Vec::new();
    for (operand, operand_labels) in labels.iter().enumerate() {
        for (axis, label) in operand_labels.iter().enumerate() {
            if !extents.iter().any(|(existing, _)| existing == label) {
                extents.push((*label, shapes[operand][axis]));
            }
        }
    }
    out_labels
        .iter()
        .map(|label| {
            extents
                .iter()
                .find(|(existing, _)| existing == label)
                .map(|(_, extent)| *extent)
                .unwrap_or(1)
        })
        .collect()
}

/// Contract two operand payloads over their shared labels, in the external scalar's arithmetic.
///
/// The output index space is walked once and, for each of its points, the contracted index space is
/// summed. A label an operand names but the output does not is summed, which is what ordinary einsum
/// notation means by it, and a label that repeats inside one operand was refused when the operation
/// was built.
fn contract_in_scalar(
    op: &'static str,
    labels: &[Box<[u32]>],
    out_labels: &[u32],
    lhs_values: &[Df64],
    lhs_shape: &[usize],
    rhs_values: &[Df64],
    rhs_shape: &[usize],
) -> tenferro_runtime::Result<Vec<Df64>> {
    if element_count(lhs_shape) != lhs_values.len() || element_count(rhs_shape) != rhs_values.len()
    {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                op,
                "inputs",
                "the payload length does not match the declared shape",
            ),
        ));
    }

    let mut extents: Vec<(u32, usize)> = Vec::new();
    let label_extent = |label: u32, extent: usize, extents: &mut Vec<(u32, usize)>| match extents
        .iter()
        .find(|(existing, _)| *existing == label)
    {
        Some((_, existing)) => *existing == extent,
        None => {
            extents.push((label, extent));
            true
        }
    };
    let mut ok = true;
    for (axis, label) in labels[0].iter().enumerate() {
        ok &= label_extent(*label, lhs_shape[axis], &mut extents);
    }
    for (axis, label) in labels[1].iter().enumerate() {
        ok &= label_extent(*label, rhs_shape[axis], &mut extents);
    }
    if !ok {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                op,
                "inputs",
                "the inputs disagree on the extent of a shared label",
            ),
        ));
    }
    let extent_of = |label: u32, extents: &[(u32, usize)]| {
        extents
            .iter()
            .find(|(existing, _)| *existing == label)
            .map(|(_, extent)| *extent)
            .unwrap_or(1)
    };
    let out_shape: Vec<usize> = out_labels
        .iter()
        .map(|label| extent_of(*label, &extents))
        .collect();
    let mut summed_labels: Vec<u32> = Vec::new();
    for label in labels[0].iter().chain(labels[1].iter()) {
        if !out_labels.contains(label) && !summed_labels.contains(label) {
            summed_labels.push(*label);
        }
    }
    let summed_shape: Vec<usize> = summed_labels
        .iter()
        .map(|label| extent_of(*label, &extents))
        .collect();

    let out_count = element_count(&out_shape);
    let summed_count = element_count(&summed_shape);
    let mut result = vec![Df64::zero(); out_count];
    let mut out_index = vec![0usize; out_shape.len()];
    let mut summed_index = vec![0usize; summed_shape.len()];
    for slot in result.iter_mut() {
        for value in summed_index.iter_mut() {
            *value = 0;
        }
        let mut accumulator = Df64::zero();
        for _ in 0..summed_count {
            let lhs_offset = offset_for(
                &labels[0],
                lhs_shape,
                out_labels,
                &out_index,
                &summed_labels,
                &summed_index,
            );
            let rhs_offset = offset_for(
                &labels[1],
                rhs_shape,
                out_labels,
                &out_index,
                &summed_labels,
                &summed_index,
            );
            accumulator = accumulator + lhs_values[lhs_offset] * rhs_values[rhs_offset];
            advance(&mut summed_index, &summed_shape);
        }
        *slot = accumulator;
        advance(&mut out_index, &out_shape);
    }
    Ok(result)
}

/// Wrap a payload as an external tensor of the shape the labels describe.
fn external_of(
    op: &'static str,
    values: Vec<Df64>,
    shape: Vec<usize>,
) -> tenferro_runtime::Result<Tensor> {
    let tensor = HostTensor::from_vec_col_major(shape, values)
        .map_err(|source| tenferro_tensor::Error::validation(op, source))
        .map_err(tenferro_runtime::Error::from)?;
    Ok(Tensor::external(ErasedHostTensor::new(tensor)))
}

/// Contract every operand, folding from the left.
///
/// An intermediate keeps exactly the labels the remaining operands or the output still need, so a
/// label that only the already-contracted operands name is summed by that step, which is what the
/// notation means by a label the output omits.
/// Fold a list of operands from the left, returning the contracted values and their shape.
///
/// An intermediate keeps exactly the labels the remaining operands or the output still need, so a
/// label only the already-contracted operands name is summed by that step, which is what the notation
/// means by a label the output omits.
fn fold_in_scalar(
    op: &'static str,
    labels: &[Box<[u32]>],
    out_labels: &[u32],
    operands: &[(Vec<Df64>, Vec<usize>)],
) -> tenferro_runtime::Result<(Vec<Df64>, Vec<usize>)> {
    let mut accumulator = operands[0].0.clone();
    let mut accumulator_shape = operands[0].1.clone();
    let mut accumulator_labels: Vec<u32> = labels[0].to_vec();
    for index in 1..labels.len() {
        let next_labels: Vec<u32> = labels[index].to_vec();
        let keep: Vec<u32> = if index + 1 == labels.len() {
            out_labels.to_vec()
        } else {
            let mut keep: Vec<u32> = Vec::new();
            for label in accumulator_labels.iter().chain(next_labels.iter()) {
                let needed_later = out_labels.contains(label)
                    || labels[index + 1..].iter().any(|rest| rest.contains(label));
                if needed_later && !keep.contains(label) {
                    keep.push(*label);
                }
            }
            keep
        };
        let pair = [
            accumulator_labels.clone().into_boxed_slice(),
            next_labels.clone().into_boxed_slice(),
        ];
        let contracted = contract_in_scalar(
            op,
            &pair,
            &keep,
            &accumulator,
            &accumulator_shape,
            &operands[index].0,
            &operands[index].1,
        )?;
        let contracted_shape = shape_of_labels(
            &pair,
            &[accumulator_shape.clone(), operands[index].1.clone()],
            &keep,
        );
        accumulator = contracted;
        accumulator_shape = contracted_shape;
        accumulator_labels = keep;
    }
    Ok((accumulator, accumulator_shape))
}

/// Contract every operand, returning the result as an external tensor.
fn einsum_of(
    labels: &[Box<[u32]>],
    out_labels: &[u32],
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let op = "df64_einsum";
    let resolved = inputs_of(op, session, inputs)?;
    if resolved.len() != labels.len() || labels.len() < 2 {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                op,
                "input",
                "a contraction takes one input per operand",
            ),
        ));
    }
    let mut operands: Vec<(Vec<Df64>, Vec<usize>)> = Vec::with_capacity(resolved.len());
    for operand in &resolved {
        operands.push((
            payload_of::<Df64>(op, operand.tensor())?,
            operand.tensor().shape().to_vec(),
        ));
    }
    let (values, shape) = fold_in_scalar(op, labels, out_labels, &operands)?;
    Ok(vec![external_of(op, values, shape)?])
}

/// The adjoint of a contraction: the cotangent of each operand, in the extended scalar.
///
/// The labels rotate rather than change, so the adjoint contracts the output cotangent with the
/// other operand and lands on the operand it differentiates.
fn einsum_vjp_of(
    labels: &[Box<[u32]>],
    out_labels: &[u32],
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let op = "df64_einsum_vjp";
    let resolved = inputs_of(op, session, inputs)?;
    if resolved.len() != labels.len() + 1 || labels.len() < 2 {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                op,
                "input",
                "the adjoint takes one input per operand and the output cotangent",
            ),
        ));
    }
    let mut operands: Vec<(Vec<Df64>, Vec<usize>)> = Vec::with_capacity(labels.len());
    for operand in &resolved[..labels.len()] {
        operands.push((
            payload_of::<Df64>(op, operand.tensor())?,
            operand.tensor().shape().to_vec(),
        ));
    }
    let cotangent = payload_of::<Df64>(op, resolved[labels.len()].tensor())?;
    let cotangent_shape = resolved[labels.len()].tensor().shape().to_vec();

    // The cotangent of each operand is the contraction of the output cotangent with every other
    // operand, so the cotangent takes that operand's place and carries the output's labels.
    let mut outputs = Vec::with_capacity(labels.len());
    for position in 0..labels.len() {
        let mut substituted: Vec<(Vec<Df64>, Vec<usize>)> = Vec::with_capacity(labels.len());
        for (index, operand) in operands.iter().enumerate() {
            if index == position {
                substituted.push((cotangent.clone(), cotangent_shape.clone()));
            } else {
                substituted.push(operand.clone());
            }
        }
        let mut step_labels: Vec<Box<[u32]>> = labels.to_vec();
        step_labels[position] = out_labels.to_vec().into_boxed_slice();
        let (values, shape) = fold_in_scalar(op, &step_labels, &labels[position], &substituted)?;
        outputs.push(external_of(op, values, shape)?);
    }
    Ok(outputs)
}

/// The forward tangent of a contraction: each operand's tangent contracted with the other.
fn einsum_jvp_of(
    labels: &[Box<[u32]>],
    out_labels: &[u32],
    tangents: &[bool],
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let op = "df64_einsum_jvp";
    let resolved = inputs_of(op, session, inputs)?;
    let expected = labels.len() + tangents.iter().filter(|present| **present).count();
    if resolved.len() != expected || labels.len() != tangents.len() || labels.len() < 2 {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                op,
                "input",
                "the tangent takes one input per operand and one per tangent",
            ),
        ));
    }
    let mut operands: Vec<(Vec<Df64>, Vec<usize>)> = Vec::with_capacity(labels.len());
    let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(labels.len());
    for operand in &resolved[..labels.len()] {
        operands.push((
            payload_of::<Df64>(op, operand.tensor())?,
            operand.tensor().shape().to_vec(),
        ));
        shapes.push(operand.tensor().shape().to_vec());
    }
    let mut dots: Vec<Option<(Vec<Df64>, Vec<usize>)>> = vec![None; labels.len()];
    let mut slot = labels.len();
    for (index, present) in tangents.iter().enumerate() {
        if !present {
            continue;
        }
        dots[index] = Some((
            payload_of::<Df64>(op, resolved[slot].tensor())?,
            resolved[slot].tensor().shape().to_vec(),
        ));
        slot += 1;
    }
    let out_shape = shape_of_labels(labels, &shapes, out_labels);

    // Each tangent takes its operand's place, under that operand's own labels, and the parts sum.
    let mut total: Option<Vec<Df64>> = None;
    for (position, dot) in dots.iter().enumerate() {
        let Some((values, shape)) = dot else {
            continue;
        };
        let mut substituted = operands.clone();
        substituted[position] = (values.clone(), shape.clone());
        let (part, _) = fold_in_scalar(op, labels, out_labels, &substituted)?;
        total = Some(match total {
            Some(existing) => existing
                .iter()
                .zip(&part)
                .map(|(left, right)| *left + *right)
                .collect(),
            None => part,
        });
    }
    let values = total.ok_or_else(|| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
            op,
            "tangents",
            "at least one operand must carry a tangent",
        ))
    })?;
    Ok(vec![external_of(op, values, out_shape)?])
}

/// The column-major offset an input's labels select at one output and contracted index.
fn offset_for(
    input_labels: &[u32],
    input_shape: &[usize],
    out_labels: &[u32],
    out_index: &[usize],
    summed_labels: &[u32],
    summed_index: &[usize],
) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, label) in input_labels.iter().enumerate() {
        let position = out_labels
            .iter()
            .position(|candidate| candidate == label)
            .map(|index| out_index[index])
            .or_else(|| {
                summed_labels
                    .iter()
                    .position(|candidate| candidate == label)
                    .map(|index| summed_index[index])
            })
            .unwrap_or(0);
        offset += position * stride;
        stride *= input_shape[axis];
    }
    offset
}

/// Advance a column-major odometer by one, wrapping the fastest-varying axis first.
fn advance(index: &mut [usize], shape: &[usize]) {
    for axis in 0..shape.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

/// The number of elements a shape describes.
fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// The externally defined payload of a tensor.
fn payload_of<T: tenferro_tensor_core::Scalar>(
    op: &'static str,
    tensor: &Tensor,
) -> tenferro_runtime::Result<Vec<T>> {
    external_payload::<T>(op, tensor)
        .map(|payload| payload.as_slice().to_vec())
        .map_err(tenferro_runtime::Error::from)
}

/// Fill a tensor of `shape` with the scalar input's value.
fn expand_of(
    shape: &[usize],
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let input = sole_input("df64_expand", session, inputs)?;
    let tensor = input.tensor();
    let payload =
        external_payload::<Df64>("df64_expand", tensor).map_err(tenferro_runtime::Error::from)?;
    let value = payload.as_slice().first().copied().ok_or_else(|| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
            "df64_expand",
            "input",
            "df64_expand takes a scalar payload",
        ))
    })?;
    let count: usize = shape.iter().product();
    let output = HostTensor::from_vec_col_major(shape.to_vec(), vec![value; count])
        .map_err(|source| tenferro_tensor::Error::validation("df64_expand", source))
        .map_err(tenferro_runtime::Error::from)?;
    Ok(vec![Tensor::external(ErasedHostTensor::new(output))])
}

fn total_of(
    session: Option<&mut dyn tenferro_tensor::BackendSession>,
    inputs: &[TensorRead<'_>],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let input = sole_input("df64_total", session, inputs)?;
    let tensor = input.tensor();
    let payload =
        external_payload::<Df64>("df64_total", tensor).map_err(tenferro_runtime::Error::from)?;
    let total = scalar_fold::<Df64, Df64Add>("df64_total", payload, Df64::zero())
        .map_err(tenferro_runtime::Error::from)?;
    let output = HostTensor::from_vec_col_major(vec![], vec![total])
        .map_err(|source| tenferro_tensor::Error::validation("df64_total", source))
        .map_err(tenferro_runtime::Error::from)?;
    Ok(vec![Tensor::external(ErasedHostTensor::new(output))])
}

/// Which numerical body a prepared operation runs.
#[derive(Debug)]
enum Df64Body {
    /// Total sum of the input.
    Total,
    /// The scalar input broadcast to this shape.
    Expand(Box<[usize]>),
    /// Reduced QR factorization of the matrix input.
    Qr,
    /// Narrow the external scalar to `f64`.
    ToF64,
    /// Widen a preset `f64` tensor into the external scalar.
    FromF64,
    /// The adjoint of the factorization, with the cotangents the caller supplied.
    QrVjp((bool, bool)),
    /// The tangent of the factorization.
    QrJvp,
    /// The forward tangent of a pairwise contraction.
    EinsumJvp {
        /// The labels of each operand, in that operand's own axis order.
        inputs: Box<[Box<[u32]>]>,
        /// The labels of the contraction's output, in the output's axis order.
        out: Box<[u32]>,
        /// Which operands carry a tangent, in operand order.
        tangents: Box<[bool]>,
    },
    /// The adjoint of a pairwise contraction.
    EinsumVjp {
        /// The labels of each operand, in that operand's own axis order.
        inputs: Box<[Box<[u32]>]>,
        /// The labels of the contraction's output, in the output's axis order.
        out: Box<[u32]>,
    },
    /// A pairwise contraction of two external tensors over their shared labels.
    Einsum {
        /// The labels of each input, in that input's own axis order.
        inputs: Box<[Box<[u32]>]>,
        /// The labels of the output, in the output's axis order.
        out: Box<[u32]>,
    },
}

impl Df64Body {
    fn execute(
        &self,
        session: Option<&mut dyn tenferro_tensor::BackendSession>,
        caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        match self {
            Self::Total => total_of(session, inputs),
            Self::Expand(shape) => expand_of(shape, session, inputs),
            Self::Qr => qr_of(session, inputs),
            Self::ToF64 => to_f64_of(session, inputs),
            Self::FromF64 => from_f64_of(session, inputs),
            Self::QrVjp(mask) => qr_vjp_of(*mask, session, caches, inputs),
            Self::QrJvp => qr_jvp_of(session, inputs),
            Self::Einsum {
                inputs: labels,
                out,
            } => einsum_of(labels, out, session, inputs),
            Self::EinsumVjp {
                inputs: labels,
                out,
            } => einsum_vjp_of(labels, out, session, inputs),
            Self::EinsumJvp {
                inputs: labels,
                out,
                tangents,
            } => einsum_jvp_of(labels, out, tangents, session, inputs),
        }
    }
}

#[derive(Debug)]
struct Df64Prepared {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    body: Df64Body,
}

impl PreparedOperation for Df64Prepared {
    fn binding(&self) -> &PreparedOperationBinding {
        &self.binding
    }

    fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

impl PreparedOperationExecutor for Df64Prepared {
    fn execute(
        &self,
        _context: &mut ErasedExecutionContext<'_>,
        caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        // The erased context does not expose a session, so a body that needs one relies
        // on the session entry point and gathers what it can without one.
        self.body.execute(None, caches, inputs)
    }

    fn supports_session(&self) -> bool {
        true
    }

    fn execute_in_session(
        &self,
        session: &mut dyn tenferro_tensor::BackendSession,
        caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        self.body.execute(Some(session), caches, inputs)
    }
}

#[derive(Debug)]
struct Df64Engine {
    family_id: &'static str,
    engine_id: EngineId,
}

impl ExtensionEngine for Df64Engine {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<CpuBackend>()
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        let operation = request.operation().as_any();
        let body = if let Some(expand) = operation.downcast_ref::<Df64Expand>() {
            Df64Body::Expand(expand.shape.clone())
        } else if operation.downcast_ref::<Df64Qr>().is_some() {
            Df64Body::Qr
        } else if operation.downcast_ref::<Df64ToF64>().is_some() {
            Df64Body::ToF64
        } else if operation.downcast_ref::<Df64FromF64>().is_some() {
            Df64Body::FromF64
        } else if let Some(adjoint) = operation.downcast_ref::<Df64QrVjp>() {
            Df64Body::QrVjp((adjoint.has_q, adjoint.has_r))
        } else if operation.downcast_ref::<Df64QrJvp>().is_some() {
            Df64Body::QrJvp
        } else if let Some(tangent) = operation.downcast_ref::<Df64EinsumJvp>() {
            Df64Body::EinsumJvp {
                inputs: tangent
                    .input_labels()
                    .iter()
                    .map(|labels| labels.clone().into_boxed_slice())
                    .collect(),
                out: tangent.out_labels().to_vec().into_boxed_slice(),
                tangents: tangent.tangents().to_vec().into_boxed_slice(),
            }
        } else if let Some(adjoint) = operation.downcast_ref::<Df64EinsumVjp>() {
            Df64Body::EinsumVjp {
                inputs: adjoint
                    .input_labels()
                    .iter()
                    .map(|labels| labels.clone().into_boxed_slice())
                    .collect(),
                out: adjoint.out_labels().to_vec().into_boxed_slice(),
            }
        } else if let Some(contraction) = operation.downcast_ref::<Df64Einsum>() {
            Df64Body::Einsum {
                inputs: contraction
                    .input_labels()
                    .iter()
                    .map(|labels| labels.clone().into_boxed_slice())
                    .collect(),
                out: contraction.out_labels().to_vec().into_boxed_slice(),
            }
        } else {
            Df64Body::Total
        };
        let prepared = Arc::new(Df64Prepared {
            binding: request.binding().clone(),
            specialization: request.specialization().clone(),
            body,
        });
        let operation: PreparedOperationHandle = Arc::clone(&prepared) as PreparedOperationHandle;
        let executor: PreparedOperationExecutorHandle = prepared as PreparedOperationExecutorHandle;
        Ok(PrepareCapability::Prepared(
            PreparedOperationPlan::executable(operation, executor),
        ))
    }
}

#[derive(Debug)]
struct Df64Config {
    family_id: &'static str,
}

impl ExtensionPlanningConfig for Df64Config {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, state: &mut dyn Hasher) {
        state.write(self.family_id.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self.family_id == other.family_id)
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct Df64TotalModule {
    module_id: ExtensionModuleId,
    engine_id: EngineId,
}

impl ExtensionModule for Df64TotalModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError> {
        registrar.register_engine(Arc::new(Df64Engine {
            family_id: DF64_OPS_FAMILY,
            engine_id: self.engine_id.clone(),
        }))?;
        registrar.register_planning_config(
            self.engine_id.clone(),
            Arc::new(Df64Config {
                family_id: DF64_OPS_FAMILY,
            }),
        )
    }
}

/// The module a downstream application installs to enable [`Df64Total`].
///
/// # Errors
///
/// Returns an error when the CPU runtime engine identifier is unavailable in this
/// process.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::module;
///
/// assert!(module().is_ok());
/// ```
pub fn module() -> Result<Arc<dyn ExtensionModule>, tenferro_runtime::RuntimeConfigError> {
    module_for_engine(tenferro_cpu::runtime_engine_id()?)
}

/// The module a downstream application installs when it composes more than one CPU
/// backend in one runtime.
///
/// The module binds the contribution's operations to `engine_id`, so an application that
/// keeps a standard backend and a contribution backend under distinct engine identities
/// installs this module against the contribution's engine.
///
/// # Errors
///
/// Returns [`tenferro_runtime::RuntimeConfigError`] when the module's configured
/// identifier is invalid.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::module_for_engine;
/// use tenferro_runtime::EngineId;
///
/// let module = module_for_engine(EngineId::new("example.df64.engine.v1")?)?;
/// assert_eq!(module.module_id().as_str(), "tenferro-df64-proof.module");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn module_for_engine(
    engine_id: EngineId,
) -> Result<Arc<dyn ExtensionModule>, tenferro_runtime::RuntimeConfigError> {
    Ok(Arc::new(Df64TotalModule {
        module_id: ExtensionModuleId::new("tenferro-df64-proof.module")?,
        engine_id,
    }))
}

/// Run [`Df64Total`] on an eagerly held external tensor.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeStateSource`] when the contribution's module
/// cannot be registered, or when the operation cannot be prepared or executed for this
/// input.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::ExtensionOp;
/// use tenferro_df64_proof::extension::{module, Df64Total};
///
/// assert_eq!(<Df64Total as ExtensionOp>::input_count(&Df64Total), 1);
/// assert!(module().is_ok());
/// ```
pub fn apply_total(input: &EagerTensor) -> tenferro_runtime::Result<Vec<EagerTensor>> {
    let module = module().map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "df64_total",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    apply_eager_with_extension_session(Arc::new(Df64Total), &[input], module)
}
