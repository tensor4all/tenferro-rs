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
    EngineId, ErasedExecutionContext, ExecutionContextIdentity, ExtensionCacheStore,
    ExtensionEngine, ExtensionModule, ExtensionModuleError, ExtensionModuleId,
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

impl ExtensionOp for Df64Total {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
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

    /// The body reads its input and writes only its own output.
    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    /// The total is a new value rather than a view of the input.
    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    /// The operation carries the contribution's externally defined scalar, so it
    /// declares that scalar's canonical identity for the value metadata.
    fn scalar_identity(&self) -> Option<&'static str> {
        Some(DF64_SCALAR_IDENTITY)
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
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
}

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

impl ExtensionOp for Df64FromF64 {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
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
}

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

impl ExtensionOp for Df64ToF64 {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
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

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
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
}

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

impl ExtensionOp for Df64QrJvp {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        3
    }

    fn output_count(&self) -> usize {
        2
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
}

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

impl ExtensionOp for Df64Qr {
    fn family_id(&self) -> &'static str {
        DF64_OPS_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        2
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
}

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
    match tensor {
        Tensor::External(payload, _) => payload.downcast_ref::<T>().ok_or_else(|| {
            tenferro_tensor::Error::unsupported_dtype(
                op,
                tensor.dtype(),
                "the external payload does not hold the expected scalar",
            )
        }),
        other => Err(tenferro_tensor::Error::unsupported_dtype(
            op,
            other.dtype(),
            "the operation takes an externally defined payload",
        )),
    }
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
fn matrix_of(op: &'static str, tensor: &Tensor) -> tenferro_runtime::Result<crate::dense::Matrix> {
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
    Ok(crate::dense::Matrix::new(rows, payload.as_slice().to_vec()))
}

/// Wrap one dense matrix as an externally defined tensor.
fn tensor_of(op: &'static str, matrix: crate::dense::Matrix) -> tenferro_runtime::Result<Tensor> {
    let columns = matrix.columns();
    let host = HostTensor::from_vec_col_major(vec![matrix.rows, columns], matrix.data).map_err(
        |source| {
            tenferro_runtime::Error::from(tenferro_tensor::Error::runtime_state_source(op, source))
        },
    )?;
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

    let q_bar_transposed = crate::dense::transpose(&q_bar);
    let m = crate::dense::subtract(
        &crate::dense::multiply(&r, &crate::dense::transpose(&r_bar)),
        &crate::dense::multiply(&q_bar_transposed, &q),
    );
    // copyltu(M) is the lower triangle plus the strict lower triangle transposed.
    let s = crate::dense::add(
        &crate::dense::lower_triangle(&m),
        &crate::dense::transpose(&crate::dense::strictly_lower_triangle(&m)),
    );
    let b = crate::dense::add(&q_bar, &crate::dense::multiply(&q, &s));
    let a_bar = crate::dense::solve_upper_from_the_right(&r, &b).ok_or_else(|| {
        tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
            op,
            "input",
            "the adjoint needs an invertible triangular factor",
        ))
    })?;
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

    let m = crate::dense::multiply(&crate::dense::transpose(&q), &a_dot);
    let r_dot = crate::dense::multiply(&crate::dense::upper_triangle(&m), &r);
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
}

impl Df64Body {
    fn execute(
        &self,
        session: Option<&mut dyn tenferro_tensor::BackendSession>,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        match self {
            Self::Total => total_of(session, inputs),
            Self::Expand(shape) => expand_of(shape, session, inputs),
            Self::Qr => qr_of(session, inputs),
            Self::ToF64 => to_f64_of(session, inputs),
            Self::FromF64 => from_f64_of(session, inputs),
            Self::QrVjp(mask) => qr_vjp_of(*mask, session, inputs),
            Self::QrJvp => qr_jvp_of(session, inputs),
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
        _caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        // The erased context does not expose a session, so a body that needs one relies
        // on the session entry point and gathers what it can without one.
        self.body.execute(None, inputs)
    }

    fn supports_session(&self) -> bool {
        true
    }

    fn execute_in_session(
        &self,
        session: &mut dyn tenferro_tensor::BackendSession,
        _caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        self.body.execute(Some(session), inputs)
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
/// Returns an error when the module identifier is invalid.
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
/// Returns the runtime's typed error when the extension cannot be prepared or
/// executed for this input.
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
///
/// Executing the operation needs the caller-owned payload ownership contract from
/// #1789, which `ext/df64-proof/tests/extension_execution.rs` records.
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
