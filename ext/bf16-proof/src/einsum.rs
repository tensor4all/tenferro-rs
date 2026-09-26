//! A bfloat16 contraction through the runtime's extension module.
//!
//! #1793's precision table lists a bf16 CPU einsum "with documented f32 internal
//! calculation/accumulation", and asks for a contraction whose result distinguishes f32
//! accumulation from repeated bfloat16 rounding, compared against an independent reference with the
//! specified output rounding. This module supplies that: the operands are widened to `f32`, the
//! contraction accumulates there, and the result is rounded to bfloat16 once.
//!
//! What it does not do is inherit a wider pattern surface. The operation is the pairwise
//! contraction the table's example needs, so a trace, a repeated label, or more than two operands is
//! refused with a typed error rather than folded, which keeps this module's body to the
//! accumulation contract the row is about.

use std::any::Any;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::sync::Arc;

use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration};
use tenferro_runtime::extension::{ExtensionOp, ExtensionShapeContext, SymDim};
use tenferro_runtime::{
    CoreCapabilityKind, EngineId, ErasedExecutionContext, ErrorPhase, ExecutionContextIdentity,
    ExtensionCacheStore, ExtensionEngine, ExtensionModule, ExtensionModuleId,
    ExtensionModuleRegistrar, ExtensionPlanningConfig, ExtensionPrepareRequest, PrepareCapability,
    PrepareError, PreparedOperation, PreparedOperationBinding, PreparedOperationExecutor,
    PreparedOperationPlan, ProviderContractError, RuntimeConfigError, SpecializationProjection,
};
use tenferro_tensor::{DType, Tensor, TensorBackend, TensorRead};

use crate::Bf16;

/// The family the bfloat16 contraction belongs to.
pub const BF16_EINSUM_FAMILY: &str = "tenferro-bf16-proof.einsum.v1";

/// The canonical identity of the bfloat16 scalar a program declares.
pub const BF16_SCALAR_IDENTITY: &str = "tenferro-bf16-proof.bf16.v1";

/// A pairwise contraction in bfloat16.
///
/// The labels are the ones an ordinary einsum subscript string names, so `"ik,kj->ij"` is
/// `(&[0, 1], &[1, 2], &[0, 2])`. The operands are widened to `f32`, the contraction accumulates
/// there, and the result is rounded to bfloat16 once.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::einsum::Bf16Einsum;
///
/// let op = Bf16Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
/// assert_eq!(op.labels(), (&[0, 1][..], &[1, 2][..], &[0, 2][..]));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bf16Einsum {
    lhs: Vec<u32>,
    rhs: Vec<u32>,
    out: Vec<u32>,
}

impl Bf16Einsum {
    /// Build the pairwise contraction `lhs,rhs->out`.
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
    /// use tenferro_bf16_proof::einsum::Bf16Einsum;
    ///
    /// assert!(Bf16Einsum::new(&[0, 1], &[1, 2], &[0, 2]).is_ok());
    /// assert!(Bf16Einsum::new(&[], &[1, 2], &[0, 2]).is_err());
    /// ```
    pub fn new(lhs: &[u32], rhs: &[u32], out: &[u32]) -> tenferro_runtime::Result<Self> {
        let invalid = |message: &str| {
            tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
                "bf16_einsum",
                "pattern",
                message,
            ))
        };
        if lhs.is_empty() || rhs.is_empty() {
            return Err(invalid("an operand must carry at least one label"));
        }
        for labels in [lhs, rhs] {
            let mut seen = labels.to_vec();
            seen.sort_unstable();
            seen.dedup();
            if seen.len() != labels.len() {
                return Err(invalid(
                    "a label repeats within one operand, which is a trace and not supported here",
                ));
            }
        }
        for label in out {
            if !lhs.contains(label) && !rhs.contains(label) {
                return Err(invalid(
                    "an output label must appear in at least one operand",
                ));
            }
        }
        Ok(Self {
            lhs: lhs.to_vec(),
            rhs: rhs.to_vec(),
            out: out.to_vec(),
        })
    }

    /// The pattern's three label lists.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_bf16_proof::einsum::Bf16Einsum;
    ///
    /// let op = Bf16Einsum::new(&[0, 1], &[1, 2], &[0, 2]).expect("a contraction");
    /// assert_eq!(op.labels(), (&[0, 1][..], &[1, 2][..], &[0, 2][..]));
    /// ```
    #[must_use]
    pub fn labels(&self) -> (&[u32], &[u32], &[u32]) {
        (&self.lhs, &self.rhs, &self.out)
    }
}

impl ExtensionOp for Bf16Einsum {
    fn family_id(&self) -> &'static str {
        BF16_EINSUM_FAMILY
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        for labels in [&self.lhs, &self.rhs, &self.out] {
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
        2
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn scalar_identity(&self) -> Option<&'static str> {
        Some(BF16_SCALAR_IDENTITY)
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "bf16_einsum",
                dtype,
                "bf16_einsum takes an externally defined scalar",
            ));
        }
        if ctx.input_dtype(1)? != dtype {
            return Err(tenferro_tensor::Error::invalid_argument(
                "bf16_einsum",
                "inputs",
                "both operands must carry the same scalar",
            ));
        }
        let lhs = ctx.input_shape(0)?;
        let rhs = ctx.input_shape(1)?;
        if lhs.len() != self.lhs.len() || rhs.len() != self.rhs.len() {
            return Err(tenferro_tensor::Error::rank_mismatch(
                "bf16_einsum",
                self.lhs.len().max(self.rhs.len()),
                lhs.len().min(rhs.len()),
            ));
        }
        let mut out_shape = Vec::with_capacity(self.out.len());
        for label in &self.out {
            let extent = self
                .lhs
                .iter()
                .position(|candidate| candidate == label)
                .map(|axis| lhs[axis].clone())
                .or_else(|| {
                    self.rhs
                        .iter()
                        .position(|candidate| candidate == label)
                        .map(|axis| rhs[axis].clone())
                })
                .ok_or_else(|| {
                    tenferro_tensor::Error::invalid_argument(
                        "bf16_einsum",
                        "pattern",
                        "an output label must appear in at least one operand",
                    )
                })?;
            out_shape.push(extent);
        }
        Ok(vec![(dtype, out_shape)])
    }
}

/// The number of elements a shape describes.
fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Advance a column-major odometer by one.
fn advance(index: &mut [usize], shape: &[usize]) {
    for axis in 0..shape.len() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

/// The column-major offset an operand's labels select at one output and contracted index.
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

/// The stored values of a bfloat16 tensor, widened to `f32`.
fn values_of(op: &'static str, tensor: &Tensor) -> tenferro_runtime::Result<Vec<f32>> {
    match tensor.external_payload() {
        Some(payload) => payload
            .downcast_ref::<Bf16>()
            .map(|stored| {
                stored
                    .as_slice()
                    .iter()
                    .map(|value| value.to_f32())
                    .collect()
            })
            .ok_or_else(|| {
                tenferro_runtime::Error::from(tenferro_tensor::Error::invalid_argument(
                    op,
                    "input",
                    "the operand does not carry a bfloat16 payload",
                ))
            }),
        None => Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::unsupported_dtype(
                op,
                tensor.dtype(),
                "bf16_einsum takes an externally defined bfloat16 scalar",
            ),
        )),
    }
}

/// Contract two operands, accumulating in `f32` and rounding once.
fn contract(op: &dyn ExtensionOp, inputs: &[&Tensor]) -> tenferro_runtime::Result<Vec<Tensor>> {
    let name = "bf16_einsum";
    let Some(contraction) = op.as_any().downcast_ref::<Bf16Einsum>() else {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                name,
                "payload",
                "the operation is not a bfloat16 contraction",
            ),
        ));
    };
    if inputs.len() != 2 {
        return Err(tenferro_runtime::Error::from(
            tenferro_tensor::Error::invalid_argument(
                name,
                "input",
                "a bf16 contraction takes two operands",
            ),
        ));
    }
    let (lhs_labels, rhs_labels, out_labels) = contraction.labels();
    let lhs_values = values_of(name, inputs[0])?;
    let rhs_values = values_of(name, inputs[1])?;
    let lhs_shape = inputs[0].shape().to_vec();
    let rhs_shape = inputs[1].shape().to_vec();

    let mut extents: Vec<(u32, usize)> = Vec::new();
    for (labels, shape) in [(lhs_labels, &lhs_shape), (rhs_labels, &rhs_shape)] {
        for (axis, label) in labels.iter().enumerate() {
            match extents.iter().find(|(existing, _)| existing == label) {
                Some((_, existing)) if *existing != shape[axis] => {
                    return Err(tenferro_runtime::Error::from(
                        tenferro_tensor::Error::invalid_argument(
                            name,
                            "inputs",
                            "the operands disagree on the extent of a shared label",
                        ),
                    ))
                }
                Some(_) => {}
                None => extents.push((*label, shape[axis])),
            }
        }
    }
    let extent_of = |label: u32| {
        extents
            .iter()
            .find(|(existing, _)| *existing == label)
            .map(|(_, extent)| *extent)
            .unwrap_or(1)
    };
    let out_shape: Vec<usize> = out_labels.iter().map(|label| extent_of(*label)).collect();
    let mut summed_labels: Vec<u32> = Vec::new();
    for label in lhs_labels.iter().chain(rhs_labels.iter()) {
        if !out_labels.contains(label) && !summed_labels.contains(label) {
            summed_labels.push(*label);
        }
    }
    let summed_shape: Vec<usize> = summed_labels
        .iter()
        .map(|label| extent_of(*label))
        .collect();

    let out_count = element_count(&out_shape);
    let summed_count = element_count(&summed_shape);
    let mut accumulated = vec![0.0_f32; out_count];
    let mut out_index = vec![0usize; out_shape.len()];
    let mut summed_index = vec![0usize; summed_shape.len()];
    for slot in accumulated.iter_mut() {
        for value in summed_index.iter_mut() {
            *value = 0;
        }
        let mut total = 0.0_f32;
        for _ in 0..summed_count {
            let lhs_offset = offset_for(
                lhs_labels,
                &lhs_shape,
                out_labels,
                &out_index,
                &summed_labels,
                &summed_index,
            );
            let rhs_offset = offset_for(
                rhs_labels,
                &rhs_shape,
                out_labels,
                &out_index,
                &summed_labels,
                &summed_index,
            );
            total += lhs_values[lhs_offset] * rhs_values[rhs_offset];
            advance(&mut summed_index, &summed_shape);
        }
        *slot = total;
        advance(&mut out_index, &out_shape);
    }

    // One rounding, after the accumulation rather than at every step.
    let rounded: Vec<Bf16> = accumulated.into_iter().map(Bf16::from_f32).collect();
    let tensor = tenferro_tensor_core::HostTensor::from_vec_col_major(out_shape, rounded)
        .map_err(|source| tenferro_tensor::Error::validation(name, source))
        .map_err(tenferro_runtime::Error::from)?;
    Ok(vec![Tensor::external(
        tenferro_tensor_core::ErasedHostTensor::new(tensor),
    )])
}

/// The engine the bfloat16 contraction is registered under.
#[derive(Debug)]
struct Bf16EinsumEngine<B: std::fmt::Debug + Send + Sync> {
    family_id: &'static str,
    engine_id: EngineId,
    _backend: PhantomData<B>,
}

impl<B: TensorBackend + std::fmt::Debug + Send + Sync + 'static> ExtensionEngine
    for Bf16EinsumEngine<B>
{
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<tenferro_cpu::CpuBackend>()
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        if request.operation().family_id() != self.family_id {
            return Err(PrepareError::ProviderContract {
                source: ProviderContractError::WrongOperationFamily {
                    expected: CoreCapabilityKind::Elementwise,
                    operation: self.family_id,
                },
            });
        }
        let prepared = Arc::new(Bf16EinsumPrepared::<B> {
            binding: request.binding().clone(),
            specialization: request.specialization().clone(),
            op: request.operation().clone_arc(),
            _backend: PhantomData,
        });
        Ok(PrepareCapability::Prepared(
            PreparedOperationPlan::executable(prepared.clone(), prepared),
        ))
    }
}

/// The planning config the runtime keys the bfloat16 family under.
///
/// The runtime keeps one planning config per engine identity, so the family states its identity
/// here rather than leaving the engine without one.
#[derive(Debug)]
struct Bf16EinsumPlanning {
    family_id: &'static str,
}

impl ExtensionPlanningConfig for Bf16EinsumPlanning {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, _state: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other.family_id() == self.family_id
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

/// The prepared bfloat16 contraction.
#[derive(Debug)]
struct Bf16EinsumPrepared<B: std::fmt::Debug + Send + Sync> {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    op: Arc<dyn ExtensionOp>,
    _backend: PhantomData<B>,
}

impl<B: TensorBackend + std::fmt::Debug + Send + Sync + 'static> PreparedOperation
    for Bf16EinsumPrepared<B>
{
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

impl<B: TensorBackend + std::fmt::Debug + Send + Sync + 'static> PreparedOperationExecutor
    for Bf16EinsumPrepared<B>
{
    fn execute(
        &self,
        context: &mut ErasedExecutionContext<'_>,
        extension_caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        // The body needs contiguous inputs, which a session on the binding's backend provides.
        let backend = context
            .downcast_mut::<B>(self.binding.context_identity())
            .map_err(|source| {
                tenferro_runtime::Error::runtime_state_source(
                    "extension",
                    ErrorPhase::Execution,
                    source,
                )
            })?;
        let _ = extension_caches;
        let materialized = backend
            .with_backend_session(|exec| {
                inputs
                    .iter()
                    .cloned()
                    .map(|input| exec.to_contiguous_read(input))
                    .collect::<tenferro_tensor::Result<Vec<Tensor>>>()
            })
            .map_err(tenferro_runtime::Error::from)?;
        let borrowed: Vec<&Tensor> = materialized.iter().collect();
        contract(self.op.as_ref(), &borrowed)
    }
}

/// The module an application installs to run the bfloat16 contraction.
#[derive(Debug)]
struct Bf16EinsumModule<B: std::fmt::Debug + Send + Sync> {
    module_id: ExtensionModuleId,
    engine_id: EngineId,
    _backend: PhantomData<B>,
}

impl<B: TensorBackend + std::fmt::Debug + Send + Sync + 'static> ExtensionModule
    for Bf16EinsumModule<B>
{
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), tenferro_runtime::ExtensionModuleError> {
        registrar.register_engine(Arc::new(Bf16EinsumEngine::<B> {
            family_id: BF16_EINSUM_FAMILY,
            engine_id: self.engine_id.clone(),
            _backend: PhantomData,
        }))?;
        registrar.register_planning_config(
            self.engine_id.clone(),
            Arc::new(Bf16EinsumPlanning {
                family_id: BF16_EINSUM_FAMILY,
            }),
        )?;
        Ok(())
    }
}

/// The module for the CPU backend's engine identity.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when the module identity cannot be built or the backend has no
/// engine identity.
///
/// # Examples
///
/// ```rust
/// use tenferro_bf16_proof::einsum::module;
///
/// assert!(module().is_ok());
/// ```
pub fn module() -> Result<Arc<dyn ExtensionModule>, RuntimeConfigError> {
    Ok(Arc::new(Bf16EinsumModule::<tenferro_cpu::CpuBackend> {
        module_id: ExtensionModuleId::new("tenferro-bf16-proof.module")?,
        engine_id: tenferro_cpu::runtime_engine_id()?,
        _backend: PhantomData,
    }))
}
