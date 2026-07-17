use std::any::Any;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_einsum::Subscripts;
#[cfg(feature = "autodiff")]
use tenferro_ops::ad::{transpose_input::TransposeInputRef, PrimitiveRuleBuilder};
#[cfg(feature = "autodiff")]
use tenferro_ops::ext_op::{ExtensionLinearTransposeRule, ExtensionLinearizeRule};
use tenferro_ops::ext_op::{ExtensionOp, HostReference};
#[cfg(feature = "autodiff")]
use tenferro_ops::std_tensor_op::StdTensorOp;
#[cfg(feature = "autodiff")]
use tenferro_ops::ShapeGuardContext;
use tenferro_ops::SymDim;
#[cfg(feature = "autodiff")]
use tenferro_ops::{ExtensionRegistryError, ExtensionRuleSet};
use tenferro_runtime::{ExtensionExecutor, ExtensionRuntimeRegistryError, HostReferenceRuntime};
#[cfg(feature = "autodiff")]
use tenferro_tensor::TensorScalar;
use tenferro_tensor::{DType, Tensor, TensorBackend};
#[cfg(feature = "autodiff")]
use tidu::{ADRuleError, ADRuleKind, ADRuleResult, PrimitiveTransposeInput};

use crate::einsum::tropical_einsum_subscripts_with_argmax;
use crate::error::unsupported_dtype;
#[cfg(feature = "autodiff")]
use crate::einsum::TropicalArgmaxStep;
use crate::TropicalKind;

pub(crate) const TROPICAL_EINSUM_FAMILY_ID: &str = "tenferro-ext-tropical.einsum.v1";
#[cfg(feature = "autodiff")]
const TROPICAL_EINSUM_JVP_FAMILY_ID: &str = "tenferro-ext-tropical.einsum_jvp.v1";
#[cfg(feature = "autodiff")]
const TROPICAL_EINSUM_VJP_FAMILY_ID: &str = "tenferro-ext-tropical.einsum_vjp.v1";

fn invalid_config(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(op, "configuration", message)
}

/// Register tropical extension runtimes on a graph or eager executor.
///
/// The runtime executor is intentionally thin: it delegates to each tropical
/// extension op's optional host reference implementation.
/// AD rules are registered separately through `tropical_ad_rules` when the
/// `autodiff` feature is enabled.
///
/// # Errors
///
/// Returns [`tenferro_runtime::ExtensionRuntimeRegistryError`] if runtime
/// registration fails.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::GraphExecutor;
///
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor
///     .register_extension(tenferro_ext_tropical::register_runtime)
///     .unwrap();
/// assert!(executor
///     .extension_executor()
///     .registry()
///     .contains("tenferro-ext-tropical.einsum.v1"));
/// ```
pub fn register_runtime<B: TensorBackend + 'static>(
    executor: &mut ExtensionExecutor<B>,
) -> Result<(), ExtensionRuntimeRegistryError> {
    executor
        .registry_mut()
        .register(Arc::new(HostReferenceRuntime::<B>::new(
            TROPICAL_EINSUM_FAMILY_ID,
        )))?;
    #[cfg(feature = "autodiff")]
    {
        executor
            .registry_mut()
            .register(Arc::new(HostReferenceRuntime::<B>::new(
                TROPICAL_EINSUM_JVP_FAMILY_ID,
            )))?;
        executor
            .registry_mut()
            .register(Arc::new(HostReferenceRuntime::<B>::new(
                TROPICAL_EINSUM_VJP_FAMILY_ID,
            )))?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TropicalEinsumOp {
    kind: TropicalKind,
    subscripts: Subscripts,
}

impl TropicalEinsumOp {
    pub(crate) fn new(kind: TropicalKind, subscripts: Subscripts) -> Self {
        Self { kind, subscripts }
    }
}

impl ExtensionOp for TropicalEinsumOp {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_kind(self.kind, hasher);
        hash_subscripts(&self.subscripts, hasher);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
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

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtypes = [ctx.input_dtype(0)?, ctx.input_dtype(1)?];
        let input_shapes = [ctx.input_shape(0)?.to_vec(), ctx.input_shape(1)?.to_vec()];
        let meta = infer_tropical_output_meta(
            ctx,
            &self.subscripts,
            &input_dtypes,
            &input_shapes,
            "tropical_einsum",
        )?;
        Ok(vec![meta])
    }

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for TropicalEinsumOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        let result = tropical_einsum_subscripts_with_argmax(self.kind, inputs, &self.subscripts)?;
        Ok(vec![result.output])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TropicalEinsumJvpOp {
    kind: TropicalKind,
    subscripts: Subscripts,
    active_inputs: Vec<usize>,
}

#[cfg(feature = "autodiff")]
impl TropicalEinsumJvpOp {
    fn new(kind: TropicalKind, subscripts: Subscripts, active_inputs: Vec<usize>) -> Self {
        Self {
            kind,
            subscripts,
            active_inputs,
        }
    }
}

#[cfg(feature = "autodiff")]
impl ExtensionOp for TropicalEinsumJvpOp {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_JVP_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_kind(self.kind, hasher);
        hash_subscripts(&self.subscripts, hasher);
        hasher.write_usize(self.active_inputs.len());
        for &active in &self.active_inputs {
            hasher.write_usize(active);
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        2 + self.active_inputs.len()
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtypes = (0..self.input_count())
            .map(|input| ctx.input_dtype(input))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let input_shapes = (0..self.input_count())
            .map(|input| ctx.input_shape(input).map(<[_]>::to_vec))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let primal = infer_tropical_output_meta(
            ctx,
            &self.subscripts,
            &input_dtypes[..2],
            &input_shapes[..2],
            "tropical_einsum_jvp",
        )?;
        for (active_pos, &active) in self.active_inputs.iter().enumerate() {
            let tangent_idx = 2 + active_pos;
            if active >= 2 || input_dtypes[tangent_idx] != input_dtypes[active] {
                return Err(invalid_config(
                    "tropical_einsum_jvp",
                    "active input tangent dtype does not match primal dtype",
                ));
            }
            ctx.require_same_shape(tangent_idx, active)?;
        }
        Ok(vec![primal])
    }

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

#[cfg(feature = "autodiff")]
impl HostReference for TropicalEinsumJvpOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_tropical_jvp_inputs(inputs, &self.subscripts, &self.active_inputs)?;
        let primal = tropical_einsum_subscripts_with_argmax(
            self.kind,
            &[inputs[0], inputs[1]],
            &self.subscripts,
        )?;
        let step = single_argmax_step(&primal.argmax)?;
        match primal.output.dtype() {
            DType::F32 => {
                execute_jvp_typed::<f32>(inputs, &self.subscripts, step, &self.active_inputs)
                    .map(|tensor| vec![tensor])
            }
            DType::F64 => {
                execute_jvp_typed::<f64>(inputs, &self.subscripts, step, &self.active_inputs)
                    .map(|tensor| vec![tensor])
            }
            dtype => Err(unsupported_dtype("tropical_einsum_jvp", dtype)),
        }
    }
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TropicalEinsumVjpOp {
    kind: TropicalKind,
    subscripts: Subscripts,
    active_input: usize,
}

#[cfg(feature = "autodiff")]
impl TropicalEinsumVjpOp {
    fn new(kind: TropicalKind, subscripts: Subscripts, active_input: usize) -> Self {
        Self {
            kind,
            subscripts,
            active_input,
        }
    }
}

#[cfg(feature = "autodiff")]
impl ExtensionOp for TropicalEinsumVjpOp {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_VJP_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_kind(self.kind, hasher);
        hash_subscripts(&self.subscripts, hasher);
        hasher.write_usize(self.active_input);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        3
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if self.active_input >= 2 {
            return Err(invalid_config(
                "tropical_einsum_vjp",
                format!(
                    "expected active input < 2, got active_input={}",
                    self.active_input
                ),
            ));
        }
        let input_dtypes = [
            ctx.input_dtype(0)?,
            ctx.input_dtype(1)?,
            ctx.input_dtype(2)?,
        ];
        let input_shapes = [
            ctx.input_shape(0)?.to_vec(),
            ctx.input_shape(1)?.to_vec(),
            ctx.input_shape(2)?.to_vec(),
        ];
        let (_, primal_output_shape) = infer_tropical_output_meta(
            ctx,
            &self.subscripts,
            &input_dtypes[..2],
            &input_shapes[..2],
            "tropical_einsum_vjp",
        )?;
        if input_dtypes[2] != input_dtypes[self.active_input] {
            return Err(tenferro_tensor::Error::dtype_mismatch(
                "tropical_einsum_vjp",
                input_dtypes[self.active_input],
                input_dtypes[2],
            ));
        }
        if input_shapes[2].len() != primal_output_shape.len() {
            return Err(tenferro_tensor::Error::rank_mismatch(
                "tropical_einsum_vjp",
                primal_output_shape.len(),
                input_shapes[2].len(),
            ));
        }
        for (cotangent_dim, output_dim) in input_shapes[2].iter().cloned().zip(primal_output_shape)
        {
            ctx.require_equal(cotangent_dim, output_dim)?;
        }
        Ok(vec![(
            input_dtypes[self.active_input],
            input_shapes[self.active_input].clone(),
        )])
    }

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

#[cfg(feature = "autodiff")]
impl HostReference for TropicalEinsumVjpOp {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        validate_tropical_vjp_inputs(inputs, &self.subscripts, self.active_input)?;
        let primal = tropical_einsum_subscripts_with_argmax(
            self.kind,
            &[inputs[0], inputs[1]],
            &self.subscripts,
        )?;
        let step = single_argmax_step(&primal.argmax)?;
        match inputs[self.active_input].dtype() {
            DType::F32 => {
                execute_vjp_typed::<f32>(inputs, &self.subscripts, step, self.active_input)
                    .map(|tensor| vec![tensor])
            }
            DType::F64 => {
                execute_vjp_typed::<f64>(inputs, &self.subscripts, step, self.active_input)
                    .map(|tensor| vec![tensor])
            }
            dtype => Err(unsupported_dtype("tropical_einsum_vjp", dtype)),
        }
    }
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct TropicalEinsumAdRule;

#[cfg(feature = "autodiff")]
impl ExtensionLinearizeRule for TropicalEinsumAdRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_FAMILY_ID
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_primal_op(op, ADRuleKind::Jvp)?;
        validate_ad_supported(&op.subscripts, ADRuleKind::Jvp)?;
        let active_inputs: Vec<usize> = tangent_in
            .iter()
            .enumerate()
            .filter_map(|(idx, tangent)| tangent.is_some().then_some(idx))
            .collect();
        if active_inputs.is_empty() {
            return Ok(vec![None]);
        }

        let mut inputs = vec![
            ValueRef::External(primal_in[0].clone()),
            ValueRef::External(primal_in[1].clone()),
        ];
        for &active in &active_inputs {
            // `active_inputs` is derived from `tangent_in.is_some()` above; keep
            // this branch explicit so future audits do not classify it as a
            // user-reachable unwrap.
            let Some(tangent) = tangent_in.get(active).copied().flatten() else {
                return Ok(vec![None]);
            };
            inputs.push(ValueRef::Local(tangent));
        }
        let active_mask = std::iter::repeat_n(false, 2)
            .chain(std::iter::repeat_n(true, active_inputs.len()))
            .collect();
        let out = builder.add_operation(
            StdTensorOp::Extension(Arc::new(TropicalEinsumJvpOp::new(
                op.kind,
                op.subscripts.clone(),
                active_inputs,
            ))),
            inputs,
            OperationRole::Linearized { active_mask },
        );
        Ok(vec![Some(out[0])])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct TropicalEinsumJvpAdRule;

#[cfg(feature = "autodiff")]
impl ExtensionLinearTransposeRule for TropicalEinsumJvpAdRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_JVP_FAMILY_ID
    }

    fn linear_transpose(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[PrimitiveTransposeInput<StdTensorOp>],
        active_mask: &[bool],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_jvp_op(op, ADRuleKind::Transpose)?;
        validate_ad_supported(&op.subscripts, ADRuleKind::Transpose)?;
        let inputs: Vec<_> = inputs.iter().map(TransposeInputRef::new).collect();
        let Some(ct) = cotangent_out.first().copied().flatten() else {
            return Ok(vec![None; op.input_count()]);
        };
        let lhs = inputs[0].fixed_value("tropical einsum VJP", 0)?;
        let rhs = inputs[1].fixed_value("tropical einsum VJP", 1)?;

        let mut result = vec![None; op.input_count()];
        for (active_pos, &active_input) in op.active_inputs.iter().enumerate() {
            let tangent_input_idx = 2 + active_pos;
            if !active_mask.get(tangent_input_idx).copied().unwrap_or(false) {
                continue;
            }
            let out = builder.add_operation(
                StdTensorOp::Extension(Arc::new(TropicalEinsumVjpOp::new(
                    op.kind,
                    op.subscripts.clone(),
                    active_input,
                ))),
                vec![lhs.clone(), rhs.clone(), ValueRef::Local(ct)],
                OperationRole::Linearized {
                    active_mask: vec![false, false, true],
                },
            );
            result[tangent_input_idx] = Some(out[0]);
        }
        Ok(result)
    }
}

/// Build an explicit AD rule set for tropical traced einsum extensions.
///
/// The returned set is intended for `tenferro_ad::AdContext`. It registers
/// the primal tropical einsum linearization rule and the transposed JVP rule
/// used for reverse-mode cotangent routing.
///
/// # Errors
///
/// Returns [`tenferro_ops::ExtensionRegistryError`] if rule registration into
/// the fresh rule set fails.
///
/// # Examples
///
/// ```
/// let rules = tenferro_ext_tropical::tropical_ad_rules().unwrap();
///
/// assert!(rules.is_linearize_registered("tenferro-ext-tropical.einsum.v1"));
/// assert!(rules.is_linear_transpose_registered("tenferro-ext-tropical.einsum_jvp.v1"));
/// ```
#[cfg(feature = "autodiff")]
pub fn tropical_ad_rules() -> Result<ExtensionRuleSet, ExtensionRegistryError> {
    let mut rules = ExtensionRuleSet::new();
    rules.register_linearize(Arc::new(TropicalEinsumAdRule))?;
    rules.register_linear_transpose(Arc::new(TropicalEinsumJvpAdRule))?;
    Ok(rules)
}

fn infer_tropical_output_meta(
    ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    subscripts: &Subscripts,
    input_dtypes: &[DType],
    input_shapes: &[Vec<SymDim>],
    op: &'static str,
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    if input_shapes.len() != 2 || input_dtypes.len() != input_shapes.len() {
        return Err(invalid_config(
            op,
            format!(
                "tropical einsum expected two input metadata records, got dtypes={} shapes={}",
                input_dtypes.len(),
                input_shapes.len()
            ),
        ));
    }
    if subscripts.inputs.len() != 2 {
        return Err(invalid_config(op, "tropical einsum requires two inputs"));
    }
    if input_dtypes[0] != input_dtypes[1] {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            op,
            input_dtypes[0],
            input_dtypes[1],
        ));
    }
    if !matches!(input_dtypes[0], DType::F32 | DType::F64) {
        return Err(unsupported_dtype(op, input_dtypes[0]));
    }

    let mut label_dims: HashMap<u32, SymDim> = HashMap::new();
    for (labels, shape) in subscripts.inputs.iter().zip(input_shapes.iter()) {
        if labels.len() != shape.len() {
            return Err(tenferro_tensor::Error::rank_mismatch(
                op,
                labels.len(),
                shape.len(),
            ));
        }
        for (&label, dim) in labels.iter().zip(shape.iter()) {
            if let Some(existing) = label_dims.get(&label) {
                ctx.require_equal(existing.clone(), dim.clone())?;
            } else {
                label_dims.insert(label, dim.clone());
            }
        }
    }
    let output_shape = subscripts
        .output
        .iter()
        .map(|label| label_dims.get(label).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid_config(op, "output labels must be present in input metadata"))?;
    Ok((input_dtypes[0], output_shape))
}

#[cfg(feature = "autodiff")]
fn validate_tropical_primal_host_meta(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    op: &'static str,
) -> tenferro_tensor::Result<Vec<usize>> {
    if inputs.len() != 2 || subscripts.inputs.len() != 2 {
        return Err(invalid_config(
            op,
            format!(
                "expected two primal inputs, got tensors={} subscripts={}",
                inputs.len(),
                subscripts.inputs.len()
            ),
        ));
    }
    let dtype = inputs[0].dtype();
    if inputs[1].dtype() != dtype {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            op,
            dtype,
            inputs[1].dtype(),
        ));
    }
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(unsupported_dtype(op, dtype));
    }

    let mut label_dims = HashMap::new();
    for (input_idx, (labels, input)) in subscripts.inputs.iter().zip(inputs).enumerate() {
        if labels.len() != input.shape().len() {
            return Err(tenferro_tensor::Error::rank_mismatch(
                op,
                labels.len(),
                input.shape().len(),
            ));
        }
        for (&label, &dim) in labels.iter().zip(input.shape()) {
            if let Some(existing) = label_dims.insert(label, dim) {
                if existing != dim {
                    return Err(tenferro_tensor::Error::shape_mismatch(
                        op,
                        vec![existing],
                        vec![dim],
                    ));
                }
            }
        }
    }
    subscripts
        .output
        .iter()
        .map(|label| {
            label_dims
                .get(label)
                .copied()
                .ok_or_else(|| invalid_config(op, format!("output label {label} is absent")))
        })
        .collect()
}

#[cfg(feature = "autodiff")]
fn validate_tropical_jvp_inputs(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    active_inputs: &[usize],
) -> tenferro_tensor::Result<()> {
    let expected = 2 + active_inputs.len();
    if inputs.len() != expected {
        return Err(invalid_config(
            "tropical_einsum_jvp",
            format!("expected {expected} inputs, got {}", inputs.len()),
        ));
    }
    validate_tropical_primal_host_meta(&inputs[..2], subscripts, "tropical_einsum_jvp")?;
    for (active_pos, &active) in active_inputs.iter().enumerate() {
        if active >= 2 {
            return Err(invalid_config(
                "tropical_einsum_jvp",
                format!("invalid active input {active}"),
            ));
        }
        let tangent = inputs[2 + active_pos];
        if tangent.dtype() != inputs[active].dtype() {
            return Err(tenferro_tensor::Error::dtype_mismatch(
                "tropical_einsum_jvp",
                inputs[active].dtype(),
                tangent.dtype(),
            ));
        }
        if tangent.shape() != inputs[active].shape() {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "tropical_einsum_jvp",
                inputs[active].shape().to_vec(),
                tangent.shape().to_vec(),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "autodiff")]
fn validate_tropical_vjp_inputs(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    active_input: usize,
) -> tenferro_tensor::Result<()> {
    if inputs.len() != 3 {
        return Err(invalid_config(
            "tropical_einsum_vjp",
            format!("expected 3 inputs, got {}", inputs.len()),
        ));
    }
    if active_input >= 2 {
        return Err(invalid_config(
            "tropical_einsum_vjp",
            format!("invalid active input {active_input}"),
        ));
    }
    let output_shape =
        validate_tropical_primal_host_meta(&inputs[..2], subscripts, "tropical_einsum_vjp")?;
    let cotangent = inputs[2];
    if cotangent.dtype() != inputs[active_input].dtype() {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            "tropical_einsum_vjp",
            inputs[active_input].dtype(),
            cotangent.dtype(),
        ));
    }
    if cotangent.shape() != output_shape {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "tropical_einsum_vjp",
            output_shape,
            cotangent.shape().to_vec(),
        ));
    }
    Ok(())
}

#[cfg(feature = "autodiff")]
fn validate_ad_supported(subscripts: &Subscripts, kind: ADRuleKind) -> ADRuleResult<()> {
    if subscripts.inputs.len() != 2 {
        return Err(ADRuleError::unsupported(
            "tropical einsum AD supports only binary inputs",
            kind,
        ));
    }
    if has_repeated_labels(&subscripts.output)
        || subscripts
            .inputs
            .iter()
            .any(|labels| has_repeated_labels(labels))
    {
        return Err(ADRuleError::unsupported(
            "tropical einsum AD does not support repeated labels",
            kind,
        ));
    }
    let has_contracted = subscripts.inputs[0]
        .iter()
        .any(|label| subscripts.inputs[1].contains(label) && !subscripts.output.contains(label));
    if !has_contracted {
        return Err(ADRuleError::unsupported(
            "tropical einsum AD requires contracted modes",
            kind,
        ));
    }
    Ok(())
}

#[cfg(feature = "autodiff")]
fn has_repeated_labels(labels: &[u32]) -> bool {
    labels
        .iter()
        .enumerate()
        .any(|(idx, label)| labels[..idx].contains(label))
}

#[cfg(feature = "autodiff")]
fn downcast_primal_op(op: &dyn ExtensionOp, kind: ADRuleKind) -> ADRuleResult<&TropicalEinsumOp> {
    op.as_any()
        .downcast_ref::<TropicalEinsumOp>()
        .ok_or_else(|| ADRuleError::unsupported("tropical einsum payload type mismatch", kind))
}

#[cfg(feature = "autodiff")]
fn downcast_jvp_op(op: &dyn ExtensionOp, kind: ADRuleKind) -> ADRuleResult<&TropicalEinsumJvpOp> {
    op.as_any()
        .downcast_ref::<TropicalEinsumJvpOp>()
        .ok_or_else(|| ADRuleError::unsupported("tropical einsum JVP payload type mismatch", kind))
}

#[cfg(feature = "autodiff")]
fn execute_jvp_typed<T>(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    step: &TropicalArgmaxStep,
    active_inputs: &[usize],
) -> tenferro_tensor::Result<Tensor>
where
    T: TensorScalar + Copy + Default + std::ops::AddAssign,
{
    let output_shape = step.output_shape().to_vec();
    let output_len = element_count(&output_shape)?;
    let mut out = vec![T::default(); output_len];
    for (active_pos, &active_input) in active_inputs.iter().enumerate() {
        let tangent = typed_slice::<T>(inputs[2 + active_pos])?;
        let labels = &subscripts.inputs[active_input];
        for (output_index, out_value) in out.iter_mut().enumerate() {
            let offset = routed_input_offset(
                step,
                labels,
                inputs[active_input].shape(),
                output_index,
                "tropical_einsum_jvp",
            )?;
            let tangent_value = tangent.get(offset).ok_or_else(|| {
                invalid_config("tropical_einsum_jvp", "tangent offset is out of bounds")
            })?;
            *out_value += *tangent_value;
        }
    }
    Tensor::from_vec_col_major(output_shape, out)
}

#[cfg(feature = "autodiff")]
fn execute_vjp_typed<T>(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    step: &TropicalArgmaxStep,
    active_input: usize,
) -> tenferro_tensor::Result<Tensor>
where
    T: TensorScalar + Copy + Default + std::ops::AddAssign,
{
    let cotangent = typed_slice::<T>(inputs[2])?;
    let output_len = element_count(step.output_shape())?;
    if cotangent.len() != output_len {
        return Err(invalid_config(
            "tropical_einsum_vjp",
            format!(
                "cotangent length {} does not match tropical output length {output_len}",
                cotangent.len()
            ),
        ));
    }
    let active_shape = inputs[active_input].shape().to_vec();
    let mut out = vec![T::default(); element_count(&active_shape)?];
    let labels = &subscripts.inputs[active_input];
    for (output_index, &ct) in cotangent.iter().enumerate() {
        let offset = routed_input_offset(
            step,
            labels,
            &active_shape,
            output_index,
            "tropical_einsum_vjp",
        )?;
        let slot = out.get_mut(offset).ok_or_else(|| {
            invalid_config("tropical_einsum_vjp", "scatter offset is out of bounds")
        })?;
        *slot += ct;
    }
    Tensor::from_vec_col_major(active_shape, out)
}

#[cfg(feature = "autodiff")]
fn routed_input_offset(
    step: &TropicalArgmaxStep,
    input_labels: &[u32],
    input_shape: &[usize],
    output_index: usize,
    op: &'static str,
) -> tenferro_tensor::Result<usize> {
    if input_labels.len() != input_shape.len() {
        return Err(invalid_config(
            op,
            "input labels do not match active input rank",
        ));
    }
    let output_coords = decode_col_major_index(output_index, step.output_shape())
        .ok_or_else(|| invalid_config(op, "output index is outside argmax output shape"))?;
    let winner_coords = step
        .winner_coordinates(output_index)
        .ok_or_else(|| invalid_config(op, "argmax winner is outside contracted shape"))?;
    let strides = col_major_strides(input_shape)?;
    input_labels
        .iter()
        .zip(strides.iter())
        .try_fold(0usize, |offset, (&label, &stride)| {
            let coordinate = if let Some(axis) = step
                .output_subscripts()
                .iter()
                .position(|candidate| *candidate == label)
            {
                output_coords[axis]
            } else if let Some(axis) = step
                .contracted_subscripts()
                .iter()
                .position(|candidate| *candidate == label)
            {
                winner_coords[axis]
            } else {
                return Err(invalid_config(
                    op,
                    format!("input label {label} requires unsupported pre-reduction"),
                ));
            };
            offset
                .checked_add(coordinate.checked_mul(stride).ok_or_else(|| {
                    invalid_config(op, "routed offset multiplication overflows usize")
                })?)
                .ok_or_else(|| invalid_config(op, "routed offset addition overflows usize"))
        })
}

#[cfg(feature = "autodiff")]
fn typed_slice<'a, T>(tensor: &'a Tensor) -> tenferro_tensor::Result<&'a [T]>
where
    T: TensorScalar,
{
    tensor.as_slice::<T>()
}

#[cfg(feature = "autodiff")]
fn single_argmax_step(
    steps: &[TropicalArgmaxStep],
) -> tenferro_tensor::Result<&TropicalArgmaxStep> {
    if steps.len() != 1 {
        return Err(invalid_config(
            "tropical_einsum_ad",
            format!("expected one argmax step, got {}", steps.len()),
        ));
    }
    Ok(&steps[0])
}

fn hash_kind(kind: TropicalKind, hasher: &mut dyn Hasher) {
    match kind {
        TropicalKind::MaxPlus => hasher.write_u8(0),
        TropicalKind::MinPlus => hasher.write_u8(1),
    }
}

fn hash_subscripts(subscripts: &Subscripts, hasher: &mut dyn Hasher) {
    hasher.write_usize(subscripts.inputs.len());
    for input in &subscripts.inputs {
        hasher.write_usize(input.len());
        for label in input {
            hasher.write_u32(*label);
        }
    }
    hasher.write_usize(subscripts.output.len());
    for label in &subscripts.output {
        hasher.write_u32(*label);
    }
}

#[cfg(feature = "autodiff")]
fn col_major_strides(shape: &[usize]) -> tenferro_tensor::Result<Vec<usize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| invalid_config("tropical_einsum_ad", "shape overflows usize"))?;
    }
    Ok(strides)
}

#[cfg(feature = "autodiff")]
fn element_count(shape: &[usize]) -> tenferro_tensor::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &extent| {
        acc.checked_mul(extent)
            .ok_or_else(|| invalid_config("tropical_einsum_ad", "shape overflows usize"))
    })
}

#[cfg(feature = "autodiff")]
fn decode_col_major_index(mut flat: usize, shape: &[usize]) -> Option<Vec<usize>> {
    let total = shape
        .iter()
        .try_fold(1usize, |acc, &extent| acc.checked_mul(extent))?;
    if flat >= total {
        return None;
    }
    let mut coordinates = Vec::with_capacity(shape.len());
    for &extent in shape {
        if extent == 0 {
            return None;
        }
        coordinates.push(flat % extent);
        flat /= extent;
    }
    Some(coordinates)
}

#[cfg(all(test, feature = "autodiff"))]
mod tests;
