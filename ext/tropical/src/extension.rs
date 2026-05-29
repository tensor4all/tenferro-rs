use std::any::Any;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use chainrules_core::{ADRuleError, ADRuleKind, ADRuleResult};
#[cfg(feature = "autodiff")]
use computegraph::fragment::FragmentBuilder;
#[cfg(feature = "autodiff")]
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
#[cfg(feature = "autodiff")]
use computegraph::OpEmitter;
use tenferro_einsum::Subscripts;
use tenferro_ops::ext_op::ExtensionOp;
#[cfg(feature = "autodiff")]
use tenferro_ops::ext_op::{register_extension_rule, ExtensionAdRule};
#[cfg(feature = "autodiff")]
use tenferro_ops::std_tensor_op::StdTensorOp;
#[cfg(feature = "autodiff")]
use tenferro_ops::ShapeGuardContext;
use tenferro_ops::SymDim;
#[cfg(feature = "autodiff")]
use tenferro_ops::{ExtensionRegistryError, ExtensionRuleSet};
use tenferro_runtime::{
    ExtensionExecutionContext, ExtensionExecutor, ExtensionRuntime, ExtensionRuntimeRegistryError,
};
#[cfg(not(feature = "autodiff"))]
use tenferro_tensor::TensorBackend;
use tenferro_tensor::{DType, Tensor};
#[cfg(feature = "autodiff")]
use tenferro_tensor::{TensorBackend, TensorScalar};

use crate::einsum::tropical_einsum_subscripts_with_argmax;
#[cfg(feature = "autodiff")]
use crate::einsum::TropicalArgmaxStep;
use crate::TropicalKind;

pub(crate) const TROPICAL_EINSUM_FAMILY_ID: &str = "tenferro-ext-tropical.einsum.v1";
#[cfg(feature = "autodiff")]
const TROPICAL_EINSUM_JVP_FAMILY_ID: &str = "tenferro-ext-tropical.einsum_jvp.v1";
#[cfg(feature = "autodiff")]
const TROPICAL_EINSUM_VJP_FAMILY_ID: &str = "tenferro-ext-tropical.einsum_vjp.v1";

#[derive(Debug)]
struct TropicalRuntime {
    family_id: &'static str,
}

impl<B: TensorBackend + 'static> ExtensionRuntime<B> for TropicalRuntime {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn execute(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[&Tensor],
        _ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        op.eager_execute(inputs)
    }
}

/// Register tropical extension runtimes on a graph or eager executor.
///
/// The runtime executor is intentionally thin: it delegates to each tropical
/// extension op's [`tenferro_runtime::extension::ExtensionOpTrait::eager_execute`].
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
    executor.registry_mut().register(Arc::new(TropicalRuntime {
        family_id: TROPICAL_EINSUM_FAMILY_ID,
    }))?;
    #[cfg(feature = "autodiff")]
    {
        executor.registry_mut().register(Arc::new(TropicalRuntime {
            family_id: TROPICAL_EINSUM_JVP_FAMILY_ID,
        }))?;
        executor.registry_mut().register(Arc::new(TropicalRuntime {
            family_id: TROPICAL_EINSUM_VJP_FAMILY_ID,
        }))?;
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

    fn n_inputs(&self) -> usize {
        2
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![infer_tropical_output_meta(
            &self.subscripts,
            input_dtypes,
            input_shapes,
        )]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
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

    fn n_inputs(&self) -> usize {
        2 + self.active_inputs.len()
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        assert_eq!(
            input_dtypes.len(),
            self.n_inputs(),
            "tropical einsum JVP expects {} inputs, got {}",
            self.n_inputs(),
            input_dtypes.len()
        );
        assert_eq!(
            input_shapes.len(),
            self.n_inputs(),
            "tropical einsum JVP expects {} input shapes, got {}",
            self.n_inputs(),
            input_shapes.len()
        );
        let primal =
            infer_tropical_output_meta(&self.subscripts, &input_dtypes[..2], &input_shapes[..2]);
        for &active in &self.active_inputs {
            let tangent_idx = 2 + self
                .active_inputs
                .iter()
                .position(|candidate| *candidate == active)
                .expect("active input is present");
            assert_eq!(
                input_dtypes[tangent_idx], input_dtypes[active],
                "tropical einsum JVP tangent dtype must match active primal input"
            );
        }
        vec![primal]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        if inputs.len() != self.n_inputs() {
            return Err(invalid_config(
                "tropical_einsum_jvp",
                format!("expected {} inputs, got {}", self.n_inputs(), inputs.len()),
            ));
        }
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
            dtype => Err(invalid_config(
                "tropical_einsum_jvp",
                format!("unsupported output dtype {dtype:?}"),
            )),
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

    fn n_inputs(&self) -> usize {
        3
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        assert_eq!(
            input_dtypes.len(),
            3,
            "tropical einsum VJP expects lhs, rhs, and cotangent inputs"
        );
        assert_eq!(
            input_shapes.len(),
            3,
            "tropical einsum VJP expects lhs, rhs, and cotangent shapes"
        );
        let _ =
            infer_tropical_output_meta(&self.subscripts, &input_dtypes[..2], &input_shapes[..2]);
        assert!(
            self.active_input < 2,
            "tropical einsum VJP active input must be 0 or 1"
        );
        assert_eq!(
            input_dtypes[2], input_dtypes[self.active_input],
            "tropical einsum VJP cotangent dtype must match active input"
        );
        vec![(
            input_dtypes[self.active_input],
            input_shapes[self.active_input].to_vec(),
        )]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        if inputs.len() != 3 {
            return Err(invalid_config(
                "tropical_einsum_vjp",
                format!("expected 3 inputs, got {}", inputs.len()),
            ));
        }
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
            dtype => Err(invalid_config(
                "tropical_einsum_vjp",
                format!("unsupported active input dtype {dtype:?}"),
            )),
        }
    }
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct TropicalEinsumAdRule;

#[cfg(feature = "autodiff")]
impl ExtensionAdRule for TropicalEinsumAdRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_FAMILY_ID
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut FragmentBuilder<StdTensorOp>,
        primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let op = downcast_primal_op(op, ADRuleKind::Linearize)?;
        validate_ad_supported(&op.subscripts, ADRuleKind::Linearize)?;
        let active_inputs: Vec<usize> = tangent_in
            .iter()
            .enumerate()
            .filter_map(|(idx, tangent)| tangent.is_some().then_some(idx))
            .collect();
        if active_inputs.is_empty() {
            return Ok(vec![None]);
        }

        let mut inputs = vec![
            ValRef::External(primal_in[0].clone()),
            ValRef::External(primal_in[1].clone()),
        ];
        for &active in &active_inputs {
            inputs.push(ValRef::Local(
                tangent_in[active].expect("active tangent is present"),
            ));
        }
        let active_mask = std::iter::repeat_n(false, 2)
            .chain(std::iter::repeat_n(true, active_inputs.len()))
            .collect();
        let out = builder.add_op(
            StdTensorOp::Extension(Arc::new(TropicalEinsumJvpOp::new(
                op.kind,
                op.subscripts.clone(),
                active_inputs,
            ))),
            inputs,
            OpMode::Linear { active_mask },
        );
        Ok(vec![Some(out[0])])
    }

    fn transpose_rule(
        &self,
        _op: &dyn ExtensionOp,
        _emitter: &mut dyn OpEmitter<StdTensorOp>,
        _cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        Err(ADRuleError::unsupported(
            "tropical einsum transpose is supported via its linearized JVP op",
            ADRuleKind::Transpose,
        ))
    }
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct TropicalEinsumJvpAdRule;

#[cfg(feature = "autodiff")]
impl ExtensionAdRule for TropicalEinsumJvpAdRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_JVP_FAMILY_ID
    }

    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        _tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        Err(ADRuleError::unsupported(
            TROPICAL_EINSUM_JVP_FAMILY_ID,
            ADRuleKind::Linearize,
        ))
    }

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<StdTensorOp>],
        mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let op = downcast_jvp_op(op, ADRuleKind::Transpose)?;
        validate_ad_supported(&op.subscripts, ADRuleKind::Transpose)?;
        let Some(ct) = cotangent_out.first().copied().flatten() else {
            return Ok(vec![None; op.n_inputs()]);
        };
        let active_mask = match mode {
            OpMode::Linear { active_mask } => active_mask,
            OpMode::Primal => return Ok(vec![None; op.n_inputs()]),
        };

        let mut result = vec![None; op.n_inputs()];
        for (active_pos, &active_input) in op.active_inputs.iter().enumerate() {
            let tangent_input_idx = 2 + active_pos;
            if !active_mask.get(tangent_input_idx).copied().unwrap_or(false) {
                continue;
            }
            let out = emitter.add_op(
                StdTensorOp::Extension(Arc::new(TropicalEinsumVjpOp::new(
                    op.kind,
                    op.subscripts.clone(),
                    active_input,
                ))),
                vec![inputs[0].clone(), inputs[1].clone(), ValRef::Local(ct)],
                OpMode::Linear {
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
/// assert!(rules.is_rule_registered("tenferro-ext-tropical.einsum.v1"));
/// assert!(rules.is_rule_registered("tenferro-ext-tropical.einsum_jvp.v1"));
/// ```
#[cfg(feature = "autodiff")]
pub fn tropical_ad_rules() -> Result<ExtensionRuleSet, ExtensionRegistryError> {
    let mut rules = ExtensionRuleSet::new();
    rules.register_rule(Arc::new(TropicalEinsumAdRule))?;
    rules.register_rule(Arc::new(TropicalEinsumJvpAdRule))?;
    Ok(rules)
}

/// Register tropical traced einsum AD rules in the process-global registry.
///
/// Prefer [`tropical_ad_rules`] with an explicit `tenferro_ad::AdContext` for
/// tests and applications. This helper exists for compatibility with global
/// extension-rule lookup and treats duplicate registration as success.
///
/// # Errors
///
/// Returns [`tenferro_ops::ExtensionRegistryError`] if a malformed family id is
/// detected while registering the rules.
///
/// # Examples
///
/// ```
/// tenferro_ext_tropical::register_tropical_ad_rules().unwrap();
/// ```
#[cfg(feature = "autodiff")]
pub fn register_tropical_ad_rules() -> Result<(), ExtensionRegistryError> {
    register_rule_idempotent(Arc::new(TropicalEinsumAdRule))?;
    register_rule_idempotent(Arc::new(TropicalEinsumJvpAdRule))?;
    Ok(())
}

#[cfg(feature = "autodiff")]
fn register_rule_idempotent(rule: Arc<dyn ExtensionAdRule>) -> Result<(), ExtensionRegistryError> {
    let family_id = rule.family_id();
    match register_extension_rule(rule) {
        Ok(()) => Ok(()),
        Err(ExtensionRegistryError::DuplicateRule {
            family_id: duplicate,
        }) if duplicate == family_id => Ok(()),
        Err(err) => Err(err),
    }
}

fn infer_tropical_output_meta(
    subscripts: &Subscripts,
    input_dtypes: &[DType],
    input_shapes: &[&[SymDim]],
) -> (DType, Vec<SymDim>) {
    assert_eq!(
        input_shapes.len(),
        2,
        "tropical einsum extension supports exactly two inputs, got {}",
        input_shapes.len()
    );
    assert_eq!(
        input_dtypes.len(),
        input_shapes.len(),
        "tropical einsum extension expects dtype and shape arity to match"
    );
    assert_eq!(
        subscripts.inputs.len(),
        2,
        "tropical einsum subscripts must describe exactly two inputs"
    );
    assert_eq!(
        input_dtypes[0], input_dtypes[1],
        "tropical einsum input dtypes must match"
    );

    let mut label_dims: HashMap<u32, SymDim> = HashMap::new();
    for (labels, shape) in subscripts.inputs.iter().zip(input_shapes.iter()) {
        assert_eq!(
            labels.len(),
            shape.len(),
            "tropical einsum input rank mismatch: labels={}, shape={}",
            labels.len(),
            shape.len()
        );
        for (&label, dim) in labels.iter().zip(shape.iter()) {
            if let Some(existing) = label_dims.get(&label) {
                if let (Some(lhs), Some(rhs)) = (existing.constant_value(), dim.constant_value()) {
                    assert_eq!(
                        lhs, rhs,
                        "tropical einsum label {label} has inconsistent concrete sizes {lhs} vs {rhs}"
                    );
                }
            } else {
                label_dims.insert(label, dim.clone());
            }
        }
    }
    let output_shape = subscripts
        .output
        .iter()
        .map(|label| {
            label_dims.get(label).cloned().unwrap_or_else(|| {
                panic!("unknown size for label {label} in tropical einsum output")
            })
        })
        .collect();
    (input_dtypes[0], output_shape)
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
        let tangent = typed_slice::<T>(inputs[2 + active_pos], "tropical_einsum_jvp tangent")?;
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
    Ok(Tensor::from_vec_col_major(output_shape, out))
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
    let cotangent = typed_slice::<T>(inputs[2], "tropical_einsum_vjp cotangent")?;
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
    Ok(Tensor::from_vec_col_major(active_shape, out))
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
fn typed_slice<'a, T>(tensor: &'a Tensor, op: &'static str) -> tenferro_tensor::Result<&'a [T]>
where
    T: TensorScalar,
{
    tensor.as_slice::<T>().ok_or_else(|| {
        invalid_config(
            op,
            format!("expected compact host {:?} tensor", tensor.dtype()),
        )
    })
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

#[cfg(feature = "autodiff")]
fn invalid_config(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::InvalidConfig {
        op,
        message: message.into(),
    }
}
