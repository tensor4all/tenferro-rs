use std::any::Any;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use tenferro_ad::semantic_extension::{
    AdValue, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRegistryError,
    SemanticExtensionRuleSet, SemanticLinearTransposeRequest, SemanticLinearTransposeRule,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
    SemanticPrimalVjpRequest, SemanticPrimalVjpRule,
};
use tenferro_ops::ext_op::{
    ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp, HostReference,
};
use tenferro_ops::SymDim;
use tenferro_runtime::extension::{
    apply, ExtensionExecutor, ExtensionRuntimeRegistryError, HostReferenceRuntime,
};
#[cfg(feature = "autodiff")]
use tenferro_runtime::program::{ProgramValue, SemanticProgramBuilder};
use tenferro_runtime::{Error as RuntimeError, Result as RuntimeResult};
use tenferro_tensor::{DType, Error, Result, Tensor, TensorBackend};

use crate::sparse::{
    coordinates_tensor, validate_traced_values, validate_value_tensor, SparseCooTracedTensor,
};

const FAMILY_ID: &str = "tenferro-ext-sparse.matmul.v1";
#[cfg(feature = "autodiff")]
const JVP_FAMILY_ID: &str = "tenferro-ext-sparse.matmul_jvp.v1";
#[cfg(feature = "autodiff")]
const VJP_FAMILY_ID: &str = "tenferro-ext-sparse.matmul_vjp.v1";
const OP: &str = "tenferro-ext-sparse";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Contribution {
    out: usize,
    left: usize,
    right: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SparseMatmulPlan {
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
    left_nnz: usize,
    right_nnz: usize,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) output_coords: Vec<[usize; 2]>,
    contributions: Vec<Contribution>,
}

impl SparseMatmulPlan {
    pub(crate) fn new(
        left_shape: &[usize],
        left_entries: &[[usize; 2]],
        right_shape: &[usize],
        right_entries: &[[usize; 2]],
    ) -> Result<Self> {
        if left_shape.len() != 2 {
            return Err(tenferro_tensor::Error::rank_mismatch(
                OP,
                2,
                left_shape.len(),
            ));
        }
        if right_shape.len() != 2 {
            return Err(tenferro_tensor::Error::rank_mismatch(
                OP,
                2,
                right_shape.len(),
            ));
        }
        if left_shape[1] != right_shape[0] {
            return Err(tenferro_tensor::Error::shape_mismatch(
                OP,
                vec![left_shape[1]],
                vec![right_shape[0]],
            ));
        }

        let mut raw = Vec::new();
        let mut output_coords = Vec::<[usize; 2]>::new();
        for (left_idx, &[row, contract_left]) in left_entries.iter().enumerate() {
            for (right_idx, &[contract_right, col]) in right_entries.iter().enumerate() {
                if contract_left != contract_right {
                    continue;
                }
                let coord = [row, col];
                raw.push((coord, left_idx, right_idx));
                if !output_coords.contains(&coord) {
                    output_coords.push(coord);
                }
            }
        }
        output_coords.sort_by_key(|&[row, col]| (col, row));
        let output_index: HashMap<[usize; 2], usize> = output_coords
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, coord)| (coord, idx))
            .collect();
        let contributions = raw
            .into_iter()
            .map(|(coord, left, right)| Contribution {
                out: output_index[&coord],
                left,
                right,
            })
            .collect();

        Ok(Self {
            left_shape: left_shape.to_vec(),
            right_shape: right_shape.to_vec(),
            left_nnz: left_entries.len(),
            right_nnz: right_entries.len(),
            output_shape: vec![left_shape[0], right_shape[1]],
            output_coords,
            contributions,
        })
    }

    fn output_nnz(&self) -> usize {
        self.output_coords.len()
    }

    fn left_nnz(&self) -> usize {
        self.left_nnz
    }

    fn right_nnz(&self) -> usize {
        self.right_nnz
    }
}

/// Register sparse extension runtimes on a graph or eager executor.
///
/// # Errors
///
/// Returns [`ExtensionRuntimeRegistryError::MalformedFamilyId`] when a
/// runtime family identifier is invalid, or
/// [`ExtensionRuntimeRegistryError::PoisonedLock`] if registry state was
/// poisoned while registering the family.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::GraphExecutor;
///
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_ext_sparse::register_runtime).unwrap();
/// assert!(executor
///     .extension_executor()
///     .registry()
///     .contains("tenferro-ext-sparse.matmul.v1"));
/// ```
pub fn register_runtime<B: TensorBackend + 'static>(
    executor: &mut ExtensionExecutor<B>,
) -> std::result::Result<(), ExtensionRuntimeRegistryError> {
    executor
        .registry_mut()
        .register(Arc::new(HostReferenceRuntime::<B>::new(FAMILY_ID)))?;
    #[cfg(feature = "autodiff")]
    {
        executor
            .registry_mut()
            .register(Arc::new(HostReferenceRuntime::<B>::new(JVP_FAMILY_ID)))?;
        executor
            .registry_mut()
            .register(Arc::new(HostReferenceRuntime::<B>::new(VJP_FAMILY_ID)))?;
    }
    Ok(())
}

/// Multiply two traced sparse COO matrices with a fixed sparse pattern.
///
/// # Errors
///
/// Returns an error when sparse shapes are incompatible or extension graph
/// construction fails.
///
/// # Examples
///
/// ```
/// use tenferro_ext_sparse::{sparse_matmul, SparseCooTracedTensor};
/// use tenferro_runtime::TracedTensor;
/// use tenferro_tensor::Tensor;
///
/// let coords = Tensor::from_vec_col_major(vec![2, 1], vec![0_i64, 0])?;
/// let a = SparseCooTracedTensor::from_parts(
///     vec![1, 1],
///     coords.clone(),
///     TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64])?,
/// )?;
/// let b = SparseCooTracedTensor::from_parts(
///     vec![1, 1],
///     coords,
///     TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64])?,
/// )?;
/// let out = sparse_matmul(&a, &b)?;
/// assert_eq!(out.shape(), &[1, 1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn sparse_matmul(
    lhs: &SparseCooTracedTensor,
    rhs: &SparseCooTracedTensor,
) -> RuntimeResult<SparseCooTracedTensor> {
    let plan = SparseMatmulPlan::new(lhs.shape(), lhs.entries(), rhs.shape(), rhs.entries())
        .map_err(RuntimeError::from)?;
    validate_traced_values(lhs.values(), plan.left_nnz())?;
    validate_traced_values(rhs.values(), plan.right_nnz())?;
    let outputs = apply(
        Arc::new(SparseMatmulOp { plan: plan.clone() }),
        &[lhs.values(), rhs.values()],
    )?;
    let [values] = outputs
        .try_into()
        .map_err(|_| RuntimeError::Internal("sparse matmul returned wrong output count".into()))?;
    SparseCooTracedTensor::from_parts(
        plan.output_shape.clone(),
        coordinates_tensor(&plan.output_coords).map_err(RuntimeError::from)?,
        values,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SparseMatmulOp {
    plan: SparseMatmulPlan,
}

impl ExtensionOp for SparseMatmulOp {
    fn family_id(&self) -> &'static str {
        FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_plan(&self.plan, hasher);
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

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtypes = [ctx.input_dtype(0)?, ctx.input_dtype(1)?];
        let input_shapes = [ctx.input_shape(0)?, ctx.input_shape(1)?];
        validate_primal_meta(&input_dtypes, &input_shapes)?;
        require_primal_shape_constraints(ctx, &self.plan)?;
        Ok(vec![(
            input_dtypes[0],
            vec![SymDim::from(self.plan.output_nnz())],
        )])
    }

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for SparseMatmulOp {
    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        validate_primal_inputs(&self.plan, inputs)?;
        apply_sparse_matmul(&self.plan, inputs[0], inputs[1]).map(|tensor| vec![tensor])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct SparseMatmulJvpOp {
    plan: SparseMatmulPlan,
    active_inputs: Vec<usize>,
}

#[cfg(feature = "autodiff")]
impl ExtensionOp for SparseMatmulJvpOp {
    fn family_id(&self) -> &'static str {
        JVP_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_plan(&self.plan, hasher);
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

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtypes = (0..self.input_count())
            .map(|input| ctx.input_dtype(input))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let input_shapes = (0..self.input_count())
            .map(|input| ctx.input_shape(input).map(<[_]>::to_vec))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let primal_shapes = [&input_shapes[0][..], &input_shapes[1][..]];
        validate_primal_meta(&input_dtypes[..2], &primal_shapes)?;
        require_primal_shape_constraints(ctx, &self.plan)?;
        for (active_pos, &active) in self.active_inputs.iter().enumerate() {
            if active >= 2 {
                return Err(invalid(format!("invalid active sparse input {active}")));
            }
            let tangent_idx = 2 + active_pos;
            if input_dtypes[tangent_idx] != input_dtypes[active] {
                return Err(Error::dtype_mismatch(
                    OP,
                    input_dtypes[active],
                    input_dtypes[tangent_idx],
                ));
            }
            if !is_rank1_shape(&input_shapes[tangent_idx]) {
                return Err(Error::rank_mismatch(OP, 1, input_shapes[tangent_idx].len()));
            }
            ctx.require_same_shape(tangent_idx, active)?;
        }
        Ok(vec![(
            input_dtypes[0],
            vec![SymDim::from(self.plan.output_nnz())],
        )])
    }

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

#[cfg(feature = "autodiff")]
impl HostReference for SparseMatmulJvpOp {
    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        validate_jvp_inputs(&self.plan, inputs, &self.active_inputs)?;
        execute_jvp(&self.plan, inputs, &self.active_inputs).map(|tensor| vec![tensor])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct SparseMatmulVjpOp {
    plan: SparseMatmulPlan,
    active_input: usize,
}

#[cfg(feature = "autodiff")]
impl ExtensionOp for SparseMatmulVjpOp {
    fn family_id(&self) -> &'static str {
        VJP_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_plan(&self.plan, hasher);
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

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> Result<Vec<(DType, Vec<SymDim>)>> {
        if self.active_input >= 2 {
            return Err(invalid("invalid sparse VJP metadata"));
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
        let primal_shapes = [&input_shapes[0][..], &input_shapes[1][..]];
        validate_primal_meta(&input_dtypes[..2], &primal_shapes)?;
        require_primal_shape_constraints(ctx, &self.plan)?;
        if input_dtypes[2] != input_dtypes[self.active_input] {
            return Err(Error::dtype_mismatch(
                OP,
                input_dtypes[self.active_input],
                input_dtypes[2],
            ));
        }
        if !is_rank1_shape(&input_shapes[2]) {
            return Err(Error::rank_mismatch(OP, 1, input_shapes[2].len()));
        }
        ctx.require_equal(ctx.input_axis(2, 0)?, SymDim::from(self.plan.output_nnz()))?;
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
impl HostReference for SparseMatmulVjpOp {
    fn execute(&self, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        validate_vjp_inputs(&self.plan, inputs, self.active_input)?;
        execute_vjp(&self.plan, inputs, self.active_input).map(|tensor| vec![tensor])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct SparseMatmulAdRule;

#[cfg(feature = "autodiff")]
impl SemanticLinearizeRule for SparseMatmulAdRule {
    fn family_id(&self) -> &'static str {
        FAMILY_ID
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<SemanticLinearizeResult, SemanticAdError> {
        let op = semantic_matmul_op(request.op(), SemanticAdRuleRole::Linearize)?;
        if !request.active_outputs()[0] {
            return Ok(SemanticLinearizeResult::new([AdValue::Absent], []));
        }
        let active_inputs: Vec<usize> = request
            .tangent_inputs()
            .iter()
            .enumerate()
            .filter_map(|(index, tangent)| matches!(tangent, AdValue::Value(_)).then_some(index))
            .collect();
        if active_inputs.is_empty() {
            return Ok(SemanticLinearizeResult::new([AdValue::Absent], []));
        }
        let mut inputs = request.primal_inputs().to_vec();
        inputs.extend(active_inputs.iter().filter_map(|&index| {
            request
                .tangent_inputs()
                .get(index)
                .copied()
                .and_then(AdValue::value)
        }));
        let tangent = builder.add_extension(
            Arc::new(SparseMatmulJvpOp {
                plan: op.plan.clone(),
                active_inputs,
            }),
            &inputs,
        )?[0];
        Ok(SemanticLinearizeResult::new([AdValue::Value(tangent)], []))
    }
}

#[cfg(feature = "autodiff")]
impl SemanticLinearTransposeRule for SparseMatmulAdRule {
    fn family_id(&self) -> &'static str {
        FAMILY_ID
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        semantic_sparse_vjp(
            request.op(),
            request.primal_inputs(),
            request.cotangent_outputs()[0],
            request.active_inputs(),
            builder,
            SemanticAdRuleRole::LinearTranspose,
        )
    }
}

#[cfg(feature = "autodiff")]
impl SemanticPrimalVjpRule for SparseMatmulAdRule {
    fn family_id(&self) -> &'static str {
        FAMILY_ID
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        semantic_sparse_vjp(
            request.op(),
            request.primal_inputs(),
            request.cotangent_outputs()[0],
            request.active_inputs(),
            builder,
            SemanticAdRuleRole::PrimalVjp,
        )
    }
}

/// Build sparse extension semantic-program AD rules.
///
/// # Errors
///
/// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] if the sparse
/// family identifier is invalid, or
/// [`SemanticExtensionRegistryError::DuplicateRule`] if a semantic rule role
/// is already registered.
///
/// # Examples
///
/// ```
/// let rules = tenferro_ext_sparse::sparse_semantic_ad_rules().unwrap();
/// assert!(rules
///     .lookup_linearize("tenferro-ext-sparse.matmul.v1")
///     .is_some());
/// assert!(rules
///     .lookup_linear_transpose("tenferro-ext-sparse.matmul.v1")
///     .is_some());
/// assert!(rules
///     .lookup_primal_vjp("tenferro-ext-sparse.matmul.v1")
///     .is_some());
/// ```
#[cfg(feature = "autodiff")]
pub fn sparse_semantic_ad_rules(
) -> std::result::Result<SemanticExtensionRuleSet, SemanticExtensionRegistryError> {
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(SparseMatmulAdRule))?
        .with_linear_transpose(Arc::new(SparseMatmulAdRule))?
        .with_primal_vjp(Arc::new(SparseMatmulAdRule))
}

#[cfg(feature = "autodiff")]
fn semantic_sparse_vjp(
    op: &dyn ExtensionOp,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
    let op = semantic_matmul_op(op, role)?;
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(vec![AdValue::Absent; op.input_count()].into_boxed_slice());
    };
    active_inputs
        .iter()
        .copied()
        .enumerate()
        .map(|(active_input, active)| {
            if !active {
                return Ok(AdValue::Absent);
            }
            let mut inputs = primal_inputs.to_vec();
            inputs.push(cotangent);
            let value = builder.add_extension(
                Arc::new(SparseMatmulVjpOp {
                    plan: op.plan.clone(),
                    active_input,
                }),
                &inputs,
            )?[0];
            Ok(AdValue::Value(value))
        })
        .collect()
}

#[cfg(feature = "autodiff")]
fn semantic_matmul_op(
    op: &dyn ExtensionOp,
    role: SemanticAdRuleRole,
) -> std::result::Result<&SparseMatmulOp, SemanticAdError> {
    op.as_any()
        .downcast_ref::<SparseMatmulOp>()
        .ok_or_else(|| SemanticAdError::Unsupported {
            family_id: FAMILY_ID,
            role,
            message: "sparse matmul semantic AD received an incompatible payload".into(),
        })
}

pub(crate) fn apply_sparse_matmul(
    plan: &SparseMatmulPlan,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor> {
    validate_primal_inputs(plan, &[lhs, rhs])?;
    let lhs = lhs.as_slice::<f64>()?;
    let rhs = rhs.as_slice::<f64>()?;
    let mut output = vec![0.0_f64; plan.output_nnz()];
    for contribution in &plan.contributions {
        output[contribution.out] += lhs[contribution.left] * rhs[contribution.right];
    }
    Tensor::from_vec_col_major(vec![plan.output_nnz()], output)
}

#[cfg(feature = "autodiff")]
fn execute_jvp(
    plan: &SparseMatmulPlan,
    inputs: &[&Tensor],
    active_inputs: &[usize],
) -> Result<Tensor> {
    let lhs = inputs[0].as_slice::<f64>()?;
    let rhs = inputs[1].as_slice::<f64>()?;
    let mut output = vec![0.0_f64; plan.output_nnz()];
    for (active_pos, &active) in active_inputs.iter().enumerate() {
        let tangent = inputs[2 + active_pos].as_slice::<f64>()?;
        for contribution in &plan.contributions {
            output[contribution.out] += match active {
                0 => tangent[contribution.left] * rhs[contribution.right],
                1 => lhs[contribution.left] * tangent[contribution.right],
                _ => return Err(invalid(format!("invalid active sparse input {active}"))),
            };
        }
    }
    Tensor::from_vec_col_major(vec![plan.output_nnz()], output)
}

#[cfg(feature = "autodiff")]
fn execute_vjp(plan: &SparseMatmulPlan, inputs: &[&Tensor], active_input: usize) -> Result<Tensor> {
    let lhs = inputs[0].as_slice::<f64>()?;
    let rhs = inputs[1].as_slice::<f64>()?;
    let cotangent = inputs[2].as_slice::<f64>()?;
    let mut output = match active_input {
        0 => vec![0.0_f64; plan.left_nnz()],
        1 => vec![0.0_f64; plan.right_nnz()],
        _ => {
            return Err(invalid(format!(
                "invalid active sparse input {active_input}"
            )))
        }
    };
    for contribution in &plan.contributions {
        match active_input {
            0 => output[contribution.left] += cotangent[contribution.out] * rhs[contribution.right],
            1 => output[contribution.right] += lhs[contribution.left] * cotangent[contribution.out],
            _ => unreachable!(),
        }
    }
    Tensor::from_vec_col_major(vec![output.len()], output)
}

fn validate_primal_meta(input_dtypes: &[DType], input_shapes: &[&[SymDim]]) -> Result<()> {
    if input_dtypes.len() != 2 || input_shapes.len() != 2 {
        return Err(invalid(format!(
            "sparse matmul expected 2 inputs, got dtypes={} shapes={}",
            input_dtypes.len(),
            input_shapes.len()
        )));
    }
    if input_dtypes[0] != DType::F64 {
        return Err(Error::dtype_mismatch(OP, DType::F64, input_dtypes[0]));
    }
    if input_dtypes[1] != DType::F64 {
        return Err(Error::dtype_mismatch(OP, DType::F64, input_dtypes[1]));
    }
    if !is_rank1_shape(input_shapes[0]) {
        return Err(Error::rank_mismatch(OP, 1, input_shapes[0].len()));
    }
    if !is_rank1_shape(input_shapes[1]) {
        return Err(Error::rank_mismatch(OP, 1, input_shapes[1].len()));
    }
    Ok(())
}

fn require_primal_shape_constraints(
    ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    plan: &SparseMatmulPlan,
) -> Result<()> {
    ctx.require_equal(ctx.input_axis(0, 0)?, SymDim::from(plan.left_nnz()))?;
    ctx.require_equal(ctx.input_axis(1, 0)?, SymDim::from(plan.right_nnz()))?;
    Ok(())
}

fn is_rank1_shape(shape: &[SymDim]) -> bool {
    shape.len() == 1
}

fn validate_primal_inputs(plan: &SparseMatmulPlan, inputs: &[&Tensor]) -> Result<()> {
    if inputs.len() != 2 {
        return Err(invalid(format!(
            "sparse matmul expected 2 inputs, got {}",
            inputs.len()
        )));
    }
    validate_value_tensor(inputs[0], plan.left_nnz())?;
    validate_value_tensor(inputs[1], plan.right_nnz())?;
    Ok(())
}

#[cfg(feature = "autodiff")]
fn validate_jvp_inputs(
    plan: &SparseMatmulPlan,
    inputs: &[&Tensor],
    active_inputs: &[usize],
) -> Result<()> {
    let expected = 2 + active_inputs.len();
    if inputs.len() != expected {
        return Err(invalid(format!(
            "sparse JVP expected {expected} inputs, got {}",
            inputs.len()
        )));
    }
    validate_primal_inputs(plan, &inputs[..2])?;
    for (active_pos, &active) in active_inputs.iter().enumerate() {
        let expected_nnz = match active {
            0 => plan.left_nnz(),
            1 => plan.right_nnz(),
            _ => return Err(invalid(format!("invalid active sparse input {active}"))),
        };
        validate_value_tensor(inputs[2 + active_pos], expected_nnz)?;
    }
    Ok(())
}

#[cfg(feature = "autodiff")]
fn validate_vjp_inputs(
    plan: &SparseMatmulPlan,
    inputs: &[&Tensor],
    active_input: usize,
) -> Result<()> {
    if inputs.len() != 3 {
        return Err(invalid(format!(
            "sparse VJP expected 3 inputs, got {}",
            inputs.len()
        )));
    }
    if active_input >= 2 {
        return Err(invalid(format!(
            "invalid active sparse input {active_input}"
        )));
    }
    validate_primal_inputs(plan, &inputs[..2])?;
    validate_value_tensor(inputs[2], plan.output_nnz())
}

fn hash_plan(plan: &SparseMatmulPlan, hasher: &mut dyn Hasher) {
    for &dim in &plan.left_shape {
        hasher.write_usize(dim);
    }
    hasher.write_u8(0xff);
    for &dim in &plan.right_shape {
        hasher.write_usize(dim);
    }
    hasher.write_u8(0xfe);
    for &[row, col] in &plan.output_coords {
        hasher.write_usize(row);
        hasher.write_usize(col);
    }
    hasher.write_u8(0xfd);
    for contribution in &plan.contributions {
        hasher.write_usize(contribution.out);
        hasher.write_usize(contribution.left);
        hasher.write_usize(contribution.right);
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::invalid_argument(OP, "configuration", message)
}

#[cfg(all(test, feature = "autodiff"))]
mod tests;
