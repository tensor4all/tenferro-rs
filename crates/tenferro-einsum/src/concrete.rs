//! Concrete einsum bodies and the prepared plan.
//!
//! The public grammar (`EinsumOps`, `TensordotOps`, the plan's
//! `prepare`/`execute`/`execute_into`) lives in `crate::api` and forwards to
//! the crate-internal entries here.

use tenferro_tensor::{DType, DotGeneralAccumulation, Engine, Tensor, TensorRead, TensorWrite};
use tinyvec::TinyVec;

use crate::binary_dot::BinaryDotOperandOrder;
use crate::eager::{
    binary_dot_config_for_into, binary_dot_plan_for_shapes, eager_einsum_exec,
    eager_einsum_exec_read, eager_einsum_exec_read_into, eager_einsum_exec_read_into_accum,
    eager_einsum_read_subscripts_on_session, eager_einsum_subscripts_on_session,
    execute_binary_dot_read_into, execute_binary_dot_read_into_accum, execute_binary_dot_shortcut,
    plan_subscripts, ConjugatedOperands,
};
use crate::ellipsis::resolve_einsum_notation;
use crate::TensorDotAxes;
use crate::{
    parse_einsum_notation, ContractionTree, EinsumNotation, EinsumSubscripts, Error, Result,
    Subscripts,
};

const TENSOR_EINSUM_INTO_OP: &str = "einsum_into";
const TENSOR_READ_EINSUM_INTO_OP: &str = "einsum_into";
const PLAN_EXECUTE_OP: &str = "ConcreteEinsumPlan::execute";

/// Tensordot of two owned erased tensors (the former `Tensor` entry).
pub(crate) fn tensordot_owned(
    lhs: &Tensor,
    rhs: &Tensor,
    axes: TensorDotAxes<'_>,
    session: &mut dyn Engine,
) -> Result<Tensor> {
    let config = crate::tensordot::dot_general_config(axes, lhs.shape().len(), rhs.shape().len())?;
    crate::tensordot::validate_concrete_contract_dims(lhs.shape(), rhs.shape(), &config)?;
    session
        .dot_general_read(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            &config,
        )
        .map_err(Error::from)
}

/// Tensordot of two borrowed reads (the former typed entry, dtype-erased).
pub(crate) fn tensordot_reads(
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    axes: TensorDotAxes<'_>,
    session: &mut dyn Engine,
) -> Result<Tensor> {
    let config = crate::tensordot::dot_general_config(axes, lhs.shape().len(), rhs.shape().len())?;
    crate::tensordot::validate_concrete_contract_dims(lhs.shape(), rhs.shape(), &config)?;
    session
        .dot_general_read(lhs, rhs, &config)
        .map_err(Error::from)
}

pub(crate) trait OwnedEinsum {
    fn einsum(&self, subscripts: &str, session: &mut dyn Engine) -> Result<Tensor>;

    fn einsum_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn Engine,
    ) -> Result<Tensor>;

    fn einsum_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn Engine,
    ) -> Result<Tensor>;
}

impl OwnedEinsum for [&Tensor] {
    fn einsum(&self, subscripts: &str, session: &mut dyn Engine) -> Result<Tensor> {
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_notation(&notation, session)
    }

    fn einsum_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn Engine,
    ) -> Result<Tensor> {
        let subscripts = resolve_tensor_notation(self, notation)?;
        eager_einsum_subscripts_on_session(session, self, &subscripts).map_err(Error::from)
    }

    fn einsum_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn Engine,
    ) -> Result<Tensor> {
        let subscripts = Subscripts::from(subscripts);
        eager_einsum_subscripts_on_session(session, self, &subscripts).map_err(Error::from)
    }
}

pub(crate) trait OwnedEinsumInto {
    fn einsum_into(
        &self,
        subscripts: &str,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    fn einsum_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    fn einsum_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()>;
}

impl OwnedEinsumInto for [&Tensor] {
    fn einsum_into(
        &self,
        subscripts: &str,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        if let ([lhs, rhs], Some((a, b, c))) = (self, parse_fast_ascii_binary_labels(subscripts)) {
            let reads = [TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs)];
            if let Some((order, config)) = read_binary_dot_config_for_labels(&reads, a, b, c, &out)
            {
                return execute_binary_dot_config_read_into(session, &reads, order, &config, out);
            }
        }
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_into_notation(&notation, session, out)
    }

    fn einsum_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let subscripts = resolve_tensor_notation(self, notation)?;
        tensor_einsum_into_subscripts(session, self, &subscripts, out, TENSOR_EINSUM_INTO_OP)
    }

    fn einsum_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        let subscripts = Subscripts::from(subscripts);
        tensor_einsum_into_subscripts(session, self, &subscripts, out, TENSOR_EINSUM_INTO_OP)
    }
}

pub(crate) trait ReadEinsum {
    fn einsum_read(&self, subscripts: &str, session: &mut dyn Engine) -> Result<Tensor>;

    fn einsum_read_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn Engine,
    ) -> Result<Tensor>;

    fn einsum_read_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn Engine,
    ) -> Result<Tensor>;
}

impl<'a> ReadEinsum for [TensorRead<'a>] {
    fn einsum_read(&self, subscripts: &str, session: &mut dyn Engine) -> Result<Tensor> {
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_read_notation(&notation, session)
    }

    fn einsum_read_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn Engine,
    ) -> Result<Tensor> {
        let subscripts = resolve_read_notation(self, notation)?;
        eager_einsum_read_subscripts_on_session(session, self, &subscripts).map_err(Error::from)
    }

    fn einsum_read_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn Engine,
    ) -> Result<Tensor> {
        let subscripts = Subscripts::from(subscripts);
        eager_einsum_read_subscripts_on_session(session, self, &subscripts).map_err(Error::from)
    }
}

pub(crate) trait ReadEinsumInto {
    fn einsum_read_into(
        &self,
        subscripts: &str,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    fn einsum_read_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()>;

    fn einsum_read_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()>;
}

impl<'a> ReadEinsumInto for [TensorRead<'a>] {
    fn einsum_read_into(
        &self,
        subscripts: &str,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        if let Some((lhs, rhs, output)) = parse_fast_ascii_binary_labels(subscripts) {
            if let Some((order, config)) =
                read_binary_dot_config_for_labels(self, lhs, rhs, output, &out)
            {
                return execute_binary_dot_config_read_into(session, self, order, &config, out);
            }
        }
        let notation = parse_einsum_notation(subscripts)?;
        self.einsum_read_into_notation(&notation, session, out)
    }

    fn einsum_read_into_notation(
        &self,
        notation: &EinsumNotation,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        if let Some([lhs, rhs, output]) = borrowed_notation_labels(notation) {
            if let Some((order, config)) =
                read_binary_dot_config_for_labels(self, &lhs, &rhs, &output, &out)
            {
                return execute_binary_dot_config_read_into(session, self, order, &config, out);
            }
        }
        let subscripts = resolve_read_notation(self, notation)?;
        tensor_read_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out,
            TENSOR_READ_EINSUM_INTO_OP,
        )
    }

    fn einsum_read_into_subscripts(
        &self,
        subscripts: &EinsumSubscripts,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()> {
        if let [lhs, rhs] = subscripts.inputs.as_slice() {
            if let Some((order, config)) =
                read_binary_dot_config_for_labels(self, lhs, rhs, &subscripts.output, &out)
            {
                return execute_binary_dot_config_read_into(session, self, order, &config, out);
            }
        }
        let subscripts = Subscripts::from(subscripts);
        tensor_read_einsum_into_subscripts(
            session,
            self,
            &subscripts,
            out,
            TENSOR_READ_EINSUM_INTO_OP,
        )
    }
}

/// Prepared concrete einsum plan for repeated executions with fixed input
/// dtype and shape metadata.
///
/// Preparing a plan parses and optimizes the contraction tree once. Execution
/// validates the later inputs against the prepared dtype and shape contract,
/// then runs the stored tree without re-planning.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuEngine;
/// use tenferro_einsum::ConcreteEinsumPlan;
/// use tenferro_tensor::Tensor;
///
/// let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
/// let rhs = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
/// let plan = ConcreteEinsumPlan::prepare(&[&lhs, &rhs], "ij,jk->ik", &Default::default())?;
///
/// let mut backend = CpuEngine::serial();
/// let out = plan.execute(&[&lhs, &rhs], &mut backend)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct ConcreteEinsumPlan {
    tree: ContractionTree,
    inputs: Vec<ConcreteEinsumInputSpec>,
    output_shape: Vec<usize>,
    binary_dot: Option<crate::binary_dot::BinaryDotPlan>,
    cpu_plan: crate::cpu_concrete::Cache,
    /// Per-operand conjugation chosen at preparation (#2552 `EinsumOptions::conj`).
    conj: crate::api::ConjMask,
}

impl ConcreteEinsumPlan {
    /// Release this plan's retained CPU numerical descriptors.
    ///
    /// The semantic contraction tree and the owning backend's workspaces are unchanged.
    ///
    /// # Examples
    /// ```
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::Tensor;
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
    /// let mut plan =
    ///     ConcreteEinsumPlan::prepare(&[&a, &a, &a], "ij,jk,kl->il", &Default::default())?;
    /// plan.clear_cpu_cache();
    /// assert_eq!(plan.cpu_cache_stats()?.entries, 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn clear_cpu_cache(&mut self) {
        self.cpu_plan.clear();
    }

    /// Inspect this plan's bounded CPU numerical descriptor cache.
    ///
    /// Counts exclude backend-owned scratch and allocator RSS. Retained bytes
    /// currently describe host layout descriptors and opaque lower plan headers.
    ///
    /// # Examples
    /// ```
    /// use tenferro_einsum::ConcreteEinsumPlan;
    /// use tenferro_tensor::Tensor;
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
    /// let plan =
    ///     ConcreteEinsumPlan::prepare(&[&a, &a, &a], "ij,jk,kl->il", &Default::default())?;
    /// assert_eq!(plan.cpu_cache_stats()?.entries, 0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error`] when no CPU plan cache exists for this plan.
    pub fn cpu_cache_stats(&self) -> Result<tenferro_tensor::CacheStats> {
        self.cpu_plan.stats().map_err(Error::from)
    }

    pub(crate) fn prepare_owned_subscripts<'a, I>(
        inputs: I,
        subscripts: &EinsumSubscripts,
    ) -> Result<Self>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let subscripts = Subscripts::from(subscripts);
        Self::prepare_subscripts_internal(input_specs(inputs.as_ref()), &subscripts)
    }

    pub(crate) fn prepare_read_subscripts<'a, I>(
        inputs: I,
        subscripts: &EinsumSubscripts,
    ) -> Result<Self>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let subscripts = Subscripts::from(subscripts);
        Self::prepare_subscripts_internal(read_input_specs(inputs.as_ref()), &subscripts)
    }

    /// Number of binary contraction steps in the prepared tree (diagnostics).
    pub(crate) fn step_count(&self) -> usize {
        self.tree.step_count()
    }

    pub(crate) fn run_owned<'a, I>(&self, inputs: I, session: &mut dyn Engine) -> Result<Tensor>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let inputs = inputs.as_ref();
        self.validate_tensor_inputs(inputs, PLAN_EXECUTE_OP)?;
        if let ([lhs, rhs], Some(plan)) = (inputs, self.binary_dot_shortcut()) {
            let reads = [TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs)];
            return execute_binary_dot_shortcut(session, &reads, plan).map_err(Error::from);
        }
        if inputs.len() >= 3 {
            let reads: Vec<_> = inputs
                .iter()
                .map(|tensor| TensorRead::from_tensor(tensor))
                .collect();
            if let Some(result) = crate::cpu_concrete::execute_cached(
                session,
                &reads,
                &self.tree,
                Some(&self.cpu_plan),
            ) {
                return result.map_err(Error::from);
            }
        }
        eager_einsum_exec(session, inputs, &self.tree).map_err(Error::from)
    }

    pub(crate) fn run_reads<'a, I>(&self, inputs: I, session: &mut dyn Engine) -> Result<Tensor>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let inputs = inputs.as_ref();
        self.validate_read_inputs(inputs, PLAN_EXECUTE_OP)?;
        if let Some(plan) = self.binary_dot_shortcut() {
            return execute_binary_dot_shortcut(session, inputs, plan).map_err(Error::from);
        }
        if let Some(result) =
            crate::cpu_concrete::execute_cached(session, inputs, &self.tree, Some(&self.cpu_plan))
        {
            return result.map_err(Error::from);
        }
        eager_einsum_exec_read(session, inputs, &self.tree).map_err(Error::from)
    }

    pub(crate) fn run_owned_into<'a, I>(
        &self,
        inputs: I,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[&'a Tensor]>,
    {
        let inputs = inputs.as_ref();
        self.validate_tensor_inputs(inputs, PLAN_EXECUTE_OP)?;
        self.validate_cached_output(&out, PLAN_EXECUTE_OP)?;
        if let Some(binary_dot) = &self.binary_dot {
            if let [lhs, rhs] = inputs {
                let reads = [TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs)];
                return execute_binary_dot_read_into(session, &reads, binary_dot, out)
                    .map_err(Error::from);
            }
        }
        let reads: Vec<_> = inputs
            .iter()
            .map(|tensor| TensorRead::from_tensor(tensor))
            .collect();
        self.execute_cpu_or_fallback_into(session, &reads, out)
            .map_err(Error::from)
    }

    pub(crate) fn run_reads_into<'a, I>(
        &self,
        inputs: I,
        session: &mut dyn Engine,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let inputs = inputs.as_ref();
        self.validate_read_inputs(inputs, PLAN_EXECUTE_OP)?;
        self.validate_cached_output(&out, PLAN_EXECUTE_OP)?;
        if let Some(binary_dot) = &self.binary_dot {
            return execute_binary_dot_read_into(session, inputs, binary_dot, out)
                .map_err(Error::from);
        }
        self.execute_cpu_or_fallback_into(session, inputs, out)
            .map_err(Error::from)
    }

    pub(crate) fn run_reads_into_accum<'a, I>(
        &self,
        inputs: I,
        session: &mut dyn Engine,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> Result<()>
    where
        I: AsRef<[TensorRead<'a>]>,
    {
        let inputs = inputs.as_ref();
        self.validate_read_inputs(inputs, PLAN_EXECUTE_OP)?;
        self.validate_cached_output(&out, PLAN_EXECUTE_OP)?;
        if let Some(binary_dot) = &self.binary_dot {
            return execute_binary_dot_read_into_accum(
                session,
                inputs,
                binary_dot,
                accumulation,
                out,
            )
            .map_err(Error::from);
        }
        if crate::cpu_concrete::supports(session, inputs) {
            let mut accumulation = accumulation;
            let conjugated = ConjugatedOperands::apply(session, inputs, &mut accumulation)?;
            let result = self.run_reads(conjugated.reads(inputs), session)?;
            let mut out = out;
            return tenferro_tensor::backend::accumulate_dot_result_into(
                &result,
                accumulation,
                &mut out,
            )
            .map_err(Error::from);
        }
        eager_einsum_exec_read_into_accum(session, inputs, &self.tree, accumulation, out)
            .map_err(Error::from)
    }

    fn execute_cpu_or_fallback_into(
        &self,
        session: &mut dyn Engine,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        if crate::cpu_concrete::supports(session, inputs) {
            crate::cpu_concrete::execute_into(
                session,
                inputs,
                &self.tree,
                Some(&self.cpu_plan),
                out,
            )
        } else {
            eager_einsum_exec_read_into(session, inputs, &self.tree, out)
        }
    }

    /// The prepared binary-dot plan when `execute*` may skip the contraction
    /// tree: operands in written order, same dtype (see
    /// `binary_dot_shortcut_plan`). The input count and metadata are validated
    /// by the caller before this is used.
    fn binary_dot_shortcut(&self) -> Option<&crate::binary_dot::BinaryDotPlan> {
        let plan = self.binary_dot.as_ref()?;
        let [lhs, rhs] = self.inputs.as_slice() else {
            return None;
        };
        (plan.operand_order == BinaryDotOperandOrder::Original && lhs.dtype == rhs.dtype)
            .then_some(plan)
    }

    fn prepare_subscripts_internal(
        inputs: Vec<ConcreteEinsumInputSpec>,
        subscripts: &Subscripts,
    ) -> Result<Self> {
        let shapes: Vec<&[usize]> = inputs.iter().map(|input| input.shape.as_slice()).collect();
        let binary_dot = binary_dot_plan_for_shapes(&shapes, subscripts);
        let tree = plan_subscripts(subscripts, &shapes)?;
        let output_shape = tree.output_shape();
        Ok(Self {
            tree,
            inputs,
            output_shape,
            binary_dot,
            cpu_plan: crate::cpu_concrete::Cache::default(),
            conj: crate::api::ConjMask::default(),
        })
    }

    /// Prepare from input metadata with an explicit contraction order and
    /// conjugation (#2552). The automatic order without annealing plans
    /// exactly as [`Self::prepare_subscripts_internal`]; every other order is
    /// honoured as given.
    pub(crate) fn prepare_planned(
        inputs: Vec<ConcreteEinsumInputSpec>,
        subscripts: &Subscripts,
        optimize: &crate::EinsumOptimize,
        conj: crate::api::ConjMask,
    ) -> Result<Self> {
        if conj.len() > inputs.len() {
            return Err(Error::invalid_argument(
                "einsum",
                "conj",
                format!(
                    "the conjugation mask has {} entries for {} inputs",
                    conj.len(),
                    inputs.len()
                ),
            ));
        }
        let mut plan = match optimize {
            crate::EinsumOptimize::Auto(options) if !options.anneals() => {
                Self::prepare_subscripts_internal(inputs, subscripts)?
            }
            optimize => {
                if inputs.is_empty() || subscripts.inputs.len() != inputs.len() {
                    // Same validation and message as the automatic route.
                    return Self::prepare_subscripts_internal(inputs, subscripts);
                }
                let shapes: Vec<&[usize]> =
                    inputs.iter().map(|input| input.shape.as_slice()).collect();
                let binary_dot = binary_dot_plan_for_shapes(&shapes, subscripts);
                let (_, tree) = crate::optimize::resolve_einsum_strategy_with_spec(
                    optimize.clone(),
                    subscripts,
                    &shapes,
                )?;
                let output_shape = tree.output_shape();
                Self {
                    tree,
                    inputs,
                    output_shape,
                    binary_dot,
                    cpu_plan: crate::cpu_concrete::Cache::default(),
                    conj: crate::api::ConjMask::default(),
                }
            }
        };
        plan.conj = conj;
        Ok(plan)
    }

    /// The conjugation chosen at preparation.
    pub(crate) fn conj_mask(&self) -> &crate::api::ConjMask {
        &self.conj
    }

    /// The operand pair contracted at `step` of the prepared tree.
    #[cfg(test)]
    pub(crate) fn tree_step_pair(&self, step: usize) -> Option<(usize, usize)> {
        self.tree.step_pair(step)
    }

    fn validate_tensor_inputs(&self, actual: &[&Tensor], op: &'static str) -> Result<()> {
        self.validate_input_metadata(
            actual.iter().map(|tensor| (tensor.dtype(), tensor.shape())),
            op,
        )
    }

    fn validate_read_inputs(&self, actual: &[TensorRead<'_>], op: &'static str) -> Result<()> {
        self.validate_input_metadata(
            actual.iter().map(|tensor| (tensor.dtype(), tensor.shape())),
            op,
        )
    }

    fn validate_input_metadata<'a>(
        &self,
        actual: impl ExactSizeIterator<Item = (DType, &'a [usize])>,
        op: &'static str,
    ) -> Result<()> {
        if actual.len() != self.inputs.len() {
            return Err(Error::invalid_argument(
                op,
                "inputs",
                format!(
                    "prepared einsum expects {} inputs, got {}",
                    self.inputs.len(),
                    actual.len()
                ),
            ));
        }
        for (expected, (actual_dtype, actual_shape)) in self.inputs.iter().zip(actual) {
            if expected.dtype != actual_dtype {
                return Err(Error::dtype_mismatch(op, expected.dtype, actual_dtype));
            }
            if expected.shape != actual_shape {
                return Err(Error::shape_mismatch(
                    op,
                    expected.shape.clone(),
                    actual_shape.to_vec(),
                ));
            }
        }
        Ok(())
    }

    fn validate_cached_output(&self, out: &TensorWrite<'_>, op: &'static str) -> Result<()> {
        let dtype = self
            .inputs
            .first()
            .map(|input| input.dtype)
            .ok_or_else(|| {
                Error::invalid_argument(op, "inputs", "einsum requires at least one input tensor")
            })?;
        for input in &self.inputs[1..] {
            if input.dtype != dtype {
                return Err(Error::dtype_mismatch(op, dtype, input.dtype));
            }
        }
        if out.dtype() != dtype {
            return Err(Error::dtype_mismatch(op, dtype, out.dtype()));
        }
        if out.shape() != self.output_shape {
            return Err(Error::shape_mismatch(
                op,
                out.shape().to_vec(),
                self.output_shape.clone(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ConcreteEinsumInputSpec {
    dtype: DType,
    shape: Vec<usize>,
}

fn resolve_shapes(notation: &EinsumNotation, shapes: Vec<&[usize]>) -> Result<Subscripts> {
    resolve_einsum_notation(notation, &shapes)
}

fn resolve_tensor_notation(inputs: &[&Tensor], notation: &EinsumNotation) -> Result<Subscripts> {
    resolve_shapes(
        notation,
        inputs.iter().map(|tensor| tensor.shape()).collect(),
    )
}

pub(crate) fn resolve_read_notation<'a>(
    inputs: &[TensorRead<'a>],
    notation: &EinsumNotation,
) -> Result<Subscripts> {
    resolve_shapes(notation, inputs.iter().map(|input| input.shape()).collect())
}

fn input_specs(inputs: &[&Tensor]) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

pub(crate) fn read_input_specs(inputs: &[TensorRead<'_>]) -> Vec<ConcreteEinsumInputSpec> {
    inputs
        .iter()
        .map(|tensor| ConcreteEinsumInputSpec {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
        })
        .collect()
}

fn read_binary_dot_config_for_labels<L: Copy + PartialEq>(
    inputs: &[TensorRead<'_>],
    lhs_labels: &[L],
    rhs_labels: &[L],
    output_labels: &[L],
    out: &TensorWrite<'_>,
) -> Option<(BinaryDotOperandOrder, tenferro_tensor::DotGeneralConfig)> {
    if inputs.len() != 2
        || inputs[0].dtype() != inputs[1].dtype()
        || out.dtype() != inputs[0].dtype()
    {
        return None;
    }
    binary_dot_config_for_into(
        inputs[0].shape(),
        inputs[1].shape(),
        lhs_labels,
        rhs_labels,
        output_labels,
        out.shape(),
    )
}

fn read_binary_dot_config(
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
    out: &TensorWrite<'_>,
) -> Option<(BinaryDotOperandOrder, tenferro_tensor::DotGeneralConfig)> {
    let [lhs, rhs] = subscripts.inputs.as_slice() else {
        return None;
    };
    read_binary_dot_config_for_labels(inputs, lhs, rhs, &subscripts.output, out)
}

fn execute_binary_dot_config_read_into(
    session: &mut dyn Engine,
    inputs: &[TensorRead<'_>],
    order: BinaryDotOperandOrder,
    config: &tenferro_tensor::DotGeneralConfig,
    out: TensorWrite<'_>,
) -> Result<()> {
    let (lhs, rhs) = match order {
        BinaryDotOperandOrder::Original => (0, 1),
        BinaryDotOperandOrder::Swapped => (1, 0),
    };
    session
        .dot_general_read_into(inputs[lhs].clone(), inputs[rhs].clone(), config, out)
        .map_err(Error::from)
}

fn tensor_einsum_into_subscripts(
    session: &mut dyn Engine,
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    out: TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    if inputs.len() == 2 {
        let reads = [
            TensorRead::from_tensor(inputs[0]),
            TensorRead::from_tensor(inputs[1]),
        ];
        if let Some((order, config)) = read_binary_dot_config(&reads, subscripts, &out) {
            return execute_binary_dot_config_read_into(session, &reads, order, &config, out);
        }
    }
    let plan = ConcreteEinsumPlan::prepare_subscripts_internal(input_specs(inputs), subscripts)?;
    validate_output(&plan.inputs, &plan.tree, &out, op)?;
    plan.run_owned_into(inputs, session, out)
}

fn parse_fast_ascii_binary_labels(notation: &str) -> Option<(&[u8], &[u8], &[u8])> {
    let (terms, output) = notation.split_once("->")?;
    let (lhs, rhs) = terms.split_once(',')?;
    // Borrow labels directly. Separators or unsupported labels in any term send
    // the entire expression through the canonical parser, including its errors.
    [lhs, rhs, output]
        .iter()
        .all(|term| term.bytes().all(|byte| byte.is_ascii_alphabetic()))
        .then_some((lhs.as_bytes(), rhs.as_bytes(), output.as_bytes()))
}

fn borrowed_notation_labels(notation: &EinsumNotation) -> Option<[TinyVec<[u32; 8]>; 3]> {
    let [lhs, rhs] = notation.inputs.as_slice() else {
        return None;
    };
    let labels = |axes: &[crate::EinsumAxis]| {
        axes.iter()
            .map(|axis| match axis {
                crate::EinsumAxis::Label(label) => Some(*label),
                crate::EinsumAxis::Ellipsis => None,
            })
            .collect::<Option<TinyVec<[u32; 8]>>>()
    };
    Some([labels(lhs)?, labels(rhs)?, labels(&notation.output)?])
}

fn tensor_read_einsum_into_subscripts(
    session: &mut dyn Engine,
    inputs: &[TensorRead<'_>],
    subscripts: &Subscripts,
    out: TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    if let Some((order, config)) = read_binary_dot_config(inputs, subscripts, &out) {
        return execute_binary_dot_config_read_into(session, inputs, order, &config, out);
    }
    let plan =
        ConcreteEinsumPlan::prepare_subscripts_internal(read_input_specs(inputs), subscripts)?;
    validate_output(&plan.inputs, &plan.tree, &out, op)?;
    plan.run_reads_into(inputs, session, out)
}

fn validate_output(
    inputs: &[ConcreteEinsumInputSpec],
    tree: &ContractionTree,
    out: &TensorWrite<'_>,
    op: &'static str,
) -> Result<()> {
    let expected = output_spec(inputs, tree, op)?;
    if out.dtype() != expected.dtype {
        return Err(Error::dtype_mismatch(op, expected.dtype, out.dtype()));
    }
    if out.shape() != expected.shape.as_slice() {
        return Err(Error::shape_mismatch(
            op,
            out.shape().to_vec(),
            expected.shape.clone(),
        ));
    }
    Ok(())
}

fn output_spec(
    inputs: &[ConcreteEinsumInputSpec],
    tree: &ContractionTree,
    op: &'static str,
) -> Result<ConcreteEinsumInputSpec> {
    let dtype = inputs
        .first()
        .ok_or_else(|| {
            Error::invalid_argument(op, "inputs", "einsum requires at least one input tensor")
        })?
        .dtype;
    for input in inputs {
        if input.dtype != dtype {
            return Err(Error::dtype_mismatch(op, dtype, input.dtype));
        }
    }

    for (input, labels) in inputs.iter().zip(tree.subscripts.inputs.iter()) {
        if labels.len() != input.shape.len() {
            return Err(Error::rank_mismatch(op, labels.len(), input.shape.len()));
        }
    }
    let output_shape = tree.output_shape();
    if output_shape.len() != tree.subscripts.output.len() {
        return Err(Error::invalid_argument(
            op,
            "output labels",
            "an output label is missing from all inputs",
        ));
    }
    Ok(ConcreteEinsumInputSpec {
        dtype,
        shape: output_shape,
    })
}
