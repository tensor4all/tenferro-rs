//! Private record of eager semantic op applications.
//!
//! An [`EagerTensor`](super::EagerTensor) carries an immutable reference into this
//! record. Nothing here infers metadata, decides identity or lowers to execution:
//! the record is replayed into one [`SemanticProgramBuilder`] when a backward pass
//! needs the canonical program, so metadata, shape guards and semantic identity
//! stay owned by the program builder alone.
//!
//! A value is a node plus one of its outputs, because one recorded operation can
//! produce several values (three for an SVD) and edges and residual identities
//! have to name the value, not the operation.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::ad_support::RetainedValue;
use tenferro_runtime::program::{
    CoreSemanticOp, FrozenProgram, ProgramValue, SemanticOpRef, SemanticProgramBuilder,
};
use tenferro_runtime::ErrorPhase;
use tenferro_tensor::DType;

use crate::error::{Error, Result};

/// One immutable node of the eager semantic record.
pub(crate) enum SemanticNode {
    /// A bound leaf: an eager leaf value or an untracked input used as a constant.
    Leaf {
        /// Identifier of this leaf inside every program that replays it.
        key: TensorInputKey,
        dtype: DType,
        rank: usize,
        data: Arc<RetainedValue>,
    },
    /// One applied semantic operation.
    Op {
        op: StdTensorOp,
        inputs: Box<[SemanticValue]>,
        outputs: u8,
    },
    /// A frozen program whose inputs are recorded values.
    ///
    /// Recorded for derivative results, so a second-order backward can replay the
    /// derivative program instead of rebuilding a graph for it.
    Program {
        frozen: Arc<FrozenProgram>,
        inputs: Box<[SemanticValue]>,
    },
}

/// One value of the eager semantic record: a node and one of its outputs.
#[derive(Clone)]
pub(crate) struct SemanticValue {
    node: Arc<SemanticNode>,
    output: u8,
    /// Whether this value's symbolic shape is fully concrete.
    ///
    /// Eager recording appends operations without shape hints, so an operation
    /// output is never concrete there; only a reshape to constant extents, a cast
    /// of a concrete value, and a program output with exact metadata are. This
    /// mirrors the shape-hint test the previous graph carrier used when it
    /// exactified `Concatenate` inputs.
    concrete: bool,
}

impl std::fmt::Debug for SemanticValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticValue")
            .field("node", &self.node_id())
            .field("output", &self.output)
            .field("concrete", &self.concrete)
            .finish()
    }
}

impl SemanticValue {
    /// Record a bound leaf.
    pub(crate) fn leaf(
        key: TensorInputKey,
        dtype: DType,
        rank: usize,
        data: Arc<RetainedValue>,
    ) -> Self {
        Self {
            node: Arc::new(SemanticNode::Leaf {
                key,
                dtype,
                rank,
                data,
            }),
            output: 0,
            concrete: false,
        }
    }

    /// Record one applied operation with its output count.
    pub(crate) fn op(op: StdTensorOp, inputs: Box<[SemanticValue]>, outputs: usize) -> Self {
        Self::op_with_concreteness(op, inputs, outputs, false)
    }

    /// Record one applied operation whose symbolic shape is known to be concrete.
    pub(crate) fn op_with_concreteness(
        op: StdTensorOp,
        inputs: Box<[SemanticValue]>,
        outputs: usize,
        concrete: bool,
    ) -> Self {
        debug_assert!(outputs > 0 && outputs <= u8::MAX as usize);
        Self {
            node: Arc::new(SemanticNode::Op {
                op,
                inputs,
                outputs: outputs as u8,
            }),
            output: 0,
            concrete,
        }
    }

    /// Record a frozen program over recorded inputs.
    pub(crate) fn program(
        frozen: Arc<FrozenProgram>,
        output: usize,
        inputs: Box<[SemanticValue]>,
    ) -> Self {
        debug_assert!(output < frozen.program.outputs().len());
        let concrete = program_output_is_concrete(&frozen, output);
        Self {
            node: Arc::new(SemanticNode::Program { frozen, inputs }),
            output: output as u8,
            concrete,
        }
    }

    /// Return whether this value's symbolic shape is fully concrete.
    pub(crate) fn is_concrete_shape(&self) -> bool {
        self.concrete
    }

    /// Return whether this value is a leaf.
    pub(crate) fn is_leaf(&self) -> bool {
        matches!(self.node.as_ref(), SemanticNode::Leaf { .. })
    }

    /// Stable identity of this value for deduplication and reachability.
    pub(crate) fn identity(&self) -> (usize, u8) {
        (self.node_id(), self.output)
    }

    /// Stable identity of this value's node.
    pub(crate) fn node_id(&self) -> usize {
        Arc::as_ptr(&self.node) as usize
    }

    /// Return the child values this value depends on directly.
    fn children(&self) -> &[SemanticValue] {
        match self.node.as_ref() {
            SemanticNode::Leaf { .. } => &[],
            SemanticNode::Op { inputs, .. } | SemanticNode::Program { inputs, .. } => inputs,
        }
    }

    /// Return a copy of this value with another output index.
    pub(crate) fn at_output(&self, output: u8) -> Self {
        Self {
            node: Arc::clone(&self.node),
            output,
            concrete: self.concrete,
        }
    }
}

fn program_output_is_concrete(frozen: &FrozenProgram, output: usize) -> bool {
    let Some(value) = frozen.program.outputs().get(output).copied() else {
        return false;
    };
    let Ok(metadata) = frozen.program.value_metadata(value) else {
        return false;
    };
    metadata
        .shape()
        .iter()
        .all(|extent| matches!(extent, ShapeExtent::Exact(DimExpr::Const(_))))
}

/// Inputs and outputs of one replayed program.
pub(crate) struct ReplayedProgram {
    frozen: FrozenProgram,
    /// Leaves in program input order; index `i` is program input `i`.
    leaves: Vec<SemanticValue>,
    /// Recorded data of each leaf, parallel to `leaves`.
    data: Vec<Arc<RetainedValue>>,
    /// Program input index per leaf node identity.
    leaf_inputs: HashMap<usize, usize>,
    /// Program output index per requested root.
    outputs: Vec<usize>,
}

impl ReplayedProgram {
    /// Borrow the frozen semantic program.
    pub(crate) fn frozen(&self) -> &FrozenProgram {
        &self.frozen
    }

    /// Borrow the replayed leaves in program input order.
    pub(crate) fn leaves(&self) -> &[SemanticValue] {
        &self.leaves
    }

    /// Borrow the recorded data of each replayed leaf, parallel to [`leaves`](Self::leaves).
    pub(crate) fn leaf_data(&self) -> &[Arc<RetainedValue>] {
        &self.data
    }

    /// Return the program input index of a leaf value.
    pub(crate) fn input_index_of(&self, value: &SemanticValue) -> Option<usize> {
        if !value.is_leaf() {
            return None;
        }
        self.leaf_inputs.get(&value.node_id()).copied()
    }

    /// Borrow the program output indices of the requested roots.
    pub(crate) fn outputs(&self) -> &[usize] {
        &self.outputs
    }
}

/// Replay the recorded closure of `roots` into one semantic program.
///
/// The program's inputs are the reachable leaves in first-visit order, each bound
/// to its recorded data, and its outputs are `roots` in order.
pub(crate) fn replay(roots: &[SemanticValue]) -> Result<ReplayedProgram> {
    let mut builder = ReplayBuilder {
        builder: SemanticProgramBuilder::new(),
        values: HashMap::new(),
        leaves: Vec::new(),
        data: Vec::new(),
        leaf_inputs: HashMap::new(),
    };
    let mut outputs = Vec::with_capacity(roots.len());
    for root in roots {
        outputs.push(builder.emit(root)?);
    }
    let frozen = builder.builder.finish(&outputs).map_err(|source| {
        Error::runtime_state_source("eager semantic replay", ErrorPhase::GraphBuild, source)
    })?;
    Ok(ReplayedProgram {
        frozen,
        leaves: builder.leaves,
        data: builder.data,
        leaf_inputs: builder.leaf_inputs,
        outputs: (0..outputs.len()).collect(),
    })
}

struct ReplayBuilder {
    builder: SemanticProgramBuilder,
    values: HashMap<(usize, u8), ProgramValue>,
    leaves: Vec<SemanticValue>,
    data: Vec<Arc<RetainedValue>>,
    leaf_inputs: HashMap<usize, usize>,
}

impl ReplayBuilder {
    fn emit(&mut self, value: &SemanticValue) -> Result<ProgramValue> {
        if let Some(&emitted) = self.values.get(&value.identity()) {
            return Ok(emitted);
        }
        match value.node.as_ref() {
            SemanticNode::Leaf {
                key,
                dtype,
                rank,
                data,
            } => {
                let index = self.leaves.len();
                let shape = (0..*rank).map(|axis| DimExpr::InputDim {
                    input_idx: index,
                    axis,
                });
                let spec = tenferro_runtime::program::ProgramInputSpec::new(*dtype, shape);
                let input = self.builder.input(spec).map_err(build_error)?;
                self.builder
                    .bind_input_retained(input, Arc::clone(data))
                    .map_err(build_error)?;
                let _ = key;
                self.leaf_inputs.insert(value.node_id(), index);
                self.leaves.push(value.clone());
                self.data.push(Arc::clone(data));
                self.values.insert(value.identity(), input);
                Ok(input)
            }
            SemanticNode::Op { op, outputs, .. } => {
                let inputs = self.emit_all(value.children())?;
                let appended = append_op(&mut self.builder, op, &inputs)?;
                if appended.len() != *outputs as usize {
                    return Err(Error::Internal(format!(
                        "eager semantic replay appended {} outputs for {op:?}, recorded {outputs}",
                        appended.len()
                    )));
                }
                for (output, &emitted) in appended.iter().enumerate() {
                    self.values.insert((value.node_id(), output as u8), emitted);
                }
                self.values.get(&value.identity()).copied().ok_or_else(|| {
                    Error::Internal(
                        "eager semantic replay lost an appended operation output".into(),
                    )
                })
            }
            SemanticNode::Program { frozen, .. } => {
                let inputs = self.emit_all(value.children())?;
                let outputs = self.inline(frozen, &inputs)?;
                for (output, &emitted) in outputs.iter().enumerate() {
                    self.values.insert((value.node_id(), output as u8), emitted);
                }
                self.values.get(&value.identity()).copied().ok_or_else(|| {
                    Error::Internal("eager semantic replay lost an inlined program output".into())
                })
            }
        }
    }

    fn emit_all(&mut self, values: &[SemanticValue]) -> Result<Vec<ProgramValue>> {
        let mut emitted = Vec::with_capacity(values.len());
        for value in values {
            emitted.push(self.emit(value)?);
        }
        Ok(emitted)
    }

    /// Inline one frozen program with its inputs substituted by `inputs`.
    fn inline(
        &mut self,
        frozen: &FrozenProgram,
        inputs: &[ProgramValue],
    ) -> Result<Vec<ProgramValue>> {
        let source_inputs = frozen.program.inputs();
        if source_inputs.len() != inputs.len() {
            return Err(Error::Internal(format!(
                "eager semantic replay inlines a program with {} inputs from {} recorded values",
                source_inputs.len(),
                inputs.len()
            )));
        }
        let mut mapped: HashMap<ProgramValue, ProgramValue> = source_inputs
            .iter()
            .copied()
            .zip(inputs.iter().copied())
            .collect();
        for operation in frozen.program.operations() {
            let mut operation_inputs = Vec::with_capacity(operation.inputs().len());
            for input in operation.inputs() {
                let Some(&emitted) = mapped.get(input) else {
                    return Err(Error::Internal(
                        "eager semantic replay inlined a program operation whose input is missing"
                            .into(),
                    ));
                };
                operation_inputs.push(emitted);
            }
            let appended = match operation.op() {
                SemanticOpRef::Core(op) => self
                    .builder
                    .add_op(op.clone(), &operation_inputs)
                    .map_err(build_error)?,
                SemanticOpRef::Extension(op) => self
                    .builder
                    .add_extension(op.clone_arc(), &operation_inputs)
                    .map_err(build_error)?,
                _ => {
                    return Err(Error::Internal(
                        "eager semantic replay inlined an unsupported semantic operation variant"
                            .into(),
                    ));
                }
            };
            if appended.len() != operation.outputs().len() {
                return Err(Error::Internal(format!(
                    "eager semantic replay inlined an operation with {} outputs for {} recorded outputs",
                    appended.len(),
                    operation.outputs().len()
                )));
            }
            for (source, emitted) in operation.outputs().iter().zip(appended.iter()) {
                mapped.insert(*source, *emitted);
            }
        }
        frozen
            .program
            .outputs()
            .iter()
            .map(|output| {
                mapped.get(output).copied().ok_or_else(|| {
                    Error::Internal(
                        "eager semantic replay inlined a program whose output is missing".into(),
                    )
                })
            })
            .collect()
    }
}

fn append_op(
    builder: &mut SemanticProgramBuilder,
    op: &StdTensorOp,
    inputs: &[ProgramValue],
) -> Result<Box<[ProgramValue]>> {
    match op {
        StdTensorOp::Extension(extension) => builder
            .add_extension(extension.clone_arc(), inputs)
            .map_err(build_error),
        op => {
            let core = CoreSemanticOp::try_from(op).map_err(|source| {
                Error::runtime_state_source("eager semantic replay", ErrorPhase::GraphBuild, source)
            })?;
            builder.add_op(core, inputs).map_err(build_error)
        }
    }
}

fn build_error(source: tenferro_runtime::program::ProgramBuildError) -> Error {
    Error::runtime_state_source("eager semantic replay", ErrorPhase::GraphBuild, source)
}

/// Collect the leaf-node identities reachable from `root`.
pub(crate) fn reachable_leaf_nodes(root: &SemanticValue) -> HashSet<usize> {
    let mut reachable = HashSet::new();
    let mut visited = HashSet::new();
    let mut pending = vec![root.clone()];
    while let Some(value) = pending.pop() {
        if !visited.insert(value.identity()) {
            continue;
        }
        if value.is_leaf() {
            reachable.insert(value.node_id());
            continue;
        }
        pending.extend(value.children().iter().cloned());
    }
    reachable
}

#[cfg(test)]
mod tests;
