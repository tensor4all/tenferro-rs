use std::fmt;

use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::ShapeExtent;
use tenferro_tensor::{DType, DotGeneralConfig};

use crate::exec::{ExecInstruction, ExecOp, ExecProgram};

/// Read-only lowering view over a compiled graph program.
///
/// This view is for peer executor crates that need to translate a
/// [`CompiledGraph`](super::CompiledGraph) without mutating the runtime-owned
/// execution program.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{GraphCompiler, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let y = x.neg().unwrap();
/// let program = compiler.compile(&y).unwrap();
/// let view = program.lowering_view();
/// assert_eq!(view.output_slots().len(), 1);
/// ```
#[derive(Clone, Copy)]
pub struct GraphProgramLoweringView<'a> {
    exec: &'a ExecProgram,
}

impl fmt::Debug for GraphProgramLoweringView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphProgramLoweringView")
            .field("slot_count", &self.slot_count())
            .field("input_count", &self.input_slots().len())
            .field("output_count", &self.output_slots().len())
            .field("instruction_count", &self.exec.instructions.len())
            .finish()
    }
}

impl<'a> GraphProgramLoweringView<'a> {
    pub(crate) fn new(exec: &'a ExecProgram) -> Self {
        Self { exec }
    }

    /// Return the number of execution slots used by the program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// assert!(program.lowering_view().slot_count() >= 1);
    /// ```
    pub fn slot_count(&self) -> usize {
        self.exec.n_slots
    }

    /// Return the execution slots populated by graph inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// assert_eq!(program.lowering_view().input_slots().len(), 1);
    /// ```
    pub fn input_slots(&self) -> &'a [usize] {
        &self.exec.input_slots
    }

    /// Return the execution slots used as program outputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// assert_eq!(program.lowering_view().output_slots().len(), 1);
    /// ```
    pub fn output_slots(&self) -> &'a [usize] {
        &self.exec.output_slots
    }

    /// Iterate over read-only instruction views in execution order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// assert!(program.lowering_view().instructions().count() >= 1);
    /// ```
    pub fn instructions(&self) -> impl ExactSizeIterator<Item = GraphInstructionView<'a>> + '_ {
        self.exec.instructions.iter().map(GraphInstructionView::new)
    }
}

/// Read-only lowering view over one execution instruction.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{GraphCompiler, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let y = x.neg().unwrap();
/// let program = compiler.compile(&y).unwrap();
/// let inst = program.lowering_view().instructions().next().unwrap();
/// assert_eq!(inst.output_slots().len(), 1);
/// ```
#[derive(Clone, Copy)]
pub struct GraphInstructionView<'a> {
    inst: &'a ExecInstruction,
}

impl fmt::Debug for GraphInstructionView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphInstructionView")
            .field("op", &self.op_name())
            .field("input_count", &self.input_slots().len())
            .field("output_count", &self.output_slots().len())
            .field("dtype", &self.dtype())
            .finish()
    }
}

impl<'a> GraphInstructionView<'a> {
    fn new(inst: &'a ExecInstruction) -> Self {
        Self { inst }
    }

    /// Return the operation view for this instruction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, GraphOpView, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let inst = program.lowering_view().instructions().next().unwrap();
    /// assert!(matches!(inst.op(), GraphOpView::Negate));
    /// ```
    pub fn op(&self) -> GraphOpView<'a> {
        match &self.inst.op {
            ExecOp::Constant { dtype, bytes } => GraphOpView::Constant {
                dtype: *dtype,
                bytes,
            },
            ExecOp::Add => GraphOpView::Add,
            ExecOp::Multiply => GraphOpView::Multiply,
            ExecOp::Negate => GraphOpView::Negate,
            ExecOp::Divide => GraphOpView::Divide,
            ExecOp::Abs => GraphOpView::Abs,
            ExecOp::Exp => GraphOpView::Exp,
            ExecOp::Log => GraphOpView::Log,
            ExecOp::Sin => GraphOpView::Sin,
            ExecOp::Cos => GraphOpView::Cos,
            ExecOp::Tanh => GraphOpView::Tanh,
            ExecOp::Sqrt => GraphOpView::Sqrt,
            ExecOp::Rsqrt => GraphOpView::Rsqrt,
            ExecOp::Pow => GraphOpView::Pow,
            ExecOp::Expm1 => GraphOpView::Expm1,
            ExecOp::Log1p => GraphOpView::Log1p,
            ExecOp::Convert { to } => GraphOpView::Convert { to: *to },
            ExecOp::Reshape { .. } => GraphOpView::Reshape,
            ExecOp::BroadcastInDim { dims, .. } => GraphOpView::BroadcastInDim { dims },
            ExecOp::Transpose { perm } => GraphOpView::Transpose { perm },
            ExecOp::ReduceSum { axes } => GraphOpView::ReduceSum { axes },
            ExecOp::DotGeneral(config) => GraphOpView::DotGeneral { config },
            ExecOp::Extension(op) => GraphOpView::Extension { op: op.as_ref() },
            other => GraphOpView::Unsupported {
                name: exec_op_name(other),
            },
        }
    }

    /// Return a stable operation name for diagnostics.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let inst = program.lowering_view().instructions().next().unwrap();
    /// assert_eq!(inst.op_name(), "Negate");
    /// ```
    pub fn op_name(&self) -> &'static str {
        self.op().name()
    }

    /// Return the input slots consumed by this instruction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let y = (&x + &x).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&y).unwrap();
    /// let inst = program.lowering_view().instructions().next().unwrap();
    /// assert_eq!(inst.input_slots().len(), 2);
    /// ```
    pub fn input_slots(&self) -> &'a [usize] {
        &self.inst.input_slots
    }

    /// Return the output slots written by this instruction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let inst = program.lowering_view().instructions().next().unwrap();
    /// assert_eq!(inst.output_slots().len(), 1);
    /// ```
    pub fn output_slots(&self) -> &'a [usize] {
        &self.inst.output_slots
    }

    /// Return the dtype of this instruction's output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let inst = program.lowering_view().instructions().next().unwrap();
    /// assert_eq!(inst.dtype(), DType::F64);
    /// ```
    pub fn dtype(&self) -> DType {
        self.inst.dtype
    }

    /// Resolve an exact static output shape for this instruction.
    ///
    /// `input_shapes` must be ordered the same way as [`Self::input_slots`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let inst = program.lowering_view().instructions().next().unwrap();
    /// assert_eq!(inst.static_output_shape(0, &[&[1]]).unwrap(), vec![1]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`GraphProgramLoweringShapeError::MissingOutput`] when
    /// `output_index` is absent, [`GraphProgramLoweringShapeError::NonStatic`]
    /// when an extent is only an upper bound, or
    /// [`GraphProgramLoweringShapeError::InvalidDimExpr`] when a symbolic
    /// dimension cannot be evaluated for `input_shapes`.
    pub fn static_output_shape(
        &self,
        output_index: usize,
        input_shapes: &[&[usize]],
    ) -> std::result::Result<Vec<usize>, GraphProgramLoweringShapeError> {
        let extents = self.inst.output_extents.get(output_index).ok_or(
            GraphProgramLoweringShapeError::MissingOutput {
                op: self.op_name(),
                output_index,
            },
        )?;
        let mut shape = Vec::with_capacity(extents.len());
        for (axis, extent) in extents.iter().enumerate() {
            match extent {
                ShapeExtent::Exact(dim) => shape.push(dim.eval(input_shapes).map_err(|err| {
                    GraphProgramLoweringShapeError::InvalidDimExpr {
                        op: self.op_name(),
                        output_index,
                        axis,
                        source: err,
                    }
                })?),
                ShapeExtent::UpperBound(_) => {
                    return Err(GraphProgramLoweringShapeError::NonStatic {
                        op: self.op_name(),
                        output_index,
                        axis,
                        kind: "an upper bound",
                    });
                }
                ShapeExtent::Unknown => {
                    return Err(GraphProgramLoweringShapeError::NonStatic {
                        op: self.op_name(),
                        output_index,
                        axis,
                        kind: "unknown",
                    });
                }
            }
        }
        Ok(shape)
    }
}

/// Read-only operation view for graph lowering integrations.
///
/// Unsupported operation families are represented as [`GraphOpView::Unsupported`]
/// so peer executors can emit precise diagnostics without depending on the raw
/// execution IR.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{GraphCompiler, GraphOpView, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let y = x.neg().unwrap();
/// let program = compiler.compile(&y).unwrap();
/// let op = program.lowering_view().instructions().next().unwrap().op();
/// assert!(matches!(op, GraphOpView::Negate));
/// ```
#[derive(Clone, Copy)]
pub enum GraphOpView<'a> {
    /// Scalar constant payload.
    Constant { dtype: DType, bytes: &'a [u8] },
    /// Elementwise addition.
    Add,
    /// Elementwise multiplication.
    Multiply,
    /// Elementwise negation.
    Negate,
    /// Elementwise division.
    Divide,
    /// Elementwise absolute value.
    Abs,
    /// Elementwise exponential.
    Exp,
    /// Elementwise natural logarithm.
    Log,
    /// Elementwise sine.
    Sin,
    /// Elementwise cosine.
    Cos,
    /// Elementwise hyperbolic tangent.
    Tanh,
    /// Elementwise square root.
    Sqrt,
    /// Elementwise reciprocal square root.
    Rsqrt,
    /// Elementwise power.
    Pow,
    /// Elementwise exponential minus one.
    Expm1,
    /// Elementwise natural logarithm of one plus input.
    Log1p,
    /// Dtype conversion.
    Convert { to: DType },
    /// Shape-only reshape.
    Reshape,
    /// Broadcast with output-to-input dimension mapping.
    BroadcastInDim { dims: &'a [usize] },
    /// Transpose with output dimension permutation.
    Transpose { perm: &'a [usize] },
    /// Sum reduction.
    ReduceSum { axes: &'a [usize] },
    /// General dot/contraction.
    DotGeneral { config: &'a DotGeneralConfig },
    /// Extension operation with an owner-provided optional standard-op lowering.
    Extension { op: &'a dyn ExtensionOp },
    /// Operation outside the stable public lowering view.
    Unsupported { name: &'static str },
}

impl fmt::Debug for GraphOpView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Constant { dtype, bytes } => f
                .debug_struct("Constant")
                .field("dtype", dtype)
                .field("byte_len", &bytes.len())
                .finish(),
            Self::Add => f.write_str("Add"),
            Self::Multiply => f.write_str("Multiply"),
            Self::Negate => f.write_str("Negate"),
            Self::Divide => f.write_str("Divide"),
            Self::Abs => f.write_str("Abs"),
            Self::Exp => f.write_str("Exp"),
            Self::Log => f.write_str("Log"),
            Self::Sin => f.write_str("Sin"),
            Self::Cos => f.write_str("Cos"),
            Self::Tanh => f.write_str("Tanh"),
            Self::Sqrt => f.write_str("Sqrt"),
            Self::Rsqrt => f.write_str("Rsqrt"),
            Self::Pow => f.write_str("Pow"),
            Self::Expm1 => f.write_str("Expm1"),
            Self::Log1p => f.write_str("Log1p"),
            Self::Convert { to } => f.debug_struct("Convert").field("to", to).finish(),
            Self::Reshape => f.write_str("Reshape"),
            Self::BroadcastInDim { dims } => f
                .debug_struct("BroadcastInDim")
                .field("dims", dims)
                .finish(),
            Self::Transpose { perm } => f.debug_struct("Transpose").field("perm", perm).finish(),
            Self::ReduceSum { axes } => f.debug_struct("ReduceSum").field("axes", axes).finish(),
            Self::DotGeneral { config } => f
                .debug_struct("DotGeneral")
                .field("config", config)
                .finish(),
            Self::Extension { op } => f
                .debug_struct("Extension")
                .field("family_id", &op.family_id())
                .finish(),
            Self::Unsupported { name } => {
                f.debug_struct("Unsupported").field("name", name).finish()
            }
        }
    }
}

impl GraphOpView<'_> {
    /// Return the stable operation name used in diagnostics.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let op = program.lowering_view().instructions().next().unwrap().op();
    /// assert_eq!(op.name(), "Negate");
    /// ```
    pub fn name(&self) -> &'static str {
        match self {
            Self::Constant { .. } => "Constant",
            Self::Add => "Add",
            Self::Multiply => "Multiply",
            Self::Negate => "Negate",
            Self::Divide => "Divide",
            Self::Abs => "Abs",
            Self::Exp => "Exp",
            Self::Log => "Log",
            Self::Sin => "Sin",
            Self::Cos => "Cos",
            Self::Tanh => "Tanh",
            Self::Sqrt => "Sqrt",
            Self::Rsqrt => "Rsqrt",
            Self::Pow => "Pow",
            Self::Expm1 => "Expm1",
            Self::Log1p => "Log1p",
            Self::Convert { .. } => "Convert",
            Self::Reshape => "Reshape",
            Self::BroadcastInDim { .. } => "BroadcastInDim",
            Self::Transpose { .. } => "Transpose",
            Self::ReduceSum { .. } => "ReduceSum",
            Self::DotGeneral { .. } => "DotGeneral",
            Self::Extension { .. } => "Extension",
            Self::Unsupported { name } => name,
        }
    }
}

/// Error returned when a lowering view cannot resolve an exact output shape.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::GraphProgramLoweringShapeError;
///
/// let err = GraphProgramLoweringShapeError::MissingOutput {
///     op: "Example",
///     output_index: 0,
/// };
/// assert!(err.to_string().contains("Example"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GraphProgramLoweringShapeError {
    /// The instruction has no metadata for the requested output.
    #[error("ExecOp::{op} missing output_extents for output {output_index}")]
    MissingOutput {
        op: &'static str,
        output_index: usize,
    },
    /// The instruction output has dynamic or unknown extent metadata.
    #[error("ExecOp::{op} output {output_index} axis {axis} has non-static extent: {kind}")]
    NonStatic {
        op: &'static str,
        output_index: usize,
        axis: usize,
        kind: &'static str,
    },
    /// Static shape evaluation failed for an exact dimension expression.
    #[error(
        "ExecOp::{op} output {output_index} axis {axis} has invalid dimension expression: {source}"
    )]
    InvalidDimExpr {
        op: &'static str,
        output_index: usize,
        axis: usize,
        source: tenferro_ops::dim_expr::DimExprEvalError,
    },
}

fn exec_op_name(op: &ExecOp) -> &'static str {
    match op {
        ExecOp::Transpose { .. } => "Transpose",
        ExecOp::Reshape { .. } => "Reshape",
        ExecOp::BroadcastInDim { .. } => "BroadcastInDim",
        ExecOp::Convert { .. } => "Convert",
        ExecOp::Constant { .. } => "Constant",
        ExecOp::DotGeneral(_) => "DotGeneral",
        ExecOp::DotGeneralWithConj { .. } => "DotGeneralWithConj",
        ExecOp::ReduceSum { .. } => "ReduceSum",
        ExecOp::ExtractDiag { .. } => "ExtractDiag",
        ExecOp::EmbedDiag { .. } => "EmbedDiag",
        ExecOp::Tril { .. } => "Tril",
        ExecOp::Triu { .. } => "Triu",
        ExecOp::Add => "Add",
        ExecOp::Subtract => "Subtract",
        ExecOp::Multiply => "Multiply",
        ExecOp::Negate => "Negate",
        ExecOp::Conj => "Conj",
        ExecOp::Divide => "Divide",
        ExecOp::Remainder => "Remainder",
        ExecOp::Abs => "Abs",
        ExecOp::Sign => "Sign",
        ExecOp::Maximum => "Maximum",
        ExecOp::Minimum => "Minimum",
        ExecOp::Compare(_) => "Compare",
        ExecOp::Select => "Select",
        ExecOp::Clamp => "Clamp",
        ExecOp::Exp => "Exp",
        ExecOp::Log => "Log",
        ExecOp::Sin => "Sin",
        ExecOp::Cos => "Cos",
        ExecOp::Tanh => "Tanh",
        ExecOp::Sqrt => "Sqrt",
        ExecOp::Rsqrt => "Rsqrt",
        ExecOp::Pow => "Pow",
        ExecOp::Expm1 => "Expm1",
        ExecOp::Log1p => "Log1p",
        ExecOp::Gather(_) => "Gather",
        ExecOp::GatherDynamicSliceSizes { .. } => "GatherDynamicSliceSizes",
        ExecOp::Scatter(_) => "Scatter",
        ExecOp::Slice(_) => "Slice",
        ExecOp::DynamicSlice { .. } => "DynamicSlice",
        ExecOp::DynamicUpdateSlice => "DynamicUpdateSlice",
        ExecOp::Pad(_) => "Pad",
        ExecOp::Concatenate { .. } => "Concatenate",
        ExecOp::Reverse { .. } => "Reverse",
        ExecOp::ShapeOf { .. } => "ShapeOf",
        ExecOp::DynamicTruncate { .. } => "DynamicTruncate",
        ExecOp::PadToMatch { .. } => "PadToMatch",
        ExecOp::ReduceProd { .. } => "ReduceProd",
        ExecOp::ReduceMax { .. } => "ReduceMax",
        ExecOp::ReduceMin { .. } => "ReduceMin",
        ExecOp::Extension(_) => "Extension",
    }
}

#[cfg(test)]
mod tests;
