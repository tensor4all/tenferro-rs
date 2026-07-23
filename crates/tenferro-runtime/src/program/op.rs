use std::sync::Arc;

use computegraph::GraphOperation;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

use super::metadata::SemanticProvenance;
use super::{
    Alias, Effect, ProgramValue, SemanticPlacementConstraint, SemanticProvenanceView, ShapeGuard,
};

/// Closed backend-neutral vocabulary of core semantic tensor operations.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CoreSemanticOp {
    Add,
    Sub,
    Mul,
    Neg,
    Conj,
    DotGeneral {
        config: DotGeneralConfig,
    },
    Transpose {
        perm: Vec<usize>,
    },
    Reshape {
        to_shape: Vec<DimExpr>,
    },
    BroadcastInDim {
        shape: Vec<DimExpr>,
        dims: Vec<usize>,
    },
    Convert {
        from: DType,
        to: DType,
    },
    Constant {
        dtype: DType,
        bytes: Vec<u8>,
    },
    ReduceSum {
        axes: Vec<usize>,
    },
    Div,
    Rem,
    Abs,
    Sign,
    Maximum,
    Minimum,
    Compare(CompareDir),
    Select,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
    ExtractDiag {
        axis_a: usize,
        axis_b: usize,
    },
    EmbedDiag {
        axis_a: usize,
        axis_b: usize,
    },
    Tril {
        k: i64,
    },
    Triu {
        k: i64,
    },
    Gather(GatherConfig),
    GatherDynamicSliceSizes {
        offset_dims: Vec<usize>,
        collapsed_slice_dims: Vec<usize>,
        start_index_map: Vec<usize>,
        index_vector_dim: usize,
        slice_sizes: Vec<DimExpr>,
    },
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice {
        slice_sizes: Vec<usize>,
    },
    DynamicUpdateSlice,
    Pad(PadConfig),
    Concatenate {
        axis: usize,
        input_count: usize,
    },
    Reverse {
        axes: Vec<usize>,
    },
    ShapeOf {
        axis: usize,
    },
    DynamicTruncate {
        axis: usize,
    },
    PadToMatch {
        axis: usize,
    },
    ReduceProd {
        axes: Vec<usize>,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceMin {
        axes: Vec<usize>,
    },
}

impl CoreSemanticOp {
    pub(crate) fn input_count(&self) -> usize {
        let standard = StdTensorOp::from(self);
        GraphOperation::input_count(&standard)
    }

    pub(crate) fn output_count(&self) -> usize {
        let standard = StdTensorOp::from(self);
        GraphOperation::output_count(&standard)
    }
}

impl From<&CoreSemanticOp> for StdTensorOp {
    fn from(op: &CoreSemanticOp) -> Self {
        match op {
            CoreSemanticOp::Add => Self::Add,
            CoreSemanticOp::Sub => Self::Sub,
            CoreSemanticOp::Mul => Self::Mul,
            CoreSemanticOp::Neg => Self::Neg,
            CoreSemanticOp::Conj => Self::Conj,
            CoreSemanticOp::DotGeneral { config } => Self::DotGeneral {
                config: config.clone(),
            },
            CoreSemanticOp::Transpose { perm } => Self::Transpose { perm: perm.clone() },
            CoreSemanticOp::Reshape { to_shape } => Self::Reshape {
                to_shape: to_shape.clone(),
            },
            CoreSemanticOp::BroadcastInDim { shape, dims } => Self::BroadcastInDim {
                shape: shape.clone(),
                dims: dims.clone(),
            },
            CoreSemanticOp::Convert { from, to } => Self::Convert {
                from: *from,
                to: *to,
            },
            CoreSemanticOp::Constant { dtype, bytes } => Self::Constant {
                dtype: *dtype,
                bytes: bytes.clone(),
            },
            CoreSemanticOp::ReduceSum { axes } => Self::ReduceSum { axes: axes.clone() },
            CoreSemanticOp::Div => Self::Div,
            CoreSemanticOp::Rem => Self::Rem,
            CoreSemanticOp::Abs => Self::Abs,
            CoreSemanticOp::Sign => Self::Sign,
            CoreSemanticOp::Maximum => Self::Maximum,
            CoreSemanticOp::Minimum => Self::Minimum,
            CoreSemanticOp::Compare(direction) => Self::Compare(direction.clone()),
            CoreSemanticOp::Select => Self::Select,
            CoreSemanticOp::Clamp => Self::Clamp,
            CoreSemanticOp::Exp => Self::Exp,
            CoreSemanticOp::Log => Self::Log,
            CoreSemanticOp::Sin => Self::Sin,
            CoreSemanticOp::Cos => Self::Cos,
            CoreSemanticOp::Tanh => Self::Tanh,
            CoreSemanticOp::Sqrt => Self::Sqrt,
            CoreSemanticOp::Rsqrt => Self::Rsqrt,
            CoreSemanticOp::Pow => Self::Pow,
            CoreSemanticOp::Expm1 => Self::Expm1,
            CoreSemanticOp::Log1p => Self::Log1p,
            CoreSemanticOp::ExtractDiag { axis_a, axis_b } => Self::ExtractDiag {
                axis_a: *axis_a,
                axis_b: *axis_b,
            },
            CoreSemanticOp::EmbedDiag { axis_a, axis_b } => Self::EmbedDiag {
                axis_a: *axis_a,
                axis_b: *axis_b,
            },
            CoreSemanticOp::Tril { k } => Self::Tril { k: *k },
            CoreSemanticOp::Triu { k } => Self::Triu { k: *k },
            CoreSemanticOp::Gather(config) => Self::Gather(config.clone()),
            CoreSemanticOp::GatherDynamicSliceSizes {
                offset_dims,
                collapsed_slice_dims,
                start_index_map,
                index_vector_dim,
                slice_sizes,
            } => Self::GatherDynamicSliceSizes {
                offset_dims: offset_dims.clone(),
                collapsed_slice_dims: collapsed_slice_dims.clone(),
                start_index_map: start_index_map.clone(),
                index_vector_dim: *index_vector_dim,
                slice_sizes: slice_sizes.clone(),
            },
            CoreSemanticOp::Scatter(config) => Self::Scatter(config.clone()),
            CoreSemanticOp::Slice(config) => Self::Slice(config.clone()),
            CoreSemanticOp::DynamicSlice { slice_sizes } => Self::DynamicSlice {
                slice_sizes: slice_sizes.clone(),
            },
            CoreSemanticOp::DynamicUpdateSlice => Self::DynamicUpdateSlice,
            CoreSemanticOp::Pad(config) => Self::Pad(config.clone()),
            CoreSemanticOp::Concatenate { axis, input_count } => Self::Concatenate {
                axis: *axis,
                input_count: *input_count,
            },
            CoreSemanticOp::Reverse { axes } => Self::Reverse { axes: axes.clone() },
            CoreSemanticOp::ShapeOf { axis } => Self::ShapeOf { axis: *axis },
            CoreSemanticOp::DynamicTruncate { axis } => Self::DynamicTruncate { axis: *axis },
            CoreSemanticOp::PadToMatch { axis } => Self::PadToMatch { axis: *axis },
            CoreSemanticOp::ReduceProd { axes } => Self::ReduceProd { axes: axes.clone() },
            CoreSemanticOp::ReduceMax { axes } => Self::ReduceMax { axes: axes.clone() },
            CoreSemanticOp::ReduceMin { axes } => Self::ReduceMin { axes: axes.clone() },
        }
    }
}

pub(crate) enum SemanticOp {
    Core(CoreSemanticOp),
    Extension(Arc<dyn ExtensionOp>),
}

pub(crate) struct SemanticOperation {
    pub(crate) op: SemanticOp,
    pub(crate) inputs: Box<[ProgramValue]>,
    pub(crate) outputs: Box<[ProgramValue]>,
    pub(crate) effects: Box<[Effect]>,
    pub(crate) aliases: Box<[Alias]>,
    pub(crate) shape_guards: Box<[ShapeGuard]>,
    pub(crate) placement: SemanticPlacementConstraint,
    pub(crate) provenance: SemanticProvenance,
}

/// Borrowed semantic operation payload.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum SemanticOpRef<'a> {
    /// Closed core operation.
    Core(&'a CoreSemanticOp),
    /// Extension semantic payload.
    Extension(&'a dyn ExtensionOp),
}

impl std::fmt::Debug for SemanticOpRef<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core(_) => formatter.write_str("SemanticOpRef::Core(<bounded>)"),
            Self::Extension(op) => formatter
                .debug_tuple("SemanticOpRef::Extension")
                .field(&op.family_id())
                .finish(),
        }
    }
}

/// Allocation-free immutable view of one semantic operation.
#[derive(Clone, Copy)]
pub struct SemanticOperationView<'a> {
    operation: &'a SemanticOperation,
}

impl<'a> SemanticOperationView<'a> {
    pub(crate) const fn new(operation: &'a SemanticOperation) -> Self {
        Self { operation }
    }

    /// Borrow the semantic operation payload.
    pub fn op(self) -> SemanticOpRef<'a> {
        match &self.operation.op {
            SemanticOp::Core(op) => SemanticOpRef::Core(op),
            SemanticOp::Extension(op) => SemanticOpRef::Extension(op.as_ref()),
        }
    }

    /// Borrow ordered SSA inputs.
    pub fn inputs(self) -> &'a [ProgramValue] {
        &self.operation.inputs
    }

    /// Borrow ordered SSA outputs.
    pub fn outputs(self) -> &'a [ProgramValue] {
        &self.operation.outputs
    }

    /// Borrow ordered observable effects.
    pub fn effects(self) -> &'a [Effect] {
        &self.operation.effects
    }

    /// Borrow output alias declarations.
    pub fn aliases(self) -> &'a [Alias] {
        &self.operation.aliases
    }

    /// Borrow operation-local symbolic guards.
    pub fn shape_guards(self) -> &'a [ShapeGuard] {
        &self.operation.shape_guards
    }

    /// Return bounded diagnostic provenance without source identities.
    pub fn provenance(self) -> SemanticProvenanceView<'a> {
        self.operation.provenance.view()
    }

    /// Return the unresolved placement constraint.
    pub fn placement(self) -> SemanticPlacementConstraint {
        self.operation.placement
    }
}

impl std::fmt::Debug for SemanticOperationView<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticOperationView")
            .field("op", &self.op())
            .field("inputs", &self.inputs().len())
            .field("outputs", &self.outputs().len())
            .field("effects", &self.effects().len())
            .field("aliases", &self.aliases().len())
            .field("shape_guards", &self.shape_guards().len())
            .finish()
    }
}
