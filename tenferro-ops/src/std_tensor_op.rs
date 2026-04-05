use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::GraphOp;

use crate::input_key::TensorInputKey;
use crate::semiring_ops::SemiringOps;
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum StdTensorOp {
    // Tier 1: semiring
    Add,
    Mul,
    Neg,
    Conj,
    DotGeneral(DotGeneralConfig),
    Transpose {
        perm: Vec<usize>,
    },
    Reshape {
        from_shape: Vec<usize>,
        to_shape: Vec<usize>,
    },
    BroadcastInDim {
        shape: Vec<usize>,
        dims: Vec<usize>,
    },
    ReduceSum {
        axes: Vec<usize>,
        input_shape: Vec<usize>,
    },

    // Tier 2: elementwise
    Div,
    Abs,
    Sign,
    Maximum,
    Minimum,
    Compare(CompareDir),
    Select,
    Clamp,

    // Tier 2: analytic
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

    // Tier 1: diagonal extraction / embedding (AD-closed pair)
    ExtractDiag {
        axis_a: usize,
        axis_b: usize,
    },
    EmbedDiag {
        axis_a: usize,
        axis_b: usize,
    },

    // Tier 2: indexing
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice {
        slice_sizes: Vec<usize>,
    },
    Pad(PadConfig),
    Concatenate {
        axis: usize,
    },
    Reverse {
        axes: Vec<usize>,
    },

    // Tier 2: reductions
    ReduceProd {
        axes: Vec<usize>,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceMin {
        axes: Vec<usize>,
    },

    // Linalg
    Cholesky,
    Svd,
    Qr,
    Eigh,
    Solve,
}

impl GraphOp for StdTensorOp {
    type Operand = tenferro_tensor::Tensor;
    type Context = ();
    type InputKey = TensorInputKey;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul | Self::DotGeneral(_) | Self::Gather(_) => 2,
            Self::Neg
            | Self::Conj
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::BroadcastInDim { .. }
            | Self::ReduceSum { .. }
            | Self::ExtractDiag { .. }
            | Self::EmbedDiag { .. }
            | Self::Slice(_)
            | Self::Pad(_)
            | Self::Reverse { .. } => 1,
            Self::Div | Self::Maximum | Self::Minimum | Self::Pow | Self::DynamicSlice { .. } => 2,
            Self::Scatter(_) => 3,
            Self::Concatenate { .. } => {
                todo!(
                    "n_inputs not yet implemented for variable-arity op {:?}",
                    self
                )
            }
            Self::Abs
            | Self::Sign
            | Self::Exp
            | Self::Log
            | Self::Sin
            | Self::Cos
            | Self::Tanh
            | Self::Sqrt
            | Self::Rsqrt
            | Self::Expm1
            | Self::Log1p => 1,
            Self::Select | Self::Clamp => 3,
            Self::Compare(_) => 2,
            Self::Cholesky | Self::Svd | Self::Qr | Self::Eigh => 1,
            Self::Solve => 2,
            _ => todo!("n_inputs not yet implemented for {:?}", self),
        }
    }

    fn n_outputs(&self) -> usize {
        match self {
            Self::Add
            | Self::Mul
            | Self::Neg
            | Self::Conj
            | Self::DotGeneral(_)
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::BroadcastInDim { .. }
            | Self::ReduceSum { .. }
            | Self::Div
            | Self::Abs
            | Self::Sign
            | Self::Maximum
            | Self::Minimum
            | Self::Compare(_)
            | Self::Select
            | Self::Clamp
            | Self::Exp
            | Self::Log
            | Self::Sin
            | Self::Cos
            | Self::Tanh
            | Self::Sqrt
            | Self::Rsqrt
            | Self::Pow
            | Self::Expm1
            | Self::Log1p
            | Self::ExtractDiag { .. }
            | Self::EmbedDiag { .. }
            | Self::Gather(_)
            | Self::Scatter(_)
            | Self::Slice(_)
            | Self::DynamicSlice { .. }
            | Self::Pad(_)
            | Self::Reverse { .. } => 1,
            Self::Cholesky | Self::Solve => 1,
            Self::Svd => 3,  // U, S, Vt
            Self::Qr => 2,   // Q, R
            Self::Eigh => 2, // eigenvalues, eigenvectors
            Self::Concatenate { .. } => todo!(
                "n_outputs not yet implemented for variable-arity op {:?}",
                self
            ),
            _ => todo!("n_outputs not yet implemented for {:?}", self),
        }
    }
}

impl PrimitiveOp for StdTensorOp {
    fn add() -> Self {
        StdTensorOp::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
    ) -> Vec<Option<LocalValId>> {
        crate::ad::linearize(self, builder, primal_in, primal_out, tangent_in)
    }

    fn transpose_rule(
        &self,
        builder: &mut FragmentBuilder<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
    ) -> Vec<Option<LocalValId>> {
        crate::ad::transpose_rule(self, builder, cotangent_out, inputs, mode)
    }
}

impl SemiringOps for StdTensorOp {
    fn add_op() -> Self {
        StdTensorOp::Add
    }

    fn mul_op() -> Self {
        StdTensorOp::Mul
    }

    fn dot_general(config: DotGeneralConfig) -> Self {
        StdTensorOp::DotGeneral(config)
    }

    fn reduce_sum(axes: Vec<usize>, input_shape: Vec<usize>) -> Self {
        StdTensorOp::ReduceSum { axes, input_shape }
    }

    fn transpose_op(perm: Vec<usize>) -> Self {
        StdTensorOp::Transpose { perm }
    }

    fn reshape(from_shape: Vec<usize>, to_shape: Vec<usize>) -> Self {
        StdTensorOp::Reshape {
            from_shape,
            to_shape,
        }
    }

    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self {
        StdTensorOp::BroadcastInDim { shape, dims }
    }

    fn extract_diag(axis_a: usize, axis_b: usize) -> Self {
        StdTensorOp::ExtractDiag { axis_a, axis_b }
    }

    fn embed_diag(axis_a: usize, axis_b: usize) -> Self {
        StdTensorOp::EmbedDiag { axis_a, axis_b }
    }
}
