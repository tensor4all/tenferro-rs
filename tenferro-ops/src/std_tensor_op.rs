use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::GraphOp;

use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::input_key::TensorInputKey;
use crate::semiring_ops::SemiringOps;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum StdTensorOp {
    // Tier 1: semiring
    Add,
    Mul,
    Neg,
    Conj,
    DotGeneral(DotGeneralConfig),
    Transpose { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
    ReduceSum { axes: Vec<usize> },

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

    // Tier 1: diagonal extraction (AD-closed with Scatter)
    ExtractDiag { axis_a: usize, axis_b: usize },

    // Tier 2: indexing
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice,
    Pad(PadConfig),
    Concatenate { axis: usize },
    Reverse { axes: Vec<usize> },

    // Tier 2: reductions
    ReduceProd { axes: Vec<usize> },
    ReduceMax { axes: Vec<usize> },
    ReduceMin { axes: Vec<usize> },

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
            Self::Add | Self::Mul | Self::DotGeneral(_) => 2,
            Self::Neg
            | Self::Conj
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::BroadcastInDim { .. }
            | Self::ReduceSum { .. }
            | Self::ExtractDiag { .. } => 1,
            Self::Div | Self::Maximum | Self::Minimum | Self::Pow => 2,
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
            | Self::ExtractDiag { .. } => 1,
            _ => todo!("n_outputs not yet implemented for {:?}", self),
        }
    }

    fn eval(&self, _ctx: &mut Self::Context, inputs: &[&Self::Operand]) -> Vec<Self::Operand> {
        use computegraph::Operand;
        match self {
            Self::Add => vec![inputs[0].add(inputs[1])],
            Self::Mul => vec![inputs[0].multiply(inputs[1])],
            Self::Neg => vec![inputs[0].neg()],
            Self::Conj => vec![inputs[0].conj()],
            Self::DotGeneral(config) => vec![inputs[0].dot_general(
                inputs[1],
                &config.lhs_contracting_dims,
                &config.rhs_contracting_dims,
                &config.lhs_batch_dims,
                &config.rhs_batch_dims,
            )],
            Self::Transpose { perm } => vec![inputs[0].transpose_perm(perm)],
            Self::Reshape { shape } => vec![inputs[0].reshape(shape)],
            Self::BroadcastInDim { shape, dims } => {
                vec![inputs[0].broadcast_in_dim(shape, dims)]
            }
            Self::ReduceSum { axes } => vec![inputs[0].reduce_sum(axes)],
            Self::ExtractDiag { axis_a, axis_b } => {
                vec![inputs[0].extract_diagonal(*axis_a, *axis_b)]
            }
            _ => todo!("eval not yet implemented for {:?}", self),
        }
    }
}

impl PrimitiveOp for StdTensorOp {
    fn add() -> Self {
        StdTensorOp::Add
    }

    fn linearize(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        _tangent_in: &[Option<LocalValId>],
    ) -> Vec<Option<LocalValId>> {
        todo!()
    }

    fn transpose_rule(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
    ) -> Vec<Option<LocalValId>> {
        todo!()
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

    fn reduce_sum(axes: Vec<usize>) -> Self {
        StdTensorOp::ReduceSum { axes }
    }

    fn transpose_op(perm: Vec<usize>) -> Self {
        StdTensorOp::Transpose { perm }
    }

    fn reshape(shape: Vec<usize>) -> Self {
        StdTensorOp::Reshape { shape }
    }

    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self {
        StdTensorOp::BroadcastInDim { shape, dims }
    }

    fn extract_diag(axis_a: usize, axis_b: usize) -> Self {
        StdTensorOp::ExtractDiag { axis_a, axis_b }
    }
}
