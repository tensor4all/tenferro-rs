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
    type Operand = tenferro_tensor::v2::Tensor;
    type Context = ();
    type InputKey = TensorInputKey;

    fn n_inputs(&self) -> usize {
        todo!()
    }

    fn n_outputs(&self) -> usize {
        todo!()
    }

    fn eval(&self, _ctx: &mut Self::Context, _inputs: &[&Self::Operand]) -> Vec<Self::Operand> {
        todo!()
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
}
