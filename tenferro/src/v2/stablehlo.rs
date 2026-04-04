use tenferro_ops::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::v2::DType;

#[derive(Clone, Debug)]
pub enum StableHloOp {
    Add,
    Multiply,
    Negate,
    Conj,
    DotGeneral(DotGeneralConfig),
    Transpose { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
    ReduceSum { axes: Vec<usize> },
    Divide,
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
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice,
    Pad(PadConfig),
    Concatenate { axis: usize },
    Reverse { axes: Vec<usize> },
    ReduceProd { axes: Vec<usize> },
    ReduceMax { axes: Vec<usize> },
    ReduceMin { axes: Vec<usize> },
    Cholesky,
    Svd,
    Qr,
    Eigh,
    Solve,
}

#[derive(Clone, Debug)]
pub struct StableHloInstruction {
    pub op: StableHloOp,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub dtype: DType,
}

#[derive(Clone, Debug)]
pub struct StableHloProgram {
    pub instructions: Vec<StableHloInstruction>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
}
