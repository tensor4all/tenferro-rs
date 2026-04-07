use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

#[derive(Clone, Debug)]
pub enum StableHloOp {
    // Tier 1 (AD-closed core)
    Add,
    Multiply,
    Negate,
    Conj,
    DotGeneral(DotGeneralConfig),
    Transpose {
        perm: Vec<usize>,
    },
    Reshape {
        shape: Vec<usize>,
    },
    BroadcastInDim {
        shape: Vec<usize>,
        dims: Vec<usize>,
    },
    Convert {
        to: DType,
    },
    Constant {
        dtype: DType,
        bytes: Vec<u8>,
    },
    ReduceSum {
        axes: Vec<usize>,
    },
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
    // Tier 2 elementwise
    Divide,
    Abs,
    Sign,
    Maximum,
    Minimum,
    Compare(CompareDir),
    Select,
    Clamp,
    // Tier 2 analytic
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
    // Indexing
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
    // Additional reductions
    ReduceProd {
        axes: Vec<usize>,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    // Linalg: only Cholesky is a direct StableHLO op
    Cholesky,
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    },
    // All other linalg (SVD, QR, Eigh, Solve) lower to custom_call
    CustomCall {
        target: String,
    },
    // Multi-output access
    GetTupleElement {
        index: usize,
    },
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
