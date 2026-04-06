use std::hash::{Hash, Hasher};

use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::GraphOp;
use num_complex::{Complex32, Complex64};

use crate::input_key::TensorInputKey;
use crate::semiring_ops::SemiringOps;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

#[derive(Clone, Debug, PartialEq)]
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
    Constant {
        dtype: DType,
        bytes: Vec<u8>,
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
    Tril {
        k: i64,
    },
    Triu {
        k: i64,
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
    Cholesky {
        input_shape: Vec<usize>,
    },
    Svd {
        eps: f64,
        input_shape: Vec<usize>,
    },
    Qr {
        input_shape: Vec<usize>,
    },
    Eigh {
        eps: f64,
        input_shape: Vec<usize>,
    },
    Solve {
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },
}

impl StdTensorOp {
    /// Create an `f64` scalar constant op.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    ///
    /// let op = StdTensorOp::constant_f64(1.5);
    /// ```
    pub fn constant_f64(value: f64) -> Self {
        Self::Constant {
            dtype: DType::F64,
            bytes: value.to_le_bytes().to_vec(),
        }
    }

    /// Create an `f32` scalar constant op.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    ///
    /// let op = StdTensorOp::constant_f32(1.5_f32);
    /// ```
    pub fn constant_f32(value: f32) -> Self {
        Self::Constant {
            dtype: DType::F32,
            bytes: value.to_le_bytes().to_vec(),
        }
    }

    /// Create a `Complex64` scalar constant op.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    ///
    /// let op = StdTensorOp::constant_c64(Complex64::new(1.0, -2.0));
    /// ```
    pub fn constant_c64(value: Complex64) -> Self {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&value.re.to_le_bytes());
        bytes.extend_from_slice(&value.im.to_le_bytes());
        Self::Constant {
            dtype: DType::C64,
            bytes,
        }
    }

    /// Create a `Complex32` scalar constant op.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex32;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    ///
    /// let op = StdTensorOp::constant_c32(Complex32::new(1.0, -2.0));
    /// ```
    pub fn constant_c32(value: Complex32) -> Self {
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&value.re.to_le_bytes());
        bytes.extend_from_slice(&value.im.to_le_bytes());
        Self::Constant {
            dtype: DType::C32,
            bytes,
        }
    }
}

impl Eq for StdTensorOp {}

impl Hash for StdTensorOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Add
            | Self::Mul
            | Self::Neg
            | Self::Conj
            | Self::Div
            | Self::Abs
            | Self::Sign
            | Self::Maximum
            | Self::Minimum
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
            | Self::Log1p => {}
            Self::Svd { eps, input_shape } => {
                hash_f64(*eps, state);
                input_shape.hash(state);
            }
            Self::Qr { input_shape } | Self::Cholesky { input_shape } => {
                input_shape.hash(state);
            }
            Self::Eigh { eps, input_shape } => {
                hash_f64(*eps, state);
                input_shape.hash(state);
            }
            Self::Solve {
                lhs_shape,
                rhs_shape,
            } => {
                lhs_shape.hash(state);
                rhs_shape.hash(state);
            }
            Self::DotGeneral(config) => config.hash(state),
            Self::Transpose { perm } => perm.hash(state),
            Self::Reshape {
                from_shape,
                to_shape,
            } => {
                from_shape.hash(state);
                to_shape.hash(state);
            }
            Self::BroadcastInDim { shape, dims } => {
                shape.hash(state);
                dims.hash(state);
            }
            Self::Constant { dtype, bytes } => {
                dtype.hash(state);
                bytes.hash(state);
            }
            Self::ReduceSum { axes, input_shape } => {
                axes.hash(state);
                input_shape.hash(state);
            }
            Self::Compare(dir) => dir.hash(state),
            Self::ExtractDiag { axis_a, axis_b } | Self::EmbedDiag { axis_a, axis_b } => {
                axis_a.hash(state);
                axis_b.hash(state);
            }
            Self::Tril { k } | Self::Triu { k } => k.hash(state),
            Self::Gather(config) => config.hash(state),
            Self::Scatter(config) => config.hash(state),
            Self::Slice(config) => config.hash(state),
            Self::DynamicSlice { slice_sizes } => slice_sizes.hash(state),
            Self::Pad(config) => config.hash(state),
            Self::Concatenate { axis } => axis.hash(state),
            Self::Reverse { axes } => axes.hash(state),
            Self::ReduceProd { axes } | Self::ReduceMax { axes } | Self::ReduceMin { axes } => {
                axes.hash(state);
            }
            Self::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                lhs_shape,
                rhs_shape,
            } => {
                left_side.hash(state);
                lower.hash(state);
                transpose_a.hash(state);
                unit_diagonal.hash(state);
                lhs_shape.hash(state);
                rhs_shape.hash(state);
            }
        }
    }
}

fn hash_f64<H: Hasher>(value: f64, state: &mut H) {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    bits.hash(state);
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
            | Self::Tril { .. }
            | Self::Triu { .. }
            | Self::Slice(_)
            | Self::Pad(_)
            | Self::Reverse { .. } => 1,
            Self::Div | Self::Maximum | Self::Minimum | Self::Pow | Self::DynamicSlice { .. } => 2,
            Self::Constant { .. } => 0,
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
            Self::Cholesky { .. } | Self::Svd { .. } | Self::Qr { .. } | Self::Eigh { .. } => 1,
            Self::Solve { .. } | Self::TriangularSolve { .. } => 2,
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
            | Self::Constant { .. }
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
            | Self::Tril { .. }
            | Self::Triu { .. }
            | Self::Gather(_)
            | Self::Scatter(_)
            | Self::Slice(_)
            | Self::DynamicSlice { .. }
            | Self::Pad(_)
            | Self::Reverse { .. } => 1,
            Self::Cholesky { .. } | Self::Solve { .. } | Self::TriangularSolve { .. } => 1,
            Self::Svd { .. } => 3,  // U, S, Vt
            Self::Qr { .. } => 2,   // Q, R
            Self::Eigh { .. } => 2, // eigenvalues, eigenvectors
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
