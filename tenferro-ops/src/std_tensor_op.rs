use std::hash::{Hash, Hasher};

use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{GraphOp, OpEmitter};
use num_complex::{Complex32, Complex64};

use crate::dim_expr::DimExpr;
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
        from_shape: Vec<DimExpr>,
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
        input_shape: Vec<DimExpr>,
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
    /// N-ary einsum kept as a single graph node.
    /// Contraction path is optimized at execution time from actual input shapes.
    NaryEinsum {
        subscripts: String,
        n_inputs: usize,
    },
    Concatenate {
        axis: usize,
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

    // Tier 2: reductions
    ReduceProd {
        axes: Vec<usize>,
        input_shape: Vec<DimExpr>,
    },
    ReduceMax {
        axes: Vec<usize>,
        input_shape: Vec<DimExpr>,
    },
    ReduceMin {
        axes: Vec<usize>,
        input_shape: Vec<DimExpr>,
    },

    // Linalg
    Cholesky {
        input_shape: Vec<DimExpr>,
    },
    Lu {
        input_shape: Vec<DimExpr>,
    },
    Svd {
        eps: f64,
        input_shape: Vec<DimExpr>,
    },
    Qr {
        input_shape: Vec<DimExpr>,
    },
    Eigh {
        eps: f64,
        input_shape: Vec<DimExpr>,
    },
    Eig {
        input_dtype: DType,
        input_shape: Vec<DimExpr>,
    },
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        lhs_shape: Vec<DimExpr>,
        rhs_shape: Vec<DimExpr>,
    },
    ValidateNonsingular {
        input_shape: Vec<DimExpr>,
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
            Self::Qr { input_shape }
            | Self::Cholesky { input_shape }
            | Self::Lu { input_shape } => {
                input_shape.hash(state);
            }
            Self::Eig {
                input_dtype,
                input_shape,
            } => {
                input_dtype.hash(state);
                input_shape.hash(state);
            }
            Self::Eigh { eps, input_shape } => {
                hash_f64(*eps, state);
                input_shape.hash(state);
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
            Self::Convert { from, to } => {
                from.hash(state);
                to.hash(state);
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
            Self::NaryEinsum {
                subscripts,
                n_inputs,
            } => {
                subscripts.hash(state);
                n_inputs.hash(state);
            }
            Self::Concatenate { axis } => axis.hash(state),
            Self::Reverse { axes } => axes.hash(state),
            Self::ShapeOf { axis } | Self::DynamicTruncate { axis } | Self::PadToMatch { axis } => {
                axis.hash(state)
            }
            Self::ReduceProd { axes, input_shape }
            | Self::ReduceMax { axes, input_shape }
            | Self::ReduceMin { axes, input_shape } => {
                axes.hash(state);
                input_shape.hash(state);
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
            Self::ValidateNonsingular { input_shape } => {
                input_shape.hash(state);
            }
        }
    }
}

fn hash_f64<H: Hasher>(value: f64, state: &mut H) {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    bits.hash(state);
}

fn n_inputs_from_dim_exprs(min_inputs: usize, exprs: &[&[DimExpr]]) -> usize {
    let max_idx = exprs
        .iter()
        .flat_map(|exprs| exprs.iter())
        .filter_map(DimExpr::max_input_idx)
        .max()
        .map_or(0, |max_idx| max_idx + 1);
    max_idx.max(min_inputs)
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
            | Self::Convert { .. }
            | Self::ExtractDiag { .. }
            | Self::EmbedDiag { .. }
            | Self::Tril { .. }
            | Self::Triu { .. }
            | Self::Slice(_)
            | Self::Pad(_)
            | Self::Reverse { .. }
            | Self::ShapeOf { .. } => 1,
            Self::DynamicTruncate { .. } | Self::PadToMatch { .. } => 2,
            Self::Reshape {
                from_shape,
                to_shape,
            } => n_inputs_from_dim_exprs(1, &[from_shape, to_shape]),
            Self::BroadcastInDim { shape, .. } => n_inputs_from_dim_exprs(1, &[shape]),
            Self::ReduceSum { input_shape, .. }
            | Self::ReduceProd { input_shape, .. }
            | Self::ReduceMax { input_shape, .. }
            | Self::ReduceMin { input_shape, .. } => n_inputs_from_dim_exprs(1, &[input_shape]),
            Self::Div | Self::Maximum | Self::Minimum | Self::Pow | Self::DynamicSlice { .. } => 2,
            Self::Constant { .. } => 0,
            Self::Scatter(_) => 3,
            Self::NaryEinsum { n_inputs, .. } => *n_inputs,
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
            Self::Cholesky { input_shape }
            | Self::Lu { input_shape }
            | Self::Svd { input_shape, .. }
            | Self::Qr { input_shape }
            | Self::Eigh { input_shape, .. }
            | Self::Eig { input_shape, .. } => n_inputs_from_dim_exprs(1, &[input_shape]),
            Self::TriangularSolve {
                lhs_shape,
                rhs_shape,
                ..
            } => n_inputs_from_dim_exprs(2, &[lhs_shape, rhs_shape]),
            Self::ValidateNonsingular { input_shape } => n_inputs_from_dim_exprs(1, &[input_shape]),
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
            | Self::Convert { .. }
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
            | Self::NaryEinsum { .. }
            | Self::Reverse { .. }
            | Self::ShapeOf { .. }
            | Self::DynamicTruncate { .. }
            | Self::PadToMatch { .. }
            | Self::ReduceProd { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. } => 1,
            Self::Cholesky { .. }
            | Self::TriangularSolve { .. }
            | Self::ValidateNonsingular { .. } => 1,
            Self::Lu { .. } => 4,
            Self::Svd { .. } => 3,  // U, S, Vt
            Self::Qr { .. } => 2,   // Q, R
            Self::Eigh { .. } => 2, // eigenvalues, eigenvectors
            Self::Eig { .. } => 2,  // eigenvalues, eigenvectors
            Self::Concatenate { .. } => todo!(
                "n_outputs not yet implemented for variable-arity op {:?}",
                self
            ),
        }
    }
}

impl PrimitiveOp for StdTensorOp {
    type ADContext = crate::ad::context::ShapeGuardContext;

    fn add() -> Self {
        StdTensorOp::Add
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<Self>,
        primal_in: &[GlobalValKey<Self>],
        primal_out: &[GlobalValKey<Self>],
        tangent_in: &[Option<LocalValId>],
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>> {
        crate::ad::linearize(self, builder, primal_in, primal_out, tangent_in, ctx)
    }

    fn transpose_rule(
        &self,
        emitter: &mut impl OpEmitter<Self>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<Self>],
        mode: &OpMode,
        ctx: &mut Self::ADContext,
    ) -> Vec<Option<LocalValId>> {
        crate::ad::transpose_rule(self, emitter, cotangent_out, inputs, mode, ctx)
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

    fn reduce_sum(axes: Vec<usize>, input_shape: Vec<DimExpr>) -> Self {
        StdTensorOp::ReduceSum { axes, input_shape }
    }

    fn transpose_op(perm: Vec<usize>) -> Self {
        StdTensorOp::Transpose { perm }
    }

    fn reshape(from_shape: Vec<DimExpr>, to_shape: Vec<DimExpr>) -> Self {
        StdTensorOp::Reshape {
            from_shape,
            to_shape,
        }
    }

    fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self {
        StdTensorOp::BroadcastInDim { shape, dims }
    }

    fn extract_diag(axis_a: usize, axis_b: usize) -> Self {
        StdTensorOp::ExtractDiag { axis_a, axis_b }
    }

    fn embed_diag(axis_a: usize, axis_b: usize) -> Self {
        StdTensorOp::EmbedDiag { axis_a, axis_b }
    }
}
