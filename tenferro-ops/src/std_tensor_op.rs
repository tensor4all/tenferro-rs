use std::hash::{Hash, Hasher};
use std::sync::Arc;

use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{GraphOp, OpEmitter};
use num_complex::{Complex32, Complex64};

use crate::dim_expr::DimExpr;
use crate::ext_op::{ext_op_eq, hash_extension, ExtensionOp};
use crate::input_key::TensorInputKey;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};

#[derive(Clone, Debug)]
pub enum StdTensorOp {
    // Tier 1: semiring
    Add,
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
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceMin {
        axes: Vec<usize>,
    },

    // Linalg
    Cholesky,
    Lu,
    Svd {
        eps: f64,
    },
    Qr,
    Eigh {
        eps: f64,
    },
    Eig {
        input_dtype: DType,
    },
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    },
    ValidateNonsingular,

    /// Out-of-tree extension carrier.
    ///
    /// See [`crate::ext_op`] and `docs/spec/extension-op.md`. Identity,
    /// hashing, equality, arity, shape inference, and AD rules are delegated
    /// to the inner [`ExtensionOp`] trait object.
    Extension(Arc<dyn ExtensionOp>),
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

impl PartialEq for StdTensorOp {
    fn eq(&self, other: &Self) -> bool {
        if std::mem::discriminant(self) != std::mem::discriminant(other) {
            return false;
        }
        match (self, other) {
            (Self::Add, Self::Add)
            | (Self::Mul, Self::Mul)
            | (Self::Neg, Self::Neg)
            | (Self::Conj, Self::Conj)
            | (Self::Div, Self::Div)
            | (Self::Abs, Self::Abs)
            | (Self::Sign, Self::Sign)
            | (Self::Maximum, Self::Maximum)
            | (Self::Minimum, Self::Minimum)
            | (Self::Select, Self::Select)
            | (Self::Clamp, Self::Clamp)
            | (Self::Exp, Self::Exp)
            | (Self::Log, Self::Log)
            | (Self::Sin, Self::Sin)
            | (Self::Cos, Self::Cos)
            | (Self::Tanh, Self::Tanh)
            | (Self::Sqrt, Self::Sqrt)
            | (Self::Rsqrt, Self::Rsqrt)
            | (Self::Pow, Self::Pow)
            | (Self::Expm1, Self::Expm1)
            | (Self::Log1p, Self::Log1p)
            | (Self::Cholesky, Self::Cholesky)
            | (Self::Lu, Self::Lu)
            | (Self::Qr, Self::Qr)
            | (Self::ValidateNonsingular, Self::ValidateNonsingular) => true,
            (Self::DotGeneral { config: a }, Self::DotGeneral { config: b }) => a == b,
            (Self::Transpose { perm: a }, Self::Transpose { perm: b }) => a == b,
            (Self::Reshape { to_shape: a }, Self::Reshape { to_shape: b }) => a == b,
            (
                Self::BroadcastInDim {
                    shape: sa,
                    dims: da,
                },
                Self::BroadcastInDim {
                    shape: sb,
                    dims: db,
                },
            ) => sa == sb && da == db,
            (Self::Convert { from: fa, to: ta }, Self::Convert { from: fb, to: tb }) => {
                fa == fb && ta == tb
            }
            (
                Self::Constant {
                    dtype: da,
                    bytes: ba,
                },
                Self::Constant {
                    dtype: db,
                    bytes: bb,
                },
            ) => da == db && ba == bb,
            (Self::ReduceSum { axes: a }, Self::ReduceSum { axes: b })
            | (Self::ReduceProd { axes: a }, Self::ReduceProd { axes: b })
            | (Self::ReduceMax { axes: a }, Self::ReduceMax { axes: b })
            | (Self::ReduceMin { axes: a }, Self::ReduceMin { axes: b })
            | (Self::Reverse { axes: a }, Self::Reverse { axes: b }) => a == b,
            (Self::Compare(a), Self::Compare(b)) => a == b,
            (
                Self::ExtractDiag {
                    axis_a: aa,
                    axis_b: ba,
                },
                Self::ExtractDiag {
                    axis_a: ab,
                    axis_b: bb,
                },
            )
            | (
                Self::EmbedDiag {
                    axis_a: aa,
                    axis_b: ba,
                },
                Self::EmbedDiag {
                    axis_a: ab,
                    axis_b: bb,
                },
            ) => aa == ab && ba == bb,
            (Self::Tril { k: a }, Self::Tril { k: b })
            | (Self::Triu { k: a }, Self::Triu { k: b }) => a == b,
            (Self::Gather(a), Self::Gather(b)) => a == b,
            (Self::Scatter(a), Self::Scatter(b)) => a == b,
            (Self::Slice(a), Self::Slice(b)) => a == b,
            (Self::DynamicSlice { slice_sizes: a }, Self::DynamicSlice { slice_sizes: b }) => {
                a == b
            }
            (Self::Pad(a), Self::Pad(b)) => a == b,
            (Self::NaryEinsum { subscripts: a }, Self::NaryEinsum { subscripts: b }) => a == b,
            (Self::Concatenate { axis: a }, Self::Concatenate { axis: b }) => a == b,
            (Self::ShapeOf { axis: a }, Self::ShapeOf { axis: b })
            | (Self::DynamicTruncate { axis: a }, Self::DynamicTruncate { axis: b })
            | (Self::PadToMatch { axis: a }, Self::PadToMatch { axis: b }) => a == b,
            (Self::Svd { eps: a }, Self::Svd { eps: b })
            | (Self::Eigh { eps: a }, Self::Eigh { eps: b }) => a.to_bits() == b.to_bits(),
            (Self::Eig { input_dtype: a }, Self::Eig { input_dtype: b }) => a == b,
            (
                Self::TriangularSolve {
                    left_side: lsa,
                    lower: la,
                    transpose_a: ta,
                    unit_diagonal: ua,
                },
                Self::TriangularSolve {
                    left_side: lsb,
                    lower: lb,
                    transpose_a: tb,
                    unit_diagonal: ub,
                },
            ) => lsa == lsb && la == lb && ta == tb && ua == ub,
            (Self::Extension(a), Self::Extension(b)) => ext_op_eq(a.as_ref(), b.as_ref()),
            _ => unreachable!("discriminant mismatch should be caught earlier"),
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
            Self::Svd { eps } => {
                hash_f64(*eps, state);
            }
            Self::Qr | Self::Cholesky | Self::Lu => {}
            Self::Eig { input_dtype } => {
                input_dtype.hash(state);
            }
            Self::Eigh { eps } => {
                hash_f64(*eps, state);
            }
            Self::DotGeneral { config } => {
                config.hash(state);
            }
            Self::Transpose { perm } => perm.hash(state),
            Self::Reshape { to_shape } => {
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
            Self::ReduceSum { axes } => {
                axes.hash(state);
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
            Self::NaryEinsum { subscripts } => {
                subscripts.hash(state);
            }
            Self::Concatenate { axis } => axis.hash(state),
            Self::Reverse { axes } => axes.hash(state),
            Self::ShapeOf { axis } | Self::DynamicTruncate { axis } | Self::PadToMatch { axis } => {
                axis.hash(state)
            }
            Self::ReduceProd { axes } | Self::ReduceMax { axes } | Self::ReduceMin { axes } => {
                axes.hash(state);
            }
            Self::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            } => {
                left_side.hash(state);
                lower.hash(state);
                transpose_a.hash(state);
                unit_diagonal.hash(state);
            }
            Self::ValidateNonsingular => {}
            Self::Extension(op) => hash_extension(op.as_ref(), state),
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

/// Count the number of inputs in an einsum subscript string.
///
/// The subscript string is expected to be in the form `"<in_1>,<in_2>,...-><out>"`.
/// This counts comma-separated input subscripts before the `->` marker.
pub(crate) fn n_inputs_from_einsum_subscripts(subscripts: &str) -> usize {
    let input_part = subscripts.split_once("->").map_or(subscripts, |(i, _)| i);
    input_part.split(',').count()
}

impl GraphOp for StdTensorOp {
    type Operand = tenferro_tensor::Tensor;
    type Context = ();
    type InputKey = TensorInputKey;

    fn n_inputs(&self) -> usize {
        match self {
            Self::Add | Self::Mul | Self::DotGeneral { .. } | Self::Gather(_) => 2,
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
            Self::Reshape { to_shape } => n_inputs_from_dim_exprs(1, &[to_shape]),
            Self::BroadcastInDim { shape, .. } => n_inputs_from_dim_exprs(1, &[shape]),
            Self::ReduceSum { .. }
            | Self::ReduceProd { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. } => 1,
            Self::Div | Self::Maximum | Self::Minimum | Self::Pow | Self::DynamicSlice { .. } => 2,
            Self::Constant { .. } => 0,
            Self::Scatter(_) => 3,
            Self::NaryEinsum { subscripts } => n_inputs_from_einsum_subscripts(subscripts),
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
            Self::Cholesky
            | Self::Lu
            | Self::Svd { .. }
            | Self::Qr
            | Self::Eigh { .. }
            | Self::Eig { .. }
            | Self::ValidateNonsingular => 1,
            Self::TriangularSolve { .. } => 2,
            Self::Extension(op) => ExtensionOp::n_inputs(op.as_ref()),
        }
    }

    fn n_outputs(&self) -> usize {
        match self {
            Self::Add
            | Self::Mul
            | Self::Neg
            | Self::Conj
            | Self::DotGeneral { .. }
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
            Self::Cholesky | Self::TriangularSolve { .. } | Self::ValidateNonsingular => 1,
            Self::Lu => 4,
            Self::Svd { .. } => 3,  // U, S, Vt
            Self::Qr => 2,          // Q, R
            Self::Eigh { .. } => 2, // eigenvalues, eigenvectors
            Self::Eig { .. } => 2,  // eigenvalues, eigenvectors
            Self::Concatenate { .. } => todo!(
                "n_outputs not yet implemented for variable-arity op {:?}",
                self
            ),
            Self::Extension(op) => ExtensionOp::n_outputs(op.as_ref()),
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
