use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[cfg(all(test, feature = "autodiff"))]
use crate::ad::{ADRuleResult, PrimitiveTransposeInput};
#[cfg(all(test, feature = "autodiff"))]
use computegraph::types::{LocalValueId, OperationRole, ValueKey};
use computegraph::GraphOperation;
use num_complex::{Complex32, Complex64};

use crate::dim_expr::DimExpr;
use crate::ext_op::{ext_op_eq, hash_extension, ExtensionOp};
use crate::input_key::TensorInputKey;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    TensorScalar,
};

/// Scalar values that can be encoded as tensor constant operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_ops::std_tensor_op::ConstantScalar;
///
/// assert_eq!(1.0_f64.constant_bytes(), 1.0_f64.to_le_bytes().to_vec());
/// ```
pub trait ConstantScalar: TensorScalar + private::Sealed {
    /// Encode the scalar value as little-endian constant bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ops::std_tensor_op::ConstantScalar;
    ///
    /// assert_eq!(true.constant_bytes(), vec![1]);
    /// ```
    fn constant_bytes(self) -> Vec<u8>;
}

mod private {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for i64 {}
    impl Sealed for i32 {}
    impl Sealed for bool {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
}

impl ConstantScalar for f64 {
    fn constant_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ConstantScalar for f32 {
    fn constant_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ConstantScalar for i64 {
    fn constant_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ConstantScalar for i32 {
    fn constant_bytes(self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ConstantScalar for bool {
    fn constant_bytes(self) -> Vec<u8> {
        vec![u8::from(self)]
    }
}

impl ConstantScalar for Complex64 {
    fn constant_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&self.re.to_le_bytes());
        bytes.extend_from_slice(&self.im.to_le_bytes());
        bytes
    }
}

impl ConstantScalar for Complex32 {
    fn constant_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8);
        bytes.extend_from_slice(&self.re.to_le_bytes());
        bytes.extend_from_slice(&self.im.to_le_bytes());
        bytes
    }
}

tenferro_core_ops::define_std_tensor_op!();

impl StdTensorOp {
    /// Create a scalar constant op from any supported tensor scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_ops::std_tensor_op::StdTensorOp;
    /// use tenferro_tensor::DType;
    ///
    /// let real = StdTensorOp::constant(1.5_f64);
    /// let complex = StdTensorOp::constant(Complex64::new(1.0, -2.0));
    ///
    /// assert!(matches!(real, StdTensorOp::Constant { dtype: DType::F64, .. }));
    /// assert!(matches!(complex, StdTensorOp::Constant { dtype: DType::C64, .. }));
    /// ```
    pub fn constant<T: ConstantScalar>(value: T) -> Self {
        Self::Constant {
            dtype: T::dtype(),
            bytes: value.constant_bytes(),
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
            | (Self::Sub, Self::Sub)
            | (Self::Mul, Self::Mul)
            | (Self::Neg, Self::Neg)
            | (Self::Conj, Self::Conj)
            | (Self::Div, Self::Div)
            | (Self::Rem, Self::Rem)
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
            | (Self::DynamicUpdateSlice, Self::DynamicUpdateSlice) => true,
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
            | (Self::ReduceSumSquares { axes: a }, Self::ReduceSumSquares { axes: b })
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
            (
                Self::GatherDynamicSliceSizes {
                    offset_dims: oa,
                    collapsed_slice_dims: ca,
                    start_index_map: sa,
                    index_vector_dim: ia,
                    slice_sizes: za,
                },
                Self::GatherDynamicSliceSizes {
                    offset_dims: ob,
                    collapsed_slice_dims: cb,
                    start_index_map: sb,
                    index_vector_dim: ib,
                    slice_sizes: zb,
                },
            ) => oa == ob && ca == cb && sa == sb && ia == ib && za == zb,
            (Self::Scatter(a), Self::Scatter(b)) => a == b,
            (Self::Slice(a), Self::Slice(b)) => a == b,
            (Self::DynamicSlice { slice_sizes: a }, Self::DynamicSlice { slice_sizes: b }) => {
                a == b
            }
            (Self::Pad(a), Self::Pad(b)) => a == b,
            (
                Self::Concatenate {
                    axis: a,
                    input_count: na,
                },
                Self::Concatenate {
                    axis: b,
                    input_count: nb,
                },
            ) => a == b && na == nb,
            (Self::ShapeOf { axis: a }, Self::ShapeOf { axis: b })
            | (Self::DynamicTruncate { axis: a }, Self::DynamicTruncate { axis: b })
            | (Self::PadToMatch { axis: a }, Self::PadToMatch { axis: b }) => a == b,
            (Self::Extension(a), Self::Extension(b)) => ext_op_eq(a.as_ref(), b.as_ref()),
            _ => false,
        }
    }
}

impl Eq for StdTensorOp {}

impl Hash for StdTensorOp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Neg
            | Self::Conj
            | Self::Div
            | Self::Rem
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
            Self::ReduceSum { axes } | Self::ReduceSumSquares { axes } => {
                axes.hash(state);
            }
            Self::Compare(dir) => dir.hash(state),
            Self::ExtractDiag { axis_a, axis_b } | Self::EmbedDiag { axis_a, axis_b } => {
                axis_a.hash(state);
                axis_b.hash(state);
            }
            Self::Tril { k } | Self::Triu { k } => k.hash(state),
            Self::Gather(config) => config.hash(state),
            Self::GatherDynamicSliceSizes {
                offset_dims,
                collapsed_slice_dims,
                start_index_map,
                index_vector_dim,
                slice_sizes,
            } => {
                offset_dims.hash(state);
                collapsed_slice_dims.hash(state);
                start_index_map.hash(state);
                index_vector_dim.hash(state);
                slice_sizes.hash(state);
            }
            Self::Scatter(config) => config.hash(state),
            Self::Slice(config) => config.hash(state),
            Self::DynamicSlice { slice_sizes } => slice_sizes.hash(state),
            Self::DynamicUpdateSlice => {}
            Self::Pad(config) => config.hash(state),
            Self::Concatenate { axis, input_count } => {
                axis.hash(state);
                input_count.hash(state);
            }
            Self::Reverse { axes } => axes.hash(state),
            Self::ShapeOf { axis } | Self::DynamicTruncate { axis } | Self::PadToMatch { axis } => {
                axis.hash(state)
            }
            Self::ReduceProd { axes } | Self::ReduceMax { axes } | Self::ReduceMin { axes } => {
                axes.hash(state);
            }
            Self::Extension(op) => hash_extension(op.as_ref(), state),
        }
    }
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

impl GraphOperation for StdTensorOp {
    type Operand = tenferro_tensor::Tensor;
    type Context = ();
    type InputKey = TensorInputKey;

    fn input_count(&self) -> usize {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::DotGeneral { .. } | Self::Gather(_) => 2,
            Self::GatherDynamicSliceSizes { slice_sizes, .. } => {
                n_inputs_from_dim_exprs(2, &[slice_sizes])
            }
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
            | Self::ReduceSumSquares { .. }
            | Self::ReduceProd { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. } => 1,
            Self::Div
            | Self::Rem
            | Self::Maximum
            | Self::Minimum
            | Self::Pow
            | Self::DynamicSlice { .. } => 2,
            Self::Constant { .. } => 0,
            Self::Scatter(_) | Self::DynamicUpdateSlice => 3,
            Self::Concatenate { input_count, .. } => *input_count,
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
            Self::Extension(op) => ExtensionOp::input_count(op.as_ref()),
        }
    }

    fn output_count(&self) -> usize {
        match self {
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Neg
            | Self::Conj
            | Self::DotGeneral { .. }
            | Self::Transpose { .. }
            | Self::Reshape { .. }
            | Self::BroadcastInDim { .. }
            | Self::Convert { .. }
            | Self::ReduceSum { .. }
            | Self::ReduceSumSquares { .. }
            | Self::Div
            | Self::Rem
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
            | Self::GatherDynamicSliceSizes { .. }
            | Self::Scatter(_)
            | Self::Slice(_)
            | Self::DynamicSlice { .. }
            | Self::DynamicUpdateSlice
            | Self::Pad(_)
            | Self::Reverse { .. }
            | Self::ShapeOf { .. }
            | Self::DynamicTruncate { .. }
            | Self::PadToMatch { .. }
            | Self::ReduceProd { .. }
            | Self::ReduceMax { .. }
            | Self::ReduceMin { .. } => 1,
            Self::Concatenate { .. } => 1,
            Self::Extension(op) => ExtensionOp::output_count(op.as_ref()),
        }
    }
}

#[cfg(all(test, feature = "autodiff"))]
impl StdTensorOp {
    pub(crate) fn jvp_rule(
        &self,
        builder: &mut computegraph::graph::GraphBuilder<Self>,
        primal_in: &[ValueKey<Self>],
        primal_out: &[ValueKey<Self>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut crate::ad::context::ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        crate::ad::linearize(self, builder, primal_in, primal_out, tangent_in, ctx)
    }

    pub(crate) fn transpose_rule(
        &self,
        builder: &mut computegraph::graph::GraphBuilder<Self>,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[computegraph::ValueRef<Self>],
        mode: &OperationRole,
        ctx: &mut crate::ad::context::ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let inputs = inputs
            .iter()
            .map(|input| match input {
                computegraph::ValueRef::Local(local_id) => {
                    let key = builder.global_key(*local_id).clone();
                    PrimitiveTransposeInput::Residual(key)
                }
                computegraph::ValueRef::External(key) => {
                    PrimitiveTransposeInput::Residual(key.clone())
                }
            })
            .collect::<Vec<_>>();
        crate::ad::transpose_rule(self, builder, cotangent_out, inputs.as_slice(), mode, ctx)
    }
}
