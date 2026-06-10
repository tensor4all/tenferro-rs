use cubecl::prelude::{AddressType, CubeElement, CubePrimitive};

use crate::backend::{ElementwiseFusionOp, ElementwiseFusionPlan};
use crate::{DType, Tensor, TypedTensor};

pub(crate) struct ClassifiedFusion<'a, T> {
    pub(crate) plan: &'a ElementwiseFusionPlan,
    pub(crate) inputs: Vec<&'a TypedTensor<T>>,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) n_elements: usize,
    pub(crate) address_type: AddressType,
}

pub(crate) fn classify<'a, T>(
    inputs: &[&'a Tensor],
    plan: &'a ElementwiseFusionPlan,
) -> crate::Result<Option<ClassifiedFusion<'a, T>>>
where
    T: FusionElement,
{
    if plan.dtype() != T::DTYPE || plan.input_count() != inputs.len() || plan.outputs().is_empty() {
        return Ok(None);
    }
    if !plan.ops().iter().all(|inst| T::supports_op(&inst.op())) {
        return Ok(None);
    }

    let mut typed_inputs = Vec::with_capacity(inputs.len());
    for tensor in inputs {
        typed_inputs.push(T::tensor_ref(tensor).ok_or_else(|| {
            crate::Error::backend_failure(
                "fused_elementwise",
                format!(
                    "plan dtype {:?} does not match runtime tensor dtype {:?}",
                    plan.dtype(),
                    tensor.dtype()
                ),
            )
        })?);
    }

    let Some(first) = typed_inputs.first() else {
        return Ok(None);
    };
    for input in &typed_inputs[1..] {
        if input.shape() != first.shape() {
            return Err(crate::Error::ShapeMismatch {
                op: "fused_elementwise",
                lhs: first.shape().to_vec(),
                rhs: input.shape().to_vec(),
            });
        }
    }

    let output_shape = first.shape().to_vec();
    let n_elements = first.n_elements();
    Ok(Some(ClassifiedFusion {
        plan,
        inputs: typed_inputs,
        output_shape,
        n_elements,
        address_type: AddressType::from_len(n_elements),
    }))
}

pub(crate) trait FusionElement: CubeElement + CubePrimitive + Clone + 'static {
    const DTYPE: DType;

    fn tensor_ref(tensor: &Tensor) -> Option<&TypedTensor<Self>>;

    fn supports_op(op: &ElementwiseFusionOp) -> bool;
}

impl FusionElement for f32 {
    const DTYPE: DType = DType::F32;

    fn tensor_ref(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::F32(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn supports_op(op: &ElementwiseFusionOp) -> bool {
        matches!(
            op,
            ElementwiseFusionOp::Add
                | ElementwiseFusionOp::Multiply
                | ElementwiseFusionOp::Negate
                | ElementwiseFusionOp::Divide
                | ElementwiseFusionOp::Abs
                | ElementwiseFusionOp::Maximum
                | ElementwiseFusionOp::Minimum
                | ElementwiseFusionOp::Clamp
                | ElementwiseFusionOp::Exp
                | ElementwiseFusionOp::Log
                | ElementwiseFusionOp::Sin
                | ElementwiseFusionOp::Cos
                | ElementwiseFusionOp::Tanh
                | ElementwiseFusionOp::Sqrt
                | ElementwiseFusionOp::Rsqrt
                | ElementwiseFusionOp::Pow
                | ElementwiseFusionOp::Expm1
                | ElementwiseFusionOp::Log1p
        )
    }
}

impl FusionElement for f64 {
    const DTYPE: DType = DType::F64;

    fn tensor_ref(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::F64(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn supports_op(op: &ElementwiseFusionOp) -> bool {
        <f32 as FusionElement>::supports_op(op)
    }
}

impl FusionElement for num_complex::Complex32 {
    const DTYPE: DType = DType::C32;

    fn tensor_ref(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::C32(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn supports_op(op: &ElementwiseFusionOp) -> bool {
        matches!(
            op,
            ElementwiseFusionOp::Add
                | ElementwiseFusionOp::Multiply
                | ElementwiseFusionOp::Negate
                | ElementwiseFusionOp::Conj
                | ElementwiseFusionOp::Divide
        )
    }
}

impl FusionElement for num_complex::Complex64 {
    const DTYPE: DType = DType::C64;

    fn tensor_ref(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::C64(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn supports_op(op: &ElementwiseFusionOp) -> bool {
        <num_complex::Complex32 as FusionElement>::supports_op(op)
    }
}
