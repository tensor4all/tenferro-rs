use chainrules::{AdResult, AutodiffError, Differentiable, NodeId, ReverseRule, TrackedValue};
use tenferro_algebra::{HasAlgebra, Scalar, Semiring};
use tenferro_device::{Error, Result};
use tenferro_einsum::Subscripts;
use tenferro_prims::TensorSemiringCore;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::argmax::ArgmaxTracker;

use super::backward::tropical_backward;
use super::common::contracted_modes;
use super::rules::tracked_forward;
use super::TropicalScalar;

/// Reverse-mode rule for tropical einsum, for integration with [`chainrules::Tape`].
///
/// The pullback returns standard-real gradients even though the forward pass
/// uses tropical scalars.
///
/// # Examples
///
/// ```ignore
/// use chainrules::ReverseRule;
/// use tenferro_tropical::ad::TropicalEinsumReverseRule;
/// use tenferro_tropical::MaxPlus;
/// use tenferro_tensor::Tensor;
/// ```
pub struct TropicalEinsumReverseRule<T: TropicalScalar> {
    subscripts: Subscripts,
    primals: Vec<Tensor<T>>,
    tracker: ArgmaxTracker,
    input_node_ids: Vec<Option<NodeId>>,
    contracted: Vec<u32>,
}

impl<T> ReverseRule<Tensor<T::Inner>> for TropicalEinsumReverseRule<T>
where
    T: TropicalScalar,
    T::Inner: Scalar,
    Tensor<T::Inner>: Differentiable<Tangent = Tensor<T::Inner>>,
{
    fn pullback(&self, cotangent: &Tensor<T::Inner>) -> AdResult<Vec<(NodeId, Tensor<T::Inner>)>> {
        let primal_refs: Vec<&Tensor<T>> = self.primals.iter().collect();
        let grads = tropical_backward(
            &primal_refs,
            cotangent,
            &self.tracker,
            &self.subscripts,
            &self.contracted,
        )
        .map_err(|e| AutodiffError::InvalidArgument(format!("{e}")))?;

        let mut results = Vec::new();
        for (i, grad) in grads.into_iter().enumerate() {
            if let Some(id) = self.input_node_ids[i] {
                results.push((id, grad));
            }
        }
        Ok(results)
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.iter().filter_map(|id| *id).collect()
    }
}

/// Tracked tropical einsum for reverse-mode AD.
///
/// Runs the tropical forward pass, records winner indices, and returns a
/// tracked tensor containing the standard-real output values.
///
/// # Examples
///
/// ```ignore
/// use chainrules::Tape;
/// use tenferro_tropical::ad::tracked_tropical_einsum;
/// use tenferro_tropical::{MaxPlus, MaxPlusAlgebra};
/// use tenferro_prims::{CpuBackend, CpuContext};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let tape = Tape::<Tensor<f64>>::new();
/// let a_data = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let b_data = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let a = tape.leaf(a_data);
/// let b = tape.leaf(b_data);
///
/// let c = tracked_tropical_einsum::<MaxPlus<f64>, MaxPlusAlgebra<f64>, CpuBackend>(
///     "ij,jk->ik", &[&a, &b],
/// ).unwrap();
///
/// let grads = tape.pullback(&c).unwrap();
/// assert_eq!(grads.len(), 2);
/// ```
pub fn tracked_tropical_einsum<T, Alg, Backend>(
    subscripts: &str,
    operands: &[&TrackedValue<Tensor<T::Inner>>],
) -> AdResult<TrackedValue<Tensor<T::Inner>>>
where
    Alg: Semiring<Scalar = T>,
    T: TropicalScalar + HasAlgebra<Algebra = Alg> + 'static,
    T::Inner: Scalar + HasAlgebra,
    Backend: TensorSemiringCore<Alg>,
    Tensor<T::Inner>: Differentiable<Tangent = Tensor<T::Inner>>,
{
    if operands.is_empty() || operands.len() > 2 {
        return Err(AutodiffError::InvalidArgument(
            "tracked_tropical_einsum supports 1 or 2 operands".into(),
        ));
    }

    let subs = Subscripts::parse(subscripts)
        .map_err(|e| AutodiffError::InvalidArgument(format!("{e}")))?;
    let contracted = contracted_modes(&subs);
    let tropical_operands: Vec<Tensor<T>> = operands
        .iter()
        .map(|op| promote_to_tropical::<T>(op.value()))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| AutodiffError::InvalidArgument(format!("{e}")))?;
    let tropical_refs: Vec<&Tensor<T>> = tropical_operands.iter().collect();
    let (output_tropical, tracker) = tracked_forward(&tropical_refs, &subs, &contracted)
        .map_err(|e| AutodiffError::InvalidArgument(format!("{e}")))?;
    let output_inner = extract_inner::<T>(&output_tropical)
        .map_err(|e| AutodiffError::InvalidArgument(format!("{e}")))?;

    if !operands.iter().any(|op| op.requires_grad()) {
        return Ok(TrackedValue::new(output_inner));
    }

    let tape = operands
        .iter()
        .filter(|op| op.requires_grad())
        .find_map(|op| op.tape())
        .ok_or(AutodiffError::MissingNode)?
        .clone();

    for op in operands.iter().filter(|op| op.requires_grad()) {
        if let Some(op_tape) = op.tape() {
            if !tape.same_tape(op_tape) {
                return Err(AutodiffError::InvalidArgument(
                    "tracked_tropical_einsum: operands belong to different AD tapes".into(),
                ));
            }
        }
    }

    let rule = TropicalEinsumReverseRule::<T> {
        subscripts: subs,
        primals: tropical_operands,
        tracker,
        input_node_ids: operands.iter().map(|op| op.node_id()).collect(),
        contracted,
    };
    Ok(tape.record_op(output_inner, Box::new(rule), None))
}

/// Promote a standard-real tensor to a tropical scalar tensor.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_tropical::MaxPlus;
/// use tenferro_tropical::ad::promote_to_tropical;
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let tropical = promote_to_tropical::<MaxPlus<f64>>(&t).unwrap();
/// assert_eq!(tropical.dims(), &[2]);
/// ```
pub fn promote_to_tropical<T: TropicalScalar>(tensor: &Tensor<T::Inner>) -> Result<Tensor<T>> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let data = contiguous.buffer().as_slice().ok_or_else(|| {
        Error::DeviceError("tensor materialization produced a non-CPU buffer".into())
    })?;
    let tropical_data: Vec<T> = data.iter().map(|&v| T::from_inner(v)).collect();
    Tensor::<T>::from_slice(&tropical_data, tensor.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}

/// Extract the inner real values from a tropical tensor.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{MemoryOrder, Tensor};
/// use tenferro_tropical::MaxPlus;
/// use tenferro_tropical::ad::extract_inner;
///
/// let t = Tensor::<MaxPlus<f64>>::from_slice(
///     &[MaxPlus(1.0), MaxPlus(2.0)], &[2], MemoryOrder::ColumnMajor,
/// ).unwrap();
/// let inner = extract_inner::<MaxPlus<f64>>(&t).unwrap();
/// assert_eq!(inner.dims(), &[2]);
/// ```
pub fn extract_inner<T: TropicalScalar>(tensor: &Tensor<T>) -> Result<Tensor<T::Inner>> {
    tensor
        .buffer()
        .as_slice()
        .ok_or_else(|| Error::DeviceError("GPU tensor passed to CPU backend".into()))?;
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let data = contiguous.buffer().as_slice().ok_or_else(|| {
        Error::DeviceError("tensor materialization produced a non-CPU buffer".into())
    })?;
    let inner_data: Vec<T::Inner> = data.iter().map(|value| value.inner()).collect();
    Tensor::<T::Inner>::from_slice(&inner_data, tensor.dims(), MemoryOrder::ColumnMajor)
        .map_err(|e| Error::InvalidArgument(format!("{e}")))
}
