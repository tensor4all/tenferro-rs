use tenferro_tensor::MemoryOrder;
use tidu::Tape;

use super::super::ScalarType;
use super::layout::with_axis_classes_ad_tensor_typed;
use super::Tensor;
use crate::structured::StructuredTensor;
use crate::{backward, AdTensor, BackwardOptions, DynTensor, Error, Result};

#[cfg(test)]
fn reverse_tape_from_anchor(anchor: &Tensor, op_name: &'static str) -> Result<Tape<DynTensor>> {
    let tape = match anchor {
        Tensor::F32(value) => value.reverse_tape(),
        Tensor::F64(value) => value.reverse_tape(),
        Tensor::C32(value) => value.reverse_tape(),
        Tensor::C64(value) => value.reverse_tape(),
    };
    tape.ok_or_else(|| Error::InvalidAdTensor {
        message: format!("{op_name} requires a reverse-mode Tensor anchor"),
    })
}

pub(crate) fn ensure_common_reverse_tape(operands: &[&Tensor]) -> Result<Option<Tape<DynTensor>>> {
    let mut tape: Option<Tape<DynTensor>> = None;

    for operand in operands {
        let current = match operand {
            Tensor::F32(value) => value.reverse_tape(),
            Tensor::F64(value) => value.reverse_tape(),
            Tensor::C32(value) => value.reverse_tape(),
            Tensor::C64(value) => value.reverse_tape(),
        };
        if let Some(current) = current {
            if let Some(expected) = &tape {
                if !expected.same_tape(&current) {
                    return Err(Error::MixedReverseTape {
                        expected: expected.id() as u64,
                        found: current.id() as u64,
                    });
                }
            } else {
                tape = Some(current);
            }
        }
    }

    if operands.iter().any(|operand| operand.requires_grad()) {
        let tape = tape.unwrap_or_else(Tape::new);
        for operand in operands {
            match operand {
                Tensor::F32(value) => value.ensure_reverse_leaf_on(&tape)?,
                Tensor::F64(value) => value.ensure_reverse_leaf_on(&tape)?,
                Tensor::C32(value) => value.ensure_reverse_leaf_on(&tape)?,
                Tensor::C64(value) => value.ensure_reverse_leaf_on(&tape)?,
            }
        }
        return Ok(Some(tape));
    }

    Ok(tape)
}

impl Tensor {
    /// Creates a primal tensor from a typed dense tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::Tensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
    ///
    /// let x = Tensor::from_tensor(
    ///     DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    /// );
    /// assert!(x.is_dense());
    /// ```
    pub fn from_tensor<T>(tensor: tenferro_tensor::Tensor<T>) -> Self
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        Self::new_primal(StructuredTensor(
            tenferro_tensor::StructuredTensor::from_dense(tensor),
        ))
    }

    pub(crate) fn from_structured<T>(tensor: StructuredTensor<T>) -> Self
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        Self::new_primal(tensor)
    }

    /// Creates a primal tensor from a typed slice and logical dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn from_slice<T>(data: &[T], dims: &[usize]) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + Clone + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tensor = tenferro_tensor::Tensor::<T>::from_slice(data, dims, MemoryOrder::ColumnMajor)
            .map_err(Error::from)?;
        Ok(Self::from_tensor(tensor))
    }

    /// Reinterprets a dense payload tensor as a structured tensor with the
    /// provided axis equivalence classes.
    ///
    /// `axis_classes` must already be canonicalized to first-appearance order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::Tensor;
    ///
    /// let payload = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let x = Tensor::with_axis_classes(payload, &[0, 0, 1, 1]).unwrap();
    /// assert_eq!(x.dims(), &[2, 2, 2, 2]);
    /// assert_eq!(x.axis_classes(), &[0, 0, 1, 1]);
    /// assert!(!x.is_dense());
    /// ```
    pub fn with_axis_classes(payload: Self, axis_classes: &[usize]) -> Result<Self> {
        match payload {
            Self::F32(value) => Ok(Self::F32(with_axis_classes_ad_tensor_typed(
                &value,
                axis_classes,
            )?)),
            Self::F64(value) => Ok(Self::F64(with_axis_classes_ad_tensor_typed(
                &value,
                axis_classes,
            )?)),
            Self::C32(value) => Ok(Self::C32(with_axis_classes_ad_tensor_typed(
                &value,
                axis_classes,
            )?)),
            Self::C64(value) => Ok(Self::C64(with_axis_classes_ad_tensor_typed(
                &value,
                axis_classes,
            )?)),
        }
    }

    pub(crate) fn new_primal<T>(tensor: impl Into<StructuredTensor<T>>) -> Self
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        AdTensor::new_primal(tensor).into()
    }

    #[cfg(test)]
    pub(crate) fn new_forward<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        Ok(AdTensor::new_forward(primal, tangent)?.into())
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_leaf<T>(primal: impl Into<StructuredTensor<T>>) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tape = Tape::new();
        Ok(AdTensor::new_reverse_leaf(primal, &tape)?.into())
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_leaf_with_tangent<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tape = Tape::new();
        Ok(AdTensor::new_reverse_leaf_with_tangent(primal, tangent, &tape)?.into())
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_sibling<T>(
        &self,
        primal: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tape = reverse_tape_from_anchor(self, "Tensor::new_reverse_sibling")?;
        Ok(AdTensor::new_reverse_leaf(primal, &tape)?.into())
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_sibling_with_tangent<T>(
        &self,
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar + super::super::DynTensorTyped + 'static,
        AdTensor<T>: Into<Self>,
    {
        let tape = reverse_tape_from_anchor(self, "Tensor::new_reverse_sibling_with_tangent")?;
        Ok(AdTensor::new_reverse_leaf_with_tangent(primal, tangent, &tape)?.into())
    }

    /// Returns runtime scalar type.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::F32(_) => ScalarType::F32,
            Self::F64(_) => ScalarType::F64,
            Self::C32(_) => ScalarType::C32,
            Self::C64(_) => ScalarType::C64,
        }
    }

    /// Returns whether this tensor participates in reverse-mode AD.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::Tensor;
    ///
    /// let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// assert!(!x.requires_grad());
    /// x.set_requires_grad(true).unwrap();
    /// assert!(x.requires_grad());
    /// ```
    pub fn requires_grad(&self) -> bool {
        match self {
            Self::F32(v) => v.requires_grad(),
            Self::F64(v) => v.requires_grad(),
            Self::C32(v) => v.requires_grad(),
            Self::C64(v) => v.requires_grad(),
        }
    }

    /// Enables or disables reverse-mode gradient tracking for a leaf tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::Tensor;
    ///
    /// let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// x.set_requires_grad(true).unwrap();
    /// assert!(x.requires_grad());
    /// ```
    pub fn set_requires_grad(&mut self, enabled: bool) -> Result<()> {
        match self {
            Self::F32(v) => v.set_requires_grad(enabled),
            Self::F64(v) => v.set_requires_grad(enabled),
            Self::C32(v) => v.set_requires_grad(enabled),
            Self::C64(v) => v.set_requires_grad(enabled),
        }
    }

    /// Returns the accumulated gradient for a reverse leaf, if available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::{backward, set_default_runtime, BackwardOptions, RuntimeContext, Tensor};
    /// use tenferro_prims::CpuContext;
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// x.set_requires_grad(true).unwrap();
    /// let out = x.exp().unwrap().sum().unwrap();
    /// backward(&[&out], None, &[&x], BackwardOptions::default()).unwrap();
    /// assert!(x.grad().is_some());
    /// ```
    pub fn grad(&self) -> Option<Self> {
        match self {
            Self::F32(v) => v.grad().map(|grad| Self::F32(AdTensor::new_primal(grad))),
            Self::F64(v) => v.grad().map(|grad| Self::F64(AdTensor::new_primal(grad))),
            Self::C32(v) => v.grad().map(|grad| Self::C32(AdTensor::new_primal(grad))),
            Self::C64(v) => v.grad().map(|grad| Self::C64(AdTensor::new_primal(grad))),
        }
    }

    /// Clears accumulated reverse-mode gradients on a leaf tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::{backward, set_default_runtime, BackwardOptions, RuntimeContext, Tensor};
    /// use tenferro_prims::CpuContext;
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// x.set_requires_grad(true).unwrap();
    /// let out = x.exp().unwrap().sum().unwrap();
    /// backward(&[&out], None, &[&x], BackwardOptions::default()).unwrap();
    /// x.zero_grad().unwrap();
    /// assert!(x.grad().is_none());
    /// ```
    pub fn zero_grad(&self) -> Result<()> {
        match self {
            Self::F32(v) => v.zero_grad(),
            Self::F64(v) => v.zero_grad(),
            Self::C32(v) => v.zero_grad(),
            Self::C64(v) => v.zero_grad(),
        }
    }

    /// Runs reverse-mode accumulation from this output tensor into `inputs`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro::{set_default_runtime, BackwardOptions, RuntimeContext, Tensor};
    /// use tenferro_prims::CpuContext;
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// x.set_requires_grad(true).unwrap();
    /// let out = x.exp().unwrap().sum().unwrap();
    /// out.backward(None, &[&x], BackwardOptions::default()).unwrap();
    /// assert!(x.grad().is_some());
    /// ```
    pub fn backward(
        &self,
        grad_output: Option<&Self>,
        inputs: &[&Self],
        options: BackwardOptions,
    ) -> Result<()> {
        let grad_outputs = grad_output.map(std::slice::from_ref);
        backward(&[self], grad_outputs, inputs, options)
    }

    pub(crate) fn accumulate_grad(&self, grad: &Self) -> Result<()> {
        match (self, grad) {
            (Self::F32(value), Self::F32(grad)) => {
                value.accumulate_leaf_grad(grad.structured_primal().clone())
            }
            (Self::F64(value), Self::F64(grad)) => {
                value.accumulate_leaf_grad(grad.structured_primal().clone())
            }
            (Self::C32(value), Self::C32(grad)) => {
                value.accumulate_leaf_grad(grad.structured_primal().clone())
            }
            (Self::C64(value), Self::C64(grad)) => {
                value.accumulate_leaf_grad(grad.structured_primal().clone())
            }
            _ => Err(Error::InvalidAdTensor {
                message: format!(
                    "gradient dtype {:?} does not match input dtype {:?}",
                    grad.scalar_type(),
                    self.scalar_type()
                ),
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn node_id(&self) -> Option<crate::core::NodeId> {
        match self {
            Self::F32(v) => v.node_id(),
            Self::F64(v) => v.node_id(),
            Self::C32(v) => v.node_id(),
            Self::C64(v) => v.node_id(),
        }
    }

    #[cfg(test)]
    pub(crate) fn shares_reverse_graph(&self, other: &Self) -> bool {
        match (
            reverse_tape_from_anchor(self, "Tensor::shares_reverse_graph"),
            reverse_tape_from_anchor(other, "Tensor::shares_reverse_graph"),
        ) {
            (Ok(lhs), Ok(rhs)) => lhs.same_tape(&rhs),
            _ => false,
        }
    }
}
