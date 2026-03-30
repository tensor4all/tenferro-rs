use tenferro_internal_ad_core::{DynAdTensor, DynAdTensorRef};
use tenferro_tensor::MemoryOrder;
use tidu::expert::Tape;

use super::super::ScalarType;
use super::layout::with_axis_classes_ad_tensor_typed;
use super::Tensor;
use crate::structured::StructuredTensor;
use crate::{backward, BackwardOptions, DynTensor, Error, Result};

#[cfg(test)]
fn reverse_tape_from_anchor(anchor: &Tensor, op_name: &'static str) -> Result<Tape<DynTensor>> {
    let tape = anchor.as_dyn_ad_ref().reverse_tape();
    tape.ok_or_else(|| Error::InvalidAdTensor {
        message: format!("{op_name} requires a reverse-mode Tensor anchor"),
    })
}

pub(crate) fn ensure_common_reverse_tape_impl(
    operands: &[&Tensor],
) -> Result<Option<Tape<DynTensor>>> {
    let mut tape: Option<Tape<DynTensor>> = None;

    for operand in operands {
        let current = operand.as_dyn_ad_ref().reverse_tape();
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
            operand.as_dyn_ad_ref().ensure_reverse_leaf_on(&tape)?;
        }
        return Ok(Some(tape));
    }

    Ok(tape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_from_adtensor_roundtrips_through_dyn_carrier() {
        let dense =
            tenferro_tensor::Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor)
                .unwrap();
        let tensor = Tensor::new_primal(StructuredTensor(
            tenferro_tensor::StructuredTensor::from_dense(dense),
        ));

        assert_eq!(tensor.scalar_type(), ScalarType::F64);
        assert_eq!(tensor.dims(), &[2]);
        assert_eq!(
            tensor
                .as_f64()
                .unwrap()
                .primal()
                .buffer()
                .as_slice()
                .unwrap(),
            &[1.0, 2.0]
        );
    }
}

impl Tensor {
    /// Creates a primal tensor from a typed dense tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
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
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        Self::new_primal(StructuredTensor(
            tenferro_tensor::StructuredTensor::from_dense(tensor),
        ))
    }

    pub(crate) fn from_structured<T>(tensor: StructuredTensor<T>) -> Self
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        Self::new_primal(tensor)
    }

    /// Creates a primal tensor from a typed slice and logical dimensions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn from_slice<T>(data: &[T], dims: &[usize]) -> Result<Self>
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + Clone
            + 'static,
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
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let payload = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    /// let x = Tensor::with_axis_classes(payload, &[0, 0, 1, 1]).unwrap();
    /// assert_eq!(x.dims(), &[2, 2, 2, 2]);
    /// assert_eq!(x.axis_classes(), &[0, 0, 1, 1]);
    /// assert!(!x.is_dense());
    /// ```
    pub fn with_axis_classes(payload: Self, axis_classes: &[usize]) -> Result<Self> {
        match payload.as_dyn_ad_ref() {
            DynAdTensorRef::F32(value) => Ok(Self::from(with_axis_classes_ad_tensor_typed(
                value,
                axis_classes,
            )?)),
            DynAdTensorRef::F64(value) => Ok(Self::from(with_axis_classes_ad_tensor_typed(
                value,
                axis_classes,
            )?)),
            DynAdTensorRef::C32(value) => Ok(Self::from(with_axis_classes_ad_tensor_typed(
                value,
                axis_classes,
            )?)),
            DynAdTensorRef::C64(value) => Ok(Self::from(with_axis_classes_ad_tensor_typed(
                value,
                axis_classes,
            )?)),
        }
    }

    pub(crate) fn new_primal<T>(tensor: impl Into<StructuredTensor<T>>) -> Self
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        Self::from(DynAdTensor::new_primal(tensor))
    }

    pub(crate) fn new_forward<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        Ok(Self::from(DynAdTensor::new_forward(primal, tangent)?))
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_leaf<T>(primal: impl Into<StructuredTensor<T>>) -> Result<Self>
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        let tape = Tape::new();
        Ok(Self::from(DynAdTensor::new_reverse_leaf(primal, &tape)?))
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_leaf_on<T>(
        primal: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        Ok(Self::from(DynAdTensor::new_reverse_leaf(primal, tape)?))
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_leaf_with_tangent<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        let tape = Tape::new();
        Ok(Self::from(DynAdTensor::new_reverse_leaf_with_tangent(
            primal, tangent, &tape,
        )?))
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_leaf_with_tangent_on<T>(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
        tape: &Tape<DynTensor>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        Ok(Self::from(DynAdTensor::new_reverse_leaf_with_tangent(
            primal, tangent, tape,
        )?))
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_sibling<T>(
        &self,
        primal: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        let tape = reverse_tape_from_anchor(self, "Tensor::new_reverse_sibling")?;
        Ok(Self::from(DynAdTensor::new_reverse_leaf(primal, &tape)?))
    }

    #[cfg(test)]
    pub(crate) fn new_reverse_sibling_with_tangent<T>(
        &self,
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Result<Self>
    where
        T: tenferro_algebra::Scalar
            + super::super::DynTensorTyped
            + tenferro_internal_ad_core::DynAdTensorTyped
            + 'static,
    {
        let tape = reverse_tape_from_anchor(self, "Tensor::new_reverse_sibling_with_tangent")?;
        Ok(Self::from(DynAdTensor::new_reverse_leaf_with_tangent(
            primal, tangent, &tape,
        )?))
    }

    /// Returns runtime scalar type.
    pub fn scalar_type(&self) -> ScalarType {
        self.0.scalar_type()
    }

    /// Returns whether this tensor participates in reverse-mode AD.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// assert!(!x.requires_grad());
    /// x.set_requires_grad(true).unwrap();
    /// assert!(x.requires_grad());
    /// ```
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad()
    }

    /// Returns whether this tensor is a leaf in the reverse-mode graph.
    ///
    /// Primal and forward-mode tensors are treated as leaf values.
    pub fn is_leaf(&self) -> bool {
        self.0.is_leaf()
    }

    /// Returns a tensor that requires reverse-mode gradients.
    ///
    /// If the tensor is already a grad-requiring leaf, this is idempotent.
    /// Otherwise the returned tensor is detached from any existing AD state and
    /// becomes a new reverse leaf.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap()
    ///     .with_requires_grad(true)
    ///     .unwrap();
    /// assert!(x.requires_grad());
    /// assert!(x.is_leaf());
    /// ```
    pub fn with_requires_grad(&self, enabled: bool) -> Result<Self> {
        self.0.with_requires_grad(enabled).map(Self::from)
    }

    /// Enables or disables reverse-mode gradient tracking for a leaf tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::Tensor;
    ///
    /// let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// x.set_requires_grad(true).unwrap();
    /// assert!(x.requires_grad());
    /// ```
    pub fn set_requires_grad(&mut self, enabled: bool) -> Result<()> {
        match self.as_dyn_ad_mut_ref() {
            tenferro_internal_ad_core::DynAdTensorMutRef::F32(v) => v.set_requires_grad(enabled),
            tenferro_internal_ad_core::DynAdTensorMutRef::F64(v) => v.set_requires_grad(enabled),
            tenferro_internal_ad_core::DynAdTensorMutRef::C32(v) => v.set_requires_grad(enabled),
            tenferro_internal_ad_core::DynAdTensorMutRef::C64(v) => v.set_requires_grad(enabled),
        }
    }

    /// Returns the accumulated gradient for a reverse leaf, if available.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{backward, set_default_runtime, BackwardOptions, RuntimeContext, Tensor};
    /// use tenferro_prims::CpuContext;
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// x.set_requires_grad(true).unwrap();
    /// let out = x.exp().unwrap().sum().unwrap();
    /// backward(&[&out], None, &[&x], BackwardOptions::default()).unwrap();
    /// assert!(x.grad().unwrap().is_some());
    /// ```
    pub fn grad(&self) -> Result<Option<Self>> {
        Ok(self.0.grad().map(Self::from))
    }

    /// Clears accumulated reverse-mode gradients on a leaf tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{backward, set_default_runtime, BackwardOptions, RuntimeContext, Tensor};
    /// use tenferro_prims::CpuContext;
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let mut x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// x.set_requires_grad(true).unwrap();
    /// let out = x.exp().unwrap().sum().unwrap();
    /// backward(&[&out], None, &[&x], BackwardOptions::default()).unwrap();
    /// x.zero_grad().unwrap();
    /// assert!(x.grad().unwrap().is_none());
    /// ```
    pub fn zero_grad(&self) -> Result<()> {
        self.0.zero_grad()
    }

    /// Runs reverse-mode accumulation from this output tensor into `inputs`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{set_default_runtime, BackwardOptions, RuntimeContext, Tensor};
    /// use tenferro_prims::CpuContext;
    ///
    /// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    /// let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    /// x.set_requires_grad(true).unwrap();
    /// let out = x.exp().unwrap().sum().unwrap();
    /// out.backward(None, &[&x], BackwardOptions::default()).unwrap();
    /// assert!(x.grad().unwrap().is_some());
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
        self.0.accumulate_input_grad_from(&grad.0)
    }

    #[cfg(test)]
    pub(crate) fn node_id(&self) -> Option<crate::core::NodeId> {
        self.0.node_id()
    }
}
