use std::marker::PhantomData;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::{DynAdTensor, DynAdTensorRef};
use tenferro_internal_frontend_core::{DynTensor, ScalarType, StructuredTensor};
use tenferro_tensor::Tensor as DenseTensor;
use tidu::expert::{NodeId, Tape, TrackedValue};

use super::Tensor;

/// Read-only typed view returned by [`Tensor::as_f32`], [`Tensor::as_f64`],
/// [`Tensor::as_c32`], and [`Tensor::as_c64`].
#[derive(Clone, Copy)]
pub struct TypedTensorRef<'a, T: Scalar> {
    inner: &'a Tensor,
    marker: PhantomData<&'a T>,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}

#[doc(hidden)]
pub trait TypedTensorScalarTag: sealed::Sealed + Scalar + 'static {
    fn scalar_type_tag() -> ScalarType;
}

#[doc(hidden)]
pub trait TypedTensorBorrowTyped: TypedTensorScalarTag {
    fn primal_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>>;
    fn structured_primal_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>>;
    fn tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>>;
    fn structured_tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>>;
    fn reverse_edge_value_from_dyn_ad(
        value: &DynAdTensor,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<Self>>>>;
}

impl TypedTensorScalarTag for f32 {
    fn scalar_type_tag() -> ScalarType {
        ScalarType::F32
    }
}

impl TypedTensorScalarTag for f64 {
    fn scalar_type_tag() -> ScalarType {
        ScalarType::F64
    }
}

impl TypedTensorScalarTag for Complex32 {
    fn scalar_type_tag() -> ScalarType {
        ScalarType::C32
    }
}

impl TypedTensorScalarTag for Complex64 {
    fn scalar_type_tag() -> ScalarType {
        ScalarType::C64
    }
}

impl TypedTensorBorrowTyped for f32 {
    fn primal_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>> {
        value.primal_as::<Self>()
    }

    fn structured_primal_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>> {
        value.structured_primal_as::<Self>()
    }

    fn tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>> {
        value.tangent_as::<Self>()
    }

    fn structured_tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>> {
        value.structured_tangent_as::<Self>()
    }

    fn reverse_edge_value_from_dyn_ad(
        value: &DynAdTensor,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<Self>>>> {
        value.reverse_edge_value_as::<Self>()
    }
}

impl TypedTensorBorrowTyped for f64 {
    fn primal_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>> {
        value.primal_as::<Self>()
    }

    fn structured_primal_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>> {
        value.structured_primal_as::<Self>()
    }

    fn tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>> {
        value.tangent_as::<Self>()
    }

    fn structured_tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>> {
        value.structured_tangent_as::<Self>()
    }

    fn reverse_edge_value_from_dyn_ad(
        value: &DynAdTensor,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<Self>>>> {
        value.reverse_edge_value_as::<Self>()
    }
}

impl TypedTensorBorrowTyped for Complex32 {
    fn primal_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>> {
        value.primal_as::<Self>()
    }

    fn structured_primal_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>> {
        value.structured_primal_as::<Self>()
    }

    fn tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>> {
        value.tangent_as::<Self>()
    }

    fn structured_tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>> {
        value.structured_tangent_as::<Self>()
    }

    fn reverse_edge_value_from_dyn_ad(
        value: &DynAdTensor,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<Self>>>> {
        value.reverse_edge_value_as::<Self>()
    }
}

impl TypedTensorBorrowTyped for Complex64 {
    fn primal_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>> {
        value.primal_as::<Self>()
    }

    fn structured_primal_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>> {
        value.structured_primal_as::<Self>()
    }

    fn tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&DenseTensor<Self>> {
        value.tangent_as::<Self>()
    }

    fn structured_tangent_from_dyn_ad(value: &DynAdTensor) -> Option<&StructuredTensor<Self>> {
        value.structured_tangent_as::<Self>()
    }

    fn reverse_edge_value_from_dyn_ad(
        value: &DynAdTensor,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<Self>>>> {
        value.reverse_edge_value_as::<Self>()
    }
}

impl<'a, T: TypedTensorBorrowTyped> TypedTensorRef<'a, T> {
    pub(crate) fn new(inner: &'a Tensor) -> Option<Self> {
        T::structured_primal_from_dyn_ad(&inner.0).map(|_| Self {
            inner,
            marker: PhantomData,
        })
    }

    pub fn scalar_type(&self) -> ScalarType {
        T::scalar_type_tag()
    }

    pub fn mode(&self) -> crate::core::AdMode {
        self.inner.mode()
    }

    pub fn primal(&self) -> &'a DenseTensor<T> {
        T::primal_from_dyn_ad(&self.inner.0)
            .expect("TypedTensorRef must be constructed from a matching runtime dtype")
    }

    pub fn structured_primal(&self) -> &'a StructuredTensor<T> {
        T::structured_primal_from_dyn_ad(&self.inner.0)
            .expect("TypedTensorRef must be constructed from a matching runtime dtype")
    }

    pub fn tangent(&self) -> Option<&'a DenseTensor<T>> {
        T::tangent_from_dyn_ad(&self.inner.0)
    }

    pub fn structured_tangent(&self) -> Option<&'a StructuredTensor<T>> {
        T::structured_tangent_from_dyn_ad(&self.inner.0)
    }

    pub fn dims(&self) -> &'a [usize] {
        self.inner.0.dims()
    }

    pub fn axis_classes(&self) -> &'a [usize] {
        self.inner.0.axis_classes()
    }

    pub fn is_dense(&self) -> bool {
        self.inner.0.is_dense()
    }

    pub fn is_diag(&self) -> bool {
        self.inner.0.is_diag()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.0.requires_grad()
    }

    pub fn is_leaf(&self) -> bool {
        self.inner.0.is_leaf()
    }

    pub fn node_id(&self) -> Option<NodeId> {
        self.inner.0.node_id()
    }

    pub fn tape(&self) -> Option<Tape<DynTensor>> {
        self.inner.0.tape()
    }

    pub(crate) fn reverse_edge_value(
        &self,
    ) -> Option<std::sync::Arc<tidu::Value<StructuredTensor<T>>>> {
        T::reverse_edge_value_from_dyn_ad(&self.inner.0)
    }

    pub(crate) fn as_tracked(&self) -> Option<TrackedValue<DynTensor>> {
        self.inner.0.as_tracked()
    }
}

impl Tensor {
    /// Returns the current AD mode of the tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::{AdMode, Tensor};
    ///
    /// let x = Tensor::from_slice(&[1.0_f64], &[1]).unwrap();
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn mode(&self) -> crate::core::AdMode {
        self.as_dyn_ad_ref().mode()
    }

    /// Returns a typed tensor view when dtype is `f32`.
    pub fn as_f32(&self) -> Option<TypedTensorRef<'_, f32>> {
        TypedTensorRef::new(self)
    }

    /// Returns a typed tensor view when dtype is `f64`.
    pub fn as_f64(&self) -> Option<TypedTensorRef<'_, f64>> {
        TypedTensorRef::new(self)
    }

    /// Returns a typed tensor view when dtype is `Complex32`.
    pub fn as_c32(&self) -> Option<TypedTensorRef<'_, Complex32>> {
        TypedTensorRef::new(self)
    }

    /// Returns a typed tensor view when dtype is `Complex64`.
    pub fn as_c64(&self) -> Option<TypedTensorRef<'_, Complex64>> {
        TypedTensorRef::new(self)
    }

    pub(crate) fn reverse_tape(&self) -> Option<Tape<DynTensor>> {
        self.0.tape()
    }

    pub(crate) fn reverse_handle(&self) -> Option<(NodeId, Tape<DynTensor>)> {
        self.0.reverse_handle()
    }

    pub(crate) fn shares_reverse_graph(&self, other: &Self) -> bool {
        match (self.reverse_tape(), other.reverse_tape()) {
            (Some(lhs), Some(rhs)) => lhs.same_tape(&rhs),
            (None, None) => match (self.as_dyn_ad_ref(), other.as_dyn_ad_ref()) {
                (DynAdTensorRef::F32(lhs), DynAdTensorRef::F32(rhs)) => {
                    match (lhs.reverse_edge_value(), rhs.reverse_edge_value()) {
                        (Some(lhs), Some(rhs)) => lhs.shares_reverse_graph(rhs.as_ref()),
                        (None, None) => true,
                        _ => false,
                    }
                }
                (DynAdTensorRef::F64(lhs), DynAdTensorRef::F64(rhs)) => {
                    match (lhs.reverse_edge_value(), rhs.reverse_edge_value()) {
                        (Some(lhs), Some(rhs)) => lhs.shares_reverse_graph(rhs.as_ref()),
                        (None, None) => true,
                        _ => false,
                    }
                }
                (DynAdTensorRef::C32(lhs), DynAdTensorRef::C32(rhs)) => {
                    match (lhs.reverse_edge_value(), rhs.reverse_edge_value()) {
                        (Some(lhs), Some(rhs)) => lhs.shares_reverse_graph(rhs.as_ref()),
                        (None, None) => true,
                        _ => false,
                    }
                }
                (DynAdTensorRef::C64(lhs), DynAdTensorRef::C64(rhs)) => {
                    match (lhs.reverse_edge_value(), rhs.reverse_edge_value()) {
                        (Some(lhs), Some(rhs)) => lhs.shares_reverse_graph(rhs.as_ref()),
                        (None, None) => true,
                        _ => false,
                    }
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub(crate) fn as_tracked(&self) -> Option<TrackedValue<DynTensor>> {
        self.0.as_tracked()
    }

    pub(crate) fn as_dyn_ad_ref(&self) -> DynAdTensorRef<'_> {
        (&self.0).into()
    }

    pub(crate) fn as_dyn_ad_mut_ref(&mut self) -> tenferro_internal_ad_core::DynAdTensorMutRef<'_> {
        (&mut self.0).into()
    }

    /// Returns true when scalar dtype is complex.
    pub fn is_complex(&self) -> bool {
        matches!(self.scalar_type(), ScalarType::C32 | ScalarType::C64)
    }

    /// Returns true when scalar dtype is real.
    pub fn is_real(&self) -> bool {
        matches!(self.scalar_type(), ScalarType::F32 | ScalarType::F64)
    }
}
