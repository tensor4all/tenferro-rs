use chainrules_core::AutodiffError;
use std::fmt;
use tenferro_internal_ad_linalg::{
    cholesky_dyn_value, det_dyn_value, eig_dyn_value, eigen_dyn_value, inv_dyn_value,
    lstsq_dyn_values, lu_dyn_value, matrix_exp_dyn_value, norm_dyn_value, pinv_dyn_value,
    qr_dyn_value, slogdet_dyn_value, solve_dyn_values, solve_triangular_dyn_value, svd_dyn_value,
};
use tenferro_internal_ad_ops::{add_dyn_values, einsum_dyn_values, exp_dyn_value, sum_dyn_value};
use tenferro_internal_frontend_core::tensor_ops::tensor_element;
use tenferro_internal_frontend_core::{DynTensor, DynTensorTyped, ScalarType, StructuredTensor};
use tenferro_linalg::{LuPivot, MatrixNormOrd, NormKind, SvdOptions, VectorNormOrd};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use super::{EigResult, EighResult, LstsqResult, LuResult, QrResult, SlogdetResult, SvdResult};
use crate::{jvp, Error, Result};

pub struct Tensor {
    inner: tidu::Value<DynTensor>,
}

impl Tensor {
    pub fn new(primal: DynTensor) -> Self {
        Self {
            inner: tidu::Value::new(primal),
        }
    }

    pub(crate) fn from_value(inner: tidu::Value<DynTensor>) -> Self {
        Self { inner }
    }

    pub(crate) fn value(&self) -> &tidu::Value<DynTensor> {
        &self.inner
    }

    pub(crate) fn primal(&self) -> &DynTensor {
        self.inner.primal()
    }

    pub(crate) fn forward_id(&self) -> usize {
        jvp::forward_id(self)
    }

    pub fn from_slice<T>(data: &[T], dims: &[usize]) -> Result<Self>
    where
        T: DynTensorTyped + Copy,
    {
        let payload = DenseTensor::<T>::from_slice(data, dims, MemoryOrder::ColumnMajor)?;
        Ok(Self::from(payload))
    }

    pub fn scalar_type(&self) -> ScalarType {
        self.primal().scalar_type()
    }

    pub fn dims(&self) -> &[usize] {
        self.primal().dims()
    }

    pub fn ndim(&self) -> usize {
        self.primal().ndim()
    }

    pub fn len(&self) -> usize {
        self.primal().len()
    }

    pub fn is_empty(&self) -> bool {
        self.primal().is_empty()
    }

    pub fn axis_classes(&self) -> &[usize] {
        self.primal().axis_classes()
    }

    pub fn is_dense(&self) -> bool {
        self.primal().is_dense()
    }

    pub fn is_diag(&self) -> bool {
        self.primal().is_diag()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    pub fn with_requires_grad(self, enabled: bool) -> Self {
        Self {
            inner: self.inner.with_requires_grad(enabled),
        }
    }

    pub fn detach(&self) -> Self {
        Self::new(self.primal().clone())
    }

    pub fn to_dense(&self) -> Result<Self> {
        Ok(Self::new(self.primal().to_dense()?))
    }

    pub fn grad(&self) -> Result<Option<Self>> {
        Ok(self.inner.grad()?.map(Self::new))
    }

    pub fn zero_grad(&self) -> Result<()> {
        Ok(self.inner.zero_grad()?)
    }

    pub fn backward(&self) -> Result<()> {
        Ok(self.inner.backward()?)
    }

    pub fn backward_with_seed(&self, seed: &Self) -> Result<()> {
        Ok(self.inner.backward_with_seed(seed.primal().clone())?)
    }

    pub fn shares_reverse_graph(&self, other: &Self) -> bool {
        self.inner.shares_reverse_graph(&other.inner)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        let output = Self::from_value(add_dyn_values(self.value(), rhs.value())?);
        jvp::add_tangent(self, rhs, &output)?;
        Ok(output)
    }

    pub fn exp(&self) -> Result<Self> {
        let output = Self::from_value(exp_dyn_value(self.value())?);
        jvp::exp_tangent(self, &output)?;
        Ok(output)
    }

    pub fn sum(&self) -> Result<Self> {
        let output = Self::from_value(sum_dyn_value(self.value())?);
        jvp::sum_tangent(self, &output)?;
        Ok(output)
    }

    pub fn einsum(subscripts: &str, operands: &[&Self]) -> Result<Self> {
        let values = operands
            .iter()
            .map(|tensor| tensor.value())
            .collect::<Vec<_>>();
        let output = Self::from_value(einsum_dyn_values(subscripts, &values)?);
        jvp::einsum_tangent(subscripts, operands, &output)?;
        Ok(output)
    }

    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        let output = Self::from_value(solve_dyn_values(self.value(), rhs.value())?);
        jvp::solve_tangent(self, rhs, &output)?;
        Ok(output)
    }

    pub fn lstsq(&self, rhs: &Self) -> Result<LstsqResult> {
        let result = lstsq_dyn_values(self.value(), rhs.value())?;
        let solution = Self::from_value(result.solution);
        let residuals = Self::from_value(result.residuals);
        jvp::lstsq_tangents(self, rhs, &solution, &residuals)?;
        Ok(LstsqResult {
            solution,
            residuals,
            rank: result.rank,
            singular_values: Tensor::from(result.singular_values),
        })
    }

    pub fn solve_triangular(&self, rhs: &Self, upper: bool) -> Result<Self> {
        let output = Self::from_value(solve_triangular_dyn_value(
            self.value(),
            rhs.value(),
            upper,
        )?);
        jvp::solve_triangular_tangent(self, rhs, &output, upper)?;
        Ok(output)
    }

    pub fn det(&self) -> Result<Self> {
        let output = Self::from_value(det_dyn_value(self.value())?);
        jvp::det_tangent(self, &output)?;
        Ok(output)
    }

    pub fn inv(&self) -> Result<Self> {
        let output = Self::from_value(inv_dyn_value(self.value())?);
        jvp::inv_tangent(self, &output)?;
        Ok(output)
    }

    pub fn slogdet(&self) -> Result<SlogdetResult> {
        let result: SlogdetResult = slogdet_dyn_value(self.value())?.into();
        jvp::slogdet_tangents(self, &result.sign, &result.logabsdet)?;
        Ok(result)
    }

    pub fn cholesky(&self) -> Result<Self> {
        let output = Self::from_value(cholesky_dyn_value(self.value())?);
        jvp::cholesky_tangent(self, &output)?;
        Ok(output)
    }

    pub fn lu(&self, pivot: LuPivot) -> Result<LuResult> {
        let result: LuResult = lu_dyn_value(self.value(), pivot)?.into();
        jvp::lu_tangents(self, &result.p, &result.l, &result.u, pivot)?;
        Ok(result)
    }

    pub fn norm(&self, kind: NormKind) -> Result<Self> {
        let output = Self::from_value(norm_dyn_value(self.value(), kind)?);
        jvp::norm_tangent(self, &output, kind)?;
        Ok(output)
    }

    pub fn vector_norm(
        &self,
        ord: VectorNormOrd,
        dim: Option<&[isize]>,
        keepdim: bool,
    ) -> Result<Self> {
        validate_vector_norm_request(self.ndim(), dim, keepdim)?;
        let kind = jvp::map_vector_norm_ord(ord)?;
        let output = Self::from_value(norm_dyn_value(self.value(), kind)?);
        jvp::vector_norm_tangent(self, &output, ord)?;
        Ok(output)
    }

    pub fn matrix_norm(
        &self,
        ord: MatrixNormOrd,
        dim: Option<(isize, isize)>,
        keepdim: bool,
    ) -> Result<Self> {
        validate_matrix_norm_request(self.ndim(), dim, keepdim)?;
        let kind = jvp::map_matrix_norm_ord(ord)?;
        let output = Self::from_value(norm_dyn_value(self.value(), kind)?);
        jvp::matrix_norm_tangent(self, &output, ord)?;
        Ok(output)
    }

    pub fn qr(&self) -> Result<QrResult> {
        crate::with_default_runtime(|_| Ok(()))?;
        let result: QrResult = qr_dyn_value(self.value())?.into();
        jvp::qr_tangents(self, &result.q, &result.r)?;
        Ok(result)
    }

    pub fn svd(&self, options: Option<SvdOptions>) -> Result<SvdResult> {
        let result: SvdResult = svd_dyn_value(self.value(), options.clone())?.into();
        jvp::svd_tangents(self, &result.u, &result.s, &result.vt, options)?;
        Ok(result)
    }

    pub fn eig(&self) -> Result<EigResult> {
        let result: EigResult = eig_dyn_value(self.value())?.into();
        jvp::eig_tangents(self, &result.values, &result.vectors)?;
        Ok(result)
    }

    pub fn eigh(&self) -> Result<EighResult> {
        let result: EighResult = eigen_dyn_value(self.value())?.into();
        jvp::eigen_tangents(self, &result.values, &result.vectors)?;
        Ok(result)
    }

    pub fn pinv(&self, rcond: Option<f64>) -> Result<Self> {
        let output = Self::from_value(pinv_dyn_value(self.value(), rcond)?);
        jvp::pinv_tangent(self, &output, rcond)?;
        Ok(output)
    }

    pub fn matrix_exp(&self) -> Result<Self> {
        let output = Self::from_value(matrix_exp_dyn_value(self.value())?);
        jvp::matrix_exp_tangent(self, &output)?;
        Ok(output)
    }

    pub fn try_to_vec<T>(&self) -> Result<Vec<T>>
    where
        T: DynTensorTyped + Copy,
    {
        let structured = T::structured_ref(self.primal()).ok_or_else(|| {
            invalid_argument(format!(
                "dtype mismatch in try_to_vec: tensor={:?}",
                self.scalar_type()
            ))
        })?;
        let dense = structured.to_dense()?;
        let contiguous = dense.contiguous(MemoryOrder::ColumnMajor);
        let slice = contiguous
            .buffer()
            .as_slice()
            .ok_or_else(|| invalid_argument("try_to_vec requires host-accessible dense payload"))?;
        Ok(slice.to_vec())
    }

    pub fn try_get<T>(&self, index: &[usize]) -> Result<T>
    where
        T: DynTensorTyped + Copy,
    {
        let structured = T::structured_ref(self.primal()).ok_or_else(|| {
            invalid_argument(format!(
                "dtype mismatch in try_get: tensor={:?}",
                self.scalar_type()
            ))
        })?;
        tensor_element(structured.payload(), index)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("scalar_type", &self.scalar_type())
            .field("dims", &self.dims())
            .field("requires_grad", &self.requires_grad())
            .finish()
    }
}

impl<T> From<DenseTensor<T>> for Tensor
where
    T: DynTensorTyped + Copy,
{
    fn from(value: DenseTensor<T>) -> Self {
        Self::from(StructuredTensor::from(value))
    }
}

impl<T> From<StructuredTensor<T>> for Tensor
where
    T: DynTensorTyped + Copy,
{
    fn from(value: StructuredTensor<T>) -> Self {
        Self::new(T::into_dyn(value))
    }
}

impl From<DynTensor> for Tensor {
    fn from(value: DynTensor) -> Self {
        Self::new(value)
    }
}

fn invalid_argument(message: impl Into<String>) -> Error {
    AutodiffError::InvalidArgument(message.into()).into()
}

fn validate_vector_norm_request(ndim: usize, dim: Option<&[isize]>, keepdim: bool) -> Result<()> {
    if keepdim {
        return Err(invalid_argument(
            "vector_norm currently supports keepdim=false only",
        ));
    }
    if ndim != 1 {
        return Err(invalid_argument(format!(
            "vector_norm currently expects a rank-1 tensor, got ndim={ndim}",
        )));
    }
    if dim.is_some() {
        return Err(invalid_argument(
            "vector_norm currently supports dim=None only",
        ));
    }
    Ok(())
}

fn validate_matrix_norm_request(
    ndim: usize,
    dim: Option<(isize, isize)>,
    keepdim: bool,
) -> Result<()> {
    if keepdim {
        return Err(invalid_argument(
            "matrix_norm currently supports keepdim=false only",
        ));
    }
    if ndim != 2 {
        return Err(invalid_argument(format!(
            "matrix_norm currently expects a rank-2 tensor, got ndim={ndim}",
        )));
    }
    if let Some(dim) = dim {
        if dim != (0, 1) && dim != (1, 0) {
            return Err(invalid_argument(format!(
                "matrix_norm currently supports dim=(0, 1) only, got {dim:?}",
            )));
        }
    }
    Ok(())
}
