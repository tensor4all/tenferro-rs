use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, TensorBackend};

use crate::eager::EagerTensor;
use crate::error::{Error, Result};

impl<B: TensorBackend> EagerTensor<B> {
    /// Singular value decomposition: `A = U diag(S) Vt`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]));
    /// let (u, s, vt) = a.svd().unwrap();
    ///
    /// assert_eq!(u.data().shape(), &[2, 2]);
    /// assert_eq!(s.data().shape(), &[2]);
    /// assert_eq!(vt.data().shape(), &[2, 2]);
    /// ```
    pub fn svd(&self) -> Result<(Self, Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Svd {
                    eps: 1.0e-12,
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                3,
            )?
            .into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(u), Some(s), Some(vt), None) => Ok((u, s, vt)),
            _ => Err(Error::Internal(
                "svd eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// QR decomposition: `A = Q R`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]));
    /// let (q, r) = a.qr().unwrap();
    ///
    /// assert_eq!(q.data().shape(), &[2, 2]);
    /// assert_eq!(r.data().shape(), &[2, 2]);
    /// ```
    pub fn qr(&self) -> Result<(Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Qr {
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                2,
            )?
            .into_iter();
        match (outputs.next(), outputs.next(), outputs.next()) {
            (Some(q), Some(r), None) => Ok((q, r)),
            _ => Err(Error::Internal(
                "qr eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// LU decomposition with partial pivoting: `P A = L U`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]));
    /// let (p, l, u, parity) = a.lu().unwrap();
    ///
    /// assert_eq!(p.data().shape(), &[2, 2]);
    /// assert_eq!(l.data().shape(), &[2, 2]);
    /// assert_eq!(u.data().shape(), &[2, 2]);
    /// assert_eq!(parity.data().shape(), &[] as &[usize]);
    /// ```
    pub fn lu(&self) -> Result<(Self, Self, Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Lu {
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                4,
            )?
            .into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(p), Some(l), Some(u), Some(parity), None) => Ok((p, l, u, parity)),
            _ => Err(Error::Internal(
                "lu eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// Cholesky factorization: `A = L L^T` for real inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]));
    /// let l = a.cholesky().unwrap();
    ///
    /// assert_eq!(l.data().shape(), &[2, 2]);
    /// assert_eq!(l.data().as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn cholesky(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Cholesky {
            input_shape: DimExpr::from_concrete(self.data.shape()),
        })
    }

    /// Symmetric or Hermitian eigendecomposition: `A = V diag(W) V^T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]));
    /// let (values, vectors) = a.eigh().unwrap();
    ///
    /// assert_eq!(values.data().shape(), &[2]);
    /// assert_eq!(vectors.data().shape(), &[2, 2]);
    /// ```
    pub fn eigh(&self) -> Result<(Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Eigh {
                    eps: 1.0e-12,
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                2,
            )?
            .into_iter();
        match (outputs.next(), outputs.next(), outputs.next()) {
            (Some(values), Some(vectors), None) => Ok((values, vectors)),
            _ => Err(Error::Internal(
                "eigh eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// General eigendecomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]));
    /// let (values, vectors) = a.eig().unwrap();
    ///
    /// assert_eq!(values.data().shape(), &[2]);
    /// assert_eq!(vectors.data().shape(), &[2, 2]);
    /// ```
    pub fn eig(&self) -> Result<(Self, Self)> {
        let input_dtype: DType = self.data.dtype();
        let mut outputs = self
            .multi_output_unary_op(
                StdTensorOp::Eig {
                    input_dtype,
                    input_shape: DimExpr::from_concrete(self.data.shape()),
                },
                2,
            )?
            .into_iter();
        match (outputs.next(), outputs.next(), outputs.next()) {
            (Some(values), Some(vectors), None) => Ok((values, vectors)),
            _ => Err(Error::Internal(
                "eig eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// Solve a triangular linear system.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{EagerTensor, Tensor};
    ///
    /// let a = EagerTensor::from_tensor(Tensor::new(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]));
    /// let b = EagerTensor::from_tensor(Tensor::new(vec![2, 1], vec![4.0_f64, 8.0]));
    /// let x = a
    ///     .triangular_solve(&b, true, true, false, false)
    ///     .unwrap();
    ///
    /// assert_eq!(x.data().shape(), &[2, 1]);
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[2.0, 2.0]);
    /// ```
    pub fn triangular_solve(
        &self,
        b: &Self,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Self> {
        self.binary_op(
            b,
            StdTensorOp::TriangularSolve {
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
                lhs_shape: DimExpr::from_concrete(self.data.shape()),
                rhs_shape: DimExpr::from_concrete(b.data.shape()),
            },
        )
    }
}
