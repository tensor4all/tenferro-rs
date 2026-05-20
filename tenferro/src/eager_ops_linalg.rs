use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::DType;

use crate::eager::EagerTensor;
use crate::error::{Error, Result};

impl EagerTensor {
    /// Singular value decomposition: `A = U diag(S) Vh`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]), ctx.clone());
    /// let (u, s, vh) = a.svd().unwrap();
    ///
    /// assert_eq!(u.data().shape(), &[2, 2]);
    /// assert_eq!(s.data().shape(), &[2]);
    /// assert_eq!(vh.data().shape(), &[2, 2]);
    /// ```
    pub fn svd(&self) -> Result<(Self, Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(StdTensorOp::Svd { eps: 0.0 }, 3)?
            .into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(u), Some(s), Some(vh), None) => Ok((u, s, vh)),
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
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]), ctx.clone());
    /// let (q, r) = a.qr().unwrap();
    ///
    /// assert_eq!(q.data().shape(), &[2, 2]);
    /// assert_eq!(r.data().shape(), &[2, 2]);
    /// ```
    pub fn qr(&self) -> Result<(Self, Self)> {
        let mut outputs = self.multi_output_unary_op(StdTensorOp::Qr, 2)?.into_iter();
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
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]), ctx.clone());
    /// let (p, l, u, parity) = a.lu().unwrap();
    ///
    /// assert_eq!(p.data().shape(), &[2, 2]);
    /// assert_eq!(l.data().shape(), &[2, 2]);
    /// assert_eq!(u.data().shape(), &[2, 2]);
    /// assert_eq!(parity.data().shape(), &[] as &[usize]);
    /// ```
    pub fn lu(&self) -> Result<(Self, Self, Self, Self)> {
        let mut outputs = self.multi_output_unary_op(StdTensorOp::Lu, 4)?.into_iter();
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

    /// LU decomposition with complete pivoting: `P A Q^T = L U`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]), ctx.clone());
    /// let (p, l, u, q, parity) = a.full_piv_lu().unwrap();
    ///
    /// assert_eq!(p.data().shape(), &[2, 2]);
    /// assert_eq!(l.data().shape(), &[2, 2]);
    /// assert_eq!(u.data().shape(), &[2, 2]);
    /// assert_eq!(q.data().shape(), &[2, 2]);
    /// assert_eq!(parity.data().shape(), &[] as &[usize]);
    /// ```
    pub fn full_piv_lu(&self) -> Result<(Self, Self, Self, Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(StdTensorOp::FullPivLu, 5)?
            .into_iter();
        match (
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
            outputs.next(),
        ) {
            (Some(p), Some(l), Some(u), Some(q), Some(parity), None) => Ok((p, l, u, q, parity)),
            _ => Err(Error::Internal(
                "full_piv_lu eager op returned an unexpected number of outputs".to_string(),
            )),
        }
    }

    /// Solve `A x = b` using complete-pivoting LU factorization.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]), ctx.clone());
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 1], vec![-1.0_f64, 5.0]), ctx.clone());
    /// let x = a.full_piv_lu_solve(&b).unwrap();
    ///
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[4.0, -1.0]);
    /// ```
    pub fn full_piv_lu_solve(&self, b: &Self) -> Result<Self> {
        self.binary_op(b, StdTensorOp::FullPivLuSolve { transpose_a: false })
    }

    /// Solve `A x = rhs` using complete-pivoting LU factorization.
    ///
    /// This is a shorter alias for [`Self::full_piv_lu_solve`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]), ctx.clone());
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 1], vec![4.0_f64, 8.0]), ctx);
    /// let x = a.solve(&b).unwrap();
    ///
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[2.0, 2.0]);
    /// ```
    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        self.full_piv_lu_solve(rhs)
    }

    /// Solve `x A = rhs`, returning `rhs * A^{-1}` without forming an inverse.
    ///
    /// This uses the identity `rhs * A^{-1} = (A^T \ rhs^T)^T`, so AD flows
    /// through the existing linear solve rule.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]), ctx.clone());
    /// let rhs = EagerTensor::from_tensor_in(Tensor::from_vec(vec![1, 2], vec![4.0_f64, 8.0]), ctx);
    /// let x = a.right_solve(&rhs).unwrap();
    ///
    /// assert_eq!(x.data().shape(), &[1, 2]);
    /// assert_eq!(x.data().as_slice::<f64>().unwrap(), &[2.0, 2.0]);
    /// ```
    pub fn right_solve(&self, rhs: &Self) -> Result<Self> {
        let lhs_shape = self.data().shape();
        let rhs_shape = rhs.data().shape();
        if lhs_shape.len() != 2 {
            return Err(tenferro_tensor::Error::RankMismatch {
                op: "right_solve",
                expected: 2,
                actual: lhs_shape.len(),
            }
            .into());
        }
        if rhs_shape.len() != 2 {
            return Err(tenferro_tensor::Error::RankMismatch {
                op: "right_solve",
                expected: 2,
                actual: rhs_shape.len(),
            }
            .into());
        }
        if lhs_shape[1] != rhs_shape[1] {
            return Err(tenferro_tensor::Error::ShapeMismatch {
                op: "right_solve",
                lhs: lhs_shape.to_vec(),
                rhs: rhs_shape.to_vec(),
            }
            .into());
        }
        let a_t = self.transpose(&[1, 0])?;
        let rhs_t = rhs.transpose(&[1, 0])?;
        a_t.full_piv_lu_solve(&rhs_t)?.transpose(&[1, 0])
    }

    /// Cholesky factorization: `A = L L^T` for real inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]), ctx.clone());
    /// let l = a.cholesky().unwrap();
    ///
    /// assert_eq!(l.data().shape(), &[2, 2]);
    /// assert_eq!(l.data().as_slice::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn cholesky(&self) -> Result<Self> {
        self.unary_op(StdTensorOp::Cholesky)
    }

    /// Symmetric or Hermitian eigendecomposition: `A = V diag(W) V^T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]), ctx.clone());
    /// let (values, vectors) = a.eigh().unwrap();
    ///
    /// assert_eq!(values.data().shape(), &[2]);
    /// assert_eq!(vectors.data().shape(), &[2, 2]);
    /// ```
    pub fn eigh(&self) -> Result<(Self, Self)> {
        let mut outputs = self
            .multi_output_unary_op(StdTensorOp::Eigh { eps: 0.0 }, 2)?
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
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]), ctx.clone());
    /// let (values, vectors) = a.eig().unwrap();
    ///
    /// assert_eq!(values.data().shape(), &[2]);
    /// assert_eq!(vectors.data().shape(), &[2, 2]);
    /// ```
    pub fn eig(&self) -> Result<(Self, Self)> {
        let input_dtype: DType = self.data.dtype();
        let mut outputs = self
            .multi_output_unary_op(StdTensorOp::Eig { input_dtype }, 2)?
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
    /// use tenferro::{CpuBackend, EagerContext, EagerTensor, Tensor};
    ///
    /// let ctx = EagerContext::with_cpu_backend(CpuBackend::new());
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]), ctx.clone());
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec(vec![2, 1], vec![4.0_f64, 8.0]), ctx.clone());
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
            },
        )
    }
}
