use crate::{Error, Result, Tensor, TensorBackend, TensorScalar, TypedTensor};

fn try_into_typed_result<T: TensorScalar>(
    op: &'static str,
    tensor: Tensor,
) -> Result<TypedTensor<T>> {
    let actual = tensor.dtype();
    T::try_into_typed(tensor).ok_or_else(|| Error::DTypeMismatch {
        op,
        lhs: actual,
        rhs: T::dtype(),
    })
}

impl<T: TensorScalar> TypedTensor<T> {
    /// Singular value decomposition: `A = U diag(S) Vt`.
    ///
    /// Returns `(U, S, Vt)` using the thin/economy SVD.
    ///
    /// For complex inputs, this wrapper expects the backend to return the
    /// singular values as `TypedTensor<T::Real>`. If the backend still returns
    /// a complex tensor for `S`, this method returns [`Error::DTypeMismatch`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, TensorBackend, TypedTensor};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]);
    /// let (u, s, vt) = a.svd(&mut ctx).unwrap();
    ///
    /// assert_eq!(u.shape, vec![2, 2]);
    /// assert_eq!(s.shape, vec![2]);
    /// assert_eq!(vt.shape, vec![2, 2]);
    /// ```
    pub fn svd(&self, ctx: &mut impl TensorBackend) -> Result<(Self, TypedTensor<T::Real>, Self)> {
        let tensor = T::into_tensor(self.shape.clone(), self.host_data().to_vec());
        let (u, s, vt) = tensor.svd(ctx)?;
        Ok((
            try_into_typed_result("svd", u)?,
            try_into_typed_result("svd", s)?,
            try_into_typed_result("svd", vt)?,
        ))
    }

    /// QR decomposition: `A = Q R`.
    ///
    /// Returns `(Q, R)` using the thin/economy QR decomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, TensorBackend, TypedTensor};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// let (q, r) = a.qr(&mut ctx).unwrap();
    ///
    /// assert_eq!(q.shape, vec![2, 2]);
    /// assert_eq!(r.shape, vec![2, 2]);
    /// ```
    pub fn qr(&self, ctx: &mut impl TensorBackend) -> Result<(Self, Self)> {
        let tensor = T::into_tensor(self.shape.clone(), self.host_data().to_vec());
        let (q, r) = tensor.qr(ctx)?;
        Ok((
            try_into_typed_result("qr", q)?,
            try_into_typed_result("qr", r)?,
        ))
    }

    /// Cholesky factorization: `A = L L^T` or `A = L L^H`.
    ///
    /// Returns the lower-triangular factor `L`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, TensorBackend, TypedTensor};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]);
    /// let l = a.cholesky(&mut ctx).unwrap();
    ///
    /// assert_eq!(l.shape, vec![2, 2]);
    /// ```
    pub fn cholesky(&self, ctx: &mut impl TensorBackend) -> Result<Self> {
        let tensor = T::into_tensor(self.shape.clone(), self.host_data().to_vec());
        let factor = tensor.cholesky(ctx)?;
        try_into_typed_result("cholesky", factor)
    }

    /// Symmetric or Hermitian eigendecomposition: `A = V diag(W) V^T`.
    ///
    /// Returns `(eigenvalues, eigenvectors)`.
    ///
    /// For complex inputs, this wrapper expects the backend to return the
    /// eigenvalues as `TypedTensor<T::Real>`. If the backend still returns a
    /// complex tensor for `W`, this method returns [`Error::DTypeMismatch`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, TensorBackend, TypedTensor};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]);
    /// let (w, v) = a.eigh(&mut ctx).unwrap();
    ///
    /// assert_eq!(w.shape, vec![2]);
    /// assert_eq!(v.shape, vec![2, 2]);
    /// ```
    pub fn eigh(&self, ctx: &mut impl TensorBackend) -> Result<(TypedTensor<T::Real>, Self)> {
        let tensor = T::into_tensor(self.shape.clone(), self.host_data().to_vec());
        let (w, v) = tensor.eigh(ctx)?;
        Ok((
            try_into_typed_result("eigh", w)?,
            try_into_typed_result("eigh", v)?,
        ))
    }
}
