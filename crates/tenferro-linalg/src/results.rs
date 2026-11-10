//! Named results of the multi-output linalg operations (#2552 rule 5).
//!
//! Each struct is generic over the tensor type of its outputs, so the same struct serves
//! every layer: `Tensor` or `TypedTensor<T>` on the concrete layer, `TracedTensor` on the
//! traced layer and `EagerTensor` on the eager layer. Where an operation returns outputs
//! of two dtypes, the second type parameter carries the other dtype (real singular values
//! of a complex SVD, for example); it defaults to the first.

/// Singular value decomposition `A = U · diag(S) · Vᴴ`.
///
/// `s` is real and descending; `u` and `vh` have the input dtype. Their shapes depend on
/// [`SvdShape`](crate::SvdShape).
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_linalg::{Svd, TensorLinalgExt};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 2.0])?;
/// let Svd { u, s, vh } = a.svd(&mut CpuEngine::serial())?;
/// assert_eq!(s.as_slice::<f64>()?, &[3.0, 2.0]);
/// assert_eq!((u.shape(), vh.shape()), (&[2, 2][..], &[2, 2][..]));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Svd<V, R = V> {
    /// Left singular vectors.
    pub u: V,
    /// Singular values, real dtype.
    pub s: R,
    /// Conjugate-transposed right singular vectors.
    pub vh: V,
}

/// QR decomposition `A = Q · R`.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_linalg::{Qr, TensorLinalgExt};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0])?;
/// let Qr { q, r } = a.qr(&mut CpuEngine::serial())?;
/// assert_eq!((q.shape(), r.shape()), (&[2, 2][..], &[2, 2][..]));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Qr<V> {
    /// Factor with orthonormal columns.
    pub q: V,
    /// Upper triangular factor.
    pub r: V,
}

/// LU decomposition with partial pivoting, `A = P · L · U`.
///
/// `parity` is the permutation sign (`±1`) in the real dtype.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_linalg::{Lu, TensorLinalgExt};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0])?;
/// let Lu { p, l, u, parity } = a.lu(&mut CpuEngine::serial())?;
/// assert_eq!((p.shape(), l.shape(), u.shape()), (&[2, 2][..], &[2, 2][..], &[2, 2][..]));
/// assert_eq!(parity.as_slice::<f64>()?, &[-1.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Lu<V, R = V> {
    /// Row permutation matrix.
    pub p: V,
    /// Unit lower triangular factor.
    pub l: V,
    /// Upper triangular factor.
    pub u: V,
    /// Permutation sign, real dtype.
    pub parity: R,
}

/// LU decomposition with complete pivoting, `A = P · L · U · Q`.
///
/// `parity` is the combined row and column permutation sign in the real dtype.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_linalg::{FullPivLu, TensorLinalgExt};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0])?;
/// let FullPivLu { p, l, u, q, parity } = a.full_piv_lu(&mut CpuEngine::serial())?;
/// assert_eq!((p.shape(), q.shape()), (&[2, 2][..], &[2, 2][..]));
/// assert_eq!((l.shape(), u.shape(), parity.shape()), (&[2, 2][..], &[2, 2][..], &[][..]));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct FullPivLu<V, R = V> {
    /// Row permutation matrix.
    pub p: V,
    /// Unit lower triangular factor.
    pub l: V,
    /// Upper triangular factor.
    pub u: V,
    /// Column permutation matrix.
    pub q: V,
    /// Permutation sign, real dtype.
    pub parity: R,
}

/// General eigendecomposition `A · V = V · diag(W)`, in the complex dtype.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_linalg::{Eig, TensorLinalgExt};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0])?;
/// let Eig { values, vectors } = a.eig(&mut CpuEngine::serial())?;
/// assert_eq!((values.shape(), vectors.shape()), (&[2][..], &[2, 2][..]));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Eig<C> {
    /// Eigenvalues.
    pub values: C,
    /// Eigenvectors, one per column.
    pub vectors: C,
}

/// Hermitian eigendecomposition `A · V = V · diag(W)` with real ascending eigenvalues.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_linalg::{Eigh, TensorLinalgExt};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 1.0])?;
/// let Eigh { values, vectors } = a.eigh(&mut CpuEngine::serial())?;
/// assert_eq!(values.as_slice::<f64>()?, &[1.0, 2.0]);
/// assert_eq!(vectors.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Eigh<V, R = V> {
    /// Eigenvalues, real dtype.
    pub values: R,
    /// Eigenvectors, one per column, input dtype.
    pub vectors: V,
}

/// Sign and log-absolute value of the determinant, `det(A) = sign · exp(logabsdet)`.
///
/// `sign` has the input dtype (a unit-modulus phase for complex input); `logabsdet` is real.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuEngine;
/// use tenferro_linalg::{Slogdet, TensorLinalgExt};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0])?;
/// let Slogdet { sign, logabsdet } = a.slogdet(&mut CpuEngine::serial())?;
/// assert_eq!(sign.as_slice::<f64>()?, &[1.0]);
/// assert!((logabsdet.as_slice::<f64>()?[0] - 8.0_f64.ln()).abs() < 1e-12);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Slogdet<V, R = V> {
    /// Sign or phase of the determinant, input dtype.
    pub sign: V,
    /// Natural log of the absolute determinant, real dtype.
    pub logabsdet: R,
}
