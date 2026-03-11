use super::*;

macro_rules! eager_unary {
    ($(#[$meta:meta])* fn $name:ident -> $ret:ty => $builder:ident ; where { $($bounds:tt)* }) => {
        $(#[$meta])*
        pub fn $name<T: Scalar>(tensor: &AdTensor<T>) -> Result<$ret>
        where
            $($bounds)*
        {
            super::$builder(tensor).run()
        }
    };
}

macro_rules! eager_binary {
    ($(#[$meta:meta])* fn $name:ident -> $ret:ty => $builder:ident ; where { $($bounds:tt)* }) => {
        $(#[$meta])*
        pub fn $name<T: Scalar>(a: &AdTensor<T>, b: &AdTensor<T>) -> Result<$ret>
        where
            $($bounds)*
        {
            super::$builder(a, b).run()
        }
    };
}

/// Eager AD einsum.
///
/// Equivalent to `crate::einsum_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::einsum("ij,jk->ik", &[&a, &b])?;
/// ```
pub fn einsum<'a, T>(subscripts: &'a str, operands: &'a [&'a AdTensor<T>]) -> Result<AdTensor<T>>
where
    T: EinsumRuntimeValue,
{
    super::einsum_ad(subscripts, operands).run()
}

/// Eager AD full reduction / sum.
///
/// Equivalent to `crate::sum_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::sum(&x)?;
/// ```
pub fn sum<T>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: ScalarRuntimeValue,
{
    super::sum_ad(tensor).run()
}

eager_unary!(
    /// Eager AD SVD.
    ///
    /// Equivalent to `crate::svd_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::svd(&a)?;
    /// ```
    fn svd -> AdSvdResult<T> => svd_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD QR.
    ///
    /// Equivalent to `crate::qr_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::qr(&a)?;
    /// ```
    fn qr -> AdQrResult<T> => qr_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD LU (partial pivot by default).
    ///
    /// Equivalent to `crate::lu_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::lu(&a)?;
    /// ```
    fn lu -> AdLuResult<T> => lu_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD symmetric/Hermitian eigen decomposition.
    ///
    /// Equivalent to `crate::eigen_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::eigen(&a)?;
    /// ```
    fn eigen -> AdEigenResult<T> => eigen_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_binary!(
    /// Eager AD least-squares solve.
    ///
    /// Equivalent to `crate::lstsq_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::lstsq(&a, &b)?;
    /// ```
    fn lstsq -> AdLstsqResult<T> => lstsq_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD Cholesky.
    ///
    /// Equivalent to `crate::cholesky_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::cholesky(&a)?;
    /// ```
    fn cholesky -> AdTensor<T> => cholesky_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_binary!(
    /// Eager AD linear solve.
    ///
    /// Equivalent to `crate::solve_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::solve(&a, &b)?;
    /// ```
    fn solve -> AdTensor<T> => solve_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD inverse.
    ///
    /// Equivalent to `crate::inv_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::inv(&a)?;
    /// ```
    fn inv -> AdTensor<T> => inv_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD determinant.
    ///
    /// Equivalent to `crate::det_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::det(&a)?;
    /// ```
    fn det -> AdTensor<T> => det_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD slogdet.
    ///
    /// Equivalent to `crate::slogdet_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::slogdet(&a)?;
    /// ```
    fn slogdet -> AdSlogdetResult<T> => slogdet_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD general eigendecomposition.
    ///
    /// Equivalent to `crate::eig_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::eig(&a)?;
    /// ```
    fn eig -> AdEigResult<T> => eig_ad;
    where {
        T: ComplexLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD pseudoinverse.
    ///
    /// Equivalent to `crate::pinv_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::pinv(&a)?;
    /// ```
    fn pinv -> AdTensor<T> => pinv_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD matrix exponential.
    ///
    /// Equivalent to `crate::matrix_exp_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::matrix_exp(&a)?;
    /// ```
    fn matrix_exp -> AdTensor<T> => matrix_exp_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_binary!(
    /// Eager AD triangular solve (upper=true by default).
    ///
    /// Equivalent to `crate::solve_triangular_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro_dyadtensor::ad::solve_triangular(&a, &b)?;
    /// ```
    fn solve_triangular -> AdTensor<T> => solve_triangular_ad;
    where {
        T: LinalgRuntimeValue,
    }
);

/// Eager AD norm (Frobenius by default).
///
/// Equivalent to `crate::norm_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::ad::norm(&a)?;
/// ```
pub fn norm<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: RealLinalgRuntimeValue,
{
    super::norm_ad(tensor).kind(NormKind::Fro).run()
}
