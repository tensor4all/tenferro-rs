use super::*;

pub struct SvdBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    options: Option<&'a SvdOptions>,
}

impl<'a, T> SvdBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets optional SVD options.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.options(&options);
    /// ```
    pub fn options(mut self, options: &'a SvdOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Executes SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<SvdResult<T, T::Real>> {
        with_runtime_cpu_only("svd", |ctx| {
            tenferro_linalg::svd::<T, CpuContext>(ctx, self.tensor, self.options)
                .map_err(Error::from)
        })
    }
}

/// Creates an SVD builder.
/// # Examples
///
/// ```ignore
/// let _ = svd(/* ... */);
/// ```
pub fn svd<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> SvdBuilder<'a, T> {
    SvdBuilder {
        tensor,
        options: None,
    }
}

/// Builder for QR.
/// # Examples
///
/// ```text
/// // Construct `QrBuilder` via its corresponding operation constructor.
/// ```
pub struct QrBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> QrBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes QR.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<QrResult<T>> {
        with_runtime_cpu_only("qr", |ctx| {
            tenferro_linalg::qr::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a QR builder.
/// # Examples
///
/// ```ignore
/// let _ = qr(/* ... */);
/// ```
pub fn qr<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> QrBuilder<'a, T> {
    QrBuilder { tensor }
}

/// Builder for LU.
/// # Examples
///
/// ```text
/// // Construct `LuBuilder` via its corresponding operation constructor.
/// ```
pub struct LuBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    pivot: LuPivot,
}

impl<'a, T> LuBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets LU pivoting policy.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivot(pivot);
    /// ```
    pub fn pivot(mut self, pivot: LuPivot) -> Self {
        self.pivot = pivot;
        self
    }

    /// Executes LU.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LuResult<T>> {
        with_runtime_cpu_only("lu", |ctx| {
            tenferro_linalg::lu::<T, CpuContext>(ctx, self.tensor, self.pivot).map_err(Error::from)
        })
    }
}

/// Creates an LU builder.
/// # Examples
///
/// ```ignore
/// let _ = lu(/* ... */);
/// ```
pub fn lu<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> LuBuilder<'a, T> {
    LuBuilder {
        tensor,
        pivot: LuPivot::Partial,
    }
}

/// Builder for packed LU factorization.
/// # Examples
///
/// ```text
/// // Construct `LuFactorBuilder` via its corresponding operation constructor.
/// ```
pub struct LuFactorBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> LuFactorBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes packed LU factorization.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LuFactorResult<T>> {
        with_runtime_cpu_only("lu_factor", |ctx| {
            tenferro_linalg::lu_factor::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a packed LU factorization builder.
/// # Examples
///
/// ```ignore
/// let _ = lu_factor(/* ... */);
/// ```
pub fn lu_factor<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> LuFactorBuilder<'a, T> {
    LuFactorBuilder { tensor }
}

/// Builder for packed LU factorization with numerical status.
/// # Examples
///
/// ```text
/// // Construct `LuFactorExBuilder` via its corresponding operation constructor.
/// ```
pub struct LuFactorExBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> LuFactorExBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes structured LU factorization.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LuFactorExResult<T>> {
        with_runtime_cpu_only("lu_factor_ex", |ctx| {
            tenferro_linalg::lu_factor_ex::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a structured LU factorization builder.
/// # Examples
///
/// ```ignore
/// let _ = lu_factor_ex(/* ... */);
/// ```
pub fn lu_factor_ex<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> LuFactorExBuilder<'a, T> {
    LuFactorExBuilder { tensor }
}

/// Builder for eigen decomposition.
/// # Examples
///
/// ```text
/// // Construct `EigenBuilder` via its corresponding operation constructor.
/// ```
pub struct EigenBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> EigenBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes eigen decomposition.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<EigenResult<T, T::Real>> {
        with_runtime_cpu_only("eigen", |ctx| {
            tenferro_linalg::eigen::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an eigen builder.
/// # Examples
///
/// ```ignore
/// let _ = eigen(/* ... */);
/// ```
pub fn eigen<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> EigenBuilder<'a, T> {
    EigenBuilder { tensor }
}

/// Builder for least squares solve.
/// # Examples
///
/// ```text
/// // Construct `LstsqBuilder` via its corresponding operation constructor.
/// ```
pub struct LstsqBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
}

impl<'a, T> LstsqBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes least squares.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LstsqResult<T>> {
        with_runtime_cpu_only("lstsq", |ctx| {
            tenferro_linalg::lstsq::<T, CpuContext>(ctx, self.a, self.b).map_err(Error::from)
        })
    }
}

/// Creates an lstsq builder.
/// # Examples
///
/// ```ignore
/// let _ = lstsq(/* ... */);
/// ```
pub fn lstsq<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> LstsqBuilder<'a, T> {
    LstsqBuilder { a, b }
}

/// Builder for Cholesky.
/// # Examples
///
/// ```text
/// // Construct `CholeskyBuilder` via its corresponding operation constructor.
/// ```
pub struct CholeskyBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> CholeskyBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes Cholesky.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("cholesky", |ctx| {
            tenferro_linalg::cholesky::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a cholesky builder.
/// # Examples
///
/// ```ignore
/// let _ = cholesky(/* ... */);
/// ```
pub fn cholesky<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> CholeskyBuilder<'a, T> {
    CholeskyBuilder { tensor }
}

/// Builder for structured Cholesky with numerical status.
/// # Examples
///
/// ```text
/// // Construct `CholeskyExBuilder` via its corresponding operation constructor.
/// ```
pub struct CholeskyExBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> CholeskyExBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes structured Cholesky.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<CholeskyExResult<T>> {
        with_runtime_cpu_only("cholesky_ex", |ctx| {
            tenferro_linalg::cholesky_ex::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a structured cholesky builder.
/// # Examples
///
/// ```ignore
/// let _ = cholesky_ex(/* ... */);
/// ```
pub fn cholesky_ex<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> CholeskyExBuilder<'a, T> {
    CholeskyExBuilder { tensor }
}

/// Builder for solve.
/// # Examples
///
/// ```text
/// // Construct `SolveBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
}

impl<'a, T> SolveBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes linear solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("solve", |ctx| {
            tenferro_linalg::solve::<T, CpuContext>(ctx, self.a, self.b).map_err(Error::from)
        })
    }
}

/// Creates a solve builder.
/// # Examples
///
/// ```ignore
/// let _ = solve(/* ... */);
/// ```
pub fn solve<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> SolveBuilder<'a, T> {
    SolveBuilder { a, b }
}

/// Builder for structured solve with numerical status.
/// # Examples
///
/// ```text
/// // Construct `SolveExBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveExBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
}

impl<'a, T> SolveExBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes structured solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<SolveExResult<T>> {
        with_runtime_cpu_only("solve_ex", |ctx| {
            tenferro_linalg::solve_ex::<T, CpuContext>(ctx, self.a, self.b).map_err(Error::from)
        })
    }
}

/// Creates a structured solve builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_ex(/* ... */);
/// ```
pub fn solve_ex<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> SolveExBuilder<'a, T> {
    SolveExBuilder { a, b }
}

/// Builder for LU-based solve from packed factors.
/// # Examples
///
/// ```text
/// // Construct `LuSolveBuilder` via its corresponding operation constructor.
/// ```
pub struct LuSolveBuilder<'a, T: LinalgScalar> {
    factors: &'a Tensor<T>,
    b: &'a Tensor<T>,
    pivots: Option<&'a [usize]>,
}

impl<'a, T> LuSolveBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets forward row-permutation indices from `lu_factor`.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivots(&pivots);
    /// ```
    pub fn pivots(mut self, pivots: &'a [usize]) -> Self {
        self.pivots = Some(pivots);
        self
    }

    /// Executes packed LU solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        let pivots = self.pivots.ok_or_else(|| {
            Error::Backend(tenferro_device::Error::InvalidArgument(
                "lu_solve builder requires `.pivots(&[..])` before `run()`".into(),
            ))
        })?;
        with_runtime_cpu_only("lu_solve", |ctx| {
            tenferro_linalg::lu_solve::<T, CpuContext>(ctx, self.factors, pivots, self.b)
                .map_err(Error::from)
        })
    }
}

/// Creates an LU solve builder.
/// # Examples
///
/// ```ignore
/// let _ = lu_solve(/* ... */);
/// ```
pub fn lu_solve<'a, T: LinalgScalar>(
    factors: &'a Tensor<T>,
    b: &'a Tensor<T>,
) -> LuSolveBuilder<'a, T> {
    LuSolveBuilder {
        factors,
        b,
        pivots: None,
    }
}

/// Builder for matrix inverse.
/// # Examples
///
/// ```text
/// // Construct `InvBuilder` via its corresponding operation constructor.
/// ```
pub struct InvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> InvBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes matrix inverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("inv", |ctx| {
            tenferro_linalg::inv::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an inv builder.
/// # Examples
///
/// ```ignore
/// let _ = inv(/* ... */);
/// ```
pub fn inv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> InvBuilder<'a, T> {
    InvBuilder { tensor }
}

/// Builder for structured inverse with numerical status.
/// # Examples
///
/// ```text
/// // Construct `InvExBuilder` via its corresponding operation constructor.
/// ```
pub struct InvExBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> InvExBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes structured inverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<InvExResult<T>> {
        with_runtime_cpu_only("inv_ex", |ctx| {
            tenferro_linalg::inv_ex::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a structured inv builder.
/// # Examples
///
/// ```ignore
/// let _ = inv_ex(/* ... */);
/// ```
pub fn inv_ex<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> InvExBuilder<'a, T> {
    InvExBuilder { tensor }
}

/// Builder for determinant.
/// # Examples
///
/// ```text
/// // Construct `DetBuilder` via its corresponding operation constructor.
/// ```
pub struct DetBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> DetBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes determinant.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("det", |ctx| {
            tenferro_linalg::det::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a det builder.
/// # Examples
///
/// ```ignore
/// let _ = det(/* ... */);
/// ```
pub fn det<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> DetBuilder<'a, T> {
    DetBuilder { tensor }
}

/// Builder for slogdet.
/// # Examples
///
/// ```text
/// // Construct `SlogdetBuilder` via its corresponding operation constructor.
/// ```
pub struct SlogdetBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> SlogdetBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Executes slogdet.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<SlogdetResult<T>> {
        with_runtime_cpu_only("slogdet", |ctx| {
            tenferro_linalg::slogdet::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an slogdet builder.
/// # Examples
///
/// ```ignore
/// let _ = slogdet(/* ... */);
/// ```
pub fn slogdet<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> SlogdetBuilder<'a, T> {
    SlogdetBuilder { tensor }
}

/// Builder for general eigendecomposition.
/// # Examples
///
/// ```text
/// // Construct `EigBuilder` via its corresponding operation constructor.
/// ```
pub struct EigBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> EigBuilder<'a, T>
where
    T: LinalgScalar<Real = T, Complex = Complex<T>> + Float + CpuLinalgScalar,
{
    /// Executes eig.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<EigResult<T>> {
        with_runtime_cpu_only("eig", |ctx| {
            tenferro_linalg::eig::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates an eig builder.
/// # Examples
///
/// ```ignore
/// let _ = eig(/* ... */);
/// ```
pub fn eig<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> EigBuilder<'a, T> {
    EigBuilder { tensor }
}

/// Builder for pseudoinverse.
/// # Examples
///
/// ```text
/// // Construct `PinvBuilder` via its corresponding operation constructor.
/// ```
pub struct PinvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    rcond: Option<f64>,
}

impl<'a, T> PinvBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets rcond.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.rcond(1e-12);
    /// ```
    pub fn rcond(mut self, rcond: f64) -> Self {
        self.rcond = Some(rcond);
        self
    }

    /// Executes pseudoinverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("pinv", |ctx| {
            tenferro_linalg::pinv::<T, CpuContext>(ctx, self.tensor, self.rcond)
                .map_err(Error::from)
        })
    }
}

/// Creates a pinv builder.
/// # Examples
///
/// ```ignore
/// let _ = pinv(/* ... */);
/// ```
pub fn pinv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> PinvBuilder<'a, T> {
    PinvBuilder {
        tensor,
        rcond: None,
    }
}

/// Builder for matrix exponential.
/// # Examples
///
/// ```text
/// // Construct `MatrixExpBuilder` via its corresponding operation constructor.
/// ```
pub struct MatrixExpBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
}

impl<'a, T> MatrixExpBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes matrix exponential.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("matrix_exp", |ctx| {
            tenferro_linalg::matrix_exp::<T, CpuContext>(ctx, self.tensor).map_err(Error::from)
        })
    }
}

/// Creates a matrix_exp builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_exp(/* ... */);
/// ```
pub fn matrix_exp<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> MatrixExpBuilder<'a, T> {
    MatrixExpBuilder { tensor }
}

/// Builder for integer matrix powers.
/// # Examples
///
/// ```text
/// // Construct `MatrixPowerBuilder` via its corresponding operation constructor.
/// ```
pub struct MatrixPowerBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    exponent: i64,
}

impl<'a, T> MatrixPowerBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets the integer exponent.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.exponent(3);
    /// ```
    pub fn exponent(mut self, exponent: i64) -> Self {
        self.exponent = exponent;
        self
    }

    /// Executes matrix_power.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("matrix_power", |ctx| {
            tenferro_linalg::matrix_power::<T, CpuContext>(ctx, self.tensor, self.exponent)
                .map_err(Error::from)
        })
    }
}

/// Creates a matrix_power builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_power(/* ... */);
/// ```
pub fn matrix_power<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> MatrixPowerBuilder<'a, T> {
    MatrixPowerBuilder {
        tensor,
        exponent: 1,
    }
}

/// Builder for triangular solve.
/// # Examples
///
/// ```text
/// // Construct `SolveTriangularBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveTriangularBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
    upper: bool,
}

impl<'a, T> SolveTriangularBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets whether the matrix is upper triangular.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.upper(true);
    /// ```
    pub fn upper(mut self, upper: bool) -> Self {
        self.upper = upper;
        self
    }

    /// Executes triangular solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("solve_triangular", |ctx| {
            tenferro_linalg::solve_triangular::<T, CpuContext>(ctx, self.a, self.b, self.upper)
                .map_err(Error::from)
        })
    }
}

/// Creates a solve_triangular builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_triangular(/* ... */);
/// ```
pub fn solve_triangular<'a, T: LinalgScalar>(
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
) -> SolveTriangularBuilder<'a, T> {
    SolveTriangularBuilder { a, b, upper: true }
}

/// Builder for norm.
/// # Examples
///
/// ```text
/// // Construct `NormBuilder` via its corresponding operation constructor.
/// ```
pub struct NormBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    kind: NormKind,
}

impl<'a, T> NormBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets norm kind.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.kind(kind);
    /// ```
    pub fn kind(mut self, kind: NormKind) -> Self {
        self.kind = kind;
        self
    }

    /// Executes norm.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("norm", |ctx| {
            tenferro_linalg::norm::<T, CpuContext>(ctx, self.tensor, self.kind).map_err(Error::from)
        })
    }
}

/// Creates a norm builder.
/// # Examples
///
/// ```ignore
/// let _ = norm(/* ... */);
/// ```
pub fn norm<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> NormBuilder<'a, T> {
    NormBuilder {
        tensor,
        kind: NormKind::Fro,
    }
}

/// Builder for condition numbers.
/// # Examples
///
/// ```text
/// // Construct `CondBuilder` via its corresponding operation constructor.
/// ```
pub struct CondBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    kind: NormKind,
}

impl<'a, T> CondBuilder<'a, T>
where
    T: LinalgScalar<Real = T> + Float + CpuLinalgScalar,
{
    /// Sets the norm kind used in the condition number.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.kind(NormKind::Spectral);
    /// ```
    pub fn kind(mut self, kind: NormKind) -> Self {
        self.kind = kind;
        self
    }

    /// Executes cond.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("cond", |ctx| {
            tenferro_linalg::cond::<T, CpuContext>(ctx, self.tensor, self.kind).map_err(Error::from)
        })
    }
}

/// Creates a cond builder.
/// # Examples
///
/// ```ignore
/// let _ = cond(/* ... */);
/// ```
pub fn cond<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> CondBuilder<'a, T> {
    CondBuilder {
        tensor,
        kind: NormKind::Spectral,
    }
}

/// Builder for cross products.
/// # Examples
///
/// ```text
/// // Construct `CrossBuilder` via its corresponding operation constructor.
/// ```
pub struct CrossBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
}

impl<'a, T> CrossBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes cross.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("cross", |ctx| {
            tenferro_linalg::cross::<T, CpuContext>(ctx, self.a, self.b).map_err(Error::from)
        })
    }
}

/// Creates a cross builder.
/// # Examples
///
/// ```ignore
/// let _ = cross(/* ... */);
/// ```
pub fn cross<'a, T: LinalgScalar>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> CrossBuilder<'a, T> {
    CrossBuilder { a, b }
}

/// Builder for Householder products.
/// # Examples
///
/// ```text
/// // Construct `HouseholderProductBuilder` via its corresponding operation constructor.
/// ```
pub struct HouseholderProductBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    tau: &'a Tensor<T>,
}

impl<'a, T> HouseholderProductBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Executes householder_product.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("householder_product", |ctx| {
            tenferro_linalg::householder_product::<T, CpuContext>(ctx, self.a, self.tau)
                .map_err(Error::from)
        })
    }
}

/// Creates a householder_product builder.
/// # Examples
///
/// ```ignore
/// let _ = householder_product(/* ... */);
/// ```
pub fn householder_product<'a, T: LinalgScalar>(
    a: &'a Tensor<T>,
    tau: &'a Tensor<T>,
) -> HouseholderProductBuilder<'a, T> {
    HouseholderProductBuilder { a, tau }
}

/// Builder for Vandermonde matrices.
/// # Examples
///
/// ```text
/// // Construct `VanderBuilder` via its corresponding operation constructor.
/// ```
pub struct VanderBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    columns: Option<usize>,
    increasing: bool,
}

impl<'a, T> VanderBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets the output column count.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.columns(4);
    /// ```
    pub fn columns(mut self, columns: usize) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Sets whether powers increase from left to right.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.increasing(true);
    /// ```
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Executes vander.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("vander", |ctx| {
            tenferro_linalg::vander::<T, CpuContext>(
                ctx,
                self.tensor,
                self.columns,
                self.increasing,
            )
            .map_err(Error::from)
        })
    }
}

/// Creates a vander builder.
/// # Examples
///
/// ```ignore
/// let _ = vander(/* ... */);
/// ```
pub fn vander<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> VanderBuilder<'a, T> {
    VanderBuilder {
        tensor,
        columns: None,
        increasing: false,
    }
}

/// Builder for tensor inverses.
/// # Examples
///
/// ```text
/// // Construct `TensorinvBuilder` via its corresponding operation constructor.
/// ```
pub struct TensorinvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    ind: usize,
}

impl<'a, T> TensorinvBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets the partition point between left and right tensor dimensions.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.ind(2);
    /// ```
    pub fn ind(mut self, ind: usize) -> Self {
        self.ind = ind;
        self
    }

    /// Executes tensorinv.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("tensorinv", |ctx| {
            tenferro_linalg::tensorinv::<T, CpuContext>(ctx, self.tensor, self.ind)
                .map_err(Error::from)
        })
    }
}

/// Creates a tensorinv builder.
/// # Examples
///
/// ```ignore
/// let _ = tensorinv(/* ... */);
/// ```
pub fn tensorinv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> TensorinvBuilder<'a, T> {
    TensorinvBuilder { tensor, ind: 1 }
}

/// Builder for tensorized linear solves.
/// # Examples
///
/// ```text
/// // Construct `TensorsolveBuilder` via its corresponding operation constructor.
/// ```
pub struct TensorsolveBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
    dims: Option<&'a [usize]>,
}

impl<'a, T> TensorsolveBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets the solution axes to move before solving.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.dims(&[3, 2]);
    /// ```
    pub fn dims(mut self, dims: &'a [usize]) -> Self {
        self.dims = Some(dims);
        self
    }

    /// Executes tensorsolve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        with_runtime_cpu_only("tensorsolve", |ctx| {
            tenferro_linalg::tensorsolve::<T, CpuContext>(ctx, self.a, self.b, self.dims)
                .map_err(Error::from)
        })
    }
}

/// Creates a tensorsolve builder.
/// # Examples
///
/// ```ignore
/// let _ = tensorsolve(/* ... */);
/// ```
pub fn tensorsolve<'a, T: LinalgScalar>(
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
) -> TensorsolveBuilder<'a, T> {
    TensorsolveBuilder { a, b, dims: None }
}
