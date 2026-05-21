use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::{RuntimeCacheControl, Tensor, TensorRead};

/// Canonical elementwise fusion plan shared between segmented execution and backends.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ElementwiseFusionPlan {
    pub dtype: crate::DType,
    pub n_inputs: usize,
    pub outputs: Vec<usize>,
    pub ops: Vec<ElementwiseFusionInst>,
}

/// One node in a canonical elementwise fusion plan.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ElementwiseFusionInst {
    pub op: ElementwiseFusionOp,
    pub inputs: Vec<usize>,
}

/// Elementwise op kinds supported by backend fusion implementations.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ElementwiseFusionOp {
    Add,
    Multiply,
    Negate,
    Conj,
    Divide,
    Abs,
    Maximum,
    Minimum,
    Compare(CompareDir),
    Select,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
}

/// Execution session surface for dense tensor backends.
///
/// All operations run within a backend-owned execution scope such as a CPU
/// rayon pool or a GPU stream. Individual ops must not try to re-enter that
/// scope.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{cpu::CpuBackend, Tensor, TensorBackend, TypedTensor};
///
/// let mut backend = CpuBackend::new();
/// let a = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
/// let b = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, 4.0]));
/// let sum = backend
///     .with_exec_session(|exec| exec.add(&a, &b))
///     .unwrap();
/// assert_eq!(sum.shape(), &[2]);
/// ```
pub trait TensorExec {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor>;
    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;

    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor>;

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor>;
    fn convert(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor>;
    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;
    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor>;

    #[doc(hidden)]
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => self.dot_general(lhs, rhs, config),
            _ => {
                let lhs = lhs.to_tensor();
                let rhs = rhs.to_tensor();
                self.dot_general(&lhs, &rhs, config)
            }
        }
    }

    #[doc(hidden)]
    fn dot_general_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general(lhs, rhs, config)
    }

    #[doc(hidden)]
    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        if !lhs_conj && !rhs_conj {
            return self.dot_general(lhs, rhs, config);
        }

        let lhs_tmp;
        let lhs_ref = if lhs_conj {
            lhs_tmp = self.conj(lhs)?;
            &lhs_tmp
        } else {
            lhs
        };
        let rhs_tmp;
        let rhs_ref = if rhs_conj {
            rhs_tmp = self.conj(rhs)?;
            &rhs_tmp
        } else {
            rhs
        };
        self.dot_general(lhs_ref, rhs_ref, config)
    }

    #[doc(hidden)]
    fn dot_general_with_conj_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_with_conj(lhs, rhs, config, lhs_conj, rhs_conj)
    }

    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor>;
    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor>;
    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor>;
    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor>;
    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    fn cholesky(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> crate::Result<Tensor>;
    fn lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn full_piv_lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> crate::Result<Tensor>;
    fn svd(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn qr(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn eigh(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn eig(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;

    fn reclaim_buffer(&mut self, tensor: Tensor);

    #[doc(hidden)]
    fn execute_elementwise_fusion(
        &mut self,
        _inputs: &[&Tensor],
        _plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        Ok(None)
    }
}

struct BackendExecAdapter<'a, B: TensorBackend + ?Sized> {
    backend: &'a mut B,
}

macro_rules! forward_exec_to_backend {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                self.backend.$name($($arg),*)
            }
        )+
    };
}

impl<B: TensorBackend + ?Sized> TensorExec for BackendExecAdapter<'_, B> {
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.backend.dot_general_read(lhs, rhs, config)
    }

    forward_exec_to_backend! {
        add(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        neg(input: &Tensor) -> crate::Result<Tensor>;
        conj(input: &Tensor) -> crate::Result<Tensor>;
        div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        abs(input: &Tensor) -> crate::Result<Tensor>;
        sign(input: &Tensor) -> crate::Result<Tensor>;
        maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor>;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
        exp(input: &Tensor) -> crate::Result<Tensor>;
        log(input: &Tensor) -> crate::Result<Tensor>;
        sin(input: &Tensor) -> crate::Result<Tensor>;
        cos(input: &Tensor) -> crate::Result<Tensor>;
        tanh(input: &Tensor) -> crate::Result<Tensor>;
        sqrt(input: &Tensor) -> crate::Result<Tensor>;
        rsqrt(input: &Tensor) -> crate::Result<Tensor>;
        pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
        expm1(input: &Tensor) -> crate::Result<Tensor>;
        log1p(input: &Tensor) -> crate::Result<Tensor>;
        transpose(input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
        reshape(input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
        broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> crate::Result<Tensor>;
        convert(input: &Tensor, to: crate::DType) -> crate::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> crate::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> crate::Result<Tensor>;
        reduce_sum(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        reduce_prod(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        reduce_max(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        reduce_min(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> crate::Result<Tensor>;
        dot_general_with_conj(
            lhs: &Tensor,
            rhs: &Tensor,
            config: &DotGeneralConfig,
            lhs_conj: bool,
            rhs_conj: bool
        ) -> crate::Result<Tensor>;
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> crate::Result<Tensor>;
        scatter(
            operand: &Tensor,
            scatter_indices: &Tensor,
            updates: &Tensor,
            config: &ScatterConfig
        ) -> crate::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> crate::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> crate::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
        cholesky(input: &Tensor) -> crate::Result<Tensor>;
        triangular_solve(
            a: &Tensor,
            b: &Tensor,
            left_side: bool,
            lower: bool,
            transpose_a: bool,
            unit_diagonal: bool
        ) -> crate::Result<Tensor>;
        lu(input: &Tensor) -> crate::Result<Vec<Tensor>>;
        full_piv_lu(input: &Tensor) -> crate::Result<Vec<Tensor>>;
        full_piv_lu_solve(a: &Tensor, b: &Tensor, transpose_a: bool) -> crate::Result<Tensor>;
        svd(input: &Tensor) -> crate::Result<Vec<Tensor>>;
        qr(input: &Tensor) -> crate::Result<Vec<Tensor>>;
        eigh(input: &Tensor) -> crate::Result<Vec<Tensor>>;
        eig(input: &Tensor) -> crate::Result<Vec<Tensor>>;
        reclaim_buffer(tensor: Tensor) -> ();
        execute_elementwise_fusion(
            inputs: &[&Tensor],
            plan: &ElementwiseFusionPlan
        ) -> crate::Result<Option<Vec<Tensor>>>;
    }
}

/// Run a closure using the default execution-session adapter.
///
/// This forwards [`TensorExec`] calls back to the backend's existing
/// [`TensorBackend`] methods, which is suitable for backends whose individual
/// ops already manage their own execution context.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{cpu::CpuBackend, default_exec_session};
///
/// let mut backend = CpuBackend::new();
/// let _ = default_exec_session(&mut backend, |_exec| 1usize);
/// ```
pub fn default_exec_session<B: TensorBackend + ?Sized, R: Send>(
    backend: &mut B,
    f: impl FnOnce(&mut dyn TensorExec) -> R + Send,
) -> R {
    let mut adapter = BackendExecAdapter { backend };
    f(&mut adapter)
}

/// Standard runtime backend over dynamic [`Tensor`] values.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{cpu::CpuBackend, TensorBackend};
///
/// let mut backend = CpuBackend::new();
/// ```
pub trait TensorBackend {
    #[doc(hidden)]
    type RuntimeCache: RuntimeCacheControl;

    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor>;
    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;

    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor>;

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor>;
    fn convert(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor>;
    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;
    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor>;

    #[doc(hidden)]
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => self.dot_general(lhs, rhs, config),
            _ => {
                let lhs = self.upload_host_tensor(&lhs.to_tensor())?;
                let rhs = self.upload_host_tensor(&rhs.to_tensor())?;
                self.dot_general(&lhs, &rhs, config)
            }
        }
    }

    #[doc(hidden)]
    fn dot_general_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general(lhs, rhs, config)
    }

    #[doc(hidden)]
    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        if !lhs_conj && !rhs_conj {
            return self.dot_general(lhs, rhs, config);
        }

        let lhs_tmp;
        let lhs_ref = if lhs_conj {
            lhs_tmp = self.conj(lhs)?;
            &lhs_tmp
        } else {
            lhs
        };
        let rhs_tmp;
        let rhs_ref = if rhs_conj {
            rhs_tmp = self.conj(rhs)?;
            &rhs_tmp
        } else {
            rhs
        };
        self.dot_general(lhs_ref, rhs_ref, config)
    }

    #[doc(hidden)]
    fn dot_general_with_conj_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_with_conj(lhs, rhs, config, lhs_conj, rhs_conj)
    }

    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor>;
    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor>;
    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor>;
    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor>;
    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    fn cholesky(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> crate::Result<Tensor>;
    fn lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn full_piv_lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> crate::Result<Tensor>;
    fn svd(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn qr(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn eigh(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn eig(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>>;
    fn solve(&mut self, a: &Tensor, b: &Tensor) -> crate::Result<Tensor>;

    /// Execute a batch of operations inside the backend's execution context.
    ///
    /// Backends can override this to establish one shared scope for many ops,
    /// such as a rayon pool install on CPU.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{cpu::CpuBackend, TensorBackend};
    ///
    /// let mut backend = CpuBackend::new();
    /// let _value = backend.with_exec_session(|_exec| 1usize);
    /// ```
    fn with_exec_session<R: Send>(&mut self, f: impl FnOnce(&mut dyn TensorExec) -> R + Send) -> R {
        default_exec_session(self, f)
    }

    /// Execute a batch of operations with an externally owned runtime cache.
    ///
    /// The default implementation ignores the cache. Backends can override
    /// this to use Engine-owned prepared plans or analysis caches while keeping
    /// cache lifetime and clearing under the caller's control.
    #[doc(hidden)]
    fn with_exec_session_cached<R: Send>(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        f: impl FnOnce(&mut dyn TensorExec) -> R + Send,
    ) -> R {
        self.with_exec_session(f)
    }

    /// Materialize a backend tensor into host memory.
    ///
    /// Backends that already operate on host tensors can keep the default
    /// implementation, which clones the input tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, Tensor, TensorBackend, TypedTensor};
    ///
    /// let mut backend = CpuBackend::new();
    /// let tensor = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
    /// let host = backend.download_to_host(&tensor).unwrap();
    /// assert_eq!(host.shape(), &[2]);
    /// ```
    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        Ok(tensor.clone())
    }

    /// Upload a host tensor into backend-owned storage when needed.
    ///
    /// Backends that already use host tensors can keep the default
    /// implementation, which clones the input tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{cpu::CpuBackend, Tensor, TensorBackend, TypedTensor};
    ///
    /// let mut backend = CpuBackend::new();
    /// let tensor = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
    /// let uploaded = backend.upload_host_tensor(&tensor).unwrap();
    /// assert_eq!(uploaded.shape(), &[2]);
    /// ```
    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        Ok(tensor.clone())
    }

    /// Reclaim a tensor buffer for backend-specific reuse.
    ///
    /// Backends that do not pool buffers can ignore the tensor and let it drop.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{cpu::CpuBackend, Tensor, TensorBackend, TypedTensor};
    ///
    /// let mut backend = CpuBackend::new();
    /// let tensor = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
    /// backend.reclaim_buffer(tensor);
    /// ```
    fn reclaim_buffer(&mut self, _tensor: Tensor) {}

    #[doc(hidden)]
    fn execute_elementwise_fusion(
        &mut self,
        _inputs: &[&Tensor],
        _plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        Ok(None)
    }
}
