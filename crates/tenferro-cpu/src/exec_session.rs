use crate::buffer_pool::BufferPool;
use crate::{Tensor, TensorRead, TensorValue, TensorWrite};
use tenferro_tensor::backend::{
    dot_general_accum_via_temp, grouped_gemm_via_sequential, validate_dot_general_accumulation,
    validate_grouped_gemm, ElementwiseFusionPlan, GroupedGemmConfig,
};
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::{
    DotGeneralAccumulation, SessionCachedDot, TensorAnalytic, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};

use super::backend::reclaim_typed;
use super::{
    analytic, elementwise, gemm, indexing, materialize_tensor_read, reduction, structural,
    CpuContext,
};
use super::{CpuBackendKind, DotGeneralProvider};

pub(crate) struct CpuExecSession<'a> {
    #[cfg_attr(feature = "cpu-blas", allow(dead_code))]
    pub(crate) ctx: &'a CpuContext,
    pub(crate) buffers: &'a mut BufferPool,
    pub(crate) gemm_analysis_cache: &'a mut gemm::GemmAnalysisCache,
    pub(crate) kind: CpuBackendKind,
    pub(crate) dot_general_provider: DotGeneralProvider,
}

impl CpuExecSession<'_> {
    fn run_native<R: Send>(&mut self, op: impl FnOnce(&mut BufferPool) -> R + Send) -> R {
        let buffers = &mut *self.buffers;
        match self.kind {
            CpuBackendKind::Faer => op(buffers),
            CpuBackendKind::Blas => self.ctx.install(|| op(buffers)),
        }
    }
}

impl TensorDeviceTransfer for CpuExecSession<'_> {
    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        if tensor.is_backend_buffer() {
            return Err(crate::Error::backend_failure(
                "CpuBackend::download_to_host",
                "CPU backend received a backend buffer; download the tensor to host with its owning backend before CPU execution",
            ));
        }
        Ok(tensor.clone())
    }

    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        if tensor.is_backend_buffer() {
            return Err(crate::Error::backend_failure(
                "CpuBackend::upload_host_tensor",
                "CPU backend upload_host_tensor expects a host tensor; download backend buffers to host before CPU execution",
            ));
        }
        Ok(tensor.clone())
    }
}

/// Simple delegation: no dtype dispatch, no install.
macro_rules! delegate {
    ($name:ident($($arg:ident : $ty:ty),*) => $body:expr) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> {
            self.run_native(|_| $body)
        }
    };
}

/// Delegation for operations whose outputs can be allocated from the session pool.
macro_rules! delegate_with_pool {
    ($name:ident($($arg:ident : $ty:ty),*) => $callee:path) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> {
            self.run_native(|buffers| $callee(buffers, $($arg),*))
        }
    };
}

impl TensorElementwise for CpuExecSession<'_> {
    // Elementwise — direct delegation, no install
    delegate_with_pool!(add(lhs: &Tensor, rhs: &Tensor) => elementwise::add_with_pool);

    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.run_native(|buffers| elementwise::add_read_with_pool(buffers, lhs, rhs))
    }

    delegate_with_pool!(sub(lhs: &Tensor, rhs: &Tensor) => elementwise::sub_with_pool);
    delegate_with_pool!(sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::sub_read_with_pool);
    delegate_with_pool!(mul(lhs: &Tensor, rhs: &Tensor) => elementwise::mul_with_pool);
    delegate_with_pool!(mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::mul_read_with_pool);
    delegate_with_pool!(neg(input: &Tensor) => elementwise::neg_with_pool);
    delegate_with_pool!(neg_read(input: TensorRead<'_>) => elementwise::neg_read_with_pool);
    delegate_with_pool!(conj(input: &Tensor) => elementwise::conj_with_pool);
    delegate_with_pool!(conj_read(input: TensorRead<'_>) => elementwise::conj_read_with_pool);
    delegate_with_pool!(div(lhs: &Tensor, rhs: &Tensor) => elementwise::div_with_pool);
    delegate_with_pool!(div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::div_read_with_pool);
    delegate_with_pool!(rem(lhs: &Tensor, rhs: &Tensor) => elementwise::rem_with_pool);
    delegate_with_pool!(rem_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::rem_read_with_pool);
    delegate_with_pool!(abs(input: &Tensor) => elementwise::abs_with_pool);
    delegate_with_pool!(abs_read(input: TensorRead<'_>) => elementwise::abs_read_with_pool);
    delegate_with_pool!(sign(input: &Tensor) => elementwise::sign_with_pool);
    delegate_with_pool!(sign_read(input: TensorRead<'_>) => elementwise::sign_read_with_pool);
    delegate_with_pool!(maximum(lhs: &Tensor, rhs: &Tensor) => elementwise::maximum_with_pool);
    delegate_with_pool!(maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::maximum_read_with_pool);
    delegate_with_pool!(minimum(lhs: &Tensor, rhs: &Tensor) => elementwise::minimum_with_pool);
    delegate_with_pool!(minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => elementwise::minimum_read_with_pool);
    delegate_with_pool!(compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) => elementwise::compare_with_pool);
    delegate_with_pool!(compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) => elementwise::compare_read_with_pool);
    delegate_with_pool!(select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) => elementwise::select_with_pool);
    delegate_with_pool!(select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) => elementwise::select_read_with_pool);
    delegate_with_pool!(clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) => elementwise::clamp_with_pool);
    delegate_with_pool!(clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) => elementwise::clamp_read_with_pool);
}

impl TensorAnalytic for CpuExecSession<'_> {
    // Analytic
    delegate_with_pool!(exp(input: &Tensor) => analytic::exp_with_pool);
    delegate_with_pool!(exp_read(input: TensorRead<'_>) => analytic::exp_read_with_pool);
    delegate_with_pool!(log(input: &Tensor) => analytic::log_with_pool);
    delegate_with_pool!(log_read(input: TensorRead<'_>) => analytic::log_read_with_pool);
    delegate_with_pool!(sin(input: &Tensor) => analytic::sin_with_pool);
    delegate_with_pool!(sin_read(input: TensorRead<'_>) => analytic::sin_read_with_pool);
    delegate_with_pool!(cos(input: &Tensor) => analytic::cos_with_pool);
    delegate_with_pool!(cos_read(input: TensorRead<'_>) => analytic::cos_read_with_pool);
    delegate_with_pool!(tanh(input: &Tensor) => analytic::tanh_with_pool);
    delegate_with_pool!(tanh_read(input: TensorRead<'_>) => analytic::tanh_read_with_pool);
    delegate_with_pool!(sqrt(input: &Tensor) => analytic::sqrt_with_pool);
    delegate_with_pool!(sqrt_read(input: TensorRead<'_>) => analytic::sqrt_read_with_pool);
    delegate_with_pool!(rsqrt(input: &Tensor) => analytic::rsqrt_with_pool);
    delegate_with_pool!(rsqrt_read(input: TensorRead<'_>) => analytic::rsqrt_read_with_pool);
    delegate_with_pool!(pow(lhs: &Tensor, rhs: &Tensor) => analytic::pow_with_pool);
    delegate_with_pool!(pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) => analytic::pow_read_with_pool);
    delegate_with_pool!(expm1(input: &Tensor) => analytic::expm1_with_pool);
    delegate_with_pool!(expm1_read(input: TensorRead<'_>) => analytic::expm1_read_with_pool);
    delegate_with_pool!(log1p(input: &Tensor) => analytic::log1p_with_pool);
    delegate_with_pool!(log1p_read(input: TensorRead<'_>) => analytic::log1p_read_with_pool);
}

impl TensorStructural for CpuExecSession<'_> {
    // Structural
    delegate_with_pool!(transpose(input: &Tensor, perm: &[usize]) => structural::transpose_with_pool);
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        self.run_native(|buffers| {
            if let Some(input) = input.as_tensor() {
                return structural::transpose_with_pool(buffers, input, perm);
            }
            let input = materialize_tensor_read("transpose", input)?;
            structural::transpose_with_pool(buffers, &input, perm)
        })
    }

    delegate!(reshape(input: &Tensor, shape: &[usize]) => structural::reshape(input, shape));
    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        self.run_native(|_| {
            if let Some(input) = input.as_tensor() {
                return structural::reshape(input, shape);
            }
            let input = materialize_tensor_read("reshape", input)?;
            structural::reshape(&input, shape)
        })
    }

    delegate_with_pool!(broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) => structural::broadcast_in_dim_with_pool);
    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.run_native(|buffers| {
            if let Some(input) = input.as_tensor() {
                return structural::broadcast_in_dim_with_pool(buffers, input, shape, dims);
            }
            let input = materialize_tensor_read("broadcast_in_dim", input)?;
            structural::broadcast_in_dim_with_pool(buffers, &input, shape, dims)
        })
    }

    delegate_with_pool!(cast(input: &Tensor, to: crate::DType) => structural::cast_with_pool);
    delegate_with_pool!(extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::extract_diagonal_with_pool);
    delegate_with_pool!(embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::embed_diagonal_with_pool);
    delegate_with_pool!(tril(input: &Tensor, k: i64) => structural::tril_with_pool);
    delegate_with_pool!(triu(input: &Tensor, k: i64) => structural::triu_with_pool);
}

impl TensorReduction for CpuExecSession<'_> {
    // Reduction
    delegate!(reduce_sum(input: &Tensor, axes: &[usize]) => reduction::reduce_sum(input, axes));

    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native(|_| reduction::reduce_sum_read(input, axes))
    }

    delegate!(reduce_prod(input: &Tensor, axes: &[usize]) => reduction::reduce_prod(input, axes));

    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native(|_| reduction::reduce_prod_read(input, axes))
    }

    delegate!(reduce_max(input: &Tensor, axes: &[usize]) => reduction::reduce_max(input, axes));

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native(|_| reduction::reduce_max_read(input, axes))
    }

    delegate!(reduce_min(input: &Tensor, axes: &[usize]) => reduction::reduce_min(input, axes));

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native(|_| reduction::reduce_min_read(input, axes))
    }
}

impl CpuExecSession<'_> {
    fn with_base_dot_general_provider<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let saved = self.dot_general_provider;
        self.dot_general_provider = DotGeneralProvider::Base;
        let result = f(self);
        self.dot_general_provider = saved;
        result
    }

    #[cfg(feature = "cpu-tblis-provider")]
    fn tblis_not_applicable<T>(
        &self,
        op: &'static str,
        value: Option<T>,
    ) -> crate::Result<Option<T>> {
        if value.is_some() || self.dot_general_provider != DotGeneralProvider::TblisRequired {
            return Ok(value);
        }
        Err(super::backend::tblis_required_not_applicable(op))
    }

    #[cfg(not(feature = "cpu-tblis-provider"))]
    fn tblis_unavailable_for_required<T>(&self, op: &'static str) -> crate::Result<Option<T>> {
        if self.dot_general_provider == DotGeneralProvider::TblisRequired {
            Err(super::backend::tblis_required_unavailable(op))
        } else {
            Ok(None)
        }
    }
}

impl TensorDot for CpuExecSession<'_> {
    // GEMM — dtype dispatch, pool + ctx
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general_cached(None, lhs, rhs, config)
    }

    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match self.dot_general_provider {
            DotGeneralProvider::Base => {}
            DotGeneralProvider::TblisIfAvailable | DotGeneralProvider::TblisRequired => {
                #[cfg(feature = "cpu-tblis-provider")]
                {
                    let direct = gemm::dot_general_tblis_read_cached(
                        self.buffers,
                        lhs.clone(),
                        rhs.clone(),
                        config,
                    )?;
                    if let Some(result) = self.tblis_not_applicable("dot_general", direct)? {
                        return Ok(result);
                    }
                }
                #[cfg(not(feature = "cpu-tblis-provider"))]
                {
                    if let Some(result) =
                        self.tblis_unavailable_for_required::<Tensor>("dot_general")?
                    {
                        return Ok(result);
                    }
                }
            }
        }

        let direct = match self.kind {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    gemm::dot_general_faer_read_cached(
                        self.buffers,
                        self.gemm_analysis_cache,
                        None,
                        self.ctx,
                        lhs.clone(),
                        rhs.clone(),
                        config,
                    )?
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    return Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "dot_general",
                    ));
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    gemm::dot_general_blas_read_cached(
                        self.buffers,
                        self.gemm_analysis_cache,
                        None,
                        lhs.clone(),
                        rhs.clone(),
                        config,
                    )?
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    return Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "dot_general",
                    ));
                }
            }
        };
        if let Some(result) = direct {
            return Ok(result);
        }

        let lhs = materialize_tensor_read("dot_general", lhs)?;
        let rhs = materialize_tensor_read("dot_general", rhs)?;
        self.with_base_dot_general_provider(|this| {
            this.dot_general_cached(None, &lhs, &rhs, config)
        })
    }

    fn dot_general_read_into(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        let accumulation = DotGeneralAccumulation::overwrite(lhs.dtype())?;
        self.dot_general_read_into_accum(lhs, rhs, config, accumulation, out)
    }

    fn dot_general_read_into_accum(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        SessionCachedDot::dot_general_read_into_accum_cached(
            self,
            None,
            lhs,
            rhs,
            config,
            accumulation,
            out,
        )
    }

    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_with_conj_cached(None, lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

impl SessionCachedDot for CpuExecSession<'_> {
    fn dot_general_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match self.dot_general_provider {
            DotGeneralProvider::Base => {}
            DotGeneralProvider::TblisIfAvailable | DotGeneralProvider::TblisRequired => {
                #[cfg(feature = "cpu-tblis-provider")]
                {
                    let direct = match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            gemm::dot_general_tblis_cached(self.buffers, a, b, config)
                                .map(|result| result.map(Tensor::F32))
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            gemm::dot_general_tblis_cached(self.buffers, a, b, config)
                                .map(|result| result.map(Tensor::F64))
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            gemm::dot_general_tblis_cached(self.buffers, a, b, config)
                                .map(|result| result.map(Tensor::C32))
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            gemm::dot_general_tblis_cached(self.buffers, a, b, config)
                                .map(|result| result.map(Tensor::C64))
                        }
                        _ if lhs.dtype() == rhs.dtype() => Ok(None),
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    }?;
                    if let Some(result) = self.tblis_not_applicable("dot_general", direct)? {
                        return Ok(result);
                    }
                }
                #[cfg(not(feature = "cpu-tblis-provider"))]
                {
                    if let Some(result) =
                        self.tblis_unavailable_for_required::<Tensor>("dot_general")?
                    {
                        return Ok(result);
                    }
                }
            }
        }

        match self.kind {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_faer_cached(
                            self.buffers,
                            self.gemm_analysis_cache,
                            cache_slot,
                            self.ctx,
                            a,
                            b,
                            config,
                        )
                        .map(Tensor::F32),
                        (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_faer_cached(
                            self.buffers,
                            self.gemm_analysis_cache,
                            cache_slot,
                            self.ctx,
                            a,
                            b,
                            config,
                        )
                        .map(Tensor::F64),
                        (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_faer_cached(
                            self.buffers,
                            self.gemm_analysis_cache,
                            cache_slot,
                            self.ctx,
                            a,
                            b,
                            config,
                        )
                        .map(Tensor::C32),
                        (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_faer_cached(
                            self.buffers,
                            self.gemm_analysis_cache,
                            cache_slot,
                            self.ctx,
                            a,
                            b,
                            config,
                        )
                        .map(Tensor::C64),
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    }
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "dot_general",
                    ))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_blas_cached(
                            self.buffers,
                            self.gemm_analysis_cache,
                            cache_slot,
                            a,
                            b,
                            config,
                        )
                        .map(Tensor::F32),
                        (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_blas_cached(
                            self.buffers,
                            self.gemm_analysis_cache,
                            cache_slot,
                            a,
                            b,
                            config,
                        )
                        .map(Tensor::F64),
                        (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_blas_cached(
                            self.buffers,
                            self.gemm_analysis_cache,
                            cache_slot,
                            a,
                            b,
                            config,
                        )
                        .map(Tensor::C32),
                        (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_blas_cached(
                            self.buffers,
                            self.gemm_analysis_cache,
                            cache_slot,
                            a,
                            b,
                            config,
                        )
                        .map(Tensor::C64),
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    }
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "dot_general",
                    ))
                }
            }
        }
    }

    fn dot_general_with_conj_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        match self.dot_general_provider {
            DotGeneralProvider::Base => {}
            DotGeneralProvider::TblisIfAvailable | DotGeneralProvider::TblisRequired => {
                #[cfg(feature = "cpu-tblis-provider")]
                {
                    let direct = match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            gemm::dot_general_tblis_with_conj_cached(
                                self.buffers,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(|result| result.map(Tensor::F32))
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            gemm::dot_general_tblis_with_conj_cached(
                                self.buffers,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(|result| result.map(Tensor::F64))
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            gemm::dot_general_tblis_with_conj_cached(
                                self.buffers,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(|result| result.map(Tensor::C32))
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            gemm::dot_general_tblis_with_conj_cached(
                                self.buffers,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(|result| result.map(Tensor::C64))
                        }
                        _ if lhs.dtype() == rhs.dtype() => Ok(None),
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    }?;
                    if let Some(result) = self.tblis_not_applicable("dot_general", direct)? {
                        return Ok(result);
                    }
                }
                #[cfg(not(feature = "cpu-tblis-provider"))]
                {
                    if let Some(result) =
                        self.tblis_unavailable_for_required::<Tensor>("dot_general")?
                    {
                        return Ok(result);
                    }
                }
            }
        }
        match self.kind {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            gemm::dot_general_faer_with_conj_cached(
                                self.buffers,
                                self.gemm_analysis_cache,
                                cache_slot,
                                self.ctx,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            gemm::dot_general_faer_with_conj_cached(
                                self.buffers,
                                self.gemm_analysis_cache,
                                cache_slot,
                                self.ctx,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            gemm::dot_general_faer_with_conj_cached(
                                self.buffers,
                                self.gemm_analysis_cache,
                                cache_slot,
                                self.ctx,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            gemm::dot_general_faer_with_conj_cached(
                                self.buffers,
                                self.gemm_analysis_cache,
                                cache_slot,
                                self.ctx,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(Tensor::C64)
                        }
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    }
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "dot_general",
                    ))
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    match (lhs, rhs) {
                        (Tensor::F32(a), Tensor::F32(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                self.buffers,
                                self.gemm_analysis_cache,
                                cache_slot,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(Tensor::F32)
                        }
                        (Tensor::F64(a), Tensor::F64(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                self.buffers,
                                self.gemm_analysis_cache,
                                cache_slot,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(Tensor::F64)
                        }
                        (Tensor::C32(a), Tensor::C32(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                self.buffers,
                                self.gemm_analysis_cache,
                                cache_slot,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(Tensor::C32)
                        }
                        (Tensor::C64(a), Tensor::C64(b)) => {
                            gemm::dot_general_blas_with_conj_cached(
                                self.buffers,
                                self.gemm_analysis_cache,
                                cache_slot,
                                a,
                                b,
                                config,
                                lhs_conj,
                                rhs_conj,
                            )
                            .map(Tensor::C64)
                        }
                        _ => Err(crate::Error::DTypeMismatch {
                            op: "dot_general",
                            lhs: lhs.dtype(),
                            rhs: rhs.dtype(),
                        }),
                    }
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "dot_general",
                    ))
                }
            }
        }
    }

    fn dot_general_read_into_accum_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
        accumulation: DotGeneralAccumulation,
        mut out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        validate_dot_general_accumulation(&lhs, &rhs, config, accumulation, &out, "dot_general")?;
        match self.dot_general_provider {
            DotGeneralProvider::Base => {}
            DotGeneralProvider::TblisIfAvailable | DotGeneralProvider::TblisRequired => {
                #[cfg(feature = "cpu-tblis-provider")]
                {
                    let direct = gemm::dot_general_tblis_read_into_accum_cached(
                        lhs.clone(),
                        rhs.clone(),
                        config,
                        accumulation,
                        &mut out,
                    )?;
                    if direct {
                        return Ok(());
                    }
                    self.tblis_not_applicable::<()>("dot_general", None)?;
                }
                #[cfg(not(feature = "cpu-tblis-provider"))]
                {
                    self.tblis_unavailable_for_required::<()>("dot_general")?;
                }
            }
        }
        let direct = match self.kind {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    gemm::dot_general_faer_read_into_accum_cached(
                        self.gemm_analysis_cache,
                        cache_slot,
                        self.ctx,
                        lhs.clone(),
                        rhs.clone(),
                        config,
                        accumulation,
                        &mut out,
                    )?
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    return Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "dot_general",
                    ));
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    gemm::dot_general_blas_read_into_accum_cached(
                        self.buffers,
                        self.gemm_analysis_cache,
                        cache_slot,
                        lhs.clone(),
                        rhs.clone(),
                        config,
                        accumulation,
                        &mut out,
                    )?
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    return Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "dot_general",
                    ));
                }
            }
        };
        if direct {
            return Ok(());
        }

        self.with_base_dot_general_provider(|this| {
            dot_general_accum_via_temp(this, lhs, rhs, config, accumulation, out)
        })
    }

    fn grouped_gemm_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &GroupedGemmConfig<'_>,
        mut out: TensorWrite<'_>,
    ) -> crate::Result<()> {
        validate_grouped_gemm(&lhs, &rhs, &out, config, "grouped_gemm")?;
        let direct = match self.kind {
            CpuBackendKind::Faer => {
                #[cfg(feature = "cpu-faer")]
                {
                    gemm::grouped_gemm_faer_cached(
                        self.ctx,
                        lhs.clone(),
                        rhs.clone(),
                        config,
                        &mut out,
                    )?
                }
                #[cfg(not(feature = "cpu-faer"))]
                {
                    return Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "grouped_gemm",
                    ));
                }
            }
            CpuBackendKind::Blas => {
                #[cfg(feature = "cpu-blas")]
                {
                    gemm::grouped_gemm_blas_cached(lhs.clone(), rhs.clone(), config, &mut out)?
                }
                #[cfg(not(feature = "cpu-blas"))]
                {
                    return Err(super::backend::unavailable_cpu_backend_kind(
                        self.kind,
                        "grouped_gemm",
                    ));
                }
            }
        };
        if direct {
            return Ok(());
        }

        grouped_gemm_via_sequential(self, lhs, rhs, config, out)
    }
}

impl TensorIndexing for CpuExecSession<'_> {
    // Indexing
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        self.run_native(|buffers| {
            indexing::gather_with_pool(buffers, operand, start_indices, config)
        })
    }
    delegate_with_pool!(scatter(operand: &Tensor, indices: &Tensor, updates: &Tensor, config: &ScatterConfig) => indexing::scatter_with_pool);
    delegate_with_pool!(slice(input: &Tensor, config: &SliceConfig) => indexing::try_slice_with_pool);
    delegate_with_pool!(dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) => indexing::dynamic_slice_with_pool);
    delegate_with_pool!(dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) => indexing::dynamic_update_slice_with_pool);
    delegate_with_pool!(pad(input: &Tensor, config: &PadConfig) => indexing::try_pad_with_pool);
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.run_native(|buffers| indexing::try_concatenate_with_pool(buffers, inputs, axis))
    }
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.run_native(|buffers| indexing::reverse_with_pool(buffers, input, axes))
    }
}

impl TensorBuffer for CpuExecSession<'_> {
    fn reclaim_buffer(&mut self, tensor: Tensor) {
        match tensor {
            Tensor::F32(t) => reclaim_typed(self.buffers, t),
            Tensor::F64(t) => reclaim_typed(self.buffers, t),
            Tensor::I32(t) => reclaim_typed(self.buffers, t),
            Tensor::I64(t) => reclaim_typed(self.buffers, t),
            Tensor::Bool(t) => reclaim_typed(self.buffers, t),
            Tensor::C32(t) => reclaim_typed(self.buffers, t),
            Tensor::C64(t) => reclaim_typed(self.buffers, t),
        }
    }
}

impl TensorFusion for CpuExecSession<'_> {
    fn execute_elementwise_fusion(
        &mut self,
        inputs: &[&Tensor],
        plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        self.run_native(|buffers| elementwise::elementwise_fusion_with_pool(buffers, inputs, plan))
    }

    fn execute_broadcast_multiply(
        &mut self,
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<Tensor>> {
        self.run_native(|buffers| {
            elementwise::broadcast_multiply_read_with_pool(
                buffers, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
        })
    }

    fn execute_broadcast_multiply_value(
        &mut self,
        lhs: TensorRead<'_>,
        lhs_shape: &[usize],
        lhs_dims: &[usize],
        rhs: TensorRead<'_>,
        rhs_shape: &[usize],
        rhs_dims: &[usize],
    ) -> crate::Result<Option<TensorValue>> {
        self.run_native(|buffers| {
            elementwise::broadcast_multiply_value_with_pool(
                buffers, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims,
            )
        })
    }
}

#[cfg(all(test, feature = "cpu-blas"))]
mod tests {
    use super::*;
    use crate::{process_cpu_affinity, CpuSet};

    #[test]
    fn blas_session_native_scope_enters_the_pinned_rayon_engine() {
        let allowed = process_cpu_affinity().expect("Linux test requires process affinity");
        let cpus = CpuSet::new([allowed.as_slice()[0]]).unwrap();
        let context = CpuContext::with_pinned_cpus(cpus, 1).unwrap();
        let mut buffers = BufferPool::new();
        let mut gemm_analysis_cache = gemm::GemmAnalysisCache::default();
        let mut session = CpuExecSession {
            ctx: &context,
            buffers: &mut buffers,
            gemm_analysis_cache: &mut gemm_analysis_cache,
            kind: CpuBackendKind::Blas,
            dot_general_provider: DotGeneralProvider::Base,
        };

        assert!(rayon::current_thread_index().is_none());
        assert_eq!(
            session.run_native(|_| rayon::current_thread_index()),
            Some(0)
        );
        assert!(rayon::current_thread_index().is_none());
    }
}
