use crate::backend::BackendSession;
use crate::buffer_pool::BufferPool;
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::{Tensor, TensorRead};

use super::backend::{reclaim_typed, unsupported_dtype};
use super::{analytic, elementwise, gemm, indexing, linalg, reduction, structural, CpuContext};

pub(crate) struct CpuExecSession<'a> {
    #[cfg_attr(feature = "cpu-blas", allow(dead_code))]
    pub(crate) ctx: &'a CpuContext,
    pub(crate) buffers: &'a mut BufferPool,
    pub(crate) gemm_analysis_cache: &'a mut gemm::GemmAnalysisCache,
}

/// Simple delegation: no dtype dispatch, no install.
macro_rules! delegate {
    ($name:ident($($arg:ident : $ty:ty),*) => $body:expr) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> { $body }
    };
}

/// Delegation for operations whose outputs can be allocated from the session pool.
macro_rules! delegate_with_pool {
    ($name:ident($($arg:ident : $ty:ty),*) => $callee:path) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> {
            $callee(self.buffers, $($arg),*)
        }
    };
}

/// Unary linalg returning single Tensor — faer path (returns Result).
#[cfg(feature = "cpu-faer")]
macro_rules! linalg_single {
    ($name:ident) => {
        fn $name(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            match input {
                Tensor::F32(t) => linalg::$name(self.ctx, self.buffers, t).map(Tensor::F32),
                Tensor::F64(t) => linalg::$name(self.ctx, self.buffers, t).map(Tensor::F64),
                Tensor::C32(t) => linalg::$name(self.ctx, self.buffers, t).map(Tensor::C32),
                Tensor::C64(t) => linalg::$name(self.ctx, self.buffers, t).map(Tensor::C64),
                _ => Err(unsupported_dtype(stringify!($name), input.dtype())),
            }
        }
    };
}

/// Unary linalg returning single Tensor — blas path.
#[cfg(feature = "cpu-blas")]
macro_rules! linalg_single {
    ($name:ident) => {
        fn $name(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            match input {
                Tensor::F32(t) => linalg::$name(self.buffers, t).map(Tensor::F32),
                Tensor::F64(t) => linalg::$name(self.buffers, t).map(Tensor::F64),
                Tensor::C32(t) => linalg::$name(self.buffers, t).map(Tensor::C32),
                Tensor::C64(t) => linalg::$name(self.buffers, t).map(Tensor::C64),
                _ => Err(unsupported_dtype(stringify!($name), input.dtype())),
            }
        }
    };
}

/// Unary linalg returning Vec<Tensor> — faer path.
#[cfg(feature = "cpu-faer")]
macro_rules! linalg_multi {
    ($name:ident) => {
        fn $name(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
            match input {
                Tensor::F32(t) => linalg::$name(self.ctx, self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                Tensor::F64(t) => linalg::$name(self.ctx, self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                Tensor::C32(t) => linalg::$name(self.ctx, self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                Tensor::C64(t) => linalg::$name(self.ctx, self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                _ => Err(unsupported_dtype(stringify!($name), input.dtype())),
            }
        }
    };
}

/// Unary linalg returning Vec<Tensor> and internal Result — faer path.
#[cfg(feature = "cpu-faer")]
macro_rules! linalg_multi_result {
    ($name:ident) => {
        fn $name(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
            match input {
                Tensor::F32(t) => linalg::$name(self.ctx, self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                Tensor::F64(t) => linalg::$name(self.ctx, self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                Tensor::C32(t) => linalg::$name(self.ctx, self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                Tensor::C64(t) => linalg::$name(self.ctx, self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                _ => Err(unsupported_dtype(stringify!($name), input.dtype())),
            }
        }
    };
}

/// Unary linalg returning Vec<Tensor> — blas path.
#[cfg(feature = "cpu-blas")]
macro_rules! linalg_multi {
    ($name:ident) => {
        fn $name(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
            match input {
                Tensor::F32(t) => linalg::$name(self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
                Tensor::F64(t) => linalg::$name(self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
                Tensor::C32(t) => linalg::$name(self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
                Tensor::C64(t) => linalg::$name(self.buffers, t)
                    .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
                _ => Err(unsupported_dtype(stringify!($name), input.dtype())),
            }
        }
    };
}

impl BackendSession for CpuExecSession<'_> {
    // Elementwise — direct delegation, no install
    delegate_with_pool!(add(lhs: &Tensor, rhs: &Tensor) => elementwise::add_with_pool);
    delegate_with_pool!(mul(lhs: &Tensor, rhs: &Tensor) => elementwise::mul_with_pool);
    delegate_with_pool!(neg(input: &Tensor) => elementwise::neg_with_pool);
    delegate_with_pool!(conj(input: &Tensor) => elementwise::conj_with_pool);
    delegate_with_pool!(div(lhs: &Tensor, rhs: &Tensor) => elementwise::div_with_pool);
    delegate_with_pool!(abs(input: &Tensor) => elementwise::abs_with_pool);
    delegate_with_pool!(sign(input: &Tensor) => elementwise::sign_with_pool);
    delegate_with_pool!(maximum(lhs: &Tensor, rhs: &Tensor) => elementwise::maximum_with_pool);
    delegate_with_pool!(minimum(lhs: &Tensor, rhs: &Tensor) => elementwise::minimum_with_pool);
    delegate_with_pool!(compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) => elementwise::compare_with_pool);
    delegate_with_pool!(select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) => elementwise::select_with_pool);
    delegate_with_pool!(clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) => elementwise::clamp_with_pool);

    // Analytic
    delegate_with_pool!(exp(input: &Tensor) => analytic::exp_with_pool);
    delegate_with_pool!(log(input: &Tensor) => analytic::log_with_pool);
    delegate_with_pool!(sin(input: &Tensor) => analytic::sin_with_pool);
    delegate_with_pool!(cos(input: &Tensor) => analytic::cos_with_pool);
    delegate_with_pool!(tanh(input: &Tensor) => analytic::tanh_with_pool);
    delegate_with_pool!(sqrt(input: &Tensor) => analytic::sqrt_with_pool);
    delegate_with_pool!(rsqrt(input: &Tensor) => analytic::rsqrt_with_pool);
    delegate_with_pool!(pow(lhs: &Tensor, rhs: &Tensor) => analytic::pow_with_pool);
    delegate_with_pool!(expm1(input: &Tensor) => analytic::expm1_with_pool);
    delegate_with_pool!(log1p(input: &Tensor) => analytic::log1p_with_pool);

    // Structural
    delegate_with_pool!(transpose(input: &Tensor, perm: &[usize]) => structural::transpose_with_pool);
    delegate!(reshape(input: &Tensor, shape: &[usize]) => structural::reshape(input, shape));
    delegate_with_pool!(broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) => structural::broadcast_in_dim_with_pool);
    delegate_with_pool!(convert(input: &Tensor, to: crate::DType) => structural::convert_with_pool);
    delegate_with_pool!(extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::extract_diagonal_with_pool);
    delegate_with_pool!(embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::embed_diagonal_with_pool);
    delegate_with_pool!(tril(input: &Tensor, k: i64) => structural::tril_with_pool);
    delegate_with_pool!(triu(input: &Tensor, k: i64) => structural::triu_with_pool);

    // Reduction
    delegate!(reduce_sum(input: &Tensor, axes: &[usize]) => reduction::reduce_sum(input, axes));
    delegate!(reduce_prod(input: &Tensor, axes: &[usize]) => reduction::reduce_prod(input, axes));
    delegate!(reduce_max(input: &Tensor, axes: &[usize]) => reduction::reduce_max(input, axes));
    delegate!(reduce_min(input: &Tensor, axes: &[usize]) => reduction::reduce_min(input, axes));

    // GEMM — dtype dispatch, pool + ctx
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general_cached(None, lhs, rhs, config)
    }

    fn dot_general_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                self.ctx,
                a,
                b,
                config,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                self.ctx,
                a,
                b,
                config,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                self.ctx,
                a,
                b,
                config,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                self.ctx,
                a,
                b,
                config,
            )
            .map(Tensor::C64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                a,
                b,
                config,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                a,
                b,
                config,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                a,
                b,
                config,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_cached(
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

    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        #[cfg(feature = "cpu-faer")]
        if let Some(result) = gemm::dot_general_read_cached(
            self.buffers,
            self.gemm_analysis_cache,
            None,
            self.ctx,
            lhs,
            rhs,
            config,
        )? {
            return Ok(result);
        }
        #[cfg(feature = "cpu-blas")]
        if let Some(result) = gemm::dot_general_read_cached(
            self.buffers,
            self.gemm_analysis_cache,
            None,
            lhs,
            rhs,
            config,
        )? {
            return Ok(result);
        }

        let lhs = lhs.to_tensor();
        let rhs = rhs.to_tensor();
        self.dot_general_cached(None, &lhs, &rhs, config)
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

    fn dot_general_with_conj_cached(
        &mut self,
        cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        match (lhs, rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_with_conj_cached(
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
            .map(Tensor::F32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_with_conj_cached(
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
            .map(Tensor::F64),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_with_conj_cached(
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
            .map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_with_conj_cached(
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
            .map(Tensor::C64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => gemm::dot_general_with_conj_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                a,
                b,
                config,
                lhs_conj,
                rhs_conj,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => gemm::dot_general_with_conj_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                a,
                b,
                config,
                lhs_conj,
                rhs_conj,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => gemm::dot_general_with_conj_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                a,
                b,
                config,
                lhs_conj,
                rhs_conj,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => gemm::dot_general_with_conj_cached(
                self.buffers,
                self.gemm_analysis_cache,
                cache_slot,
                a,
                b,
                config,
                lhs_conj,
                rhs_conj,
            )
            .map(Tensor::C64),
            _ => Err(crate::Error::DTypeMismatch {
                op: "dot_general",
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            }),
        }
    }

    // Indexing
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        indexing::gather_with_pool(self.buffers, operand, start_indices, config)
    }
    delegate_with_pool!(scatter(operand: &Tensor, indices: &Tensor, updates: &Tensor, config: &ScatterConfig) => indexing::scatter_with_pool);
    delegate_with_pool!(slice(input: &Tensor, config: &SliceConfig) => indexing::try_slice_with_pool);
    delegate_with_pool!(dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) => indexing::dynamic_slice_with_pool);
    delegate_with_pool!(dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) => indexing::dynamic_update_slice_with_pool);
    delegate_with_pool!(pad(input: &Tensor, config: &PadConfig) => indexing::try_pad_with_pool);
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        indexing::try_concatenate_with_pool(self.buffers, inputs, axis)
    }
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        indexing::reverse_with_pool(self.buffers, input, axes)
    }

    // Linalg — macro-generated dtype dispatch
    linalg_single!(cholesky);
    linalg_multi!(lu);
    linalg_multi!(full_piv_lu);
    #[cfg(feature = "cpu-faer")]
    linalg_multi_result!(svd);
    #[cfg(feature = "cpu-blas")]
    linalg_multi!(svd);
    linalg_multi!(qr);
    #[cfg(feature = "cpu-faer")]
    linalg_multi_result!(eigh);
    #[cfg(feature = "cpu-blas")]
    linalg_multi!(eigh);

    fn eig(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        if !matches!(
            input,
            Tensor::F32(_) | Tensor::F64(_) | Tensor::C32(_) | Tensor::C64(_)
        ) {
            return Err(unsupported_dtype("eig", input.dtype()));
        }
        #[cfg(feature = "cpu-faer")]
        {
            linalg::eig(self.ctx, self.buffers, input)
        }
        #[cfg(feature = "cpu-blas")]
        {
            linalg::eig(self.buffers, input)
        }
    }

    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> crate::Result<Tensor> {
        match (a, b) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => linalg::triangular_solve(
                self.ctx,
                self.buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => linalg::triangular_solve(
                self.ctx,
                self.buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => linalg::triangular_solve(
                self.ctx,
                self.buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => linalg::triangular_solve(
                self.ctx,
                self.buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => linalg::triangular_solve(
                self.buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => linalg::triangular_solve(
                self.buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => linalg::triangular_solve(
                self.buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => linalg::triangular_solve(
                self.buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C64),
            _ => {
                if a.dtype() != b.dtype() {
                    Err(crate::Error::DTypeMismatch {
                        op: "triangular_solve",
                        lhs: a.dtype(),
                        rhs: b.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("triangular_solve", a.dtype()))
                }
            }
        }
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> crate::Result<Tensor> {
        match (a, b) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::full_piv_lu_solve(self.ctx, self.buffers, a, b, transpose_a)
                    .map(Tensor::F32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::full_piv_lu_solve(self.ctx, self.buffers, a, b, transpose_a)
                    .map(Tensor::F64)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::full_piv_lu_solve(self.ctx, self.buffers, a, b, transpose_a)
                    .map(Tensor::C32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::full_piv_lu_solve(self.ctx, self.buffers, a, b, transpose_a)
                    .map(Tensor::C64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::full_piv_lu_solve(self.buffers, a, b, transpose_a).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::full_piv_lu_solve(self.buffers, a, b, transpose_a).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::full_piv_lu_solve(self.buffers, a, b, transpose_a).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::full_piv_lu_solve(self.buffers, a, b, transpose_a).map(Tensor::C64)
            }
            _ => {
                if a.dtype() != b.dtype() {
                    Err(crate::Error::DTypeMismatch {
                        op: "full_piv_lu_solve",
                        lhs: a.dtype(),
                        rhs: b.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("full_piv_lu_solve", a.dtype()))
                }
            }
        }
    }

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
