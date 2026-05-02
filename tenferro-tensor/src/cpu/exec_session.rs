use crate::backend::TensorExec;
use crate::buffer_pool::BufferPool;
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::Tensor;

use super::backend::{catch_backend_panic, reclaim_typed, unsupported_dtype};
use super::{analytic, elementwise, gemm, indexing, linalg, reduction, structural, CpuContext};

pub(crate) struct CpuExecSession<'a> {
    #[cfg_attr(feature = "cpu-blas", allow(dead_code))]
    pub(crate) ctx: &'a CpuContext,
    pub(crate) buffers: &'a mut BufferPool,
}

/// Simple delegation: no dtype dispatch, no install.
macro_rules! delegate {
    ($name:ident($($arg:ident : $ty:ty),*) => $body:expr) => {
        fn $name(&mut self, $($arg: $ty),*) -> crate::Result<Tensor> { $body }
    };
}

/// Unary linalg returning single Tensor — faer path (returns Result).
#[cfg(feature = "cpu-faer")]
macro_rules! linalg_single {
    ($name:ident) => {
        fn $name(&mut self, input: &Tensor) -> crate::Result<Tensor> {
            match input {
                Tensor::F64(t) => catch_backend_panic(stringify!($name), || {
                    linalg::$name(self.ctx, self.buffers, t)
                })
                .and_then(|r| r)
                .map(Tensor::F64),
                Tensor::C64(t) => catch_backend_panic(stringify!($name), || {
                    linalg::$name(self.ctx, self.buffers, t)
                })
                .and_then(|r| r)
                .map(Tensor::C64),
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
                Tensor::F64(t) => {
                    catch_backend_panic(stringify!($name), || linalg::$name(self.buffers, t))
                        .and_then(|r| r)
                        .map(Tensor::F64)
                }
                Tensor::C64(t) => {
                    catch_backend_panic(stringify!($name), || linalg::$name(self.buffers, t))
                        .and_then(|r| r)
                        .map(Tensor::C64)
                }
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
                Tensor::F64(t) => catch_backend_panic(stringify!($name), || {
                    linalg::$name(self.ctx, self.buffers, t)
                        .into_iter()
                        .map(Tensor::F64)
                        .collect()
                }),
                Tensor::C64(t) => catch_backend_panic(stringify!($name), || {
                    linalg::$name(self.ctx, self.buffers, t)
                        .into_iter()
                        .map(Tensor::C64)
                        .collect()
                }),
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
                Tensor::F64(t) => catch_backend_panic(stringify!($name), || {
                    linalg::$name(self.buffers, t)
                        .into_iter()
                        .map(Tensor::F64)
                        .collect()
                }),
                Tensor::C64(t) => catch_backend_panic(stringify!($name), || {
                    linalg::$name(self.buffers, t)
                        .into_iter()
                        .map(Tensor::C64)
                        .collect()
                }),
                _ => Err(unsupported_dtype(stringify!($name), input.dtype())),
            }
        }
    };
}

impl TensorExec for CpuExecSession<'_> {
    // Elementwise — direct delegation, no install
    delegate!(add(lhs: &Tensor, rhs: &Tensor) => elementwise::add(lhs, rhs));
    delegate!(mul(lhs: &Tensor, rhs: &Tensor) => elementwise::mul(lhs, rhs));
    delegate!(neg(input: &Tensor) => elementwise::neg(input));
    delegate!(conj(input: &Tensor) => elementwise::conj(input));
    delegate!(div(lhs: &Tensor, rhs: &Tensor) => elementwise::div(lhs, rhs));
    delegate!(abs(input: &Tensor) => elementwise::abs(input));
    delegate!(sign(input: &Tensor) => elementwise::sign(input));
    delegate!(maximum(lhs: &Tensor, rhs: &Tensor) => elementwise::maximum(lhs, rhs));
    delegate!(minimum(lhs: &Tensor, rhs: &Tensor) => elementwise::minimum(lhs, rhs));
    delegate!(compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) => elementwise::compare(lhs, rhs, dir));
    delegate!(select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) => elementwise::select(pred, on_true, on_false));
    delegate!(clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) => elementwise::clamp(input, lower, upper));

    // Analytic
    delegate!(exp(input: &Tensor) => analytic::exp(input));
    delegate!(log(input: &Tensor) => analytic::log(input));
    delegate!(sin(input: &Tensor) => analytic::sin(input));
    delegate!(cos(input: &Tensor) => analytic::cos(input));
    delegate!(tanh(input: &Tensor) => analytic::tanh(input));
    delegate!(sqrt(input: &Tensor) => analytic::sqrt(input));
    delegate!(rsqrt(input: &Tensor) => analytic::rsqrt(input));
    delegate!(pow(lhs: &Tensor, rhs: &Tensor) => analytic::pow(lhs, rhs));
    delegate!(expm1(input: &Tensor) => analytic::expm1(input));
    delegate!(log1p(input: &Tensor) => analytic::log1p(input));

    // Structural
    delegate!(transpose(input: &Tensor, perm: &[usize]) => structural::transpose(input, perm));
    delegate!(reshape(input: &Tensor, shape: &[usize]) => structural::reshape(input, shape));
    delegate!(broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) => structural::broadcast_in_dim(input, shape, dims));
    delegate!(convert(input: &Tensor, to: crate::DType) => Ok(structural::convert(input, to)));
    delegate!(extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::extract_diagonal(input, axis_a, axis_b));
    delegate!(embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) => structural::embed_diagonal(input, axis_a, axis_b));
    delegate!(tril(input: &Tensor, k: i64) => structural::tril(input, k));
    delegate!(triu(input: &Tensor, k: i64) => structural::triu(input, k));

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
        match (lhs, rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                gemm::dot_general(self.buffers, self.ctx, a, b, config).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                gemm::dot_general(self.buffers, self.ctx, a, b, config).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                gemm::dot_general(self.buffers, self.ctx, a, b, config).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                gemm::dot_general(self.buffers, self.ctx, a, b, config).map(Tensor::C64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                gemm::dot_general(self.buffers, a, b, config).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                gemm::dot_general(self.buffers, a, b, config).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                gemm::dot_general(self.buffers, a, b, config).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                gemm::dot_general(self.buffers, a, b, config).map(Tensor::C64)
            }
            _ => Err(crate::Error::DTypeMismatch {
                op: "dot_general",
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            }),
        }
    }

    // Indexing
    delegate!(gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) => indexing::gather(operand, start_indices, config));
    delegate!(scatter(operand: &Tensor, indices: &Tensor, updates: &Tensor, config: &ScatterConfig) => indexing::scatter(operand, indices, updates, config));
    delegate!(slice(input: &Tensor, config: &SliceConfig) => indexing::try_slice(input, config));
    delegate!(dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) => indexing::dynamic_slice(input, starts, slice_sizes));
    delegate!(pad(input: &Tensor, config: &PadConfig) => indexing::try_pad(input, config));
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        indexing::try_concatenate(inputs, axis)
    }
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        indexing::reverse(input, axes)
    }

    // Linalg — macro-generated dtype dispatch
    linalg_single!(cholesky);
    linalg_multi!(lu);
    linalg_multi!(full_piv_lu);
    linalg_multi!(svd);
    linalg_multi!(qr);
    linalg_multi!(eigh);

    fn eig(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        if !matches!(input, Tensor::F64(_) | Tensor::C64(_)) {
            return Err(unsupported_dtype("eig", input.dtype()));
        }
        catch_backend_panic("eig", || {
            #[cfg(feature = "cpu-faer")]
            {
                linalg::eig(self.ctx, self.buffers, input)
            }
            #[cfg(feature = "cpu-blas")]
            {
                linalg::eig(self.buffers, input)
            }
        })
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
            (Tensor::F64(a), Tensor::F64(b)) => catch_backend_panic("triangular_solve", || {
                Tensor::F64(linalg::triangular_solve(
                    self.ctx,
                    self.buffers,
                    a,
                    b,
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                ))
            }),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => catch_backend_panic("triangular_solve", || {
                Tensor::C64(linalg::triangular_solve(
                    self.ctx,
                    self.buffers,
                    a,
                    b,
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                ))
            }),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => catch_backend_panic("triangular_solve", || {
                Tensor::F64(linalg::triangular_solve(
                    self.buffers,
                    a,
                    b,
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                ))
            }),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => catch_backend_panic("triangular_solve", || {
                Tensor::C64(linalg::triangular_solve(
                    self.buffers,
                    a,
                    b,
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                ))
            }),
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
            (Tensor::F64(a), Tensor::F64(b)) => catch_backend_panic("full_piv_lu_solve", || {
                linalg::full_piv_lu_solve(self.ctx, self.buffers, a, b, transpose_a)
                    .map(Tensor::F64)
            })
            .and_then(|result| result),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => catch_backend_panic("full_piv_lu_solve", || {
                linalg::full_piv_lu_solve(self.ctx, self.buffers, a, b, transpose_a)
                    .map(Tensor::C64)
            })
            .and_then(|result| result),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => catch_backend_panic("full_piv_lu_solve", || {
                linalg::full_piv_lu_solve(self.buffers, a, b, transpose_a).map(Tensor::F64)
            })
            .and_then(|result| result),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => catch_backend_panic("full_piv_lu_solve", || {
                linalg::full_piv_lu_solve(self.buffers, a, b, transpose_a).map(Tensor::C64)
            })
            .and_then(|result| result),
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
            Tensor::I64(t) => reclaim_typed(self.buffers, t),
            Tensor::C32(t) => reclaim_typed(self.buffers, t),
            Tensor::C64(t) => reclaim_typed(self.buffers, t),
        }
    }
}
