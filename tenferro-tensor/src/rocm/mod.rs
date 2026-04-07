use tenferro_algebra::Semiring;

use crate::backend::{SemiringBackend, TensorBackend};
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::{Tensor, TypedTensor};

#[derive(Default)]
pub struct RocmBackend;

impl RocmBackend {
    pub fn new() -> Self {
        Self
    }
}

macro_rules! impl_stub_backend {
    ($ty:ty, $label:literal) => {
        impl TensorBackend for $ty {
            fn add(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
                todo!(concat!($label, " add"))
            }
            fn mul(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
                todo!(concat!($label, " mul"))
            }
            fn neg(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " neg"))
            }
            fn conj(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " conj"))
            }
            fn div(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
                todo!(concat!($label, " div"))
            }
            fn abs(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " abs"))
            }
            fn sign(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " sign"))
            }
            fn maximum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
                todo!(concat!($label, " maximum"))
            }
            fn minimum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
                todo!(concat!($label, " minimum"))
            }
            fn compare(&mut self, _lhs: &Tensor, _rhs: &Tensor, _dir: &CompareDir) -> Tensor {
                todo!(concat!($label, " compare"))
            }
            fn select(&mut self, _pred: &Tensor, _on_true: &Tensor, _on_false: &Tensor) -> Tensor {
                todo!(concat!($label, " select"))
            }
            fn clamp(&mut self, _input: &Tensor, _lower: &Tensor, _upper: &Tensor) -> Tensor {
                todo!(concat!($label, " clamp"))
            }
            fn exp(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " exp"))
            }
            fn log(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " log"))
            }
            fn sin(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " sin"))
            }
            fn cos(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " cos"))
            }
            fn tanh(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " tanh"))
            }
            fn sqrt(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " sqrt"))
            }
            fn rsqrt(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " rsqrt"))
            }
            fn pow(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
                todo!(concat!($label, " pow"))
            }
            fn expm1(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " expm1"))
            }
            fn log1p(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " log1p"))
            }
            fn transpose(&mut self, _input: &Tensor, _perm: &[usize]) -> Tensor {
                todo!(concat!($label, " transpose"))
            }
            fn reshape(&mut self, _input: &Tensor, _shape: &[usize]) -> Tensor {
                todo!(concat!($label, " reshape"))
            }
            fn broadcast_in_dim(
                &mut self,
                _input: &Tensor,
                _shape: &[usize],
                _dims: &[usize],
            ) -> Tensor {
                todo!(concat!($label, " broadcast_in_dim"))
            }
            fn extract_diagonal(
                &mut self,
                _input: &Tensor,
                _axis_a: usize,
                _axis_b: usize,
            ) -> Tensor {
                todo!(concat!($label, " extract_diagonal"))
            }
            fn embed_diagonal(
                &mut self,
                _input: &Tensor,
                _axis_a: usize,
                _axis_b: usize,
            ) -> Tensor {
                todo!(concat!($label, " embed_diagonal"))
            }
            fn tril(&mut self, _input: &Tensor, _k: i64) -> Tensor {
                todo!(concat!($label, " tril"))
            }
            fn triu(&mut self, _input: &Tensor, _k: i64) -> Tensor {
                todo!(concat!($label, " triu"))
            }
            fn reduce_sum(&mut self, _input: &Tensor, _axes: &[usize]) -> Tensor {
                todo!(concat!($label, " reduce_sum"))
            }
            fn reduce_prod(&mut self, _input: &Tensor, _axes: &[usize]) -> Tensor {
                todo!(concat!($label, " reduce_prod"))
            }
            fn reduce_max(&mut self, _input: &Tensor, _axes: &[usize]) -> Tensor {
                todo!(concat!($label, " reduce_max"))
            }
            fn reduce_min(&mut self, _input: &Tensor, _axes: &[usize]) -> Tensor {
                todo!(concat!($label, " reduce_min"))
            }
            fn dot_general(
                &mut self,
                _lhs: &Tensor,
                _rhs: &Tensor,
                _config: &DotGeneralConfig,
            ) -> Tensor {
                todo!(concat!($label, " dot_general"))
            }
            fn gather(
                &mut self,
                _operand: &Tensor,
                _start_indices: &Tensor,
                _config: &GatherConfig,
            ) -> Tensor {
                todo!(concat!($label, " gather"))
            }
            fn scatter(
                &mut self,
                _operand: &Tensor,
                _scatter_indices: &Tensor,
                _updates: &Tensor,
                _config: &ScatterConfig,
            ) -> Tensor {
                todo!(concat!($label, " scatter"))
            }
            fn slice(&mut self, _input: &Tensor, _config: &SliceConfig) -> Tensor {
                todo!(concat!($label, " slice"))
            }
            fn dynamic_slice(
                &mut self,
                _input: &Tensor,
                _starts: &Tensor,
                _slice_sizes: &[usize],
            ) -> Tensor {
                todo!(concat!($label, " dynamic_slice"))
            }
            fn pad(&mut self, _input: &Tensor, _config: &PadConfig) -> Tensor {
                todo!(concat!($label, " pad"))
            }
            fn concatenate(&mut self, _inputs: &[&Tensor], _axis: usize) -> Tensor {
                todo!(concat!($label, " concatenate"))
            }
            fn reverse(&mut self, _input: &Tensor, _axes: &[usize]) -> Tensor {
                todo!(concat!($label, " reverse"))
            }
            fn cholesky(&mut self, _input: &Tensor) -> Tensor {
                todo!(concat!($label, " cholesky"))
            }
            fn triangular_solve(
                &mut self,
                _a: &Tensor,
                _b: &Tensor,
                _left_side: bool,
                _lower: bool,
                _transpose_a: bool,
                _unit_diagonal: bool,
            ) -> Tensor {
                todo!(concat!($label, " triangular_solve"))
            }
            fn lu(&mut self, _input: &Tensor) -> Vec<Tensor> {
                todo!(concat!($label, " lu"))
            }
            fn svd(&mut self, _input: &Tensor) -> Vec<Tensor> {
                todo!(concat!($label, " svd"))
            }
            fn qr(&mut self, _input: &Tensor) -> Vec<Tensor> {
                todo!(concat!($label, " qr"))
            }
            fn eigh(&mut self, _input: &Tensor) -> Vec<Tensor> {
                todo!(concat!($label, " eigh"))
            }
            fn eig(&mut self, _input: &Tensor) -> Vec<Tensor> {
                todo!(concat!($label, " eig"))
            }
            fn solve(&mut self, _a: &Tensor, _b: &Tensor) -> Tensor {
                todo!(concat!($label, " solve"))
            }
        }

        impl<Alg: Semiring> SemiringBackend<Alg> for $ty {
            fn batched_gemm(
                &mut self,
                _lhs: &TypedTensor<Alg::Scalar>,
                _rhs: &TypedTensor<Alg::Scalar>,
                _config: &DotGeneralConfig,
            ) -> TypedTensor<Alg::Scalar> {
                todo!(concat!($label, " batched_gemm"))
            }
        }
    };
}

impl_stub_backend!(RocmBackend, "ROCm");
