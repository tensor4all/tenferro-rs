use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{
    cpu::CpuBackend, BackendCachedDot, BackendRuntimeCache, BackendSessionHost, CompareDir, DType,
    DotGeneralConfig, Error, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural, TensorView,
    TypedTensor,
};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

#[test]
fn default_svd_view_returns_explicit_backend_boundary_error() {
    struct DefaultOnlyLinalgBackend;

    macro_rules! panic_backend_methods {
        ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
            $(
                fn $name(&mut self, $($arg: $argty),*) -> $ret {
                    $(let _ = &$arg;)*
                    panic!(concat!(stringify!($name), " should not be called by this test"))
                }
            )+
        };
    }

    impl BackendRuntimeCache for DefaultOnlyLinalgBackend {
        type RuntimeCache = ();
    }

    impl TensorElementwise for DefaultOnlyLinalgBackend {
        panic_backend_methods! {
            add(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
            mul(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
            neg(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            conj(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            div(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
            abs(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            sign(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            maximum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
            minimum(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
            compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> tenferro_tensor::Result<Tensor>;
            select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> tenferro_tensor::Result<Tensor>;
            clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> tenferro_tensor::Result<Tensor>;
        }
    }

    impl TensorAnalytic for DefaultOnlyLinalgBackend {
        panic_backend_methods! {
            exp(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            log(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            sin(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            cos(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            tanh(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            sqrt(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            rsqrt(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            pow(lhs: &Tensor, rhs: &Tensor) -> tenferro_tensor::Result<Tensor>;
            expm1(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            log1p(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
        }
    }

    impl TensorStructural for DefaultOnlyLinalgBackend {
        panic_backend_methods! {
            transpose(input: &Tensor, perm: &[usize]) -> tenferro_tensor::Result<Tensor>;
            reshape(input: &Tensor, shape: &[usize]) -> tenferro_tensor::Result<Tensor>;
            broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> tenferro_tensor::Result<Tensor>;
            convert(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
            extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
            embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
            tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
            triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
        }
    }

    impl TensorReduction for DefaultOnlyLinalgBackend {
        panic_backend_methods! {
            reduce_sum(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
            reduce_prod(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
            reduce_max(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
            reduce_min(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        }
    }

    impl TensorIndexing for DefaultOnlyLinalgBackend {
        panic_backend_methods! {
            gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> tenferro_tensor::Result<Tensor>;
            scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> tenferro_tensor::Result<Tensor>;
            slice(input: &Tensor, config: &SliceConfig) -> tenferro_tensor::Result<Tensor>;
            dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> tenferro_tensor::Result<Tensor>;
            dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> tenferro_tensor::Result<Tensor>;
            pad(input: &Tensor, config: &PadConfig) -> tenferro_tensor::Result<Tensor>;
            concatenate(inputs: &[&Tensor], axis: usize) -> tenferro_tensor::Result<Tensor>;
            reverse(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        }
    }

    impl TensorDot for DefaultOnlyLinalgBackend {
        panic_backend_methods! {
            dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> tenferro_tensor::Result<Tensor>;
        }
    }

    impl TensorFusion for DefaultOnlyLinalgBackend {}
    impl TensorBuffer for DefaultOnlyLinalgBackend {}
    impl TensorDeviceTransfer for DefaultOnlyLinalgBackend {}
    impl BackendCachedDot for DefaultOnlyLinalgBackend {}
    impl BackendSessionHost for DefaultOnlyLinalgBackend {}
    impl TensorBackend for DefaultOnlyLinalgBackend {}

    impl LinalgBackend for DefaultOnlyLinalgBackend {
        panic_backend_methods! {
            cholesky(input: &Tensor) -> tenferro_tensor::Result<Tensor>;
            triangular_solve(a: &Tensor, b: &Tensor, left_side: bool, lower: bool, transpose_a: bool, unit_diagonal: bool) -> tenferro_tensor::Result<Tensor>;
            lu(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
            full_piv_lu(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
            full_piv_lu_solve(a: &Tensor, b: &Tensor, transpose_a: bool) -> tenferro_tensor::Result<Tensor>;
            svd(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
            qr(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
            eigh(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
            eig(input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
            solve(a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor>;
        }
    }

    let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]);
    let mut backend = DefaultOnlyLinalgBackend;

    let err = backend
        .svd_view(TensorView::F64(input.as_view()))
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "svd",
            ref message,
        } if message.contains("borrowed tensor views")
    ));
}

#[test]
fn cholesky_rejects_rank_less_than_two_even_when_zero_dim() {
    let input = f64_tensor(vec![0], Vec::new());
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.cholesky(&input)));

    assert!(result.is_ok(), "cholesky should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::RankMismatch {
            op: "cholesky",
            expected: 2,
            actual: 1,
        }
    ));
}

#[test]
fn solve_rejects_singular_matrix() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]);
    let b = f64_tensor(vec![2, 1], vec![1.0, 2.0]);
    let mut backend = CpuBackend::new();

    let err = backend.solve(&a, &b).unwrap_err();

    assert!(matches!(err, Error::BackendFailure { op: "solve", .. }));
}

#[test]
fn triangular_solve_rejects_batch_mismatch_without_backend_panic() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let b = f64_tensor(vec![2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = catch_unwind(AssertUnwindSafe(|| {
        backend.triangular_solve(&a, &b, true, true, false, false)
    }));

    assert!(
        result.is_ok(),
        "triangular_solve should return Err on batch mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "triangular_solve",
            ..
        }
    ));
}

#[test]
fn full_piv_lu_solve_rejects_batch_mismatch_without_backend_panic() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let b = f64_tensor(vec![2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = catch_unwind(AssertUnwindSafe(|| {
        backend.full_piv_lu_solve(&a, &b, false)
    }));

    assert!(
        result.is_ok(),
        "full_piv_lu_solve should return Err on batch mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "full_piv_lu_solve",
            ..
        }
    ));
}
