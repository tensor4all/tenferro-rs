use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSessionHost, Buffer, BufferHandle, CompareDir,
    DType, DotGeneralConfig, Error, GatherConfig, MemoryKind, PadConfig, Placement, ScatterConfig,
    SliceConfig, Tensor, TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer,
    TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
    TensorView, TypedTensor,
};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn c64_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn c32_tensor(shape: Vec<usize>, data: Vec<Complex32>) -> Tensor {
    Tensor::C32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn i32_tensor(shape: Vec<usize>, data: Vec<i32>) -> Tensor {
    Tensor::I32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn f64_values(tensor: &Tensor) -> Vec<f64> {
    match tensor {
        Tensor::F64(tensor) => tensor.host_data().unwrap().to_vec(),
        other => panic!("expected F64 tensor, got {:?}", other.dtype()),
    }
}

fn c64_values(tensor: &Tensor) -> Vec<Complex64> {
    match tensor {
        Tensor::C64(tensor) => tensor.host_data().unwrap().to_vec(),
        other => panic!("expected C64 tensor, got {:?}", other.dtype()),
    }
}

fn opaque_backend_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::Device,
        device: None,
    }
}

fn backend_f64_tensor(shape: Vec<usize>, handle_id: u64) -> Tensor {
    let len = shape.iter().product();
    Tensor::F64(
        TypedTensor::<f64>::from_buffer_col_major(
            shape,
            Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(handle_id, len))),
            opaque_backend_placement(),
        )
        .unwrap(),
    )
}

fn assert_backend_download_error<T>(result: tenferro_tensor::Result<T>, expected_op: &'static str) {
    let err = match result {
        Ok(_) => panic!("expected {expected_op} to reject backend buffer"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        Error::BackendFailure {
            op,
            ref message,
        } if op == expected_op && message.contains("download")
    ));
}

fn assert_no_panic_backend_download_error<T>(
    expected_op: &'static str,
    f: impl FnOnce() -> tenferro_tensor::Result<T>,
) {
    let result = catch_unwind(AssertUnwindSafe(f));

    assert!(
        result.is_ok(),
        "{expected_op} should return Err for backend buffers, not panic"
    );
    assert_backend_download_error(result.unwrap(), expected_op);
}

#[test]
fn default_svd_read_returns_explicit_backend_boundary_error() {
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
            cast(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
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

    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]).unwrap();
    let mut backend = DefaultOnlyLinalgBackend;

    let err = backend.lu_factor(&Tensor::F64(input.clone())).unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "lu_factor",
            ref message,
        } if message.contains("does not implement")
    ));

    let err = backend.svd_values(&Tensor::F64(input.clone())).unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "svd_values",
            ref message,
        } if message.contains("does not implement")
    ));

    let err = backend
        .svd_read(TensorView::F64(input.as_view()))
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "svd",
            ref message,
        } if message.contains("borrowed tensor views")
    ));

    let err = backend
        .qr_read(TensorView::F64(input.as_view()))
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "qr",
            ref message,
        } if message.contains("borrowed tensor views")
    ));

    let err = backend
        .eigh_read(TensorView::F64(input.as_view()))
        .unwrap_err();

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "eigh",
            ref message,
        } if message.contains("borrowed tensor views")
    ));

    let err = backend
        .eigh_values(&Tensor::F64(input.clone()))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "eigh_values",
            ref message,
        } if message.contains("does not implement")
    ));

    let pivots = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1, 2]).unwrap());
    let err = backend
        .lu_solve_prepared(
            &Tensor::F64(input.clone()),
            &Tensor::F64(input.clone()),
            &pivots,
            &Tensor::F64(input.clone()),
            false,
            false,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "lu_solve_prepared",
            ref message,
        } if message.contains("does not implement")
    ));

    let err = backend
        .cholesky_read(TensorView::F64(input.as_view()))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "cholesky",
            ref message,
        } if message.contains("borrowed tensor views")
    ));

    let err = backend
        .lu_read(TensorView::F64(input.as_view()))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "lu",
            ref message,
        } if message.contains("borrowed tensor views")
    ));

    let err = backend
        .full_piv_lu_read(TensorView::F64(input.as_view()))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "full_piv_lu",
            ref message,
        } if message.contains("borrowed tensor views")
    ));

    let err = backend
        .eig_read(TensorView::F64(input.as_view()))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "eig",
            ref message,
        } if message.contains("borrowed tensor views")
    ));
}

#[test]
fn cpu_lu_solve_prepared_consumes_packed_factor_outputs() {
    let mut backend = CpuBackend::new();

    let a = f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 3.0]);
    let b = f64_tensor(vec![2, 1], vec![4.0, 9.0]);
    let factors = backend.lu_factor(&a).unwrap();
    let x = backend
        .lu_solve_prepared(&a, &factors[0], &factors[1], &b, false, false)
        .unwrap();
    assert_eq!(f64_values(&x), vec![2.0, 3.0]);

    let a = f64_tensor(vec![2, 2], vec![1.0, 0.0, 2.0, 3.0]);
    let b = f64_tensor(vec![2, 1], vec![5.0, 31.0]);
    let factors = backend.lu_factor(&a).unwrap();
    let x = backend
        .lu_solve_prepared(&a, &factors[0], &factors[1], &b, true, false)
        .unwrap();
    let values = f64_values(&x);
    assert!((values[0] - 5.0).abs() < 1.0e-12);
    assert!((values[1] - 7.0).abs() < 1.0e-12);

    let a = c64_tensor(
        vec![2, 2],
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
        ],
    );
    let b = c64_tensor(
        vec![2, 1],
        vec![Complex64::new(2.0, -2.0), Complex64::new(6.0, 3.0)],
    );
    let factors = backend.lu_factor(&a).unwrap();
    let x = backend
        .lu_solve_prepared(&a, &factors[0], &factors[1], &b, true, true)
        .unwrap();
    let values = c64_values(&x);
    assert!((values[0] - Complex64::new(2.0, 0.0)).norm() < 1.0e-12);
    assert!((values[1] - Complex64::new(3.0, 0.0)).norm() < 1.0e-12);
}

#[test]
fn cpu_lu_factor_covers_pivoted_real_and_complex_dtypes() {
    let mut backend = CpuBackend::new();

    let a = f32_tensor(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]);
    let factors = backend.lu_factor(&a).unwrap();
    assert!(matches!(&factors[0], Tensor::F32(t) if t.shape() == [2, 2]));
    assert!(matches!(&factors[1], Tensor::I32(t) if t.host_data().unwrap() == [2, 2]));
    assert!(matches!(&factors[2], Tensor::F32(t) if t.host_data().unwrap() == [-1.0]));

    let a = c32_tensor(
        vec![2, 2],
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.0, 1.0),
        ],
    );
    let factors = backend.lu_factor(&a).unwrap();
    assert!(matches!(&factors[0], Tensor::C32(t) if t.shape() == [2, 2]));
    assert!(matches!(&factors[1], Tensor::I32(t) if t.host_data().unwrap() == [1, 2]));
    assert!(
        matches!(&factors[2], Tensor::C32(t) if t.host_data().unwrap() == [Complex32::new(1.0, 0.0)])
    );
}

#[test]
fn cpu_values_only_decompositions_cover_real_complex_and_batched_inputs() {
    let mut backend = CpuBackend::new();

    let s = backend
        .svd_values(&f32_tensor(vec![2, 2], vec![3.0, 0.0, 0.0, 4.0]))
        .unwrap();
    assert!(matches!(s, Tensor::F32(ref t) if t.shape() == [2]));

    let s = backend
        .svd_values(&f64_tensor(
            vec![2, 2, 2],
            vec![3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0],
        ))
        .unwrap();
    assert!(matches!(s, Tensor::F64(ref t) if t.shape() == [2, 2]));

    let s = backend
        .svd_values(&c32_tensor(
            vec![2, 2],
            vec![
                Complex32::new(3.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(4.0, 0.0),
            ],
        ))
        .unwrap();
    assert!(matches!(s, Tensor::F32(ref t) if t.shape() == [2]));

    let s = backend
        .svd_values(&c64_tensor(
            vec![2, 2],
            vec![
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        ))
        .unwrap();
    assert!(matches!(s, Tensor::F64(ref t) if t.shape() == [2]));

    let values = backend
        .eigh_values(&f32_tensor(vec![2, 2], vec![3.0, 0.0, 0.0, 4.0]))
        .unwrap();
    assert!(matches!(values, Tensor::F32(ref t) if t.shape() == [2]));

    let values = backend
        .eigh_values(&f64_tensor(
            vec![2, 2, 2],
            vec![3.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0],
        ))
        .unwrap();
    assert!(matches!(values, Tensor::F64(ref t) if t.shape() == [2, 2]));

    let values = backend
        .eigh_values(&c32_tensor(
            vec![2, 2],
            vec![
                Complex32::new(3.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(4.0, 0.0),
            ],
        ))
        .unwrap();
    assert!(matches!(values, Tensor::F32(ref t) if t.shape() == [2]));

    let values = backend
        .eigh_values(&c64_tensor(
            vec![2, 2],
            vec![
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        ))
        .unwrap();
    assert!(matches!(values, Tensor::F64(ref t) if t.shape() == [2]));
}

#[test]
fn cpu_lu_solve_prepared_restores_vector_rhs_and_validates_inputs() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]);
    let factors = backend.lu_factor(&a).unwrap();
    let b = f64_tensor(vec![2], vec![6.0, 20.0]);

    let x = backend
        .lu_solve_prepared(&a, &factors[0], &factors[1], &b, false, false)
        .unwrap();
    assert_eq!(x.shape(), &[2]);
    assert_eq!(f64_values(&x), vec![3.0, 5.0]);

    let empty_a = f64_tensor(vec![0, 0], Vec::new());
    let empty_b = f64_tensor(vec![0, 1], Vec::new());
    let empty_pivots = i32_tensor(vec![0], Vec::new());
    let x = backend
        .lu_solve_prepared(&empty_a, &empty_a, &empty_pivots, &empty_b, false, false)
        .unwrap();
    assert_eq!(x.shape(), &[0, 1]);

    let bad_pivots = f64_tensor(vec![2], vec![1.0, 2.0]);
    let err = backend
        .lu_solve_prepared(&a, &factors[0], &bad_pivots, &b, false, false)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            op: "lu_solve_prepared",
            ..
        }
    ));

    let bad_b = c64_tensor(
        vec![2, 1],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    );
    let err = backend
        .lu_solve_prepared(&a, &factors[0], &factors[1], &bad_b, false, false)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            op: "lu_solve_prepared",
            ..
        }
    ));

    let bad_pivots = i32_tensor(vec![2], vec![0, 2]);
    let err = backend
        .lu_solve_prepared(&a, &a, &bad_pivots, &b, false, false)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "lu_solve_prepared",
            ref message,
        } if message.contains("1-based")
    ));
}

#[test]
fn cpu_lu_solve_prepared_rejects_rank_less_than_two() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![1.0, 2.0]);
    let pivots = i32_tensor(vec![2], vec![1, 2]);

    let err = backend
        .lu_solve_prepared(&a, &a, &pivots, &b, true, false)
        .unwrap_err();

    assert!(matches!(
        err,
        Error::RankMismatch {
            op: "lu_solve_prepared",
            ..
        }
    ));
}

#[test]
fn cpu_linalg_rejects_backend_buffers_without_panicking_or_downloading() {
    let a = backend_f64_tensor(vec![2, 2], 101);
    let b = backend_f64_tensor(vec![2, 1], 102);
    let mut backend = CpuBackend::new();

    assert_no_panic_backend_download_error("cholesky", || backend.cholesky(&a));
    assert_no_panic_backend_download_error("triangular_solve", || {
        backend.triangular_solve(&a, &b, true, true, false, false)
    });
    assert_no_panic_backend_download_error("lu", || backend.lu(&a));
    assert_no_panic_backend_download_error("full_piv_lu", || backend.full_piv_lu(&a));
    assert_no_panic_backend_download_error("full_piv_lu_solve", || {
        backend.full_piv_lu_solve(&a, &b, false)
    });
    assert_no_panic_backend_download_error("svd", || backend.svd(&a));
    assert_no_panic_backend_download_error("qr", || backend.qr(&a));
    assert_no_panic_backend_download_error("eigh", || backend.eigh(&a));
    assert_no_panic_backend_download_error("eig", || backend.eig(&a));
    assert_no_panic_backend_download_error("solve", || backend.solve(&a, &b));
}

#[test]
fn cpu_linalg_rejects_backend_rhs_before_zero_dim_fast_paths() {
    let a = f64_tensor(vec![0, 0], Vec::new());
    let b = backend_f64_tensor(vec![0, 1], 103);
    let mut backend = CpuBackend::new();

    assert_no_panic_backend_download_error("solve", || backend.solve(&a, &b));
    assert_no_panic_backend_download_error("full_piv_lu_solve", || {
        backend.full_piv_lu_solve(&a, &b, false)
    });
}

#[test]
fn solve_rejects_invalid_dtype_pairs_before_zero_dim_fast_path() {
    let mut backend = CpuBackend::new();
    let f64_a = f64_tensor(vec![0, 0], Vec::new());
    let c64_b = c64_tensor(vec![0, 1], Vec::new());
    let i32_a = i32_tensor(vec![0, 0], Vec::new());
    let i32_b = i32_tensor(vec![0, 1], Vec::new());

    let err = backend.solve(&f64_a, &c64_b).unwrap_err();
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            op: "solve",
            lhs: DType::F64,
            rhs: DType::C64,
        }
    ));

    let err = backend.solve(&i32_a, &i32_b).unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "solve",
            ref message,
        } if message.contains("unsupported dtype I32")
    ));
}

#[test]
fn full_piv_lu_solve_rejects_invalid_dtype_pairs_before_zero_dim_fast_path() {
    let mut backend = CpuBackend::new();
    let f64_a = f64_tensor(vec![0, 0], Vec::new());
    let c64_b = c64_tensor(vec![0, 1], Vec::new());
    let i32_a = i32_tensor(vec![0, 0], Vec::new());
    let i32_b = i32_tensor(vec![0, 1], Vec::new());

    let err = backend
        .full_piv_lu_solve(&f64_a, &c64_b, false)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            op: "full_piv_lu_solve",
            lhs: DType::F64,
            rhs: DType::C64,
        }
    ));

    let err = backend
        .full_piv_lu_solve(&i32_a, &i32_b, false)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "full_piv_lu_solve",
            ref message,
        } if message.contains("unsupported dtype I32")
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
