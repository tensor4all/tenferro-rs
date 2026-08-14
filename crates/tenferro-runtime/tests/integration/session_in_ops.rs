//! Parity and single-session-entry tests for the `_in` session surfaces.
//!
//! Covers, per surface (dynamic [`Tensor`] and typed [`TypedTensor`]):
//! equal-shape ops, real broadcast, invalid-broadcast structured-error parity
//! with the one-shot path, dtype parity, typed output dtype validation
//! (`into_typed_result`), and a deterministic proof that an `_in` chain
//! executes inside exactly one backend session entry while the one-shot
//! counterpart enters one session per op.

use std::cell::Cell;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{
    Tensor, TensorOpsExt, TensorSessionOpsExt, TypedTensor, TypedTensorOpsExt,
    TypedTensorSessionOpsExt,
};
use tenferro_tensor::backend::{
    BackendCachedDot, TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};
use tenferro_tensor::{
    BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir, DType, DotGeneralConfig,
    Error, GatherConfig, PadConfig, ScatterConfig, ShapeMismatch, SliceConfig, TensorBackend,
    TensorRead, TensorWrite, ValidationError,
};

type TensorResult = tenferro_tensor::Result<Tensor>;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

fn assert_incompatible_shapes_error(error: Error) {
    match error {
        Error::Validation {
            op: "broadcast",
            source: ValidationError::ShapeMismatch(shape),
        } => assert!(
            matches!(
                shape.as_ref(),
                ShapeMismatch::IncompatibleShapes { lhs, rhs }
                    if lhs.as_slice() == [2] && rhs.as_slice() == [3]
            ),
            "unexpected shape payload: {shape:?}"
        ),
        other => panic!("expected broadcast ShapeMismatch, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Dynamic surface (TensorSessionOpsExt)
// ---------------------------------------------------------------------------

#[test]
fn session_in_dynamic_equal_shape_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.5_f64; 8];
    let b_values = vec![2.25_f64; 8];
    let a = Tensor::from_vec_col_major(vec![8], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&b, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&b, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });

    // Independent value check in plain scalar math: the chain computes
    // sum_i exp(a_i + b_i) * b_i without any tenferro op, so a shared
    // one-shot regression cannot satisfy it.
    let expected: f64 = a_values
        .iter()
        .zip(&b_values)
        .map(|(&x, &y)| (x + y).exp() * y)
        .sum();
    assert_close(session.as_slice::<f64>().unwrap(), &[expected]);

    assert_eq!(session.shape(), one_shot.shape());
    assert_close(
        session.as_slice::<f64>().unwrap(),
        one_shot.as_slice::<f64>().unwrap(),
    );
}

#[test]
fn session_in_dynamic_broadcast_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.0_f64];
    let b_values = vec![2.0_f64; 8];
    let a = Tensor::from_vec_col_major(vec![1], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });

    // Independent value check: the broadcast chain computes
    // sum_i exp(a_0 + b_i) * a_0 in plain scalar math.
    let expected: f64 = b_values
        .iter()
        .map(|&y| (a_values[0] + y).exp() * a_values[0])
        .sum();
    assert_close(session.as_slice::<f64>().unwrap(), &[expected]);

    assert_eq!(session.shape(), one_shot.shape());
    assert_close(
        session.as_slice::<f64>().unwrap(),
        one_shot.as_slice::<f64>().unwrap(),
    );
}

#[test]
fn session_in_dynamic_invalid_broadcast_matches_one_shot_error() {
    let mut backend = CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let one_shot_error = a.add(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.add_in(&b, s))
        .unwrap_err();

    assert_incompatible_shapes_error(one_shot_error);
    assert_incompatible_shapes_error(session_error);
}

#[test]
fn session_in_dynamic_dtype_error_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![1_i32; 2]).unwrap();

    let one_shot_error = a.add(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.add_in(&b, s))
        .unwrap_err();

    // Assert the full payload (op name, source lhs dtype as expected, rhs
    // dtype as actual) for both paths, not just the error kind.
    for error in [one_shot_error, session_error] {
        let Error::Validation {
            op: "add",
            source: ValidationError::DTypeMismatch { expected, actual },
        } = &error
        else {
            panic!("expected add DTypeMismatch, got {error:?}");
        };
        assert_eq!(*expected, tenferro_tensor::core::DType::F64);
        assert_eq!(*actual, tenferro_tensor::core::DType::I32);
    }
}

// ---------------------------------------------------------------------------
// Typed surface (TypedTensorSessionOpsExt)
// ---------------------------------------------------------------------------

#[test]
fn session_in_typed_equal_shape_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.5_f64; 8];
    let b_values = vec![2.25_f64; 8];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![8], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&b, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&b, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });

    // Independent value check in plain scalar math (see the dynamic twin).
    let expected: f64 = a_values
        .iter()
        .zip(&b_values)
        .map(|(&x, &y)| (x + y).exp() * y)
        .sum();
    assert_close(session.host_data().unwrap(), &[expected]);

    assert_eq!(session.shape(), one_shot.shape());
    assert_close(session.host_data().unwrap(), one_shot.host_data().unwrap());
}

#[test]
fn session_in_typed_broadcast_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.0_f64];
    let b_values = vec![2.0_f64; 8];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![1], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });

    // Independent value check: sum_i exp(a_0 + b_i) * a_0 in plain scalar math.
    let expected: f64 = b_values
        .iter()
        .map(|&y| (a_values[0] + y).exp() * a_values[0])
        .sum();
    assert_close(session.host_data().unwrap(), &[expected]);

    assert_eq!(session.shape(), one_shot.shape());
    assert_close(session.host_data().unwrap(), one_shot.host_data().unwrap());
}

#[test]
fn session_in_typed_invalid_broadcast_matches_one_shot_error() {
    let mut backend = CpuBackend::new();
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0; 2]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0; 3]).unwrap();

    let one_shot_error = a.add(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.add_in(&b, s))
        .unwrap_err();

    assert_incompatible_shapes_error(one_shot_error);
    assert_incompatible_shapes_error(session_error);
}

#[test]
fn session_in_typed_validates_output_dtype() {
    let mut backend = WrongDTypeSessionBackend;
    let a = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();

    let add_error = backend
        .with_backend_session(|s| a.add_in(&a, s))
        .unwrap_err();
    let mul_error = backend
        .with_backend_session(|s| a.mul_in(&a, s))
        .unwrap_err();
    let exp_error = backend.with_backend_session(|s| a.exp_in(s)).unwrap_err();
    let reduce_error = backend
        .with_backend_session(|s| a.reduce_sum_in(&[0], s))
        .unwrap_err();
    for (op, error) in [
        ("add", add_error),
        ("mul", mul_error),
        ("exp", exp_error),
        ("reduce_sum", reduce_error),
    ] {
        let Error::Validation {
            op: error_op,
            source: ValidationError::DTypeMismatch { expected, actual },
        } = &error
        else {
            panic!("expected {op} DTypeMismatch, got {error:?}");
        };
        assert_eq!(*error_op, op);
        assert_eq!(*expected, tenferro_tensor::core::DType::I32);
        assert_eq!(*actual, tenferro_tensor::core::DType::F64);
    }
}

// ---------------------------------------------------------------------------
// Single-session-entry proof
// ---------------------------------------------------------------------------

#[test]
fn session_in_chain_enters_one_session_one_shot_enters_ten() {
    let mut backend = SessionCountingBackend::new();
    let a = Tensor::from_vec_col_major(vec![1], vec![0.5_f64]).unwrap();
    let b = Tensor::from_vec_col_major(vec![8], vec![1.0_f64; 8]).unwrap();

    // 10 one-shot ops (3x add->exp->mul + final reduce_sum): one session entry
    // per op.
    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();
    assert_eq!(
        backend.entries.get(),
        10,
        "one-shot chain must enter one session per op"
    );

    backend.entries.set(0);
    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        let x = x.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        let x = x.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });
    assert_eq!(
        backend.entries.get(),
        1,
        "session chain must enter exactly one session"
    );

    assert_close(
        session.as_slice::<f64>().unwrap(),
        one_shot.as_slice::<f64>().unwrap(),
    );
}

// ---------------------------------------------------------------------------
// Test backends
// ---------------------------------------------------------------------------

/// Test backend counting `with_backend_session` entries while delegating real
/// execution to an inner [`CpuBackend`]. All direct op methods are unreachable
/// in these tests and panic.
struct SessionCountingBackend {
    inner: CpuBackend,
    entries: Cell<usize>,
}

impl SessionCountingBackend {
    fn new() -> Self {
        Self {
            inner: CpuBackend::new(),
            entries: Cell::new(0),
        }
    }
}

/// Test backend whose session ops return an `F64` tensor regardless of the
/// requested dtype, so the typed `_in` surface must reject the output through
/// `into_typed_result`.
struct WrongDTypeSessionBackend;

macro_rules! panic_backend_methods {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                let _ = ($($arg),*);
                panic!(concat!(stringify!($name), " should not be called in this test"))
            }
        )+
    };
}

/// Panic-only implementations of every op trait except the elementwise,
/// analytic, and reduction families (which some test backends override with
/// real ops). Also excludes `BackendSessionHost`, which each backend
/// implements explicitly.
macro_rules! test_backend_impls {
    ($ty:ident, $marker:ident) => {
        struct $marker;

        impl BackendRuntimeCache for $ty {
            type RuntimeCache = <CpuBackend as BackendRuntimeCache>::RuntimeCache;
        }

        impl TensorStructural for $ty {
            fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult {
                CpuBackend::new().to_contiguous_read(input)
            }

            fn copy_read_into(
                &mut self,
                src: TensorRead<'_>,
                dst: TensorWrite<'_>,
            ) -> tenferro_tensor::Result<()> {
                CpuBackend::new().copy_read_into(src, dst)
            }

            panic_backend_methods! {
                transpose(input: &Tensor, perm: &[usize]) -> TensorResult;
                reshape(input: &Tensor, shape: &[usize]) -> TensorResult;
                broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> TensorResult;
                cast(input: &Tensor, to: DType) -> TensorResult;
                convert(input: &Tensor, to: DType) -> TensorResult;
                extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult;
                embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult;
                tril(input: &Tensor, k: i64) -> TensorResult;
                triu(input: &Tensor, k: i64) -> TensorResult;
            }
        }

        impl TensorIndexing for $ty {
            panic_backend_methods! {
                gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> TensorResult;
                scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> TensorResult;
                slice(input: &Tensor, config: &SliceConfig) -> TensorResult;
                dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> TensorResult;
                dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> TensorResult;
                pad(input: &Tensor, config: &PadConfig) -> TensorResult;
                concatenate(inputs: &[&Tensor], axis: usize) -> TensorResult;
                reverse(input: &Tensor, axes: &[usize]) -> TensorResult;
            }
        }

        impl TensorDot for $ty {
            fn dot_general(
                &mut self,
                _lhs: &Tensor,
                _rhs: &Tensor,
                _config: &DotGeneralConfig,
            ) -> TensorResult {
                panic!("dot_general should not be called in this test")
            }
        }

        impl TensorFusion for $ty {}
        impl TensorBuffer for $ty {}

        impl TensorDeviceTransfer for $ty {
            fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
                Err(Error::unsupported(
                    concat!(stringify!($ty), "::download_to_host"),
                    "test backend does not transfer tensors",
                ))
            }

            fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
                Err(Error::unsupported(
                    concat!(stringify!($ty), "::upload_host_tensor"),
                    "test backend does not transfer tensors",
                ))
            }
        }

        impl BackendCachedDot for $ty {}

        impl BackendSession for $ty {
            fn session_type_id(&self) -> std::any::TypeId {
                std::any::TypeId::of::<$marker>()
            }

            unsafe fn session_data_mut(&mut self) -> *mut () {
                self as *mut Self as *mut ()
            }
        }

        impl TensorBackend for $ty {}
    };
}

macro_rules! panic_elementwise {
    ($ty:ident) => {
        impl TensorElementwise for $ty {
            panic_backend_methods! {
                add(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                sub(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                mul(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                neg(input: &Tensor) -> TensorResult;
                div(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                abs(input: &Tensor) -> TensorResult;
                sign(input: &Tensor) -> TensorResult;
                maximum(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                minimum(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> TensorResult;
                select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> TensorResult;
                clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult;
            }

            fn conj(&mut self, input: &Tensor) -> TensorResult {
                CpuBackend::new().conj(input)
            }
        }
    };
}

macro_rules! panic_analytic {
    ($ty:ident) => {
        impl TensorAnalytic for $ty {
            panic_backend_methods! {
                exp(input: &Tensor) -> TensorResult;
                log(input: &Tensor) -> TensorResult;
                sin(input: &Tensor) -> TensorResult;
                cos(input: &Tensor) -> TensorResult;
                tanh(input: &Tensor) -> TensorResult;
                sqrt(input: &Tensor) -> TensorResult;
                rsqrt(input: &Tensor) -> TensorResult;
                pow(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                expm1(input: &Tensor) -> TensorResult;
                log1p(input: &Tensor) -> TensorResult;
            }
        }
    };
}

macro_rules! panic_reduction {
    ($ty:ident) => {
        impl TensorReduction for $ty {
            panic_backend_methods! {
                reduce_sum(input: &Tensor, axes: &[usize]) -> TensorResult;
                reduce_prod(input: &Tensor, axes: &[usize]) -> TensorResult;
                reduce_max(input: &Tensor, axes: &[usize]) -> TensorResult;
                reduce_min(input: &Tensor, axes: &[usize]) -> TensorResult;
            }
        }
    };
}

test_backend_impls!(SessionCountingBackend, SessionCountingBackendMarker);
panic_elementwise!(SessionCountingBackend);
panic_analytic!(SessionCountingBackend);
panic_reduction!(SessionCountingBackend);

impl BackendSessionHost for SessionCountingBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        self.entries.set(self.entries.get() + 1);
        self.inner.with_backend_session(f)
    }
}

test_backend_impls!(WrongDTypeSessionBackend, WrongDTypeSessionBackendMarker);

impl TensorElementwise for WrongDTypeSessionBackend {
    panic_backend_methods! {
        sub(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
        neg(input: &Tensor) -> TensorResult;
        div(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
        abs(input: &Tensor) -> TensorResult;
        sign(input: &Tensor) -> TensorResult;
        maximum(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
        minimum(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
        compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> TensorResult;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> TensorResult;
        clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult;
    }

    fn add(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn add_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn mul(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn mul_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn conj(&mut self, input: &Tensor) -> TensorResult {
        CpuBackend::new().conj(input)
    }
}

impl TensorAnalytic for WrongDTypeSessionBackend {
    panic_backend_methods! {
        log(input: &Tensor) -> TensorResult;
        sin(input: &Tensor) -> TensorResult;
        cos(input: &Tensor) -> TensorResult;
        tanh(input: &Tensor) -> TensorResult;
        sqrt(input: &Tensor) -> TensorResult;
        rsqrt(input: &Tensor) -> TensorResult;
        pow(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
        expm1(input: &Tensor) -> TensorResult;
        log1p(input: &Tensor) -> TensorResult;
    }

    fn exp(&mut self, _input: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn exp_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }
}

impl TensorReduction for WrongDTypeSessionBackend {
    panic_backend_methods! {
        reduce_prod(input: &Tensor, axes: &[usize]) -> TensorResult;
        reduce_max(input: &Tensor, axes: &[usize]) -> TensorResult;
        reduce_min(input: &Tensor, axes: &[usize]) -> TensorResult;
    }

    fn reduce_sum(&mut self, _input: &Tensor, _axes: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn reduce_sum_read(&mut self, _input: TensorRead<'_>, _axes: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }
}

impl BackendSessionHost for WrongDTypeSessionBackend {}

fn wrong_dtype_tensor() -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![1.0]).unwrap())
}
