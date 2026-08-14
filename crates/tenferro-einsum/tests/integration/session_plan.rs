//! Single-session-entry and typed-result tests for the
//! `ConcreteEinsumPlan::execute*` session surfaces.
//!
//! A mixed chain (`plan.execute` + runtime `exp` + `reduce_sum`) must execute
//! inside exactly one backend session entry, matching an independent scalar
//! expected value. The typed session surface must reject a backend-returned
//! wrong dtype through `into_typed_result`.

use std::cell::Cell;

use tenferro_cpu::CpuBackend;
use tenferro_einsum::{ConcreteEinsumPlan, Error};
use tenferro_runtime::{Tensor, TensorSessionOpsExt};
use tenferro_tensor::backend::{
    BackendCachedDot, TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};
use tenferro_tensor::{
    BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir, DType, DotGeneralConfig,
    GatherConfig, PadConfig, ScatterConfig, SliceConfig, TensorBackend, TensorRead, TensorWrite,
};

type TensorResult = tenferro_tensor::Result<Tensor>;

/// Test backend counting `with_backend_session` entries while delegating real
/// einsum (`dot_general`), `exp`, and `reduce_sum` execution to an inner
/// [`CpuBackend`]. All other op methods are unreachable in these tests and
/// panic.
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

/// Test backend whose `dot_general` returns an `F64` tensor regardless of the
/// requested dtype, so the typed `_in_session` surface must reject the output
/// through `into_typed_result`.
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

/// Panic-only implementations of every op trait except `TensorDot`,
/// `TensorAnalytic`, and `TensorReduction`, which test backends override with
/// real or wrong-dtype ops. Excludes `BackendSessionHost`, which each backend
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

        impl TensorFusion for $ty {}
        impl TensorBuffer for $ty {}

        impl TensorDeviceTransfer for $ty {
            fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
                Err(tenferro_tensor::Error::unsupported(
                    concat!(stringify!($ty), "::download_to_host"),
                    "test backend does not transfer tensors",
                ))
            }

            fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
                Err(tenferro_tensor::Error::unsupported(
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

impl TensorDot for SessionCountingBackend {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> TensorResult {
        self.inner.dot_general(lhs, rhs, config)
    }
}

impl TensorElementwise for SessionCountingBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> TensorResult {
        self.inner.add(lhs, rhs)
    }

    fn sub(&mut self, lhs: &Tensor, rhs: &Tensor) -> TensorResult {
        self.inner.sub(lhs, rhs)
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> TensorResult {
        self.inner.mul(lhs, rhs)
    }

    fn neg(&mut self, input: &Tensor) -> TensorResult {
        self.inner.neg(input)
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> TensorResult {
        self.inner.div(lhs, rhs)
    }

    fn abs(&mut self, input: &Tensor) -> TensorResult {
        self.inner.abs(input)
    }

    fn sign(&mut self, input: &Tensor) -> TensorResult {
        self.inner.sign(input)
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> TensorResult {
        self.inner.maximum(lhs, rhs)
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> TensorResult {
        self.inner.minimum(lhs, rhs)
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> TensorResult {
        self.inner.compare(lhs, rhs, dir)
    }

    fn select(&mut self, pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> TensorResult {
        self.inner.select(pred, on_true, on_false)
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult {
        self.inner.clamp(input, lower, upper)
    }

    fn conj(&mut self, input: &Tensor) -> TensorResult {
        self.inner.conj(input)
    }
}

impl TensorAnalytic for SessionCountingBackend {
    fn exp(&mut self, input: &Tensor) -> TensorResult {
        self.inner.exp(input)
    }

    fn log(&mut self, input: &Tensor) -> TensorResult {
        self.inner.log(input)
    }

    fn sin(&mut self, input: &Tensor) -> TensorResult {
        self.inner.sin(input)
    }

    fn cos(&mut self, input: &Tensor) -> TensorResult {
        self.inner.cos(input)
    }

    fn tanh(&mut self, input: &Tensor) -> TensorResult {
        self.inner.tanh(input)
    }

    fn sqrt(&mut self, input: &Tensor) -> TensorResult {
        self.inner.sqrt(input)
    }

    fn rsqrt(&mut self, input: &Tensor) -> TensorResult {
        self.inner.rsqrt(input)
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> TensorResult {
        self.inner.pow(lhs, rhs)
    }

    fn expm1(&mut self, input: &Tensor) -> TensorResult {
        self.inner.expm1(input)
    }

    fn log1p(&mut self, input: &Tensor) -> TensorResult {
        self.inner.log1p(input)
    }
}

impl TensorReduction for SessionCountingBackend {
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> TensorResult {
        self.inner.reduce_sum(input, axes)
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> TensorResult {
        self.inner.reduce_prod(input, axes)
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> TensorResult {
        self.inner.reduce_max(input, axes)
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> TensorResult {
        self.inner.reduce_min(input, axes)
    }
}

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

impl TensorDot for WrongDTypeSessionBackend {
    fn dot_general(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _config: &DotGeneralConfig,
    ) -> TensorResult {
        Ok(Tensor::F64(
            tenferro_tensor::TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0; 4]).unwrap(),
        ))
    }
}

panic_elementwise!(WrongDTypeSessionBackend);
panic_analytic!(WrongDTypeSessionBackend);
panic_reduction!(WrongDTypeSessionBackend);
impl BackendSessionHost for WrongDTypeSessionBackend {}

// ---------------------------------------------------------------------------
// Single-session-entry proof
// ---------------------------------------------------------------------------

#[test]
fn einsum_plan_mixed_chain_enters_one_session() {
    let mut backend = SessionCountingBackend::new();
    let lhs =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();

    // 10 iterations of (einsum + exp + reduce_sum) must execute inside
    // exactly one backend session entry.
    backend.entries.set(0);
    let session = backend
        .with_backend_session(|session| -> tenferro_einsum::Result<Tensor> {
            let mut x = lhs.duplicate().unwrap();
            for _ in 0..10 {
                x = plan.execute([&lhs, &rhs], session)?;
                x = x.exp(session)?;
                x = x.reduce_sum(&[1], session)?;
            }
            Ok(x)
        })
        .unwrap();
    assert_eq!(
        backend.entries.get(),
        1,
        "mixed einsum plan + session chain must enter exactly one session"
    );

    // Independent scalar math: each iteration is
    // einsum(lhs, rhs) -> exp -> reduce_sum(axis 1). The einsum over the
    // [2,3] x [3,2] inputs (values 1..6 each) is [[22, 28], [49, 64]] in
    // col-major storage, so the final value is
    // [exp(22) + exp(49), exp(28) + exp(64)].
    let expected = [
        22.0_f64.exp() + 49.0_f64.exp(),
        28.0_f64.exp() + 64.0_f64.exp(),
    ];
    let actual = session.as_slice::<f64>().unwrap();
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs() / expected;
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, rel error={error}"
        );
    }
}

#[test]
fn einsum_plan_execute_in_session_adds_no_nested_entry_to_caller_session() {
    let mut backend = SessionCountingBackend::new();
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik").unwrap();

    backend.entries.set(0);
    let result = backend.with_backend_session(|session| {
        let x = plan
            .execute([&lhs, &rhs], session)
            .expect("einsum plan should execute inside the session");
        x.exp(session).expect("exp should run in the session")
    });
    assert_eq!(
        backend.entries.get(),
        1,
        "execute must not add a nested session entry"
    );
    assert_eq!(result.shape(), &[2, 2]);
}

// ---------------------------------------------------------------------------
// Typed-result validation through into_typed_result
// ---------------------------------------------------------------------------

#[test]
fn einsum_plan_typed_execute_rejects_wrong_backend_dtype() {
    let typed_lhs =
        tenferro_tensor::TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0_f32; 4])
            .unwrap();
    let typed_rhs =
        tenferro_tensor::TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0_f32; 4])
            .unwrap();
    let plan = ConcreteEinsumPlan::prepare_typed([&typed_lhs, &typed_rhs], "ij,jk->ik").unwrap();
    let mut backend = WrongDTypeSessionBackend;

    // The backend returns an F64 tensor regardless of the requested dtype, so
    // the typed session surface must reject it through into_typed_result.
    let in_session = backend
        .with_backend_session(|session| plan.execute_typed([&typed_lhs, &typed_rhs], session))
        .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Tensor(tenferro_tensor::Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
}
