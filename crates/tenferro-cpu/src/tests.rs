#![allow(dead_code)]

use std::panic::AssertUnwindSafe;

use num_complex::{Complex32, Complex64};

use crate::{
    abs, add, broadcast_in_dim, clamp, compare, conj, div, dynamic_slice, dynamic_update_slice,
    embed_diagonal, extract_diagonal, gather, maximum, minimum, mul, neg, pad, pow, reduce_max,
    reduce_min, reduce_prod, reduce_sum, reduce_sum_squares, rem, reshape, scatter, select, sign,
    transpose, tril, triu, with_cpu_engine, CpuEngine, Error,
};
#[cfg(feature = "blas")]
use tenferro_tensor::StridedSliceSpec;
use tenferro_tensor::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, TensorAnalytic, TensorDot, TensorElementwise,
    TensorStructural,
};
use tenferro_tensor::{
    DType, Tensor, TensorRead, TensorView, TensorViewMut, TensorWrite, TypedTensor,
    TypedTensorView, TypedTensorViewMut,
};

pub(crate) use crate::test_support::with_cpu_session;

#[test]
fn with_cpu_engine_checks_exact_marker_and_scopes_borrow() {
    let mut backend = crate::CpuEngine::serial();
    let value = crate::test_support::on_engine(&mut backend, |session| {
        // The session the owner builds is the CPU execution session, which
        // is what the capability bridge recognizes.
        assert!(session
            .native_engine()
            .is_some_and(|token| { token.has_marker::<crate::exec_session::CpuEngineMarker>() }));
        with_cpu_engine(session, |session: &mut CpuEngine| {
            let _: &mut CpuEngine = session;
            17usize
        })
    })
    .unwrap()
    .expect("CpuEngine must expose its scoped CpuEngine");
    assert_eq!(value, 17);
}

fn get_f64(t: &Tensor, idx: &[usize]) -> f64 {
    *t.as_typed::<f64>()
        .expect("expected F64 tensor")
        .get(idx)
        .unwrap()
}

fn get_c64(t: &Tensor, idx: &[usize]) -> Complex64 {
    *t.as_typed::<Complex64>()
        .expect("expected C64 tensor")
        .get(idx)
        .unwrap()
}

fn get_f32(t: &Tensor, idx: &[usize]) -> f32 {
    *t.as_typed::<f32>()
        .expect("expected F32 tensor")
        .get(idx)
        .unwrap()
}

fn get_c32(t: &Tensor, idx: &[usize]) -> Complex32 {
    *t.as_typed::<Complex32>()
        .expect("expected C32 tensor")
        .get(idx)
        .unwrap()
}

fn get_i64(t: &Tensor, idx: &[usize]) -> i64 {
    *t.as_typed::<i64>()
        .expect("expected I64 tensor")
        .get(idx)
        .unwrap()
}

fn get_i32(t: &Tensor, idx: &[usize]) -> i32 {
    *t.as_typed::<i32>()
        .expect("expected I32 tensor")
        .get(idx)
        .unwrap()
}

fn get_bool(t: &Tensor, idx: &[usize]) -> bool {
    *t.as_typed::<bool>()
        .expect("expected Bool tensor")
        .get(idx)
        .unwrap()
}

fn assert_f64_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-12,
        "expected {expected}, got {actual}"
    );
}

fn assert_f64_close_tol(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "expected {expected}, got {actual}, tol={tol}"
    );
}

fn assert_c64_close(actual: Complex64, expected: Complex64) {
    assert_f64_close(actual.re, expected.re);
    assert_f64_close(actual.im, expected.im);
}

fn assert_c64_close_tol(actual: Complex64, expected: Complex64, tol: f64) {
    assert_f64_close_tol(actual.re, expected.re, tol);
    assert_f64_close_tol(actual.im, expected.im, tol);
}

fn col_major_index(rows: usize, row: usize, col: usize) -> usize {
    row + col * rows
}

fn transpose_f64(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}

fn conjugate_transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
        }
    }
    out
}

fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn matmul_c64(
    lhs: &[Complex64],
    rhs: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}

fn diag_f64(values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}

fn diag_c64(values: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}

fn batch_matrix_f64_from_tensor(
    t: &Tensor,
    rows: usize,
    cols: usize,
    batch_idx: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(rows, i, j)] = get_f64(t, &[i, j, batch_idx]);
        }
    }
    out
}

fn matrix_f64_from_tensor(t: &Tensor, rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(rows, i, j)] = get_f64(t, &[i, j]);
        }
    }
    out
}

fn batch_vector_f64_from_tensor(t: &Tensor, len: usize, batch_idx: usize) -> Vec<f64> {
    let mut out = vec![0.0; len];
    for (i, value) in out.iter_mut().enumerate().take(len) {
        *value = get_f64(t, &[i, batch_idx]);
    }
    out
}

fn matrix_c64_from_tensor(t: &Tensor, rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(rows, i, j)] = get_c64(t, &[i, j]);
        }
    }
    out
}

fn batch_matrix_c64_from_tensor(
    t: &Tensor,
    rows: usize,
    cols: usize,
    batch_idx: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(rows, i, j)] = get_c64(t, &[i, j, batch_idx]);
        }
    }
    out
}

fn vector_c64_from_tensor(t: &Tensor, len: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    for (i, value) in out.iter_mut().enumerate().take(len) {
        *value = get_c64(t, &[i]);
    }
    out
}

fn batch_vector_c64_from_tensor(t: &Tensor, len: usize, batch_idx: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); len];
    for (i, value) in out.iter_mut().enumerate().take(len) {
        *value = get_c64(t, &[i, batch_idx]);
    }
    out
}

fn simple_gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    }
}

fn diagonal_scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0, 1],
        scatter_dims_to_operand_dims: vec![0, 1],
        index_vector_dim: 1,
    }
}

#[test]
fn cpu_elementwise_kernels_live_in_internal_crate() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let cpu_manifest = std::fs::read_to_string(manifest_dir.join("Cargo.toml"))
        .expect("tenferro-cpu manifest must be readable");
    assert!(
        cpu_manifest.contains("tenferro-internal-cpu-kernels"),
        "tenferro-cpu must depend on the internal CPU kernel crate"
    );
    assert!(
        !manifest_dir.join("src/elementwise.rs").exists(),
        "elementwise kernels must not move back into the public tenferro-cpu crate"
    );
    assert!(
        !manifest_dir.join("src/buffer_pool.rs").exists(),
        "buffer pool implementation must not move back into the public tenferro-cpu crate"
    );

    let internal_dir = manifest_dir
        .parent()
        .expect("crate has workspace crates directory")
        .join("tenferro-internal-cpu-kernels");
    assert!(
        internal_dir.join("src/elementwise.rs").exists(),
        "internal CPU kernel crate must own elementwise kernels"
    );
    assert!(
        internal_dir
            .join("../tenferro-cpu-basic/src/buffer_pool.rs")
            .exists(),
        "basic CPU crate must own the shared buffer pool"
    );
    assert!(
        internal_dir
            .join("../tenferro-cpu-fused/src/lib.rs")
            .exists(),
        "fused CPU crate must own fused elementwise kernels"
    );
}

#[path = "tests/cpu_tests/backend_misc.rs"]
mod backend_misc;
#[path = "tests/cpu_tests/basic_ops.rs"]
mod basic_ops;
#[path = "tests/cpu_tests/blas1.rs"]
mod blas1;
#[path = "tests/cpu_tests/capability.rs"]
mod capability;
mod cast_matrix_coverage_tests;
#[path = "tests/cpu_tests/context.rs"]
mod context;
#[path = "tests/cpu_tests/dot_structural_analytic.rs"]
mod dot_structural_analytic;
#[path = "tests/cpu_tests/elementwise_reduction_helpers.rs"]
mod elementwise_reduction_helpers;
#[path = "tests/cpu_tests/erf.rs"]
mod erf;
#[path = "tests/cpu_tests/indexing.rs"]
mod indexing;
#[path = "tests/cpu_indexing_coverage_tests.rs"]
mod indexing_coverage;
#[path = "tests/cpu_tests/managed_copy_boundaries.rs"]
mod managed_copy_boundaries;
#[path = "tests/cpu_tests/reshape_tensor_pool.rs"]
mod reshape_tensor_pool;
#[cfg(feature = "native")]
/// The analytic read view carries one arm per preset scalar, and the read entry for `pow` takes it for both
/// operands. The owned-tensor tests reach the floating arms only, so this drives every dtype, including the
/// boolean arm, whose analytic view is a unit marker and is therefore refused.
#[test]
fn analytic_read_view_covers_every_preset_scalar() {
    let mut buffers = crate::buffer_pool::BufferPool::new();

    macro_rules! pow_output {
        ($scalar:ty, $values:expr) => {{
            let values: Vec<$scalar> = $values;
            let lhs: Tensor =
                Tensor::from_vec_col_major(vec![values.len()], values.clone()).unwrap();
            let rhs: Tensor = Tensor::from_vec_col_major(vec![values.len()], values).unwrap();
            crate::analytic::pow_read_with_pool(
                &mut buffers,
                tenferro_tensor::TensorRead::from_tensor(&lhs),
                tenferro_tensor::TensorRead::from_tensor(&rhs),
            )
            .expect("pow admits this preset scalar")
        }};
    }

    macro_rules! pow_case_exact {
        ($scalar:ty, $values:expr, $expected:expr) => {{
            let output = pow_output!($scalar, $values);
            assert_eq!(output.as_slice::<$scalar>().unwrap(), $expected.as_slice());
        }};
    }

    // The complex arm computes through `exp`/`log`, so its result is exact only to the
    // precision of that pair; compare within a tolerance instead of demanding bit equality.
    macro_rules! pow_case_close {
        ($scalar:ty, $values:expr, $expected:expr) => {{
            let output = pow_output!($scalar, $values);
            let actual = output.as_slice::<$scalar>().unwrap();
            let expected: Vec<$scalar> = $expected;
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.iter().zip(&expected) {
                assert!(
                    (a.re - e.re).abs() < 1e-5 && (a.im - e.im).abs() < 1e-5,
                    "pow({}) = {:?}, expected {:?}",
                    stringify!($scalar),
                    actual,
                    expected
                );
            }
        }};
    }

    pow_case_exact!(f32, vec![2.0_f32, 3.0], vec![4.0_f32, 27.0]);
    pow_case_exact!(f64, vec![2.0_f64, 3.0], vec![4.0_f64, 27.0]);
    pow_case_exact!(i32, vec![2_i32, 3], vec![4_i32, 27]);
    pow_case_exact!(i64, vec![2_i64, 3], vec![4_i64, 27]);
    pow_case_close!(
        Complex32,
        vec![Complex32::new(2.0, 0.0), Complex32::new(3.0, 0.0)],
        vec![Complex32::new(4.0, 0.0), Complex32::new(27.0, 0.0)]
    );
    pow_case_close!(
        Complex64,
        vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)],
        vec![Complex64::new(4.0, 0.0), Complex64::new(27.0, 0.0)]
    );

    // The boolean analytic view is a unit marker, so `pow` has no arm for it and reports a
    // dtype mismatch rather than silently returning a value.
    let bools: Tensor = Tensor::from_vec_col_major(vec![2], vec![false, true]).unwrap();
    let error = crate::analytic::pow_read_with_pool(
        &mut buffers,
        tenferro_tensor::TensorRead::from_tensor(&bools),
        tenferro_tensor::TensorRead::from_tensor(&bools),
    )
    .expect_err("pow refuses the boolean analytic view");
    assert!(matches!(
        error,
        tenferro_tensor::Error::Validation {
            op: "pow",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        }
    ));
}

/// Shared fixture: run one large native map through the session's parallelism.
#[cfg(test)]
pub(crate) mod native_participants {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread::ThreadId;
    use std::time::Duration;

    const LARGE_NATIVE_TEST_LEN: usize = 1 << 17;

    #[derive(Default)]
    pub(crate) struct NativeParticipants {
        active: AtomicUsize,
        max_active: AtomicUsize,
        thread_ids: Mutex<Vec<ThreadId>>,
        require_two: bool,
        rendezvous_released: AtomicBool,
        rendezvous_lock: Mutex<()>,
        rendezvous: Condvar,
    }

    impl NativeParticipants {
        fn requiring_two() -> Self {
            Self {
                require_two: true,
                ..Self::default()
            }
        }

        fn observe(&self) {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            {
                let id = std::thread::current().id();
                let mut ids = self.thread_ids.lock().unwrap();
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            if self.require_two && !self.rendezvous_released.load(Ordering::Acquire) {
                let guard = self.rendezvous_lock.lock().unwrap();
                if active >= 2 {
                    self.rendezvous_released.store(true, Ordering::Release);
                    self.rendezvous.notify_all();
                } else if !self.rendezvous_released.load(Ordering::Acquire) {
                    let _guard = self
                        .rendezvous
                        .wait_timeout_while(guard, Duration::from_secs(2), |_| {
                            !self.rendezvous_released.load(Ordering::Acquire)
                        })
                        .unwrap();
                }
            }
            for _ in 0..32 {
                std::hint::spin_loop();
            }
            self.active.fetch_sub(1, Ordering::SeqCst);
        }

        pub(crate) fn max_active(&self) -> usize {
            self.max_active.load(Ordering::SeqCst)
        }

        pub(crate) fn thread_count(&self) -> usize {
            self.thread_ids.lock().unwrap().len()
        }

        pub(crate) fn thread_ids(&self) -> Vec<ThreadId> {
            self.thread_ids.lock().unwrap().clone()
        }
    }

    /// Run one native strided map of `len` elements and report which threads
    /// executed it together with the mapped values.
    pub(crate) fn run_native_map_of_len(
        len: usize,
        require_two: bool,
    ) -> (Arc<NativeParticipants>, Vec<f64>) {
        let source =
            strided_kernel::StridedArray::<f64>::from_fn_col_major(&[len], |index| index[0] as f64);
        let mut destination = strided_kernel::StridedArray::<f64>::col_major(&[len]);
        let participants = Arc::new(if require_two {
            NativeParticipants::requiring_two()
        } else {
            NativeParticipants::default()
        });
        let observed = Arc::clone(&participants);
        strided_kernel::map_into(&mut destination.view_mut(), &source.view(), |value| {
            observed.observe();
            value + 1.0
        })
        .unwrap();
        let output = destination.into_data();
        assert_eq!(output[0], 1.0);
        assert_eq!(output[len - 1], len as f64);
        (participants, output)
    }

    pub(crate) fn run_unscoped_native_map(require_two: bool) -> Arc<NativeParticipants> {
        run_native_map_of_len(LARGE_NATIVE_TEST_LEN, require_two).0
    }
}
