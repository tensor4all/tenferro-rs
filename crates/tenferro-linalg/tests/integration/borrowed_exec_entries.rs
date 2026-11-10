//! The borrowed-`Exec` factorization entries against the engine entries (#2462).
//!
//! The contract on the issue: each entry gives the result the owned-tensor engine entry gives for
//! the same view materialized with `from_vec_col_major`, value bits and placement included; a
//! negative stride is `Unsupported`; an invalid input fails with the same variant and the same
//! downcastable source; and every buffer comes from the caller's pool and no other.

#![cfg(feature = "native")]

use std::error::Error as _;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
use tenferro_linalg::{
    eig_values_with_exec, eigh_values_with_exec, eigh_with_exec, qr_with_exec,
    svd_values_with_exec, svd_with_exec, LinalgScalar, SvdOptions, SvdShape, TensorLinalgExt,
    TensorReadLinalgExt,
};
use tenferro_tensor::{
    ErrorKind, StridedSliceSpec, Tensor, TensorRead, TensorScalar, TensorView, TypedTensor,
    TypedTensorView,
};

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

/// A scalar the fixtures can build from two reals.
trait Sample: LinalgScalar + Copy + std::fmt::Debug {
    fn sample(re: f64, im: f64) -> Self;
}
impl Sample for f32 {
    fn sample(re: f64, _im: f64) -> Self {
        re as f32
    }
}
impl Sample for f64 {
    fn sample(re: f64, _im: f64) -> Self {
        re
    }
}
impl Sample for Complex32 {
    fn sample(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
}
impl Sample for Complex64 {
    fn sample(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

/// A deterministic, well-conditioned compact tensor of `shape`.
fn base<T: Sample>(shape: &[usize]) -> TypedTensor<T> {
    let len: usize = shape.iter().product();
    let data = (0..len)
        .map(|index| {
            let x = index as f64;
            T::sample(1.0 + (x * 0.37).sin() + 0.1 * x, 0.25 * (x * 0.71).cos())
        })
        .collect();
    TypedTensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

/// A Hermitian batch `[n, n, batch]` (real symmetric for a real scalar).
fn hermitian<T: Sample>(n: usize, batch: usize) -> TypedTensor<T> {
    let mut data = Vec::with_capacity(n * n * batch);
    for b in 0..batch {
        for col in 0..n {
            for row in 0..n {
                let (lo, hi) = (row.min(col) as f64, row.max(col) as f64);
                let re = if row == col {
                    3.0 + row as f64 + b as f64
                } else {
                    0.5 / (1.0 + hi - lo)
                };
                let im = match row.cmp(&col) {
                    std::cmp::Ordering::Less => 0.125 * (1.0 + lo),
                    std::cmp::Ordering::Greater => -0.125 * (1.0 + lo),
                    std::cmp::Ordering::Equal => 0.0,
                };
                data.push(T::sample(re, im));
            }
        }
    }
    TypedTensor::from_vec_col_major(vec![n, n, batch], data).unwrap()
}

/// The view's logical elements, materialized as the owned compact tensor the engine is fed.
fn materialize<T: Sample>(view: &TypedTensorView<'_, T>) -> Tensor {
    let shape = view.shape().to_vec();
    let len: usize = shape.iter().product();
    let mut data = Vec::with_capacity(len);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..len {
        data.push(*view.get(&index).unwrap());
        for (axis, value) in index.iter_mut().enumerate() {
            *value += 1;
            if *value < shape[axis] {
                break;
            }
            *value = 0;
        }
    }
    Tensor::from_typed::<T>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

/// Bit equality of two tensors, plus shape and placement.
///
/// `Debug` of a float is its shortest round-trip text, so equal text is equal bits for every
/// non-NaN value, signed zero included.
fn assert_same<S: TensorScalar + std::fmt::Debug>(
    label: &str,
    borrowed: &TypedTensor<S>,
    engine: &Tensor,
) {
    assert_eq!(borrowed.shape(), engine.shape(), "{label}: shape");
    assert_eq!(
        borrowed.placement(),
        engine.placement(),
        "{label}: placement"
    );
    let engine_data = engine
        .as_slice::<S>()
        .unwrap_or_else(|error| panic!("{label}: engine dtype differs: {error}"));
    assert_eq!(
        format!("{:?}", borrowed.host_data().unwrap()),
        format!("{engine_data:?}"),
        "{label}: value bits"
    );
}

/// Run `body` on a fresh serial engine's bundle with a fresh caller pool.
fn borrowed<R>(
    body: impl FnOnce(&mut tcontract_exec::Exec<'_>, &mut BufferPool) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    let mut engine = CpuEngine::serial();
    let mut buffers = BufferPool::new();
    engine.with_exec(|exec| body(exec, &mut buffers))?
}

// ---------------------------------------------------------------------------------------------
// Equality with the engine entry
// ---------------------------------------------------------------------------------------------

/// Every general (non-square-only) factorization of `view`, borrowed against engine.
fn check_general<T: Sample>(label: &str, view: &TypedTensorView<'_, T>)
where
    T::RealScalar: std::fmt::Debug,
{
    let owned = materialize(view);
    let mut engine = CpuEngine::serial();

    for full in [false, true] {
        let (u, s, vt) = borrowed(|exec, buffers| svd_with_exec(exec, buffers, view, full))
            .unwrap_or_else(|error| panic!("{label} svd(full={full}): {error}"));
        let (eu, es, evt) = if full {
            owned
                .svd_with(
                    &SvdOptions::default().with_shape(SvdShape::Full),
                    &mut engine,
                )
                .unwrap()
        } else {
            owned.svd(&mut engine).unwrap()
        };
        let tag = format!("{label} svd(full={full})");
        assert_same(&format!("{tag} U"), &u, &eu);
        assert_same(&format!("{tag} S"), &s, &es);
        assert_same(&format!("{tag} Vh"), &vt, &evt);
    }

    let s = borrowed(|exec, buffers| svd_values_with_exec(exec, buffers, view)).unwrap();
    assert_same(
        &format!("{label} svd_values"),
        &s,
        &owned.svdvals(&mut engine).unwrap(),
    );

    let (q, r) = borrowed(|exec, buffers| qr_with_exec(exec, buffers, view)).unwrap();
    let (eq, er) = owned.qr(&mut engine).unwrap();
    assert_same(&format!("{label} qr Q"), &q, &eq);
    assert_same(&format!("{label} qr R"), &r, &er);
}

/// Every square-only factorization of `view`, borrowed against engine.
fn check_square<T: Sample>(label: &str, view: &TypedTensorView<'_, T>)
where
    T::RealScalar: std::fmt::Debug,
    T::Complex: std::fmt::Debug,
{
    let owned = materialize(view);
    let mut engine = CpuEngine::serial();

    let (values, vectors) = borrowed(|exec, buffers| eigh_with_exec(exec, buffers, view)).unwrap();
    let (evalues, evectors) = owned.eigh(&mut engine).unwrap();
    assert_same(&format!("{label} eigh values"), &values, &evalues);
    assert_same(&format!("{label} eigh vectors"), &vectors, &evectors);

    let values = borrowed(|exec, buffers| eigh_values_with_exec(exec, buffers, view)).unwrap();
    assert_same(
        &format!("{label} eigh_values"),
        &values,
        &owned.eigvalsh(&mut engine).unwrap(),
    );

    let values = borrowed(|exec, buffers| eig_values_with_exec(exec, buffers, view)).unwrap();
    assert_same(
        &format!("{label} eig_values"),
        &values,
        &owned.eigvals(&mut engine).unwrap(),
    );
}

fn equality_for<T: Sample>(dtype: &str)
where
    T::RealScalar: std::fmt::Debug,
    T::Complex: std::fmt::Debug,
{
    // Non-square, compact.
    for shape in [[3, 2], [2, 3]] {
        let a = base::<T>(&shape);
        check_general(&format!("{dtype} {shape:?}"), &a.as_view());
    }

    // Non-square with arbitrary non-negative strides: every other row of a taller matrix,
    // transposed, so neither axis is unit-stride.
    let tall = base::<T>(&[6, 3]);
    let strided = tall
        .as_view()
        .slice_view(&[StridedSliceSpec::new(0, None, 2), StridedSliceSpec::all()])
        .unwrap()
        .transpose_view([1, 0])
        .unwrap();
    assert_eq!(strided.shape(), &[3, 3]);
    check_general(&format!("{dtype} strided"), &strided);

    // Zero dimensions, including the full SVD's identity factors.
    for shape in [[0, 3], [3, 0], [0, 0]] {
        let a = base::<T>(&shape);
        check_general(&format!("{dtype} empty {shape:?}"), &a.as_view());
    }
    let empty = base::<T>(&[0, 0]);
    check_square(&format!("{dtype} empty square"), &empty.as_view());

    // Batched `[m, n, b]` read in place: a `[b, m, n]` tensor permuted to `[m, n, b]`, so the
    // batch axis is the unit-stride one and the matrix axes are not.
    let stacked = base::<T>(&[2, 4, 3]);
    let batched = stacked.as_view().transpose_view([1, 2, 0]).unwrap();
    assert_eq!(batched.shape(), &[4, 3, 2]);
    check_general(&format!("{dtype} batched"), &batched);

    // Batched square: Hermitian, permuted the same way.
    let herm = hermitian::<T>(3, 2);
    check_square(&format!("{dtype} batched square"), &herm.as_view());

    // Square with non-unit strides: the Hermitian batch transposed in its matrix axes.
    let herm_t = herm.as_view().transpose_view([1, 0, 2]).unwrap();
    check_square(&format!("{dtype} strided square"), &herm_t);
}

#[test]
fn each_entry_matches_the_engine_entry_on_value_bits_and_placement() {
    equality_for::<f32>("f32");
    equality_for::<f64>("f64");
    equality_for::<Complex32>("c32");
    equality_for::<Complex64>("c64");
}

fn assert_close(label: &str, gathered: &[f64], direct: &[f64], tol: f64) {
    assert_eq!(gathered.len(), direct.len(), "{label}: length");
    for (index, (g, d)) in gathered.iter().zip(direct).enumerate() {
        assert!(
            (g - d).abs() <= tol * d.abs().max(1.0),
            "{label}[{index}]: {g} vs {d}"
        );
    }
}

#[test]
fn complex_spectra_are_real_typed_and_agree_with_the_values_only_route() {
    // The factor entries gather the spectrum into the real scalar after the provider writes it in
    // `T`; the values-only entries have the provider write the real scalar directly. Equality with
    // the engine cannot see a gather bug (both paths share it), so compare the two routes.
    let a = base::<Complex64>(&[4, 3, 2]);
    let (_, s, _) =
        borrowed(|exec, buffers| svd_with_exec(exec, buffers, &a.as_view(), false)).unwrap();
    let s: TypedTensor<f64> = s;
    let direct =
        borrowed(|exec, buffers| svd_values_with_exec(exec, buffers, &a.as_view())).unwrap();
    assert_eq!(s.shape(), &[3, 2]);
    assert_close(
        "c64 svd spectrum",
        s.host_data().unwrap(),
        direct.host_data().unwrap(),
        1e-12,
    );

    let h = hermitian::<Complex64>(3, 2);
    let (values, vectors) =
        borrowed(|exec, buffers| eigh_with_exec(exec, buffers, &h.as_view())).unwrap();
    let _: &TypedTensor<Complex64> = &vectors;
    let direct =
        borrowed(|exec, buffers| eigh_values_with_exec(exec, buffers, &h.as_view())).unwrap();
    assert_close(
        "c64 eigh spectrum",
        values.host_data().unwrap(),
        direct.host_data().unwrap(),
        1e-12,
    );

    let h32 = hermitian::<Complex32>(3, 1);
    let (values, _) =
        borrowed(|exec, buffers| eigh_with_exec(exec, buffers, &h32.as_view())).unwrap();
    let _: TypedTensor<f32> = values;

    let r = base::<f32>(&[2, 2]);
    let values =
        borrowed(|exec, buffers| eig_values_with_exec(exec, buffers, &r.as_view())).unwrap();
    let _: TypedTensor<Complex32> = values;
}

// ---------------------------------------------------------------------------------------------
// Negative strides
// ---------------------------------------------------------------------------------------------

#[test]
fn a_negative_stride_is_unsupported_for_every_entry() {
    let a = base::<f64>(&[3, 3]);
    let reversed = a
        .as_view()
        .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::all()])
        .unwrap();
    assert!(reversed.strides()[0] < 0);

    // The same rule has to hold when there is nothing to read: a zero-length axis taken from the
    // reversed view keeps the negative stride, and the zero-dimension shortcut must not return
    // success before the layout is validated (#2505). `[3, 0]` is the matrix shape the factor
    // entries read, `[0, 0]` the square one the eigensolvers require.
    let empty_wide = reversed
        .slice_view(&[
            StridedSliceSpec::all(),
            StridedSliceSpec::new(0, Some(0), 1),
        ])
        .unwrap();
    let empty_square = reversed
        .slice_view(&[
            StridedSliceSpec::new(0, Some(0), 1),
            StridedSliceSpec::new(0, Some(0), 1),
        ])
        .unwrap();
    assert_eq!(empty_wide.shape(), &[3, 0]);
    assert_eq!(empty_square.shape(), &[0, 0]);

    let errors = [
        (
            "svd",
            borrowed(|e, b| svd_with_exec(e, b, &reversed, false)).map(|_| ()),
        ),
        (
            "svd_full",
            borrowed(|e, b| svd_with_exec(e, b, &reversed, true)).map(|_| ()),
        ),
        (
            "svd_values",
            borrowed(|e, b| svd_values_with_exec(e, b, &reversed)).map(|_| ()),
        ),
        (
            "qr",
            borrowed(|e, b| qr_with_exec(e, b, &reversed)).map(|_| ()),
        ),
        (
            "eigh",
            borrowed(|e, b| eigh_with_exec(e, b, &reversed)).map(|_| ()),
        ),
        (
            "eigh_values",
            borrowed(|e, b| eigh_values_with_exec(e, b, &reversed)).map(|_| ()),
        ),
        (
            "eig_values",
            borrowed(|e, b| eig_values_with_exec(e, b, &reversed)).map(|_| ()),
        ),
    ];
    let mut non_empty = Vec::new();
    for (op, result) in errors {
        let error = result.expect_err(op);
        assert_eq!(error.kind(), ErrorKind::Unsupported, "{op}: {error}");
        assert_eq!(
            error.to_string(),
            format!(
                "{op}: unsupported operation: a negative stride is not a supported linalg input layout"
            ),
            "{op}"
        );
        non_empty.push(error.to_string());
    }

    let empty_errors = [
        (
            "svd",
            borrowed(|e, b| svd_with_exec(e, b, &empty_wide, false)).map(|_| ()),
        ),
        (
            "svd_full",
            borrowed(|e, b| svd_with_exec(e, b, &empty_wide, true)).map(|_| ()),
        ),
        (
            "svd_values",
            borrowed(|e, b| svd_values_with_exec(e, b, &empty_wide)).map(|_| ()),
        ),
        (
            "qr",
            borrowed(|e, b| qr_with_exec(e, b, &empty_wide)).map(|_| ()),
        ),
        (
            "eigh",
            borrowed(|e, b| eigh_with_exec(e, b, &empty_square)).map(|_| ()),
        ),
        (
            "eigh_values",
            borrowed(|e, b| eigh_values_with_exec(e, b, &empty_square)).map(|_| ()),
        ),
        (
            "eig_values",
            borrowed(|e, b| eig_values_with_exec(e, b, &empty_square)).map(|_| ()),
        ),
    ];
    // Same variant, same message as the non-empty operand: the empty operand's shape is a result
    // the shortcut may answer, but its layout is not, so it cannot be answered with outputs.
    for ((op, result), message) in empty_errors.into_iter().zip(&non_empty) {
        let error = result.expect_err(op);
        assert_eq!(
            error.kind(),
            ErrorKind::Unsupported,
            "{op} on an empty operand: {error}"
        );
        assert_eq!(&error.to_string(), message, "{op} on an empty operand");
    }

    // No engine entry refuses this view: the `*_read` entries gather a negative stride by design
    // (contract point 3), which is exactly the rule the borrowed entries do not promise.
    let mut engine = CpuEngine::serial();
    let read = || TensorRead::from_view(TensorView::F64(reversed.clone()));
    assert!(read().svd_read(&mut engine).is_ok());
    assert!(read().eigvals_read(&mut engine).is_ok());
}

// ---------------------------------------------------------------------------------------------
// Invalid inputs: same variant, same downcastable source
// ---------------------------------------------------------------------------------------------

/// Compare two errors on kind, rendered variant and the downcast linalg payload.
fn assert_same_error(
    label: &str,
    borrowed: tenferro_tensor::Error,
    engine: tenferro_tensor::Error,
) {
    assert_eq!(borrowed.kind(), engine.kind(), "{label}: kind");
    assert_eq!(
        format!("{borrowed:?}"),
        format!("{engine:?}"),
        "{label}: variant"
    );
    let payload = |error: &tenferro_tensor::Error| {
        error
            .source()
            .and_then(|source| source.downcast_ref::<tenferro_linalg::Error>())
            .map(|payload| format!("{payload:?}"))
    };
    assert_eq!(payload(&borrowed), payload(&engine), "{label}: source");
}

#[test]
fn invalid_inputs_fail_with_the_same_variant_and_source_as_the_engine() {
    let mut engine = CpuEngine::serial();

    // Rank 1: every entry refuses the rank.
    let vector = base::<f64>(&[3]);
    let owned = materialize(&vector.as_view());
    let view = vector.as_view();
    assert_same_error(
        "rank-1 svd",
        borrowed(|e, b| svd_with_exec(e, b, &view, false)).unwrap_err(),
        owned.svd(&mut engine).unwrap_err(),
    );
    assert_same_error(
        "rank-1 svd_values",
        borrowed(|e, b| svd_values_with_exec(e, b, &view)).unwrap_err(),
        owned.svdvals(&mut engine).unwrap_err(),
    );
    assert_same_error(
        "rank-1 qr",
        borrowed(|e, b| qr_with_exec(e, b, &view)).unwrap_err(),
        owned.qr(&mut engine).unwrap_err(),
    );

    // Non-square: the eigensolvers refuse the shape.
    let rect = base::<f64>(&[3, 2]);
    let owned = materialize(&rect.as_view());
    let view = rect.as_view();
    assert_same_error(
        "non-square eigh",
        borrowed(|e, b| eigh_with_exec(e, b, &view)).unwrap_err(),
        owned.eigh(&mut engine).unwrap_err(),
    );
    assert_same_error(
        "non-square eigh_values",
        borrowed(|e, b| eigh_values_with_exec(e, b, &view)).unwrap_err(),
        owned.eigvalsh(&mut engine).unwrap_err(),
    );
    assert_same_error(
        "non-square eig_values",
        borrowed(|e, b| eig_values_with_exec(e, b, &view)).unwrap_err(),
        owned.eigvals(&mut engine).unwrap_err(),
    );

    // Non-finite input: the provider's own failure, mapped by `map_error`, wherever the provider
    // reports one. Both paths must agree on success or on the exact failure.
    let mut data = base::<f64>(&[3, 3]).host_data().unwrap().to_vec();
    data[4] = f64::NAN;
    let nan = TypedTensor::<f64>::from_vec_col_major(vec![3, 3], data).unwrap();
    let owned = materialize(&nan.as_view());
    let view = nan.as_view();
    type Outcome = Result<(), tenferro_tensor::Error>;
    let cases: [(&str, Outcome, Outcome); 6] = [
        (
            "nan svd",
            borrowed(|e, b| svd_with_exec(e, b, &view, false)).map(|_| ()),
            owned.svd(&mut engine).map(|_| ()),
        ),
        (
            "nan svd_values",
            borrowed(|e, b| svd_values_with_exec(e, b, &view)).map(|_| ()),
            owned.svdvals(&mut engine).map(|_| ()),
        ),
        (
            "nan qr",
            borrowed(|e, b| qr_with_exec(e, b, &view)).map(|_| ()),
            owned.qr(&mut engine).map(|_| ()),
        ),
        (
            "nan eigh",
            borrowed(|e, b| eigh_with_exec(e, b, &view)).map(|_| ()),
            owned.eigh(&mut engine).map(|_| ()),
        ),
        (
            "nan eigh_values",
            borrowed(|e, b| eigh_values_with_exec(e, b, &view)).map(|_| ()),
            owned.eigvalsh(&mut engine).map(|_| ()),
        ),
        (
            "nan eig_values",
            borrowed(|e, b| eig_values_with_exec(e, b, &view)).map(|_| ()),
            owned.eigvals(&mut engine).map(|_| ()),
        ),
    ];
    let mut provider_failures = 0;
    for (label, borrowed, engine) in cases {
        match (borrowed, engine) {
            (Ok(()), Ok(())) => {}
            (Err(borrowed), Err(engine)) => {
                provider_failures += 1;
                assert_same_error(label, borrowed, engine);
            }
            (borrowed, engine) => {
                panic!("{label}: borrowed {borrowed:?} but engine {engine:?}")
            }
        }
    }
    assert!(
        provider_failures > 0,
        "the non-finite fixture must reach at least one provider failure"
    );
}

// ---------------------------------------------------------------------------------------------
// Storage: the caller's pool and no other
// ---------------------------------------------------------------------------------------------

/// One call of every entry on `exec`/`buffers`, dropping every output.
fn run_all(exec: &mut tcontract_exec::Exec<'_>, buffers: &mut BufferPool) {
    let a = base::<Complex64>(&[4, 3, 2]);
    let h = hermitian::<Complex64>(3, 2);
    let (a, h) = (a.as_view(), h.as_view());
    drop(svd_with_exec(exec, buffers, &a, false).unwrap());
    drop(svd_with_exec(exec, buffers, &a, true).unwrap());
    drop(svd_values_with_exec(exec, buffers, &a).unwrap());
    drop(qr_with_exec(exec, buffers, &a).unwrap());
    drop(eigh_with_exec(exec, buffers, &h).unwrap());
    drop(eigh_values_with_exec(exec, buffers, &h).unwrap());
    drop(eig_values_with_exec(exec, buffers, &h).unwrap());
}

#[test]
fn every_buffer_comes_from_the_callers_pool_and_reaches_a_steady_state() {
    let mut engine = CpuEngine::serial();
    let engine_pool_before = engine.buffer_pool_stats();
    let mut lane_pool = BufferPool::new();
    assert!(lane_pool.is_empty());

    engine
        .with_exec(|exec| run_all(exec, &mut lane_pool))
        .unwrap();
    // The outputs and the transient spectra returned to the lane's pool, not the engine's.
    let warm = lane_pool.stats();
    assert!(
        warm.buffers > 0,
        "outputs must recycle into the caller's pool"
    );
    assert_eq!(
        engine.buffer_pool_stats(),
        engine_pool_before,
        "the engine's own pool must not be touched"
    );

    // A second round reuses what the first returned: the pool does not grow.
    engine
        .with_exec(|exec| run_all(exec, &mut lane_pool))
        .unwrap();
    assert_eq!(
        lane_pool.stats(),
        warm,
        "steady state: no growth beyond the pool"
    );
    assert_eq!(engine.buffer_pool_stats(), engine_pool_before);
}

#[test]
fn two_execs_with_two_pools_do_not_share_storage() {
    let mut first_engine = CpuEngine::serial();
    let mut second_engine = CpuEngine::serial();
    let mut first_pool = BufferPool::new();
    let mut second_pool = BufferPool::new();
    let a = base::<f64>(&[4, 4]);

    let first = first_engine
        .with_exec(|exec| svd_with_exec(exec, &mut first_pool, &a.as_view(), false))
        .unwrap()
        .unwrap();
    let second = second_engine
        .with_exec(|exec| svd_with_exec(exec, &mut second_pool, &a.as_view(), false))
        .unwrap()
        .unwrap();

    // Same values, distinct storage.
    assert_eq!(first.1.host_data().unwrap(), second.1.host_data().unwrap());
    for (x, y) in [(&first.0, &second.0), (&first.2, &second.2)] {
        assert_ne!(
            x.host_data().unwrap().as_ptr(),
            y.host_data().unwrap().as_ptr(),
            "two pools handed out the same storage"
        );
    }
    assert_ne!(
        first.1.host_data().unwrap().as_ptr(),
        second.1.host_data().unwrap().as_ptr()
    );

    // Dropping the first lane's outputs fills only the first pool.
    let second_before = second_pool.stats();
    drop(first);
    assert!(first_pool.stats().buffers > 0);
    assert_eq!(second_pool.stats(), second_before);

    // And the second lane's outputs return only to the second pool.
    let first_before = first_pool.stats();
    drop(second);
    assert_eq!(first_pool.stats(), first_before);
    assert!(second_pool.stats().buffers > second_before.buffers);
}
