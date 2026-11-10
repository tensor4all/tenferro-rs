//! The factorization entries inside a lane of `CpuEngine::with_lanes` against the
//! `*_with_exec` entries (#2552 S2b, the P7 parity tests).
//!
//! A lane is an engine over a share of the thread budget and a caller-owned pool. The ordinary
//! entries (`svd_read` and its siblings, and `TypedTensorLinalgExt`, which goes through the read
//! path) run on it and must give bit-identical results to the borrowed entries, fail with the
//! same error, and draw every buffer from the lane's pool. Every other linalg operation is
//! `Unsupported` on a lane.

#![cfg(feature = "native")]

use std::error::Error as _;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
use tenferro_linalg::{
    eig_values_with_exec, eigh_values_with_exec, eigh_with_exec, qr_with_exec,
    svd_values_with_exec, svd_with_exec, EighGauge, EighOptions, LinalgScalar, SvdGauge,
    SvdOptions, TensorLinalgExt, TensorReadLinalgExt, TypedTensorLinalgExt,
};
use tenferro_tensor::{
    DType, Engine, ErrorKind, StridedSliceSpec, Tensor, TensorRead, TensorScalar, TypedTensor,
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

/// Bit equality of a typed borrowed-entry result and a lane tensor, plus shape and placement.
///
/// `Debug` of a float is its shortest round-trip text, so equal text is equal bits for every
/// non-NaN value, signed zero included.
fn assert_same<S: TensorScalar + std::fmt::Debug>(
    label: &str,
    free: &TypedTensor<S>,
    lane: &Tensor,
) {
    assert_eq!(free.shape(), lane.shape(), "{label}: shape");
    assert_eq!(free.placement(), lane.placement(), "{label}: placement");
    let lane_data = lane
        .as_slice::<S>()
        .unwrap_or_else(|error| panic!("{label}: lane dtype differs: {error}"));
    assert_eq!(
        format!("{:?}", free.host_data().unwrap()),
        format!("{lane_data:?}"),
        "{label}: value bits"
    );
}

/// The element text of an erased tensor of a linalg dtype (`Debug` of a float is its shortest
/// round-trip text, so equal text is equal bits for every non-NaN value).
fn element_text(tensor: &Tensor) -> String {
    match tensor.dtype() {
        DType::F32 => format!("{:?}", tensor.as_slice::<f32>().unwrap()),
        DType::F64 => format!("{:?}", tensor.as_slice::<f64>().unwrap()),
        DType::C32 => format!("{:?}", tensor.as_slice::<Complex32>().unwrap()),
        DType::C64 => format!("{:?}", tensor.as_slice::<Complex64>().unwrap()),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// Bit equality of two erased tensors: shape, placement, dtype and elements.
fn assert_same_tensors(label: &str, left: &Tensor, right: &Tensor) {
    assert_eq!(left.shape(), right.shape(), "{label}: shape");
    assert_eq!(left.placement(), right.placement(), "{label}: placement");
    assert_eq!(left.dtype(), right.dtype(), "{label}: dtype");
    assert_eq!(element_text(left), element_text(right), "{label}: bits");
}

/// An owned erased copy of a typed tensor.
fn owned_of<T: Sample>(typed: &TypedTensor<T>) -> Tensor {
    Tensor::from_typed::<T>(
        TypedTensor::from_vec_col_major(
            typed.shape().to_vec(),
            typed.host_data().unwrap().to_vec(),
        )
        .unwrap(),
    )
}

/// The erased read of a typed view, tied to the same borrow.
fn read_of<'a, T: Sample>(view: &TypedTensorView<'a, T>) -> TensorRead<'a> {
    TensorRead::from_view(T::tensor_view(view.clone()))
}

/// Run `body` on one lane of a fresh serial engine with a fresh lane pool.
fn on_lane<R: Send>(body: impl Fn(&mut dyn Engine) -> R + Sync) -> R {
    let mut engine = CpuEngine::serial();
    let mut pools = [BufferPool::new()];
    let mut out = engine
        .with_lanes(&[1], &mut pools, |_, lane| body(lane))
        .unwrap();
    out.pop().unwrap()
}

/// Run `body` on a fresh serial engine's bundle with a fresh caller pool (the `*_with_exec` way).
fn borrowed<R>(
    body: impl FnOnce(&mut tcontract_exec::Exec<'_>, &mut BufferPool) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    let mut engine = CpuEngine::serial();
    let mut buffers = BufferPool::new();
    engine.with_exec(|exec| body(exec, &mut buffers))?
}

// ---------------------------------------------------------------------------------------------
// Result parity with the `*_with_exec` entries
// ---------------------------------------------------------------------------------------------

fn check_general<T: Sample>(label: &str, view: &TypedTensorView<'_, T>)
where
    T::RealScalar: std::fmt::Debug,
{
    for full in [false, true] {
        let (u, s, vh) = borrowed(|e, b| svd_with_exec(e, b, view, full)).unwrap();
        let (lu, ls, lvh) = on_lane(|lane| {
            if full {
                read_of(view).svd_full_read(lane)
            } else {
                read_of(view).svd_read(lane)
            }
            .unwrap()
        });
        let tag = format!("{label} svd(full={full})");
        assert_same(&format!("{tag} U"), &u, &lu);
        assert_same(&format!("{tag} S"), &s, &ls);
        assert_same(&format!("{tag} Vh"), &vh, &lvh);
    }

    let s = borrowed(|e, b| svd_values_with_exec(e, b, view)).unwrap();
    let lane = on_lane(|lane| read_of(view).svdvals_read(lane).unwrap());
    assert_same(&format!("{label} svdvals"), &s, &lane);

    let (q, r) = borrowed(|e, b| qr_with_exec(e, b, view)).unwrap();
    let (lq, lr) = on_lane(|lane| read_of(view).qr_read(lane).unwrap());
    assert_same(&format!("{label} qr Q"), &q, &lq);
    assert_same(&format!("{label} qr R"), &r, &lr);
}

fn check_square<T: Sample>(label: &str, view: &TypedTensorView<'_, T>)
where
    T::RealScalar: std::fmt::Debug,
    T::Complex: std::fmt::Debug,
{
    let (values, vectors) = borrowed(|e, b| eigh_with_exec(e, b, view)).unwrap();
    let (lvalues, lvectors) = on_lane(|lane| read_of(view).eigh_read(lane).unwrap());
    assert_same(&format!("{label} eigh values"), &values, &lvalues);
    assert_same(&format!("{label} eigh vectors"), &vectors, &lvectors);

    let values = borrowed(|e, b| eigh_values_with_exec(e, b, view)).unwrap();
    let lane = on_lane(|lane| read_of(view).eigvalsh_read(lane).unwrap());
    assert_same(&format!("{label} eigvalsh"), &values, &lane);

    let values = borrowed(|e, b| eig_values_with_exec(e, b, view)).unwrap();
    let lane = on_lane(|lane| read_of(view).eigvals_read(lane).unwrap());
    assert_same(&format!("{label} eigvals"), &values, &lane);
}

fn parity_for<T: Sample>(dtype: &str)
where
    T::RealScalar: std::fmt::Debug,
    T::Complex: std::fmt::Debug,
{
    for shape in [[3, 2], [2, 3]] {
        let a = base::<T>(&shape);
        check_general(&format!("{dtype} {shape:?}"), &a.as_view());
    }
    let tall = base::<T>(&[6, 3]);
    let strided = tall
        .as_view()
        .slice_view(&[StridedSliceSpec::new(0, None, 2), StridedSliceSpec::all()])
        .unwrap()
        .transpose_view([1, 0])
        .unwrap();
    check_general(&format!("{dtype} strided"), &strided);
    for shape in [[0, 3], [3, 0], [0, 0]] {
        let a = base::<T>(&shape);
        check_general(&format!("{dtype} empty {shape:?}"), &a.as_view());
    }
    let stacked = base::<T>(&[2, 4, 3]);
    let batched = stacked.as_view().transpose_view([1, 2, 0]).unwrap();
    check_general(&format!("{dtype} batched"), &batched);
    let herm = hermitian::<T>(3, 2);
    check_square(&format!("{dtype} batched square"), &herm.as_view());
    let herm_t = herm.as_view().transpose_view([1, 0, 2]).unwrap();
    check_square(&format!("{dtype} strided square"), &herm_t);
    let empty = base::<T>(&[0, 0]);
    check_square(&format!("{dtype} empty square"), &empty.as_view());
}

#[test]
fn lane_results_are_bit_identical_to_the_with_exec_entries_f64() {
    parity_for::<f64>("f64");
}

#[test]
fn lane_results_are_bit_identical_to_the_with_exec_entries_complex64() {
    parity_for::<Complex64>("c64");
}

#[test]
fn lane_results_are_bit_identical_to_the_with_exec_entries_single_precision() {
    parity_for::<f32>("f32");
    parity_for::<Complex32>("c32");
}

#[test]
fn every_lane_entry_spelling_gives_the_same_bits() {
    // Owned `Tensor` methods, the read methods and the typed trait are one route on a lane.
    let typed = base::<Complex64>(&[5, 4]);
    let owned = owned_of(&typed);
    let herm = hermitian::<Complex64>(4, 1);
    let owned_herm = owned_of(&herm);
    on_lane(|lane| {
        let (u, s, vh) = borrowed(|e, b| svd_with_exec(e, b, &typed.as_view(), false)).unwrap();
        let from_owned = owned.svd(lane).unwrap();
        assert_same("owned svd U", &u, &from_owned.0);
        assert_same("owned svd S", &s, &from_owned.1);
        assert_same("owned svd Vh", &vh, &from_owned.2);
        let from_typed = TypedTensorLinalgExt::svd(&typed, lane).unwrap();
        assert_same("typed svd U", &u, &Tensor::from_typed(from_typed.0));
        assert_same("typed svd S", &s, &Tensor::from_typed(from_typed.1));
        assert_same("typed svd Vh", &vh, &Tensor::from_typed(from_typed.2));
        let full = owned.svd_full(lane).unwrap();
        assert_eq!(full.0.shape(), &[5, 5]);
        assert_eq!(owned.svdvals(lane).unwrap().shape(), &[4]);
        assert_eq!(owned.qr(lane).unwrap().0.shape(), &[5, 4]);
        assert_eq!(owned_herm.eigh(lane).unwrap().1.shape(), &[4, 4, 1]);
        assert_eq!(owned_herm.eigvalsh(lane).unwrap().shape(), &[4, 1]);
        assert_eq!(owned_herm.eigvals(lane).unwrap().shape(), &[4, 1]);
        let typed_values = TypedTensorLinalgExt::eigvalsh(&herm, lane).unwrap();
        assert_eq!(typed_values.shape(), &[4, 1]);
    });
}

// ---------------------------------------------------------------------------------------------
// Error parity
// ---------------------------------------------------------------------------------------------

fn same_error(label: &str, lane: tenferro_tensor::Error, free: tenferro_tensor::Error) {
    assert_eq!(lane.kind(), free.kind(), "{label}: kind");
    assert_eq!(format!("{lane:?}"), format!("{free:?}"), "{label}: variant");
    let payload = |error: &tenferro_tensor::Error| {
        error
            .source()
            .and_then(|source| source.downcast_ref::<tenferro_linalg::Error>())
            .map(|payload| format!("{payload:?}"))
    };
    assert_eq!(payload(&lane), payload(&free), "{label}: source");
}

#[test]
fn lane_errors_have_the_same_kind_variant_and_source_as_the_with_exec_entries() {
    let vector = base::<f64>(&[3]);
    let v = vector.as_view();
    same_error(
        "rank-1 svd",
        on_lane(|lane| read_of(&v).svd_read(lane).map(|_| ()).unwrap_err()),
        borrowed(|e, b| svd_with_exec(e, b, &v, false)).unwrap_err(),
    );
    same_error(
        "rank-1 qr",
        on_lane(|lane| read_of(&v).qr_read(lane).map(|_| ()).unwrap_err()),
        borrowed(|e, b| qr_with_exec(e, b, &v)).unwrap_err(),
    );
    let rect = base::<f64>(&[3, 2]);
    let r = rect.as_view();
    same_error(
        "non-square eigh",
        on_lane(|lane| read_of(&r).eigh_read(lane).map(|_| ()).unwrap_err()),
        borrowed(|e, b| eigh_with_exec(e, b, &r)).unwrap_err(),
    );
    same_error(
        "non-square eigvals",
        on_lane(|lane| read_of(&r).eigvals_read(lane).map(|_| ()).unwrap_err()),
        borrowed(|e, b| eig_values_with_exec(e, b, &r)).unwrap_err(),
    );

    // Non-finite input: the provider's own mapped failure, wherever it reports one.
    let mut data = base::<f64>(&[3, 3]).host_data().unwrap().to_vec();
    data[4] = f64::NAN;
    let nan = TypedTensor::<f64>::from_vec_col_major(vec![3, 3], data).unwrap();
    let n = nan.as_view();
    let mut failures = 0;
    let cases = [
        (
            "nan svd",
            on_lane(|lane| read_of(&n).svd_read(lane).map(|_| ())),
            borrowed(|e, b| svd_with_exec(e, b, &n, false)).map(|_| ()),
        ),
        (
            "nan qr",
            on_lane(|lane| read_of(&n).qr_read(lane).map(|_| ())),
            borrowed(|e, b| qr_with_exec(e, b, &n)).map(|_| ()),
        ),
        (
            "nan eigh",
            on_lane(|lane| read_of(&n).eigh_read(lane).map(|_| ())),
            borrowed(|e, b| eigh_with_exec(e, b, &n)).map(|_| ()),
        ),
        (
            "nan eigvals",
            on_lane(|lane| read_of(&n).eigvals_read(lane).map(|_| ())),
            borrowed(|e, b| eig_values_with_exec(e, b, &n)).map(|_| ()),
        ),
    ];
    for (label, lane, free) in cases {
        match (lane, free) {
            (Ok(()), Ok(())) => {}
            (Err(lane), Err(free)) => {
                failures += 1;
                same_error(label, lane, free);
            }
            (lane, free) => panic!("{label}: lane {lane:?} but free {free:?}"),
        }
    }
    assert!(
        failures > 0,
        "the NaN fixture must reach a provider failure"
    );
}

// ---------------------------------------------------------------------------------------------
// Operations that do not run in a lane
// ---------------------------------------------------------------------------------------------

#[test]
fn every_other_operation_on_a_lane_is_unsupported() {
    let a = Tensor::from_typed::<f64>(hermitian::<f64>(3, 1));
    let square = Tensor::from_vec_col_major(vec![3, 3], (0..9).map(f64::from).collect()).unwrap();
    let typed = hermitian::<f64>(3, 1);
    on_lane(|lane| {
        let refused: Vec<(&str, tenferro_tensor::Error)> = vec![
            ("solve", square.solve(&square, lane).unwrap_err()),
            ("cholesky", square.cholesky(lane).unwrap_err()),
            ("lu", square.lu(lane).map(|_| ()).unwrap_err()),
            ("eig", square.eig(lane).map(|_| ()).unwrap_err()),
            ("inv", square.inv(lane).unwrap_err()),
            ("det", square.det(lane).unwrap_err()),
            ("pinv", square.pinv(lane).unwrap_err()),
            (
                "qr_with_options",
                a.qr_with_options(Default::default(), lane)
                    .map(|_| ())
                    .unwrap_err(),
            ),
            (
                "typed solve",
                typed.solve(&typed, lane).map(|_| ()).unwrap_err(),
            ),
            (
                "engine add",
                lane.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
                    .map(|_| ())
                    .unwrap_err(),
            ),
        ];
        for (op, error) in refused {
            assert_eq!(error.kind(), ErrorKind::Unsupported, "{op}: {error}");
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Layout (F1): a negative stride is gathered from the lane's pool
// ---------------------------------------------------------------------------------------------

#[test]
fn a_negative_stride_is_gathered_into_the_lane_pool_and_matches_the_engine_route() {
    let a = base::<f64>(&[4, 3]);
    let reversed = a
        .as_view()
        .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::all()])
        .unwrap();
    assert!(reversed.strides()[0] < 0);
    let compact = base::<f64>(&[4, 3]);
    let empty = reversed
        .slice_view(&[
            StridedSliceSpec::all(),
            StridedSliceSpec::new(0, Some(0), 1),
        ])
        .unwrap();
    assert_eq!(empty.shape(), &[4, 0]);

    // Cold lane pools, outputs dropped inside the lane: whatever the pool holds afterwards was
    // taken from it during the call. The reversed input needs one more buffer (the gather
    // target, handed back when the call ends) than the compact input of the same shape.
    let mut engine = CpuEngine::serial();
    let engine_pool_before = engine.buffer_pool_stats();
    let mut held_after = |view: &TypedTensorView<'_, f64>| {
        let mut pools = [BufferPool::new()];
        engine
            .with_lanes(&[1], &mut pools, |_, lane| {
                drop(read_of(view).qr_read(lane).unwrap());
            })
            .unwrap();
        pools[0].stats()
    };
    let compact_stats = held_after(&compact.as_view());
    let reversed_stats = held_after(&reversed);
    assert!(
        reversed_stats.buffers > compact_stats.buffers
            && reversed_stats.capacity_bytes > compact_stats.capacity_bytes,
        "the gather buffer must come from and return to the lane pool: \
         compact {compact_stats:?}, reversed {reversed_stats:?}"
    );
    assert_eq!(
        engine.buffer_pool_stats(),
        engine_pool_before,
        "the engine's own pool is untouched"
    );
    let mut pools = [BufferPool::new()];
    let qr_bits = engine
        .with_lanes(&[1], &mut pools, |_, lane| {
            read_of(&reversed).qr_read(lane).unwrap()
        })
        .unwrap()
        .pop()
        .unwrap();
    // Steady state with the gather in play: a second call neither grows the pool nor touches the
    // engine's.
    let warm = pools[0].stats();
    drop(qr_bits);
    engine
        .with_lanes(&[1], &mut pools, |_, lane| {
            drop(read_of(&reversed).qr_read(lane).unwrap());
        })
        .unwrap();
    assert!(pools[0].stats().buffers >= warm.buffers);
    assert_eq!(engine.buffer_pool_stats(), engine_pool_before);
    let qr_bits = engine
        .with_lanes(&[1], &mut pools, |_, lane| {
            read_of(&reversed).qr_read(lane).unwrap()
        })
        .unwrap()
        .pop()
        .unwrap();

    // The engine route gathers too, so both factor the same compact copy.
    let mut cpu = CpuEngine::serial();
    let (eq, er) = read_of(&reversed).qr_read(&mut cpu).unwrap();
    assert_same_tensors("reversed qr Q", &eq, &qr_bits.0);
    assert_same_tensors("reversed qr R", &er, &qr_bits.1);

    // Every lane entry accepts the reversed view and the empty one, like the engine entry.
    on_lane(|lane| {
        let mut cpu = CpuEngine::serial();
        for (tag, view) in [("reversed", &reversed), ("empty", &empty)] {
            assert_same_tensors(
                &format!("{tag} svdvals"),
                &read_of(view).svdvals_read(&mut cpu).unwrap(),
                &read_of(view).svdvals_read(lane).unwrap(),
            );
            assert_same_tensors(
                &format!("{tag} svd S"),
                &read_of(view).svd_read(&mut cpu).unwrap().1,
                &read_of(view).svd_read(lane).unwrap().1,
            );
            assert_same_tensors(
                &format!("{tag} svd_full U"),
                &read_of(view).svd_full_read(&mut cpu).unwrap().0,
                &read_of(view).svd_full_read(lane).unwrap().0,
            );
        }
    });
}

#[test]
fn a_negative_stride_batch_and_square_operand_run_in_a_lane() {
    let herm = hermitian::<Complex64>(3, 2);
    let reversed = herm
        .as_view()
        .slice_view(&[
            StridedSliceSpec::reverse(),
            StridedSliceSpec::reverse(),
            StridedSliceSpec::all(),
        ])
        .unwrap();
    assert!(reversed.strides()[0] < 0);
    let mut cpu = CpuEngine::serial();
    let want = read_of(&reversed).eigvalsh_read(&mut cpu).unwrap();
    let want_vals = read_of(&reversed).eigvals_read(&mut cpu).unwrap();
    on_lane(|lane| {
        assert_same_tensors(
            "eigvalsh",
            &want,
            &read_of(&reversed).eigvalsh_read(lane).unwrap(),
        );
        assert_same_tensors(
            "eigvals",
            &want_vals,
            &read_of(&reversed).eigvals_read(lane).unwrap(),
        );
        assert_eq!(
            read_of(&reversed).eigh_read(lane).unwrap().1.shape(),
            &[3, 3, 2]
        );
    });
}

// ---------------------------------------------------------------------------------------------
// Gauge (F2): the lane applies the engine route's post-processing
// ---------------------------------------------------------------------------------------------

#[test]
fn svd_and_eigh_gauges_on_a_lane_match_the_engine_route() {
    let a = base::<Complex64>(&[5, 3]);
    let herm = hermitian::<Complex64>(4, 2);
    let owned = owned_of(&a);
    let owned_herm = owned_of(&herm);
    let svd_options = SvdOptions::default().gauge(SvdGauge::CanonicalPivot);
    let eigh_options = EighOptions::default().gauge(EighGauge::CanonicalPivot);
    let mut cpu = CpuEngine::serial();
    let want_svd = owned.svd_with_options(svd_options, &mut cpu).unwrap();
    let want_svd_read = read_of(&a.as_view())
        .svd_with_options_read(svd_options, &mut cpu)
        .unwrap();
    let want_eigh = owned_herm
        .eigh_with_options(eigh_options, &mut cpu)
        .unwrap();
    let want_eigh_read = read_of(&herm.as_view())
        .eigh_with_options_read(eigh_options, &mut cpu)
        .unwrap();
    let raw = owned.svd(&mut cpu).unwrap();
    // The fixture must make the gauge visible, or this test checks nothing.
    assert_ne!(element_text(&raw.0), element_text(&want_svd.0));

    on_lane(|lane| {
        let got = owned.svd_with_options(svd_options, lane).unwrap();
        assert_same_tensors("svd_with_options U", &want_svd.0, &got.0);
        assert_same_tensors("svd_with_options S", &want_svd.1, &got.1);
        assert_same_tensors("svd_with_options Vh", &want_svd.2, &got.2);
        let got = read_of(&a.as_view())
            .svd_with_options_read(svd_options, lane)
            .unwrap();
        assert_same_tensors("svd_with_options_read U", &want_svd_read.0, &got.0);
        assert_same_tensors("svd_with_options_read Vh", &want_svd_read.2, &got.2);
        let got = owned_herm.eigh_with_options(eigh_options, lane).unwrap();
        assert_same_tensors("eigh_with_options values", &want_eigh.0, &got.0);
        assert_same_tensors("eigh_with_options vectors", &want_eigh.1, &got.1);
        let got = read_of(&herm.as_view())
            .eigh_with_options_read(eigh_options, lane)
            .unwrap();
        assert_same_tensors("eigh_with_options_read vectors", &want_eigh_read.1, &got.1);
        // A default-gauge option set is the plain entry.
        let plain = owned.svd_with_options(SvdOptions::default(), lane).unwrap();
        assert_same_tensors("default-gauge svd U", &raw.0, &plain.0);
    });
}

#[test]
fn an_invalid_derivative_eps_fails_on_a_lane_like_on_the_engine_route() {
    let a = Tensor::from_typed::<f64>(base::<f64>(&[3, 3]));
    let herm = Tensor::from_typed::<f64>(hermitian::<f64>(3, 1));
    let bad_svd = SvdOptions::default().derivative_eps(-1.0);
    let bad_eigh = EighOptions::default().derivative_eps(f64::NAN);
    let mut cpu = CpuEngine::serial();
    let want_svd = a.svd_with_options(bad_svd, &mut cpu).unwrap_err();
    let want_eigh = herm.eigh_with_options(bad_eigh, &mut cpu).unwrap_err();
    on_lane(|lane| {
        let got = a.svd_with_options(bad_svd, lane).unwrap_err();
        assert_eq!(got.kind(), want_svd.kind());
        assert_eq!(format!("{got:?}"), format!("{want_svd:?}"));
        let got = herm.eigh_with_options(bad_eigh, lane).unwrap_err();
        assert_eq!(got.kind(), want_eigh.kind());
        assert_eq!(format!("{got:?}"), format!("{want_eigh:?}"));
    });
}

// ---------------------------------------------------------------------------------------------
// The lane's pool, not the engine's
// ---------------------------------------------------------------------------------------------

fn run_all(lane: &mut dyn Engine) {
    let a = base::<Complex64>(&[4, 3, 2]);
    let h = hermitian::<Complex64>(3, 2);
    let (a, h) = (a.as_view(), h.as_view());
    drop(read_of(&a).svd_read(lane).unwrap());
    drop(read_of(&a).svd_full_read(lane).unwrap());
    drop(read_of(&a).svdvals_read(lane).unwrap());
    drop(read_of(&a).qr_read(lane).unwrap());
    drop(read_of(&h).eigh_read(lane).unwrap());
    drop(read_of(&h).eigvalsh_read(lane).unwrap());
    drop(read_of(&h).eigvals_read(lane).unwrap());
}

#[test]
fn every_buffer_comes_from_the_lane_pool_and_the_engine_pool_is_untouched() {
    let mut host = CpuEngine::serial();
    let engine_before = host.buffer_pool_stats();
    let mut pools = [BufferPool::new()];
    host.with_lanes(&[1], &mut pools, |_, lane| run_all(lane))
        .unwrap();
    let warm = pools[0].stats();
    assert!(warm.buffers > 0, "outputs recycle into the lane pool");
    assert_eq!(
        host.buffer_pool_stats(),
        engine_before,
        "engine pool untouched"
    );
    host.with_lanes(&[1], &mut pools, |_, lane| run_all(lane))
        .unwrap();
    assert_eq!(
        pools[0].stats(),
        warm,
        "steady state: the pool does not grow"
    );
    assert_eq!(host.buffer_pool_stats(), engine_before);
}

#[test]
fn two_lanes_with_two_pools_run_concurrent_factorizations_with_the_reference_bits() {
    let a = base::<Complex64>(&[24, 18]);
    let h = hermitian::<Complex64>(16, 3);
    let (want_u, want_s, want_vh) =
        borrowed(|e, b| svd_with_exec(e, b, &a.as_view(), false)).unwrap();
    let want_vals = borrowed(|e, b| eigh_values_with_exec(e, b, &h.as_view())).unwrap();

    let pool = tenferro_cpu::Pool::builder()
        .threads(4)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap();
    let mut host = CpuEngine::new(pool);
    let mut pools = [BufferPool::new(), BufferPool::new()];
    let (tx, rx) = std::sync::mpsc::channel();
    // Run on a helper thread with a watchdog so a deadlock fails instead of hanging.
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let out = host
                .with_lanes(&[2, 2], &mut pools, |lane, engine| {
                    if lane == 0 {
                        (Some(read_of(&a.as_view()).svd_read(engine).unwrap()), None)
                    } else {
                        (
                            None,
                            Some(read_of(&h.as_view()).eigvalsh_read(engine).unwrap()),
                        )
                    }
                })
                .unwrap();
            let _ = tx.send(out);
        });
        let out = rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("no result within 60 s: deadlock or lost lane");
        let svd = out[0].0.as_ref().unwrap();
        assert_same("lane svd U", &want_u, &svd.0);
        assert_same("lane svd S", &want_s, &svd.1);
        assert_same("lane svd Vh", &want_vh, &svd.2);
        assert_same("lane eigvalsh", &want_vals, out[1].1.as_ref().unwrap());
    });
    assert!(pools.iter().all(|p| p.stats().buffers > 0));
    assert_eq!(host.buffer_pool_stats().buffers, 0);
    assert!(!host.is_broken());
}
