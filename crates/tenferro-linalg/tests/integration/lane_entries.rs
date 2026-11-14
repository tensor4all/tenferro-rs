//! The factorization entries inside a lane of `CpuEngine::with_lanes` against the owned-engine
//! entries (#2552 S2b, the P7 parity tests; #2462 for the contract the former `*_with_exec`
//! entries carried).
//!
//! A lane is an engine over a share of the thread budget and a caller-owned pool. The ordinary
//! grammar (`view.svd(lane)` and its siblings from `LinalgOps`) runs on it and must give the
//! result the owned-tensor engine entry gives for the same logical matrix (value bits and
//! placement included), fail with the same error, accept a negative stride by gathering it once
//! into the lane's pool, and draw every buffer from the lane's pool and no other. Every other
//! linalg operation is `Unsupported` on a lane.

use std::error::Error as _;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{linalg_interop::BufferPool, CpuEngine};
use tenferro_linalg::{
    EighGauge, EighOptions, LinalgOps, LinalgScalar, Svd, SvdGauge, SvdOptions, SvdShape,
};
use tenferro_tensor::{
    DType, Engine, ErrorKind, StridedSliceSpec, Tensor, TensorRead, TypedTensor, TypedTensorView,
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

/// Run `body` on one lane of a fresh serial engine with a fresh lane pool.
fn on_lane<R: Send>(body: impl Fn(&mut dyn Engine) -> R + Sync) -> R {
    let mut engine = CpuEngine::serial();
    let mut pools = [BufferPool::new()];
    let mut out = engine
        .with_lanes(&[1], &mut pools, |_, lane| body(lane))
        .unwrap();
    out.pop().unwrap()
}

// ---------------------------------------------------------------------------------------------
// Result parity with the owned engine entry
// ---------------------------------------------------------------------------------------------

fn check_general<T: Sample>(label: &str, view: &TypedTensorView<'_, T>) {
    let owned = materialize(view);
    let mut engine = CpuEngine::serial();

    for full in [false, true] {
        let want = if full {
            owned
                .svd_with(&SvdOptions::default().shape(SvdShape::Full), &mut engine)
                .unwrap()
        } else {
            owned.svd(&mut engine).unwrap()
        };
        let got = on_lane(|lane| {
            if full {
                read_of(view)
                    .svd_with(&SvdOptions::default().shape(SvdShape::Full), lane)
                    .unwrap()
            } else {
                read_of(view).svd(lane).unwrap()
            }
        });
        let tag = format!("{label} svd(full={full})");
        assert_same_tensors(&format!("{tag} U"), &want.u, &got.u);
        assert_same_tensors(&format!("{tag} S"), &want.s, &got.s);
        assert_same_tensors(&format!("{tag} Vh"), &want.vh, &got.vh);
    }

    let want = owned.svdvals(&mut engine).unwrap();
    let got = on_lane(|lane| read_of(view).svdvals(lane).unwrap());
    assert_same_tensors(&format!("{label} svdvals"), &want, &got);

    let want = owned.qr(&mut engine).unwrap();
    let got = on_lane(|lane| read_of(view).qr(lane).unwrap());
    assert_same_tensors(&format!("{label} qr Q"), &want.q, &got.q);
    assert_same_tensors(&format!("{label} qr R"), &want.r, &got.r);
}

fn check_square<T: Sample>(label: &str, view: &TypedTensorView<'_, T>) {
    let owned = materialize(view);
    let mut engine = CpuEngine::serial();

    let want = owned.eigh(&mut engine).unwrap();
    let got = on_lane(|lane| read_of(view).eigh(lane).unwrap());
    assert_same_tensors(&format!("{label} eigh values"), &want.values, &got.values);
    assert_same_tensors(
        &format!("{label} eigh vectors"),
        &want.vectors,
        &got.vectors,
    );

    let want = owned.eigvalsh(&mut engine).unwrap();
    let got = on_lane(|lane| read_of(view).eigvalsh(lane).unwrap());
    assert_same_tensors(&format!("{label} eigvalsh"), &want, &got);

    let want = owned.eigvals(&mut engine).unwrap();
    let got = on_lane(|lane| read_of(view).eigvals(lane).unwrap());
    assert_same_tensors(&format!("{label} eigvals"), &want, &got);
}

fn parity_for<T: Sample>(dtype: &str) {
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
    // Batched `[m, n, b]` read in place: the batch axis is the unit-stride one.
    let stacked = base::<T>(&[2, 4, 3]);
    let batched = stacked.as_view().transpose_view([1, 2, 0]).unwrap();
    assert_eq!(batched.shape(), &[4, 3, 2]);
    check_general(&format!("{dtype} batched"), &batched);
    let herm = hermitian::<T>(3, 2);
    check_square(&format!("{dtype} batched square"), &herm.as_view());
    let herm_t = herm.as_view().transpose_view([1, 0, 2]).unwrap();
    check_square(&format!("{dtype} strided square"), &herm_t);
    let empty = base::<T>(&[0, 0]);
    check_square(&format!("{dtype} empty square"), &empty.as_view());
}

#[test]
fn lane_results_are_bit_identical_to_the_engine_entries_f64() {
    parity_for::<f64>("f64");
}

#[test]
fn lane_results_are_bit_identical_to_the_engine_entries_complex64() {
    parity_for::<Complex64>("c64");
}

#[test]
fn lane_results_are_bit_identical_to_the_engine_entries_single_precision() {
    parity_for::<f32>("f32");
    parity_for::<Complex32>("c32");
}

#[test]
fn every_lane_entry_spelling_gives_the_same_bits() {
    // Owned `Tensor`, the borrowed read and the typed tensor are one route on a lane.
    let typed = base::<Complex64>(&[5, 4]);
    let owned = owned_of(&typed);
    let herm = hermitian::<Complex64>(4, 1);
    let owned_herm = owned_of(&herm);
    let mut engine = CpuEngine::serial();
    let want = owned.svd(&mut engine).unwrap();
    on_lane(|lane| {
        let from_owned = owned.svd(lane).unwrap();
        assert_same_tensors("owned svd U", &want.u, &from_owned.u);
        assert_same_tensors("owned svd S", &want.s, &from_owned.s);
        assert_same_tensors("owned svd Vh", &want.vh, &from_owned.vh);
        let from_typed = typed.svd(lane).unwrap();
        assert_same_tensors("typed svd U", &want.u, &Tensor::from_typed(from_typed.u));
        assert_same_tensors("typed svd S", &want.s, &Tensor::from_typed(from_typed.s));
        assert_same_tensors("typed svd Vh", &want.vh, &Tensor::from_typed(from_typed.vh));
        let from_view = typed.as_view().svd(lane).unwrap();
        assert_same_tensors("view svd U", &want.u, &Tensor::from_typed(from_view.u));
        let full = owned
            .svd_with(&SvdOptions::default().shape(SvdShape::Full), lane)
            .unwrap();
        assert_eq!(full.u.shape(), &[5, 5]);
        assert_eq!(owned.svdvals(lane).unwrap().shape(), &[4]);
        assert_eq!(owned.qr(lane).unwrap().q.shape(), &[5, 4]);
        assert_eq!(owned_herm.eigh(lane).unwrap().vectors.shape(), &[4, 4, 1]);
        assert_eq!(owned_herm.eigvalsh(lane).unwrap().shape(), &[4, 1]);
        assert_eq!(owned_herm.eigvals(lane).unwrap().shape(), &[4, 1]);
        let typed_values = herm.eigvalsh(lane).unwrap();
        assert_eq!(typed_values.shape(), &[4, 1]);
    });
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
    let h = hermitian::<Complex64>(3, 2);
    let h32 = hermitian::<Complex32>(3, 1);
    let r = base::<f32>(&[2, 2]);
    on_lane(|lane| {
        let Svd { s, .. } = a.as_view().svd(lane).unwrap();
        let s: TypedTensor<f64> = s;
        let direct = a.as_view().svdvals(lane).unwrap();
        assert_eq!(s.shape(), &[3, 2]);
        assert_close(
            "c64 svd spectrum",
            s.host_data().unwrap(),
            direct.host_data().unwrap(),
            1e-12,
        );

        let eigh = h.as_view().eigh(lane).unwrap();
        let _: &TypedTensor<Complex64> = &eigh.vectors;
        let direct = h.as_view().eigvalsh(lane).unwrap();
        assert_close(
            "c64 eigh spectrum",
            eigh.values.host_data().unwrap(),
            direct.host_data().unwrap(),
            1e-12,
        );

        let values: TypedTensor<f32> = h32.as_view().eigh(lane).unwrap().values;
        drop(values);
        let values: TypedTensor<Complex32> = r.as_view().eigvals(lane).unwrap();
        drop(values);
    });
}

// ---------------------------------------------------------------------------------------------
// Error parity
// ---------------------------------------------------------------------------------------------

fn same_error(label: &str, lane: tenferro_tensor::Error, engine: tenferro_tensor::Error) {
    assert_eq!(lane.kind(), engine.kind(), "{label}: kind");
    assert_eq!(
        format!("{lane:?}"),
        format!("{engine:?}"),
        "{label}: variant"
    );
    let payload = |error: &tenferro_tensor::Error| {
        error
            .source()
            .and_then(|source| source.downcast_ref::<tenferro_linalg::Error>())
            .map(|payload| format!("{payload:?}"))
    };
    assert_eq!(payload(&lane), payload(&engine), "{label}: source");
}

#[test]
fn lane_errors_have_the_same_kind_variant_and_source_as_the_engine_entries() {
    let mut engine = CpuEngine::serial();

    // Rank 1: every entry refuses the rank.
    let vector = base::<f64>(&[3]);
    let v = vector.as_view();
    let owned = materialize(&v);
    same_error(
        "rank-1 svd",
        on_lane(|lane| read_of(&v).svd(lane).map(|_| ()).unwrap_err()),
        owned.svd(&mut engine).map(|_| ()).unwrap_err(),
    );
    same_error(
        "rank-1 svdvals",
        on_lane(|lane| read_of(&v).svdvals(lane).map(|_| ()).unwrap_err()),
        owned.svdvals(&mut engine).map(|_| ()).unwrap_err(),
    );
    same_error(
        "rank-1 qr",
        on_lane(|lane| read_of(&v).qr(lane).map(|_| ()).unwrap_err()),
        owned.qr(&mut engine).map(|_| ()).unwrap_err(),
    );

    // Non-square: the eigensolvers refuse the shape.
    let rect = base::<f64>(&[3, 2]);
    let r = rect.as_view();
    let owned = materialize(&r);
    same_error(
        "non-square eigh",
        on_lane(|lane| read_of(&r).eigh(lane).map(|_| ()).unwrap_err()),
        owned.eigh(&mut engine).map(|_| ()).unwrap_err(),
    );
    same_error(
        "non-square eigvalsh",
        on_lane(|lane| read_of(&r).eigvalsh(lane).map(|_| ()).unwrap_err()),
        owned.eigvalsh(&mut engine).map(|_| ()).unwrap_err(),
    );
    same_error(
        "non-square eigvals",
        on_lane(|lane| read_of(&r).eigvals(lane).map(|_| ()).unwrap_err()),
        owned.eigvals(&mut engine).map(|_| ()).unwrap_err(),
    );

    // Non-finite input: the provider's own mapped failure, wherever it reports one.
    let mut data = base::<f64>(&[3, 3]).host_data().unwrap().to_vec();
    data[4] = f64::NAN;
    let nan = TypedTensor::<f64>::from_vec_col_major(vec![3, 3], data).unwrap();
    let n = nan.as_view();
    let owned = materialize(&n);
    type Outcome = Result<(), tenferro_tensor::Error>;
    let cases: [(&str, Outcome, Outcome); 6] = [
        (
            "nan svd",
            on_lane(|lane| read_of(&n).svd(lane).map(|_| ())),
            owned.svd(&mut engine).map(|_| ()),
        ),
        (
            "nan svdvals",
            on_lane(|lane| read_of(&n).svdvals(lane).map(|_| ())),
            owned.svdvals(&mut engine).map(|_| ()),
        ),
        (
            "nan qr",
            on_lane(|lane| read_of(&n).qr(lane).map(|_| ())),
            owned.qr(&mut engine).map(|_| ()),
        ),
        (
            "nan eigh",
            on_lane(|lane| read_of(&n).eigh(lane).map(|_| ())),
            owned.eigh(&mut engine).map(|_| ()),
        ),
        (
            "nan eigvalsh",
            on_lane(|lane| read_of(&n).eigvalsh(lane).map(|_| ())),
            owned.eigvalsh(&mut engine).map(|_| ()),
        ),
        (
            "nan eigvals",
            on_lane(|lane| read_of(&n).eigvals(lane).map(|_| ())),
            owned.eigvals(&mut engine).map(|_| ()),
        ),
    ];
    let mut failures = 0;
    for (label, lane, engine) in cases {
        match (lane, engine) {
            (Ok(()), Ok(())) => {}
            (Err(lane), Err(engine)) => {
                failures += 1;
                same_error(label, lane, engine);
            }
            (lane, engine) => panic!("{label}: lane {lane:?} but engine {engine:?}"),
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
    let square = Tensor::from_vec_col_major(vec![3, 3], (0..9).map(f64::from).collect()).unwrap();
    let a = Tensor::from_typed::<f64>(hermitian::<f64>(3, 1));
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
                "qr_with (non-default gauge)",
                a.qr_with(
                    &tenferro_linalg::QrOptions::default()
                        .gauge(tenferro_linalg::QrGauge::PositiveDiagonal),
                    lane,
                )
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

/// Every lane entry on `view` against the owned engine entry on the materialized logical
/// matrix, bit for bit. `square` also runs the eigensolvers.
fn assert_lane_matches_materialized(tag: &str, view: &TypedTensorView<'_, f64>, square: bool) {
    let owned = materialize(view);
    let mut engine = CpuEngine::serial();
    let want = owned.svd(&mut engine).unwrap();
    let want_full = owned
        .svd_with(&SvdOptions::default().shape(SvdShape::Full), &mut engine)
        .unwrap();
    let want_vals = owned.svdvals(&mut engine).unwrap();
    let want_qr = owned.qr(&mut engine).unwrap();
    let want_eigh = square.then(|| owned.eigh(&mut engine).unwrap());
    let want_eigvalsh = square.then(|| owned.eigvalsh(&mut engine).unwrap());
    let want_eigvals = square.then(|| owned.eigvals(&mut engine).unwrap());
    on_lane(|lane| {
        let got = read_of(view).svd(lane).unwrap();
        assert_same_tensors(&format!("{tag} svd U"), &want.u, &got.u);
        assert_same_tensors(&format!("{tag} svd S"), &want.s, &got.s);
        assert_same_tensors(&format!("{tag} svd Vh"), &want.vh, &got.vh);
        let got = read_of(view)
            .svd_with(&SvdOptions::default().shape(SvdShape::Full), lane)
            .unwrap();
        assert_same_tensors(&format!("{tag} svd_full U"), &want_full.u, &got.u);
        assert_same_tensors(&format!("{tag} svd_full Vh"), &want_full.vh, &got.vh);
        assert_same_tensors(
            &format!("{tag} svdvals"),
            &want_vals,
            &read_of(view).svdvals(lane).unwrap(),
        );
        let got = read_of(view).qr(lane).unwrap();
        assert_same_tensors(&format!("{tag} qr Q"), &want_qr.q, &got.q);
        assert_same_tensors(&format!("{tag} qr R"), &want_qr.r, &got.r);
        if let (Some(eigh), Some(eigvalsh), Some(eigvals)) =
            (&want_eigh, &want_eigvalsh, &want_eigvals)
        {
            let got = read_of(view).eigh(lane).unwrap();
            assert_same_tensors(&format!("{tag} eigh values"), &eigh.values, &got.values);
            assert_same_tensors(&format!("{tag} eigh vectors"), &eigh.vectors, &got.vectors);
            assert_same_tensors(
                &format!("{tag} eigvalsh"),
                eigvalsh,
                &read_of(view).eigvalsh(lane).unwrap(),
            );
            assert_same_tensors(
                &format!("{tag} eigvals"),
                eigvals,
                &read_of(view).eigvals(lane).unwrap(),
            );
        }
    });
}

#[test]
fn a_negative_stride_operand_runs_in_a_lane_like_its_materialized_matrix() {
    // Replaces `a_negative_stride_is_unsupported_for_every_entry`: the borrowed `*_with_exec`
    // entries refused a negative stride (and an empty operand that kept one, #2505); the lane
    // gathers it once into its pool, so every entry accepts it and gives the bits the owned
    // engine entry gives for the materialized logical matrix.
    let a = base::<f64>(&[3, 3]);
    let reversed = a
        .as_view()
        .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::all()])
        .unwrap();
    assert!(reversed.strides()[0] < 0);
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

    assert_lane_matches_materialized("reversed", &reversed, true);
    assert_lane_matches_materialized("empty wide", &empty_wide, false);
    assert_lane_matches_materialized("empty square", &empty_square, true);
}

#[test]
fn a_negative_stride_is_gathered_into_the_lane_pool_and_matches_the_engine_route() {
    let a = base::<f64>(&[4, 3]);
    let reversed = a
        .as_view()
        .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::all()])
        .unwrap();
    assert!(reversed.strides()[0] < 0);
    let compact = base::<f64>(&[4, 3]);
    let input_bytes = 4 * 3 * std::mem::size_of::<f64>();

    // Cold lane pools, outputs dropped inside the lane: whatever the pool holds afterwards was
    // taken from it during the call. The reversed input needs one more buffer (the gather
    // target, handed back when the call ends) than the compact input of the same shape.
    let mut engine = CpuEngine::serial();
    let engine_pool_before = engine.buffer_pool_stats();
    let mut held_after = |view: &TypedTensorView<'_, f64>| {
        let mut pools = [BufferPool::new()];
        engine
            .with_lanes(&[1], &mut pools, |_, lane| {
                drop(read_of(view).qr(lane).unwrap());
            })
            .unwrap();
        pools[0].stats()
    };
    let compact_stats = held_after(&compact.as_view());
    let reversed_stats = held_after(&reversed);
    assert!(
        reversed_stats.buffers > compact_stats.buffers
            && reversed_stats.capacity_bytes >= compact_stats.capacity_bytes + input_bytes,
        "the gather buffer must come from and return to the lane pool: \
         compact {compact_stats:?}, reversed {reversed_stats:?}"
    );
    assert_eq!(
        engine.buffer_pool_stats(),
        engine_pool_before,
        "the engine's own pool is untouched"
    );

    // Steady state with the gather in play: a second call neither grows the pool nor touches the
    // engine's.
    let mut pools = [BufferPool::new()];
    let first = engine
        .with_lanes(&[1], &mut pools, |_, lane| {
            read_of(&reversed).qr(lane).unwrap()
        })
        .unwrap()
        .pop()
        .unwrap();
    drop(first);
    let warm = pools[0].stats();
    engine
        .with_lanes(&[1], &mut pools, |_, lane| {
            drop(read_of(&reversed).qr(lane).unwrap());
        })
        .unwrap();
    assert!(pools[0].stats().buffers >= warm.buffers);
    assert_eq!(engine.buffer_pool_stats(), engine_pool_before);
    let qr_bits = engine
        .with_lanes(&[1], &mut pools, |_, lane| {
            read_of(&reversed).qr(lane).unwrap()
        })
        .unwrap()
        .pop()
        .unwrap();

    // The engine route on the materialized logical matrix gives the same bits.
    let mut cpu = CpuEngine::serial();
    let want = materialize(&reversed).qr(&mut cpu).unwrap();
    assert_same_tensors("reversed qr Q", &want.q, &qr_bits.q);
    assert_same_tensors("reversed qr R", &want.r, &qr_bits.r);
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
    let owned = materialize(&reversed);
    let mut cpu = CpuEngine::serial();
    let want = owned.eigvalsh(&mut cpu).unwrap();
    let want_vals = owned.eigvals(&mut cpu).unwrap();
    on_lane(|lane| {
        assert_same_tensors(
            "eigvalsh",
            &want,
            &read_of(&reversed).eigvalsh(lane).unwrap(),
        );
        assert_same_tensors(
            "eigvals",
            &want_vals,
            &read_of(&reversed).eigvals(lane).unwrap(),
        );
        assert_eq!(
            read_of(&reversed).eigh(lane).unwrap().vectors.shape(),
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
    let want_svd = owned.svd_with(&svd_options, &mut cpu).unwrap();
    let want_svd_read = read_of(&a.as_view())
        .svd_with(&svd_options, &mut cpu)
        .unwrap();
    let want_eigh = owned_herm.eigh_with(&eigh_options, &mut cpu).unwrap();
    let want_eigh_read = read_of(&herm.as_view())
        .eigh_with(&eigh_options, &mut cpu)
        .unwrap();
    let raw = owned.svd(&mut cpu).unwrap();
    // The fixture must make the gauge visible, or this test checks nothing.
    assert_ne!(element_text(&raw.u), element_text(&want_svd.u));

    on_lane(|lane| {
        let got = owned.svd_with(&svd_options, lane).unwrap();
        assert_same_tensors("svd_with U", &want_svd.u, &got.u);
        assert_same_tensors("svd_with S", &want_svd.s, &got.s);
        assert_same_tensors("svd_with Vh", &want_svd.vh, &got.vh);
        let got = read_of(&a.as_view()).svd_with(&svd_options, lane).unwrap();
        assert_same_tensors("svd_with (read) U", &want_svd_read.u, &got.u);
        assert_same_tensors("svd_with (read) Vh", &want_svd_read.vh, &got.vh);
        let got = owned_herm.eigh_with(&eigh_options, lane).unwrap();
        assert_same_tensors("eigh_with values", &want_eigh.values, &got.values);
        assert_same_tensors("eigh_with vectors", &want_eigh.vectors, &got.vectors);
        let got = read_of(&herm.as_view())
            .eigh_with(&eigh_options, lane)
            .unwrap();
        assert_same_tensors(
            "eigh_with (read) vectors",
            &want_eigh_read.vectors,
            &got.vectors,
        );
        // A default-gauge option set is the plain entry.
        let plain = owned.svd_with(&SvdOptions::default(), lane).unwrap();
        assert_same_tensors("default-gauge svd U", &raw.u, &plain.u);
    });
}

#[test]
fn an_invalid_derivative_eps_fails_on_a_lane_like_on_the_engine_route() {
    let a = Tensor::from_typed::<f64>(base::<f64>(&[3, 3]));
    let herm = Tensor::from_typed::<f64>(hermitian::<f64>(3, 1));
    let bad_svd = SvdOptions::default().derivative_eps(-1.0);
    let bad_eigh = EighOptions::default().derivative_eps(f64::NAN);
    let mut cpu = CpuEngine::serial();
    let want_svd = a.svd_with(&bad_svd, &mut cpu).unwrap_err();
    let want_eigh = herm.eigh_with(&bad_eigh, &mut cpu).unwrap_err();
    on_lane(|lane| {
        let got = a.svd_with(&bad_svd, lane).unwrap_err();
        assert_eq!(got.kind(), want_svd.kind());
        assert_eq!(format!("{got:?}"), format!("{want_svd:?}"));
        let got = herm.eigh_with(&bad_eigh, lane).unwrap_err();
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
    drop(read_of(&a).svd(lane).unwrap());
    drop(
        read_of(&a)
            .svd_with(&SvdOptions::default().shape(SvdShape::Full), lane)
            .unwrap(),
    );
    drop(read_of(&a).svdvals(lane).unwrap());
    drop(read_of(&a).qr(lane).unwrap());
    drop(read_of(&h).eigh(lane).unwrap());
    drop(read_of(&h).eigvalsh(lane).unwrap());
    drop(read_of(&h).eigvals(lane).unwrap());
}

#[test]
fn every_buffer_comes_from_the_lane_pool_and_the_engine_pool_is_untouched() {
    let mut host = CpuEngine::serial();
    let engine_before = host.buffer_pool_stats();
    let mut pools = [BufferPool::new()];
    assert!(pools[0].is_empty());
    host.with_lanes(&[1], &mut pools, |_, lane| run_all(lane))
        .unwrap();
    // The outputs and the transient spectra returned to the lane's pool, not the engine's.
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
fn two_lanes_with_two_pools_do_not_share_storage() {
    let mut first_engine = CpuEngine::serial();
    let mut second_engine = CpuEngine::serial();
    let mut first_pools = [BufferPool::new()];
    let mut second_pools = [BufferPool::new()];
    let a = base::<f64>(&[4, 4]);

    let first = first_engine
        .with_lanes(&[1], &mut first_pools, |_, lane| {
            a.as_view().svd(lane).unwrap()
        })
        .unwrap()
        .pop()
        .unwrap();
    let second = second_engine
        .with_lanes(&[1], &mut second_pools, |_, lane| {
            a.as_view().svd(lane).unwrap()
        })
        .unwrap()
        .pop()
        .unwrap();

    // Same values, distinct storage.
    assert_eq!(first.s.host_data().unwrap(), second.s.host_data().unwrap());
    for (x, y) in [(&first.u, &second.u), (&first.vh, &second.vh)] {
        assert_ne!(
            x.host_data().unwrap().as_ptr(),
            y.host_data().unwrap().as_ptr(),
            "two pools handed out the same storage"
        );
    }
    assert_ne!(
        first.s.host_data().unwrap().as_ptr(),
        second.s.host_data().unwrap().as_ptr()
    );

    // Dropping the first lane's outputs fills only the first pool.
    let second_before = second_pools[0].stats();
    drop(first);
    assert!(first_pools[0].stats().buffers > 0);
    assert_eq!(second_pools[0].stats(), second_before);

    // And the second lane's outputs return only to the second pool.
    let first_before = first_pools[0].stats();
    drop(second);
    assert_eq!(first_pools[0].stats(), first_before);
    assert!(second_pools[0].stats().buffers > second_before.buffers);
}

#[test]
fn two_lanes_with_two_pools_run_concurrent_factorizations_with_the_reference_bits() {
    let a = base::<Complex64>(&[24, 18]);
    let h = hermitian::<Complex64>(16, 3);
    let mut reference = CpuEngine::serial();
    let want = owned_of(&a).svd(&mut reference).unwrap();
    let want_vals = owned_of(&h).eigvalsh(&mut reference).unwrap();

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
                        (Some(read_of(&a.as_view()).svd(engine).unwrap()), None)
                    } else {
                        (None, Some(read_of(&h.as_view()).eigvalsh(engine).unwrap()))
                    }
                })
                .unwrap();
            let _ = tx.send(out);
        });
        let out = rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("no result within 60 s: deadlock or lost lane");
        let svd = out[0].0.as_ref().unwrap();
        assert_same_tensors("lane svd U", &want.u, &svd.u);
        assert_same_tensors("lane svd S", &want.s, &svd.s);
        assert_same_tensors("lane svd Vh", &want.vh, &svd.vh);
        assert_same_tensors("lane eigvalsh", &want_vals, out[1].1.as_ref().unwrap());
    });
    assert!(pools.iter().all(|p| p.stats().buffers > 0));
    assert_eq!(host.buffer_pool_stats().buffers, 0);
    assert!(!host.is_broken());
}
