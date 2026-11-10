//! Typed borrowed views against the owned-tensor entries (#2462, #2552).
//!
//! Each `TensorLinalgExt` call on a `TypedTensorView` gives the result the same call gives on the
//! same view materialized with `from_vec_col_major`, value bits and placement included; a
//! negative-stride view is gathered and agrees with the compact input; and an invalid input fails
//! with the same variant and the same downcastable source.

#![cfg(feature = "native")]

use std::error::Error as _;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{Eigh, LinalgScalar, Qr, Svd, SvdOptions, SvdShape, TensorLinalgExt};
use tenferro_tensor::{StridedSliceSpec, Tensor, TensorScalar, TypedTensor, TypedTensorView};

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

// ---------------------------------------------------------------------------------------------
// Equality with the engine entry
// ---------------------------------------------------------------------------------------------

/// Every general (non-square-only) factorization of `view`, on the view against the owned tensor.
fn check_general<T: Sample>(label: &str, view: &TypedTensorView<'_, T>)
where
    <T as TensorScalar>::Real: std::fmt::Debug,
{
    let owned = materialize(view);
    let mut engine = CpuEngine::serial();

    for full in [false, true] {
        let shape = if full { SvdShape::Full } else { SvdShape::Thin };
        let options = SvdOptions::default().with_shape(shape);
        let Svd { u, s, vh: vt } = view
            .svd_with(&options, &mut engine)
            .unwrap_or_else(|error| panic!("{label} svd(full={full}): {error}"));
        let Svd {
            u: eu,
            s: es,
            vh: evt,
        } = owned.svd_with(&options, &mut engine).unwrap();
        let tag = format!("{label} svd(full={full})");
        assert_same(&format!("{tag} U"), &u, &eu);
        assert_same(&format!("{tag} S"), &s, &es);
        assert_same(&format!("{tag} Vh"), &vt, &evt);
    }

    let s = view.svdvals(&mut engine).unwrap();
    assert_same(
        &format!("{label} svd_values"),
        &s,
        &owned.svdvals(&mut engine).unwrap(),
    );

    let Qr { q, r } = view.qr(&mut engine).unwrap();
    let Qr { q: eq, r: er } = owned.qr(&mut engine).unwrap();
    assert_same(&format!("{label} qr Q"), &q, &eq);
    assert_same(&format!("{label} qr R"), &r, &er);
}

/// Every square-only factorization of `view`, on the view against the owned tensor.
fn check_square<T: Sample>(label: &str, view: &TypedTensorView<'_, T>)
where
    <T as TensorScalar>::Real: std::fmt::Debug,
    T::Complex: std::fmt::Debug,
{
    let owned = materialize(view);
    let mut engine = CpuEngine::serial();

    let Eigh { values, vectors } = view.eigh(&mut engine).unwrap();
    let Eigh {
        values: evalues,
        vectors: evectors,
    } = owned.eigh(&mut engine).unwrap();
    assert_same(&format!("{label} eigh values"), &values, &evalues);
    assert_same(&format!("{label} eigh vectors"), &vectors, &evectors);

    let values = view.eigvalsh(&mut engine).unwrap();
    assert_same(
        &format!("{label} eigh_values"),
        &values,
        &owned.eigvalsh(&mut engine).unwrap(),
    );

    let values = view.eigvals(&mut engine).unwrap();
    assert_same(
        &format!("{label} eig_values"),
        &values,
        &owned.eigvals(&mut engine).unwrap(),
    );
}

fn equality_for<T: Sample>(dtype: &str)
where
    <T as TensorScalar>::Real: std::fmt::Debug,
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
fn each_view_entry_matches_the_owned_entry_on_value_bits_and_placement() {
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
    // the owned entry cannot see a gather bug (both paths share it), so compare the two routes.
    let mut engine = CpuEngine::serial();
    let a = base::<Complex64>(&[4, 3, 2]);
    let Svd { s, .. } = a
        .as_view()
        .svd_with(&SvdOptions::default(), &mut engine)
        .unwrap();
    let s: TypedTensor<f64> = s;
    let direct = a.as_view().svdvals(&mut engine).unwrap();
    assert_eq!(s.shape(), &[3, 2]);
    assert_close(
        "c64 svd spectrum",
        s.host_data().unwrap(),
        direct.host_data().unwrap(),
        1e-12,
    );

    let h = hermitian::<Complex64>(3, 2);
    let Eigh { values, vectors } = h.as_view().eigh(&mut engine).unwrap();
    let _: &TypedTensor<Complex64> = &vectors;
    let direct = h.as_view().eigvalsh(&mut engine).unwrap();
    assert_close(
        "c64 eigh spectrum",
        values.host_data().unwrap(),
        direct.host_data().unwrap(),
        1e-12,
    );

    let h32 = hermitian::<Complex32>(3, 1);
    let Eigh { values, .. } = h32.as_view().eigh(&mut engine).unwrap();
    let _: TypedTensor<f32> = values;

    let r = base::<f32>(&[2, 2]);
    let values = r.as_view().eigvals(&mut engine).unwrap();
    let _: TypedTensor<Complex32> = values;
}

// ---------------------------------------------------------------------------------------------
// Negative strides
// ---------------------------------------------------------------------------------------------

#[test]
fn a_negative_stride_view_is_gathered_and_matches_the_compact_input() {
    let a = base::<f64>(&[3, 3]);
    let reversed = a
        .as_view()
        .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::all()])
        .unwrap();
    assert!(reversed.strides()[0] < 0);

    let mut engine = CpuEngine::serial();
    let compact = materialize(&reversed);
    let from_view = reversed.svdvals(&mut engine).unwrap();
    let from_compact = compact.svdvals(&mut engine).unwrap();
    assert_eq!(from_view.shape(), &[3]);
    assert_close(
        "negative-stride svdvals",
        from_view.host_data().unwrap(),
        from_compact.as_slice::<f64>().unwrap(),
        1e-12,
    );

    // A zero-length axis taken from the reversed view keeps the negative stride; the
    // zero-dimension shortcut must still return the empty result (#2505).
    let empty_wide = reversed
        .slice_view(&[
            StridedSliceSpec::all(),
            StridedSliceSpec::new(0, Some(0), 1),
        ])
        .unwrap();
    assert_eq!(empty_wide.shape(), &[3, 0]);
    let empty = empty_wide.svdvals(&mut engine).unwrap();
    assert_eq!(empty.shape(), &[0]);
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
fn invalid_inputs_fail_with_the_same_variant_and_source_as_the_owned_entry() {
    let mut engine = CpuEngine::serial();

    // Rank 1: every entry refuses the rank.
    let vector = base::<f64>(&[3]);
    let owned = materialize(&vector.as_view());
    let view = vector.as_view();
    assert_same_error(
        "rank-1 svd",
        view.svd(&mut engine).unwrap_err(),
        owned.svd(&mut engine).unwrap_err(),
    );
    assert_same_error(
        "rank-1 svd_values",
        view.svdvals(&mut engine).unwrap_err(),
        owned.svdvals(&mut engine).unwrap_err(),
    );
    assert_same_error(
        "rank-1 qr",
        view.qr(&mut engine).unwrap_err(),
        owned.qr(&mut engine).unwrap_err(),
    );

    // Non-square: the eigensolvers refuse the shape.
    let rect = base::<f64>(&[3, 2]);
    let owned = materialize(&rect.as_view());
    let view = rect.as_view();
    assert_same_error(
        "non-square eigh",
        view.eigh(&mut engine).unwrap_err(),
        owned.eigh(&mut engine).unwrap_err(),
    );
    assert_same_error(
        "non-square eigh_values",
        view.eigvalsh(&mut engine).unwrap_err(),
        owned.eigvalsh(&mut engine).unwrap_err(),
    );
    assert_same_error(
        "non-square eig_values",
        view.eigvals(&mut engine).unwrap_err(),
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
            view.svd(&mut engine).map(|_| ()),
            owned.svd(&mut engine).map(|_| ()),
        ),
        (
            "nan svd_values",
            view.svdvals(&mut engine).map(|_| ()),
            owned.svdvals(&mut engine).map(|_| ()),
        ),
        (
            "nan qr",
            view.qr(&mut engine).map(|_| ()),
            owned.qr(&mut engine).map(|_| ()),
        ),
        (
            "nan eigh",
            view.eigh(&mut engine).map(|_| ()),
            owned.eigh(&mut engine).map(|_| ()),
        ),
        (
            "nan eigh_values",
            view.eigvalsh(&mut engine).map(|_| ()),
            owned.eigvalsh(&mut engine).map(|_| ()),
        ),
        (
            "nan eig_values",
            view.eigvals(&mut engine).map(|_| ()),
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
