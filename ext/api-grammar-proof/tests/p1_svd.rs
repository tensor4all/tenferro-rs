//! Prototype 1 tests: complex SVD across input forms. See findings/p1.md.

use num_complex::Complex64;
use tenferro_api_grammar_proof::p1_svd::{
    downcast_checked, svd_typed_view_direct, Svd, SvdOps, SvdOptions, SvdShape,
};
use tenferro_api_grammar_proof::policy::MaterializationPolicy::{Allow, Forbid};
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{SvdDriver, SvdGauge, SvdOptions as LinalgSvdOptions, TensorReadLinalgExt};
use tenferro_tensor::{
    DType, Error, Tensor, TensorRead, TensorScalar, TypedTensor, TypedTensorView,
};

type C = Complex64;

fn cval(k: usize) -> C {
    let x = k as f64;
    C::new((1.0 + 0.7 * x).sin() + 0.1 * x, (0.3 * x * x).cos() - 0.2)
}

/// Column-major data for a matrix (or batch) of the given shape.
fn data_c(shape: &[usize]) -> Vec<C> {
    (0..shape.iter().product::<usize>()).map(cval).collect()
}

fn typed_c(shape: &[usize], data: Vec<C>) -> TypedTensor<C> {
    TypedTensor::<C>::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn tensor_c(shape: &[usize], data: Vec<C>) -> Tensor {
    Tensor::from_typed::<C>(typed_c(shape, data))
}

/// Logical elements of a (possibly strided) view, column-major.
fn logical(view: &TypedTensorView<'_, C>) -> Vec<C> {
    let shape = view.shape().to_vec();
    let total: usize = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    for lin in 0..total {
        let mut rem = lin;
        let idx: Vec<usize> = shape
            .iter()
            .map(|&d| {
                let i = rem % d;
                rem /= d;
                i
            })
            .collect();
        out.push(*view.get(&idx).unwrap());
    }
    out
}

fn erase(r: Svd<TypedTensor<C>, TypedTensor<f64>>) -> Svd<Tensor> {
    Svd {
        u: Tensor::from_typed::<C>(r.u),
        s: Tensor::from_typed::<f64>(r.s),
        vh: Tensor::from_typed::<C>(r.vh),
    }
}

/// Check shapes, dtypes and reconstruction; return `s` (all batches, col-major).
fn check(
    svd: &Svd<Tensor>,
    a: &[C],
    m: usize,
    n: usize,
    batch: usize,
    shape: SvdShape,
) -> Vec<f64> {
    let k = m.min(n);
    let (uc, vr) = match shape {
        SvdShape::Thin => (k, k),
        SvdShape::Full => (m, n),
    };
    assert_eq!(svd.u.dtype(), DType::C64);
    assert_eq!(svd.vh.dtype(), DType::C64);
    assert_eq!(svd.s.dtype(), DType::F64, "s of a complex SVD must be real");
    let tail: Vec<usize> = if batch > 1 { vec![batch] } else { vec![] };
    let want = |mut v: Vec<usize>| {
        v.extend(&tail);
        v
    };
    assert_eq!(svd.u.shape(), want(vec![m, uc]).as_slice());
    assert_eq!(svd.s.shape(), want(vec![k]).as_slice());
    assert_eq!(svd.vh.shape(), want(vec![vr, n]).as_slice());
    let u = C::as_slice(&svd.u).unwrap();
    let vh = C::as_slice(&svd.vh).unwrap();
    let s = f64::as_slice(&svd.s).unwrap();
    for b in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = C::new(0.0, 0.0);
                for l in 0..k {
                    acc += u[i + l * m + b * m * uc] * s[l + b * k] * vh[l + j * vr + b * vr * n];
                }
                let diff = (acc - a[i + j * m + b * m * n]).norm();
                assert!(diff < 1e-12, "recon error {diff} at ({i},{j},{b})");
            }
        }
        // descending, non-negative
        for l in 1..k {
            assert!(s[l - 1] >= s[l] && s[l] >= 0.0);
        }
    }
    s.to_vec()
}

fn assert_close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 1e-12, "{x} vs {y}");
    }
}

fn is_unsupported(e: &Error) -> bool {
    matches!(e, Error::Unsupported { .. })
}

/// Every positive-stride form on one shape; returns nothing, asserts inside.
fn all_forms(shape_full: &[usize], shape_opt: SvdShape) {
    let (m, n) = (shape_full[0], shape_full[1]);
    let batch = shape_full.get(2).copied().unwrap_or(1);
    let a = data_c(shape_full);
    let opts = SvdOptions::default().shape(shape_opt);
    let mut eng = CpuEngine::serial();

    // 1. owned erased
    let owned = tensor_c(shape_full, a.clone());
    let r_owned = owned.svd_with(&opts, &mut eng, Allow).unwrap();
    let s_ref = check(&r_owned, &a, m, n, batch, shape_opt);

    // 2. borrowed erased
    let read = TensorRead::from_tensor(&owned);
    let r_read = read.svd_with(&opts, &mut eng, Allow).unwrap();
    assert_close(&check(&r_read, &a, m, n, batch, shape_opt), &s_ref);

    // 3. typed owned
    let typed = typed_c(shape_full, a.clone());
    let r_typed = erase(typed.svd_with(&opts, &mut eng, Allow).unwrap());
    assert_close(&check(&r_typed, &a, m, n, batch, shape_opt), &s_ref);

    // 4. typed view
    let view = TypedTensorView::from_col_major(shape_full, &a).unwrap();
    let r_view = erase(view.svd_with(&opts, &mut eng, Allow).unwrap());
    assert_close(&check(&r_view, &a, m, n, batch, shape_opt), &s_ref);

    // 5. transposed view: store B = A^T (swap first two axes), view it back.
    let mut bshape = shape_full.to_vec();
    bshape.swap(0, 1);
    let mut b = vec![C::new(0.0, 0.0); a.len()];
    for bt in 0..batch {
        for i in 0..m {
            for j in 0..n {
                b[j + i * n + bt * m * n] = a[i + j * m + bt * m * n];
            }
        }
    }
    let bview = TypedTensorView::from_col_major(&bshape, &b).unwrap();
    let mut axes: Vec<usize> = (0..shape_full.len()).collect();
    axes.swap(0, 1);
    let tview = bview.transpose_view(axes).unwrap();
    assert_eq!(tview.shape(), shape_full);
    assert_eq!(logical(&tview), a);
    let r_t = erase(tview.svd_with(&opts, &mut eng, Allow).unwrap());
    assert_close(&check(&r_t, &a, m, n, batch, shape_opt), &s_ref);
    // same through the erased borrowed form
    let r_t2 = tview
        .clone()
        .into_tensor_read()
        .unwrap()
        .svd_with(&opts, &mut eng, Allow)
        .unwrap();
    assert_close(&check(&r_t2, &a, m, n, batch, shape_opt), &s_ref);

    // 6. direct typed route (no erase/downcast): positive strides work.
    let r_direct = erase(svd_typed_view_direct(&view, shape_opt, &mut eng, Allow).unwrap());
    assert_close(&check(&r_direct, &a, m, n, batch, shape_opt), &s_ref);
    let r_direct_t = erase(svd_typed_view_direct(&tview, shape_opt, &mut eng, Allow).unwrap());
    assert_close(&check(&r_direct_t, &a, m, n, batch, shape_opt), &s_ref);
}

#[test]
fn c64_thin_tall_3x2_all_forms() {
    all_forms(&[3, 2], SvdShape::Thin);
}
#[test]
fn c64_thin_wide_2x3_all_forms() {
    all_forms(&[2, 3], SvdShape::Thin);
}
#[test]
fn c64_thin_batch_3x2x2_all_forms() {
    all_forms(&[3, 2, 2], SvdShape::Thin);
}
#[test]
fn c64_full_tall_3x2_all_forms() {
    all_forms(&[3, 2], SvdShape::Full);
}
#[test]
fn c64_full_wide_2x3_all_forms() {
    all_forms(&[2, 3], SvdShape::Full);
}
#[test]
fn c64_full_batch_2x3x2_all_forms() {
    all_forms(&[2, 3, 2], SvdShape::Full);
}

#[test]
fn typed_outputs_have_real_s_at_the_type_level() {
    let a = typed_c(&[3, 2], data_c(&[3, 2]));
    let mut eng = CpuEngine::serial();
    let r = a.svd(&mut eng).unwrap();
    // Compile-time: s is TypedTensor<f64>, u/vh are TypedTensor<Complex64>.
    let _: &TypedTensor<f64> = &r.s;
    let _: &TypedTensor<C> = &r.u;
    let (u, s, vh) = r.into_parts();
    assert_eq!(
        (u.shape(), s.shape(), vh.shape()),
        (&[3, 2][..], &[2][..], &[2, 2][..])
    );
}

#[test]
fn f64_thin_and_full_typed_and_erased() {
    let shape = [3usize, 2];
    let data: Vec<f64> = (0..6)
        .map(|k| ((k * k) as f64 * 0.37 + 1.0).sin())
        .collect();
    let mut eng = CpuEngine::serial();
    let typed = TypedTensor::<f64>::from_vec_col_major(shape.to_vec(), data.clone()).unwrap();
    let r = typed.svd(&mut eng).unwrap();
    // For real input Svd<V, R = V> degenerates to one type.
    let _: &Svd<TypedTensor<f64>, TypedTensor<f64>> = &r;
    let erased = Tensor::from_typed::<f64>(typed);
    let re = erased.svd(&mut eng).unwrap();
    assert_eq!(re.s.dtype(), DType::F64);
    let st = r.s.map_read().unwrap().to_vec();
    assert_close(&st, f64::as_slice(&re.s).unwrap());
    // reconstruction (thin)
    let (u, s, vh) = (
        r.u.map_read().unwrap().to_vec(),
        st,
        r.vh.map_read().unwrap().to_vec(),
    );
    for i in 0..3 {
        for j in 0..2 {
            let acc: f64 = (0..2).map(|l| u[i + l * 3] * s[l] * vh[l + j * 2]).sum();
            assert!((acc - data[i + j * 3]).abs() < 1e-12);
        }
    }
    // full
    let full = erased
        .svd_with(
            &SvdOptions::default().shape(SvdShape::Full),
            &mut eng,
            Allow,
        )
        .unwrap();
    assert_eq!(full.u.shape(), &[3, 3]);
    assert_eq!(full.vh.shape(), &[2, 2]);
    assert_eq!(full.s.shape(), &[2]);
}

#[test]
fn negative_stride_view_allow_route_gathers_direct_route_refuses() {
    let (m, n) = (3usize, 2usize);
    let a = data_c(&[m, n]);
    // rows reversed: (i, j) -> a[m-1-i, j]
    let view = TypedTensorView::from_slice(vec![m, n], vec![-1, m as isize], (m - 1) as isize, &a)
        .unwrap();
    let expected = logical(&view);
    assert_eq!(expected[0], a[m - 1]);
    let mut eng = CpuEngine::serial();

    // Erased borrowed read: gathers (the `*_read` rule).
    let read = view.clone().into_tensor_read().unwrap();
    let r_read = read.svd(&mut eng).unwrap();
    check(&r_read, &expected, m, n, 1, SvdShape::Thin);

    // Typed view through the one implementation: same gather.
    let r_view = erase(view.svd(&mut eng).unwrap());
    check(&r_view, &expected, m, n, 1, SvdShape::Thin);

    // Direct typed route (`svd_with_exec` rule): negative stride is Unsupported,
    // even though policy is Allow. Two input rules survive behind one grammar.
    let err = svd_typed_view_direct(&view, SvdShape::Thin, &mut eng, Allow).unwrap_err();
    assert!(is_unsupported(&err), "got {err:?}");
}

#[test]
fn full_with_non_raw_gauge_is_not_expressible_today() {
    let a = tensor_c(&[3, 2], data_c(&[3, 2]));
    let mut eng = CpuEngine::serial();
    let opts = SvdOptions::default()
        .shape(SvdShape::Full)
        .gauge(SvdGauge::CanonicalPivot);
    let err = a.svd_with(&opts, &mut eng, Allow).unwrap_err();
    assert!(is_unsupported(&err), "got {err:?}");
    assert!(err.to_string().contains("Full"));

    // Thin + the same gauge works, so the gap is exactly the combination.
    let ok = a
        .svd_with(
            &SvdOptions::default().gauge(SvdGauge::CanonicalPivot),
            &mut eng,
            Allow,
        )
        .unwrap();
    assert_eq!(ok.u.shape(), &[3, 2]);

    // Underlying API shape: `svd_full_read` takes no options at all, and
    // `svd_with_options_read` is thin-only (shape [3,2], not [3,3]).
    let thin_gauge = TensorReadLinalgExt::svd_with_options_read(
        TensorRead::from_tensor(&a),
        LinalgSvdOptions::default().gauge(SvdGauge::CanonicalPivot),
        &mut eng,
    )
    .unwrap();
    assert_eq!(thin_gauge.0.shape(), &[3, 2]);
}

#[test]
fn derivative_eps_is_validated_by_thin_path_only_in_current_linalg() {
    let a = tensor_c(&[3, 2], data_c(&[3, 2]));
    let mut eng = CpuEngine::serial();
    // Current linalg: thin + bad eps is refused...
    assert!(TensorReadLinalgExt::svd_with_options_read(
        TensorRead::from_tensor(&a),
        LinalgSvdOptions::default().derivative_eps(-1.0),
        &mut eng,
    )
    .is_err());
    // ...while svd_full_read has no eps parameter, so nothing could validate it.
    // The adapter validates once up front so both shapes agree.
    for shape in [SvdShape::Thin, SvdShape::Full] {
        let opts = SvdOptions::default().shape(shape).derivative_eps(-1.0);
        let err = a.svd_with(&opts, &mut eng, Allow).unwrap_err();
        assert!(matches!(err, Error::Validation { .. }), "got {err:?}");
    }
}

#[test]
fn driver_option_is_accepted_but_has_no_effect_on_svd_today() {
    let shape = [3usize, 2];
    let a = tensor_c(&shape, data_c(&shape));
    let mut eng = CpuEngine::serial();
    let base = a.svd(&mut eng).unwrap();
    for driver in [SvdDriver::Jacobi, SvdDriver::QrBased] {
        let r = a
            .svd_with(&SvdOptions::default().driver(driver), &mut eng, Allow)
            .unwrap();
        // Bit-identical output: the CPU backend's svd_with_options_read ignores `driver`.
        assert_eq!(C::as_slice(&r.u).unwrap(), C::as_slice(&base.u).unwrap());
        assert_eq!(
            f64::as_slice(&r.s).unwrap(),
            f64::as_slice(&base.s).unwrap()
        );
    }
}

#[test]
fn forbid_is_refused_with_typed_unsupported_for_every_form() {
    let a = data_c(&[3, 2]);
    let owned = tensor_c(&[3, 2], a.clone());
    let typed = typed_c(&[3, 2], a.clone());
    let view = TypedTensorView::from_col_major(&[3, 2], &a).unwrap();
    let mut eng = CpuEngine::serial();
    let o = SvdOptions::default();
    let errs = [
        owned.svd_with(&o, &mut eng, Forbid).unwrap_err(),
        TensorRead::from_tensor(&owned)
            .svd_with(&o, &mut eng, Forbid)
            .unwrap_err(),
        typed.svd_with(&o, &mut eng, Forbid).unwrap_err(),
        view.svd_with(&o, &mut eng, Forbid).unwrap_err(),
        svd_typed_view_direct(&view, SvdShape::Thin, &mut eng, Forbid).unwrap_err(),
    ];
    for e in &errs {
        assert!(is_unsupported(e), "got {e:?}");
        assert!(e.to_string().contains("not implemented"), "{e}");
    }
}

#[test]
fn downcast_is_checked_not_assumed() {
    let t = tensor_c(&[2, 2], data_c(&[2, 2]));
    let err = downcast_checked::<f64>(t).unwrap_err();
    assert!(matches!(err, Error::Validation { .. }), "got {err:?}");
    let t = tensor_c(&[2, 2], data_c(&[2, 2]));
    assert!(downcast_checked::<C>(t).is_ok());
}

#[test]
fn rank_one_input_is_a_validation_error() {
    let v = tensor_c(&[3], data_c(&[3]));
    let mut eng = CpuEngine::serial();
    let err = v.svd(&mut eng).unwrap_err();
    assert!(matches!(err, Error::Validation { .. }), "got {err:?}");
}
