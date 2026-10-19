use std::sync::Mutex;

use num_complex::{Complex32, Complex64};
use tenferro_tensor::{MemoryKind, Placement};

use super::*;
use tenferro_tensor::TensorRead;

static MATERIALIZATION_TEST_LOCK: Mutex<()> = Mutex::new(());

fn run_blas1(threads: usize, mut f: impl FnMut(&mut CpuEngine)) {
    let mut backend = crate::test_support::cpu_engine(threads).unwrap();
    f(&mut backend);
}

fn vdot(backend: &mut CpuEngine, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    crate::test_support::on_engine(backend, |session| {
        session.vdot_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
    })
    .unwrap()
    .unwrap()
}

fn norm_squared(backend: &mut CpuEngine, input: &Tensor) -> Tensor {
    crate::test_support::on_engine(backend, |session| {
        session.norm_squared_read(TensorRead::from_tensor(input))
    })
    .unwrap()
    .unwrap()
}

fn axpby(
    backend: &mut CpuEngine,
    alpha: ContractionScalar,
    x: &Tensor,
    beta: ContractionScalar,
    y: &mut Tensor,
) {
    crate::test_support::on_engine(backend, |session| {
        session.axpby_read_into_accum(
            alpha,
            TensorRead::from_tensor(x),
            beta,
            TensorWrite::from_tensor(y),
        )
    })
    .unwrap()
    .unwrap();
}

#[test]
fn blas1_compact_f32_f64_c32_c64_matches_scalar_references() {
    let _guard = MATERIALIZATION_TEST_LOCK.lock().unwrap();
    for threads in [1, 2] {
        run_blas1(threads, |backend| {
            crate::blas1::reset_materializations_for_test();
            let xf = Tensor::from_vec_col_major(vec![4], vec![1.0_f32, -2.0, 3.0, 4.0]).unwrap();
            let yf = Tensor::from_vec_col_major(vec![4], vec![5.0_f32, 6.0, -7.0, 8.0]).unwrap();
            assert_eq!(vdot(backend, &xf, &yf).as_slice::<f32>().unwrap(), &[4.0]);
            assert_eq!(
                norm_squared(backend, &xf).as_slice::<f32>().unwrap(),
                &[30.0]
            );
            let mut out = Tensor::from_vec_col_major(vec![4], vec![10.0_f32; 4]).unwrap();
            axpby(
                backend,
                ContractionScalar::F32(2.0),
                &xf,
                ContractionScalar::F32(0.5),
                &mut out,
            );
            assert_eq!(out.as_slice::<f32>().unwrap(), &[7.0, 1.0, 11.0, 13.0]);

            let xd = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, -2.0, 3.0, 4.0]).unwrap();
            let yd = Tensor::from_vec_col_major(vec![4], vec![5.0_f64, 6.0, -7.0, 8.0]).unwrap();
            assert_eq!(vdot(backend, &xd, &yd).as_slice::<f64>().unwrap(), &[4.0]);
            assert_eq!(
                norm_squared(backend, &xd).as_slice::<f64>().unwrap(),
                &[30.0]
            );
            let mut out = Tensor::from_vec_col_major(vec![4], vec![10.0_f64; 4]).unwrap();
            axpby(
                backend,
                ContractionScalar::F64(2.0),
                &xd,
                ContractionScalar::F64(0.5),
                &mut out,
            );
            assert_eq!(out.as_slice::<f64>().unwrap(), &[7.0, 1.0, 11.0, 13.0]);

            let xc = Tensor::from_vec_col_major(
                vec![2],
                vec![Complex32::new(1.0, 2.0), Complex32::new(-3.0, 4.0)],
            )
            .unwrap();
            let yc = Tensor::from_vec_col_major(
                vec![2],
                vec![Complex32::new(5.0, -1.0), Complex32::new(2.0, 3.0)],
            )
            .unwrap();
            let expected = xc
                .as_slice::<Complex32>()
                .unwrap()
                .iter()
                .zip(yc.as_slice::<Complex32>().unwrap())
                .map(|(x, y)| x.conj() * y)
                .fold(Complex32::new(0.0, 0.0), |a, b| a + b);
            let actual = vdot(backend, &xc, &yc).as_slice::<Complex32>().unwrap()[0];
            assert!((actual - expected).norm() < 1.0e-5);
            assert_eq!(
                norm_squared(backend, &xc).as_slice::<f32>().unwrap(),
                &[30.0]
            );
            let mut out =
                Tensor::from_vec_col_major(vec![2], vec![Complex32::new(10.0, 1.0); 2]).unwrap();
            axpby(
                backend,
                ContractionScalar::C32(Complex32::new(2.0, 0.0)),
                &xc,
                ContractionScalar::C32(Complex32::new(0.5, 0.0)),
                &mut out,
            );
            assert_eq!(
                out.as_slice::<Complex32>().unwrap(),
                &[Complex32::new(7.0, 4.5), Complex32::new(-1.0, 8.5)]
            );

            let xc = Tensor::from_vec_col_major(
                vec![2],
                vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
            )
            .unwrap();
            let yc = Tensor::from_vec_col_major(
                vec![2],
                vec![Complex64::new(5.0, -1.0), Complex64::new(2.0, 3.0)],
            )
            .unwrap();
            let expected = xc
                .as_slice::<Complex64>()
                .unwrap()
                .iter()
                .zip(yc.as_slice::<Complex64>().unwrap())
                .map(|(x, y)| x.conj() * y)
                .fold(Complex64::new(0.0, 0.0), |a, b| a + b);
            let actual = vdot(backend, &xc, &yc).as_slice::<Complex64>().unwrap()[0];
            assert!((actual - expected).norm() < 1.0e-12);
            assert_eq!(
                norm_squared(backend, &xc).as_slice::<f64>().unwrap(),
                &[30.0]
            );
            let mut out =
                Tensor::from_vec_col_major(vec![2], vec![Complex64::new(10.0, 1.0); 2]).unwrap();
            axpby(
                backend,
                ContractionScalar::C64(Complex64::new(2.0, 0.0)),
                &xc,
                ContractionScalar::C64(Complex64::new(0.5, 0.0)),
                &mut out,
            );
            assert_eq!(
                out.as_slice::<Complex64>().unwrap(),
                &[Complex64::new(7.0, 4.5), Complex64::new(-1.0, 8.5)]
            );
            assert_eq!(crate::blas1::materializations_for_test(), 0);
        });
    }
}

#[test]
fn blas1_compact_view_destinations_cover_all_supported_dtypes() {
    let mut backend = crate::test_support::cpu_engine(1).unwrap();

    macro_rules! check {
        ($ty:ty, $tensor_variant:ident, $view_variant:ident, $scalar_variant:ident, $one:expr, $two:expr) => {{
            let x = Tensor::from_typed::<preset_scalar!($tensor_variant)>(
                TypedTensor::<$ty>::from_vec_col_major(vec![2], vec![$one, $two]).unwrap(),
            );
            let mut storage = vec![$two, $one];
            let view = TypedTensorViewMut::from_slice([2], [1], 0, &mut storage).unwrap();
            crate::test_support::on_engine(&mut backend, |session| {
                session.axpby_read_into_accum(
                    ContractionScalar::$scalar_variant($one),
                    TensorRead::from_tensor(&x),
                    ContractionScalar::$scalar_variant($one),
                    TensorWrite::from_view(TensorViewMut::$view_variant(view)),
                )
            })
            .unwrap()
            .unwrap();
            assert_eq!(storage, vec![$one + $two, $one + $two]);
        }};
    }

    check!(f32, F32, F32, F32, 1.0_f32, 2.0_f32);
    check!(f64, F64, F64, F64, 1.0_f64, 2.0_f64);
    check!(
        Complex32,
        C32,
        C32,
        C32,
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0)
    );
    check!(
        Complex64,
        C64,
        C64,
        C64,
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0)
    );

    let empty_x = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap();
    let mut empty_storage = Vec::<f64>::new();
    let empty_view = TypedTensorViewMut::from_slice([0], [1], 0, &mut empty_storage).unwrap();
    crate::test_support::on_engine(&mut backend, |session| {
        session.axpby_read_into_accum(
            ContractionScalar::F64(1.0),
            TensorRead::from_tensor(&empty_x),
            ContractionScalar::F64(1.0),
            TensorWrite::from_view(TensorViewMut::F64(empty_view)),
        )
    })
    .unwrap()
    .unwrap();
}

#[test]
fn blas1_rankn_and_empty_inputs_reduce_all_axes() {
    let mut backend = crate::test_support::cpu_engine(2).unwrap();
    let x = Tensor::from_vec_col_major(vec![2, 2, 2], (1..=8).map(f64::from).collect()).unwrap();
    let y = Tensor::from_vec_col_major(vec![2, 2, 2], vec![2.0_f64; 8]).unwrap();
    assert_eq!(
        vdot(&mut backend, &x, &y).as_slice::<f64>().unwrap(),
        &[72.0]
    );
    assert_eq!(
        norm_squared(&mut backend, &x).as_slice::<f64>().unwrap(),
        &[204.0]
    );

    let empty = Tensor::from_vec_col_major(vec![0, 3, 2], Vec::<f64>::new()).unwrap();
    assert_eq!(
        vdot(&mut backend, &empty, &empty)
            .as_slice::<f64>()
            .unwrap(),
        &[0.0]
    );
    assert_eq!(
        norm_squared(&mut backend, &empty)
            .as_slice::<f64>()
            .unwrap(),
        &[0.0]
    );
    let mut out = Tensor::from_vec_col_major(vec![0, 3, 2], Vec::<f64>::new()).unwrap();
    axpby(
        &mut backend,
        ContractionScalar::F64(2.0),
        &empty,
        ContractionScalar::F64(3.0),
        &mut out,
    );
    assert!(out.as_slice::<f64>().unwrap().is_empty());
}

#[test]
fn blas1_strided_reads_are_supported_and_x_is_materialized_once() {
    let _guard = MATERIALIZATION_TEST_LOCK.lock().unwrap();
    let lhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let rhs =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
            .unwrap();
    let lhs_view = lhs.as_view().transpose_view([1, 0]).unwrap();
    let rhs_view = rhs.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = crate::test_support::cpu_engine(2).unwrap();

    let dot = crate::test_support::on_engine(&mut backend, |session| {
        session.vdot_read(
            TensorRead::from_view(TensorView::F64(lhs_view.clone())),
            TensorRead::from_view(TensorView::F64(rhs_view)),
        )
    })
    .unwrap()
    .unwrap();
    assert_eq!(dot.as_slice::<f64>().unwrap(), &[182.0]);
    let norm = crate::test_support::on_engine(&mut backend, |session| {
        session.norm_squared_read(TensorRead::from_view(TensorView::F64(lhs_view.clone())))
    })
    .unwrap()
    .unwrap();
    assert_eq!(norm.as_slice::<f64>().unwrap(), &[91.0]);

    let mut out = Tensor::from_vec_col_major(vec![3, 2], vec![10.0_f64; 6]).unwrap();
    crate::blas1::reset_materializations_for_test();
    crate::test_support::on_engine(&mut backend, |session| {
        session.axpby_read_into_accum(
            ContractionScalar::F64(2.0),
            TensorRead::from_view(TensorView::F64(lhs_view.clone())),
            ContractionScalar::F64(0.5),
            TensorWrite::from_tensor(&mut out),
        )
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        &[7.0, 11.0, 15.0, 9.0, 13.0, 17.0]
    );
    // The compact destination keeps the dense `strided_kernel::axpby_accum`
    // path, whose compact source slice forces one staging pass for a strided
    // `x`. A strided destination instead runs the pinned update primitive,
    // which consumes the strided source directly, so it stages nothing.
    assert_eq!(crate::blas1::materializations_for_test(), 1);

    let mut strided_storage = vec![10.0_f64; 7];
    let strided_destination =
        TypedTensorViewMut::from_slice(vec![3, 2], vec![1, 4], 0, &mut strided_storage).unwrap();
    crate::blas1::reset_materializations_for_test();
    crate::test_support::on_engine(&mut backend, |session| {
        session.axpby_read_into_accum(
            ContractionScalar::F64(2.0),
            TensorRead::from_view(TensorView::F64(lhs_view)),
            ContractionScalar::F64(0.5),
            TensorWrite::from_view(TensorViewMut::F64(strided_destination)),
        )
    })
    .unwrap()
    .unwrap();
    assert_eq!(crate::blas1::materializations_for_test(), 0);
}

/// A strided/offset destination region inside a larger allocation accumulates
/// in place without touching the surrounding bytes.
///
/// The 2x2 region starts at offset 5 of a `[4, 3]` column-major allocation with
/// strides `(1, 4)`, so it is neither compact nor a contiguous span. The
/// reference is the same stride formula the region view encodes, and the whole
/// allocation is compared byte-for-byte afterwards.
#[test]
fn blas1_strided_destination_region_preserves_surrounding_bytes() {
    let mut backend = crate::test_support::cpu_engine(2).unwrap();
    // Column-major `[4, 3]` strides; region rows 0..2, columns 1..3.
    let region = [5_usize, 6, 9, 10];
    let alpha = 2.0_f64;
    let beta = 0.5_f64;

    let reference = |storage: &mut [f64], x: &[f64]| {
        for (position, &index) in region.iter().enumerate() {
            storage[index] = alpha * x[position] + beta * storage[index];
        }
    };

    // Case 1: compact x, non-compact destination.
    let mut storage: Vec<f64> = (0..12).map(|index| 10.0 + f64::from(index)).collect();
    let mut expected = storage.clone();
    let compact_x = [1.0_f64, 2.0, 3.0, 4.0];
    reference(&mut expected, &compact_x);

    let x = Tensor::from_vec_col_major(vec![2, 2], compact_x.to_vec()).unwrap();
    let destination =
        TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 4], 5, &mut storage).unwrap();
    crate::blas1::reset_materializations_for_test();
    crate::test_support::on_engine(&mut backend, |session| {
        session.axpby_read_into_accum(
            ContractionScalar::F64(alpha),
            TensorRead::from_tensor(&x),
            ContractionScalar::F64(beta),
            TensorWrite::from_view(TensorViewMut::F64(destination)),
        )
    })
    .unwrap()
    .unwrap();
    assert_eq!(storage, expected);
    assert_eq!(crate::blas1::materializations_for_test(), 0);

    // Case 2: strided x, non-compact destination. The strided path consumes the
    // source in place, so neither operand is staged.
    let mut storage: Vec<f64> = (0..12).map(|index| 20.0 + f64::from(index)).collect();
    let mut expected = storage.clone();
    let source: Vec<f64> = (0..12).map(|index| f64::from(index) - 6.0).collect();
    let source_before = source.clone();
    let source_region: Vec<f64> = [1_usize, 2, 5, 6].iter().map(|&i| source[i]).collect();
    reference(&mut expected, &source_region);

    let source_view = TypedTensorView::from_slice(vec![2, 2], vec![1, 4], 1, &source).unwrap();
    let destination =
        TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 4], 5, &mut storage).unwrap();
    crate::blas1::reset_materializations_for_test();
    crate::test_support::on_engine(&mut backend, |session| {
        session.axpby_read_into_accum(
            ContractionScalar::F64(alpha),
            TensorRead::from_view(TensorView::F64(source_view)),
            ContractionScalar::F64(beta),
            TensorWrite::from_view(TensorViewMut::F64(destination)),
        )
    })
    .unwrap()
    .unwrap();
    assert_eq!(storage, expected);
    assert_eq!(source, source_before);
    assert_eq!(crate::blas1::materializations_for_test(), 0);
}

fn assert_axpby_rejects_without_mutating(
    backend: &mut CpuEngine,
    alpha: ContractionScalar,
    x: &Tensor,
    beta: ContractionScalar,
    y: &mut Tensor,
) {
    let before = y.as_slice::<f64>().unwrap().to_vec();
    assert!(crate::test_support::on_engine(backend, |session| {
        session.axpby_read_into_accum(
            alpha,
            TensorRead::from_tensor(x),
            beta,
            TensorWrite::from_tensor(y),
        )
    })
    .unwrap()
    .is_err());
    assert_eq!(y.as_slice::<f64>().unwrap(), before.as_slice());
}

#[test]
fn blas1_invalid_axpby_requests_leave_y_byte_identical() {
    let mut backend = crate::test_support::cpu_engine(1).unwrap();
    let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut y = Tensor::from_vec_col_major(vec![2], vec![10.0_f64, 20.0]).unwrap();

    let integer = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    assert_axpby_rejects_without_mutating(
        &mut backend,
        ContractionScalar::F64(1.0),
        &integer,
        ContractionScalar::F64(1.0),
        &mut y,
    );
    let boolean = Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    assert_axpby_rejects_without_mutating(
        &mut backend,
        ContractionScalar::F64(1.0),
        &boolean,
        ContractionScalar::F64(1.0),
        &mut y,
    );
    assert_axpby_rejects_without_mutating(
        &mut backend,
        ContractionScalar::F32(1.0),
        &x,
        ContractionScalar::F64(1.0),
        &mut y,
    );
    let wrong_shape = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    assert_axpby_rejects_without_mutating(
        &mut backend,
        ContractionScalar::F64(1.0),
        &wrong_shape,
        ContractionScalar::F64(1.0),
        &mut y,
    );

    let mut placed_x = x.duplicate().unwrap();
    if let Some(tensor) = placed_x.as_typed_mut::<f64>() {
        tensor.set_placement(Placement {
            memory_kind: MemoryKind::PinnedHost,
            device: None,
            cpu_affinity: None,
        });
    }
    assert_axpby_rejects_without_mutating(
        &mut backend,
        ContractionScalar::F64(1.0),
        &placed_x,
        ContractionScalar::F64(1.0),
        &mut y,
    );

    let mut device_x = x.duplicate().unwrap();
    if let Some(tensor) = device_x.as_typed_mut::<f64>() {
        tensor.set_placement(Placement {
            memory_kind: MemoryKind::Device,
            device: None,
            cpu_affinity: None,
        });
    }
    let mut device_y = y.duplicate().unwrap();
    if let Some(tensor) = device_y.as_typed_mut::<f64>() {
        tensor.set_placement(Placement {
            memory_kind: MemoryKind::Device,
            device: None,
            cpu_affinity: None,
        });
    }
    assert_axpby_rejects_without_mutating(
        &mut backend,
        ContractionScalar::F64(1.0),
        &device_x,
        ContractionScalar::F64(1.0),
        &mut device_y,
    );

    let mut noncompact_storage = vec![10.0_f64, 20.0, 30.0, 40.0];
    let before = noncompact_storage.clone();
    // The shape and dtype rejections below keep a strided/offset view
    // destination: relaxing compactness must not weaken the checks that still
    // apply, and an invalid request must leave the whole allocation untouched.
    let wrong_shape_x = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    let wrong_shape_y =
        TypedTensorViewMut::from_slice(vec![2], vec![2], 0, &mut noncompact_storage).unwrap();
    assert!(crate::test_support::on_engine(&mut backend, |session| {
        session.axpby_read_into_accum(
            ContractionScalar::F64(1.0),
            TensorRead::from_tensor(&wrong_shape_x),
            ContractionScalar::F64(1.0),
            TensorWrite::from_view(TensorViewMut::F64(wrong_shape_y)),
        )
    })
    .unwrap()
    .is_err());
    assert_eq!(noncompact_storage, before);

    let wrong_dtype_x = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let wrong_dtype_y =
        TypedTensorViewMut::from_slice(vec![2], vec![2], 0, &mut noncompact_storage).unwrap();
    assert!(crate::test_support::on_engine(&mut backend, |session| {
        session.axpby_read_into_accum(
            ContractionScalar::F32(1.0),
            TensorRead::from_tensor(&wrong_dtype_x),
            ContractionScalar::F32(1.0),
            TensorWrite::from_view(TensorViewMut::F64(wrong_dtype_y)),
        )
    })
    .unwrap()
    .is_err());
    assert_eq!(noncompact_storage, before);
}

#[test]
fn blas1_vdot_and_norm_reject_invalid_metadata_and_placement() {
    let mut backend = crate::CpuEngine::serial();
    let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let wrong_shape = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    let wrong_dtype = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    for rhs in [&wrong_shape, &wrong_dtype] {
        assert!(
            crate::test_support::on_engine(&mut backend, |session| session
                .vdot_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(rhs),))
            .unwrap()
            .is_err()
        );
    }

    let mut device = x.duplicate().unwrap();
    if let Some(tensor) = device.as_typed_mut::<f64>() {
        tensor.set_placement(Placement {
            memory_kind: MemoryKind::Device,
            device: None,
            cpu_affinity: None,
        });
    }
    assert!(
        crate::test_support::on_engine(&mut backend, |session| session.vdot_read(
            TensorRead::from_tensor(&device),
            TensorRead::from_tensor(&device),
        ))
        .unwrap()
        .is_err()
    );
    assert!(
        crate::test_support::on_engine(&mut backend, |session| session
            .norm_squared_read(TensorRead::from_tensor(&device)))
        .unwrap()
        .is_err()
    );
}

#[test]
fn blas1_norm_rejects_integer_and_bool_without_allocating_a_result() {
    let mut backend = crate::CpuEngine::serial();
    for input in [
        Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
    ] {
        let result = crate::test_support::on_engine(&mut backend, |session| {
            session.norm_squared_read(TensorRead::from_tensor(&input))
        })
        .unwrap();
        assert!(matches!(result, Err(Error::Unsupported { .. })));
    }
}

#[test]
fn blas1_cg_microfixture_uses_session_primitives_without_element_loops() {
    let mut backend = crate::test_support::cpu_engine(2).unwrap();
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 0.0, 0.0, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 2.0]).unwrap();
    let mut x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0]).unwrap();
    let mut r = b.duplicate().unwrap();
    let mut p = r.duplicate().unwrap();
    let mut ap = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let mut rs = norm_squared(&mut backend, &r);

    for _ in 0..2 {
        crate::test_support::on_engine(&mut backend, |__s| {
            __s.copy_read_into(
                TensorRead::from_tensor(&p),
                TensorWrite::from_tensor(&mut ap),
            )
        })
        .unwrap()
        .unwrap();
        let computed_ap = crate::test_support::on_engine(&mut backend, |__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&ap),
                &DotGeneralConfig {
                    lhs_contracting_dims: [1].as_slice().into(),
                    rhs_contracting_dims: [0].as_slice().into(),
                    lhs_batch_dims: [].as_slice().into(),
                    rhs_batch_dims: [].as_slice().into(),
                },
            )
        })
        .unwrap()
        .unwrap();
        crate::test_support::on_engine(&mut backend, |__s| {
            __s.copy_read_into(
                TensorRead::from_tensor(&computed_ap),
                TensorWrite::from_tensor(&mut ap),
            )
        })
        .unwrap()
        .unwrap();
        let rs_value = rs.as_slice::<f64>().unwrap()[0];
        let pap = vdot(&mut backend, &p, &ap).as_slice::<f64>().unwrap()[0];
        let alpha = rs_value / pap;
        axpby(
            &mut backend,
            ContractionScalar::F64(alpha),
            &p,
            ContractionScalar::F64(1.0),
            &mut x,
        );
        axpby(
            &mut backend,
            ContractionScalar::F64(-alpha),
            &ap,
            ContractionScalar::F64(1.0),
            &mut r,
        );
        let next_rs = norm_squared(&mut backend, &r);
        let beta = next_rs.as_slice::<f64>().unwrap()[0] / rs_value;
        axpby(
            &mut backend,
            ContractionScalar::F64(1.0),
            &r,
            ContractionScalar::F64(beta),
            &mut p,
        );
        rs = next_rs;
    }

    assert!((x.as_slice::<f64>().unwrap()[0] - 1.0).abs() < 1.0e-12);
    assert!((x.as_slice::<f64>().unwrap()[1] - 1.0).abs() < 1.0e-12);
}
