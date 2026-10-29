//! Lower-provider error mapping for the contraction boundary.
//!
//! The new provider errors (`Workspace`, declined strategies) must reach the
//! tensor boundary as typed sources, not as Internal or string-only failures.

use std::error::Error as _;

use super::LowerContractionError;
use tenferro_tensor::{Error as TensorError, ErrorKind};

fn has_source<T: std::error::Error + 'static>(error: &TensorError) -> bool {
    error
        .source()
        .and_then(|source| source.downcast_ref::<T>())
        .is_some()
}

#[test]
fn cpueinsum_workspace_is_a_typed_runtime_state_failure() {
    let error = cpueinsum::Error::Workspace {
        needed: 16,
        have: 8,
    }
    .into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::RuntimeState);
    assert!(has_source::<cpueinsum::Error>(&error));
}

#[test]
fn cpueinsum_declined_route_keeps_its_typed_source() {
    let declined = cpueinsum::Error::Contract {
        step: 0,
        source: cpueinsum::tcontract::Error::Unsupported(
            cpueinsum::tcontract::api::Unsupported::Reason("test decline"),
        ),
    };
    let error = declined.into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(has_source::<cpueinsum::Error>(&error));
}

#[test]
fn bare_tcontract_decline_keeps_its_typed_source() {
    let error = cpueinsum::tcontract::Error::Unsupported(
        cpueinsum::tcontract::api::Unsupported::Reason("test decline"),
    )
    .into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert!(has_source::<cpueinsum::tcontract::Error>(&error));
}

#[cfg(feature = "blas")]
#[test]
fn blas_workspace_is_a_typed_runtime_state_failure() {
    let error =
        cpueinsum_blas::BlasError::Work { need: 16, got: 8 }.into_crate_error("dot_general");
    assert_eq!(error.kind(), ErrorKind::RuntimeState);
    assert!(has_source::<cpueinsum_blas::BlasError>(&error));
}

/// Plan-slot reuse (#2314): the fallback paths of [`BinaryCache::plan`].
///
/// A miss rebuilds the slot's plan in place only when the stored plan is
/// exclusively owned and of the requested scalar type. These cover the two
/// fallbacks and the failed preparation that must leave the slot untouched.
#[cfg(any(feature = "native", feature = "blas"))]
mod plan_slot_reuse {
    use super::super::{
        BinaryCache, BinaryPlan, BufferPool, ExecutionResources, Key, Layout,
        LowerContractionError, Scalar, SliceAccumulationSource,
    };
    use cpueinsum::tcontract::api::DType;
    use std::sync::Arc;
    use tenferro_tensor::DotGeneralConfig;

    /// The rank-2 matmul config: `lhs[m, k] * rhs[k, n]`.
    fn matmul_config() -> DotGeneralConfig {
        DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        }
    }

    /// The cache key of a dense, column-major `lhs[m, k] * rhs[k_in, n]`.
    fn key(dtype: DType, (m, k, k_in, n): (usize, usize, usize, usize)) -> Key {
        let lhs_strides = [1isize, m as isize];
        let rhs_strides = [1isize, k_in as isize];
        let out_strides = [1isize, m as isize];
        Key {
            dtype,
            lhs: Layout::new(&[m, k], &lhs_strides, 0),
            rhs: Layout::new(&[k_in, n], &rhs_strides, 0),
            out: Layout::new(&[m, n], &out_strides, 0),
            lhs_conj: false,
            rhs_conj: false,
            serial: true,
        }
    }

    /// Run one prepared plan on dense column-major buffers.
    fn run<T: Scalar>(
        plan: &BinaryPlan<T>,
        alpha: T,
        beta: T,
        lhs: &[T],
        rhs: &[T],
        out: &mut [T],
    ) -> crate::Result<()> {
        let resources = ExecutionResources::new(None);
        let mut exec = resources.exec(Default::default());
        let mut buffers = BufferPool::new();
        #[cfg(feature = "blas")]
        let mut work = <T as crate::buffer_pool::PoolScalar>::pool_acquire_zeroed(
            &mut buffers,
            plan.work_len(),
        );
        #[cfg(feature = "native")]
        let _ = &mut buffers;
        plan.execute_slices_accum(
            &mut exec,
            alpha,
            (lhs, 0),
            (rhs, 0),
            beta,
            SliceAccumulationSource::Output,
            (out, 0),
            #[cfg(feature = "blas")]
            &mut work,
        )
        .map(|_| ())
        .map_err(|error| error.into_crate_error("dot_general"))
    }

    fn assert_matmul((m, k, n): (usize, usize, usize), (a, b, want): (&[f64], &[f64], &[f64])) {
        for l in 0..n {
            for i in 0..m {
                let mut sum = 0.0;
                for j in 0..k {
                    sum += a[i + m * j] * b[j + k * l];
                }
                assert!(
                    (want[i + m * l] - sum).abs() < 1e-12,
                    "output[{i}, {l}] = {} != {sum}",
                    want[i + m * l]
                );
            }
        }
    }

    /// A caller that still runs the old plan gets a fresh `Arc`; the plan it
    /// holds keeps its own geometry and the slot takes the new one.
    #[test]
    fn a_shared_plan_survives_a_different_key_miss() {
        let mut cache = BinaryCache::default();
        let config = matmul_config();
        let first_key = key(DType::F64, (2, 3, 3, 4));
        let second_key = key(DType::F64, (3, 2, 2, 2));
        let held = cache
            .plan::<f64>(Some(0), 1, first_key.clone(), &config)
            .expect("first plan");
        let stored = cache
            .plan::<f64>(Some(0), 1, second_key.clone(), &config)
            .expect("second plan");

        // The slot held the plan exclusively only while nobody ran it; the caller
        // still holds `held`, so the miss must allocate a new `Arc`.
        assert!(!Arc::ptr_eq(&held, &stored));
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(
            cache.slots[0].as_ref().map(|entry| entry.key.clone()),
            Some(second_key)
        );

        // The held plan still executes its own 2x3 * 3x4 geometry correctly.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 0.0, -1.0, 2.0, 0.5, 1.0, 3.0, -2.0, 1.0, 1.0, 0.0, 2.0];
        let mut want = [0.0f64; 8];
        run(&held, 1.0, 0.0, &a, &b, &mut want).expect("the held plan still runs");
        assert_matmul((2, 3, 4), (&a, &b, &want));

        // And so does the new plan the slot now holds.
        let a = [1.0, 2.0, -1.0, 0.5, 3.0, 4.0];
        let b = [2.0, -1.0, 0.5, 1.0];
        let mut want = [0.0f64; 6];
        run(&stored, 1.0, 0.0, &a, &b, &mut want).expect("the new plan runs");
        assert_matmul((3, 2, 2), (&a, &b, &want));
    }

    /// The same slot holding an exclusively owned plan of another scalar type
    /// allocates a new `Arc` rather than reinterpreting the old one.
    #[test]
    fn a_different_scalar_type_replaces_the_slot_plan() {
        let mut cache = BinaryCache::default();
        let config = matmul_config();
        let stored = cache
            .plan::<f64>(Some(0), 1, key(DType::F64, (2, 2, 2, 2)), &config)
            .expect("f64 plan");
        // Release the f64 plan so the slot owns it exclusively; the type
        // mismatch, not the refcount, is then what forces a new `Arc`.
        drop(stored);

        let stored_f32 = cache
            .plan::<f32>(Some(0), 1, key(DType::F32, (2, 2, 2, 2)), &config)
            .expect("f32 plan");
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [1.0f32, -1.0, 0.5, 2.0];
        let mut out = [0.0f32; 4];
        run(&stored_f32, 1.0, 0.0, &a, &b, &mut out).expect("the f32 plan runs");
        // Column-major: [[1, 3], [2, 4]] * [[1, 0.5], [-1, 2]].
        assert_eq!(out, [-2.0, -2.0, 6.5, 9.0]);
        assert_eq!(
            cache.slots[0].as_ref().map(|entry| entry.key.dtype),
            Some(DType::F32)
        );
    }

    /// A preparation that fails must not reach the slot: the previous entry
    /// stays byte-for-byte usable and only the miss counter moves.
    #[test]
    fn a_failed_preparation_leaves_the_previous_entry_usable() {
        let mut cache = BinaryCache::default();
        let config = matmul_config();
        let good_key = key(DType::F64, (2, 3, 3, 4));
        let plan = cache
            .plan::<f64>(Some(0), 1, good_key.clone(), &config)
            .expect("stored plan");
        let before = cache.stats();

        // Contracting 3 against 4 is rejected while the plan is prepared.
        let bad_key = key(DType::F64, (2, 3, 4, 2));
        let failed = cache.plan::<f64>(Some(0), 1, bad_key, &config);
        assert!(failed.is_err(), "a mismatched contraction must fail");
        let after = cache.stats();
        assert_eq!(after.misses, before.misses + 1);
        assert_eq!(after.hits, before.hits);
        assert_eq!(after.evictions, before.evictions);
        assert_eq!(
            cache.slots[0].as_ref().map(|entry| entry.key.clone()),
            Some(good_key)
        );

        // The untouched entry still holds the same, still-correct plan.
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut want = [0.0f64; 8];
        run(&plan, 1.0, 0.0, &a, &b, &mut want).expect("the previous plan still runs");
        assert_matmul((2, 3, 4), (&a, &b, &want));
    }
}

/// c64 products above tcontract's default `FaerLimit` volume (`1 << 17`) take
/// faer at a thread budget of one (tensor4all-rs#1040), and equal a direct
/// faer matmul bit for bit.
#[cfg(feature = "native")]
mod faer_at_budget_one {
    use crate::test_support::{cpu_engine, with_cpu_session};
    use tenferro_tensor::{Complex64, DotGeneralConfig, Tensor, TensorDot, TensorRead};

    fn config() -> DotGeneralConfig {
        DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [].as_slice().into(),
            rhs_batch_dims: [].as_slice().into(),
        }
    }

    fn data(len: usize, seed: usize) -> Vec<Complex64> {
        (0..len)
            .map(|i| {
                let x = (i * 2654435761 + seed * 40503) % 1009;
                Complex64::new(
                    x as f64 / 503.0 - 1.0,
                    ((x * 7 + i) % 211) as f64 / 105.0 - 1.0,
                )
            })
            .collect()
    }

    fn faer_product(
        a: &[Complex64],
        b: &[Complex64],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<Complex64> {
        use crate::faer::{c64, linalg::matmul::matmul, Accum, Mat, MatRef, Par};
        let a: Vec<c64> = a.iter().map(|z| c64::new(z.re, z.im)).collect();
        let b: Vec<c64> = b.iter().map(|z| c64::new(z.re, z.im)).collect();
        let mut out = Mat::<c64>::zeros(m, n);
        matmul(
            out.as_mut(),
            Accum::Replace,
            MatRef::from_column_major_slice(&a, m, k),
            MatRef::from_column_major_slice(&b, k, n),
            c64::new(1.0, 0.0),
            Par::Seq,
        );
        (0..n)
            .flat_map(|j| (0..m).map(move |i| (i, j)))
            .map(|(i, j)| {
                let z = out.as_ref().get(i, j);
                Complex64::new(z.re, z.im)
            })
            .collect()
    }

    fn run(engine: &mut crate::CpuEngine, a: &Tensor, b: &Tensor) -> Tensor {
        with_cpu_session(engine, |session| {
            session
                .dot_general_read(
                    TensorRead::from_tensor(a),
                    TensorRead::from_tensor(b),
                    &config(),
                )
                .unwrap()
        })
    }

    /// The #1040 shapes, all above the old `1 << 17` bound except `1x141x282`.
    #[test]
    fn serial_c64_takes_faer_and_matches_direct_faer_bitwise() {
        for (m, k, n) in [(1, 141, 282), (8, 141, 282), (141, 282, 141)] {
            let (av, bv) = (data(m * k, 1), data(k * n, 2));
            let a = Tensor::from_vec_col_major(vec![m, k], av.clone()).unwrap();
            let b = Tensor::from_vec_col_major(vec![k, n], bv.clone()).unwrap();
            let mut engine = cpu_engine(1).unwrap();
            let _ = crate::contraction::take_last_route_was_faer();
            let got = run(&mut engine, &a, &b);
            assert_eq!(
                crate::contraction::take_last_route_was_faer(),
                Some(true),
                "{m}x{k}x{n}: budget one must route to faer"
            );
            let want = faer_product(&av, &bv, m, k, n);
            let got = got.as_slice::<Complex64>().unwrap();
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.re.to_bits(), w.re.to_bits(), "{m}x{k}x{n}");
                assert_eq!(g.im.to_bits(), w.im.to_bits(), "{m}x{k}x{n}");
            }
        }
    }

    /// The budget class is part of the plan key: a plan prepared by a serial
    /// engine is not handed to a wider engine sharing the cache, and back.
    #[test]
    fn shared_plan_cache_separates_budget_classes() {
        let (m, k, n) = (8, 141, 282);
        let a = Tensor::from_vec_col_major(vec![m, k], data(m * k, 1)).unwrap();
        let b = Tensor::from_vec_col_major(vec![k, n], data(k * n, 2)).unwrap();
        let plans = crate::CpuPlanCache::new();
        let mut serial = cpu_engine(1).unwrap().with_plan_cache(plans.clone());
        let mut wide = cpu_engine(4).unwrap().with_plan_cache(plans);
        let _ = run(&mut serial, &a, &b);
        assert_eq!(crate::contraction::take_last_route_was_faer(), Some(true));
        let _ = run(&mut wide, &a, &b);
        assert_eq!(
            crate::contraction::take_last_route_was_faer(),
            Some(false),
            "a wider budget keeps tcontract's default FaerLimit"
        );
        let _ = run(&mut serial, &a, &b);
        assert_eq!(crate::contraction::take_last_route_was_faer(), Some(true));
    }
}
