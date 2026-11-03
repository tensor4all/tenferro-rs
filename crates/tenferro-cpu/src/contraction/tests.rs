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
            budget: 1,
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

/// How the c64 `FaerLimit` follows the host class and the thread budget.
///
/// tcontract's default sends c64 above `m*n*k = 1 << 17` to its packed driver.
/// That driver is host-dependent (tcontract-rs#156): on AVX2 at a thread budget
/// of one it never beats faer, while aarch64 (packed wins at `m = 16`) and
/// AVX-512 keep the bound. A plan is not re-selected per call and one
/// `CpuPlanCache` may be shared by engines of different widths, so the budget is
/// part of the key.
#[cfg(feature = "native")]
mod faer_limit {
    use crate::contraction::{default_faer_limit, take_last_plan_route, take_last_route_was_faer};
    use crate::test_support::{cpu_engine, with_cpu_session};
    use tenferro_tensor::{Complex64, DotGeneralConfig, Tensor, TensorDot, TensorRead};
    use threadteam::HostClass;

    /// tcontract's own bound for c64.
    const DEFAULT_BOUND: u64 = 1 << 17;
    const HOSTS: [HostClass; 6] = [
        HostClass::X86_64Avx512,
        HostClass::X86_64Avx2,
        HostClass::X86_64Scalar,
        HostClass::Aarch64Apple,
        HostClass::Aarch64,
        HostClass::Generic,
    ];

    /// Only AVX2 at a budget of one lifts the c64 bound, and no other dtype is
    /// bounded on any host.
    #[test]
    fn only_serial_avx2_lifts_the_c64_bound() {
        for host in HOSTS {
            for budget in [1, 2, 4] {
                let limit = default_faer_limit(host, budget);
                let c64 = if host == HostClass::X86_64Avx2 && budget == 1 {
                    None
                } else {
                    Some(DEFAULT_BOUND)
                };
                assert_eq!(limit.c64, c64, "{} at a budget of {budget}", host.name());
                assert_eq!(limit.f32, None, "{}: f32 is unbounded", host.name());
                assert_eq!(limit.f64, None, "{}: f64 is unbounded", host.name());
                assert_eq!(limit.c32, None, "{}: c32 is unbounded", host.name());
            }
        }
    }

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

    fn faer_f64_product(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        use crate::faer::{linalg::matmul::matmul, Accum, Mat, MatRef, Par};
        let mut out = Mat::<f64>::zeros(m, n);
        matmul(
            out.as_mut(),
            Accum::Replace,
            MatRef::from_column_major_slice(a, m, k),
            MatRef::from_column_major_slice(b, k, n),
            1.0,
            Par::Seq,
        );
        (0..n)
            .flat_map(|j| (0..m).map(move |i| (i, j)))
            .map(|(i, j)| *out.as_ref().get(i, j))
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

    /// Distance in representable steps between two finite `f64`s.
    ///
    /// Steps, not a relative tolerance: a relative bound with an absolute floor
    /// (`64 * EPSILON * max(|w|, 1)`, the version this replaced) passes a value
    /// near `1e-3` that is thousands of steps away, so it silently accepts far
    /// more error at small magnitudes than at large ones. The step count means
    /// the same thing everywhere.
    fn ulp_distance(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        // Map the sign-magnitude bit pattern onto a monotonic integer line so
        // that adjacent floats are adjacent integers across zero.
        let key = |x: f64| {
            let bits = x.to_bits() as i64;
            if bits < 0 {
                i64::MIN - bits
            } else {
                bits
            }
        };
        key(a).abs_diff(key(b))
    }

    /// Agreement to a few ULP, not bitwise.
    ///
    /// The two calls reach the same kernel family but not the same entry point:
    /// tcontract multiplies through the t4a-faer conj-aware entry, on a
    /// destination it may leave uninitialized, while the reference here calls
    /// `matmul` on an initialized one, so the accumulation order is not required
    /// to match.
    ///
    /// The bound is normwise and scaled by the contraction length, not
    /// component-wise. A component-wise bound cannot work here: a 141-term
    /// complex dot product with cancellation produces components far below the
    /// accumulation noise, and on x86_64 at `1x141x282` the largest component
    /// disagreement is 2102 steps while the tensor is still computed to its own
    /// scale. The `2 ULP` that #2326 recorded was the *first* offending
    /// component, not the maximum; that assertion stopped at the first one. A
    /// component-wise bound tight enough to catch a broken kernel fails on
    /// correct output, and one loose enough to pass accepts a kernel that lost
    /// most of its accuracy.
    ///
    /// `max|g - w| <= GAMMA_SAFETY * k * eps * max|w|` is the forward bound for a
    /// length-`k` dot product with a safety factor on top. It is scale free and
    /// still rejects what this test exists for: a missing conjugate, a route
    /// change or a degraded kernel move large components by O(|w|), thousands of
    /// times the bound. The measured error in `eps * max|w|` units and the worst
    /// component step count are printed for the record.
    const GAMMA_SAFETY: f64 = 4.0;

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

    /// The #1040 shapes: budget one routes to faer exactly where the host class
    /// lifts the c64 bound, and either route stays within the forward error
    /// bound of a direct faer product.
    #[test]
    fn serial_c64_route_follows_the_host_class() {
        let host = HostClass::detect();
        let lifted = default_faer_limit(host, 1).c64.is_none();
        for (m, k, n) in [(1, 141, 282), (8, 141, 282), (141, 282, 141)] {
            let (av, bv) = (data(m * k, 1), data(k * n, 2));
            let a = Tensor::from_vec_col_major(vec![m, k], av.clone()).unwrap();
            let b = Tensor::from_vec_col_major(vec![k, n], bv.clone()).unwrap();
            let mut engine = cpu_engine(1).unwrap();
            let _ = take_last_route_was_faer();
            let got = run(&mut engine, &a, &b);
            let below_bound = (m * k * n) as u64 <= DEFAULT_BOUND;
            assert_eq!(
                take_last_route_was_faer(),
                Some(below_bound || lifted),
                "{m}x{k}x{n}: {} at a budget of one must {}",
                host.name(),
                if below_bound || lifted {
                    "route to faer"
                } else {
                    "keep tcontract's packed driver"
                }
            );
            assert_eq!(
                take_last_plan_route().map(|(algorithm, _)| algorithm.name()),
                Some(if below_bound || lifted {
                    "faer"
                } else {
                    "packed"
                }),
                "{m}x{k}x{n}: the plan report must name the driver that ran"
            );
            let want = faer_product(&av, &bv, m, k, n);
            let got = got.as_slice::<Complex64>().unwrap();
            assert_eq!(got.len(), want.len());
            let mut worst_steps = 0u64;
            let mut max_err: f64 = 0.0;
            let mut scale: f64 = 0.0;
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    g.re.is_finite() && g.im.is_finite() && w.re.is_finite() && w.im.is_finite(),
                    "{m}x{k}x{n}: non-finite value"
                );
                worst_steps = worst_steps.max(ulp_distance(g.re, w.re));
                worst_steps = worst_steps.max(ulp_distance(g.im, w.im));
                max_err = max_err.max((g.re - w.re).abs()).max((g.im - w.im).abs());
                scale = scale.max(w.re.abs()).max(w.im.abs());
            }
            let eps = f64::EPSILON;
            let bound = GAMMA_SAFETY * k as f64 * eps * scale;
            let measured = max_err / (eps * scale);
            assert!(
                max_err <= bound,
                "{m}x{k}x{n}: max error {max_err:e} exceeds {GAMMA_SAFETY} * {k} * eps * {scale:e} = {bound:e}; measured {measured:.1} eps*scale, worst component {worst_steps} ULP"
            );
            println!(
                "{m}x{k}x{n}: {measured:.1} eps*scale measured, {:.1} allowed, worst component {worst_steps} ULP",
                GAMMA_SAFETY * k as f64
            );
        }
    }

    /// The budget is part of the plan key: a plan prepared by a serial engine is
    /// not handed to a wider engine sharing the cache, and back.
    #[test]
    fn shared_plan_cache_separates_budget_classes() {
        let (m, k, n) = (8, 141, 282);
        let lifted = default_faer_limit(HostClass::detect(), 1).c64.is_none();
        let a = Tensor::from_vec_col_major(vec![m, k], data(m * k, 1)).unwrap();
        let b = Tensor::from_vec_col_major(vec![k, n], data(k * n, 2)).unwrap();
        let plans = crate::CpuPlanCache::new();
        let mut serial = cpu_engine(1).unwrap().with_plan_cache(plans.clone());
        let mut wide = cpu_engine(4).unwrap().with_plan_cache(plans.clone());
        let _ = run(&mut serial, &a, &b);
        assert_eq!(take_last_route_was_faer(), Some(lifted));
        let misses = plans.stats().misses;
        let _ = run(&mut wide, &a, &b);
        assert_eq!(
            plans.stats().misses,
            misses + 1,
            "a wider budget must plan again, not reuse the serial plan"
        );
        assert_eq!(
            take_last_route_was_faer(),
            Some(false),
            "a wider budget keeps tcontract's default FaerLimit"
        );
        let _ = run(&mut serial, &a, &b);
        assert_eq!(take_last_route_was_faer(), Some(lifted));
    }

    /// f64 is unbounded on every host, so the c64 change cannot move it.
    ///
    /// The bits are pinned by the configuration, not by this comparison: `f64`
    /// is `None` for every class and budget, and `None` is also what the
    /// previous code produced, so the plan, kernel and accumulation order are
    /// unchanged. A direct faer matmul is a different entry point and is not
    /// required to agree bitwise, so this checks the route and the error bound.
    #[test]
    fn serial_f64_keeps_the_faer_route() {
        for (m, k, n) in [(8, 141, 282), (141, 282, 141)] {
            let real = |len: usize, seed: usize| -> Vec<f64> {
                (0..len)
                    .map(|i| ((i * 2654435761 + seed * 40503) % 1009) as f64 / 503.0 - 1.0)
                    .collect()
            };
            let (av, bv) = (real(m * k, 1), real(k * n, 2));
            let a = Tensor::from_vec_col_major(vec![m, k], av.clone()).unwrap();
            let b = Tensor::from_vec_col_major(vec![k, n], bv.clone()).unwrap();
            let mut engine = cpu_engine(1).unwrap();
            let _ = take_last_route_was_faer();
            let got = run(&mut engine, &a, &b);
            assert_eq!(
                take_last_route_was_faer(),
                Some(true),
                "{m}x{k}x{n}: f64 above the c64 bound must route to faer"
            );
            let want = faer_f64_product(&av, &bv, m, k, n);
            let got = got.as_slice::<f64>().unwrap();
            let mut max_err: f64 = 0.0;
            let mut scale: f64 = 0.0;
            for (g, w) in got.iter().zip(&want) {
                assert!(
                    g.is_finite() && w.is_finite(),
                    "{m}x{k}x{n}: non-finite value"
                );
                max_err = max_err.max((g - w).abs());
                scale = scale.max(w.abs());
            }
            let bound = GAMMA_SAFETY * k as f64 * f64::EPSILON * scale;
            assert!(
                max_err <= bound,
                "{m}x{k}x{n}: max error {max_err:e} exceeds {bound:e}"
            );
        }
    }
}

/// Which plan a thin c64 product takes on a vendor-BLAS build
/// (tensor4all-rs#1099).
///
/// On an Accelerate link, Accelerate's Complex64 GEMM kernel pays a per
/// right-operand-element read that a thin row role never amortises, so
/// [`super::blas_plan`] takes tcontract's own (faer) plan for c64 products whose
/// GEMM row role is one or two. Every other product keeps the vendor kernel:
/// f64, a real row role, and anything tcontract does not fuse onto faer. A build
/// that links a reference BLAS keeps the vendor kernel everywhere.
#[cfg(feature = "blas")]
mod blas_thin_c64 {
    use crate::contraction::{plan_config, take_last_route_was_faer};
    use crate::test_support::{cpu_engine, with_cpu_session};
    use cpueinsum::tcontract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
    use cpueinsum_blas::BlasScalar;
    use tenferro_tensor::ContractionScalar;
    use tenferro_tensor::{
        Complex64, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot, TensorRead,
        TensorWrite,
    };

    /// The gate is compiled only for an Accelerate link.
    const THIN_C64_IS_ROUTED: bool = cfg!(feature = "blas-accelerate");

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

    /// The problem of one column-major `m x k` by `k x n` product.
    fn problem<T: BlasScalar>(dtype: DType, m: usize, k: usize, n: usize) -> Problem {
        let spec = |dims: &[usize], strides: &[isize]| {
            OperandSpec::new(LayoutSpec::new(dims, strides, 0).expect("valid layout"))
        };
        let _ = core::marker::PhantomData::<T>;
        Problem::from_dot_general(
            dtype,
            spec(&[m, k], &[1, m as isize]),
            spec(&[k, n], &[1, k as isize]),
            spec(&[m, n], &[1, m as isize]),
            &DotGeneral::new(&[1], &[0], &[], &[]),
        )
        .expect("valid product")
    }

    /// Whether the cache would prepare tcontract's plan for one product.
    ///
    /// Planning only: no operand is touched, so this runs on any provider build.
    fn plans_faer<T: BlasScalar>(dtype: DType, m: usize, k: usize, n: usize) -> bool {
        plans_faer_with::<T>(
            dtype,
            m,
            k,
            n,
            &plan_config(threadteam::HostClass::detect(), 1),
        )
    }

    fn plans_faer_with<T: BlasScalar>(
        dtype: DType,
        m: usize,
        k: usize,
        n: usize,
        config: &cpueinsum::tcontract::PlanConfig,
    ) -> bool {
        match crate::contraction::blas_plan::BinaryPlan::<T>::new(
            &problem::<T>(dtype, m, k, n),
            config,
        ) {
            Ok(crate::contraction::blas_plan::BinaryPlan::Native(_)) => true,
            Ok(crate::contraction::blas_plan::BinaryPlan::Vendor(_)) => false,
            Err(error) => panic!("{m}x{k}x{n} {dtype:?}: {error}"),
        }
    }

    /// The branch itself: a thin c64 product takes tcontract's plan, and every
    /// product outside the gate keeps the vendor plan.
    #[test]
    fn only_thin_c64_takes_the_faer_plan() {
        // The #1099 shapes: a one- or two-row c64 product against a real right
        // operand.
        for (m, k, n) in [(1, 141, 141), (1, 141, 282), (1, 1, 141), (2, 141, 282)] {
            assert_eq!(
                plans_faer::<Complex64>(DType::C64, m, k, n),
                THIN_C64_IS_ROUTED,
                "{m}x{k}x{n} c64"
            );
        }
        // A real row role keeps Accelerate, which is 4.1x faster at 256^3.
        for (m, k, n) in [(8, 141, 282), (256, 256, 256)] {
            assert!(
                !plans_faer::<Complex64>(DType::C64, m, k, n),
                "{m}x{k}x{n} c64 must keep the vendor kernel"
            );
        }
        // A thin product that tcontract sends to its packed driver instead
        // (above the c64 faer bound) keeps the vendor kernel: the gate takes
        // tcontract's plan only where that plan is faer.
        assert!(
            !plans_faer_with::<Complex64>(
                DType::C64,
                1,
                1024,
                282,
                &cpueinsum::tcontract::PlanConfig::default()
            ),
            "1x1024x282 c64 is above the c64 faer bound and must keep the vendor kernel"
        );
        // f64 is already fast on the vendor kernel and every faer route for it
        // measured slower, so it is never routed.
        assert!(!plans_faer::<f64>(DType::F64, 1, 141, 282));
        assert!(!plans_faer::<f64>(DType::F64, 2, 141, 141));
        // No other dtype is bounded by tcontract, so every fusable non-c64
        // product reports `Faer` in a native build; the dtype check is what
        // keeps them on the vendor kernel.
        assert!(!plans_faer::<f32>(DType::F32, 1, 141, 282));
        assert!(!plans_faer::<num_complex::Complex32>(
            DType::C32,
            1,
            141,
            282
        ));
    }

    /// The routed product is correct and is the one that ran: a thin c64
    /// contraction must agree with a scalar complex product, and report the
    /// route the gate chose.
    ///
    /// The reference is a scalar loop rather than a second library, so the check
    /// does not move with a provider. The bound is the normwise forward bound of
    /// a length-`k` dot product, `4 * k * eps * max|d|`: a routed kernel that
    /// dropped a conjugate or lost accuracy moves large components by `O(|d|)`.
    #[test]
    fn thin_c64_runs_the_routed_plan_and_matches_a_scalar_product() {
        for (m, k, n) in [(1, 141, 282), (2, 141, 141)] {
            let (av, bv) = (data(m * k, 1), data(k * n, 2));
            let a = Tensor::from_vec_col_major(vec![m, k], av.clone()).unwrap();
            let b = Tensor::from_vec_col_major(vec![k, n], bv.clone()).unwrap();
            let mut engine = cpu_engine(1).unwrap();
            let _ = take_last_route_was_faer();
            let got = run(&mut engine, &a, &b);
            assert_eq!(
                take_last_route_was_faer(),
                Some(THIN_C64_IS_ROUTED),
                "{m}x{k}x{n}: the routed plan must be the one that ran"
            );
            let got = got.as_slice::<Complex64>().unwrap();
            let mut max_err: f64 = 0.0;
            let mut scale: f64 = 0.0;
            for j in 0..n {
                for i in 0..m {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for t in 0..k {
                        acc += av[i + t * m] * bv[t + j * k];
                    }
                    let g = got[i + j * m];
                    assert!(g.re.is_finite() && g.im.is_finite(), "non-finite value");
                    max_err = max_err
                        .max((g.re - acc.re).abs())
                        .max((g.im - acc.im).abs());
                    scale = scale.max(acc.re.abs()).max(acc.im.abs());
                }
            }
            let bound = 4.0 * k as f64 * f64::EPSILON * scale;
            assert!(
                max_err <= bound,
                "{m}x{k}x{n}: max error {max_err:e} exceeds {bound:e}"
            );
        }
    }

    /// The accepted deviation, asserted: the routed plan and the vendor kernel
    /// that ran before the route agree to 1e-13 relative, and that is the whole
    /// numeric cost of tensor4all-rs#1099 (2.2e-16 to 1.3e-15 of the output
    /// scale measured; this is the regression bound).
    ///
    /// Both plans are prepared here directly, so the comparison is the old
    /// output against the new one rather than either against a third reference.
    #[test]
    fn routed_plan_stays_within_the_recorded_deviation_of_the_vendor_kernel() {
        use core::mem::MaybeUninit;
        use cpueinsum::tcontract::SliceAccumulationSource;
        use cpueinsum::Exec;

        const RELATIVE_BOUND: f64 = 1e-13;

        for (m, k, n) in [(1, 141, 141), (1, 141, 282), (2, 141, 141), (2, 141, 282)] {
            let (av, bv) = (data(m * k, 1), data(k * n, 2));
            let config = plan_config(threadteam::HostClass::detect(), 1);
            let problem = problem::<Complex64>(DType::C64, m, k, n);
            let native =
                cpueinsum::BinaryPlan::<Complex64>::new(&problem, &config).expect("tcontract plan");
            assert_eq!(
                native.report().algorithm,
                cpueinsum::tcontract::Algorithm::Faer,
                "{m}x{k}x{n}: the baseline must be the plan the route takes"
            );
            let vendor = cpueinsum_blas::BinaryPlan::<Complex64>::new(&problem, &config)
                .expect("vendor plan");
            let mut work = vec![Complex64::new(0.0, 0.0); vendor.work_len()];

            let mut fresh = vec![MaybeUninit::<Complex64>::uninit(); m * n];
            let (got, _) = native
                .execute_uninit_slices(
                    &mut Exec::serial(),
                    Complex64::new(1.0, 0.0),
                    (&av, 0),
                    (&bv, 0),
                    Complex64::new(0.0, 0.0),
                    SliceAccumulationSource::Absent,
                    (&mut fresh, 0),
                )
                .expect("routed execution");
            let routed = got.to_vec();
            let mut fresh = vec![MaybeUninit::<Complex64>::uninit(); m * n];
            let (got, _) = vendor
                .execute_uninit_slices(
                    &mut Exec::serial(),
                    Complex64::new(1.0, 0.0),
                    (&av, 0),
                    (&bv, 0),
                    Complex64::new(0.0, 0.0),
                    SliceAccumulationSource::Absent,
                    (&mut fresh, 0),
                    &mut work,
                )
                .expect("vendor execution");
            let vendor_out = got.to_vec();

            let scale = vendor_out
                .iter()
                .map(|z| z.re.abs().max(z.im.abs()))
                .fold(0.0_f64, f64::max);
            let max_diff = routed
                .iter()
                .zip(&vendor_out)
                .map(|(r, v)| (r.re - v.re).abs().max((r.im - v.im).abs()))
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff <= RELATIVE_BOUND * scale,
                "{m}x{k}x{n}: routed output moved {max_diff:e} from the vendor kernel, \
                 above {RELATIVE_BOUND:e} of its {scale:e} scale"
            );
        }
    }

    /// The accumulating entry runs the routed plan too: an initialized output
    /// with `alpha = 2`, `beta = 3` must produce `2 * A B + 3 * C` from it.
    #[test]
    fn thin_c64_accumulates_into_an_initialized_output() {
        let (m, k, n) = (1, 141, 282);
        let (av, bv) = (data(m * k, 1), data(k * n, 2));
        let a = Tensor::from_vec_col_major(vec![m, k], av.clone()).unwrap();
        let b = Tensor::from_vec_col_major(vec![k, n], bv.clone()).unwrap();
        let prior = data(m * n, 3);
        let mut out = Tensor::from_vec_col_major(vec![m, n], prior.clone()).unwrap();
        let mut engine = cpu_engine(1).unwrap();
        let accumulation = DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::C64(Complex64::new(2.0, 0.0)),
            beta: ContractionScalar::C64(Complex64::new(3.0, 0.0)),
        };
        let _ = take_last_route_was_faer();
        with_cpu_session(&mut engine, |session| {
            session
                .dot_general_read_into_accum(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&b),
                    &config(),
                    accumulation,
                    TensorWrite::from_tensor(&mut out),
                )
                .unwrap()
        });
        assert_eq!(
            take_last_route_was_faer(),
            Some(THIN_C64_IS_ROUTED),
            "the accumulating call must run the routed plan"
        );
        let got = out.as_slice::<Complex64>().unwrap();
        let mut max_err: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for j in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for t in 0..k {
                acc += av[t] * bv[t + j * k];
            }
            let want = acc * 2.0 + prior[j] * 3.0;
            let g = got[j];
            max_err = max_err
                .max((g.re - want.re).abs())
                .max((g.im - want.im).abs());
            scale = scale.max(want.re.abs()).max(want.im.abs());
        }
        let bound = 4.0 * k as f64 * f64::EPSILON * scale * 3.0;
        assert!(max_err <= bound, "max error {max_err:e} exceeds {bound:e}");
    }
}
