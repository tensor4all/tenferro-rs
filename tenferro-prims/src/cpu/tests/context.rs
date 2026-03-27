use num_complex::Complex64;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::super::context::{CpuBackend, CpuContext};

// Do not delete or weaken this test: it keeps CpuContext constructor and accessor coverage tied to the public execution contract.
#[test]
fn cpu_context_try_new_rejects_zero_threads_and_exposes_accessors() {
    let err = CpuContext::try_new(0)
        .err()
        .expect("zero-thread context should fail");
    assert!(err.to_string().contains("num_threads >= 1"));

    let mut ctx = CpuContext::try_new(2).unwrap();
    assert_eq!(ctx.num_threads(), 2);
    assert_eq!(ctx.thread_pool().current_num_threads(), 2);
    ctx.plan_cache_mut().clear();
}

#[test]
fn cpu_context_default_constructor_uses_detected_thread_count() {
    let ctx = CpuContext::new_default();
    assert_eq!(ctx.num_threads(), CpuContext::default_num_threads());
    assert!(ctx.num_threads() >= 1);
}

#[test]
fn cpu_context_reuses_global_thread_pool_for_same_thread_count() {
    let ctx_a = CpuContext::new(3);
    let ctx_b = CpuContext::new(3);
    assert!(std::ptr::eq(ctx_a.thread_pool(), ctx_b.thread_pool()));
}

#[test]
fn cpu_backend_resolve_conj_covers_both_fast_and_materializing_paths() {
    let mut ctx = CpuContext::new(1);

    let plain = Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let resolved_plain = CpuBackend::resolve_conj(&mut ctx, &plain);
    let expected_plain = Tensor::stack(&[&plain], 0).unwrap().squeeze_dim(0).unwrap();
    assert!(!resolved_plain.is_conjugated());
    assert_eq!(
        resolved_plain.buffer().as_slice().unwrap(),
        expected_plain.buffer().as_slice().unwrap()
    );
    assert_eq!(resolved_plain.buffer().as_slice().unwrap(), &[1.0, 2.0]);

    let complex = Tensor::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .into_conj();
    let resolved_complex = CpuBackend::resolve_conj(&mut ctx, &complex);
    let expected_complex = Tensor::stack(&[&complex], 0)
        .unwrap()
        .squeeze_dim(0)
        .unwrap();
    assert!(!resolved_complex.is_conjugated());
    assert_eq!(
        resolved_complex.buffer().as_slice().unwrap(),
        expected_complex.buffer().as_slice().unwrap()
    );
    assert_eq!(
        resolved_complex.buffer().as_slice().unwrap(),
        &[Complex64::new(1.0, -2.0), Complex64::new(-3.0, -4.0)]
    );
}
