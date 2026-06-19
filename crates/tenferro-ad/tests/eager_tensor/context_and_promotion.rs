use super::*;

#[test]
fn context_id_is_unique() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    assert_ne!(ctx_a.id(), ctx_b.id());
}

#[test]
fn same_context_true_for_shared_ctx() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        ctx,
    )
    .unwrap();
    assert!(x.same_context(&y));
    assert_eq!(x.ctx_id(), y.ctx_id());
}

#[test]
fn same_context_false_for_different_ctx() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        ctx_a,
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        ctx_b,
    )
    .unwrap();
    assert!(!x.same_context(&y));
    assert_ne!(x.ctx_id(), y.ctx_id());
}

#[test]
fn constant_from_creates_untracked_leaf() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let c = ctx
        .constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    assert_eq!(c.ctx_id(), ctx.id());
    assert!(!c.tracks_grad());
    assert_eq!(f64_data(c.materialized().unwrap().as_ref()), &[1.0, 2.0]);
}

#[test]
fn variable_from_creates_tracked_leaf() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let p = ctx
        .variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    assert_eq!(p.ctx_id(), ctx.id());
    assert!(p.tracks_grad());
    // backward should work on a tracked variable
    let loss = p.exp().unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert!(p.grad().unwrap().is_some());
}

#[test]
fn cross_context_add_rejected() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx_a,
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        ctx_b,
    )
    .unwrap();
    let msg = match x.add(&y) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(msg.contains("different eager AD contexts"), "got: {msg}");
}

#[test]
fn cross_context_mul_rejected() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx_a,
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        ctx_b,
    )
    .unwrap();
    let msg = match x.mul(&y) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(msg.contains("different eager AD contexts"), "got: {msg}");
}

#[test]
fn cross_context_tracked_tensors_rejected() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx_a,
    )
    .unwrap();
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        ctx_b,
    )
    .unwrap();
    let msg = match x.add(&y) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error"),
    };
    assert!(msg.contains("different eager AD contexts"), "got: {msg}");
}

#[test]
fn constant_from_can_cross_context() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    // Import a fixed mask from a raw tensor into the same context
    let c = ctx
        .constant_from(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap())
        .unwrap();
    let z = x.add(&c).unwrap();
    assert_eq!(f64_data(z.materialized().unwrap().as_ref()), &[4.0, 6.0]);
}

#[test]
fn detach_into_different_context() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx_a,
    )
    .unwrap();
    let d = x.detach_into(&ctx_b).unwrap();
    assert_eq!(d.ctx_id(), ctx_b.id());
    assert!(!d.tracks_grad());
    assert_eq!(f64_data(d.materialized().unwrap().as_ref()), &[1.0, 2.0]);
    // Can operate with tensors from ctx_b now
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        ctx_b,
    )
    .unwrap();
    let z = d.add(&y).unwrap();
    assert_eq!(f64_data(z.materialized().unwrap().as_ref()), &[4.0, 6.0]);
}

#[test]
fn detach_into_still_accessible_in_original_context() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx_a.clone(),
    )
    .unwrap();
    let d = x.detach_into(&ctx_b).unwrap();
    // Original tensor still in ctx_a, should work fine
    let loss = x.exp().unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert!(x.grad().unwrap().is_some());
    // d is in ctx_b, x is in ctx_a
    assert_ne!(d.ctx_id(), x.ctx_id());
}

#[test]
fn promote_i64_add_f64_eager() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 1.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    // I64 + F64 should promote to F64
    let z = a.add(&b).unwrap();
    assert_eq!(z.dtype(), DType::F64);
    assert_eq!(
        z.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[1.5, 3.0]
    );
}

#[test]
fn promote_i64_mul_c64_eager() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![3_i64]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]).unwrap(),
        ctx,
    )
    .unwrap();
    // I64 * C64 should promote to C64
    let z = a.mul(&b).unwrap();
    assert_eq!(z.dtype(), DType::C64);
    assert_eq!(
        z.materialized().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(3.0, 0.0)]
    );
}

#[test]
fn clamp_promotes_all_three_operands_to_common_dtype() {
    let ctx = test_ctx();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let lower = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f32, 0.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let upper = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.5_f64, 1.5]).unwrap(),
        ctx,
    )
    .unwrap();

    let out = input.clamp(&lower, &upper).unwrap();

    assert_eq!(out.dtype(), DType::F64);
    assert_eq!(
        out.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[1.0, 1.5]
    );
}

#[test]
fn promote_f32_add_f64_eager() {
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 1.0]).unwrap(),
        ctx,
    )
    .unwrap();
    // F32 + F64 should promote to F64
    let z = a.add(&b).unwrap();
    assert_eq!(z.dtype(), DType::F64);
    assert_eq!(
        z.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[1.5, 3.0]
    );
}

#[test]
fn promote_same_dtype_no_conversion_penalty() {
    // Same-dtype ops should work without any conversion overhead
    let ctx = test_ctx();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let z = a.add(&b).unwrap();
    assert_eq!(z.dtype(), DType::F64);
    assert_eq!(
        z.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
}
