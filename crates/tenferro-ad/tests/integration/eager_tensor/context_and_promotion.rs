use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use tenferro_runtime::TensorScalar;

use super::*;

#[test]
fn context_id_is_unique() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    assert_ne!(ctx_a.id(), ctx_b.id());
}

#[test]
fn same_context_true_for_shared_ctx() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let c = ctx
        .constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    assert_eq!(c.ctx_id(), ctx.id());
    assert!(!c.tracks_grad());
    assert_eq!(f64_data(&c.to_tensor().unwrap()), &[1.0, 2.0]);
}

#[test]
fn variable_from_creates_tracked_leaf() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let p = ctx
        .variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    assert_eq!(p.ctx_id(), ctx.id());
    assert!(p.tracks_grad());
    // backward should work on a tracked variable
    let loss = p.exp().unwrap().reduce_sum(Some(&[0])).unwrap();
    let _ = loss.backward().unwrap();
    assert!(p.grad().unwrap().is_some());
}

#[test]
fn cross_context_add_rejected() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    assert_eq!(f64_data(&z.to_tensor().unwrap()), &[4.0, 6.0]);
}

#[test]
fn detach_into_different_context() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx_a,
    )
    .unwrap();
    let d = x.detach_into(&ctx_b).unwrap();
    assert_eq!(d.ctx_id(), ctx_b.id());
    assert!(!d.tracks_grad());
    assert_eq!(f64_data(&d.to_tensor().unwrap()), &[1.0, 2.0]);
    // Can operate with tensors from ctx_b now
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        ctx_b,
    )
    .unwrap();
    let z = d.add(&y).unwrap();
    assert_eq!(f64_data(&z.to_tensor().unwrap()), &[4.0, 6.0]);
}

#[test]
fn detach_into_still_accessible_in_original_context() {
    let ctx_a = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let ctx_b = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx_a.clone(),
    )
    .unwrap();
    let d = x.detach_into(&ctx_b).unwrap();
    // Original tensor still in ctx_a, should work fine
    let loss = x.exp().unwrap().reduce_sum(Some(&[0])).unwrap();
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
    assert_eq!(z.value().unwrap().as_slice::<f64>().unwrap(), &[1.5, 3.0]);
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
        z.value().unwrap().as_slice::<Complex64>().unwrap(),
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
    assert_eq!(out.value().unwrap().as_slice::<f64>().unwrap(), &[1.0, 1.5]);
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
    assert_eq!(z.value().unwrap().as_slice::<f64>().unwrap(), &[1.5, 3.0]);
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
    assert_eq!(z.value().unwrap().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}

fn assert_exact_values<T>(tensor: &EagerTensor, expected: &[T])
where
    T: TensorScalar + Debug + PartialEq,
{
    assert_eq!(
        tensor.to_tensor().unwrap().as_slice::<T>().unwrap(),
        expected
    );
}

fn eager_input<T: TensorScalar>(ctx: &Arc<EagerRuntime>, data: &[T], tracked: bool) -> EagerTensor {
    let tensor = Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).unwrap();
    if tracked {
        EagerTensor::requires_grad_in(tensor, Arc::clone(ctx)).unwrap()
    } else {
        // Keep inactive operands on the active-edge path as lazy constants.
        EagerTensor::from_tensor_in(tensor, Arc::clone(ctx))
            .unwrap()
            .reshape(&[data.len()])
            .unwrap()
    }
}

// INVARIANT: Explicit inputs and exact oracles keep both dtype pairs independently auditable.
#[allow(clippy::too_many_arguments)]
fn run_mixed_add_case<R, C>(
    real_data: &[R],
    complex_data: &[C],
    real_tangent: &[R],
    complex_tangent: &[C],
    cotangent: &[C],
    expected_sum: &[C],
    expected_real_vjp: &[R],
    expected_complex_vjp: &[C],
    expected_real_jvp: &[C],
    expected_complex_jvp: &[C],
    output_dtype: DType,
) where
    R: TensorScalar + Debug + PartialEq,
    C: TensorScalar + Debug + PartialEq,
{
    for active_real in [true, false] {
        for real_first in [true, false] {
            let ctx = test_ctx();
            let real = eager_input(&ctx, real_data, active_real);
            let complex = eager_input(&ctx, complex_data, !active_real);
            let output = if real_first {
                real.add(&complex).unwrap()
            } else {
                complex.add(&real).unwrap()
            };
            assert_eq!(output.dtype(), output_dtype);
            assert_exact_values(&output, expected_sum);

            let cotangent = eager_input(&ctx, cotangent, false);
            if active_real {
                let real_tangent = eager_input(&ctx, real_tangent, false);
                assert_exact_values(
                    &ctx.vjp(&output, &real, &cotangent).unwrap(),
                    expected_real_vjp,
                );
                assert!(ctx
                    .vjp_optional(&output, &complex, &cotangent)
                    .unwrap()
                    .is_none());
                assert_exact_values(
                    &ctx.jvp(&output, &real, &real_tangent).unwrap(),
                    expected_real_jvp,
                );
                let complex_tangent = eager_input(&ctx, complex_tangent, false);
                assert!(ctx
                    .jvp_optional(&output, &complex, &complex_tangent)
                    .unwrap()
                    .is_none());
            } else {
                let complex_tangent = eager_input(&ctx, complex_tangent, false);
                assert_exact_values(
                    &ctx.vjp(&output, &complex, &cotangent).unwrap(),
                    expected_complex_vjp,
                );
                assert!(ctx
                    .vjp_optional(&output, &real, &cotangent)
                    .unwrap()
                    .is_none());
                assert_exact_values(
                    &ctx.jvp(&output, &complex, &complex_tangent).unwrap(),
                    expected_complex_jvp,
                );
                let real_tangent = eager_input(&ctx, real_tangent, false);
                assert!(ctx
                    .jvp_optional(&output, &real, &real_tangent)
                    .unwrap()
                    .is_none());
            }
        }
    }
}

#[test]
fn mixed_add_semantic_jvp_vjp_promotes_f64_c64_and_f32_c32_in_both_orders() {
    let c64 = Complex64::new;
    run_mixed_add_case(
        &[1.25_f64, -2.5],
        &[c64(3.0, 0.5), c64(-4.0, -1.25)],
        &[0.25_f64, -0.75],
        &[c64(-1.5, 0.25), c64(2.0, -0.5)],
        &[c64(0.5, 0.75), c64(-1.25, -0.25)],
        &[c64(4.25, 0.5), c64(-6.5, -1.25)],
        &[0.5_f64, -1.25],
        &[c64(0.5, 0.75), c64(-1.25, -0.25)],
        &[c64(0.25, 0.0), c64(-0.75, 0.0)],
        &[c64(-1.5, 0.25), c64(2.0, -0.5)],
        DType::C64,
    );

    let c32 = Complex32::new;
    run_mixed_add_case(
        &[1.25_f32, -2.5],
        &[c32(3.0, 0.5), c32(-4.0, -1.25)],
        &[0.25_f32, -0.75],
        &[c32(-1.5, 0.25), c32(2.0, -0.5)],
        &[c32(0.5, 0.75), c32(-1.25, -0.25)],
        &[c32(4.25, 0.5), c32(-6.5, -1.25)],
        &[0.5_f32, -1.25],
        &[c32(0.5, 0.75), c32(-1.25, -0.25)],
        &[c32(0.25, 0.0), c32(-0.75, 0.0)],
        &[c32(-1.5, 0.25), c32(2.0, -0.5)],
        DType::C32,
    );
}

#[test]
fn mixed_concatenate_semantic_replay_promotes_inputs() {
    let ctx = test_ctx();
    let real = eager_input(&ctx, &[1.0_f64, 2.0], false);
    let complex = eager_input(
        &ctx,
        &[Complex64::new(3.0, 0.5), Complex64::new(-4.0, -1.25)],
        true,
    );
    let output = EagerTensor::concatenate(&[&real, &complex], 0).unwrap();
    assert_eq!(output.dtype(), DType::C64);
    assert_exact_values(
        &output,
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(-4.0, -1.25),
        ],
    );

    let cotangent = eager_input(
        &ctx,
        &[
            Complex64::new(0.5, 0.75),
            Complex64::new(-1.25, -0.25),
            Complex64::new(2.0, 1.0),
            Complex64::new(-3.0, 0.5),
        ],
        false,
    );
    assert_exact_values(
        &ctx.vjp(&output, &complex, &cotangent).unwrap(),
        &[Complex64::new(2.0, 1.0), Complex64::new(-3.0, 0.5)],
    );
    assert!(ctx
        .vjp_optional(&output, &real, &cotangent)
        .unwrap()
        .is_none());

    let tangent = eager_input(
        &ctx,
        &[Complex64::new(0.25, -0.5), Complex64::new(1.5, 0.75)],
        false,
    );
    assert_exact_values(
        &ctx.jvp(&output, &complex, &tangent).unwrap(),
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.25, -0.5),
            Complex64::new(1.5, 0.75),
        ],
    );
}
