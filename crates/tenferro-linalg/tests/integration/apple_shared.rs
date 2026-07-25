#![cfg(target_os = "macos")]

use num_complex::{Complex32, Complex64};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_gpu::{upload_webgpu_tensor, AppleContext, WebGpuRuntime};
use tenferro_linalg::{EagerTensorLinalgExt, LinalgBackend, TracedTensorLinalgExt};
use tenferro_runtime::{GraphCompiler, TracedTensor};
use tenferro_tensor::{Buffer, HostAccessError, Tensor};

use super::support;

fn apple_context() -> Option<AppleContext> {
    match AppleContext::new() {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping Apple shared Cholesky test: {error}");
            None
        }
    }
}

fn mapped_f32(tensor: &Tensor) -> Vec<f32> {
    let Tensor::F32(tensor) = tensor else {
        panic!("expected F32 tensor")
    };
    let Buffer::Backend(buffer) = tensor.buffer() else {
        panic!("expected managed backend storage")
    };
    buffer.map_read().unwrap().to_vec()
}

fn f32_ids(
    tensor: &Tensor,
) -> (
    Option<tenferro_tensor::AllocationDomainId>,
    Option<tenferro_tensor::AllocationId>,
) {
    let Tensor::F32(tensor) = tensor else {
        panic!("expected F32 tensor")
    };
    (tensor.allocation_domain(), tensor.allocation_id())
}

fn assert_cholesky_result(
    input: &Tensor,
    output: &Tensor,
    context: &AppleContext,
    before: tenferro_gpu::AppleTransferStats,
) {
    let values = mapped_f32(output);
    assert_eq!(values.len(), 4);
    assert_eq!(values[2], 0.0, "upper triangle must be zero");
    let reconstructed = [
        values[0] * values[0] + values[2] * values[2],
        values[1] * values[0] + values[3] * values[2],
        values[0] * values[1] + values[2] * values[3],
        values[1] * values[1] + values[3] * values[3],
    ];
    let expected = [4.0_f32, 2.0, 2.0, 3.0];
    let residual = reconstructed
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        residual <= 1.0e-5,
        "Cholesky reconstruction residual: {residual}"
    );
    let (output_domain, output_id) = f32_ids(output);
    let (_, input_id) = f32_ids(input);
    assert_eq!(output_domain, Some(context.domain_id()));
    assert_ne!(output_id, None);
    assert_ne!(output_id, input_id);
    assert_eq!(context.transfer_stats(), before);
}

fn managed_spd(context: &AppleContext) -> Tensor {
    let host = Tensor::from_vec_col_major([2, 2], vec![4.0_f32, 2.0, 2.0, 3.0]).unwrap();
    context.upload_tensor(&host).unwrap()
}

fn assert_lower_real<T>(values: &[T], to_f64: impl Fn(T) -> f64)
where
    T: Copy + std::fmt::Debug,
{
    assert_eq!(values.len(), 4);
    assert!(
        to_f64(values[2]).abs() <= 1.0e-12,
        "upper triangle: {values:?}"
    );
    let expected = [2.0_f64, 1.0, 0.0, 2.0_f64.sqrt()];
    for (actual, expected) in values.iter().copied().map(&to_f64).zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn managed_cpu_cholesky_supports_all_cpu_float_and_complex_dtypes() {
    let Some(context) = apple_context() else {
        return;
    };
    let f64_input = context
        .upload_tensor(&Tensor::from_vec_col_major([2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap())
        .unwrap();
    let before = context.transfer_stats();
    let Tensor::F64(f64_output) = context.cpu_backend().clone().cholesky(&f64_input).unwrap()
    else {
        panic!("expected F64 output")
    };
    let Buffer::Backend(buffer) = f64_output.buffer() else {
        panic!("expected managed F64 output")
    };
    assert_lower_real(&buffer.map_read().unwrap(), |value| value);
    assert_eq!(context.transfer_stats(), before);

    let c32 = |value| Complex32::new(value, 0.0);
    let c32_input = context
        .upload_tensor(
            &Tensor::from_vec_col_major([2, 2], vec![c32(4.0), c32(2.0), c32(2.0), c32(3.0)])
                .unwrap(),
        )
        .unwrap();
    let before = context.transfer_stats();
    let Tensor::C32(c32_output) = context.cpu_backend().clone().cholesky(&c32_input).unwrap()
    else {
        panic!("expected C32 output")
    };
    let Buffer::Backend(buffer) = c32_output.buffer() else {
        panic!("expected managed C32 output")
    };
    let values = buffer.map_read().unwrap();
    assert!(values.iter().all(|value| value.im.abs() <= 1.0e-6));
    assert_lower_real(&values, |value| f64::from(value.re));
    drop(values);
    assert_eq!(context.transfer_stats(), before);

    let c64 = |value| Complex64::new(value, 0.0);
    let c64_input = context
        .upload_tensor(
            &Tensor::from_vec_col_major([2, 2], vec![c64(4.0), c64(2.0), c64(2.0), c64(3.0)])
                .unwrap(),
        )
        .unwrap();
    let before = context.transfer_stats();
    let Tensor::C64(c64_output) = context.cpu_backend().clone().cholesky(&c64_input).unwrap()
    else {
        panic!("expected C64 output")
    };
    let Buffer::Backend(buffer) = c64_output.buffer() else {
        panic!("expected managed C64 output")
    };
    let values = buffer.map_read().unwrap();
    assert!(values.iter().all(|value| value.im.abs() <= 1.0e-12));
    assert_lower_real(&values, |value| value.re);
    drop(values);
    assert_eq!(context.transfer_stats(), before);
}

#[test]
fn public_concrete_eager_and_traced_cholesky_preserve_apple_domain_without_transfers() {
    let Some(context) = apple_context() else {
        return;
    };

    let input = managed_spd(&context);
    let before = context.transfer_stats();
    let direct = context.cpu_backend().clone().cholesky(&input).unwrap();
    assert_cholesky_result(&input, &direct, &context, before);

    let input = managed_spd(&context);
    let before = context.transfer_stats();
    let runtime = EagerRuntime::with_cpu_backend(context.cpu_backend().clone());
    let eager_input = EagerTensor::from_tensor_in(input.clone(), runtime).unwrap();
    let eager = eager_input.cholesky().unwrap().materialized().unwrap();
    assert_cholesky_result(&input, eager.as_ref(), &context, before);

    let input = managed_spd(&context);
    let before = context.transfer_stats();
    let traced_input = TracedTensor::from_tensor_concrete_shape(input.clone()).unwrap();
    let traced_output = traced_input.cholesky().unwrap();
    let program = GraphCompiler::new().compile(&traced_output).unwrap();
    let runtime = support::cpu_runtime_with_linalg(context.cpu_backend()).unwrap();
    let traced = runtime.run_compiled(&program, &[]).unwrap().remove(0);
    assert_cholesky_result(&input, &traced, &context, before);
}

#[test]
fn managed_cholesky_rejects_foreign_and_device_local_storage_without_transfers() {
    let (Some(first), Some(second)) = (apple_context(), apple_context()) else {
        return;
    };
    let foreign = managed_spd(&first);
    let first_before = first.transfer_stats();
    let second_before = second.transfer_stats();
    let error = second.cpu_backend().clone().cholesky(&foreign).unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::ForeignDomain { .. },
            ..
        }
    ));
    assert_eq!(first.transfer_stats(), first_before);
    assert_eq!(second.transfer_stats(), second_before);

    let Ok(runtime) = WebGpuRuntime::new_default() else {
        return;
    };
    let host = Tensor::from_vec_col_major([2, 2], vec![4.0_f32, 2.0, 2.0, 3.0]).unwrap();
    let device_local = upload_webgpu_tensor(&runtime, &host).unwrap();
    runtime.synchronize().unwrap();
    let before = first.transfer_stats();
    let error = first
        .cpu_backend()
        .clone()
        .cholesky(&device_local)
        .unwrap_err();
    assert!(matches!(
        error,
        tenferro_tensor::Error::HostAccess {
            source: HostAccessError::Unsupported { .. },
            ..
        }
    ));
    assert_eq!(first.transfer_stats(), before);
}

#[test]
fn managed_cholesky_validates_rank_shape_and_positive_definiteness_before_output() {
    let Some(context) = apple_context() else {
        return;
    };
    for host in [
        Tensor::from_vec_col_major([2, 3], vec![1.0_f32; 6]).unwrap(),
        Tensor::from_vec_col_major([2, 2, 1], vec![1.0_f32; 4]).unwrap(),
    ] {
        let managed = context.upload_tensor(&host).unwrap();
        let before = context.transfer_stats();
        let error = context
            .cpu_backend()
            .clone()
            .cholesky(&managed)
            .unwrap_err();
        assert!(matches!(error, tenferro_tensor::Error::Validation { .. }));
        assert_eq!(context.transfer_stats(), before);
    }

    let not_positive_definite = context
        .upload_tensor(&Tensor::from_vec_col_major([2, 2], vec![1.0_f32, 2.0, 2.0, 1.0]).unwrap())
        .unwrap();
    let before = context.transfer_stats();
    let error = context
        .cpu_backend()
        .clone()
        .cholesky(&not_positive_definite)
        .unwrap_err();
    assert!(matches!(error, tenferro_tensor::Error::Extension { .. }));
    assert_eq!(context.transfer_stats(), before);
}
