use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_device::Result;
use tenferro_tensor::MemoryOrder;
use tenferro_tensor::Tensor;

use super::*;

#[test]
fn cuda_backend_feature_surface_matches_tensor_prims_contract() {
    let _plan_fn: fn(&mut CudaContext, &PrimDescriptor, &[&[usize]]) -> Result<CudaPlan<f64>> =
        <CudaBackend as TensorPrims<Standard<f64>>>::plan;
    let _execute_fn: fn(
        &mut CudaContext,
        &CudaPlan<f64>,
        f64,
        &[&Tensor<f64>],
        f64,
        &mut Tensor<f64>,
    ) -> Result<()> = <CudaBackend as TensorPrims<Standard<f64>>>::execute;

    assert!(<CudaBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::Contract));
    assert!(
        <CudaBackend as TensorPrims<Standard<f64>>>::has_extension_for(Extension::ElementwiseMul)
    );
}

fn available_cutensor_library_path() -> Option<&'static str> {
    [
        "/usr/lib/x86_64-linux-gnu/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor.so.2",
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2",
    ]
    .into_iter()
    .find(|path| std::path::Path::new(path).exists())
}

fn cuda_device_zero_is_available() -> bool {
    std::panic::catch_unwind(|| {
        cudarc::runtime::result::device::get_count()
            .map(|count| count > 0)
            .unwrap_or(false)
    })
    .unwrap_or(false)
}

#[test]
fn cuda_make_contiguous_smoke_runs_on_device_tensors_when_runtime_is_available() {
    let Some(path) = available_cutensor_library_path() else {
        return;
    };

    if !cuda_device_zero_is_available() {
        return;
    }

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();
    let base = Tensor::<f32>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let input = base.permute(&[1, 0]).unwrap();
    let plan = <CudaBackend as TensorPrims<Standard<f32>>>::plan(
        &mut ctx,
        &PrimDescriptor::MakeContiguous,
        &[&[3, 2], &[3, 2]],
    )
    .unwrap();
    let input_gpu = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut output_gpu = Tensor::<f32>::zeros(
        &[3, 2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    );

    <CudaBackend as TensorPrims<Standard<f32>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input_gpu],
        0.0,
        &mut output_gpu,
    )
    .unwrap();

    let output_cpu = output_gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert!(output_cpu.is_contiguous());
    assert_eq!(output_cpu.dims(), &[3, 2]);
    assert_eq!(
        output_cpu.buffer().as_slice(),
        Some(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0][..])
    );
}
