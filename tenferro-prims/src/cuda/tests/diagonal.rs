use super::*;

#[test]
fn cuda_trace_matches_cpu_for_batched_square_case() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };
    let mut cpu_ctx = CpuContext::new(1);
    let desc = SemiringCoreDescriptor::Trace {
        modes_a: vec![0, 1, 2],
        modes_c: vec![2],
        paired: vec![(0, 1)],
    };

    let input = tensor_f64_from_col_major_fn(&[3, 3, 2], |idx| {
        (idx[0] + 1) as f64 + 10.0 * (idx[1] + 1) as f64 + 100.0 * idx[2] as f64
    });
    let cpu_plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut cpu_ctx,
        &desc,
        &[input.dims(), &[2]],
    )
    .unwrap();
    let cuda_plan = <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut cuda_ctx,
        &desc,
        &[input.dims(), &[2]],
    )
    .unwrap();

    let mut cpu_output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut cpu_ctx,
        &cpu_plan,
        1.0,
        &[&input],
        0.0,
        &mut cpu_output,
    )
    .unwrap();

    let input_gpu = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut gpu_output = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut cuda_ctx,
        &cuda_plan,
        1.0,
        &[&input_gpu],
        0.0,
        &mut gpu_output,
    )
    .unwrap();

    let got = gpu_output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(got.buffer().as_slice(), cpu_output.buffer().as_slice());
}

#[test]
fn cuda_antidiag_matches_cpu_with_alpha_beta() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };
    let mut cpu_ctx = CpuContext::new(1);
    let desc = SemiringCoreDescriptor::AntiDiag {
        modes_a: vec![0, 2],
        modes_c: vec![0, 1, 2],
        paired: vec![(0, 1)],
    };

    let input =
        tensor_f64_from_col_major_fn(&[3, 2], |idx| (idx[0] + 1) as f64 + 10.0 * idx[1] as f64);
    let cpu_plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut cpu_ctx,
        &desc,
        &[input.dims(), &[3, 3, 2]],
    )
    .unwrap();
    let cuda_plan = <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut cuda_ctx,
        &desc,
        &[input.dims(), &[3, 3, 2]],
    )
    .unwrap();

    let mut cpu_output = tensor_f64_from_col_major_fn(&[3, 3, 2], |_| 5.0);
    <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut cpu_ctx,
        &cpu_plan,
        2.0,
        &[&input],
        3.0,
        &mut cpu_output,
    )
    .unwrap();

    let input_gpu = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let existing_gpu = tensor_f64_from_col_major_fn(&[3, 3, 2], |_| 5.0)
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut gpu_output = existing_gpu;
    <CudaBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut cuda_ctx,
        &cuda_plan,
        2.0,
        &[&input_gpu],
        3.0,
        &mut gpu_output,
    )
    .unwrap();

    let got = gpu_output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(got.buffer().as_slice(), cpu_output.buffer().as_slice());
}

#[test]
fn cuda_antitrace_matches_cpu_with_batched_scalars() {
    let Some((_backend, mut cuda_ctx)) = load_cuda_backend() else {
        return;
    };
    let mut cpu_ctx = CpuContext::new(1);
    let desc = SemiringCoreDescriptor::AntiTrace {
        modes_a: vec![2],
        modes_c: vec![0, 1, 2],
        paired: vec![(0, 1)],
    };

    let input = tensor_f64(&[2.0_f64, 7.0], &[2]);
    let cpu_plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut cpu_ctx,
        &desc,
        &[input.dims(), &[3, 3, 2]],
    )
    .unwrap();
    let cuda_plan = <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan(
        &mut cuda_ctx,
        &desc,
        &[input.dims(), &[3, 3, 2]],
    )
    .unwrap();

    let mut cpu_output = tensor_f64_from_col_major_fn(&[3, 3, 2], |_| 1.0);
    <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut cpu_ctx,
        &cpu_plan,
        2.0,
        &[&input],
        3.0,
        &mut cpu_output,
    )
    .unwrap();

    let input_gpu = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let existing_gpu = tensor_f64_from_col_major_fn(&[3, 3, 2], |_| 1.0)
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut gpu_output = existing_gpu;
    <CudaBackend as TensorSemiringCore<Standard<f64>>>::execute(
        &mut cuda_ctx,
        &cuda_plan,
        2.0,
        &[&input_gpu],
        3.0,
        &mut gpu_output,
    )
    .unwrap();

    let got = gpu_output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(got.buffer().as_slice(), cpu_output.buffer().as_slice());
}
