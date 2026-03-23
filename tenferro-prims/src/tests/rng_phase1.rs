use tenferro_algebra::Standard;
use tenferro_device::{Generator, LogicalMemorySpace};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{CpuBackend, CpuContext, RngPrimsDescriptor, TensorRngPrims};
#[cfg(feature = "cuda")]
use crate::{CudaBackend, CudaContext};

fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&lhs, &rhs) in actual.iter().zip(expected) {
        assert!(
            (lhs - rhs).abs() <= tol,
            "expected {rhs} within {tol}, got {lhs}"
        );
    }
}

fn tensor_on_host<T: tenferro_algebra::Scalar>(tensor: &Tensor<T>) -> Tensor<T> {
    tensor
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap()
}

fn run_uniform_f64_cpu(seed: u64, dims: &[usize]) -> Tensor<f64> {
    let mut ctx = CpuContext::new(1);
    let mut generator = Generator::cpu(seed);
    let desc = RngPrimsDescriptor::Uniform;
    let mut output = Tensor::<f64>::zeros(
        dims,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let plan =
        <CpuBackend as TensorRngPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[output.dims()])
            .unwrap();
    <CpuBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &mut generator,
        &mut output,
    )
    .unwrap();
    output
}

fn run_normal_f64_cpu(seed: u64, dims: &[usize]) -> Tensor<f64> {
    let mut ctx = CpuContext::new(1);
    let mut generator = Generator::cpu(seed);
    let desc = RngPrimsDescriptor::Normal;
    let mut output = Tensor::<f64>::zeros(
        dims,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let plan =
        <CpuBackend as TensorRngPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[output.dims()])
            .unwrap();
    <CpuBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &mut generator,
        &mut output,
    )
    .unwrap();
    output
}

fn run_randint_i32_cpu(seed: u64, dims: &[usize], low: i32, high: i32) -> Tensor<i32> {
    let mut ctx = CpuContext::new(1);
    let mut generator = Generator::cpu(seed);
    let desc = RngPrimsDescriptor::Integer { low, high };
    let mut output = Tensor::<i32>::zeros(
        dims,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let plan =
        <CpuBackend as TensorRngPrims<Standard<i32>>>::plan(&mut ctx, &desc, &[output.dims()])
            .unwrap();
    <CpuBackend as TensorRngPrims<Standard<i32>>>::execute(
        &mut ctx,
        &plan,
        &mut generator,
        &mut output,
    )
    .unwrap();
    output
}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
fn cuda_device_zero_is_available() -> bool {
    std::panic::catch_unwind(|| {
        cudarc::runtime::result::device::get_count()
            .map(|count| count > 0)
            .unwrap_or(false)
    })
    .unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn load_cuda_backend() -> Option<(CudaBackend, CudaContext)> {
    let path = available_cutensor_library_path()?;
    if !cuda_device_zero_is_available() {
        return None;
    }
    Some(CudaBackend::load(path).unwrap())
}

#[test]
fn cpu_rng_phase1_seeded_uniform_replay_matches() {
    let lhs = run_uniform_f64_cpu(1234, &[32]);
    let rhs = run_uniform_f64_cpu(1234, &[32]);
    assert_close_slice(
        tensor_on_host(&lhs).buffer().as_slice().unwrap(),
        tensor_on_host(&rhs).buffer().as_slice().unwrap(),
        0.0,
    );
}

#[test]
fn cpu_rng_phase1_seeded_normal_replay_matches() {
    let lhs = run_normal_f64_cpu(777, &[32]);
    let rhs = run_normal_f64_cpu(777, &[32]);
    assert_close_slice(
        tensor_on_host(&lhs).buffer().as_slice().unwrap(),
        tensor_on_host(&rhs).buffer().as_slice().unwrap(),
        0.0,
    );
}

#[test]
fn cpu_rng_phase1_seeded_randint_replay_matches_and_stays_in_range() {
    let lhs = run_randint_i32_cpu(99, &[64], -3, 7);
    let rhs = run_randint_i32_cpu(99, &[64], -3, 7);
    let lhs_host = tensor_on_host(&lhs);
    let rhs_host = tensor_on_host(&rhs);
    assert_eq!(
        lhs_host.buffer().as_slice().unwrap(),
        rhs_host.buffer().as_slice().unwrap()
    );
    for &value in lhs_host.buffer().as_slice().unwrap() {
        assert!(
            (-3..7).contains(&value),
            "randint sample {value} escaped range"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rng_phase1_seeded_uniform_replay_matches() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    let dims = [32usize];
    let desc = RngPrimsDescriptor::Uniform;

    let mut lhs_gen = Generator::cuda(device_id, 1234).unwrap();
    let mut rhs_gen = Generator::cuda(device_id, 1234).unwrap();

    let mut lhs = Tensor::<f64>::zeros(
        &dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut rhs = Tensor::<f64>::zeros(
        &dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let plan =
        <CudaBackend as TensorRngPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&dims]).unwrap();
    <CudaBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &mut lhs_gen,
        &mut lhs,
    )
    .unwrap();
    <CudaBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &mut rhs_gen,
        &mut rhs,
    )
    .unwrap();

    let lhs_host = tensor_on_host(&lhs);
    let rhs_host = tensor_on_host(&rhs);
    assert_close_slice(
        lhs_host.buffer().as_slice().unwrap(),
        rhs_host.buffer().as_slice().unwrap(),
        0.0,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rng_phase1_seeded_normal_replay_matches() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    let dims = [32usize];
    let desc = RngPrimsDescriptor::Normal;

    let mut lhs_gen = Generator::cuda(device_id, 777).unwrap();
    let mut rhs_gen = Generator::cuda(device_id, 777).unwrap();

    let mut lhs = Tensor::<f64>::zeros(
        &dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut rhs = Tensor::<f64>::zeros(
        &dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let plan =
        <CudaBackend as TensorRngPrims<Standard<f64>>>::plan(&mut ctx, &desc, &[&dims]).unwrap();
    <CudaBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &mut lhs_gen,
        &mut lhs,
    )
    .unwrap();
    <CudaBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        &mut rhs_gen,
        &mut rhs,
    )
    .unwrap();

    let lhs_host = tensor_on_host(&lhs);
    let rhs_host = tensor_on_host(&rhs);
    assert_close_slice(
        lhs_host.buffer().as_slice().unwrap(),
        rhs_host.buffer().as_slice().unwrap(),
        0.0,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rng_phase1_seeded_randint_replay_matches_and_stays_in_range() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    let dims = [64usize];
    let low = -3;
    let high = 7;
    let desc = RngPrimsDescriptor::Integer { low, high };

    let mut lhs_gen = Generator::cuda(device_id, 99).unwrap();
    let mut rhs_gen = Generator::cuda(device_id, 99).unwrap();

    let mut lhs = Tensor::<i32>::zeros(
        &dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mut rhs = Tensor::<i32>::zeros(
        &dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let plan =
        <CudaBackend as TensorRngPrims<Standard<i32>>>::plan(&mut ctx, &desc, &[&dims]).unwrap();
    <CudaBackend as TensorRngPrims<Standard<i32>>>::execute(
        &mut ctx,
        &plan,
        &mut lhs_gen,
        &mut lhs,
    )
    .unwrap();
    <CudaBackend as TensorRngPrims<Standard<i32>>>::execute(
        &mut ctx,
        &plan,
        &mut rhs_gen,
        &mut rhs,
    )
    .unwrap();

    let lhs_host = tensor_on_host(&lhs);
    let rhs_host = tensor_on_host(&rhs);
    assert_eq!(
        lhs_host.buffer().as_slice().unwrap(),
        rhs_host.buffer().as_slice().unwrap()
    );
    for &value in lhs_host.buffer().as_slice().unwrap() {
        assert!(
            (low..high).contains(&value),
            "randint sample {value} escaped range"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rng_phase1_rejects_cpu_generator_for_zero_sized_gpu_outputs() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    let dims = [0usize];

    let mut cpu_uniform = Generator::cpu(111);
    let mut uniform = Tensor::<f64>::zeros(
        &dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let uniform_plan = <CudaBackend as TensorRngPrims<Standard<f64>>>::plan(
        &mut ctx,
        &RngPrimsDescriptor::Uniform,
        &[&dims],
    )
    .unwrap();
    assert!(<CudaBackend as TensorRngPrims<Standard<f64>>>::execute(
        &mut ctx,
        &uniform_plan,
        &mut cpu_uniform,
        &mut uniform,
    )
    .is_err());

    let mut cpu_integer = Generator::cpu(222);
    let mut randint = Tensor::<i32>::zeros(
        &dims,
        LogicalMemorySpace::GpuMemory { device_id },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let randint_plan = <CudaBackend as TensorRngPrims<Standard<i32>>>::plan(
        &mut ctx,
        &RngPrimsDescriptor::Integer { low: -4, high: 5 },
        &[&dims],
    )
    .unwrap();
    assert!(<CudaBackend as TensorRngPrims<Standard<i32>>>::execute(
        &mut ctx,
        &randint_plan,
        &mut cpu_integer,
        &mut randint,
    )
    .is_err());
}
