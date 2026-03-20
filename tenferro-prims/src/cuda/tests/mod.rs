mod complex;
mod organization;

use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_device::Result;
use tenferro_tensor::MemoryOrder;
use tenferro_tensor::Tensor;

use super::*;
use crate::{
    AnalyticPrimsDescriptor, AnalyticReductionOp, ScalarBinaryOp, ScalarPrimsDescriptor,
    SemiringBinaryOp, TensorAnalyticPrims, TensorScalarPrims,
};

#[test]
fn cuda_backend_feature_surface_matches_family_contracts() {
    let _core_plan_fn: fn(
        &mut CudaContext,
        &SemiringCoreDescriptor,
        &[&[usize]],
    ) -> Result<CudaPlan<f64>> = <CudaBackend as TensorSemiringCore<Standard<f64>>>::plan;
    let _fast_plan_fn: fn(
        &mut CudaContext,
        &SemiringFastPathDescriptor,
        &[&[usize]],
    ) -> Result<CudaPlan<f64>> = <CudaBackend as TensorSemiringFastPath<Standard<f64>>>::plan;
    let _execute_fn: fn(
        &mut CudaContext,
        &CudaPlan<f64>,
        f64,
        &[&Tensor<f64>],
        f64,
        &mut Tensor<f64>,
    ) -> Result<()> = <CudaBackend as TensorSemiringCore<Standard<f64>>>::execute;

    assert!(
        <CudaBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(
            SemiringFastPathDescriptor::Contract {
                modes_a: vec![0],
                modes_b: vec![0],
                modes_c: vec![0],
            }
        )
    );
    assert!(
        <CudaBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(
            SemiringFastPathDescriptor::ElementwiseBinary {
                op: SemiringBinaryOp::Mul,
            }
        )
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

pub(super) fn cuda_runtime_is_available() -> Option<&'static str> {
    let path = available_cutensor_library_path()?;
    if cuda_device_zero_is_available() {
        Some(path)
    } else {
        None
    }
}

#[test]
fn cuda_make_contiguous_smoke_runs_on_device_tensors_when_runtime_is_available() {
    let Some(path) = cuda_runtime_is_available() else {
        return;
    };

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();
    let base = Tensor::<f32>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let input = base.permute(&[1, 0]).unwrap();
    let plan = <CudaBackend as TensorSemiringCore<Standard<f32>>>::plan(
        &mut ctx,
        &SemiringCoreDescriptor::MakeContiguous,
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

    <CudaBackend as TensorSemiringCore<Standard<f32>>>::execute(
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

#[test]
fn cuda_elementwise_add_honors_alpha_beta_contract_when_runtime_is_available() {
    let Some(path) = cuda_runtime_is_available() else {
        return;
    };

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();
    let lhs = Tensor::<f32>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let rhs = Tensor::<f32>::from_slice(&[10.0, 20.0, 30.0], &[3], MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let plan = <CudaBackend as TensorSemiringFastPath<Standard<f32>>>::plan(
        &mut ctx,
        &SemiringFastPathDescriptor::ElementwiseBinary {
            op: SemiringBinaryOp::Add,
        },
        &[&[3], &[3], &[3]],
    )
    .unwrap();
    let mut output =
        Tensor::<f32>::from_slice(&[100.0, 100.0, 100.0], &[3], MemoryOrder::ColumnMajor)
            .unwrap()
            .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

    <CudaBackend as TensorSemiringFastPath<Standard<f32>>>::execute(
        &mut ctx,
        &plan,
        2.0,
        &[&lhs, &rhs],
        3.0,
        &mut output,
    )
    .unwrap();

    let output_cpu = output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        output_cpu.buffer().as_slice(),
        Some(&[322.0, 344.0, 366.0][..])
    );
}

#[test]
fn cuda_scalar_and_analytic_smoke_run_on_device_tensors_when_runtime_is_available() {
    let Some(path) = cuda_runtime_is_available() else {
        return;
    };

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();

    let scalar_plan = <CudaBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Div,
        },
        &[&[2], &[2], &[2]],
    )
    .unwrap();
    let lhs = Tensor::<f64>::from_slice(&[8.0, 9.0], &[2], MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let rhs = Tensor::<f64>::from_slice(&[2.0, 3.0], &[2], MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut scalar_out = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    );
    <CudaBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut ctx,
        &scalar_plan,
        1.0,
        &[&lhs, &rhs],
        0.0,
        &mut scalar_out,
    )
    .unwrap();
    let scalar_cpu = scalar_out
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(scalar_cpu.buffer().as_slice(), Some(&[4.0, 3.0][..]));

    let analytic_plan = <CudaBackend as TensorAnalyticPrims<Standard<f64>>>::plan(
        &mut ctx,
        &AnalyticPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op: AnalyticReductionOp::Var,
        },
        &[&[2, 2], &[2]],
    )
    .unwrap();
    let analytic_input =
        Tensor::<f64>::from_slice(&[1.0, 3.0, 5.0, 7.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap()
            .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
    let mut analytic_out = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    );
    <CudaBackend as TensorAnalyticPrims<Standard<f64>>>::execute(
        &mut ctx,
        &analytic_plan,
        1.0,
        &[&analytic_input],
        0.0,
        &mut analytic_out,
    )
    .unwrap();
    let analytic_cpu = analytic_out
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(analytic_cpu.buffer().as_slice(), Some(&[1.0, 1.0][..]));
}

#[test]
fn cuda_trace_antitrace_and_antidiag_smoke_run_on_device_tensors_when_runtime_is_available() {
    let Some(path) = cuda_runtime_is_available() else {
        return;
    };

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();

    let trace_plan = <CudaBackend as TensorSemiringCore<Standard<f32>>>::plan(
        &mut ctx,
        &SemiringCoreDescriptor::Trace {
            modes_a: vec![0, 1, 2],
            modes_c: vec![0],
            paired: vec![(1, 2)],
        },
        &[&[1, 3, 3], &[1]],
    )
    .unwrap();
    let trace_input = Tensor::<f32>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 3, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
    .unwrap();
    let mut trace_out = Tensor::<f32>::zeros(
        &[1],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    );
    <CudaBackend as TensorSemiringCore<Standard<f32>>>::execute(
        &mut ctx,
        &trace_plan,
        1.0,
        &[&trace_input],
        0.0,
        &mut trace_out,
    )
    .unwrap();
    let trace_cpu = trace_out
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(trace_cpu.buffer().as_slice(), Some(&[15.0][..]));

    let antitrace_plan = <CudaBackend as TensorSemiringCore<Standard<f32>>>::plan(
        &mut ctx,
        &SemiringCoreDescriptor::AntiTrace {
            modes_a: vec![0],
            modes_c: vec![0, 1, 2],
            paired: vec![(1, 2)],
        },
        &[&[1], &[1, 3, 3]],
    )
    .unwrap();
    let antitrace_input = Tensor::<f32>::from_slice(&[2.0], &[1], MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut antitrace_out = Tensor::<f32>::zeros(
        &[1, 3, 3],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    );
    <CudaBackend as TensorSemiringCore<Standard<f32>>>::execute(
        &mut ctx,
        &antitrace_plan,
        1.0,
        &[&antitrace_input],
        0.0,
        &mut antitrace_out,
    )
    .unwrap();
    let antitrace_cpu = antitrace_out
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        antitrace_cpu.buffer().as_slice(),
        Some(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0][..])
    );

    let antidiag_plan = <CudaBackend as TensorSemiringCore<Standard<f32>>>::plan(
        &mut ctx,
        &SemiringCoreDescriptor::AntiDiag {
            modes_a: vec![0],
            modes_c: vec![0, 1],
            paired: vec![(0, 1)],
        },
        &[&[3], &[3, 3]],
    )
    .unwrap();
    let antidiag_input =
        Tensor::<f32>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor)
            .unwrap()
            .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
    let mut antidiag_out = Tensor::<f32>::zeros(
        &[3, 3],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    );
    <CudaBackend as TensorSemiringCore<Standard<f32>>>::execute(
        &mut ctx,
        &antidiag_plan,
        1.0,
        &[&antidiag_input],
        0.0,
        &mut antidiag_out,
    )
    .unwrap();
    let antidiag_cpu = antidiag_out
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        antidiag_cpu.buffer().as_slice(),
        Some(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0][..])
    );
}
