use num_complex::Complex64;
use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::{cuda_runtime_is_available, *};
use crate::{
    AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticUnaryOp, CpuBackend, CpuContext,
    ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp, ScalarUnaryOp, TensorAnalyticPrims,
    TensorScalarPrims,
};

fn assert_complex_slice_close(actual: Option<&[Complex64]>, expected: &[Complex64], tol: f64) {
    let actual = actual.expect("expected complex slice");
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (lhs.re - rhs.re).abs() <= tol,
            "real mismatch: {lhs:?} vs {rhs:?}"
        );
        assert!(
            (lhs.im - rhs.im).abs() <= tol,
            "imag mismatch: {lhs:?} vs {rhs:?}"
        );
    }
}

fn tensor_c64(data: &[Complex64], dims: &[usize]) -> Tensor<Complex64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn cpu_scalar_unary(op: ScalarUnaryOp, input: &Tensor<Complex64>) -> Tensor<Complex64> {
    let mut ctx = CpuContext::new(1);
    let plan = <CpuBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseUnary { op },
        &[input.dims(), input.dims()],
    )
    .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        input.dims(),
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[input],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
}

fn cpu_scalar_binary(
    op: ScalarBinaryOp,
    lhs: &Tensor<Complex64>,
    rhs: &Tensor<Complex64>,
) -> Tensor<Complex64> {
    let mut ctx = CpuContext::new(1);
    let plan = <CpuBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary { op },
        &[lhs.dims(), rhs.dims(), lhs.dims()],
    )
    .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        lhs.dims(),
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[lhs, rhs],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
}

fn cpu_scalar_reduction(op: ScalarReductionOp, input: &Tensor<Complex64>) -> Tensor<Complex64> {
    let mut ctx = CpuContext::new(1);
    let plan = <CpuBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op,
        },
        &[input.dims(), &[input.dims()[1]]],
    )
    .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        &[input.dims()[1]],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[input],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
}

fn cpu_analytic_unary(op: AnalyticUnaryOp, input: &Tensor<Complex64>) -> Tensor<Complex64> {
    let mut ctx = CpuContext::new(1);
    let plan = <CpuBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &AnalyticPrimsDescriptor::PointwiseUnary { op },
        &[input.dims(), input.dims()],
    )
    .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        input.dims(),
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorAnalyticPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[input],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
}

fn cpu_analytic_binary(
    op: AnalyticBinaryOp,
    lhs: &Tensor<Complex64>,
    rhs: &Tensor<Complex64>,
) -> Tensor<Complex64> {
    let mut ctx = CpuContext::new(1);
    let plan = <CpuBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &AnalyticPrimsDescriptor::PointwiseBinary { op },
        &[lhs.dims(), rhs.dims(), lhs.dims()],
    )
    .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        lhs.dims(),
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CpuBackend as TensorAnalyticPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[lhs, rhs],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
}

fn cuda_scalar_unary(
    ctx: &mut CudaContext,
    op: ScalarUnaryOp,
    input: &Tensor<Complex64>,
) -> Tensor<Complex64> {
    let plan = <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        ctx,
        &ScalarPrimsDescriptor::PointwiseUnary { op },
        &[input.dims(), input.dims()],
    )
    .unwrap();
    let input = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        input.dims(),
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&input],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap()
}

fn cuda_scalar_binary(
    ctx: &mut CudaContext,
    op: ScalarBinaryOp,
    lhs: &Tensor<Complex64>,
    rhs: &Tensor<Complex64>,
) -> Tensor<Complex64> {
    let plan = <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        ctx,
        &ScalarPrimsDescriptor::PointwiseBinary { op },
        &[lhs.dims(), rhs.dims(), lhs.dims()],
    )
    .unwrap();
    let lhs = lhs
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let rhs = rhs
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        lhs.dims(),
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&lhs, &rhs],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap()
}

fn cuda_scalar_reduction(
    ctx: &mut CudaContext,
    op: ScalarReductionOp,
    input: &Tensor<Complex64>,
) -> Tensor<Complex64> {
    let plan = <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        ctx,
        &ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op,
        },
        &[input.dims(), &[input.dims()[1]]],
    )
    .unwrap();
    let input = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        &[input.dims()[1]],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&input],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap()
}

fn cuda_analytic_unary(
    ctx: &mut CudaContext,
    op: AnalyticUnaryOp,
    input: &Tensor<Complex64>,
) -> Tensor<Complex64> {
    let plan = <CudaBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        ctx,
        &AnalyticPrimsDescriptor::PointwiseUnary { op },
        &[input.dims(), input.dims()],
    )
    .unwrap();
    let input = input
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        input.dims(),
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorAnalyticPrims<Standard<Complex64>>>::execute(
        ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&input],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap()
}

fn cuda_analytic_binary(
    ctx: &mut CudaContext,
    op: AnalyticBinaryOp,
    lhs: &Tensor<Complex64>,
    rhs: &Tensor<Complex64>,
) -> Tensor<Complex64> {
    let plan = <CudaBackend as TensorAnalyticPrims<Standard<Complex64>>>::plan(
        ctx,
        &AnalyticPrimsDescriptor::PointwiseBinary { op },
        &[lhs.dims(), rhs.dims(), lhs.dims()],
    )
    .unwrap();
    let lhs = lhs
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let rhs = rhs
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let mut output = Tensor::<Complex64>::zeros(
        lhs.dims(),
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorAnalyticPrims<Standard<Complex64>>>::execute(
        ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&lhs, &rhs],
        Complex64::new(0.0, 0.0),
        &mut output,
    )
    .unwrap();
    output
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap()
}

#[test]
fn cuda_complex_scalar_and_analytic_smoke_run_on_device_tensors_when_runtime_is_available() {
    let Some(path) = cuda_runtime_is_available() else {
        return;
    };

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();

    let scalar_plan = <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseUnary {
            op: crate::ScalarUnaryOp::Imag,
        },
        &[&[2], &[2]],
    )
    .unwrap();
    let scalar_input = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, -2.0), Complex64::new(3.5, 4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
    .unwrap();
    let mut scalar_out = Tensor::<Complex64>::zeros(
        &[2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &scalar_plan,
        Complex64::new(1.0, 0.0),
        &[&scalar_input],
        Complex64::new(0.0, 0.0),
        &mut scalar_out,
    )
    .unwrap();
    let scalar_cpu = scalar_out
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        scalar_cpu.buffer().as_slice(),
        Some(&[Complex64::new(-2.0, 0.0), Complex64::new(4.0, 0.0)][..])
    );
}

#[test]
fn cuda_complex_scalar_supported_ops_match_cpu_when_runtime_is_available() {
    let Some(path) = cuda_runtime_is_available() else {
        return;
    };

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();

    let unary_input = tensor_c64(&[Complex64::new(1.0, -2.0), Complex64::new(3.5, 4.0)], &[2]);
    for op in [
        ScalarUnaryOp::Neg,
        ScalarUnaryOp::Conj,
        ScalarUnaryOp::Abs,
        ScalarUnaryOp::Reciprocal,
        ScalarUnaryOp::Real,
        ScalarUnaryOp::Imag,
        ScalarUnaryOp::Square,
    ] {
        let cpu = cpu_scalar_unary(op, &unary_input);
        let cuda = cuda_scalar_unary(&mut ctx, op, &unary_input);
        assert_complex_slice_close(
            cuda.buffer().as_slice(),
            cpu.buffer().as_slice().unwrap(),
            1.0e-12,
        );
    }

    let lhs = tensor_c64(&[Complex64::new(3.0, 1.0), Complex64::new(-2.0, 4.0)], &[2]);
    let rhs = tensor_c64(&[Complex64::new(1.0, -2.0), Complex64::new(5.0, 0.5)], &[2]);
    for op in [
        ScalarBinaryOp::Add,
        ScalarBinaryOp::Sub,
        ScalarBinaryOp::Mul,
        ScalarBinaryOp::Div,
    ] {
        let cpu = cpu_scalar_binary(op, &lhs, &rhs);
        let cuda = cuda_scalar_binary(&mut ctx, op, &lhs, &rhs);
        assert_complex_slice_close(
            cuda.buffer().as_slice(),
            cpu.buffer().as_slice().unwrap(),
            1.0e-12,
        );
    }

    let reduction_input = tensor_c64(
        &[
            Complex64::new(1.0, 0.5),
            Complex64::new(3.0, -1.0),
            Complex64::new(5.0, 1.5),
            Complex64::new(7.0, 2.0),
        ],
        &[2, 2],
    );
    for op in [
        ScalarReductionOp::Sum,
        ScalarReductionOp::Prod,
        ScalarReductionOp::Mean,
    ] {
        let cpu = cpu_scalar_reduction(op, &reduction_input);
        let cuda = cuda_scalar_reduction(&mut ctx, op, &reduction_input);
        assert_complex_slice_close(
            cuda.buffer().as_slice(),
            cpu.buffer().as_slice().unwrap(),
            1.0e-12,
        );
    }
}

#[test]
#[ignore = "complex analytic CUDA support is out of scope for this scalar substrate task"]
fn cuda_complex_analytic_supported_ops_match_cpu_when_runtime_is_available() {
    let Some(path) = cuda_runtime_is_available() else {
        return;
    };

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();

    let unary_input = tensor_c64(
        &[Complex64::new(0.25, 0.5), Complex64::new(-0.4, 0.2)],
        &[2],
    );
    for op in [
        AnalyticUnaryOp::Sqrt,
        AnalyticUnaryOp::Rsqrt,
        AnalyticUnaryOp::Exp,
        AnalyticUnaryOp::Expm1,
        AnalyticUnaryOp::Log,
        AnalyticUnaryOp::Log1p,
        AnalyticUnaryOp::Sin,
        AnalyticUnaryOp::Cos,
        AnalyticUnaryOp::Tan,
        AnalyticUnaryOp::Tanh,
        AnalyticUnaryOp::Asin,
        AnalyticUnaryOp::Acos,
        AnalyticUnaryOp::Atan,
        AnalyticUnaryOp::Sinh,
        AnalyticUnaryOp::Cosh,
        AnalyticUnaryOp::Asinh,
        AnalyticUnaryOp::Acosh,
        AnalyticUnaryOp::Atanh,
    ] {
        let cpu = cpu_analytic_unary(op, &unary_input);
        let cuda = cuda_analytic_unary(&mut ctx, op, &unary_input);
        assert_complex_slice_close(
            cuda.buffer().as_slice(),
            cpu.buffer().as_slice().unwrap(),
            1.0e-10,
        );
    }

    let pow_lhs = tensor_c64(&[Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)], &[2]);
    let pow_rhs = tensor_c64(&[Complex64::new(3.0, 0.0), Complex64::new(2.0, 0.0)], &[2]);
    let cpu_pow = cpu_analytic_binary(AnalyticBinaryOp::Pow, &pow_lhs, &pow_rhs);
    let cuda_pow = cuda_analytic_binary(&mut ctx, AnalyticBinaryOp::Pow, &pow_lhs, &pow_rhs);
    assert_complex_slice_close(
        cuda_pow.buffer().as_slice(),
        cpu_pow.buffer().as_slice().unwrap(),
        1.0e-12,
    );

    let xlogy_lhs = tensor_c64(&[Complex64::new(0.25, 0.5), Complex64::new(0.0, 0.0)], &[2]);
    let xlogy_rhs = tensor_c64(
        &[Complex64::new(1.5, 0.25), Complex64::new(2.0, -0.5)],
        &[2],
    );
    let cpu_xlogy = cpu_analytic_binary(AnalyticBinaryOp::Xlogy, &xlogy_lhs, &xlogy_rhs);
    let cuda_xlogy =
        cuda_analytic_binary(&mut ctx, AnalyticBinaryOp::Xlogy, &xlogy_lhs, &xlogy_rhs);
    assert_complex_slice_close(
        cuda_xlogy.buffer().as_slice(),
        cpu_xlogy.buffer().as_slice().unwrap(),
        1.0e-12,
    );
}

#[test]
fn cuda_complex_scalar_binary_sub_smoke_runs_on_device_tensors_when_runtime_is_available() {
    let Some(path) = cuda_runtime_is_available() else {
        return;
    };

    let (_backend, mut ctx) = CudaBackend::load(path).unwrap();

    let plan = <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::plan(
        &mut ctx,
        &ScalarPrimsDescriptor::PointwiseBinary {
            op: crate::ScalarBinaryOp::Sub,
        },
        &[&[2], &[2], &[2]],
    )
    .unwrap();
    let lhs = Tensor::<Complex64>::from_slice(
        &[Complex64::new(3.0, 1.0), Complex64::new(-2.0, 4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
    .unwrap();
    let rhs = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, -2.0), Complex64::new(5.0, 0.5)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
    .unwrap();
    let mut out = Tensor::<Complex64>::zeros(
        &[2],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    <CudaBackend as TensorScalarPrims<Standard<Complex64>>>::execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&lhs, &rhs],
        Complex64::new(0.0, 0.0),
        &mut out,
    )
    .unwrap();
    let cpu = out
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    assert_eq!(
        cpu.buffer().as_slice(),
        Some(&[Complex64::new(2.0, 3.0), Complex64::new(-7.0, 3.5)][..])
    );
}
