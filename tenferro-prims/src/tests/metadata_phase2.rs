use tenferro_algebra::Scalar;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    CpuContext, MetadataBinaryOp, MetadataDType, MetadataPrimsDescriptor, MetadataReductionOp,
    MetadataTensorMut, MetadataTensorRef, MetadataTernaryOp, TensorMetadataContextFor,
    TensorMetadataPrims,
};

fn tensor_i32(data: &[i32], dims: &[usize], memory_space: LogicalMemorySpace) -> Tensor<i32> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(memory_space)
        .unwrap()
}

fn tensor_u8(data: &[u8], dims: &[usize], memory_space: LogicalMemorySpace) -> Tensor<u8> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(memory_space)
        .unwrap()
}

fn tensor_on_host<T: Scalar>(tensor: &Tensor<T>) -> Tensor<T> {
    tensor
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap()
}

fn assert_tensor_eq<T: Scalar + core::fmt::Debug + PartialEq>(tensor: &Tensor<T>, expected: &[T]) {
    let host = tensor_on_host(tensor);
    assert_eq!(host.buffer().as_slice().unwrap(), expected);
}

fn execute_binary_i32<C>(
    ctx: &mut C,
    memory_space: LogicalMemorySpace,
    op: MetadataBinaryOp,
    lhs_data: &[i32],
    rhs_data: &[i32],
    expected: &[i32],
) where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let dims = [lhs_data.len()];
    let lhs = tensor_i32(lhs_data, &dims, memory_space);
    let rhs = tensor_i32(rhs_data, &dims, memory_space);
    let mut out = Tensor::<i32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let desc = MetadataPrimsDescriptor::Binary {
        op,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
    };
    assert!(<C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(desc.clone()));
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &desc,
        &[MetadataTensorRef::I32(&lhs), MetadataTensorRef::I32(&rhs)],
        MetadataTensorMut::I32(&mut out),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[MetadataTensorRef::I32(&lhs), MetadataTensorRef::I32(&rhs)],
        MetadataTensorMut::I32(&mut out),
    )
    .unwrap();
    assert_tensor_eq(&out, expected);
}

fn execute_binary_bool<C>(
    ctx: &mut C,
    memory_space: LogicalMemorySpace,
    op: MetadataBinaryOp,
    lhs_data: &[u8],
    rhs_data: &[u8],
    expected: &[u8],
) where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let dims = [lhs_data.len()];
    let lhs = tensor_u8(lhs_data, &dims, memory_space);
    let rhs = tensor_u8(rhs_data, &dims, memory_space);
    let mut out = Tensor::<u8>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let desc = MetadataPrimsDescriptor::Binary {
        op,
        lhs_dtype: MetadataDType::Bool,
        rhs_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::Bool,
    };
    assert!(<C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(desc.clone()));
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &desc,
        &[MetadataTensorRef::Bool(&lhs), MetadataTensorRef::Bool(&rhs)],
        MetadataTensorMut::Bool(&mut out),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[MetadataTensorRef::Bool(&lhs), MetadataTensorRef::Bool(&rhs)],
        MetadataTensorMut::Bool(&mut out),
    )
    .unwrap();
    assert_tensor_eq(&out, expected);
}

fn run_metadata_phase2_family<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    execute_binary_i32::<C>(
        ctx,
        memory_space,
        MetadataBinaryOp::Add,
        &[1, -2, 3, 4],
        &[5, 6, -7, 8],
        &[6, 4, -4, 12],
    );
    execute_binary_i32::<C>(
        ctx,
        memory_space,
        MetadataBinaryOp::Sub,
        &[7, 9, -1, 6],
        &[5, 2, -4, 8],
        &[2, 7, 3, -2],
    );
    execute_binary_i32::<C>(
        ctx,
        memory_space,
        MetadataBinaryOp::Mul,
        &[2, -3, 4, -5],
        &[6, 7, -2, -1],
        &[12, -21, -8, 5],
    );
    execute_binary_bool::<C>(
        ctx,
        memory_space,
        MetadataBinaryOp::BitAnd,
        &[1, 1, 0, 0],
        &[1, 0, 1, 0],
        &[1, 0, 0, 0],
    );

    let cond = tensor_u8(&[1, 0, 1, 0], &[4], memory_space);
    let on_true = tensor_i32(&[10, 20, 30, 40], &[4], memory_space);
    let on_false = tensor_i32(&[-1, -2, -3, -4], &[4], memory_space);
    let mut where_out = Tensor::<i32>::zeros(&[4], memory_space, MemoryOrder::ColumnMajor).unwrap();
    let where_desc = MetadataPrimsDescriptor::Ternary {
        op: MetadataTernaryOp::Where,
        cond_dtype: MetadataDType::Bool,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
    };
    let where_plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &where_desc,
        &[
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::I32(&on_true),
            MetadataTensorRef::I32(&on_false),
        ],
        MetadataTensorMut::I32(&mut where_out),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &where_plan,
        &[
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::I32(&on_true),
            MetadataTensorRef::I32(&on_false),
        ],
        MetadataTensorMut::I32(&mut where_out),
    )
    .unwrap();
    assert_tensor_eq(&where_out, &[10, -2, 30, -4]);

    let reduce_i32_input = tensor_i32(&[1, 2, 3, 4], &[2, 2], memory_space);
    let mut reduce_i32_out =
        Tensor::<i32>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor).unwrap();
    let sum_desc = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        input_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
        op: MetadataReductionOp::Sum,
    };
    let sum_plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &sum_desc,
        &[MetadataTensorRef::I32(&reduce_i32_input)],
        MetadataTensorMut::I32(&mut reduce_i32_out),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &sum_plan,
        &[MetadataTensorRef::I32(&reduce_i32_input)],
        MetadataTensorMut::I32(&mut reduce_i32_out),
    )
    .unwrap();
    assert_tensor_eq(&reduce_i32_out, &[3, 7]);

    let reduce_bool_input = tensor_u8(&[1, 1, 1, 0], &[2, 2], memory_space);
    let mut reduce_all_out =
        Tensor::<u8>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor).unwrap();
    let all_desc = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        input_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::Bool,
        op: MetadataReductionOp::All,
    };
    let all_plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &all_desc,
        &[MetadataTensorRef::Bool(&reduce_bool_input)],
        MetadataTensorMut::Bool(&mut reduce_all_out),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &all_plan,
        &[MetadataTensorRef::Bool(&reduce_bool_input)],
        MetadataTensorMut::Bool(&mut reduce_all_out),
    )
    .unwrap();
    assert_tensor_eq(&reduce_all_out, &[1, 0]);

    let mut reduce_any_out =
        Tensor::<u8>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor).unwrap();
    let any_desc = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        input_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::Bool,
        op: MetadataReductionOp::Any,
    };
    let any_plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &any_desc,
        &[MetadataTensorRef::Bool(&reduce_bool_input)],
        MetadataTensorMut::Bool(&mut reduce_any_out),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &any_plan,
        &[MetadataTensorRef::Bool(&reduce_bool_input)],
        MetadataTensorMut::Bool(&mut reduce_any_out),
    )
    .unwrap();
    assert_tensor_eq(&reduce_any_out, &[1, 1]);
}

fn run_metadata_phase2_shape_rejection<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let lhs = tensor_i32(&[1, 2], &[2], memory_space);
    let rhs = tensor_i32(&[3, 4, 5], &[3], memory_space);
    let mut out = Tensor::<i32>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor).unwrap();
    let desc = MetadataPrimsDescriptor::Binary {
        op: MetadataBinaryOp::Add,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
    };
    assert!(<C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &desc,
        &[MetadataTensorRef::I32(&lhs), MetadataTensorRef::I32(&rhs)],
        MetadataTensorMut::I32(&mut out),
    )
    .is_err());
}

#[test]
fn cpu_metadata_phase2_family_supports_arithmetic_and_logical_ops() {
    let mut ctx = CpuContext::new(1);
    run_metadata_phase2_family::<CpuContext>(&mut ctx, LogicalMemorySpace::MainMemory);
}

#[test]
fn cpu_metadata_phase2_rejects_mismatched_binary_shapes() {
    let mut ctx = CpuContext::new(1);
    run_metadata_phase2_shape_rejection::<CpuContext>(&mut ctx, LogicalMemorySpace::MainMemory);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_phase2_family_supports_arithmetic_and_logical_ops() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    run_metadata_phase2_family::<crate::CudaContext>(
        &mut ctx,
        LogicalMemorySpace::GpuMemory { device_id },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_phase2_rejects_mismatched_binary_shapes() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    run_metadata_phase2_shape_rejection::<crate::CudaContext>(
        &mut ctx,
        LogicalMemorySpace::GpuMemory { device_id },
    );
}

#[cfg(feature = "cuda")]
fn load_cuda_backend() -> Option<(crate::CudaBackend, crate::CudaContext)> {
    let path = [
        "/usr/lib/x86_64-linux-gnu/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor.so.2",
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2",
    ]
    .into_iter()
    .find(|path| std::path::Path::new(path).exists())?;

    Some(crate::CudaBackend::load(path).unwrap())
}
