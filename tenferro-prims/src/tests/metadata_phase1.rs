use tenferro_algebra::Scalar;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    CpuContext, MetadataBinaryOp, MetadataDType, MetadataGenerateOp, MetadataPrimsDescriptor,
    MetadataReductionOp, MetadataTensorMut, MetadataTensorRef, MetadataTernaryOp,
    TensorMetadataContextFor, TensorMetadataPrims,
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

fn run_metadata_family<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let backend = std::marker::PhantomData::<C::MetadataBackend>;
    let _ = backend;

    let dims = [4usize];

    let iota_desc = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::IotaStartZero,
        output_dtype: MetadataDType::I32,
    };
    assert!(<C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(iota_desc.clone()));
    let mut iota = Tensor::<i32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor);
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &iota_desc,
        &[],
        MetadataTensorMut::I32(&mut iota),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[],
        MetadataTensorMut::I32(&mut iota),
    )
    .unwrap();
    assert_tensor_eq(&iota, &[0, 1, 2, 3]);

    let lhs_i32 = tensor_i32(&[0, 1, 2, 3], &dims, memory_space);
    let rhs_i32 = tensor_i32(&[0, 0, 2, 9], &dims, memory_space);
    let mut neq = Tensor::<u8>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor);
    let binary_desc = MetadataPrimsDescriptor::Binary {
        op: MetadataBinaryOp::NotEqual,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::Bool,
    };
    assert!(<C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(binary_desc.clone()));
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &binary_desc,
        &[
            MetadataTensorRef::I32(&lhs_i32),
            MetadataTensorRef::I32(&rhs_i32),
        ],
        MetadataTensorMut::Bool(&mut neq),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[
            MetadataTensorRef::I32(&lhs_i32),
            MetadataTensorRef::I32(&rhs_i32),
        ],
        MetadataTensorMut::Bool(&mut neq),
    )
    .unwrap();
    assert_tensor_eq(&neq, &[0, 1, 0, 1]);

    let bool_lhs = tensor_u8(&[0, 1, 1, 0], &dims, memory_space);
    let bool_rhs = tensor_u8(&[0, 0, 1, 1], &dims, memory_space);
    let mut bool_neq = Tensor::<u8>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor);
    let bool_binary_desc = MetadataPrimsDescriptor::Binary {
        op: MetadataBinaryOp::NotEqual,
        lhs_dtype: MetadataDType::Bool,
        rhs_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::Bool,
    };
    assert!(
        <C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(bool_binary_desc.clone())
    );
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &bool_binary_desc,
        &[
            MetadataTensorRef::Bool(&bool_lhs),
            MetadataTensorRef::Bool(&bool_rhs),
        ],
        MetadataTensorMut::Bool(&mut bool_neq),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[
            MetadataTensorRef::Bool(&bool_lhs),
            MetadataTensorRef::Bool(&bool_rhs),
        ],
        MetadataTensorMut::Bool(&mut bool_neq),
    )
    .unwrap();
    assert_tensor_eq(&bool_neq, &[0, 1, 0, 1]);

    let mut where_i32 = Tensor::<i32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor);
    let cond = tensor_u8(&[1, 0, 1, 0], &dims, memory_space);
    let on_true_i32 = tensor_i32(&[10, 20, 30, 40], &dims, memory_space);
    let on_false_i32 = tensor_i32(&[-1, -2, -3, -4], &dims, memory_space);
    let ternary_desc = MetadataPrimsDescriptor::Ternary {
        op: MetadataTernaryOp::Where,
        cond_dtype: MetadataDType::Bool,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
    };
    assert!(
        <C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(ternary_desc.clone())
    );
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &ternary_desc,
        &[
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::I32(&on_true_i32),
            MetadataTensorRef::I32(&on_false_i32),
        ],
        MetadataTensorMut::I32(&mut where_i32),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::I32(&on_true_i32),
            MetadataTensorRef::I32(&on_false_i32),
        ],
        MetadataTensorMut::I32(&mut where_i32),
    )
    .unwrap();
    assert_tensor_eq(&where_i32, &[10, -2, 30, -4]);

    let mut where_bool = Tensor::<u8>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor);
    let on_true_bool = tensor_u8(&[1, 0, 1, 0], &dims, memory_space);
    let on_false_bool = tensor_u8(&[0, 1, 0, 1], &dims, memory_space);
    let bool_ternary_desc = MetadataPrimsDescriptor::Ternary {
        op: MetadataTernaryOp::Where,
        cond_dtype: MetadataDType::Bool,
        lhs_dtype: MetadataDType::Bool,
        rhs_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::Bool,
    };
    assert!(
        <C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(
            bool_ternary_desc.clone()
        )
    );
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &bool_ternary_desc,
        &[
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::Bool(&on_true_bool),
            MetadataTensorRef::Bool(&on_false_bool),
        ],
        MetadataTensorMut::Bool(&mut where_bool),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[
            MetadataTensorRef::Bool(&cond),
            MetadataTensorRef::Bool(&on_true_bool),
            MetadataTensorRef::Bool(&on_false_bool),
        ],
        MetadataTensorMut::Bool(&mut where_bool),
    )
    .unwrap();
    assert_tensor_eq(&where_bool, &[1, 1, 1, 1]);

    let reduction_desc = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        input_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::I32,
        op: MetadataReductionOp::Sum,
    };
    assert!(
        <C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(reduction_desc.clone())
    );
    let reduce_input = tensor_u8(&[1, 0, 1, 1], &[2, 2], memory_space);
    let mut reduce_output = Tensor::<i32>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor);
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &reduction_desc,
        &[MetadataTensorRef::Bool(&reduce_input)],
        MetadataTensorMut::I32(&mut reduce_output),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[MetadataTensorRef::Bool(&reduce_input)],
        MetadataTensorMut::I32(&mut reduce_output),
    )
    .unwrap();
    assert_tensor_eq(&reduce_output, &[1, 2]);

    let reduction_i32_desc = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        input_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
        op: MetadataReductionOp::Sum,
    };
    assert!(
        <C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(
            reduction_i32_desc.clone()
        )
    );
    let reduce_i32_input = tensor_i32(&[1, 2, 3, 4], &[2, 2], memory_space);
    let mut reduce_i32_output = Tensor::<i32>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor);
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &reduction_i32_desc,
        &[MetadataTensorRef::I32(&reduce_i32_input)],
        MetadataTensorMut::I32(&mut reduce_i32_output),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[MetadataTensorRef::I32(&reduce_i32_input)],
        MetadataTensorMut::I32(&mut reduce_i32_output),
    )
    .unwrap();
    assert_tensor_eq(&reduce_i32_output, &[3, 7]);
}

#[test]
fn cpu_metadata_family_builds_lu_det_parity_primitives() {
    let mut ctx = CpuContext::new(1);
    run_metadata_family::<CpuContext>(&mut ctx, LogicalMemorySpace::MainMemory);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_family_builds_lu_det_parity_primitives() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();

    run_metadata_family::<crate::CudaContext>(
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
