use tenferro_algebra::{Scalar, Standard};
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    CpuContext, MetadataCastPrimsDescriptor, MetadataDType, MetadataScalarTensorRef,
    MetadataTensorRef, TensorMetadataCastPrims,
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

fn tensor_f32(data: &[f32], dims: &[usize], memory_space: LogicalMemorySpace) -> Tensor<f32> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor)
        .unwrap()
        .to_memory_space_async(memory_space)
        .unwrap()
}

fn tensor_f64(data: &[f64], dims: &[usize], memory_space: LogicalMemorySpace) -> Tensor<f64> {
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

fn run_metadata_bridge_phase1_f32<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: crate::TensorMetadataContextFor + crate::TensorScalarContextFor<Standard<f32>>,
    C::MetadataBackend: crate::TensorMetadataPrims<Context = C>,
    C::ScalarBackend: TensorMetadataCastPrims<f32, Context = C>,
{
    let dims = [4usize];
    let mask = tensor_u8(&[1, 0, 1, 0], &dims, memory_space);
    let ints = tensor_i32(&[3, -2, 7, 5], &dims, memory_space);
    let on_true = tensor_f32(&[10.0, 20.0, 30.0, 40.0], &dims, memory_space);
    let on_false = tensor_f32(&[-1.0, -2.0, -3.0, -4.0], &dims, memory_space);

    let mut mask_as_scalar =
        Tensor::<f32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let cast_bool_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::Bool,
    };
    assert!(
        <C::ScalarBackend as TensorMetadataCastPrims<f32>>::has_metadata_cast_support(
            cast_bool_desc.clone()
        )
    );
    let cast_bool_plan = <C::ScalarBackend as TensorMetadataCastPrims<f32>>::plan(
        ctx,
        &cast_bool_desc,
        &[&dims, &dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f32>>::execute(
        ctx,
        &cast_bool_plan,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(
            &mask,
        ))],
        0.0,
        &mut mask_as_scalar,
    )
    .unwrap();
    assert_tensor_eq(&mask_as_scalar, &[1.0, 0.0, 1.0, 0.0]);

    let mut ints_as_scalar =
        Tensor::<f32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let cast_i32_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::I32,
    };
    assert!(
        <C::ScalarBackend as TensorMetadataCastPrims<f32>>::has_metadata_cast_support(
            cast_i32_desc.clone()
        )
    );
    let cast_i32_plan = <C::ScalarBackend as TensorMetadataCastPrims<f32>>::plan(
        ctx,
        &cast_i32_desc,
        &[&dims, &dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f32>>::execute(
        ctx,
        &cast_i32_plan,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::I32(
            &ints,
        ))],
        0.0,
        &mut ints_as_scalar,
    )
    .unwrap();
    assert_tensor_eq(&ints_as_scalar, &[3.0, -2.0, 7.0, 5.0]);

    let mut where_out =
        Tensor::<f32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let where_desc = MetadataCastPrimsDescriptor::Where {
        cond_dtype: MetadataDType::Bool,
    };
    let where_plan = <C::ScalarBackend as TensorMetadataCastPrims<f32>>::plan(
        ctx,
        &where_desc,
        &[&dims, &dims, &dims, &dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f32>>::execute(
        ctx,
        &where_plan,
        1.0,
        &[
            MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(&mask)),
            MetadataScalarTensorRef::Scalar(&on_true),
            MetadataScalarTensorRef::Scalar(&on_false),
        ],
        0.0,
        &mut where_out,
    )
    .unwrap();
    assert_tensor_eq(&where_out, &[10.0, -2.0, 30.0, -4.0]);
}

fn run_metadata_bridge_phase1_f64<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: crate::TensorMetadataContextFor + crate::TensorScalarContextFor<Standard<f64>>,
    C::MetadataBackend: crate::TensorMetadataPrims<Context = C>,
    C::ScalarBackend: TensorMetadataCastPrims<f64, Context = C>,
{
    let dims = [4usize];
    let mask = tensor_u8(&[0, 1, 1, 0], &dims, memory_space);
    let ints = tensor_i32(&[-3, 2, 11, 5], &dims, memory_space);
    let on_true = tensor_f64(&[1.0, 2.0, 3.0, 4.0], &dims, memory_space);
    let on_false = tensor_f64(&[9.0, 8.0, 7.0, 6.0], &dims, memory_space);

    let mut mask_as_scalar =
        Tensor::<f64>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let cast_bool_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::Bool,
    };
    let cast_bool_plan = <C::ScalarBackend as TensorMetadataCastPrims<f64>>::plan(
        ctx,
        &cast_bool_desc,
        &[&dims, &dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f64>>::execute(
        ctx,
        &cast_bool_plan,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(
            &mask,
        ))],
        0.0,
        &mut mask_as_scalar,
    )
    .unwrap();
    assert_tensor_eq(&mask_as_scalar, &[0.0, 1.0, 1.0, 0.0]);

    let mut ints_as_scalar =
        Tensor::<f64>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let cast_i32_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::I32,
    };
    let cast_i32_plan = <C::ScalarBackend as TensorMetadataCastPrims<f64>>::plan(
        ctx,
        &cast_i32_desc,
        &[&dims, &dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f64>>::execute(
        ctx,
        &cast_i32_plan,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::I32(
            &ints,
        ))],
        0.0,
        &mut ints_as_scalar,
    )
    .unwrap();
    assert_tensor_eq(&ints_as_scalar, &[-3.0, 2.0, 11.0, 5.0]);

    let mut where_out =
        Tensor::<f64>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let where_desc = MetadataCastPrimsDescriptor::Where {
        cond_dtype: MetadataDType::Bool,
    };
    let where_plan = <C::ScalarBackend as TensorMetadataCastPrims<f64>>::plan(
        ctx,
        &where_desc,
        &[&dims, &dims, &dims, &dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f64>>::execute(
        ctx,
        &where_plan,
        1.0,
        &[
            MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(&mask)),
            MetadataScalarTensorRef::Scalar(&on_true),
            MetadataScalarTensorRef::Scalar(&on_false),
        ],
        0.0,
        &mut where_out,
    )
    .unwrap();
    assert_tensor_eq(&where_out, &[9.0, 2.0, 3.0, 6.0]);
}

fn run_metadata_bridge_phase1_broadcast_f32<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: crate::TensorMetadataContextFor + crate::TensorScalarContextFor<Standard<f32>>,
    C::MetadataBackend: crate::TensorMetadataPrims<Context = C>,
    C::ScalarBackend: TensorMetadataCastPrims<f32, Context = C>,
{
    let lhs_dims = [2usize, 1usize];
    let output_dims = [2usize, 3usize];
    let mask = tensor_u8(&[1, 0], &lhs_dims, memory_space);
    let ints = tensor_i32(&[4, -7], &lhs_dims, memory_space);
    let on_true = tensor_f32(&[10.0, 20.0], &lhs_dims, memory_space);
    let on_false = tensor_f32(&[-1.0, -2.0], &lhs_dims, memory_space);

    let mut mask_as_scalar =
        Tensor::<f32>::zeros(&output_dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let cast_bool_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::Bool,
    };
    let cast_bool_plan = <C::ScalarBackend as TensorMetadataCastPrims<f32>>::plan(
        ctx,
        &cast_bool_desc,
        &[&lhs_dims, &output_dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f32>>::execute(
        ctx,
        &cast_bool_plan,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(
            &mask,
        ))],
        0.0,
        &mut mask_as_scalar,
    )
    .unwrap();
    assert_tensor_eq(&mask_as_scalar, &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

    let mut ints_as_scalar =
        Tensor::<f32>::zeros(&output_dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let cast_i32_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::I32,
    };
    let cast_i32_plan = <C::ScalarBackend as TensorMetadataCastPrims<f32>>::plan(
        ctx,
        &cast_i32_desc,
        &[&lhs_dims, &output_dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f32>>::execute(
        ctx,
        &cast_i32_plan,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::I32(
            &ints,
        ))],
        0.0,
        &mut ints_as_scalar,
    )
    .unwrap();
    assert_tensor_eq(&ints_as_scalar, &[4.0, -7.0, 4.0, -7.0, 4.0, -7.0]);

    let mut where_out =
        Tensor::<f32>::zeros(&output_dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let where_desc = MetadataCastPrimsDescriptor::Where {
        cond_dtype: MetadataDType::Bool,
    };
    let where_plan = <C::ScalarBackend as TensorMetadataCastPrims<f32>>::plan(
        ctx,
        &where_desc,
        &[&lhs_dims, &output_dims, &output_dims, &output_dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f32>>::execute(
        ctx,
        &where_plan,
        1.0,
        &[
            MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(&mask)),
            MetadataScalarTensorRef::Scalar(&on_true),
            MetadataScalarTensorRef::Scalar(&on_false),
        ],
        0.0,
        &mut where_out,
    )
    .unwrap();
    assert_tensor_eq(&where_out, &[10.0, -2.0, 10.0, -2.0, 10.0, -2.0]);
}

fn run_metadata_bridge_phase1_broadcast_f64<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: crate::TensorMetadataContextFor + crate::TensorScalarContextFor<Standard<f64>>,
    C::MetadataBackend: crate::TensorMetadataPrims<Context = C>,
    C::ScalarBackend: TensorMetadataCastPrims<f64, Context = C>,
{
    let lhs_dims = [2usize, 1usize];
    let output_dims = [2usize, 3usize];
    let mask = tensor_u8(&[0, 1], &lhs_dims, memory_space);
    let ints = tensor_i32(&[8, -5], &lhs_dims, memory_space);
    let on_true = tensor_f64(&[1.0, 2.0], &lhs_dims, memory_space);
    let on_false = tensor_f64(&[9.0, 7.0], &lhs_dims, memory_space);

    let mut mask_as_scalar =
        Tensor::<f64>::zeros(&output_dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let cast_bool_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::Bool,
    };
    let cast_bool_plan = <C::ScalarBackend as TensorMetadataCastPrims<f64>>::plan(
        ctx,
        &cast_bool_desc,
        &[&lhs_dims, &output_dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f64>>::execute(
        ctx,
        &cast_bool_plan,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(
            &mask,
        ))],
        0.0,
        &mut mask_as_scalar,
    )
    .unwrap();
    assert_tensor_eq(&mask_as_scalar, &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

    let mut ints_as_scalar =
        Tensor::<f64>::zeros(&output_dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let cast_i32_desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::I32,
    };
    let cast_i32_plan = <C::ScalarBackend as TensorMetadataCastPrims<f64>>::plan(
        ctx,
        &cast_i32_desc,
        &[&lhs_dims, &output_dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f64>>::execute(
        ctx,
        &cast_i32_plan,
        1.0,
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::I32(
            &ints,
        ))],
        0.0,
        &mut ints_as_scalar,
    )
    .unwrap();
    assert_tensor_eq(&ints_as_scalar, &[8.0, -5.0, 8.0, -5.0, 8.0, -5.0]);

    let mut where_out =
        Tensor::<f64>::zeros(&output_dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let where_desc = MetadataCastPrimsDescriptor::Where {
        cond_dtype: MetadataDType::Bool,
    };
    let where_plan = <C::ScalarBackend as TensorMetadataCastPrims<f64>>::plan(
        ctx,
        &where_desc,
        &[&lhs_dims, &output_dims, &output_dims, &output_dims],
    )
    .unwrap();
    <C::ScalarBackend as TensorMetadataCastPrims<f64>>::execute(
        ctx,
        &where_plan,
        1.0,
        &[
            MetadataScalarTensorRef::Metadata(MetadataTensorRef::Bool(&mask)),
            MetadataScalarTensorRef::Scalar(&on_true),
            MetadataScalarTensorRef::Scalar(&on_false),
        ],
        0.0,
        &mut where_out,
    )
    .unwrap();
    assert_tensor_eq(&where_out, &[9.0, 2.0, 9.0, 2.0, 9.0, 2.0]);
}

#[test]
fn cpu_metadata_bridge_phase1_supports_bool_i32_casts_and_where_for_f32() {
    let mut ctx = CpuContext::new(1);
    run_metadata_bridge_phase1_f32::<CpuContext>(&mut ctx, LogicalMemorySpace::MainMemory);
}

#[test]
fn cpu_metadata_bridge_phase1_supports_bool_i32_casts_and_where_for_f64() {
    let mut ctx = CpuContext::new(1);
    run_metadata_bridge_phase1_f64::<CpuContext>(&mut ctx, LogicalMemorySpace::MainMemory);
}

#[test]
fn cpu_metadata_bridge_phase1_supports_broadcast_bool_i32_casts_and_where_for_f32() {
    let mut ctx = CpuContext::new(1);
    run_metadata_bridge_phase1_broadcast_f32::<CpuContext>(
        &mut ctx,
        LogicalMemorySpace::MainMemory,
    );
}

#[test]
fn cpu_metadata_bridge_phase1_supports_broadcast_bool_i32_casts_and_where_for_f64() {
    let mut ctx = CpuContext::new(1);
    run_metadata_bridge_phase1_broadcast_f64::<CpuContext>(
        &mut ctx,
        LogicalMemorySpace::MainMemory,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_bridge_phase1_supports_bool_i32_casts_and_where_for_f32() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    run_metadata_bridge_phase1_f32::<crate::CudaContext>(
        &mut ctx,
        LogicalMemorySpace::GpuMemory { device_id },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_bridge_phase1_supports_bool_i32_casts_and_where_for_f64() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    run_metadata_bridge_phase1_f64::<crate::CudaContext>(
        &mut ctx,
        LogicalMemorySpace::GpuMemory { device_id },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_bridge_phase1_supports_broadcast_bool_i32_casts_and_where_for_f32() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    run_metadata_bridge_phase1_broadcast_f32::<crate::CudaContext>(
        &mut ctx,
        LogicalMemorySpace::GpuMemory { device_id },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_bridge_phase1_supports_broadcast_bool_i32_casts_and_where_for_f64() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();
    run_metadata_bridge_phase1_broadcast_f64::<crate::CudaContext>(
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
