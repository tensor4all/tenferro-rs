use tenferro_algebra::Scalar;
use tenferro_device::{unflatten_col_major_index_into, LogicalMemorySpace};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    CpuContext, MetadataBinaryOp, MetadataConstantValue, MetadataDType, MetadataGenerateOp,
    MetadataPrimsDescriptor, MetadataReductionOp, MetadataTensorMut, MetadataTensorRef,
    MetadataTernaryOp, TensorMetadataContextFor, TensorMetadataPrims,
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

fn flatten_col_major_index(idx: &[usize], dims: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (&coord, &dim) in idx.iter().zip(dims) {
        flat += coord * stride;
        stride *= dim;
    }
    flat
}

fn expected_permuted_metadata_reduction_sum(
    input_data: &[i32],
    input_dims: &[usize],
    modes_a: &[usize],
    modes_c: &[usize],
) -> Vec<i32> {
    let output_dims: Vec<usize> = modes_c.iter().map(|&axis| input_dims[axis]).collect();
    let reduced_axes: Vec<usize> = modes_a
        .iter()
        .copied()
        .filter(|axis| !modes_c.contains(axis))
        .collect();
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&axis| input_dims[axis]).collect();
    let output_total = output_dims.iter().product();
    let reduced_total = reduced_dims.iter().product();
    let mut output = vec![0i32; output_total];
    let mut out_idx = vec![0usize; output_dims.len()];
    let mut red_idx = vec![0usize; reduced_dims.len()];
    let mut in_idx = vec![0usize; input_dims.len()];
    let mut output_axis_for_input_axis = vec![None; input_dims.len()];
    for (output_axis, &input_axis) in modes_c.iter().enumerate() {
        output_axis_for_input_axis[input_axis] = Some(output_axis);
    }
    let mut reduced_axis_for_input_axis = vec![None; input_dims.len()];
    for (reduced_axis, &input_axis) in reduced_axes.iter().enumerate() {
        reduced_axis_for_input_axis[input_axis] = Some(reduced_axis);
    }

    for out_flat in 0..output_total {
        unflatten_col_major_index_into(out_flat, &output_dims, &mut out_idx).unwrap();
        let mut sum = 0i32;
        for red_flat in 0..reduced_total {
            unflatten_col_major_index_into(red_flat, &reduced_dims, &mut red_idx).unwrap();
            for axis in 0..input_dims.len() {
                if let Some(output_axis) = output_axis_for_input_axis[axis] {
                    in_idx[axis] = out_idx[output_axis];
                } else if let Some(reduced_axis) = reduced_axis_for_input_axis[axis] {
                    in_idx[axis] = red_idx[reduced_axis];
                } else {
                    unreachable!("axis {axis} not classified as kept or reduced");
                }
            }
            let flat = flatten_col_major_index(&in_idx, input_dims);
            sum += input_data[flat];
        }
        output[out_flat] = sum;
    }

    output
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
    let mut iota = Tensor::<i32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
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

    let constant_i32_desc = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::Constant(MetadataConstantValue::I32(7)),
        output_dtype: MetadataDType::I32,
    };
    assert!(
        <C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(
            constant_i32_desc.clone()
        )
    );
    let mut constant_i32 =
        Tensor::<i32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &constant_i32_desc,
        &[],
        MetadataTensorMut::I32(&mut constant_i32),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[],
        MetadataTensorMut::I32(&mut constant_i32),
    )
    .unwrap();
    assert_tensor_eq(&constant_i32, &[7, 7, 7, 7]);

    let constant_bool_desc = MetadataPrimsDescriptor::Generate {
        op: MetadataGenerateOp::Constant(MetadataConstantValue::Bool(true)),
        output_dtype: MetadataDType::Bool,
    };
    assert!(
        <C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(
            constant_bool_desc.clone()
        )
    );
    let mut constant_bool =
        Tensor::<u8>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &constant_bool_desc,
        &[],
        MetadataTensorMut::Bool(&mut constant_bool),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[],
        MetadataTensorMut::Bool(&mut constant_bool),
    )
    .unwrap();
    assert_tensor_eq(&constant_bool, &[1, 1, 1, 1]);

    let lhs_i32 = tensor_i32(&[0, 1, 2, 3], &dims, memory_space);
    let rhs_i32 = tensor_i32(&[0, 0, 2, 9], &dims, memory_space);
    let mut neq = Tensor::<u8>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
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
    let mut bool_neq = Tensor::<u8>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
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

    let mut where_i32 =
        Tensor::<i32>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
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

    let mut where_bool =
        Tensor::<u8>::zeros(&dims, memory_space, MemoryOrder::ColumnMajor).unwrap();
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
    let mut reduce_output =
        Tensor::<i32>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor).unwrap();
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
    let mut reduce_i32_output =
        Tensor::<i32>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor).unwrap();
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

fn run_permuted_metadata_reduction_family<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let input_dims = [2usize, 3, 4];
    let modes_a = vec![0u32, 1, 2];
    let modes_c = vec![2u32, 0];
    let input_data: Vec<i32> = (0..24).collect();
    let input = tensor_i32(&input_data, &input_dims, memory_space);
    let mut output = Tensor::<i32>::zeros(&[4, 2], memory_space, MemoryOrder::ColumnMajor).unwrap();
    let desc = MetadataPrimsDescriptor::Reduction {
        modes_a: modes_a.clone(),
        modes_c: modes_c.clone(),
        input_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
        op: MetadataReductionOp::Sum,
    };

    assert!(<C::MetadataBackend as TensorMetadataPrims>::has_metadata_support(desc.clone()));
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &desc,
        &[MetadataTensorRef::I32(&input)],
        MetadataTensorMut::I32(&mut output),
    )
    .unwrap();
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[MetadataTensorRef::I32(&input)],
        MetadataTensorMut::I32(&mut output),
    )
    .unwrap();

    let expected =
        expected_permuted_metadata_reduction_sum(&input_data, &input_dims, &[0, 1, 2], &[2, 0]);
    assert_tensor_eq(&output, &expected);
}

fn run_duplicate_mode_reduction_rejection<C>(ctx: &mut C, memory_space: LogicalMemorySpace)
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let input = tensor_i32(&[1, 2, 3, 4, 5, 6], &[2, 3], memory_space);

    let duplicate_input_desc = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 0],
        modes_c: vec![0],
        input_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
        op: MetadataReductionOp::Sum,
    };
    let mut duplicate_input_output =
        Tensor::<i32>::zeros(&[2], memory_space, MemoryOrder::ColumnMajor).unwrap();
    assert!(<C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &duplicate_input_desc,
        &[MetadataTensorRef::I32(&input)],
        MetadataTensorMut::I32(&mut duplicate_input_output),
    )
    .is_err());

    let duplicate_output_desc = MetadataPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1, 1],
        input_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
        op: MetadataReductionOp::Sum,
    };
    let mut duplicate_output_output =
        Tensor::<i32>::zeros(&[3, 3], memory_space, MemoryOrder::ColumnMajor).unwrap();
    assert!(<C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &duplicate_output_desc,
        &[MetadataTensorRef::I32(&input)],
        MetadataTensorMut::I32(&mut duplicate_output_output),
    )
    .is_err());
}

#[test]
fn cpu_metadata_family_builds_lu_det_parity_primitives() {
    let mut ctx = CpuContext::new(1);
    run_metadata_family::<CpuContext>(&mut ctx, LogicalMemorySpace::MainMemory);
}

#[test]
fn cpu_metadata_family_handles_permuted_reduction_modes_order() {
    let mut ctx = CpuContext::new(1);
    run_permuted_metadata_reduction_family::<CpuContext>(&mut ctx, LogicalMemorySpace::MainMemory);
}

#[test]
fn cpu_metadata_family_rejects_duplicate_reduction_modes() {
    let mut ctx = CpuContext::new(1);
    run_duplicate_mode_reduction_rejection::<CpuContext>(&mut ctx, LogicalMemorySpace::MainMemory);
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
#[test]
fn cuda_metadata_family_handles_permuted_reduction_modes_order() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();

    run_permuted_metadata_reduction_family::<crate::CudaContext>(
        &mut ctx,
        LogicalMemorySpace::GpuMemory { device_id },
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_metadata_family_rejects_duplicate_reduction_modes() {
    let Some((_backend, mut ctx)) = load_cuda_backend() else {
        return;
    };
    let device_id = ctx.device_id();

    run_duplicate_mode_reduction_rejection::<crate::CudaContext>(
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
