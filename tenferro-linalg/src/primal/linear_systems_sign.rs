use super::*;
use num_traits::{NumCast, One};
use tenferro_prims::{
    MetadataBinaryOp, MetadataCastPrimsDescriptor, MetadataConstantValue, MetadataDType,
    MetadataGenerateOp, MetadataPrimsDescriptor, MetadataReductionOp, MetadataScalarTensorRef,
    MetadataTensorMut, MetadataTensorRef, TensorMetadataCastPrims, TensorMetadataContextFor,
    TensorMetadataPrims,
};

pub(crate) fn metadata_generate_i32_same_shape<C>(
    ctx: &mut C,
    dims: &[usize],
    op: MetadataGenerateOp,
    memory_space: tenferro_device::LogicalMemorySpace,
) -> Result<Tensor<i32>>
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let mut output = Tensor::zeros(dims, memory_space, MemoryOrder::ColumnMajor)?;
    let desc = MetadataPrimsDescriptor::Generate {
        op,
        output_dtype: MetadataDType::I32,
    };
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &desc,
        &[],
        MetadataTensorMut::I32(&mut output),
    )?;
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[],
        MetadataTensorMut::I32(&mut output),
    )?;
    Ok(output)
}

pub(crate) fn metadata_binary_i32_same_shape<C>(
    ctx: &mut C,
    lhs: &Tensor<i32>,
    rhs: &Tensor<i32>,
    op: MetadataBinaryOp,
) -> Result<Tensor<i32>>
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let mut output = Tensor::zeros(
        lhs.dims(),
        lhs.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let desc = MetadataPrimsDescriptor::Binary {
        op,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::I32,
    };
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &desc,
        &[MetadataTensorRef::I32(lhs), MetadataTensorRef::I32(rhs)],
        MetadataTensorMut::I32(&mut output),
    )?;
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[MetadataTensorRef::I32(lhs), MetadataTensorRef::I32(rhs)],
        MetadataTensorMut::I32(&mut output),
    )?;
    Ok(output)
}

pub(crate) fn metadata_binary_i32_bool_same_shape<C>(
    ctx: &mut C,
    lhs: &Tensor<i32>,
    rhs: &Tensor<i32>,
    op: MetadataBinaryOp,
) -> Result<Tensor<u8>>
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    let mut output = Tensor::zeros(
        lhs.dims(),
        lhs.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let desc = MetadataPrimsDescriptor::Binary {
        op,
        lhs_dtype: MetadataDType::I32,
        rhs_dtype: MetadataDType::I32,
        output_dtype: MetadataDType::Bool,
    };
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &desc,
        &[MetadataTensorRef::I32(lhs), MetadataTensorRef::I32(rhs)],
        MetadataTensorMut::Bool(&mut output),
    )?;
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[MetadataTensorRef::I32(lhs), MetadataTensorRef::I32(rhs)],
        MetadataTensorMut::Bool(&mut output),
    )?;
    Ok(output)
}

pub(crate) fn metadata_reduce_bool_sum_keep_batch_axes<C>(
    ctx: &mut C,
    input: &Tensor<u8>,
) -> Result<Tensor<i32>>
where
    C: TensorMetadataContextFor,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
{
    if input.ndim() == 0 {
        return Err(Error::InvalidArgument(
            "metadata reduction expects rank >= 1".into(),
        ));
    }

    let output_dims = input.dims()[1..].to_vec();
    let mut output = Tensor::zeros(
        &output_dims,
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let modes_a = (0..input.ndim())
        .map(|mode| mode as u32)
        .collect::<Vec<_>>();
    let modes_c = (1..input.ndim())
        .map(|mode| mode as u32)
        .collect::<Vec<_>>();
    let desc = MetadataPrimsDescriptor::Reduction {
        modes_a,
        modes_c,
        input_dtype: MetadataDType::Bool,
        output_dtype: MetadataDType::I32,
        op: MetadataReductionOp::Sum,
    };
    let plan = <C::MetadataBackend as TensorMetadataPrims>::plan(
        ctx,
        &desc,
        &[MetadataTensorRef::Bool(input)],
        MetadataTensorMut::I32(&mut output),
    )?;
    <C::MetadataBackend as TensorMetadataPrims>::execute(
        ctx,
        &plan,
        &[MetadataTensorRef::Bool(input)],
        MetadataTensorMut::I32(&mut output),
    )?;
    Ok(output)
}

pub(crate) fn metadata_cast_i32_to_scalar_same_shape<C, S>(
    ctx: &mut C,
    input: &Tensor<i32>,
) -> Result<Tensor<S>>
where
    S: Scalar + NumCast + One + 'static,
    C: TensorMetadataContextFor
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<S>>,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
    C::ScalarBackend: TensorMetadataCastPrims<S, Context = C>,
{
    let mut output = Tensor::zeros(
        input.dims(),
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let desc = MetadataCastPrimsDescriptor::PointwiseCast {
        input_dtype: MetadataDType::I32,
    };
    let plan = <C::ScalarBackend as TensorMetadataCastPrims<S>>::plan(
        ctx,
        &desc,
        &[input.dims(), output.dims()],
    )?;
    <C::ScalarBackend as TensorMetadataCastPrims<S>>::execute(
        ctx,
        &plan,
        S::one(),
        &[MetadataScalarTensorRef::Metadata(MetadataTensorRef::I32(
            input,
        ))],
        S::zero(),
        &mut output,
    )?;
    Ok(output)
}

pub(crate) fn lu_permutation_sign_tensor<T, C>(
    ctx: &mut C,
    pivots: &Tensor<i32>,
) -> Result<Tensor<T::Real>>
where
    T: KernelLinalgScalar,
    T::Real: Scalar + NumCast + One + 'static,
    C: TensorMetadataContextFor
        + tenferro_prims::TensorScalarContextFor<tenferro_algebra::Standard<T::Real>>,
    C::MetadataBackend: TensorMetadataPrims<Context = C>,
    C::ScalarBackend: TensorMetadataCastPrims<T::Real, Context = C>,
{
    let dims = pivots.dims();
    if dims.is_empty() {
        return Err(Error::InvalidArgument(
            "lu permutation sign expects a non-scalar pivot tensor".into(),
        ));
    }

    let memory_space = pivots.logical_memory_space();
    let step_count = dims[0];
    let mut step_indices = metadata_generate_i32_same_shape(
        ctx,
        &[step_count],
        MetadataGenerateOp::IotaStartZero,
        memory_space,
    )?;
    let ones = metadata_generate_i32_same_shape(
        ctx,
        &[step_count],
        MetadataGenerateOp::Constant(MetadataConstantValue::I32(1)),
        memory_space,
    )?;
    step_indices =
        metadata_binary_i32_same_shape(ctx, &step_indices, &ones, MetadataBinaryOp::Add)?;
    for _ in &dims[1..] {
        step_indices = step_indices.unsqueeze(-1)?;
    }
    let step_indices = step_indices.broadcast(dims)?;
    let mismatches = metadata_binary_i32_bool_same_shape(
        ctx,
        &step_indices,
        pivots,
        MetadataBinaryOp::NotEqual,
    )?;
    let swap_count = metadata_reduce_bool_sum_keep_batch_axes(ctx, &mismatches)?;
    let parity_mask = metadata_generate_i32_same_shape(
        ctx,
        swap_count.dims(),
        MetadataGenerateOp::Constant(MetadataConstantValue::I32(1)),
        memory_space,
    )?;
    let parity =
        metadata_binary_i32_same_shape(ctx, &swap_count, &parity_mask, MetadataBinaryOp::BitAnd)?;
    let parity_real = metadata_cast_i32_to_scalar_same_shape::<C, T::Real>(ctx, &parity)?;
    let sign_one =
        crate::prims_bridge::full_like_constant(T::Real::one(), parity_real.dims(), memory_space)?;
    let two = crate::prims_bridge::full_like_constant(
        T::Real::one() + T::Real::one(),
        parity_real.dims(),
        memory_space,
    )?;
    let two_parity = crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &parity_real,
        &two,
        tenferro_prims::ScalarBinaryOp::Mul,
    )?;
    crate::prims_bridge::scalar_binary_same_shape(
        ctx,
        &sign_one,
        &two_parity,
        tenferro_prims::ScalarBinaryOp::Sub,
    )
}
