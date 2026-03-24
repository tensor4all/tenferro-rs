use tenferro_algebra::Standard;
use tenferro_prims::{SemiringCoreDescriptor, TensorSemiringContextFor};

use super::*;

fn matrix_modes(rank: usize) -> Vec<u32> {
    (0..rank as u32).collect()
}

fn batch_modes(rank: usize) -> Vec<u32> {
    (2..rank as u32).collect()
}

fn validate_square_output_dims<'a>(
    output_dims: &'a [usize],
    op_name: &str,
) -> Result<(usize, &'a [usize])> {
    if output_dims.len() < 2 {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects at least 2 output dimensions, got {:?}",
            output_dims
        )));
    }
    if output_dims[0] != output_dims[1] {
        return Err(Error::ShapeMismatch {
            expected: vec![output_dims[0], output_dims[0]],
            got: output_dims[..2].to_vec(),
        });
    }
    Ok((output_dims[0], &output_dims[2..]))
}

fn diag_shape_from_output_dims(output_dims: &[usize]) -> Vec<usize> {
    let mut dims = Vec::with_capacity(output_dims.len().saturating_sub(1));
    dims.push(output_dims[0]);
    dims.extend_from_slice(&output_dims[2..]);
    dims
}

enum DiagonalScatterKind {
    AntiTrace { modes_a: Vec<u32> },
    AntiDiag { modes_a: Vec<u32> },
}

fn classify_diagonal_scatter(
    input_dims: &[usize],
    output_dims: &[usize],
) -> Result<DiagonalScatterKind> {
    let (_, batch_dims) = validate_square_output_dims(output_dims, "diag_scatter")?;
    if input_dims == batch_dims {
        return Ok(DiagonalScatterKind::AntiTrace {
            modes_a: batch_modes(output_dims.len()),
        });
    }

    let expected_diag_dims = diag_shape_from_output_dims(output_dims);
    if input_dims == expected_diag_dims {
        let mut modes_a = Vec::with_capacity(input_dims.len());
        modes_a.push(0);
        modes_a.extend(batch_modes(output_dims.len()));
        return Ok(DiagonalScatterKind::AntiDiag { modes_a });
    }

    Err(Error::InvalidArgument(format!(
        "diag_scatter expects input shape {:?} (batch scalars) or {:?} (diagonal values), got {:?}",
        batch_dims, expected_diag_dims, input_dims
    )))
}

pub(crate) fn diag_extract<T: LinalgScalar>(input: &Tensor<T>) -> Result<Tensor<T>> {
    validate_square(input)?;
    let diagonal = input.diagonal(&[(0, 1)])?;
    if diagonal.ndim() <= 1 {
        return Ok(diagonal);
    }

    let mut perm = Vec::with_capacity(diagonal.ndim());
    perm.push(diagonal.ndim() - 1);
    perm.extend(0..(diagonal.ndim() - 1));
    diagonal.permute(&perm)
}

pub(crate) fn trace_tensor<T, C>(ctx: &mut C, input: &Tensor<T>) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    let (_, batch_dims) = validate_square(input)?;
    let mut output = Tensor::zeros(
        batch_dims,
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let desc = SemiringCoreDescriptor::Trace {
        modes_a: matrix_modes(input.ndim()),
        modes_c: batch_modes(input.ndim()),
        paired: vec![(0, 1)],
    };
    prims_bridge::semiring_core_single_input_into(
        ctx,
        &desc,
        input,
        T::one(),
        T::zero(),
        &mut output,
    )?;
    Ok(output)
}

fn diag_scatter_into<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    beta: T,
    output: &mut Tensor<T>,
) -> Result<()>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    if input.logical_memory_space() != output.logical_memory_space() {
        return Err(Error::InvalidArgument(
            "diag_scatter expects input and output in the same logical memory space".into(),
        ));
    }

    let modes_c = matrix_modes(output.ndim());
    let paired = vec![(0, 1)];
    let desc = match classify_diagonal_scatter(input.dims(), output.dims())? {
        DiagonalScatterKind::AntiTrace { modes_a } => SemiringCoreDescriptor::AntiTrace {
            modes_a,
            modes_c,
            paired,
        },
        DiagonalScatterKind::AntiDiag { modes_a } => SemiringCoreDescriptor::AntiDiag {
            modes_a,
            modes_c,
            paired,
        },
    };
    prims_bridge::semiring_core_single_input_into(ctx, &desc, input, T::one(), beta, output)
}

pub(crate) fn diag_scatter<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    output_dims: &[usize],
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    let mut output = Tensor::zeros(
        output_dims,
        input.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    diag_scatter_into(ctx, input, T::zero(), &mut output)?;
    Ok(output)
}

pub(crate) fn diag_embed<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    output_dims: &[usize],
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    diag_scatter(ctx, input, output_dims)
}

pub(crate) fn diag_scatter_add<T, C>(
    ctx: &mut C,
    input: &Tensor<T>,
    base: &Tensor<T>,
) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorSemiringContextFor<Standard<T>>,
{
    let mut output = Tensor::stack(&[base], 0)?.squeeze_dim(0)?;
    diag_scatter_into(ctx, input, T::one(), &mut output)?;
    Ok(output)
}
