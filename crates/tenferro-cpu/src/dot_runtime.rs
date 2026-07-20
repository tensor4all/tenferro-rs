use tenferro_tensor::{
    Buffer, DotGeneralAccumulation, DotGeneralConfig, ShapeMismatch, Tensor, TensorRead,
    TensorView, TensorViewMut, TensorWrite, ValidationError,
};

use crate::provider::{CpuContractionAxes, CpuDotGeneralRequest};
use crate::{Error, Result};

const OP: &str = "dot_general";

fn validate_axis_ranges(axes: &[usize], rank: usize) -> Result<()> {
    for &axis in axes {
        if axis >= rank {
            return Err(Error::axis_out_of_bounds(OP, axis, rank));
        }
    }
    Ok(())
}

fn role_mask(axes: &[usize], rank: usize, role: &'static str) -> Result<Option<u64>> {
    if rank > 64 {
        for (position, &axis) in axes.iter().enumerate() {
            if axes[..position].contains(&axis) {
                return Err(Error::duplicate_axis(OP, axis, role));
            }
        }
        return Ok(None);
    }

    let mut mask = 0_u64;
    for &axis in axes {
        let bit = 1_u64 << axis;
        if mask & bit != 0 {
            return Err(Error::duplicate_axis(OP, axis, role));
        }
        mask |= bit;
    }
    Ok(Some(mask))
}

fn validate_disjoint(
    first: &[usize],
    first_mask: Option<u64>,
    first_role: &'static str,
    second: &[usize],
    second_mask: Option<u64>,
    second_role: &'static str,
) -> Result<()> {
    let overlap = match (first_mask, second_mask) {
        (Some(first), Some(second)) => first & second,
        _ => 0,
    };
    let conflict = if overlap != 0 || first_mask.is_none() {
        first.iter().copied().find(|axis| second.contains(axis))
    } else {
        None
    };
    if let Some(axis) = conflict {
        return Err(Error::validation(
            OP,
            ValidationError::AxisRoleConflict {
                axis,
                first_role,
                second_role,
            },
        ));
    }
    Ok(())
}

pub(crate) fn validate_axis_groups<'a>(
    lhs_rank: usize,
    rhs_rank: usize,
    config: &'a DotGeneralConfig,
) -> Result<CpuContractionAxes<'a>> {
    validate_axis_ranges(&config.lhs_contracting_dims, lhs_rank)?;
    validate_axis_ranges(&config.rhs_contracting_dims, rhs_rank)?;
    validate_axis_ranges(&config.lhs_batch_dims, lhs_rank)?;
    validate_axis_ranges(&config.rhs_batch_dims, rhs_rank)?;

    let lhs_contracting_mask = role_mask(
        &config.lhs_contracting_dims,
        lhs_rank,
        "lhs_contracting_dims",
    )?;
    let rhs_contracting_mask = role_mask(
        &config.rhs_contracting_dims,
        rhs_rank,
        "rhs_contracting_dims",
    )?;
    let lhs_batch_mask = role_mask(&config.lhs_batch_dims, lhs_rank, "lhs_batch_dims")?;
    let rhs_batch_mask = role_mask(&config.rhs_batch_dims, rhs_rank, "rhs_batch_dims")?;

    validate_disjoint(
        &config.lhs_contracting_dims,
        lhs_contracting_mask,
        "lhs contracting",
        &config.lhs_batch_dims,
        lhs_batch_mask,
        "lhs batch",
    )?;
    validate_disjoint(
        &config.rhs_contracting_dims,
        rhs_contracting_mask,
        "rhs contracting",
        &config.rhs_batch_dims,
        rhs_batch_mask,
        "rhs batch",
    )?;

    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(Error::invalid_argument(
            OP,
            "dot_general_config",
            format!(
                "lhs/rhs contracting dim counts differ ({} vs {})",
                config.lhs_contracting_dims.len(),
                config.rhs_contracting_dims.len(),
            ),
        ));
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(Error::invalid_argument(
            OP,
            "dot_general_config",
            format!(
                "lhs/rhs batch dim counts differ ({} vs {})",
                config.lhs_batch_dims.len(),
                config.rhs_batch_dims.len(),
            ),
        ));
    }

    Ok(CpuContractionAxes::new(
        lhs_rank,
        rhs_rank,
        &config.lhs_contracting_dims,
        &config.rhs_contracting_dims,
        &config.lhs_batch_dims,
        &config.rhs_batch_dims,
        lhs_contracting_mask.zip(lhs_batch_mask).map(|(a, b)| a | b),
        rhs_contracting_mask.zip(rhs_batch_mask).map(|(a, b)| a | b),
    ))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ValidatedDotGeneral<'a> {
    axes: CpuContractionAxes<'a>,
    output_element_count: usize,
}

impl<'a> ValidatedDotGeneral<'a> {
    pub(crate) fn axes(&self) -> &CpuContractionAxes<'a> {
        &self.axes
    }

    pub(crate) fn output_element_count(&self) -> usize {
        self.output_element_count
    }

    #[allow(dead_code)]
    pub(crate) fn request<'request, 'input, 'output>(
        &'request self,
        lhs: &'request TensorRead<'input>,
        rhs: &'request TensorRead<'input>,
        output: &'request mut TensorWrite<'output>,
        accumulation: DotGeneralAccumulation,
    ) -> CpuDotGeneralRequest<'request, 'input, 'output>
    where
        'a: 'request,
    {
        CpuDotGeneralRequest::new(lhs, rhs, output, self.axes, accumulation)
    }
}

fn validate_paired_extents(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    axes: &CpuContractionAxes<'_>,
) -> Result<()> {
    for (lhs_axis, rhs_axis) in axes.contracting_pairs().chain(axes.batch_pairs()) {
        if lhs.shape()[lhs_axis] != rhs.shape()[rhs_axis] {
            return Err(Error::validation(
                OP,
                ShapeMismatch::ContractedDimensions {
                    lhs_axis,
                    lhs_size: lhs.shape()[lhs_axis],
                    rhs_axis,
                    rhs_size: rhs.shape()[rhs_axis],
                }
                .into(),
            ));
        }
    }
    Ok(())
}

fn expected_output_shape(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    axes: &CpuContractionAxes<'_>,
) -> Vec<usize> {
    axes.lhs_free_axes()
        .map(|axis| lhs.shape()[axis])
        .chain(axes.rhs_free_axes().map(|axis| rhs.shape()[axis]))
        .chain(
            axes.batch_pairs()
                .map(|(lhs_axis, _)| lhs.shape()[lhs_axis]),
        )
        .collect()
}

fn output_shape_matches(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    output: &TensorWrite<'_>,
    axes: &CpuContractionAxes<'_>,
) -> Result<()> {
    let expected_rank =
        axes.lhs_free_axes().count() + axes.rhs_free_axes().count() + axes.batch_pairs().len();
    let mut actual = output.shape().iter().copied();
    let matches = output.shape().len() == expected_rank
        && axes
            .lhs_free_axes()
            .map(|axis| lhs.shape()[axis])
            .chain(axes.rhs_free_axes().map(|axis| rhs.shape()[axis]))
            .chain(
                axes.batch_pairs()
                    .map(|(lhs_axis, _)| lhs.shape()[lhs_axis]),
            )
            .all(|expected| actual.next() == Some(expected));
    if matches {
        return Ok(());
    }

    Err(Error::validation(
        OP,
        ShapeMismatch::ExpectedActual {
            expected: expected_output_shape(lhs, rhs, axes).into(),
            actual: output.shape().to_vec().into(),
        }
        .into(),
    ))
}

fn layout_overflow() -> Error {
    Error::validation(OP, ValidationError::IntegerOverflow)
}

pub(crate) fn validate_layout_metadata(
    role: &'static str,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    storage_len: usize,
) -> Result<usize> {
    if shape.len() != strides.len() {
        return Err(Error::validation(
            OP,
            ValidationError::RankMismatch {
                expected: shape.len(),
                actual: strides.len(),
            },
        ));
    }
    let element_count = tenferro_tensor::validate::checked_shape_product(OP, role, shape)?;

    if shape.contains(&0) {
        let offset = usize::try_from(offset).map_err(|_| {
            Error::invalid_argument(OP, role, "minimum reachable offset is negative")
        })?;
        if offset > storage_len {
            return Err(Error::validation(OP, ValidationError::ViewOutOfBounds));
        }
        return Ok(element_count);
    }

    let mut minimum = offset;
    let mut maximum = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let steps = isize::try_from(extent - 1).map_err(|_| layout_overflow())?;
        let end = stride.checked_mul(steps).ok_or_else(layout_overflow)?;
        let (axis_minimum, axis_maximum) = if end < 0 { (end, 0) } else { (0, end) };
        minimum = minimum
            .checked_add(axis_minimum)
            .ok_or_else(layout_overflow)?;
        maximum = maximum
            .checked_add(axis_maximum)
            .ok_or_else(layout_overflow)?;
    }
    let minimum = usize::try_from(minimum)
        .map_err(|_| Error::invalid_argument(OP, role, "minimum reachable offset is negative"))?;
    let maximum = usize::try_from(maximum)
        .map_err(|_| Error::invalid_argument(OP, role, "maximum reachable offset is negative"))?;
    if minimum > maximum || maximum >= storage_len {
        return Err(Error::validation(OP, ValidationError::ViewOutOfBounds));
    }
    Ok(element_count)
}

macro_rules! validate_owned_layout {
    ($tensor:expr, $role:expr) => {{
        let tensor = $tensor;
        let storage_len = match tensor.buffer() {
            Buffer::Host(storage) => storage.len(),
            Buffer::Backend(_) => return Err(crate::cpu_backend_buffer_error(OP)),
        };
        validate_layout_metadata(
            $role,
            tensor.shape(),
            tensor.layout().strides(),
            tensor.layout().offset(),
            storage_len,
        )
    }};
}

macro_rules! validate_read_view_layout {
    ($view:expr, $role:expr) => {{
        let view = $view;
        let storage_len = view.host_storage()?.len();
        validate_layout_metadata(
            $role,
            view.shape(),
            view.strides(),
            view.offset(),
            storage_len,
        )
    }};
}

macro_rules! validate_write_view_layout {
    ($view:expr, $role:expr) => {{
        let view = $view;
        let storage_len = view.host_storage()?.len();
        validate_layout_metadata(
            $role,
            view.shape(),
            view.strides(),
            view.offset(),
            storage_len,
        )
    }};
}

fn validate_read_layout(tensor: &TensorRead<'_>, role: &'static str) -> Result<usize> {
    match tensor {
        TensorRead::Tensor(tensor) => match tensor {
            Tensor::F32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::F64(tensor) => validate_owned_layout!(tensor, role),
            Tensor::I32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::I64(tensor) => validate_owned_layout!(tensor, role),
            Tensor::Bool(tensor) => validate_owned_layout!(tensor, role),
            Tensor::C32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::C64(tensor) => validate_owned_layout!(tensor, role),
        },
        TensorRead::View(view) => match view {
            TensorView::F32(view) => validate_read_view_layout!(view, role),
            TensorView::F64(view) => validate_read_view_layout!(view, role),
            TensorView::I32(view) => validate_read_view_layout!(view, role),
            TensorView::I64(view) => validate_read_view_layout!(view, role),
            TensorView::Bool(view) => validate_read_view_layout!(view, role),
            TensorView::C32(view) => validate_read_view_layout!(view, role),
            TensorView::C64(view) => validate_read_view_layout!(view, role),
        },
    }
}

fn validate_write_layout(tensor: &TensorWrite<'_>, role: &'static str) -> Result<usize> {
    match tensor {
        TensorWrite::Tensor(tensor) => match tensor {
            Tensor::F32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::F64(tensor) => validate_owned_layout!(tensor, role),
            Tensor::I32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::I64(tensor) => validate_owned_layout!(tensor, role),
            Tensor::Bool(tensor) => validate_owned_layout!(tensor, role),
            Tensor::C32(tensor) => validate_owned_layout!(tensor, role),
            Tensor::C64(tensor) => validate_owned_layout!(tensor, role),
        },
        TensorWrite::View(view) => match view {
            TensorViewMut::F32(view) => validate_write_view_layout!(view, role),
            TensorViewMut::F64(view) => validate_write_view_layout!(view, role),
            TensorViewMut::I32(view) => validate_write_view_layout!(view, role),
            TensorViewMut::I64(view) => validate_write_view_layout!(view, role),
            TensorViewMut::Bool(view) => validate_write_view_layout!(view, role),
            TensorViewMut::C32(view) => validate_write_view_layout!(view, role),
            TensorViewMut::C64(view) => validate_write_view_layout!(view, role),
        },
    }
}

pub(crate) fn validate_dot_general<'a>(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    output: &TensorWrite<'_>,
    config: &'a DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
) -> Result<ValidatedDotGeneral<'a>> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::dtype_mismatch(OP, lhs.dtype(), rhs.dtype()));
    }
    if output.dtype() != lhs.dtype() {
        return Err(Error::dtype_mismatch(OP, output.dtype(), lhs.dtype()));
    }
    if accumulation.alpha.dtype() != lhs.dtype() {
        return Err(Error::dtype_mismatch(
            OP,
            lhs.dtype(),
            accumulation.alpha.dtype(),
        ));
    }
    if accumulation.beta.dtype() != lhs.dtype() {
        return Err(Error::dtype_mismatch(
            OP,
            lhs.dtype(),
            accumulation.beta.dtype(),
        ));
    }

    crate::structural::validate_cpu_host_placement(OP, "lhs", read_placement(lhs))?;
    crate::structural::validate_cpu_host_placement(OP, "rhs", read_placement(rhs))?;
    crate::structural::validate_cpu_host_placement(OP, "output", write_placement(output))?;
    validate_read_layout(lhs, "lhs")?;
    validate_read_layout(rhs, "rhs")?;
    let output_element_count = validate_write_layout(output, "output")?;

    let axes = validate_axis_groups(lhs.shape().len(), rhs.shape().len(), config)?;
    validate_paired_extents(lhs, rhs, &axes)?;
    output_shape_matches(lhs, rhs, output, &axes)?;

    Ok(ValidatedDotGeneral {
        axes,
        output_element_count,
    })
}

fn read_placement<'a>(tensor: &'a TensorRead<'_>) -> &'a tenferro_tensor::Placement {
    match tensor {
        TensorRead::Tensor(tensor) => tensor.placement(),
        TensorRead::View(view) => match view {
            tenferro_tensor::TensorView::F32(view) => view.placement(),
            tenferro_tensor::TensorView::F64(view) => view.placement(),
            tenferro_tensor::TensorView::I32(view) => view.placement(),
            tenferro_tensor::TensorView::I64(view) => view.placement(),
            tenferro_tensor::TensorView::Bool(view) => view.placement(),
            tenferro_tensor::TensorView::C32(view) => view.placement(),
            tenferro_tensor::TensorView::C64(view) => view.placement(),
        },
    }
}

fn write_placement<'a>(tensor: &'a TensorWrite<'_>) -> &'a tenferro_tensor::Placement {
    match tensor {
        TensorWrite::Tensor(tensor) => tensor.placement(),
        TensorWrite::View(view) => match view {
            tenferro_tensor::TensorViewMut::F32(view) => view.placement(),
            tenferro_tensor::TensorViewMut::F64(view) => view.placement(),
            tenferro_tensor::TensorViewMut::I32(view) => view.placement(),
            tenferro_tensor::TensorViewMut::I64(view) => view.placement(),
            tenferro_tensor::TensorViewMut::Bool(view) => view.placement(),
            tenferro_tensor::TensorViewMut::C32(view) => view.placement(),
            tenferro_tensor::TensorViewMut::C64(view) => view.placement(),
        },
    }
}

#[cfg(test)]
mod tests;
