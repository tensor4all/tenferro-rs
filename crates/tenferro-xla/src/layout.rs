#![allow(dead_code)]

#[cfg(test)]
mod tests;

use std::mem;

use crate::{Error, Result};

pub(crate) fn col_major_to_row_major<T: Clone>(shape: &[usize], col_major: &[T]) -> Result<Vec<T>> {
    convert_layout(shape, col_major, LayoutDirection::ColToRow)
}

pub(crate) fn row_major_to_col_major<T: Clone>(shape: &[usize], row_major: &[T]) -> Result<Vec<T>> {
    convert_layout(shape, row_major, LayoutDirection::RowToCol)
}

pub(crate) fn col_major_byte_strides<T>(shape: &[usize]) -> Result<Vec<i64>> {
    let element_size = mem::size_of::<T>();
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1_usize;
    for &dim in shape {
        let byte_stride =
            stride
                .checked_mul(element_size)
                .ok_or_else(|| Error::InvalidProgram {
                    message: format!("shape {:?} byte stride overflows usize", shape),
                })?;
        strides.push(
            i64::try_from(byte_stride).map_err(|_| Error::InvalidProgram {
                message: format!("byte stride {byte_stride} exceeds i64 for PJRT"),
            })?,
        );
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| Error::InvalidProgram {
                message: format!("shape {:?} element stride overflows usize", shape),
            })?;
    }
    Ok(strides)
}

#[derive(Clone, Copy)]
enum LayoutDirection {
    ColToRow,
    RowToCol,
}

fn convert_layout<T: Clone>(
    shape: &[usize],
    input: &[T],
    direction: LayoutDirection,
) -> Result<Vec<T>> {
    let len = element_count(shape)?;
    if input.len() != len {
        return Err(Error::InvalidProgram {
            message: format!(
                "layout conversion for shape {:?} expected {} elements, got {}",
                shape,
                len,
                input.len()
            ),
        });
    }
    if shape.len() <= 1 || len <= 1 {
        return Ok(input.to_vec());
    }

    let col_strides = col_major_strides(shape);
    let row_strides = row_major_strides(shape);
    let mut output = Vec::with_capacity(len);
    match direction {
        LayoutDirection::ColToRow => {
            for row_offset in 0..len {
                let index = unravel_row_major(row_offset, shape);
                output.push(input[ravel(&index, &col_strides)].clone());
            }
        }
        LayoutDirection::RowToCol => {
            for col_offset in 0..len {
                let index = unravel_col_major(col_offset, shape);
                output.push(input[ravel(&index, &row_strides)].clone());
            }
        }
    }
    Ok(output)
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| Error::InvalidProgram {
            message: format!("shape {:?} element count overflows usize", shape),
        })
    })
}

fn col_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1_usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1_usize; shape.len()];
    let mut stride = 1_usize;
    for (axis, &dim) in shape.iter().enumerate().rev() {
        strides[axis] = stride;
        stride = stride.saturating_mul(dim);
    }
    strides
}

fn unravel_row_major(mut offset: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0_usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = offset % shape[axis];
        offset /= shape[axis];
    }
    index
}

fn unravel_col_major(mut offset: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0_usize; shape.len()];
    for axis in 0..shape.len() {
        index[axis] = offset % shape[axis];
        offset /= shape[axis];
    }
    index
}

fn ravel(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(dim, stride)| dim * stride)
        .sum()
}
