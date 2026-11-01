#![allow(dead_code)]

#[cfg(test)]
mod tests;

use std::mem;

use crate::{Error, Result};

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
