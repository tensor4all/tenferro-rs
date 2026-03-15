use std::ops::Add;

use tenferro_device::{Error, Result};

/// Memory ordering for new allocations.
///
/// Specifies how elements are laid out in memory when creating new tensors
/// or copying data into a contiguous buffer. This is **not** stored on the
/// tensor itself. The tensor's strides fully describe the actual layout.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::MemoryOrder;
///
/// let order = MemoryOrder::RowMajor;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    /// Column-major (Fortran/Julia order). First dimension has stride 1.
    ColumnMajor,
    /// Row-major (C/NumPy order). Last dimension has stride 1.
    RowMajor,
}

pub(crate) fn compute_contiguous_strides(dims: &[usize], order: MemoryOrder) -> Vec<isize> {
    let ndim = dims.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![0isize; ndim];
    match order {
        MemoryOrder::ColumnMajor => {
            strides[0] = 1;
            for i in 1..ndim {
                strides[i] = strides[i - 1] * dims[i - 1] as isize;
            }
        }
        MemoryOrder::RowMajor => {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1] as isize;
            }
        }
    }
    strides
}

pub(crate) fn is_contiguous_in_order(
    dims: &[usize],
    strides: &[isize],
    order: MemoryOrder,
) -> bool {
    if dims.is_empty() || dims.contains(&0) {
        return true;
    }

    let expected = compute_contiguous_strides(dims, order);
    dims.iter()
        .zip(strides)
        .zip(expected)
        .all(|((&dim, &actual), expected)| dim <= 1 || actual == expected)
}

pub(crate) fn validate_layout_against_len(
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    data_len: usize,
) -> Result<()> {
    if strides.len() != dims.len() {
        return Err(Error::InvalidArgument(format!(
            "strides length {} doesn't match dims length {}",
            strides.len(),
            dims.len()
        )));
    }

    if dims.iter().product::<usize>() == 0 {
        return Ok(());
    }

    let mut min_pos = offset;
    let mut max_pos = offset;
    for (axis, (&dim, &stride)) in dims.iter().zip(strides).enumerate() {
        if dim == 0 {
            continue;
        }
        let extent = isize::try_from(dim - 1)
            .ok()
            .and_then(|d| d.checked_mul(stride))
            .ok_or_else(|| {
                Error::StrideError(format!(
                    "extent overflow for dimension {axis} (size={dim}, stride={stride})"
                ))
            })?;
        if extent >= 0 {
            max_pos += extent;
        } else {
            min_pos += extent;
        }
    }
    if min_pos < 0 || max_pos >= data_len as isize {
        return Err(Error::StrideError(format!(
            "layout accesses buffer positions {}..={} but buffer length is {}",
            min_pos, max_pos, data_len
        )));
    }

    Ok(())
}

pub(crate) fn copy_strided<T: Copy>(
    src: &[T],
    dims: &[usize],
    src_strides: &[isize],
    src_offset: isize,
    dst: &mut [T],
    dst_strides: &[isize],
) {
    let n_elements: usize = dims.iter().product();
    if n_elements == 0 {
        return;
    }
    if dims.is_empty() {
        dst[0] = src[src_offset as usize];
        return;
    }

    let mut index = vec![0usize; dims.len()];
    for _ in 0..n_elements {
        let src_pos = src_offset
            + index
                .iter()
                .zip(src_strides)
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        let dst_pos = index
            .iter()
            .zip(dst_strides)
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
        debug_assert!(
            src_pos >= 0 && (src_pos as usize) < src.len(),
            "copy_strided: source position {} out of bounds for buffer length {}",
            src_pos,
            src.len()
        );
        debug_assert!(
            dst_pos >= 0 && (dst_pos as usize) < dst.len(),
            "copy_strided: destination position {} out of bounds for buffer length {}",
            dst_pos,
            dst.len()
        );
        dst[dst_pos as usize] = src[src_pos as usize];

        for axis in 0..dims.len() {
            index[axis] += 1;
            if index[axis] < dims[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

pub(crate) struct StridedInput<'a, T> {
    pub(crate) data: &'a [T],
    pub(crate) strides: &'a [isize],
    pub(crate) offset: isize,
}

pub(crate) fn add_strided<T: Copy + Add<Output = T>>(
    dims: &[usize],
    a: StridedInput<'_, T>,
    b: StridedInput<'_, T>,
    dst: &mut [T],
    dst_strides: &[isize],
) {
    let n_elements: usize = dims.iter().product();
    if n_elements == 0 {
        return;
    }
    if dims.is_empty() {
        dst[0] = a.data[a.offset as usize] + b.data[b.offset as usize];
        return;
    }

    let mut index = vec![0usize; dims.len()];
    for _ in 0..n_elements {
        let a_pos = a.offset
            + index
                .iter()
                .zip(a.strides.iter())
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        let b_pos = b.offset
            + index
                .iter()
                .zip(b.strides.iter())
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>();
        let dst_pos = index
            .iter()
            .zip(dst_strides.iter())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
        debug_assert!(
            a_pos >= 0 && (a_pos as usize) < a.data.len(),
            "add_strided: input a position {} out of bounds for buffer length {}",
            a_pos,
            a.data.len()
        );
        debug_assert!(
            b_pos >= 0 && (b_pos as usize) < b.data.len(),
            "add_strided: input b position {} out of bounds for buffer length {}",
            b_pos,
            b.data.len()
        );
        debug_assert!(
            dst_pos >= 0 && (dst_pos as usize) < dst.len(),
            "add_strided: destination position {} out of bounds for buffer length {}",
            dst_pos,
            dst.len()
        );
        dst[dst_pos as usize] = a.data[a_pos as usize] + b.data[b_pos as usize];

        for axis in 0..dims.len() {
            index[axis] += 1;
            if index[axis] < dims[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}
