use super::super::state::CudaBuffer;
use super::validate_axes_list;
use crate::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataDType {
    I32,
    Bool,
}

#[derive(Debug)]
pub(crate) enum MetadataTensorRef<'a> {
    I32(&'a CudaBuffer<i32>),
    Bool(&'a CudaBuffer<u8>),
}

impl<'a> MetadataTensorRef<'a> {
    pub(crate) const fn dtype(&self) -> MetadataDType {
        match self {
            Self::I32(_) => MetadataDType::I32,
            Self::Bool(_) => MetadataDType::Bool,
        }
    }
}

#[derive(Debug)]
pub(crate) enum MetadataTensorMut<'a> {
    I32(&'a mut CudaBuffer<i32>),
    Bool(&'a mut CudaBuffer<u8>),
}

impl<'a> MetadataTensorMut<'a> {
    pub(crate) const fn dtype(&self) -> MetadataDType {
        match self {
            Self::I32(_) => MetadataDType::I32,
            Self::Bool(_) => MetadataDType::Bool,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataConstantValue {
    I32(i32),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataGenerateOp {
    IotaStartZero,
    Constant(MetadataConstantValue),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataBinaryOp {
    Equal,
    NotEqual,
    Add,
    Sub,
    Mul,
    BitAnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataTernaryOp {
    Where,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataReductionOp {
    Sum,
    All,
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetadataGenerateSpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) dst_strides: Vec<isize>,
    pub(crate) dst_offset: isize,
}

impl MetadataGenerateSpec {
    pub(crate) fn new(dims: &[usize], dst_strides: &[isize], dst_offset: isize) -> Result<Self> {
        if dims.len() != dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "metadata generate rank mismatch: dims={} dst_strides={}",
                dims.len(),
                dst_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            dst_strides: dst_strides.to_vec(),
            dst_offset,
        })
    }

    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub(crate) fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }

    pub(crate) fn dst_offset(&self) -> isize {
        self.dst_offset
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetadataBinarySpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) lhs_strides: Vec<isize>,
    pub(crate) lhs_offset: isize,
    pub(crate) rhs_strides: Vec<isize>,
    pub(crate) rhs_offset: isize,
    pub(crate) dst_strides: Vec<isize>,
    pub(crate) dst_offset: isize,
}

impl MetadataBinarySpec {
    pub(crate) fn new(
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<Self> {
        if dims.len() != lhs_strides.len()
            || dims.len() != rhs_strides.len()
            || dims.len() != dst_strides.len()
        {
            return Err(Error::InvalidArgument(format!(
                "metadata binary rank mismatch: dims={} lhs={} rhs={} dst={}",
                dims.len(),
                lhs_strides.len(),
                rhs_strides.len(),
                dst_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            lhs_strides: lhs_strides.to_vec(),
            lhs_offset,
            rhs_strides: rhs_strides.to_vec(),
            rhs_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetadataTernarySpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) cond_strides: Vec<isize>,
    pub(crate) cond_offset: isize,
    pub(crate) true_strides: Vec<isize>,
    pub(crate) true_offset: isize,
    pub(crate) false_strides: Vec<isize>,
    pub(crate) false_offset: isize,
    pub(crate) dst_strides: Vec<isize>,
    pub(crate) dst_offset: isize,
}

impl MetadataTernarySpec {
    pub(crate) fn new(
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        true_strides: &[isize],
        true_offset: isize,
        false_strides: &[isize],
        false_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<Self> {
        if dims.len() != cond_strides.len()
            || dims.len() != true_strides.len()
            || dims.len() != false_strides.len()
            || dims.len() != dst_strides.len()
        {
            return Err(Error::InvalidArgument(format!(
                "metadata ternary rank mismatch: dims={} cond={} true={} false={} dst={}",
                dims.len(),
                cond_strides.len(),
                true_strides.len(),
                false_strides.len(),
                dst_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            cond_strides: cond_strides.to_vec(),
            cond_offset,
            true_strides: true_strides.to_vec(),
            true_offset,
            false_strides: false_strides.to_vec(),
            false_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetadataCastSpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) input_strides: Vec<isize>,
    pub(crate) input_offset: isize,
    pub(crate) dst_strides: Vec<isize>,
    pub(crate) dst_offset: isize,
}

impl MetadataCastSpec {
    pub(crate) fn new(
        dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<Self> {
        if dims.len() != input_strides.len() || dims.len() != dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "metadata cast rank mismatch: dims={} input={} dst={}",
                dims.len(),
                input_strides.len(),
                dst_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            input_strides: input_strides.to_vec(),
            input_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetadataReductionSpec {
    pub(crate) input_dims: Vec<usize>,
    pub(crate) input_strides: Vec<isize>,
    pub(crate) input_offset: isize,
    pub(crate) output_dims: Vec<usize>,
    pub(crate) output_strides: Vec<isize>,
    pub(crate) output_offset: isize,
    pub(crate) kept_axes: Vec<usize>,
    pub(crate) reduced_axes: Vec<usize>,
}

impl MetadataReductionSpec {
    pub(crate) fn new(
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<Self> {
        if input_dims.len() != input_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "metadata reduction input rank mismatch: dims={} input_strides={}",
                input_dims.len(),
                input_strides.len()
            )));
        }
        if output_dims.len() != output_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "metadata reduction output rank mismatch: dims={} output_strides={}",
                output_dims.len(),
                output_strides.len()
            )));
        }
        if output_dims.len() != kept_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "metadata reduction kept-axis mismatch: output dims={} kept_axes={}",
                output_dims.len(),
                kept_axes.len()
            )));
        }

        let kept_seen = validate_axes_list(kept_axes, input_dims.len(), "metadata reduction kept")?;
        let reduced_seen =
            validate_axes_list(reduced_axes, input_dims.len(), "metadata reduction reduced")?;
        for axis in 0..input_dims.len() {
            match (kept_seen[axis], reduced_seen[axis]) {
                (true, true) => {
                    return Err(Error::InvalidArgument(format!(
                        "metadata reduction axis {axis} appears in both kept_axes and reduced_axes"
                    )));
                }
                (false, false) => {
                    return Err(Error::InvalidArgument(format!(
                        "metadata reduction axis {axis} is missing from kept_axes and reduced_axes"
                    )));
                }
                _ => {}
            }
        }

        for (output_axis, &input_axis) in kept_axes.iter().enumerate() {
            let Some(&expected_dim) = input_dims.get(input_axis) else {
                return Err(Error::InvalidArgument(format!(
                    "metadata reduction kept axis {input_axis} out of bounds"
                )));
            };
            if output_dims[output_axis] != expected_dim {
                return Err(Error::InvalidArgument(format!(
                    "metadata reduction output dim mismatch at axis {output_axis}: expected {expected_dim}, got {}",
                    output_dims[output_axis]
                )));
            }
        }

        Ok(Self {
            input_dims: input_dims.to_vec(),
            input_strides: input_strides.to_vec(),
            input_offset,
            output_dims: output_dims.to_vec(),
            output_strides: output_strides.to_vec(),
            output_offset,
            kept_axes: kept_axes.to_vec(),
            reduced_axes: reduced_axes.to_vec(),
        })
    }
}
