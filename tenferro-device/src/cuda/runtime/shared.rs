use super::kernels::*;
use crate::{Error, Result};

/// Destination layout for materialized contiguous buffers.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::ContiguousOrder;
///
/// let order = ContiguousOrder::ColumnMajor;
/// assert_eq!(order, ContiguousOrder::ColumnMajor);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContiguousOrder {
    /// Column-major / Fortran order.
    ColumnMajor,
    /// Row-major / C order.
    RowMajor,
}

/// Which triangular half to keep when materializing a matrix or batched matrix.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::TriangularHalf;
///
/// let half = TriangularHalf::Lower;
/// assert_eq!(half, TriangularHalf::Lower);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularHalf {
    /// Keep the lower triangle.
    Lower,
    /// Keep the upper triangle.
    Upper,
}

impl TriangularHalf {
    pub(super) fn as_i32(self) -> i32 {
        match self {
            TriangularHalf::Lower => 0,
            TriangularHalf::Upper => 1,
        }
    }
}

/// Low-level specification for copying a strided source layout into a destination layout.
///
/// The `dims`, `src_strides`, and `dst_strides` arrays describe the same logical tensor
/// shape. Offsets are measured in elements, not bytes.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::{ContiguousOrder, StridedCopySpec};
///
/// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
/// assert_eq!(spec.dims(), &[4, 2, 3]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StridedCopySpec {
    pub(super) dims: Vec<usize>,
    pub(super) src_strides: Vec<isize>,
    pub(super) src_offset: isize,
    pub(super) dst_strides: Vec<isize>,
    pub(super) dst_offset: isize,
}

impl StridedCopySpec {
    /// Build a strided-copy spec whose destination is contiguous in the requested order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{ContiguousOrder, StridedCopySpec};
    ///
    /// let spec = StridedCopySpec::to_contiguous(&[2, 3], &[1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// assert_eq!(spec.dst_strides(), &[1, 2]);
    /// ```
    pub fn to_contiguous(
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        order: ContiguousOrder,
    ) -> Result<Self> {
        if dims.len() != src_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "strided copy rank mismatch: dims={} src_strides={}",
                dims.len(),
                src_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
            src_offset,
            dst_strides: contiguous_strides(dims, order)?,
            dst_offset: 0,
        })
    }

    /// Returns the logical dimensions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{ContiguousOrder, StridedCopySpec};
    ///
    /// let spec = StridedCopySpec::to_contiguous(&[2, 3], &[1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// assert_eq!(spec.dims(), &[2, 3]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the destination strides in elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{ContiguousOrder, StridedCopySpec};
    ///
    /// let spec = StridedCopySpec::to_contiguous(&[2, 3], &[1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// assert_eq!(spec.dst_strides(), &[1, 2]);
    /// ```
    pub fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }
}

/// Source-side transforms supported by the Layer 0 strided-copy helper.
///
/// Phase 1 supports plain copy and complex conjugation only.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::StridedCopyTransform;
///
/// assert_eq!(StridedCopyTransform::None, StridedCopyTransform::None);
/// assert_eq!(StridedCopyTransform::Conj, StridedCopyTransform::Conj);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StridedCopyTransform {
    None,
    Conj,
}

/// Low-level specification for materializing a triangular matrix view on the GPU.
///
/// The first two dimensions are interpreted as the matrix rows and columns.
/// Any remaining dimensions are treated as batch dimensions and copied
/// elementwise. The output shape matches the input shape.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
///
/// let spec = TriangularPartSpec::new(
///     &[3, 2, 4],
///     &[1, 3, 6],
///     0,
///     &[1, 3, 6],
///     0,
///     -1,
///     TriangularHalf::Lower,
/// ).unwrap();
/// assert_eq!(spec.diagonal(), -1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriangularPartSpec {
    pub(super) dims: Vec<usize>,
    pub(super) src_strides: Vec<isize>,
    pub(super) src_offset: isize,
    pub(super) dst_strides: Vec<isize>,
    pub(super) dst_offset: isize,
    pub(super) diagonal: isize,
    pub(super) half: TriangularHalf,
}

impl TriangularPartSpec {
    /// Build a triangular-copy specification.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(
    ///     &[2, 3],
    ///     &[1, 2],
    ///     0,
    ///     &[1, 2],
    ///     0,
    ///     0,
    ///     TriangularHalf::Upper,
    /// ).unwrap();
    /// assert_eq!(spec.half(), TriangularHalf::Upper);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
        diagonal: isize,
        half: TriangularHalf,
    ) -> Result<Self> {
        if dims.len() < 2 {
            return Err(Error::InvalidArgument(
                "triangular copy requires rank >= 2".into(),
            ));
        }
        if dims.len() != src_strides.len() || dims.len() != dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "triangular copy rank mismatch: dims={} src_strides={} dst_strides={}",
                dims.len(),
                src_strides.len(),
                dst_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
            src_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
            diagonal,
            half,
        })
    }

    /// Returns the triangular diagonal offset.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 2], &[1, 2], 0, &[1, 2], 0, 1, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.diagonal(), 1);
    /// ```
    pub fn diagonal(&self) -> isize {
        self.diagonal
    }

    /// Returns which half is preserved.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 2], &[1, 2], 0, &[1, 2], 0, 0, TriangularHalf::Upper).unwrap();
    /// assert_eq!(spec.half(), TriangularHalf::Upper);
    /// ```
    pub fn half(&self) -> TriangularHalf {
        self.half
    }

    /// Returns the logical dimensions described by this triangular-copy spec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 0, &[1, 2], 0, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.dims(), &[2, 3]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the source strides described by this triangular-copy spec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 0, &[1, 2], 0, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.src_strides(), &[1, 2]);
    /// ```
    pub fn src_strides(&self) -> &[isize] {
        &self.src_strides
    }

    /// Returns the source element offset.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 4, &[1, 2], 0, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.src_offset(), 4);
    /// ```
    pub fn src_offset(&self) -> isize {
        self.src_offset
    }

    /// Returns the destination strides described by this triangular-copy spec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 0, &[1, 2], 0, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.dst_strides(), &[1, 2]);
    /// ```
    pub fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }

    /// Returns the destination element offset.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 0, &[1, 2], 5, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.dst_offset(), 5);
    /// ```
    pub fn dst_offset(&self) -> isize {
        self.dst_offset
    }
}

/// Low-level specification for merging a strict-lower source and an upper-with-diagonal source.
///
/// The logical output shape is `dims`. The first source is read when `row > col`,
/// and the second source is read otherwise. The first two dimensions are
/// interpreted as matrix rows and columns; trailing dimensions are batch dims.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::TriangularMergeSpec;
///
/// let spec = TriangularMergeSpec::new(
///     &[3, 2, 4],
///     &[1, 3, 6],
///     0,
///     &[1, 3, 6],
///     0,
///     &[1, 3, 6],
///     0,
/// ).unwrap();
/// assert_eq!(spec.dims(), &[3, 2, 4]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriangularMergeSpec {
    pub(super) dims: Vec<usize>,
    pub(super) lower_strides: Vec<isize>,
    pub(super) lower_offset: isize,
    pub(super) upper_strides: Vec<isize>,
    pub(super) upper_offset: isize,
    pub(super) dst_strides: Vec<isize>,
    pub(super) dst_offset: isize,
}

impl TriangularMergeSpec {
    /// Build a triangular-merge specification.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::TriangularMergeSpec;
    ///
    /// let spec = TriangularMergeSpec::new(
    ///     &[2, 3],
    ///     &[1, 2],
    ///     0,
    ///     &[1, 2],
    ///     0,
    ///     &[1, 2],
    ///     0,
    /// ).unwrap();
    /// assert_eq!(spec.dims(), &[2, 3]);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: &[usize],
        lower_strides: &[isize],
        lower_offset: isize,
        upper_strides: &[isize],
        upper_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<Self> {
        if dims.len() < 2 {
            return Err(Error::InvalidArgument(
                "triangular merge requires rank >= 2".into(),
            ));
        }
        if dims.len() != lower_strides.len()
            || dims.len() != upper_strides.len()
            || dims.len() != dst_strides.len()
        {
            return Err(Error::InvalidArgument(format!(
                "triangular merge rank mismatch: dims={} lower_strides={} upper_strides={} dst_strides={}",
                dims.len(),
                lower_strides.len(),
                upper_strides.len(),
                dst_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            lower_strides: lower_strides.to_vec(),
            lower_offset,
            upper_strides: upper_strides.to_vec(),
            upper_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
        })
    }

    /// Returns the logical output dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the strict-lower source strides.
    pub fn lower_strides(&self) -> &[isize] {
        &self.lower_strides
    }

    /// Returns the strict-lower source offset.
    pub fn lower_offset(&self) -> isize {
        self.lower_offset
    }

    /// Returns the upper-with-diagonal source strides.
    pub fn upper_strides(&self) -> &[isize] {
        &self.upper_strides
    }

    /// Returns the upper-with-diagonal source offset.
    pub fn upper_offset(&self) -> isize {
        self.upper_offset
    }

    /// Returns the destination strides.
    pub fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }

    /// Returns the destination offset.
    pub fn dst_offset(&self) -> isize {
        self.dst_offset
    }
}

/// Real unary operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RealUnaryOp;
///
/// let op = RealUnaryOp::Abs;
/// assert_eq!(op, RealUnaryOp::Abs);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealUnaryOp {
    Conj,
    Abs,
    Reciprocal,
    Log,
    Sqrt,
}

/// Complex-to-real unary operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::ComplexRealUnaryOp;
///
/// let op = ComplexRealUnaryOp::Abs;
/// assert_eq!(op, ComplexRealUnaryOp::Abs);
/// let op = ComplexRealUnaryOp::Real;
/// assert_eq!(op, ComplexRealUnaryOp::Real);
/// let op = ComplexRealUnaryOp::Imag;
/// assert_eq!(op, ComplexRealUnaryOp::Imag);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexRealUnaryOp {
    Abs,
    Real,
    Imag,
}

/// Real binary operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RealBinaryOp;
///
/// let op = RealBinaryOp::Add;
/// assert_eq!(op, RealBinaryOp::Add);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    Greater,
    GreaterEqual,
    Pow,
}

/// Real ternary operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RealTernaryOp;
///
/// let op = RealTernaryOp::Where;
/// assert_eq!(op, RealTernaryOp::Where);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealTernaryOp {
    Where,
}

/// Real reduction operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RealReductionOp;
///
/// let op = RealReductionOp::Sum;
/// assert_eq!(op, RealReductionOp::Sum);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealReductionOp {
    Sum,
    Max,
    Min,
    Prod,
}

/// Low-level specification for zero-filling trailing regions by batch-local keep counts.
///
/// The trailing batch dims are `dims[structural_rank..]`. `axis` is interpreted
/// within the structural prefix `[0, structural_rank)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec;
///
/// let spec = ZeroTrailingByCountsSpec::new(
///     &[2, 2, 2],
///     &[1, 2, 4],
///     0,
///     &[1, 2, 4],
///     0,
///     &[1],
///     0,
///     1,
///     2,
/// ).unwrap();
/// assert_eq!(spec.axis(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroTrailingByCountsSpec {
    pub(super) dims: Vec<usize>,
    pub(super) src_strides: Vec<isize>,
    pub(super) src_offset: isize,
    pub(super) dst_strides: Vec<isize>,
    pub(super) dst_offset: isize,
    pub(super) keep_count_strides: Vec<isize>,
    pub(super) keep_count_offset: isize,
    pub(super) axis: usize,
    pub(super) structural_rank: usize,
}

impl ZeroTrailingByCountsSpec {
    /// Build a zero-trailing specification.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec;
    ///
    /// let spec = ZeroTrailingByCountsSpec::new(
    ///     &[3, 2, 2],
    ///     &[1, 3, 6],
    ///     0,
    ///     &[1, 3, 6],
    ///     0,
    ///     &[1],
    ///     0,
    ///     0,
    ///     2,
    /// ).unwrap();
    /// assert_eq!(spec.structural_rank(), 2);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
        keep_count_strides: &[isize],
        keep_count_offset: isize,
        axis: usize,
        structural_rank: usize,
    ) -> Result<Self> {
        if dims.len() != src_strides.len() || dims.len() != dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "zero-trailing rank mismatch: dims={} src_strides={} dst_strides={}",
                dims.len(),
                src_strides.len(),
                dst_strides.len()
            )));
        }
        if structural_rank == 0 || structural_rank > dims.len() {
            return Err(Error::InvalidArgument(format!(
                "structural_rank {structural_rank} must be in 1..={}",
                dims.len()
            )));
        }
        if axis >= structural_rank {
            return Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for structural_rank {structural_rank}"
            )));
        }
        let batch_rank = dims.len() - structural_rank;
        if keep_count_strides.len() != batch_rank {
            return Err(Error::InvalidArgument(format!(
                "keep_count_strides rank {} does not match batch rank {}",
                keep_count_strides.len(),
                batch_rank
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
            src_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
            keep_count_strides: keep_count_strides.to_vec(),
            keep_count_offset,
            axis,
            structural_rank,
        })
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[cfg(test)]
    pub(crate) fn src_strides(&self) -> &[isize] {
        &self.src_strides
    }

    #[cfg(test)]
    pub(crate) fn src_offset(&self) -> isize {
        self.src_offset
    }

    #[cfg(test)]
    pub(crate) fn keep_count_strides(&self) -> &[isize] {
        &self.keep_count_strides
    }

    #[cfg(test)]
    pub(crate) fn keep_count_offset(&self) -> isize {
        self.keep_count_offset
    }

    #[cfg(test)]
    pub(crate) fn axis(&self) -> usize {
        self.axis
    }

    #[cfg(test)]
    pub(crate) fn structural_rank(&self) -> usize {
        self.structural_rank
    }
}

pub(super) trait RuntimeRealScalar: cudarc::driver::DeviceRepr + Copy + 'static {
    const UNARY_KERNEL_NAME: &'static str;
    const BINARY_KERNEL_NAME: &'static str;
    const TERNARY_KERNEL_NAME: &'static str;
    const REDUCTION_KERNEL_NAME: &'static str;
}

/// Marker trait for keep-count scalars supported by CUDA trailing zero-fill.
///
/// Implemented for `f32` and `f64`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RuntimeKeepCountScalar;
///
/// fn needs_counts<T: RuntimeKeepCountScalar>() {}
/// needs_counts::<f32>();
/// needs_counts::<f64>();
/// ```
pub trait RuntimeKeepCountScalar: cudarc::driver::DeviceRepr + Copy + 'static {
    const VALIDATE_KERNEL_NAME: &'static str;
    const ZERO_TRAILING_KERNEL_NAME: &'static str;
}

impl RuntimeRealScalar for f32 {
    const UNARY_KERNEL_NAME: &'static str = REAL_UNARY_KERNEL_NAME_F32;
    const BINARY_KERNEL_NAME: &'static str = REAL_BINARY_KERNEL_NAME_F32;
    const TERNARY_KERNEL_NAME: &'static str = REAL_TERNARY_KERNEL_NAME_F32;
    const REDUCTION_KERNEL_NAME: &'static str = REAL_REDUCTION_KERNEL_NAME_F32;
}

impl RuntimeRealScalar for f64 {
    const UNARY_KERNEL_NAME: &'static str = REAL_UNARY_KERNEL_NAME_F64;
    const BINARY_KERNEL_NAME: &'static str = REAL_BINARY_KERNEL_NAME_F64;
    const TERNARY_KERNEL_NAME: &'static str = REAL_TERNARY_KERNEL_NAME_F64;
    const REDUCTION_KERNEL_NAME: &'static str = REAL_REDUCTION_KERNEL_NAME_F64;
}

impl RuntimeKeepCountScalar for f32 {
    const VALIDATE_KERNEL_NAME: &'static str = ZERO_TRAILING_VALIDATE_KERNEL_NAME_F32;
    const ZERO_TRAILING_KERNEL_NAME: &'static str = ZERO_TRAILING_KERNEL_NAME_F32;
}

impl RuntimeKeepCountScalar for f64 {
    const VALIDATE_KERNEL_NAME: &'static str = ZERO_TRAILING_VALIDATE_KERNEL_NAME_F64;
    const ZERO_TRAILING_KERNEL_NAME: &'static str = ZERO_TRAILING_KERNEL_NAME_F64;
}
