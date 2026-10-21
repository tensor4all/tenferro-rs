//! Borrowed operand layout shared by the CPU tensor kernels.
//!
//! A kernel reads an operand as `(data, shape, strides, offset)` rather than as
//! a compact owned tensor, so an owned tensor and a borrowed view reach the same
//! kernel without a staging copy (#2214). Owned tensors are compact, so their
//! descriptor is their buffer with column-major strides and offset zero.

use tenferro_tensor::TypedTensorView;

/// Borrowed operand: element buffer plus the layout its strides describe.
pub(crate) struct Operand<'a, T> {
    /// Backing storage; `offset` and `strides` index into it.
    pub(crate) data: &'a [T],
    /// Logical dimensions.
    pub(crate) shape: &'a [usize],
    /// Element strides, in `data` element units.
    pub(crate) strides: &'a [isize],
    /// Element offset of the first logical element.
    pub(crate) offset: isize,
}

impl<'a, T: tenferro_tensor::TensorScalar + Copy + 'static> Operand<'a, T> {
    /// Descriptor of a compact owned tensor, borrowing its cached column-major strides.
    pub(crate) fn from_typed(
        op: &'static str,
        tensor: &'a tenferro_tensor::TypedTensor<T>,
    ) -> crate::Result<Self> {
        Ok(Self {
            data: crate::typed_host_data(op, tensor)?,
            shape: tensor.shape(),
            strides: tensor.strides(),
            offset: 0,
        })
    }
}

/// Descriptor of a borrowed view, keeping its strides and offset.
pub(crate) fn operand_from_view<'a, T, R>(
    view: &'a TypedTensorView<'a, T, R>,
) -> crate::Result<Operand<'a, T>>
where
    T: Copy + 'static,
    R: tenferro_tensor::TensorRank,
{
    Ok(Operand {
        data: view.host_storage()?,
        shape: view.shape(),
        strides: view.strides(),
        offset: view.offset(),
    })
}
