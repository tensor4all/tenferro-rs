use std::fmt;

use tenferro_algebra::Scalar;
use tenferro_device::LogicalMemorySpace;
use tenferro_internal_ad_core::DynAdTensorRef;
use tenferro_tensor::Tensor as DenseTensor;

use super::Tensor;
use crate::structured::StructuredTensor;

const MAX_PREVIEW_LOGICAL_ELEMENTS: usize = 16;

struct RawDebug(String);

impl fmt::Debug for RawDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let preview = match self.as_dyn_ad_ref() {
            DynAdTensorRef::F32(value) => preview_field(value.structured_primal()),
            DynAdTensorRef::F64(value) => preview_field(value.structured_primal()),
            DynAdTensorRef::C32(value) => preview_field(value.structured_primal()),
            DynAdTensorRef::C64(value) => preview_field(value.structured_primal()),
        };

        f.debug_struct("Tensor")
            .field("scalar_type", &self.scalar_type())
            .field("dims", &self.dims())
            .field("axis_classes", &self.axis_classes())
            .field("mode", &self.mode())
            .field("is_dense", &self.is_dense())
            .field("is_diag", &self.is_diag())
            .field("preview", &preview)
            .finish()
    }
}

fn preview_field<T>(structured: &StructuredTensor<T>) -> RawDebug
where
    T: Scalar + fmt::Debug,
{
    let logical_numel = structured.logical_dims().iter().product::<usize>();
    if structured.payload().logical_memory_space() != LogicalMemorySpace::MainMemory {
        return RawDebug("<omitted: non-main-memory tensor>".to_string());
    }
    if logical_numel > MAX_PREVIEW_LOGICAL_ELEMENTS {
        return RawDebug(format!("<omitted: {logical_numel} logical values>"));
    }

    match structured.to_dense() {
        Ok(dense) => match format_dense_values(&dense) {
            Some(rendered) => RawDebug(rendered),
            None => RawDebug("<unavailable: preview formatting failed>".to_string()),
        },
        Err(err) => RawDebug(format!("<unavailable: {err}>")),
    }
}

fn format_dense_values<T>(tensor: &DenseTensor<T>) -> Option<String>
where
    T: fmt::Debug,
{
    let mut index = vec![0usize; tensor.dims().len()];
    format_dense_values_axis(tensor, tensor.dims(), 0, &mut index)
}

fn format_dense_values_axis<T>(
    tensor: &DenseTensor<T>,
    dims: &[usize],
    axis: usize,
    index: &mut [usize],
) -> Option<String>
where
    T: fmt::Debug,
{
    if axis == dims.len() {
        return tensor.get(index).map(|value| format!("{value:?}"));
    }

    if dims[axis] == 0 {
        return Some("[]".to_string());
    }

    let mut rendered = String::from("[");
    for i in 0..dims[axis] {
        index[axis] = i;
        if i > 0 {
            rendered.push_str(", ");
        }
        rendered.push_str(&format_dense_values_axis(tensor, dims, axis + 1, index)?);
    }
    rendered.push(']');
    Some(rendered)
}
