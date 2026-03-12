use tenferro_algebra::Scalar;

use super::Tensor;
use crate::layout::{add_strided, compute_contiguous_strides, StridedInput};
use crate::MemoryOrder;

impl<T: Scalar> chainrules_core::Differentiable for Tensor<T> {
    type Tangent = Tensor<T>;

    fn zero_tangent(&self) -> Tensor<T> {
        Tensor::zeros(
            &self.dims,
            self.logical_memory_space,
            MemoryOrder::ColumnMajor,
        )
    }

    fn num_elements(&self) -> usize {
        self.len()
    }

    fn seed_cotangent(&self) -> Tensor<T> {
        Tensor::ones(
            &self.dims,
            self.logical_memory_space,
            MemoryOrder::ColumnMajor,
        )
    }

    fn accumulate_tangent(a: Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        assert_eq!(
            a.dims, b.dims,
            "tangent shape mismatch in accumulate_tangent"
        );

        let a_fw = a.fw_grad().cloned();
        let b_fw = b.fw_grad().cloned();

        let dst_strides = compute_contiguous_strides(&a.dims, MemoryOrder::ColumnMajor);
        let mut data = vec![T::zero(); a.len()];
        if !data.is_empty() {
            add_strided(
                &a.dims,
                StridedInput {
                    data: a.cpu_backed_slice_or_panic("accumulate_tangent"),
                    strides: &a.strides,
                    offset: a.offset,
                },
                StridedInput {
                    data: b.cpu_backed_slice_or_panic("accumulate_tangent"),
                    strides: &b.strides,
                    offset: b.offset,
                },
                &mut data,
                &dst_strides,
            );
        }

        let fw_grad = match (a_fw, b_fw) {
            (Some(fa), Some(fb)) => Some(Self::accumulate_tangent(fa, &fb)),
            (Some(fa), None) => Some(fa),
            (None, Some(fb)) => Some(fb.clone()),
            (None, None) => None,
        };

        let mut result = Tensor::from_owned_contiguous_data(
            data,
            a.dims.clone(),
            MemoryOrder::ColumnMajor,
            a.logical_memory_space,
            a.preferred_compute_device,
            false,
        );
        if let Some(fw_grad) = fw_grad {
            result.set_fw_grad(fw_grad);
        }
        result
    }
}
