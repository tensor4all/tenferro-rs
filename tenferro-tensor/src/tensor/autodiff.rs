use tenferro_algebra::Scalar;
#[cfg(feature = "cuda")]
use tenferro_device::LogicalMemorySpace;

use super::Tensor;
use crate::layout::{add_strided, compute_contiguous_strides, StridedInput};
use crate::MemoryOrder;

impl<T: Scalar> chainrules_core::Differentiable for Tensor<T> {
    type Tangent = Tensor<T>;

    fn zero_tangent(&self) -> Tensor<T> {
        self.zeros_like()
            .unwrap_or_else(|err| panic!("zero_tangent: failed to create zero tensor: {err}"))
    }

    fn num_elements(&self) -> usize {
        self.len()
    }

    fn seed_cotangent(&self) -> Tensor<T> {
        self.ones_like()
            .unwrap_or_else(|err| panic!("seed_cotangent: failed to create ones tensor: {err}"))
    }

    fn accumulate_tangent(a: Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
        assert_eq!(
            a.dims, b.dims,
            "tangent shape mismatch in accumulate_tangent"
        );

        let a_fw = a.fw_grad().cloned();
        let b_fw = b.fw_grad().cloned();

        #[cfg(feature = "cuda")]
        let result_primal =
            if matches!(a.logical_memory_space, LogicalMemorySpace::GpuMemory { .. }) {
                crate::cuda_runtime::add_strided_tensor(&a, b)
                    .unwrap_or_else(|err| panic!("accumulate_tangent: GPU addition failed: {err}"))
            } else {
                accumulate_tangent_cpu(&a, b)
            };

        #[cfg(not(feature = "cuda"))]
        let result_primal = accumulate_tangent_cpu(&a, b);

        let fw_grad = match (a_fw, b_fw) {
            (Some(fa), Some(fb)) => Some(Self::accumulate_tangent(fa, &fb)),
            (Some(fa), None) => Some(fa),
            (None, Some(fb)) => Some(fb.clone()),
            (None, None) => None,
        };

        let mut result = result_primal;
        if let Some(fw_grad) = fw_grad {
            result.set_fw_grad(fw_grad);
        }
        result
    }
}

/// CPU path for element-wise tangent accumulation.
fn accumulate_tangent_cpu<T: Scalar>(a: &Tensor<T>, b: &Tensor<T>) -> Tensor<T> {
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
    Tensor::from_owned_contiguous_data(
        data,
        a.dims.clone(),
        MemoryOrder::ColumnMajor,
        a.logical_memory_space,
        a.preferred_compute_device,
        false,
    )
}
