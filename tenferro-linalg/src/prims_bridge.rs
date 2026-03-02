use std::cell::RefCell;

use tenferro_algebra::Standard;
use tenferro_device::{Error, LogicalMemorySpace, Result};
use tenferro_prims::{CpuBackend, CpuContext, PrimDescriptor, TensorPrims};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::LinalgScalar;

thread_local! {
    static PRIMS_CTX: RefCell<CpuContext> = RefCell::new(CpuContext::new(1));
}

pub(crate) fn batched_gemm_via_prims<T>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> Result<Vec<T>>
where
    T: LinalgScalar,
{
    let a_shape = [m, k];
    let b_shape = [k, n];
    let c_shape = [m, n];

    let a_strides = [1isize, m as isize];
    let b_strides = [1isize, k as isize];
    let a_tensor = Tensor::from_vec(a.to_vec(), &a_shape, &a_strides, 0)?;
    let b_tensor = Tensor::from_vec(b.to_vec(), &b_shape, &b_strides, 0)?;
    let mut c_tensor = Tensor::zeros(
        &c_shape,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m,
        n,
        k,
    };

    PRIMS_CTX.with(|ctx_cell| {
        let mut ctx = ctx_cell.borrow_mut();
        let plan = <CpuBackend as TensorPrims<Standard<T>>>::plan(
            &mut ctx,
            &desc,
            &[&a_shape, &b_shape, &c_shape],
        )?;
        <CpuBackend as TensorPrims<Standard<T>>>::execute(
            &mut ctx,
            &plan,
            T::one(),
            &[&a_tensor, &b_tensor],
            T::zero(),
            &mut c_tensor,
        )?;

        c_tensor
            .try_into_data_vec()
            .ok_or_else(|| Error::DeviceError("expected owned CPU output tensor".into()))
    })
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};

    use super::batched_gemm_via_prims;

    #[test]
    fn batched_gemm_via_prims_multiplies_real_col_major_matrices() {
        let a = vec![1.0_f64, 2.0, 3.0, 4.0];
        let b = vec![5.0_f64, 6.0, 7.0, 8.0];

        let c = batched_gemm_via_prims(&a, 2, 2, &b, 2).unwrap();

        assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
    }

    #[test]
    fn batched_gemm_via_prims_multiplies_complex_col_major_matrices() {
        let a = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let b = vec![
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
            Complex64::new(8.0, 0.0),
        ];

        let c = batched_gemm_via_prims(&a, 2, 2, &b, 2).unwrap();

        assert_eq!(
            c,
            vec![
                Complex64::new(23.0, 0.0),
                Complex64::new(34.0, 0.0),
                Complex64::new(31.0, 0.0),
                Complex64::new(46.0, 0.0),
            ]
        );
    }

    #[test]
    fn batched_gemm_via_prims_multiplies_complex32_col_major_matrices() {
        let a = vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ];
        let b = vec![
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
            Complex32::new(7.0, 0.0),
            Complex32::new(8.0, 0.0),
        ];

        let c = batched_gemm_via_prims(&a, 2, 2, &b, 2).unwrap();

        assert_eq!(
            c,
            vec![
                Complex32::new(23.0, 0.0),
                Complex32::new(34.0, 0.0),
                Complex32::new(31.0, 0.0),
                Complex32::new(46.0, 0.0),
            ]
        );
    }
}
