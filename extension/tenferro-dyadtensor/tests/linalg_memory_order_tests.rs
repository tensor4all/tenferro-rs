use tenferro_dyadtensor::{ad, set_default_runtime, AdTensor, RuntimeContext};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

fn matmul(lhs: &Tensor<f64>, rhs: &Tensor<f64>) -> Tensor<f64> {
    assert_eq!(lhs.dims(), &[3, 3]);
    assert_eq!(rhs.dims(), &[3, 4]);
    let lhs_rm = lhs.contiguous(MemoryOrder::RowMajor);
    let rhs_rm = rhs.contiguous(MemoryOrder::RowMajor);
    let lhs_data = lhs_rm.buffer().as_slice().unwrap();
    let rhs_data = rhs_rm.buffer().as_slice().unwrap();
    let mut out = vec![0.0; 12];
    for i in 0..3 {
        for j in 0..4 {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += lhs_data[i * 3 + k] * rhs_data[k * 4 + j];
            }
            out[i * 4 + j] = acc;
        }
    }
    Tensor::from_slice(&out, &[3, 4], MemoryOrder::RowMajor).unwrap()
}

fn max_abs_diff(lhs: &Tensor<f64>, rhs: &Tensor<f64>) -> f64 {
    assert_eq!(lhs.dims(), rhs.dims());
    let lhs_rm = lhs.contiguous(MemoryOrder::RowMajor);
    let rhs_rm = rhs.contiguous(MemoryOrder::RowMajor);
    lhs_rm
        .buffer()
        .as_slice()
        .unwrap()
        .iter()
        .zip(rhs_rm.buffer().as_slice().unwrap().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn eager_qr_accepts_row_major_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let input = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        MemoryOrder::RowMajor,
    )
    .unwrap();
    let input_expected = input.contiguous(MemoryOrder::RowMajor);
    let out = ad::qr(&AdTensor::new_primal(input)).unwrap();
    let reconstructed = matmul(out.q.primal(), out.r.primal());

    assert!(
        max_abs_diff(&reconstructed, &input_expected) < 1e-10,
        "row-major QR should reconstruct the original logical matrix"
    );
}

#[test]
fn eager_svd_accepts_row_major_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let row_major = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        MemoryOrder::RowMajor,
    )
    .unwrap();
    let column_major = Tensor::<f64>::from_slice(
        &[
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
        &[3, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let row_out = ad::svd(&AdTensor::new_primal(row_major)).unwrap();
    let col_out = ad::svd(&AdTensor::new_primal(column_major)).unwrap();

    assert!(
        max_abs_diff(row_out.s.primal(), col_out.s.primal()) < 1e-10,
        "row-major SVD should match column-major singular values"
    );
}
