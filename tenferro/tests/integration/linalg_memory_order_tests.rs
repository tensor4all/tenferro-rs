use tenferro::{set_default_runtime, RuntimeContext, Tensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn matmul(lhs: &DenseTensor<f64>, rhs: &DenseTensor<f64>) -> DenseTensor<f64> {
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
    DenseTensor::from_slice(&out, &[3, 4], MemoryOrder::RowMajor).unwrap()
}

fn max_abs_diff(lhs: &DenseTensor<f64>, rhs: &DenseTensor<f64>) -> f64 {
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
    let input = DenseTensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        MemoryOrder::RowMajor,
    )
    .unwrap();
    let input_expected = input.contiguous(MemoryOrder::RowMajor);
    let out = Tensor::from_tensor(input).qr().unwrap();
    let reconstructed = matmul(
        out.q.as_f64().unwrap().primal(),
        out.r.as_f64().unwrap().primal(),
    );

    assert!(
        max_abs_diff(&reconstructed, &input_expected) < 1e-10,
        "row-major QR should reconstruct the original logical matrix"
    );
}

#[test]
fn eager_svd_accepts_row_major_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let row_major = DenseTensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        MemoryOrder::RowMajor,
    )
    .unwrap();
    let column_major = DenseTensor::<f64>::from_slice(
        &[
            1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
        ],
        &[3, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let row_out = Tensor::from_tensor(row_major).svd().unwrap();
    let col_out = Tensor::from_tensor(column_major).svd().unwrap();

    assert!(
        max_abs_diff(
            row_out.s.as_f64().unwrap().primal(),
            col_out.s.as_f64().unwrap().primal(),
        ) < 1e-10,
        "row-major SVD should match column-major singular values"
    );
}
