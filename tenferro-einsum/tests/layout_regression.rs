use tenferro_algebra::Standard;
use tenferro_einsum::{einsum_with_subscripts, Subscripts};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

fn tensor_elem_f64(t: &Tensor<f64>, idx: &[usize]) -> f64 {
    assert_eq!(idx.len(), t.dims().len());
    let mut off = t.offset();
    for (axis, &i) in idx.iter().enumerate() {
        off += (i as isize) * t.strides()[axis];
    }
    let pos = usize::try_from(off).expect("offset must be non-negative");
    t.buffer().as_slice().expect("host buffer")[pos]
}

#[test]
fn row_major_binary_rank3_rank2_matches_manual() {
    let mut ctx = CpuContext::new(1);

    let a_data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect(); // [2,3,4]
    let b_data: Vec<f64> = (0..20).map(|x| x as f64 - 3.0).collect(); // [4,5]

    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3, 4], MemoryOrder::RowMajor).unwrap();
    let b = Tensor::<f64>::from_slice(&b_data, &[4, 5], MemoryOrder::RowMajor).unwrap();

    let subs = Subscripts::new(&[&[0, 1, 2], &[2, 3]], &[0, 1, 3]);
    let out = einsum_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &[&a, &b], None)
        .unwrap();

    assert_eq!(out.dims(), &[2, 3, 5]);
    for i in 0..2 {
        for j in 0..3 {
            for l in 0..5 {
                let mut expected = 0.0_f64;
                for k in 0..4 {
                    expected += tensor_elem_f64(&a, &[i, j, k]) * tensor_elem_f64(&b, &[k, l]);
                }
                let actual = tensor_elem_f64(&out, &[i, j, l]);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "mismatch at ({i},{j},{l}): actual={actual}, expected={expected}"
                );
            }
        }
    }
}

#[test]
fn preconverted_column_major_binary_rank3_rank2_matches_manual() {
    let mut ctx = CpuContext::new(1);

    let a_data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect(); // [2,3,4]
    let b_data: Vec<f64> = (0..20).map(|x| x as f64 - 3.0).collect(); // [4,5]

    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3, 4], MemoryOrder::RowMajor)
        .unwrap()
        .into_contiguous(MemoryOrder::ColumnMajor);
    let b = Tensor::<f64>::from_slice(&b_data, &[4, 5], MemoryOrder::RowMajor)
        .unwrap()
        .into_contiguous(MemoryOrder::ColumnMajor);

    let subs = Subscripts::new(&[&[0, 1, 2], &[2, 3]], &[0, 1, 3]);
    let out = einsum_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &[&a, &b], None)
        .unwrap();

    assert_eq!(out.dims(), &[2, 3, 5]);
    for i in 0..2 {
        for j in 0..3 {
            for l in 0..5 {
                let mut expected = 0.0_f64;
                for k in 0..4 {
                    expected += tensor_elem_f64(&a, &[i, j, k]) * tensor_elem_f64(&b, &[k, l]);
                }
                let actual = tensor_elem_f64(&out, &[i, j, l]);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "mismatch at ({i},{j},{l}): actual={actual}, expected={expected}"
                );
            }
        }
    }
}

#[test]
fn row_major_three_input_matches_manual() {
    let mut ctx = CpuContext::new(1);

    let a_data: Vec<f64> = (0..24).map(|x| x as f64 + 1.0).collect(); // [2,3,4]
    let b_data: Vec<f64> = (0..20).map(|x| x as f64 + 0.25).collect(); // [4,5]
    let c_data: Vec<f64> = (0..10).map(|x| x as f64 - 1.5).collect(); // [5,2]

    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3, 4], MemoryOrder::RowMajor).unwrap();
    let b = Tensor::<f64>::from_slice(&b_data, &[4, 5], MemoryOrder::RowMajor).unwrap();
    let c = Tensor::<f64>::from_slice(&c_data, &[5, 2], MemoryOrder::RowMajor).unwrap();

    let subs = Subscripts::new(&[&[0, 1, 2], &[2, 3], &[3, 4]], &[0, 1, 4]);
    let out =
        einsum_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &[&a, &b, &c], None)
            .unwrap();

    assert_eq!(out.dims(), &[2, 3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            for m in 0..2 {
                let mut expected = 0.0_f64;
                for k in 0..4 {
                    for l in 0..5 {
                        expected += tensor_elem_f64(&a, &[i, j, k])
                            * tensor_elem_f64(&b, &[k, l])
                            * tensor_elem_f64(&c, &[l, m]);
                    }
                }
                let actual = tensor_elem_f64(&out, &[i, j, m]);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "mismatch at ({i},{j},{m}): actual={actual}, expected={expected}"
                );
            }
        }
    }
}
