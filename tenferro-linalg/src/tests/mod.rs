use super::*;
use tenferro_prims::CpuContext;

fn tensor_data(tensor: &Tensor<f64>) -> Vec<f64> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

#[test]
fn vector_norm_paths_are_covered_in_crate_unit_tests() {
    let mut ctx = CpuContext::new(1);
    let x = Tensor::from_slice(&[3.0_f64, -4.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let dx = Tensor::from_slice(&[0.25_f64, -0.5], &[2], MemoryOrder::ColumnMajor).unwrap();
    let cotangent = Tensor::from_vec(vec![2.0_f64], &[], &[], 0).unwrap();

    let lp = norm(&mut ctx, &x, NormKind::Lp(2.0)).unwrap();
    assert!((tensor_data(&lp)[0] - 5.0).abs() < 1e-12);

    let grad = norm_rrule(&mut ctx, &x, &cotangent, NormKind::Fro).unwrap();
    let grad_data = tensor_data(&grad);
    assert!((grad_data[0] - 1.2).abs() < 1e-12);
    assert!((grad_data[1] + 1.6).abs() < 1e-12);

    let (nrm, dnrm) = norm_frule(&mut ctx, &x, &dx, NormKind::Lp(2.0)).unwrap();
    assert!((tensor_data(&nrm)[0] - 5.0).abs() < 1e-12);
    assert!((tensor_data(&dnrm)[0] - 0.55).abs() < 1e-12);
}
