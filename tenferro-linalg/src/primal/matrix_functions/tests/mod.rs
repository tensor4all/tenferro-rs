use super::*;
use tenferro_prims::CpuContext;
use tenferro_tensor::MemoryOrder;

fn tensor_data<T: tenferro_algebra::Scalar + Copy>(tensor: &Tensor<T>) -> Vec<T> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.len();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

#[test]
fn matrix_power_covers_batched_identity_special_cases_and_binary_loop() {
    let mut ctx = CpuContext::new(1);

    let batched = Tensor::from_slice(
        &[
            2.0_f64, 0.0, 0.0, 3.0, //
            4.0, 0.0, 0.0, 5.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let identity = matrix_power(&mut ctx, &batched, 0).unwrap();
    assert_eq!(identity.dims(), &[2, 2, 2]);
    assert_eq!(
        tensor_data(&identity),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    );

    let a =
        Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(
        tensor_data(&matrix_power(&mut ctx, &a, 3).unwrap()),
        vec![8.0, 0.0, 0.0, 64.0]
    );
    assert_eq!(
        tensor_data(&matrix_power(&mut ctx, &a, 5).unwrap()),
        vec![32.0, 0.0, 0.0, 1024.0]
    );
    assert_eq!(
        tensor_data(&matrix_power(&mut ctx, &a, -2).unwrap()),
        vec![0.25, 0.0, 0.0, 0.0625]
    );

    let err = matrix_power(&mut ctx, &a, i64::MIN).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("i64::MIN exponent")));
}
