use super::*;

#[test]
fn softmax_tensor_covers_the_f32_and_f64_paths() {
    let mut pool = BufferPool::new();
    let f32_tensor =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
    let output = softmax_read_with_pool(
        FanOut::SERIAL,
        &mut pool,
        TensorRead::from_tensor(&f32_tensor),
        0,
    )
    .unwrap()
    .unwrap();
    for column in output.as_slice::<f32>().unwrap().chunks(3) {
        let sum: f32 = column.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    let f64_tensor =
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0f64, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
    let output = softmax_read_with_pool(
        FanOut::SERIAL,
        &mut pool,
        TensorRead::from_tensor(&f64_tensor),
        0,
    )
    .unwrap()
    .unwrap();
    for column in output.as_slice::<f64>().unwrap().chunks(3) {
        let sum: f64 = column.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn scalar_group_functions_match_the_reference() {
    let src = [1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0];
    let mut shifted = [0.0f32; 6];
    shift_group_f32(&src, &mut shifted, 3);
    assert_eq!(shifted, [-2.0, -1.0, 0.0, -2.0, -1.0, 0.0]);
    for value in shifted.iter_mut() {
        *value = value.exp();
    }
    scale_group_f32(&mut shifted, 3);
    for column in shifted.chunks(3) {
        let sum: f32 = column.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    let src = [1.0f64, 2.0, 3.0, -1.0, 0.0, 1.0];
    let mut shifted = [0.0f64; 6];
    shift_group_f64(&src, &mut shifted, 3);
    assert_eq!(shifted, [-2.0, -1.0, 0.0, -2.0, -1.0, 0.0]);
    for value in shifted.iter_mut() {
        *value = value.exp();
    }
    scale_group_f64(&mut shifted, 3);
    for column in shifted.chunks(3) {
        let sum: f64 = column.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_group_functions_match_the_scalar_reference() {
    if !std::is_x86_feature_detected!("avx2") {
        return;
    }
    let src = [1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0];
    let mut scalar = [0.0f32; 6];
    shift_group_f32(&src, &mut scalar, 3);
    let mut vector = [0.0f32; 6];
    // SAFETY: guarded by the runtime AVX2 check above.
    unsafe { shift_group_f32_avx2(&src, &mut vector, 3) };
    assert_eq!(scalar, vector);

    let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut b = a;
    // SAFETY: guarded by the runtime AVX2 check above.
    unsafe { scale_group_f32_avx2(&mut a, 3) };
    scale_group_f32(&mut b, 3);
    assert_eq!(a, b);
}
