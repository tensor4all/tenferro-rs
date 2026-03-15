use num_complex::Complex32;
use tenferro::{forward_ad, Tensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn matrix2(values: &[f64; 4]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

fn vector(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_c32(values: &[Complex32]) -> DenseTensor<Complex32> {
    DenseTensor::<Complex32>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn diag_tensor(values: &[f64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector(values))).unwrap()
}

fn as_slice(tensor: &DenseTensor<f64>) -> &[f64] {
    tensor
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

#[test]
fn dyn_ad_tensor_reshape_preserves_forward_mode() {
    let (reshaped, tangent) = forward_ad::dual_level(|fw| {
        let x = fw.make_dual(
            &Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0])),
            &Tensor::from_tensor(matrix2(&[0.5, -0.25, 0.75, 1.0])),
        )?;
        let reshaped = x.reshape(&[4])?;
        fw.unpack_dual(&reshaped)
    })
    .unwrap();

    let reshaped = reshaped.as_f64().unwrap();
    assert_eq!(reshaped.dims(), &[4]);
    assert_eq!(as_slice(reshaped.primal()), &[1.0, 2.0, 3.0, 4.0]);
    let tangent = tangent.unwrap();
    assert_eq!(
        as_slice(tangent.as_f64().unwrap().primal()),
        &[0.5, -0.25, 0.75, 1.0]
    );
}

#[test]
fn dyn_ad_tensor_reshape_pullback_restores_original_shape() {
    let mut x = Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0]));
    x.set_requires_grad(true).unwrap();

    let reshaped = x.reshape(&[4]).unwrap();
    let cotangent = Tensor::from_tensor(vector(&[1.0, -2.0, 0.5, 3.0]));
    let grad_outputs = [cotangent];
    let grads =
        tenferro::grad(&[&reshaped], &[&x], Some(&grad_outputs), Default::default()).unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2, 2]);
    assert_eq!(
        as_slice(grad.as_f64().unwrap().primal()),
        &[1.0, -2.0, 0.5, 3.0]
    );
}

#[test]
fn dyn_ad_tensor_take_prefix_preserves_forward_mode() {
    let (sliced, tangent) = forward_ad::dual_level(|fw| {
        let x = fw.make_dual(
            &Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0])),
            &Tensor::from_tensor(matrix2(&[0.5, -0.25, 0.75, 1.0])),
        )?;
        let sliced = x.take_prefix(1, 1)?;
        fw.unpack_dual(&sliced)
    })
    .unwrap();

    let sliced = sliced.as_f64().unwrap();
    assert_eq!(sliced.dims(), &[2, 1]);
    assert_eq!(as_slice(sliced.primal()), &[1.0, 2.0]);
    assert_eq!(
        as_slice(tangent.unwrap().as_f64().unwrap().primal()),
        &[0.5, -0.25]
    );
}

#[test]
fn dyn_ad_tensor_take_prefix_pullback_zero_fills_dropped_entries() {
    let mut x = Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0]));
    x.set_requires_grad(true).unwrap();

    let sliced = x.take_prefix(1, 1).unwrap();
    let cotangent = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.5, -0.5], &[2, 1], MemoryOrder::ColumnMajor).unwrap(),
    );

    let grad_outputs = [cotangent];
    let grads = tenferro::grad(&[&sliced], &[&x], Some(&grad_outputs), Default::default()).unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2, 2]);
    assert_eq!(
        as_slice(grad.as_f64().unwrap().primal()),
        &[1.5, -0.5, 0.0, 0.0]
    );
}

#[test]
fn dyn_ad_tensor_diag_embed_preserves_reverse_pullback() {
    let mut x = Tensor::from_tensor(vector(&[2.0, 3.0]));
    x.set_requires_grad(true).unwrap();

    let diag = x.diag_embed(2).unwrap();
    assert!(diag.is_diag());
    let cotangent = diag_tensor(&[0.5, -1.0]);

    let grad_outputs = [cotangent];
    let grads = tenferro::grad(&[&diag], &[&x], Some(&grad_outputs), Default::default()).unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert_eq!(grad.dims(), &[2]);
    assert_eq!(as_slice(grad.as_f64().unwrap().primal()), &[0.5, -1.0]);
}

#[test]
fn dyn_ad_tensor_diag_embed_preserves_forward_mode() {
    let (diag, tangent) = forward_ad::dual_level(|fw| {
        let x = fw.make_dual(
            &Tensor::from_tensor(vector(&[2.0, 3.0])),
            &Tensor::from_tensor(vector(&[0.5, -1.0])),
        )?;
        let diag = x.diag_embed(2)?;
        fw.unpack_dual(&diag)
    })
    .unwrap();

    assert!(diag.is_diag());
    assert_eq!(diag.dims(), &[2, 2]);
    let tangent = tangent.unwrap();
    assert!(tangent.is_diag());
    assert_eq!(tangent.dims(), &[2, 2]);
    assert_eq!(as_slice(tangent.as_f64().unwrap().primal()), &[0.5, -1.0]);
}

#[test]
fn dyn_ad_tensor_with_axis_classes_preserves_forward_mode() {
    let (structured, tangent) = forward_ad::dual_level(|fw| {
        let payload = Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0]));
        let tangent = Tensor::from_tensor(matrix2(&[0.25, 0.5, -0.75, 1.0]));
        let dual = fw.make_dual(&payload, &tangent)?;
        let structured = Tensor::with_axis_classes(dual, &[0, 0, 1, 1])?;
        fw.unpack_dual(&structured)
    })
    .unwrap();

    assert_eq!(structured.dims(), &[2, 2, 2, 2]);
    assert_eq!(structured.axis_classes(), &[0, 0, 1, 1]);
    assert_eq!(tangent.unwrap().axis_classes(), &[0, 0, 1, 1]);
}

#[test]
fn dyn_ad_tensor_with_axis_classes_reverse_pullback_restores_dense_payload() {
    let mut payload = Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0]));
    payload.set_requires_grad(true).unwrap();

    let structured = Tensor::with_axis_classes(payload.clone(), &[0, 0, 1, 1]).unwrap();
    let grad_outputs = [Tensor::with_axis_classes(
        Tensor::from_tensor(matrix2(&[0.5, -1.0, 2.0, -0.25])),
        &[0, 0, 1, 1],
    )
    .unwrap()];
    let grads = tenferro::grad(
        &[&structured],
        &[&payload],
        Some(&grad_outputs),
        Default::default(),
    )
    .unwrap();
    let grad = grads.into_iter().next().unwrap().unwrap();

    assert!(grad.is_dense());
    assert_eq!(grad.dims(), &[2, 2]);
    assert_eq!(
        as_slice(grad.as_f64().unwrap().primal()),
        &[0.5, -1.0, 2.0, -0.25]
    );
}

#[test]
fn with_axis_classes_rejects_non_dense_payloads_and_rank_mismatches() {
    let structured_payload = diag_tensor(&[1.0, 2.0]);
    let non_dense = match Tensor::with_axis_classes(structured_payload, &[0, 0]) {
        Ok(_) => panic!("expected non-dense payload to be rejected"),
        Err(err) => err,
    };
    let non_dense_message = match non_dense {
        tenferro::Error::InvalidAdTensor { message } => message,
        other => panic!("expected InvalidAdTensor, got {other:?}"),
    };
    assert!(non_dense_message.contains("supports only dense tensors"));

    let rank_mismatch =
        match Tensor::with_axis_classes(Tensor::from_tensor(vector(&[1.0, 2.0])), &[0, 1]) {
            Ok(_) => panic!("expected rank mismatch to be rejected"),
            Err(err) => err,
        };
    let rank_mismatch_message = match rank_mismatch {
        tenferro::Error::InvalidAdTensor { message } => message,
        other => panic!("expected InvalidAdTensor, got {other:?}"),
    };
    assert!(rank_mismatch_message.contains("payload rank"));
}

#[test]
fn diag_embed_rejects_non_vector_dense_inputs() {
    let err = match Tensor::from_tensor(matrix2(&[1.0, 2.0, 3.0, 4.0])).diag_embed(2) {
        Ok(_) => panic!("expected non-vector diag_embed to be rejected"),
        Err(err) => err,
    };
    let message = match err {
        tenferro::Error::InvalidAdTensor { message } => message,
        other => panic!("expected InvalidAdTensor, got {other:?}"),
    };
    assert!(message.contains("rank-1 dense tensor"));
}

#[test]
fn diag_embed_forward_mode_covers_complex_variant() {
    let (diag, tangent) = forward_ad::dual_level(|fw| {
        let x = fw.make_dual(
            &Tensor::from_tensor(vector_c32(&[
                Complex32::new(1.0, -2.0),
                Complex32::new(0.5, 0.25),
            ])),
            &Tensor::from_tensor(vector_c32(&[
                Complex32::new(0.0, 1.0),
                Complex32::new(-0.5, 0.0),
            ])),
        )?;
        let diag = x.diag_embed(2)?;
        fw.unpack_dual(&diag)
    })
    .unwrap();

    assert_eq!(diag.scalar_type(), tenferro::ScalarType::C32);
    assert!(diag.is_diag());
    assert_eq!(tangent.unwrap().scalar_type(), tenferro::ScalarType::C32);
}
