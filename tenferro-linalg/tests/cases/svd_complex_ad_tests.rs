use num_complex::Complex64;
use tenferro_linalg::{svd, svd_frule, svd_rrule, SvdCotangent};
use tenferro_prims::CpuContext;
use tenferro_tensor::Tensor;

fn make_complex_tensor(data: Vec<Complex64>, dims: &[usize]) -> Tensor<Complex64> {
    let mut strides = vec![0isize; dims.len()];
    if !dims.is_empty() {
        strides[0] = 1;
        for axis in 1..dims.len() {
            strides[axis] = strides[axis - 1] * dims[axis - 1] as isize;
        }
    }
    Tensor::from_vec(data, dims, &strides, 0).unwrap()
}

fn make_real_tensor(data: Vec<f64>, dims: &[usize]) -> Tensor<f64> {
    let mut strides = vec![0isize; dims.len()];
    if !dims.is_empty() {
        strides[0] = 1;
        for axis in 1..dims.len() {
            strides[axis] = strides[axis - 1] * dims[axis - 1] as isize;
        }
    }
    Tensor::from_vec(data, dims, &strides, 0).unwrap()
}

fn complex_tensor_data(tensor: &Tensor<Complex64>) -> Vec<Complex64> {
    let contiguous = tensor.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len: usize = contiguous.dims().iter().product();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn real_tensor_data(tensor: &Tensor<f64>) -> Vec<f64> {
    let contiguous = tensor.contiguous(tenferro_tensor::MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len: usize = contiguous.dims().iter().product();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn add_scaled(base: &[Complex64], direction: &[Complex64], scale: f64) -> Vec<Complex64> {
    base.iter()
        .zip(direction.iter())
        .map(|(x, dx)| *x + *dx * scale)
        .collect()
}

fn svd_s_loss(a: &Tensor<Complex64>, cotangent_s: &Tensor<f64>) -> f64 {
    let mut ctx = CpuContext::new(1);
    let s = svd(&mut ctx, a, None).unwrap().s;
    let s_data = real_tensor_data(&s);
    let cot_data = real_tensor_data(cotangent_s);
    s_data
        .iter()
        .zip(cot_data.iter())
        .map(|(value, co)| value * co)
        .sum()
}

fn complex_fixture() -> (Tensor<Complex64>, Tensor<Complex64>) {
    let a = make_complex_tensor(
        vec![
            Complex64::new(1.0, 0.25),
            Complex64::new(-0.5, 0.75),
            Complex64::new(0.2, -0.4),
            Complex64::new(0.3, -0.1),
            Complex64::new(1.7, 0.5),
            Complex64::new(-0.8, 0.2),
            Complex64::new(-1.1, 0.6),
            Complex64::new(0.4, -0.9),
            Complex64::new(0.9, -0.3),
        ],
        &[3, 3],
    );
    let da = make_complex_tensor(
        vec![
            Complex64::new(0.05, -0.02),
            Complex64::new(-0.04, 0.03),
            Complex64::new(0.01, 0.06),
            Complex64::new(-0.03, 0.01),
            Complex64::new(0.02, -0.05),
            Complex64::new(0.07, 0.04),
            Complex64::new(-0.06, 0.02),
            Complex64::new(0.03, -0.07),
            Complex64::new(0.04, 0.05),
        ],
        &[3, 3],
    );
    (a, da)
}

#[test]
fn svd_frule_complex64_through_s_matches_finite_difference() {
    let (a, da) = complex_fixture();
    let eps = 1e-5;

    let mut ctx = CpuContext::new(1);
    let (_, tangent) = svd_frule(&mut ctx, &a, &da, None).unwrap();
    let analytic = real_tensor_data(&tangent.s);

    let a_data = complex_tensor_data(&a);
    let da_data = complex_tensor_data(&da);
    let plus = make_complex_tensor(add_scaled(&a_data, &da_data, eps), &[3, 3]);
    let minus = make_complex_tensor(add_scaled(&a_data, &da_data, -eps), &[3, 3]);

    let mut plus_ctx = CpuContext::new(1);
    let mut minus_ctx = CpuContext::new(1);
    let s_plus = real_tensor_data(&svd(&mut plus_ctx, &plus, None).unwrap().s);
    let s_minus = real_tensor_data(&svd(&mut minus_ctx, &minus, None).unwrap().s);

    for (idx, actual) in analytic.iter().enumerate() {
        let fd = (s_plus[idx] - s_minus[idx]) / (2.0 * eps);
        assert!(
            (actual - fd).abs() < 5e-4,
            "complex svd_frule through s mismatch at {idx}: analytic={actual}, fd={fd}"
        );
    }
}

#[test]
fn svd_rrule_complex64_through_s_matches_finite_difference() {
    let (a, _) = complex_fixture();
    let cotangent_s = make_real_tensor(vec![0.5, -0.75, 0.25], &[3]);
    let eps = 1e-5;

    let mut ctx = CpuContext::new(1);
    let grad = svd_rrule(
        &mut ctx,
        &a,
        &SvdCotangent::<Complex64, f64> {
            u: None,
            s: Some(cotangent_s.clone()),
            vt: None,
        },
        None,
    )
    .unwrap();
    let analytic = complex_tensor_data(&grad);
    let base = complex_tensor_data(&a);

    for idx in 0..analytic.len() {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus[idx] += Complex64::new(eps, 0.0);
        minus[idx] -= Complex64::new(eps, 0.0);
        let fd_real = (svd_s_loss(&make_complex_tensor(plus, &[3, 3]), &cotangent_s)
            - svd_s_loss(&make_complex_tensor(minus, &[3, 3]), &cotangent_s))
            / (2.0 * eps);
        assert!(
            (analytic[idx].re - fd_real).abs() < 5e-4,
            "complex svd_rrule real mismatch at {idx}: analytic={}, fd={fd_real}",
            analytic[idx].re
        );

        let mut plus = base.clone();
        let mut minus = base.clone();
        plus[idx] += Complex64::new(0.0, eps);
        minus[idx] -= Complex64::new(0.0, eps);
        let fd_imag = (svd_s_loss(&make_complex_tensor(plus, &[3, 3]), &cotangent_s)
            - svd_s_loss(&make_complex_tensor(minus, &[3, 3]), &cotangent_s))
            / (2.0 * eps);
        assert!(
            (analytic[idx].im - fd_imag).abs() < 5e-4,
            "complex svd_rrule imag mismatch at {idx}: analytic={}, fd={fd_imag}",
            analytic[idx].im
        );
    }
}
