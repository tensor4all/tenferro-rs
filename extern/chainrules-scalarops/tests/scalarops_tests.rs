use chainrules_scalarops::{
    add, add_frule, add_rrule, conj, conj_frule, conj_rrule, div, div_frule, div_rrule,
    handle_r_to_c_f32, handle_r_to_c_f64, mul, mul_frule, mul_rrule, powf, powf_frule, powf_rrule,
    powi, powi_frule, powi_rrule, sqrt, sqrt_frule, sqrt_rrule, sub, sub_frule, sub_rrule,
};
use num_complex::{Complex32, Complex64};

#[test]
fn handle_r_to_c_projects_real_part() {
    let g32 = Complex32::new(1.25, -9.0);
    let g64 = Complex64::new(-3.5, 2.0);
    assert_eq!(handle_r_to_c_f32(g32), 1.25_f32);
    assert_eq!(handle_r_to_c_f64(g64), -3.5_f64);
}

#[test]
fn conj_rules_match_formula_complex64() {
    let x = Complex64::new(2.0, -3.0);
    let dx = Complex64::new(-0.5, 4.0);
    let g = Complex64::new(1.5, -2.0);

    let (y, dy) = conj_frule(x, dx);
    assert_eq!(y, Complex64::new(2.0, 3.0));
    assert_eq!(dy, Complex64::new(-0.5, -4.0));

    let grad = conj_rrule(g);
    assert_eq!(grad, Complex64::new(1.5, 2.0));
}

#[test]
fn sqrt_rules_match_formula_f64() {
    let x = 9.0_f64;
    let dx = 1.0_f64;
    let g = 1.0_f64;

    let (y, dy) = sqrt_frule(x, dx);
    assert!((y - 3.0).abs() < 1e-12);
    assert!((dy - (1.0 / 6.0)).abs() < 1e-12);

    let grad = sqrt_rrule(y, g);
    assert!((grad - (1.0 / 6.0)).abs() < 1e-12);
}

#[test]
fn sqrt_rules_surface_singularity_at_zero() {
    let (y, dy) = sqrt_frule(0.0_f64, 1.0_f64);
    assert_eq!(y, 0.0);
    assert!(dy.is_infinite(), "sqrt_frule at zero should be singular");

    let grad = sqrt_rrule(0.0_f64, 1.0_f64);
    assert!(grad.is_infinite(), "sqrt_rrule at zero should be singular");
}

#[test]
fn add_sub_rules_match_formula_f64() {
    let x = 5.0_f64;
    let y = 2.0_f64;
    let dx = 0.3_f64;
    let dy = 0.1_f64;
    let g = 1.5_f64;

    assert_eq!(add(x, y), 7.0_f64);
    assert_eq!(sub(x, y), 3.0_f64);

    let (add_y, add_dy) = add_frule(x, y, dx, dy);
    assert_eq!(add_y, 7.0_f64);
    assert_eq!(add_dy, 0.4_f64);
    let (add_dx, add_dy_rr) = add_rrule(g);
    assert_eq!(add_dx, g);
    assert_eq!(add_dy_rr, g);

    let (sub_y, sub_dy) = sub_frule(x, y, dx, dy);
    assert_eq!(sub_y, 3.0_f64);
    assert!((sub_dy - 0.2_f64).abs() < 1e-12);
    let (sub_dx, sub_dy_rr) = sub_rrule(g);
    assert_eq!(sub_dx, g);
    assert_eq!(sub_dy_rr, -g);
}

#[test]
fn mul_div_rules_match_formula_f64() {
    let x = 8.0_f64;
    let y = 2.0_f64;
    let dx = 0.5_f64;
    let dy = 0.25_f64;
    let g = 1.0_f64;

    assert_eq!(mul(x, y), 16.0_f64);
    assert_eq!(div(x, y), 4.0_f64);

    let (mul_y, mul_dy) = mul_frule(x, y, dx, dy);
    assert_eq!(mul_y, 16.0_f64);
    assert_eq!(mul_dy, dx * y + dy * x);
    let (mul_dx, mul_dy_rr) = mul_rrule(x, y, g);
    assert_eq!(mul_dx, g * y);
    assert_eq!(mul_dy_rr, g * x);

    let (div_y, div_dy) = div_frule(x, y, dx, dy);
    assert_eq!(div_y, 4.0_f64);
    assert_eq!(div_dy, (dx / y) - (dy * x / (y * y)));
    let (div_dx, div_dy_rr) = div_rrule(x, y, g);
    assert_eq!(div_dx, g / y);
    assert_eq!(div_dy_rr, -(g * x / (y * y)));
}

#[test]
fn powf_rules_match_formula_f64() {
    let x = 2.0_f64;
    let exponent = 3.0_f64;
    let dx = 1.0_f64;
    let g = 1.0_f64;

    let (y, dy) = powf_frule(x, exponent, dx);
    assert!((y - 8.0).abs() < 1e-12);
    assert!((dy - 12.0).abs() < 1e-12);

    let grad = powf_rrule(x, exponent, g);
    assert!((grad - 12.0).abs() < 1e-12);
}

#[test]
fn mul_div_rules_match_formula_complex64() {
    let x = Complex64::new(1.5, -0.5);
    let y = Complex64::new(-0.25, 2.0);
    let dx = Complex64::new(0.3, -0.2);
    let dy = Complex64::new(-0.1, 0.4);
    let g = Complex64::new(0.7, -0.6);

    let (mul_y, mul_dy) = mul_frule(x, y, dx, dy);
    assert!((mul_y - (x * y)).norm() < 1e-12);
    let expected_mul_tangent = dx * y.conj() + dy * x.conj();
    assert!((mul_dy - expected_mul_tangent).norm() < 1e-12);
    let (mul_dx, mul_dy_rr) = mul_rrule(x, y, g);
    assert!((mul_dx - g * y.conj()).norm() < 1e-12);
    assert!((mul_dy_rr - g * x.conj()).norm() < 1e-12);

    let (div_y, div_dy) = div_frule(x, y, dx, dy);
    assert!((div_y - (x / y)).norm() < 1e-12);
    let expected_div_tangent = dx * (Complex64::new(1.0, 0.0) / y).conj()
        + dy * ((Complex64::new(-1.0, 0.0) * x) / (y * y)).conj();
    assert!((div_dy - expected_div_tangent).norm() < 1e-12);
    let (div_dx, div_dy_rr) = div_rrule(x, y, g);
    assert!((div_dx - g * (Complex64::new(1.0, 0.0) / y).conj()).norm() < 1e-12);
    assert!((div_dy_rr - g * ((Complex64::new(-1.0, 0.0) * x) / (y * y)).conj()).norm() < 1e-12);
}

#[test]
fn powi_rules_match_formula_complex64() {
    let x = Complex64::new(1.0, 2.0);
    let exponent = 3_i32;
    let dx = Complex64::new(-1.0, 0.5);
    let g = Complex64::new(0.25, -0.75);

    let (y, dy) = powi_frule(x, exponent, dx);
    let expected_y = x * x * x;
    assert!((y - expected_y).norm() < 1e-12);

    let expected_scale = (Complex64::new(3.0, 0.0) * x.powi(2)).conj();
    assert!((dy - (dx * expected_scale)).norm() < 1e-12);

    let grad = powi_rrule(x, exponent, g);
    assert!((grad - (g * expected_scale)).norm() < 1e-12);
}

#[test]
fn pow_zero_exponent_returns_zero_gradients() {
    let x = 0.0_f64;

    let (_y_f, dy_f) = powf_frule(x, 0.0, 1.0);
    assert_eq!(dy_f, 0.0);
    assert_eq!(powf_rrule(x, 0.0, 1.0), 0.0);

    let (_y_i, dy_i) = powi_frule(x, 0, 1.0);
    assert_eq!(dy_i, 0.0);
    assert_eq!(powi_rrule(x, 0, 1.0), 0.0);
}

#[test]
fn primal_wrappers_cover_real_and_complex() {
    assert_eq!(conj(3.0_f32), 3.0_f32);
    assert!((sqrt(16.0_f32) - 4.0_f32).abs() < 1e-6);
    assert!((powf(2.0_f32, 3.5_f32) - 2.0_f32.powf(3.5_f32)).abs() < 1e-6);
    assert!((powi(2.0_f32, -2) - 0.25_f32).abs() < 1e-6);

    let z32 = Complex32::new(3.0, 4.0);
    assert_eq!(conj(z32), Complex32::new(3.0, -4.0));
    let z32_sqrt = sqrt(z32);
    assert!((z32_sqrt * z32_sqrt - z32).norm() < 1e-5);
    assert!((powf(z32, 1.25_f32) - z32.powf(1.25_f32)).norm() < 1e-5);
    assert!((powi(z32, 3) - z32.powi(3)).norm() < 1e-5);

    let z64 = Complex64::new(2.0, -1.5);
    assert_eq!(conj(z64), Complex64::new(2.0, 1.5));
    let z64_sqrt = sqrt(z64);
    assert!((z64_sqrt * z64_sqrt - z64).norm() < 1e-12);
    assert!((powf(z64, 2.5_f64) - z64.powf(2.5_f64)).norm() < 1e-12);
    assert!((powi(z64, -3) - z64.powi(-3)).norm() < 1e-12);
}

#[test]
fn complex_frules_and_rrules_cover_from_real_paths() {
    let x32 = Complex32::new(1.2, -0.3);
    let dx32 = Complex32::new(-0.4, 0.7);
    let g32 = Complex32::new(0.6, -0.2);
    let (_y32, dy32) = powf_frule(x32, 2.0_f32, dx32);
    let grad32 = powf_rrule(x32, 2.0_f32, g32);
    let expected_scale32 = (Complex32::new(2.0, 0.0) * x32.powf(1.0_f32)).conj();
    assert!((dy32 - dx32 * expected_scale32).norm() < 1e-5);
    assert!((grad32 - g32 * expected_scale32).norm() < 1e-5);

    let x64 = Complex64::new(-0.8, 1.1);
    let dx64 = Complex64::new(0.9, -0.4);
    let g64 = Complex64::new(0.3, 0.2);
    let (_y64, dy64) = powi_frule(x64, 4, dx64);
    let grad64 = powi_rrule(x64, 4, g64);
    let expected_scale64 = (Complex64::new(4.0, 0.0) * x64.powi(3)).conj();
    assert!((dy64 - dx64 * expected_scale64).norm() < 1e-12);
    assert!((grad64 - g64 * expected_scale64).norm() < 1e-12);
}
