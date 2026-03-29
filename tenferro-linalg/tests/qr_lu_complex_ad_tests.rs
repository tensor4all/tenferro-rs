use num_complex::Complex64;
use tenferro_linalg::{
    lu, lu_frule, lu_rrule, qr, qr_frule, qr_rrule, LuCotangent, LuPivot, QrCotangent,
};
use tenferro_linalg_prims::{KernelLinalgScalar, LinalgScalar};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

trait FdScalar: KernelLinalgScalar + Copy {
    fn from_f64_parts(real: f64, imag: f64) -> Self;
    fn max_abs_diff(lhs: &[Self], rhs: &[Self]) -> f64;
    fn real_pairing(lhs: &[Self], rhs: &[Self]) -> f64;
}

impl FdScalar for f64 {
    fn from_f64_parts(real: f64, _imag: f64) -> Self {
        real
    }

    fn max_abs_diff(lhs: &[Self], rhs: &[Self]) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    fn real_pairing(lhs: &[Self], rhs: &[Self]) -> f64 {
        lhs.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum()
    }
}

impl FdScalar for Complex64 {
    fn from_f64_parts(real: f64, imag: f64) -> Self {
        Self::new(real, imag)
    }

    fn max_abs_diff(lhs: &[Self], rhs: &[Self]) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(x, y)| (*x - *y).norm())
            .fold(0.0_f64, f64::max)
    }

    fn real_pairing(lhs: &[Self], rhs: &[Self]) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(x, y)| (x.conj() * *y).re)
            .sum()
    }
}

fn tensor_from_pairs<T: FdScalar>(pairs: &[(f64, f64)], dims: &[usize]) -> Tensor<T> {
    let data = pairs
        .iter()
        .map(|(real, imag)| T::from_f64_parts(*real, *imag))
        .collect();
    tensor_from_vec(data, dims)
}

fn tensor_from_vec<T: LinalgScalar>(data: Vec<T>, dims: &[usize]) -> Tensor<T> {
    let mut strides = vec![0isize; dims.len()];
    if !dims.is_empty() {
        strides[0] = 1;
        for axis in 1..dims.len() {
            strides[axis] = strides[axis - 1] * dims[axis - 1] as isize;
        }
    }
    Tensor::from_vec(data, dims, &strides, 0).unwrap()
}

fn tensor_data<T: LinalgScalar>(tensor: &Tensor<T>) -> Vec<T> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len: usize = contiguous.dims().iter().product();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn add_scaled<T: FdScalar>(base: &[T], direction: &[T], scale: f64) -> Vec<T> {
    let alpha = T::from_f64_parts(scale, 0.0);
    base.iter()
        .zip(direction.iter())
        .map(|(x, dx)| *x + *dx * alpha)
        .collect()
}

fn check_forward_fd<T, Forward, Frule>(
    a: &Tensor<T>,
    da: &Tensor<T>,
    eps: f64,
    tol: f64,
    forward: Forward,
    frule: Frule,
    label: &str,
) where
    T: FdScalar,
    Forward: Fn(&mut CpuContext, &Tensor<T>) -> Tensor<T>,
    Frule: Fn(&mut CpuContext, &Tensor<T>, &Tensor<T>) -> Tensor<T>,
{
    let mut ctx = CpuContext::new(1);
    let analytic = tensor_data(&frule(&mut ctx, a, da));

    let base = tensor_data(a);
    let direction = tensor_data(da);
    let plus = tensor_from_vec(add_scaled(&base, &direction, eps), a.dims());
    let minus = tensor_from_vec(add_scaled(&base, &direction, -eps), a.dims());

    let mut plus_ctx = CpuContext::new(1);
    let mut minus_ctx = CpuContext::new(1);
    let plus_data = tensor_data(&forward(&mut plus_ctx, &plus));
    let minus_data = tensor_data(&forward(&mut minus_ctx, &minus));
    let fd: Vec<T> = plus_data
        .iter()
        .zip(minus_data.iter())
        .map(|(p, m)| (*p - *m) * T::from_f64_parts(0.5 / eps, 0.0))
        .collect();

    let err = T::max_abs_diff(&analytic, &fd);
    assert!(err < tol, "{label} forward fd mismatch: {err}");
}

fn check_backward_directional_fd<T, Forward, Rrule>(
    a: &Tensor<T>,
    da: &Tensor<T>,
    cotangent: &Tensor<T>,
    eps: f64,
    tol: f64,
    forward: Forward,
    rrule: Rrule,
    label: &str,
) where
    T: FdScalar,
    Forward: Fn(&mut CpuContext, &Tensor<T>) -> Tensor<T>,
    Rrule: Fn(&mut CpuContext, &Tensor<T>, &Tensor<T>) -> Tensor<T>,
{
    let mut ctx = CpuContext::new(1);
    let grad = rrule(&mut ctx, a, cotangent);
    let predicted = T::real_pairing(&tensor_data(&grad), &tensor_data(da));

    let objective = |a_now: &Tensor<T>| -> f64 {
        let mut ctx = CpuContext::new(1);
        let out = forward(&mut ctx, a_now);
        T::real_pairing(&tensor_data(cotangent), &tensor_data(&out))
    };

    let base = tensor_data(a);
    let direction = tensor_data(da);
    let plus = tensor_from_vec(add_scaled(&base, &direction, eps), a.dims());
    let minus = tensor_from_vec(add_scaled(&base, &direction, -eps), a.dims());
    let fd = (objective(&plus) - objective(&minus)) / (2.0 * eps);
    let err = (predicted - fd).abs();
    assert!(err < tol, "{label} backward directional fd mismatch: {err}");
}

fn qr_square_fixture<T: FdScalar>() -> (Tensor<T>, Tensor<T>, Tensor<T>) {
    let a = tensor_from_pairs(
        &[
            (1.2, 0.2),
            (-0.4, 0.5),
            (0.3, -0.2),
            (0.7, -0.1),
            (1.5, 0.3),
            (-0.6, 0.4),
            (-0.2, 0.6),
            (0.8, -0.3),
            (1.1, 0.1),
        ],
        &[3, 3],
    );
    let da = tensor_from_pairs(
        &[
            (0.04, -0.01),
            (-0.03, 0.05),
            (0.02, 0.01),
            (-0.01, 0.03),
            (0.05, -0.02),
            (0.01, 0.04),
            (-0.02, 0.02),
            (0.03, -0.05),
            (0.02, 0.03),
        ],
        &[3, 3],
    );
    let cq = tensor_from_pairs(
        &[
            (0.2, -0.1),
            (-0.1, 0.05),
            (0.04, 0.03),
            (0.03, 0.02),
            (-0.15, 0.04),
            (0.06, -0.02),
            (-0.05, 0.01),
            (0.08, -0.06),
            (0.07, 0.02),
        ],
        &[3, 3],
    );
    (a, da, cq)
}

fn qr_wide_fixture() -> (Tensor<Complex64>, Tensor<Complex64>, Tensor<Complex64>) {
    let a = tensor_from_pairs(
        &[
            (1.0, 0.2),
            (0.4, 0.6),
            (0.3, -0.5),
            (1.3, -0.2),
            (-0.7, 0.1),
            (0.5, 0.4),
        ],
        &[2, 3],
    );
    let da = tensor_from_pairs(
        &[
            (0.05, -0.03),
            (-0.02, 0.04),
            (0.03, 0.01),
            (-0.04, 0.02),
            (0.01, -0.05),
            (0.02, 0.03),
        ],
        &[2, 3],
    );
    let cr = tensor_from_pairs(
        &[
            (0.2, -0.1),
            (-0.05, 0.02),
            (0.04, 0.03),
            (0.03, 0.01),
            (-0.08, 0.05),
            (0.06, -0.04),
        ],
        &[2, 3],
    );
    (a, da, cr)
}

fn lu_square_fixture<T: FdScalar>() -> (Tensor<T>, Tensor<T>, Tensor<T>) {
    let a = tensor_from_pairs(
        &[
            (2.2, 0.3),
            (0.5, -0.2),
            (-0.1, 0.4),
            (0.8, -0.1),
            (1.9, 0.2),
            (0.6, 0.1),
            (-0.3, 0.2),
            (0.4, -0.5),
            (1.6, 0.3),
        ],
        &[3, 3],
    );
    let da = tensor_from_pairs(
        &[
            (0.03, -0.01),
            (-0.02, 0.04),
            (0.01, 0.02),
            (0.04, 0.01),
            (-0.03, 0.02),
            (0.02, -0.04),
            (0.01, 0.03),
            (0.05, -0.02),
            (-0.04, 0.01),
        ],
        &[3, 3],
    );
    let cu = tensor_from_pairs(
        &[
            (0.2, 0.0),
            (-0.1, 0.03),
            (0.04, -0.02),
            (0.05, 0.01),
            (0.07, -0.04),
            (-0.03, 0.02),
            (0.01, 0.03),
            (0.06, -0.01),
            (0.08, 0.02),
        ],
        &[3, 3],
    );
    (a, da, cu)
}

fn lu_wide_fixture() -> (Tensor<Complex64>, Tensor<Complex64>, Tensor<Complex64>) {
    let a = tensor_from_pairs(
        &[
            (1.8, 0.2),
            (0.6, -0.1),
            (0.4, 0.5),
            (2.1, -0.3),
            (-0.2, 0.4),
            (0.7, 0.1),
        ],
        &[2, 3],
    );
    let da = tensor_from_pairs(
        &[
            (0.03, -0.02),
            (-0.01, 0.03),
            (0.02, 0.01),
            (-0.04, 0.02),
            (0.01, -0.03),
            (0.05, 0.01),
        ],
        &[2, 3],
    );
    let cu = tensor_from_pairs(
        &[
            (0.15, -0.02),
            (0.03, 0.04),
            (-0.06, 0.01),
            (0.07, -0.03),
            (0.04, 0.02),
            (-0.02, 0.05),
        ],
        &[2, 3],
    );
    (a, da, cu)
}

fn lu_tall_fixture() -> (Tensor<Complex64>, Tensor<Complex64>, Tensor<Complex64>) {
    let a = tensor_from_pairs(
        &[
            (2.0, 0.1),
            (0.5, -0.2),
            (-0.4, 0.3),
            (0.6, 0.2),
            (1.7, -0.1),
            (0.8, 0.4),
        ],
        &[3, 2],
    );
    let da = tensor_from_pairs(
        &[
            (0.04, -0.01),
            (-0.03, 0.02),
            (0.02, 0.03),
            (0.01, -0.04),
            (-0.02, 0.01),
            (0.03, 0.02),
        ],
        &[3, 2],
    );
    let cl = tensor_from_pairs(
        &[
            (0.0, 0.0),
            (0.08, -0.02),
            (-0.05, 0.03),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.07, -0.01),
        ],
        &[3, 2],
    );
    (a, da, cl)
}

#[test]
fn qr_square_q_forward_matches_fd_f64() {
    let (a, da, _) = qr_square_fixture::<f64>();
    check_forward_fd(
        &a,
        &da,
        1e-6,
        2e-5,
        |ctx, a_now| qr(ctx, a_now).unwrap().q,
        |ctx, a_now, da_now| qr_frule(ctx, a_now, da_now).unwrap().1.q,
        "qr_square_q_forward_matches_fd_f64",
    );
}

#[test]
fn qr_square_q_backward_matches_fd_f64() {
    let (a, da, cq) = qr_square_fixture::<f64>();
    check_backward_directional_fd(
        &a,
        &da,
        &cq,
        1e-6,
        2e-5,
        |ctx, a_now| qr(ctx, a_now).unwrap().q,
        |ctx, a_now, cotangent| {
            qr_rrule(
                ctx,
                a_now,
                &QrCotangent {
                    q: Some(cotangent.clone()),
                    r: None,
                },
            )
            .unwrap()
        },
        "qr_square_q_backward_matches_fd_f64",
    );
}

#[test]
fn lu_square_u_forward_matches_fd_f64() {
    let (a, da, _) = lu_square_fixture::<f64>();
    check_forward_fd(
        &a,
        &da,
        1e-6,
        2e-5,
        |ctx, a_now| lu(ctx, a_now, LuPivot::Partial).unwrap().u,
        |ctx, a_now, da_now| lu_frule(ctx, a_now, da_now, LuPivot::Partial).unwrap().1.u,
        "lu_square_u_forward_matches_fd_f64",
    );
}

#[test]
fn lu_square_u_backward_matches_fd_f64() {
    let (a, da, cu) = lu_square_fixture::<f64>();
    check_backward_directional_fd(
        &a,
        &da,
        &cu,
        1e-6,
        2e-5,
        |ctx, a_now| lu(ctx, a_now, LuPivot::Partial).unwrap().u,
        |ctx, a_now, cotangent| {
            lu_rrule(
                ctx,
                a_now,
                &LuCotangent {
                    l: None,
                    u: Some(cotangent.clone()),
                },
                LuPivot::Partial,
            )
            .unwrap()
        },
        "lu_square_u_backward_matches_fd_f64",
    );
}

#[test]
fn qr_square_q_forward_matches_fd_c64() {
    let (a, da, _) = qr_square_fixture::<Complex64>();
    check_forward_fd(
        &a,
        &da,
        1e-6,
        5e-5,
        |ctx, a_now| qr(ctx, a_now).unwrap().q,
        |ctx, a_now, da_now| qr_frule(ctx, a_now, da_now).unwrap().1.q,
        "qr_square_q_forward_matches_fd_c64",
    );
}

#[test]
fn qr_square_q_backward_matches_fd_c64() {
    let (a, da, cq) = qr_square_fixture::<Complex64>();
    check_backward_directional_fd(
        &a,
        &da,
        &cq,
        1e-6,
        5e-5,
        |ctx, a_now| qr(ctx, a_now).unwrap().q,
        |ctx, a_now, cotangent| {
            qr_rrule(
                ctx,
                a_now,
                &QrCotangent {
                    q: Some(cotangent.clone()),
                    r: None,
                },
            )
            .unwrap()
        },
        "qr_square_q_backward_matches_fd_c64",
    );
}

#[test]
fn lu_square_u_forward_matches_fd_c64() {
    let (a, da, _) = lu_square_fixture::<Complex64>();
    check_forward_fd(
        &a,
        &da,
        1e-6,
        5e-5,
        |ctx, a_now| lu(ctx, a_now, LuPivot::Partial).unwrap().u,
        |ctx, a_now, da_now| lu_frule(ctx, a_now, da_now, LuPivot::Partial).unwrap().1.u,
        "lu_square_u_forward_matches_fd_c64",
    );
}

#[test]
fn lu_square_u_backward_matches_fd_c64() {
    let (a, da, cu) = lu_square_fixture::<Complex64>();
    check_backward_directional_fd(
        &a,
        &da,
        &cu,
        1e-6,
        5e-5,
        |ctx, a_now| lu(ctx, a_now, LuPivot::Partial).unwrap().u,
        |ctx, a_now, cotangent| {
            lu_rrule(
                ctx,
                a_now,
                &LuCotangent {
                    l: None,
                    u: Some(cotangent.clone()),
                },
                LuPivot::Partial,
            )
            .unwrap()
        },
        "lu_square_u_backward_matches_fd_c64",
    );
}

#[test]
fn qr_wide_r_forward_matches_fd_c64() {
    let (a, da, _) = qr_wide_fixture();
    check_forward_fd(
        &a,
        &da,
        1e-6,
        8e-5,
        |ctx, a_now| qr(ctx, a_now).unwrap().r,
        |ctx, a_now, da_now| qr_frule(ctx, a_now, da_now).unwrap().1.r,
        "qr_wide_r_forward_matches_fd_c64",
    );
}

#[test]
fn qr_wide_r_backward_matches_fd_c64() {
    let (a, da, cr) = qr_wide_fixture();
    check_backward_directional_fd(
        &a,
        &da,
        &cr,
        1e-6,
        8e-5,
        |ctx, a_now| qr(ctx, a_now).unwrap().r,
        |ctx, a_now, cotangent| {
            qr_rrule(
                ctx,
                a_now,
                &QrCotangent {
                    q: None,
                    r: Some(cotangent.clone()),
                },
            )
            .unwrap()
        },
        "qr_wide_r_backward_matches_fd_c64",
    );
}

#[test]
fn lu_wide_u_forward_matches_fd_c64() {
    let (a, da, _) = lu_wide_fixture();
    check_forward_fd(
        &a,
        &da,
        1e-6,
        8e-5,
        |ctx, a_now| lu(ctx, a_now, LuPivot::Partial).unwrap().u,
        |ctx, a_now, da_now| lu_frule(ctx, a_now, da_now, LuPivot::Partial).unwrap().1.u,
        "lu_wide_u_forward_matches_fd_c64",
    );
}

#[test]
fn lu_wide_u_backward_matches_fd_c64() {
    let (a, da, cu) = lu_wide_fixture();
    check_backward_directional_fd(
        &a,
        &da,
        &cu,
        1e-6,
        8e-5,
        |ctx, a_now| lu(ctx, a_now, LuPivot::Partial).unwrap().u,
        |ctx, a_now, cotangent| {
            lu_rrule(
                ctx,
                a_now,
                &LuCotangent {
                    l: None,
                    u: Some(cotangent.clone()),
                },
                LuPivot::Partial,
            )
            .unwrap()
        },
        "lu_wide_u_backward_matches_fd_c64",
    );
}

#[test]
fn lu_tall_l_forward_matches_fd_c64() {
    let (a, da, _) = lu_tall_fixture();
    check_forward_fd(
        &a,
        &da,
        1e-6,
        8e-5,
        |ctx, a_now| lu(ctx, a_now, LuPivot::Partial).unwrap().l,
        |ctx, a_now, da_now| lu_frule(ctx, a_now, da_now, LuPivot::Partial).unwrap().1.l,
        "lu_tall_l_forward_matches_fd_c64",
    );
}

#[test]
fn lu_tall_l_backward_matches_fd_c64() {
    let (a, da, cl) = lu_tall_fixture();
    check_backward_directional_fd(
        &a,
        &da,
        &cl,
        1e-6,
        8e-5,
        |ctx, a_now| lu(ctx, a_now, LuPivot::Partial).unwrap().l,
        |ctx, a_now, cotangent| {
            lu_rrule(
                ctx,
                a_now,
                &LuCotangent {
                    l: Some(cotangent.clone()),
                    u: None,
                },
                LuPivot::Partial,
            )
            .unwrap()
        },
        "lu_tall_l_backward_matches_fd_c64",
    );
}

#[test]
fn qr_complex64_r_diagonal_is_real() {
    let (a, _, _) = qr_square_fixture::<Complex64>();
    let mut ctx = CpuContext::new(1);
    let result = qr(&mut ctx, &a).unwrap();
    let r = tensor_data(&result.r);
    for i in 0..3 {
        assert!(
            r[i + i * 3].im.abs() < 1e-10,
            "expected real diagonal, got {:?} at {i}",
            r[i + i * 3]
        );
    }
}
