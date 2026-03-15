use num_complex::{Complex32, Complex64};
use tenferro::{grad, set_default_runtime, GradOptions, RuntimeContext, ScalarType, Tensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn scalar_f32(value: f32) -> DenseTensor<f32> {
    DenseTensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar_f64(value: f64) -> DenseTensor<f64> {
    DenseTensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar_c32(value: Complex32) -> DenseTensor<Complex32> {
    DenseTensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar_c64(value: Complex64) -> DenseTensor<Complex64> {
    DenseTensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_f32(values: &[f32]) -> DenseTensor<f32> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_f64(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_c32(values: &[Complex32]) -> DenseTensor<Complex32> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_c64(values: &[Complex64]) -> DenseTensor<Complex64> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn matrix_f32(values: &[f32], dims: &[usize]) -> DenseTensor<f32> {
    DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn matrix_f64(values: &[f64], dims: &[usize]) -> DenseTensor<f64> {
    DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn matrix_c64(values: &[Complex64], dims: &[usize]) -> DenseTensor<Complex64> {
    DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn grad_wrt(output: &Tensor, cotangent: &Tensor, wrt: &[&Tensor]) -> Vec<Option<Tensor>> {
    let grad_outputs = [cotangent.clone()];
    grad(
        &[output],
        wrt,
        Some(&grad_outputs),
        GradOptions {
            retain_graph: true,
            ..GradOptions::default()
        },
    )
    .unwrap()
}

macro_rules! primal {
    ($tensor:expr) => {
        Tensor::from_tensor($tensor)
    };
}

macro_rules! reverse {
    ($tensor:expr) => {{
        let mut tensor = Tensor::from_tensor($tensor);
        tensor.set_requires_grad(true).unwrap();
        tensor
    }};
}

fn make_f32_matrix() -> Tensor {
    primal!(matrix_f32(&[4.0, 1.0, 1.0, 3.0], &[2, 2]))
}

fn make_f64_matrix() -> Tensor {
    primal!(matrix_f64(&[4.0, 1.0, 1.0, 3.0], &[2, 2]))
}

fn make_f32_triangular() -> Tensor {
    primal!(matrix_f32(&[2.0, 0.0, 1.0, 3.0], &[2, 2]))
}

fn make_f64_triangular() -> Tensor {
    primal!(matrix_f64(&[2.0, 0.0, 1.0, 3.0], &[2, 2]))
}

fn make_c64_triangular() -> Tensor {
    primal!(matrix_c64(
        &[
            Complex64::new(2.0, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -0.25),
            Complex64::new(3.0, 0.75),
        ],
        &[2, 2],
    ))
}

fn make_f32_rhs() -> Tensor {
    primal!(vector_f32(&[1.0, 2.0]))
}

fn make_f64_rhs() -> Tensor {
    primal!(vector_f64(&[1.0, 2.0]))
}

fn make_c64_rhs() -> Tensor {
    primal!(vector_c64(&[
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
    ]))
}

#[test]
fn tensor_lazy_reverse_graph_merges_operands_and_rejects_distinct_outputs() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = reverse!(scalar_f64(1.0));
    let y = reverse!(scalar_f64(2.0));
    let z = reverse!(scalar_f64(3.0));

    let out_xy = x.add(&y).unwrap();
    let shared = grad(
        &[&out_xy],
        &[&x, &y],
        None,
        GradOptions {
            retain_graph: true,
            ..GradOptions::default()
        },
    )
    .unwrap();
    assert!(shared[0].is_some());
    assert!(shared[1].is_some());

    let out_z = z.exp().unwrap();
    let err = match grad(&[&out_xy, &out_z], &[&x], None, GradOptions::default()) {
        Ok(_) => panic!("grad should reject outputs from distinct reverse graphs"),
        Err(err) => err,
    };
    assert!(matches!(err, tenferro::Error::MixedReverseTape { .. }));
}

#[test]
fn tensor_to_scalar_type_covers_all_explicit_cast_pairs() {
    let real32 = primal!(scalar_f32(1.5));
    let real64 = primal!(scalar_f64(2.5));
    let complex32 = primal!(scalar_c32(Complex32::new(3.0, -4.0)));
    let complex64 = primal!(scalar_c64(Complex64::new(-2.0, 5.0)));

    assert_eq!(
        real32
            .to_scalar_type(ScalarType::F64)
            .unwrap()
            .scalar_type(),
        ScalarType::F64
    );
    assert_eq!(
        real64
            .to_scalar_type(ScalarType::F32)
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        complex32
            .to_scalar_type(ScalarType::C64)
            .unwrap()
            .scalar_type(),
        ScalarType::C64
    );
    assert_eq!(
        complex64
            .to_scalar_type(ScalarType::C32)
            .unwrap()
            .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        real32
            .to_scalar_type(ScalarType::C32)
            .unwrap()
            .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        real64
            .to_scalar_type(ScalarType::C64)
            .unwrap()
            .scalar_type(),
        ScalarType::C64
    );
    assert_eq!(
        real32
            .to_scalar_type(ScalarType::C64)
            .unwrap()
            .scalar_type(),
        ScalarType::C64
    );
    assert_eq!(
        real64
            .to_scalar_type(ScalarType::C32)
            .unwrap()
            .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        complex32
            .to_scalar_type(ScalarType::F32)
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        complex32
            .to_scalar_type(ScalarType::F64)
            .unwrap()
            .scalar_type(),
        ScalarType::F64
    );
    assert_eq!(
        complex64
            .to_scalar_type(ScalarType::F64)
            .unwrap()
            .scalar_type(),
        ScalarType::F64
    );
    assert_eq!(
        complex64
            .to_scalar_type(ScalarType::F32)
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );
}

#[test]
fn tensor_dynamic_scalar_wrappers_cover_success_and_error_paths() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let unit_f32 = primal!(scalar_f32(0.25));
    let unit_f64 = primal!(scalar_f64(0.25));
    let acosh_f64 = primal!(scalar_f64(1.25));
    let complex32 = primal!(scalar_c32(Complex32::new(0.5, -0.25)));
    let complex64 = primal!(scalar_c64(Complex64::new(0.5, 0.25)));

    let _ = unit_f32.sqrt().unwrap();
    let _ = unit_f64.exp().unwrap();
    let _ = unit_f64.expm1().unwrap();
    let _ = unit_f64.log().unwrap();
    let _ = unit_f64.log1p().unwrap();
    let _ = unit_f64.sin().unwrap();
    let _ = unit_f64.sinh().unwrap();
    let _ = unit_f64.cos().unwrap();
    let _ = unit_f64.cosh().unwrap();
    let _ = unit_f64.tanh().unwrap();
    let _ = unit_f64.asin().unwrap();
    let _ = unit_f64.acos().unwrap();
    let _ = unit_f64.atan().unwrap();
    let _ = unit_f64.asinh().unwrap();
    let _ = acosh_f64.acosh().unwrap();
    let _ = unit_f64.atanh().unwrap();
    let _ = complex32.exp().unwrap();
    let _ = complex64.sqrt().unwrap();

    let _ = primal!(vector_f32(&[1.0, 3.0])).mean().unwrap();
    let _ = primal!(vector_f32(&[1.0, 3.0])).var().unwrap();
    let _ = primal!(vector_f64(&[1.0, 3.0])).std().unwrap();

    let _ = unit_f32.add(&unit_f32).unwrap();
    let _ = unit_f64.pow(&unit_f64).unwrap();
    let _ = unit_f32.atan2(&unit_f32).unwrap();
    let _ = unit_f64.hypot(&unit_f64).unwrap();
    let _ = unit_f32.add(&complex32).unwrap();
    let _ = unit_f64.add(&complex64).unwrap();

    for result in [
        complex32.var(),
        complex64.std(),
        complex32.atan2(&complex32),
        complex64.hypot(&complex64),
    ] {
        let err = match result {
            Ok(_) => panic!("complex inputs should be rejected by real-only wrappers"),
            Err(err) => err,
        };
        assert!(
            matches!(err, tenferro::Error::InvalidAdTensor { message } if message.contains("requires real-valued"))
        );
    }
}

#[test]
fn tensor_dynamic_tensor_wrappers_cover_all_variants_and_errors() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let _ = primal!(vector_f32(&[1.0, 2.0])).sum().unwrap();
    let _ = primal!(vector_f64(&[1.0, 2.0])).sum().unwrap();
    let _ = primal!(vector_c32(&[Complex32::new(1.0, 0.5)]))
        .sum()
        .unwrap();
    let _ = primal!(vector_c64(&[Complex64::new(1.0, -0.5)]))
        .sum()
        .unwrap();

    let dot_f32 = Tensor::einsum(
        "i,i->",
        &[
            &primal!(vector_f32(&[1.0, 2.0])),
            &primal!(vector_f32(&[3.0, 4.0])),
        ],
    )
    .unwrap();
    assert_eq!(dot_f32.scalar_type(), ScalarType::F32);

    let dot_c32 = Tensor::einsum(
        "i,i->",
        &[
            &primal!(vector_f32(&[1.0, 2.0])),
            &primal!(vector_c32(&[
                Complex32::new(1.0, 0.5),
                Complex32::new(-2.0, 1.0),
            ])),
        ],
    )
    .unwrap();
    assert_eq!(dot_c32.scalar_type(), ScalarType::C32);

    let dot_c64 = Tensor::einsum(
        "i,i->",
        &[
            &primal!(vector_f64(&[1.0, 2.0])),
            &primal!(vector_c64(&[
                Complex64::new(1.0, 0.5),
                Complex64::new(-2.0, 1.0),
            ])),
        ],
    )
    .unwrap();
    assert_eq!(dot_c64.scalar_type(), ScalarType::C64);

    let err = match Tensor::einsum("->", &[]) {
        Ok(_) => panic!("einsum should reject an empty operand list"),
        Err(err) => err,
    };
    assert!(
        matches!(err, tenferro::Error::InvalidAdTensor { message } if message.contains("at least one operand"))
    );
}

fn exercise_real_linalg_suite(matrix: &Tensor, triangular: &Tensor, rhs: &Tensor) {
    let svd = matrix.svd().unwrap();
    assert_eq!(svd.s.dims(), &[2]);
    let qr = matrix.qr().unwrap();
    assert_eq!(qr.q.dims(), &[2, 2]);
    let lu = matrix.lu().unwrap();
    assert_eq!(lu.l.dims(), &[2, 2]);
    let eigen = matrix.eigen().unwrap();
    assert_eq!(eigen.values.dims(), &[2]);
    let eig = matrix.eig().unwrap();
    assert_eq!(eig.values.dims(), &[2]);
    let chol = matrix.cholesky().unwrap();
    assert_eq!(chol.dims(), &[2, 2]);
    let solve = matrix.solve(rhs).unwrap();
    assert_eq!(solve.dims(), &[2]);
    let solve_triangular = triangular.solve_triangular(rhs).unwrap();
    assert_eq!(solve_triangular.dims(), &[2]);
    let det = matrix.det().unwrap();
    assert_eq!(det.dims(), &[]);
    let slogdet = matrix.slogdet().unwrap();
    assert_eq!(slogdet.sign.dims(), &[]);
    let inv = matrix.inv().unwrap();
    assert_eq!(inv.dims(), &[2, 2]);
    let pinv = matrix.pinv().unwrap();
    assert_eq!(pinv.dims(), &[2, 2]);
    let expm = matrix.matrix_exp().unwrap();
    assert_eq!(expm.dims(), &[2, 2]);
    let norm = matrix.norm().unwrap();
    assert_eq!(norm.dims(), &[]);
    let lstsq = matrix.lstsq(rhs).unwrap();
    assert_eq!(lstsq.x.dims(), &[2]);
}

#[test]
fn tensor_dynamic_linalg_wrappers_cover_success_and_error_paths() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let matrix32 = make_f32_matrix();
    let tri32 = make_f32_triangular();
    let rhs32 = make_f32_rhs();
    exercise_real_linalg_suite(&matrix32, &tri32, &rhs32);

    let matrix64 = make_f64_matrix();
    let tri64 = make_f64_triangular();
    let rhs64 = make_f64_rhs();
    exercise_real_linalg_suite(&matrix64, &tri64, &rhs64);

    let tri_c64 = make_c64_triangular();
    let rhs_c64 = make_c64_rhs();
    let solved = tri_c64.solve_triangular(&rhs_c64).unwrap();
    assert_eq!(solved.scalar_type(), ScalarType::C64);

    let complex_err = match primal!(matrix_c64(
        &[
            Complex64::new(4.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(1.0, 0.25),
            Complex64::new(3.0, 1.0),
        ],
        &[2, 2],
    ))
    .svd()
    {
        Ok(_) => panic!("complex svd should stay rejected by the dynamic wrapper"),
        Err(err) => err,
    };
    assert!(
        matches!(complex_err, tenferro::Error::InvalidAdTensor { message } if message.contains("requires a real Tensor input"))
    );

    let mismatch_err = match matrix32.solve(&rhs64) {
        Ok(_) => panic!("solve should reject mixed dtypes"),
        Err(err) => err,
    };
    assert!(
        matches!(mismatch_err, tenferro::Error::InvalidAdTensor { message } if message.contains("requires matching dtypes"))
    );

    let lstsq_err = match matrix64.lstsq(&rhs32) {
        Ok(_) => panic!("lstsq should reject mixed dtypes"),
        Err(err) => err,
    };
    assert!(
        matches!(lstsq_err, tenferro::Error::InvalidAdTensor { message } if message.contains("requires matching dtypes"))
    );
}

#[test]
fn tensor_dynamic_pullback_wrapper_covers_success_and_error_paths() {
    let x = reverse!(vector_f64(&[1.0, 2.0]));
    let alpha = reverse!(scalar_f64(3.0));
    let out = x.scale(&alpha).unwrap();
    let cotangent = primal!(vector_f64(&[0.5, 1.25]));

    let disconnected = primal!(scalar_f64(7.0));
    let grads = grad_wrt(&out, &cotangent, &[&x, &disconnected]);
    assert!(grads[0].is_some());
    assert!(grads[1].is_none());

    let cotangent_mismatch = primal!(vector_c64(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ]));
    let err = match grad(
        &[&out],
        &[&x],
        Some(&[cotangent_mismatch.clone()]),
        GradOptions::default(),
    ) {
        Ok(_) => panic!("pullback should reject mismatched cotangent dtypes"),
        Err(err) => err,
    };
    assert!(
        matches!(err, tenferro::Error::InvalidAdTensor { message } if message.contains("requires cotangent dtype"))
    );

    let other = reverse!(scalar_f64(2.0));
    let grads = grad_wrt(&out, &cotangent, &[&other]);
    assert!(grads[0].is_none());

    let primal = primal!(vector_f64(&[1.0, 2.0]));
    let err = match grad(
        &[&primal],
        &[&primal],
        Some(&[cotangent.clone()]),
        GradOptions::default(),
    ) {
        Ok(_) => panic!("primal outputs should not expose reverse pullback"),
        Err(err) => err,
    };
    assert!(
        matches!(err, tenferro::Error::InvalidAdTensor { message } if message.contains("reverse-mode output tensor"))
    );
}

#[test]
fn tensor_dynamic_pullback_wrapper_covers_all_dtype_variants() {
    fn exercise_scale_pullback(
        x: Tensor,
        alpha: Tensor,
        cotangent: Tensor,
        expected_dtype: ScalarType,
    ) {
        let out = x.scale(&alpha).unwrap();
        let grads = grad_wrt(&out, &cotangent, &[&x, &alpha]);
        assert_eq!(grads[0].as_ref().unwrap().scalar_type(), expected_dtype);
        assert_eq!(grads[1].as_ref().unwrap().scalar_type(), expected_dtype);
    }

    let x_f32 = reverse!(vector_f32(&[1.0, 2.0]));
    let alpha_f32 = reverse!(scalar_f32(3.0));
    let cotangent_f32 = primal!(vector_f32(&[0.5, 1.25]));
    exercise_scale_pullback(x_f32, alpha_f32, cotangent_f32, ScalarType::F32);

    let x_c32 = reverse!(vector_c32(&[
        Complex32::new(1.0, 0.5),
        Complex32::new(-2.0, 1.0),
    ]));
    let alpha_c32 = reverse!(scalar_c32(Complex32::new(0.5, -1.0)));
    let cotangent_c32 = primal!(vector_c32(&[
        Complex32::new(0.25, -0.5),
        Complex32::new(1.0, 0.75),
    ]));
    exercise_scale_pullback(x_c32, alpha_c32, cotangent_c32, ScalarType::C32);

    let x_c64 = reverse!(vector_c64(&[
        Complex64::new(1.0, -0.5),
        Complex64::new(2.0, 1.5),
    ]));
    let alpha_c64 = reverse!(scalar_c64(Complex64::new(-1.5, 0.25)));
    let cotangent_c64 = primal!(vector_c64(&[
        Complex64::new(0.5, 0.0),
        Complex64::new(-0.75, 1.25),
    ]));
    exercise_scale_pullback(x_c64, alpha_c64, cotangent_c64, ScalarType::C64);
}
