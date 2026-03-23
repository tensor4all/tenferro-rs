use num_complex::{Complex32, Complex64};
use tenferro::{forward_ad, set_default_runtime, RuntimeContext, ScalarType, Tensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn dense_tensor_data<T: tenferro_algebra::Scalar + Copy>(tensor: &DenseTensor<T>) -> Vec<T> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn matrix_f32(values: &[f32], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap())
}

fn matrix_f64(values: &[f64], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap())
}

fn matrix_c32(values: &[Complex32], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap())
}

fn matrix_c64(values: &[Complex64], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap())
}

fn vector_f32(values: &[f32]) -> Tensor {
    Tensor::from_tensor(
        DenseTensor::<f32>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap(),
    )
}

fn vector_f64(values: &[f64]) -> Tensor {
    Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap(),
    )
}

fn vector_c32(values: &[Complex32]) -> Tensor {
    Tensor::from_tensor(
        DenseTensor::<Complex32>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor)
            .unwrap(),
    )
}

fn vector_c64(values: &[Complex64]) -> Tensor {
    Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor)
            .unwrap(),
    )
}

fn tensor_f64(values: &[f64], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap())
}

fn tensor_f32(values: &[f32], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap())
}

fn tensor_c32(values: &[Complex32], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap())
}

#[test]
fn complex_forward_wrappers_cover_supported_svd_and_solve_paths() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = matrix_c64(
        &[
            Complex64::new(4.0, 0.5),
            Complex64::new(1.0, -0.25),
            Complex64::new(1.0, 0.25),
            Complex64::new(3.0, 1.0),
        ],
        &[2, 2],
    );
    let da = matrix_c64(
        &[
            Complex64::new(0.1, 0.0),
            Complex64::new(-0.2, 0.05),
            Complex64::new(0.3, -0.1),
            Complex64::new(-0.4, 0.2),
        ],
        &[2, 2],
    );
    let b = vector_c64(&[Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)]);
    let db = vector_c64(&[Complex64::new(0.2, -0.1), Complex64::new(-0.3, 0.4)]);

    let (svd_primal, svd_tangent) = forward_ad::dual_level(|fw| {
        let dual_a = fw.make_dual(&a, &da)?;
        let singular_values = dual_a.svd()?.s.sum()?;
        fw.unpack_dual(&singular_values)
    })
    .unwrap();
    assert_eq!(svd_primal.scalar_type(), ScalarType::F64);
    assert_eq!(svd_tangent.unwrap().scalar_type(), ScalarType::F64);

    let (solve_primal, solve_tangent) = forward_ad::dual_level(|fw| {
        let dual_a = fw.make_dual(&a, &da)?;
        let dual_b = fw.make_dual(&b, &db)?;
        let solution = dual_a.solve(&dual_b)?;
        fw.unpack_dual(&solution)
    })
    .unwrap();
    assert_eq!(solve_primal.scalar_type(), ScalarType::C64);
    assert_eq!(solve_tangent.unwrap().scalar_type(), ScalarType::C64);
}

#[test]
fn tensor_frontend_exposes_missing_primal_linalg_wrappers() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let matrix = matrix_f64(&[4.0, 1.0, 1.0, 3.0], &[2, 2]);
    let rhs = vector_f64(&[1.0, 2.0]);

    let lu_factor = matrix.lu_factor().unwrap();
    assert_eq!(lu_factor.factors.dims(), &[2, 2]);
    assert_eq!(dense_tensor_data(&lu_factor.pivots).len(), 2);

    let lu_factor_ex = matrix.lu_factor_ex().unwrap();
    assert_eq!(lu_factor_ex.factors.dims(), &[2, 2]);
    assert_eq!(dense_tensor_data(&lu_factor_ex.pivots).len(), 2);
    assert_eq!(dense_tensor_data(&lu_factor_ex.info), vec![0]);

    let lu_solved = lu_factor.factors.lu_solve(&rhs, &lu_factor.pivots).unwrap();
    assert_eq!(lu_solved.dims(), &[2]);

    let solve_ex = matrix.solve_ex(&rhs).unwrap();
    assert_eq!(solve_ex.solution.dims(), &[2]);
    assert_eq!(dense_tensor_data(&solve_ex.info), vec![0]);

    let inv_ex = matrix.inv_ex().unwrap();
    assert_eq!(inv_ex.inverse.dims(), &[2, 2]);
    assert_eq!(dense_tensor_data(&inv_ex.info), vec![0]);

    let cholesky_ex = matrix.cholesky_ex().unwrap();
    assert_eq!(cholesky_ex.l.dims(), &[2, 2]);
    assert_eq!(dense_tensor_data(&cholesky_ex.info), vec![0]);

    let squared = matrix.matrix_power(2).unwrap();
    assert_eq!(squared.dims(), &[2, 2]);

    let cond = matrix.cond().unwrap();
    assert_eq!(cond.dims(), &[]);

    let cross = vector_f64(&[1.0, 0.0, 0.0])
        .cross(&vector_f64(&[0.0, 1.0, 0.0]))
        .unwrap();
    assert_eq!(cross.dims(), &[3]);

    let reflectors = matrix_f64(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let tau = vector_f64(&[2.0]);
    let householder = reflectors.householder_product(&tau).unwrap();
    assert_eq!(householder.dims(), &[2, 2]);

    let x = vector_f64(&[2.0, 3.0]);
    let default_vander = x.vander().unwrap();
    assert_eq!(default_vander.dims(), &[2, 2]);
    let custom_vander = x.vander_with(Some(4), true).unwrap();
    assert_eq!(custom_vander.dims(), &[2, 4]);

    let tensorized = tensor_f64(
        &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[2, 2, 2, 2],
    );
    let tensor_inverse = tensorized.tensorinv(2).unwrap();
    assert_eq!(tensor_inverse.dims(), &[2, 2, 2, 2]);

    let tensor_rhs = matrix_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let tensor_solve = tensorized.tensorsolve(&tensor_rhs).unwrap();
    assert_eq!(tensor_solve.dims(), &[2, 2]);
    let reordered_tensor_solve = tensorized
        .tensorsolve_with_dims(&tensor_rhs, &[3, 2])
        .unwrap();
    assert_eq!(reordered_tensor_solve.dims(), &[2, 2]);
}

#[test]
fn complex_primal_frontend_support_extends_beyond_svd_and_qr() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let hermitian = matrix_c64(
        &[
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(1.0, 0.5),
            Complex64::new(3.0, 0.0),
        ],
        &[2, 2],
    );
    let rhs = vector_c64(&[Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)]);

    let lu = hermitian.lu().unwrap();
    assert_eq!(lu.l.scalar_type(), ScalarType::C64);

    let eigen = hermitian.eigen().unwrap();
    assert_eq!(eigen.values.scalar_type(), ScalarType::F64);
    assert_eq!(eigen.vectors.scalar_type(), ScalarType::C64);

    let cholesky = hermitian.cholesky().unwrap();
    assert_eq!(cholesky.scalar_type(), ScalarType::C64);

    let solved = hermitian.solve(&rhs).unwrap();
    assert_eq!(solved.scalar_type(), ScalarType::C64);

    let inverse = hermitian.inv().unwrap();
    assert_eq!(inverse.scalar_type(), ScalarType::C64);

    let matrix_exp = hermitian.matrix_exp().unwrap();
    assert_eq!(matrix_exp.scalar_type(), ScalarType::C64);

    let lu_factor = hermitian.lu_factor().unwrap();
    assert_eq!(lu_factor.factors.scalar_type(), ScalarType::C64);

    let solve_ex = hermitian.solve_ex(&rhs).unwrap();
    assert_eq!(solve_ex.solution.scalar_type(), ScalarType::C64);

    let inv_ex = hermitian.inv_ex().unwrap();
    assert_eq!(inv_ex.inverse.scalar_type(), ScalarType::C64);

    let matrix_power = hermitian.matrix_power(2).unwrap();
    assert_eq!(matrix_power.scalar_type(), ScalarType::C64);
}

#[test]
fn tensor_frontend_extra_wrappers_cover_remaining_f32_c32_and_error_paths() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let matrix32 = matrix_f32(&[4.0, 1.0, 1.0, 3.0], &[2, 2]);
    let rhs32 = vector_f32(&[1.0, 2.0]);
    let lu_factor32 = matrix32.lu_factor().unwrap();
    assert_eq!(lu_factor32.factors.scalar_type(), ScalarType::F32);
    assert_eq!(
        matrix32.lu_factor_ex().unwrap().factors.scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        lu_factor32
            .factors
            .lu_solve(&rhs32, &lu_factor32.pivots)
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        matrix32.solve_ex(&rhs32).unwrap().solution.scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        matrix32.inv_ex().unwrap().inverse.scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        matrix32.cholesky_ex().unwrap().l.scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        matrix32.matrix_power(3).unwrap().scalar_type(),
        ScalarType::F32
    );
    assert_eq!(matrix32.cond().unwrap().scalar_type(), ScalarType::F32);
    assert_eq!(
        vector_f32(&[1.0, 0.0, 0.0])
            .cross(&vector_f32(&[0.0, 1.0, 0.0]))
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        matrix_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2])
            .householder_product(&vector_f32(&[2.0]))
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        vector_f32(&[2.0, 3.0]).vander().unwrap().scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        vector_f32(&[2.0, 3.0])
            .vander_with(Some(3), true)
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );
    let tensorized32 = tensor_f32(
        &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[2, 2, 2, 2],
    );
    assert_eq!(
        tensorized32.tensorinv(2).unwrap().scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        tensorized32
            .tensorsolve(&matrix_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );
    assert_eq!(
        tensorized32
            .tensorsolve_with_dims(&matrix_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]), &[3, 2])
            .unwrap()
            .scalar_type(),
        ScalarType::F32
    );

    let hermitian32 = matrix_c32(
        &[
            Complex32::new(4.0, 0.0),
            Complex32::new(1.0, -0.5),
            Complex32::new(1.0, 0.5),
            Complex32::new(3.0, 0.0),
        ],
        &[2, 2],
    );
    let rhs_c32 = vector_c32(&[Complex32::new(1.0, 0.5), Complex32::new(-2.0, 1.0)]);
    let lu_factor_c32 = hermitian32.lu_factor().unwrap();
    assert_eq!(lu_factor_c32.factors.scalar_type(), ScalarType::C32);
    assert_eq!(
        hermitian32.lu_factor_ex().unwrap().factors.scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        lu_factor_c32
            .factors
            .lu_solve(&rhs_c32, &lu_factor_c32.pivots)
            .unwrap()
            .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        hermitian32
            .solve_ex(&rhs_c32)
            .unwrap()
            .solution
            .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        hermitian32.inv_ex().unwrap().inverse.scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        hermitian32.cholesky_ex().unwrap().l.scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        hermitian32.matrix_power(3).unwrap().scalar_type(),
        ScalarType::C32
    );
    let cond_err = hermitian32.cond().unwrap_err();
    assert!(matches!(cond_err, tenferro::Error::InvalidAdTensor { .. }));
    assert_eq!(
        vector_c32(&[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(0.0, 0.0),
        ])
        .cross(&vector_c32(&[
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
        ]))
        .unwrap()
        .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        matrix_c32(
            &[
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
            ],
            &[2, 2],
        )
        .householder_product(&vector_c32(&[Complex32::new(2.0, 0.0)]))
        .unwrap()
        .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        vector_c32(&[Complex32::new(2.0, 0.0), Complex32::new(3.0, 1.0)])
            .vander()
            .unwrap()
            .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        vector_c32(&[Complex32::new(2.0, 0.0), Complex32::new(3.0, 1.0)])
            .vander_with(Some(3), true)
            .unwrap()
            .scalar_type(),
        ScalarType::C32
    );
    let tensorized_c32 = tensor_c32(
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
        ],
        &[2, 2, 2, 2],
    );
    assert_eq!(
        tensorized_c32.tensorinv(2).unwrap().scalar_type(),
        ScalarType::C32
    );
    let tensor_rhs_c32 = matrix_c32(
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
        ],
        &[2, 2],
    );
    assert_eq!(
        tensorized_c32
            .tensorsolve(&tensor_rhs_c32)
            .unwrap()
            .scalar_type(),
        ScalarType::C32
    );
    assert_eq!(
        tensorized_c32
            .tensorsolve_with_dims(&tensor_rhs_c32, &[3, 2])
            .unwrap()
            .scalar_type(),
        ScalarType::C32
    );

    assert!(matrix32.solve_ex(&rhs_c32).is_err());
    assert!(lu_factor32
        .factors
        .lu_solve(&rhs_c32, &lu_factor32.pivots)
        .is_err());
    assert!(vector_f32(&[1.0, 0.0, 0.0])
        .cross(&vector_c32(&[
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
        ]))
        .is_err());
    assert!(matrix32
        .householder_product(&vector_c32(&[Complex32::new(2.0, 0.0)]))
        .is_err());
    assert!(tensorized32.tensorsolve(&tensor_rhs_c32).is_err());
}
