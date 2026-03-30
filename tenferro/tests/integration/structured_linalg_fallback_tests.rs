use tenferro::{set_default_runtime, Error, RuntimeContext, Tensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn vector(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn diag(values: &[f64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector(values))).unwrap()
}

#[test]
fn structured_diag_qr_primal_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = diag(&[1.0, 2.0]);

    let err = match x.qr() {
        Ok(_) => panic!("structured qr should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "qr"));
}

#[test]
fn structured_diag_inv_primal_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = diag(&[2.0, 4.0]);

    let err = match x.inv() {
        Ok(_) => panic!("structured inv should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "inv"));
}

#[test]
fn structured_diag_solve_primal_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = diag(&[2.0, 4.0]);
    let b = Tensor::from_tensor(vector(&[6.0, 8.0]));

    let err = match a.solve(&b) {
        Ok(_) => panic!("structured solve should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "solve"));
}

#[test]
fn structured_diag_extra_primal_wrappers_reject_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = diag(&[2.0, 4.0]);
    let b = Tensor::from_tensor(vector(&[6.0, 8.0]));
    let pivots = DenseTensor::<i32>::from_slice(&[1, 2], &[2], MemoryOrder::ColumnMajor).unwrap();

    let lu_factor_err = match a.lu_factor() {
        Ok(_) => panic!("structured lu_factor should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(lu_factor_err, Error::UnsupportedStructuredLinalg { op } if op == "lu_factor")
    );

    let lu_factor_ex_err = match a.lu_factor_ex() {
        Ok(_) => panic!("structured lu_factor_ex should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(lu_factor_ex_err, Error::UnsupportedStructuredLinalg { op } if op == "lu_factor_ex")
    );

    let lu_solve_err = match a.lu_solve(&b, &pivots) {
        Ok(_) => panic!("structured lu_solve should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(lu_solve_err, Error::UnsupportedStructuredLinalg { op } if op == "lu_solve"));

    let solve_ex_err = match a.solve_ex(&b) {
        Ok(_) => panic!("structured solve_ex should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(solve_ex_err, Error::UnsupportedStructuredLinalg { op } if op == "solve_ex"));

    let inv_ex_err = match a.inv_ex() {
        Ok(_) => panic!("structured inv_ex should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(inv_ex_err, Error::UnsupportedStructuredLinalg { op } if op == "inv_ex"));

    let cholesky_ex_err = match a.cholesky_ex() {
        Ok(_) => panic!("structured cholesky_ex should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(cholesky_ex_err, Error::UnsupportedStructuredLinalg { op } if op == "cholesky_ex")
    );
}

#[test]
fn structured_diag_qr_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut structured_x = diag(&[2.0, 3.0]);
    structured_x.set_requires_grad(true).unwrap();

    let err = match structured_x.qr() {
        Ok(_) => panic!("structured qr reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "qr"));
}

#[test]
fn structured_diag_inv_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut structured_x = diag(&[2.0, 4.0]);
    structured_x.set_requires_grad(true).unwrap();

    let err = match structured_x.inv() {
        Ok(_) => panic!("structured inv reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "inv"));
}

#[test]
fn structured_diag_solve_reverse_rejects_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut structured_a = diag(&[2.0, 4.0]);
    structured_a.set_requires_grad(true).unwrap();
    let mut structured_b = Tensor::from_tensor(vector(&[6.0, 8.0]));
    structured_b.set_requires_grad(true).unwrap();

    let err = match structured_a.solve(&structured_b) {
        Ok(_) => panic!("structured solve reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedStructuredLinalg { op } if op == "solve"));
}

#[test]
fn structured_diag_extra_reverse_wrappers_reject_non_dense_input() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let mut structured_a = diag(&[2.0, 4.0]);
    structured_a.set_requires_grad(true).unwrap();
    let mut structured_b = Tensor::from_tensor(vector(&[6.0, 8.0]));
    structured_b.set_requires_grad(true).unwrap();
    let pivots = DenseTensor::<i32>::from_slice(&[1, 2], &[2], MemoryOrder::ColumnMajor).unwrap();

    let lu_factor_err = match structured_a.lu_factor() {
        Ok(_) => panic!("structured lu_factor reverse should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(lu_factor_err, Error::UnsupportedStructuredLinalg { op } if op == "lu_factor")
    );

    let lu_factor_ex_err = match structured_a.lu_factor_ex() {
        Ok(_) => panic!("structured lu_factor_ex reverse should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(lu_factor_ex_err, Error::UnsupportedStructuredLinalg { op } if op == "lu_factor_ex")
    );

    let lu_solve_err = match structured_a.lu_solve(&structured_b, &pivots) {
        Ok(_) => panic!("structured lu_solve reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(lu_solve_err, Error::UnsupportedStructuredLinalg { op } if op == "lu_solve"));

    let solve_ex_err = match structured_a.solve_ex(&structured_b) {
        Ok(_) => panic!("structured solve_ex reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(solve_ex_err, Error::UnsupportedStructuredLinalg { op } if op == "solve_ex"));

    let inv_ex_err = match structured_a.inv_ex() {
        Ok(_) => panic!("structured inv_ex reverse should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(inv_ex_err, Error::UnsupportedStructuredLinalg { op } if op == "inv_ex"));

    let cholesky_ex_err = match structured_a.cholesky_ex() {
        Ok(_) => panic!("structured cholesky_ex reverse should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(cholesky_ex_err, Error::UnsupportedStructuredLinalg { op } if op == "cholesky_ex")
    );
}
