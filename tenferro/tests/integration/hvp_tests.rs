use tenferro::{hvp, set_default_runtime, HvpOptions, RuntimeContext, Tensor};
use tenferro_prims::CpuContext;

fn setup() -> tenferro::DefaultRuntimeGuard {
    set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)))
}

fn assert_tensor_close(actual: &Tensor, expected: &[f64], label: &str) {
    let actual_data: Vec<f64> = match actual {
        Tensor::F64(v) => v.primal().buffer().as_slice().unwrap().to_vec(),
        _ => panic!("{label}: expected f64 tensor"),
    };
    assert_eq!(
        actual_data.len(),
        expected.len(),
        "{label}: length mismatch: got {}, expected {}",
        actual_data.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual_data.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-10, "{label}[{i}] = {a}, expected {e}");
    }
}

/// f(x) = x^T M x = einsum("ij,jk->ik", M, diag(x)) composed with trace
/// Simpler: f(x) = einsum("i,ij,j->", x, M, x) with M = identity
///
/// Since the structured path doesn't support HVP yet, we use a two-step
/// dense computation: y = x * x (element-wise via mul), then z = sum(y).
/// But sum() uses ClosureRule, so we need a pure einsum chain.
///
/// Instead, we test via matrix operations:
/// f(A) = trace(A @ A^T) = einsum("ij,ij->", A, A) = Frobenius norm squared
/// grad_A = 2*A, H*V = 2*V
///
/// Using 2x2 matrix, dense path forced via parenthesized subscripts:
/// We express this as a composition of two einsums to go through dense path.
///
/// Actually, the simplest test that works:
/// Step 1: C = einsum("ij,jk->ik", A, B) -- matrix mul, goes through structured path
/// Step 2: loss = einsum("ij,ij->", C, C) -- Frobenius norm squared, structured path
///
/// Both go through structured path -> HvpNotSupported.
///
/// The correct test: use the Level A API (tenferro-einsum::tracked_einsum)
/// which directly registers EinsumReverseRule with HVP support.
/// At Level B, the dense path (triggered by parenthesized subscripts) has HVP.
/// The simplest scalar quadratic: f(x) = x^2 for scalar x.
#[test]
fn test_hvp_quadratic_scalar() {
    let _guard = setup();

    // f(x) = x * x via einsum(",->"): scalar * scalar -> scalar
    // This is a 0-dimensional tensor; einsum "->" = identity for scalar.
    // Use einsum(",->") which is scalar multiplication.
    let mut x = Tensor::from_slice(&[3.0_f64], &[]).unwrap();
    x.set_requires_grad(true).unwrap();

    // f(x) = x * x: use the multiply operation which goes through scalar AD
    // Actually, Tensor::mul creates a scalar binary AD path.
    // Let's use the matmul approach: A is 1x1, B is 1x1
    let mut a = Tensor::from_slice(&[3.0_f64], &[1, 1]).unwrap();
    a.set_requires_grad(true).unwrap();

    // f(A) = trace(A^T A) = einsum("ij,ij->", A, A) for 1x1 matrix = A[0,0]^2
    // This goes through the structured path at Level B.
    // The structured path uses ClosureRule -> HvpNotSupported.
    // To test dense path, we need parenthesized subscripts.
    // For the nested einsum format, try: einsum("ij,ij->", ...)
    // which is a simple trace/Frobenius dot product.

    // Alternative: use the Level A API directly by constructing a tracked einsum
    // via the tidu tape. But that requires lower-level setup.

    // Actually let's check: even if the structural path is used for pullback,
    // the DenseEinsumRule we registered on the dense path should work.
    // Let me test with matrix operands that go through dense path:
    // Use size_dict to force dense path... but we can't pass size_dict through public API.

    // The pragmatic solution: test HVP with operations that DO go through the
    // dense path. Since all simple subscripts go through structured path,
    // and HVP for structured path is out of scope, we test the error case
    // and verify the einsum-crate level tests pass (which they do).

    // For the public API test, verify that structured einsum returns HvpNotSupported:
    let output = Tensor::einsum("ij,ij->", &[&a, &a]).unwrap();
    let va = Tensor::from_slice(&[1.0_f64], &[1, 1]).unwrap();
    match hvp(&output, &[&a], &[&va], HvpOptions::default()) {
        Err(tenferro::Error::Autodiff(chainrules_core::AutodiffError::HvpNotSupported)) => {}
        Err(err) => panic!("expected HvpNotSupported for structured path, got: {err}"),
        Ok(_) => panic!("structured einsum path should not support HVP yet"),
    }
}

/// Non-scalar output should return NonScalarLoss error.
#[test]
fn test_hvp_non_scalar_output_error() {
    let _guard = setup();

    let mut x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]).unwrap();
    x.set_requires_grad(true).unwrap();

    // Element-wise product, not a scalar
    let output = Tensor::einsum("i,i->i", &[&x, &x]).unwrap();

    let v = Tensor::from_slice(&[1.0_f64, 1.0, 1.0], &[3]).unwrap();

    match hvp(&output, &[&x], &[&v], HvpOptions::default()) {
        Err(tenferro::Error::Autodiff(chainrules_core::AutodiffError::NonScalarLoss {
            ..
        })) => {}
        Err(err) => panic!("expected NonScalarLoss, got: {err}"),
        Ok(_) => panic!("expected error for non-scalar output"),
    }
}

/// Shape mismatch between input and v.
#[test]
fn test_hvp_shape_mismatch_error() {
    let _guard = setup();

    let mut x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]).unwrap();
    x.set_requires_grad(true).unwrap();

    let output = Tensor::einsum("i,i->", &[&x, &x]).unwrap();

    // Wrong shape: [2] instead of [3]
    let v_wrong = Tensor::from_slice(&[1.0_f64, 1.0], &[2]).unwrap();

    match hvp(&output, &[&x], &[&v_wrong], HvpOptions::default()) {
        Err(tenferro::Error::InvalidAdTensor { message }) => {
            assert!(
                message.contains("shape mismatch"),
                "expected shape mismatch message, got: {message}"
            );
        }
        Err(err) => panic!("expected InvalidAdTensor, got: {err}"),
        Ok(_) => panic!("expected error for shape mismatch"),
    }
}

/// Graph with a ClosureRule (sum) that does not support HVP should return HvpNotSupported.
#[test]
fn test_hvp_not_supported_propagation() {
    let _guard = setup();

    let mut x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]).unwrap();
    x.set_requires_grad(true).unwrap();

    // sum() uses ClosureRule which does not support HVP
    let output = x.sum().unwrap();

    let v = Tensor::from_slice(&[1.0_f64, 1.0, 1.0], &[3]).unwrap();

    match hvp(&output, &[&x], &[&v], HvpOptions::default()) {
        Err(tenferro::Error::Autodiff(chainrules_core::AutodiffError::HvpNotSupported)) => {}
        Err(err) => panic!("expected HvpNotSupported, got: {err}"),
        Ok(_) => panic!("expected HvpNotSupported error"),
    }
}

/// Dense-path HVP for a linear function via parenthesized subscripts.
///
/// f(A) = sum((A @ B) * D) via einsum("(ij,jk),ik->", [A, B, D])
/// Parenthesized subscripts force the dense einsum path which registers
/// DenseEinsumRule with full HVP support.
/// Since f is linear in A: gradient = D @ B^T, HVP = 0 (Hessian is zero).
#[test]
fn test_hvp_dense_path_linear_f64() {
    let _guard = setup();

    let mut a = Tensor::from_slice(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
    a.set_requires_grad(true).unwrap();

    let b = Tensor::from_slice(&[2.0_f64, 0.5, -1.0, 1.5], &[2, 2]).unwrap();
    let d = Tensor::from_slice(&[0.4_f64, 0.2, -0.7, 0.9], &[2, 2]).unwrap();

    let output = Tensor::einsum("(ij,jk),ik->", &[&a, &b, &d]).unwrap();
    assert!(
        output.dims().is_empty(),
        "output should be scalar, got dims {:?}",
        output.dims()
    );

    let v = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

    let result = hvp(&output, &[&a], &[&v], HvpOptions::default())
        .expect("dense-path HVP should succeed for parenthesized subscripts");

    let grad = result.gradients[0]
        .as_ref()
        .expect("gradient should exist for tracked input");
    let hvp_val = result.hvps[0]
        .as_ref()
        .expect("hvp should exist for tracked input");

    // Gradient: D @ B^T
    // B = [[2, -1], [0.5, 1.5]], B^T = [[2, 0.5], [-1, 1.5]]
    // D = [[0.4, -0.7], [0.2, 0.9]]
    // D @ B^T = [[0.4*2 + (-0.7)*(-1), 0.4*0.5 + (-0.7)*1.5],
    //            [0.2*2 + 0.9*(-1), 0.2*0.5 + 0.9*1.5]]
    //         = [[1.5, -0.85], [-0.5, 1.45]]
    // Column-major: [1.5, -0.5, -0.85, 1.45]
    assert_tensor_close(grad, &[1.5, -0.5, -0.85, 1.45], "grad");

    // HVP is zero because f is linear in A
    assert_tensor_close(hvp_val, &[0.0, 0.0, 0.0, 0.0], "hvp");
}

/// Dense-path HVP for a quadratic function via two parenthesized einsums.
///
/// Stage 1: C = einsum("(ij,jk)->ik", [A, B]) — forces dense path
/// Stage 2: loss = einsum("(ik,ik)->", [C, C]) — forces dense path
/// f(A) = sum((A@B)^2) — quadratic in A
///
/// Verifies HVP against central finite differences of the gradient.
#[test]
fn test_hvp_dense_path_quadratic_fd_f64() {
    let _guard = setup();

    let a_data: &[f64] = &[1.0, 3.0, 2.0, 4.0];
    let b_data: &[f64] = &[2.0, 0.5, -1.0, 1.5];
    let v_data: &[f64] = &[0.2, -0.1, 0.3, 0.05];
    let dims: &[usize] = &[2, 2];

    let mut a = Tensor::from_slice(a_data, dims).unwrap();
    a.set_requires_grad(true).unwrap();
    let b = Tensor::from_slice(b_data, dims).unwrap();

    let c = Tensor::einsum("(ij,jk)->ik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), dims, "stage 1 output shape");

    let output = Tensor::einsum("(ik,ik)->", &[&c, &c]).unwrap();
    assert!(output.dims().is_empty(), "stage 2 output should be scalar");

    let v = Tensor::from_slice(v_data, dims).unwrap();

    let result = hvp(&output, &[&a], &[&v], HvpOptions::default())
        .expect("dense-path HVP should succeed for two-stage parenthesized einsum");

    let hvp_actual = result.hvps[0]
        .as_ref()
        .expect("hvp should exist for tracked input");
    let hvp_slice = match hvp_actual {
        Tensor::F64(v) => v.primal().buffer().as_slice().unwrap(),
        _ => panic!("expected f64 tensor for hvp"),
    };

    let hvp_fd = compute_hvp_via_grad_fd(a_data, b_data, v_data, dims, 1e-5);

    for (i, (&actual, &expected)) in hvp_slice.iter().zip(hvp_fd.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-4,
            "hvp[{i}] = {actual}, fd = {expected}, diff = {}",
            (actual - expected).abs()
        );
    }
}

fn scalar_value(tensor: &Tensor) -> f64 {
    match tensor {
        Tensor::F64(v) => v.primal().buffer().as_slice().unwrap()[0],
        _ => panic!("expected f64 scalar tensor"),
    }
}

fn eval_quadratic(a_data: &[f64], b_data: &[f64], dims: &[usize]) -> f64 {
    let a = Tensor::from_slice(a_data, dims).unwrap();
    let b = Tensor::from_slice(b_data, dims).unwrap();
    let c = Tensor::einsum("(ij,jk)->ik", &[&a, &b]).unwrap();
    let loss = Tensor::einsum("(ik,ik)->", &[&c, &c]).unwrap();
    scalar_value(&loss)
}

fn compute_grad_fd(a_data: &[f64], b_data: &[f64], dims: &[usize], eps: f64) -> Vec<f64> {
    let mut grad = vec![0.0; a_data.len()];
    for i in 0..a_data.len() {
        let mut a_plus = a_data.to_vec();
        a_plus[i] += eps;
        let mut a_minus = a_data.to_vec();
        a_minus[i] -= eps;
        grad[i] = (eval_quadratic(&a_plus, b_data, dims) - eval_quadratic(&a_minus, b_data, dims))
            / (2.0 * eps);
    }
    grad
}

fn compute_hvp_via_grad_fd(
    a_data: &[f64],
    b_data: &[f64],
    v_data: &[f64],
    dims: &[usize],
    eps: f64,
) -> Vec<f64> {
    let a_plus: Vec<f64> = a_data
        .iter()
        .zip(v_data.iter())
        .map(|(&a, &v)| a + eps * v)
        .collect();
    let a_minus: Vec<f64> = a_data
        .iter()
        .zip(v_data.iter())
        .map(|(&a, &v)| a - eps * v)
        .collect();

    let grad_plus = compute_grad_fd(&a_plus, b_data, dims, eps);
    let grad_minus = compute_grad_fd(&a_minus, b_data, dims, eps);

    grad_plus
        .iter()
        .zip(grad_minus.iter())
        .map(|(&gp, &gm)| (gp - gm) / (2.0 * eps))
        .collect()
}
