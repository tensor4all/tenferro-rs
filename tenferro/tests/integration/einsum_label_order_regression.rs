use num_complex::Complex64;
use tenferro::{set_default_runtime, RuntimeContext, Tensor};
use tenferro_algebra::Conjugate;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

const TOL: f64 = 1.0e-10;

fn matrix(values: &[Complex64]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, &[2, 2], MemoryOrder::RowMajor).unwrap())
}

fn tensor(values: &[Complex64], dims: &[usize]) -> Tensor {
    Tensor::from_tensor(DenseTensor::from_slice(values, dims, MemoryOrder::RowMajor).unwrap())
}

fn row_major_values(tensor: &Tensor) -> Vec<Complex64> {
    let snapshot = tensor.primal_snapshot().to_dense().unwrap();
    let dense = snapshot.payload_c64().unwrap();
    let row_major = dense.contiguous(MemoryOrder::RowMajor);
    let is_conjugated = row_major.is_conjugated();
    let offset = usize::try_from(row_major.offset()).unwrap();
    let len = row_major.len();
    let slice = row_major
        .buffer()
        .as_slice()
        .unwrap()
        .get(offset..offset + len)
        .unwrap();
    if is_conjugated {
        slice.iter().copied().map(Conjugate::conj).collect()
    } else {
        slice.to_vec()
    }
}

fn assert_tensor_close(lhs: &Tensor, rhs: &Tensor) {
    let lhs = row_major_values(lhs);
    let rhs = row_major_values(rhs);
    assert_eq!(lhs.len(), rhs.len());
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        assert!(
            (*a - *b).norm() < TOL,
            "tensor mismatch: lhs={a:?}, rhs={b:?}, diff={:?}",
            (*a - *b).norm()
        );
    }
}

fn assert_row_major_values_close(tensor: &Tensor, expected: &[Complex64]) {
    let actual = row_major_values(tensor);
    assert_eq!(actual.len(), expected.len());
    for (got, want) in actual.iter().zip(expected.iter()) {
        assert!(
            (*got - *want).norm() < TOL,
            "tensor mismatch: got={got:?}, want={want:?}, diff={:?}",
            (*got - *want).norm()
        );
    }
}

#[test]
fn einsum_matches_equivalent_rhs_axis_reordering() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let lhs = matrix(&[
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.3, 1.2),
        Complex64::new(0.7, -0.8),
        Complex64::new(2.1, 0.3),
    ]);
    let rhs = matrix(&[
        Complex64::new(0.5, -0.1),
        Complex64::new(1.0, 0.3),
        Complex64::new(-0.4, 0.9),
        Complex64::new(0.2, -0.7),
    ]);

    // Same abstract contraction, different rhs axis order presentation.
    let direct = Tensor::einsum("ab,ca->bc", &[&lhs, &rhs]).unwrap();
    let rhs_reordered = rhs.permute(&[1, 0]).unwrap();
    let normalized = Tensor::einsum("ab,ac->bc", &[&lhs, &rhs_reordered]).unwrap();

    assert_eq!(direct.dims(), &[2, 2]);
    assert_tensor_close(&direct, &normalized);
}

#[test]
fn einsum_matches_equivalent_rhs_axis_reordering_with_lazy_conjugation() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let lhs = matrix(&[
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(-0.5, 0.7),
        Complex64::new(2.0, 1.5),
    ]);
    let rhs = matrix(&[
        Complex64::new(0.5, -0.3),
        Complex64::new(1.0, 0.8),
        Complex64::new(-1.2, 0.4),
        Complex64::new(0.3, -0.9),
    ]);

    let rhs_conj = rhs.conj().unwrap();
    let direct = Tensor::einsum("ab,ca->bc", &[&lhs, &rhs_conj]).unwrap();
    let rhs_reordered = rhs_conj.permute(&[1, 0]).unwrap();
    let normalized = Tensor::einsum("ab,ac->bc", &[&lhs, &rhs_reordered]).unwrap();

    assert_eq!(direct.dims(), &[2, 2]);
    assert_tensor_close(&direct, &normalized);
}

#[test]
fn chained_einsum_matches_equivalent_rhs_axis_reordering_with_lazy_conjugation() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a0 = matrix(&[
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(-0.5, 0.7),
        Complex64::new(2.0, 1.5),
    ]);
    let b0 = matrix(&[
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(-0.5, 0.7),
        Complex64::new(2.0, 1.5),
    ]);
    let a1 = matrix(&[
        Complex64::new(0.5, -0.3),
        Complex64::new(1.0, 0.8),
        Complex64::new(-1.2, 0.4),
        Complex64::new(0.3, -0.9),
    ]);

    let a0_conj = a0.conj().unwrap();
    let env = Tensor::einsum("sa,sc->ac", &[&a0_conj, &b0]).unwrap();
    let a1_conj = a1.conj().unwrap();
    let direct = Tensor::einsum("ab,ca->bc", &[&env, &a1_conj]).unwrap();
    let rhs_reordered = a1_conj.permute(&[1, 0]).unwrap();
    let normalized = Tensor::einsum("ab,ac->bc", &[&env, &rhs_reordered]).unwrap();

    assert_eq!(direct.dims(), &[2, 2]);
    assert_tensor_close(&direct, &normalized);
}

#[test]
fn einsum_ab_ca_to_bc_matches_manual_reference() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let lhs = tensor(
        &[
            Complex64::new(1.0, 0.2),
            Complex64::new(-0.5, 0.4),
            Complex64::new(2.0, -0.1),
            Complex64::new(0.3, -1.0),
            Complex64::new(1.2, 0.7),
            Complex64::new(-0.8, 0.5),
        ],
        &[2, 3],
    );
    let rhs = tensor(
        &[
            Complex64::new(0.5, -0.1),
            Complex64::new(1.0, 0.6),
            Complex64::new(-1.2, 0.3),
            Complex64::new(0.7, -0.9),
            Complex64::new(0.2, 1.1),
            Complex64::new(-0.4, 0.8),
            Complex64::new(1.3, -0.2),
            Complex64::new(0.9, 0.5),
        ],
        &[4, 2],
    );

    let out = Tensor::einsum("ab,ca->bc", &[&lhs, &rhs]).unwrap();

    let lhs_vals = row_major_values(&lhs);
    let rhs_vals = row_major_values(&rhs);
    let mut expected = Vec::new();
    for b in 0..3 {
        for c in 0..4 {
            let mut acc = Complex64::new(0.0, 0.0);
            for a in 0..2 {
                acc += lhs_vals[a * 3 + b] * rhs_vals[c * 2 + a];
            }
            expected.push(acc);
        }
    }

    assert_eq!(out.dims(), &[3, 4]);
    assert_row_major_values_close(&out, &expected);
}

#[test]
fn einsum_ab_ca_to_bc_with_lazy_conjugated_rhs_matches_manual_reference() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let lhs = tensor(
        &[
            Complex64::new(1.0, 0.2),
            Complex64::new(-0.5, 0.4),
            Complex64::new(2.0, -0.1),
            Complex64::new(0.3, -1.0),
            Complex64::new(1.2, 0.7),
            Complex64::new(-0.8, 0.5),
        ],
        &[2, 3],
    );
    let rhs_base = tensor(
        &[
            Complex64::new(0.5, -0.1),
            Complex64::new(1.0, 0.6),
            Complex64::new(-1.2, 0.3),
            Complex64::new(0.7, -0.9),
            Complex64::new(0.2, 1.1),
            Complex64::new(-0.4, 0.8),
            Complex64::new(1.3, -0.2),
            Complex64::new(0.9, 0.5),
        ],
        &[4, 2],
    );
    let rhs = rhs_base.conj().unwrap();

    let out = Tensor::einsum("ab,ca->bc", &[&lhs, &rhs]).unwrap();

    let lhs_vals = row_major_values(&lhs);
    let rhs_vals = row_major_values(&rhs);
    let mut expected = Vec::new();
    for b in 0..3 {
        for c in 0..4 {
            let mut acc = Complex64::new(0.0, 0.0);
            for a in 0..2 {
                acc += lhs_vals[a * 3 + b] * rhs_vals[c * 2 + a];
            }
            expected.push(acc);
        }
    }

    assert_eq!(out.dims(), &[3, 4]);
    assert_row_major_values_close(&out, &expected);
}

#[test]
fn einsum_ab_ba_to_scalar_matches_manual_reference() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let lhs = tensor(
        &[
            Complex64::new(1.0, 0.2),
            Complex64::new(-0.5, 0.4),
            Complex64::new(2.0, -0.1),
            Complex64::new(0.3, -1.0),
            Complex64::new(1.2, 0.7),
            Complex64::new(-0.8, 0.5),
        ],
        &[2, 3],
    );
    let rhs = tensor(
        &[
            Complex64::new(0.5, -0.1),
            Complex64::new(1.0, 0.6),
            Complex64::new(-1.2, 0.3),
            Complex64::new(0.7, -0.9),
            Complex64::new(0.2, 1.1),
            Complex64::new(-0.4, 0.8),
        ],
        &[3, 2],
    );

    let out = Tensor::einsum("ab,ba->", &[&lhs, &rhs]).unwrap();
    let lhs_vals = row_major_values(&lhs);
    let rhs_vals = row_major_values(&rhs);
    let mut expected = Complex64::new(0.0, 0.0);
    for a in 0..2 {
        for b in 0..3 {
            expected += lhs_vals[a * 3 + b] * rhs_vals[b * 2 + a];
        }
    }

    match out.try_scalar_value().unwrap() {
        tenferro::ScalarValue::C64(actual) => {
            assert!(
                (actual - expected).norm() < TOL,
                "scalar mismatch: actual={actual:?}, expected={expected:?}, diff={:?}",
                (actual - expected).norm()
            );
        }
        other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}

#[test]
fn einsum_ab_ba_to_scalar_with_lazy_conjugated_rhs_matches_manual_reference() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let lhs = tensor(
        &[
            Complex64::new(1.0, 0.2),
            Complex64::new(-0.5, 0.4),
            Complex64::new(2.0, -0.1),
            Complex64::new(0.3, -1.0),
            Complex64::new(1.2, 0.7),
            Complex64::new(-0.8, 0.5),
        ],
        &[2, 3],
    );
    let rhs_base = tensor(
        &[
            Complex64::new(0.5, -0.1),
            Complex64::new(1.0, 0.6),
            Complex64::new(-1.2, 0.3),
            Complex64::new(0.7, -0.9),
            Complex64::new(0.2, 1.1),
            Complex64::new(-0.4, 0.8),
        ],
        &[3, 2],
    );
    let rhs = rhs_base.conj().unwrap();

    let out = Tensor::einsum("ab,ba->", &[&lhs, &rhs]).unwrap();
    let lhs_vals = row_major_values(&lhs);
    let rhs_vals = row_major_values(&rhs);
    let mut expected = Complex64::new(0.0, 0.0);
    for a in 0..2 {
        for b in 0..3 {
            expected += lhs_vals[a * 3 + b] * rhs_vals[b * 2 + a];
        }
    }

    match out.try_scalar_value().unwrap() {
        tenferro::ScalarValue::C64(actual) => {
            assert!(
                (actual - expected).norm() < TOL,
                "scalar mismatch: actual={actual:?}, expected={expected:?}, diff={:?}",
                (actual - expected).norm()
            );
        }
        other => panic!("expected Complex64 scalar, got {other:?}"),
    }
}
