// Run with: cargo test --features cuda -- --ignored
use crate::backend::ElementwiseFusionPlan;
use crate::{ElementwiseFusionInst, ElementwiseFusionOp, TensorBackend};

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c32, tensor_c64, tensor_f64,
    upload,
};

/// Build a plan for: out = (a + b) * a
fn add_mul_plan() -> ElementwiseFusionPlan {
    ElementwiseFusionPlan {
        dtype: crate::DType::F64,
        n_inputs: 2,
        outputs: vec![3], // value index of final result
        ops: vec![
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Add,
                inputs: vec![0, 1], // a + b
            },
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Multiply,
                inputs: vec![2, 0], // (a+b) * a
            },
        ],
    }
}

#[test]
#[ignore]
fn test_fused_add_mul_matches_cpu() {
    let a = tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b = tensor_f64(vec![4], vec![0.5, -1.0, 2.0, 0.0]);

    let mut cpu = cpu_backend();
    let sum = cpu.add(&a, &b).unwrap();
    let expected = cpu.mul(&sum, &a).unwrap();

    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);

    let plan = add_mul_plan();
    let result = gpu
        .execute_elementwise_fusion(&[&gpu_a, &gpu_b], &plan)
        .unwrap()
        .expect("fusion should succeed for f64 add+mul");
    assert_eq!(result.len(), 1);
    let actual = download(&gpu, &result[0]);
    assert_tensor_close(&actual, &expected, 1e-12);
}

fn complex_add_conj_mul_plan(dtype: crate::DType) -> ElementwiseFusionPlan {
    ElementwiseFusionPlan {
        dtype,
        n_inputs: 2,
        outputs: vec![4],
        ops: vec![
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Add,
                inputs: vec![0, 1],
            },
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Conj,
                inputs: vec![2],
            },
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Multiply,
                inputs: vec![3, 0],
            },
        ],
    }
}

#[test]
#[ignore]
fn test_fused_complex_c64_add_conj_mul_matches_cpu() {
    let a = tensor_c64(
        vec![3],
        vec![
            num_complex::Complex64::new(1.0, 2.0),
            num_complex::Complex64::new(-3.0, 0.5),
            num_complex::Complex64::new(0.25, -1.0),
        ],
    );
    let b = tensor_c64(
        vec![3],
        vec![
            num_complex::Complex64::new(0.5, -1.0),
            num_complex::Complex64::new(2.0, 3.0),
            num_complex::Complex64::new(-4.0, 0.75),
        ],
    );

    let mut cpu = cpu_backend();
    let sum = cpu.add(&a, &b).unwrap();
    let conj = cpu.conj(&sum).unwrap();
    let expected = cpu.mul(&conj, &a).unwrap();

    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);

    let plan = complex_add_conj_mul_plan(crate::DType::C64);
    let result = gpu
        .execute_elementwise_fusion(&[&gpu_a, &gpu_b], &plan)
        .unwrap()
        .expect("fusion should succeed for c64 add+conj+mul");
    assert_eq!(result.len(), 1);
    let actual = download(&gpu, &result[0]);
    assert_tensor_close(&actual, &expected, 1e-12);
}

fn complex_div_neg_plan(dtype: crate::DType) -> ElementwiseFusionPlan {
    ElementwiseFusionPlan {
        dtype,
        n_inputs: 2,
        outputs: vec![3],
        ops: vec![
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Divide,
                inputs: vec![0, 1],
            },
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Negate,
                inputs: vec![2],
            },
        ],
    }
}

#[test]
#[ignore]
fn test_fused_complex_c32_div_neg_matches_cpu() {
    let a = tensor_c32(
        vec![3],
        vec![
            num_complex::Complex32::new(1.0, 2.0),
            num_complex::Complex32::new(-3.0, 0.5),
            num_complex::Complex32::new(0.25, -1.0),
        ],
    );
    let b = tensor_c32(
        vec![3],
        vec![
            num_complex::Complex32::new(0.5, -1.0),
            num_complex::Complex32::new(2.0, 3.0),
            num_complex::Complex32::new(-4.0, 0.75),
        ],
    );

    let mut cpu = cpu_backend();
    let div = cpu.div(&a, &b).unwrap();
    let expected = cpu.neg(&div).unwrap();

    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);

    let plan = complex_div_neg_plan(crate::DType::C32);
    let result = gpu
        .execute_elementwise_fusion(&[&gpu_a, &gpu_b], &plan)
        .unwrap()
        .expect("fusion should succeed for c32 div+neg");
    assert_eq!(result.len(), 1);
    let actual = download(&gpu, &result[0]);
    assert_tensor_close(&actual, &expected, 1e-5);
}

/// Build a plan for: out = neg(a + b)
fn add_neg_plan() -> ElementwiseFusionPlan {
    ElementwiseFusionPlan {
        dtype: crate::DType::F64,
        n_inputs: 2,
        outputs: vec![3],
        ops: vec![
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Add,
                inputs: vec![0, 1],
            },
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Negate,
                inputs: vec![2],
            },
        ],
    }
}

#[test]
#[ignore]
fn test_fused_add_neg() {
    let a = tensor_f64(vec![3], vec![1.0, -2.0, 3.0]);
    let b = tensor_f64(vec![3], vec![4.0, 5.0, -6.0]);

    let mut cpu = cpu_backend();
    let sum = cpu.add(&a, &b).unwrap();
    let expected = cpu.neg(&sum).unwrap();

    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);

    let plan = add_neg_plan();
    let result = gpu
        .execute_elementwise_fusion(&[&gpu_a, &gpu_b], &plan)
        .unwrap()
        .expect("fusion should succeed");
    let actual = download(&gpu, &result[0]);
    assert_tensor_close(&actual, &expected, 1e-12);
}

/// Plan with multiple outputs: both (a+b) and neg(a+b) are live.
fn multi_output_plan() -> ElementwiseFusionPlan {
    ElementwiseFusionPlan {
        dtype: crate::DType::F64,
        n_inputs: 2,
        outputs: vec![2, 3], // sum and neg(sum)
        ops: vec![
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Add,
                inputs: vec![0, 1],
            },
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Negate,
                inputs: vec![2],
            },
        ],
    }
}

#[test]
#[ignore]
fn test_fused_multi_output() {
    let a = tensor_f64(vec![3], vec![1.0, 2.0, 3.0]);
    let b = tensor_f64(vec![3], vec![4.0, 5.0, 6.0]);

    let mut cpu = cpu_backend();
    let sum_expected = cpu.add(&a, &b).unwrap();
    let neg_expected = cpu.neg(&sum_expected).unwrap();

    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);

    let plan = multi_output_plan();
    let result = gpu
        .execute_elementwise_fusion(&[&gpu_a, &gpu_b], &plan)
        .unwrap()
        .expect("fusion should succeed for multi-output");
    assert_eq!(result.len(), 2);
    assert_tensor_close(&download(&gpu, &result[0]), &sum_expected, 1e-12);
    assert_tensor_close(&download(&gpu, &result[1]), &neg_expected, 1e-12);
}

/// Plan with unary transcendentals: exp(sqrt(abs(a)))
fn unary_chain_plan() -> ElementwiseFusionPlan {
    ElementwiseFusionPlan {
        dtype: crate::DType::F64,
        n_inputs: 1,
        outputs: vec![3],
        ops: vec![
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Abs,
                inputs: vec![0],
            },
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Sqrt,
                inputs: vec![1],
            },
            ElementwiseFusionInst {
                op: ElementwiseFusionOp::Exp,
                inputs: vec![2],
            },
        ],
    }
}

#[test]
#[ignore]
fn test_fused_unary_chain() {
    let a = tensor_f64(vec![4], vec![-4.0, 1.0, 9.0, 0.25]);

    let mut cpu = cpu_backend();
    let t1 = cpu.abs(&a).unwrap();
    let t2 = cpu.sqrt(&t1).unwrap();
    let expected = cpu.exp(&t2).unwrap();

    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);

    let plan = unary_chain_plan();
    let result = gpu
        .execute_elementwise_fusion(&[&gpu_a], &plan)
        .unwrap()
        .expect("fusion should succeed for unary chain");
    let actual = download(&gpu, &result[0]);
    assert_tensor_close(&actual, &expected, 1e-10);
}

#[test]
#[ignore]
fn test_fused_empty_tensor() {
    let a = tensor_f64(vec![0], vec![]);
    let b = tensor_f64(vec![0], vec![]);

    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);

    let plan = add_mul_plan();
    let result = gpu
        .execute_elementwise_fusion(&[&gpu_a, &gpu_b], &plan)
        .unwrap()
        .expect("fusion should handle empty tensors");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape(), &[0]);
}
