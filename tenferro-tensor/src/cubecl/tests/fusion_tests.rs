use crate::backend::ElementwiseFusionPlan;
use crate::config::CompareDir;
use crate::{ElementwiseFusionInst, ElementwiseFusionOp, TensorBackend};

use super::{assert_tensor_close, cpu_backend, download, gpu_backend, tensor_f64, upload};

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

/// Compare produces 0/1 output.
fn compare_plan() -> ElementwiseFusionPlan {
    ElementwiseFusionPlan {
        dtype: crate::DType::F64,
        n_inputs: 2,
        outputs: vec![2],
        ops: vec![ElementwiseFusionInst {
            op: ElementwiseFusionOp::Compare(CompareDir::Gt),
            inputs: vec![0, 1],
        }],
    }
}

#[test]
#[ignore]
fn test_fused_compare() {
    let a = tensor_f64(vec![4], vec![1.0, 5.0, 3.0, 2.0]);
    let b = tensor_f64(vec![4], vec![2.0, 3.0, 3.0, 4.0]);

    let mut cpu = cpu_backend();
    let expected = cpu.compare(&a, &b, &CompareDir::Gt).unwrap();

    let mut gpu = gpu_backend();
    let gpu_a = upload(&gpu, &a);
    let gpu_b = upload(&gpu, &b);

    let plan = compare_plan();
    let result = gpu
        .execute_elementwise_fusion(&[&gpu_a, &gpu_b], &plan)
        .unwrap()
        .expect("fusion should succeed for compare");
    let actual = download(&gpu, &result[0]);
    assert_tensor_close(&actual, &expected, 1e-12);
}
