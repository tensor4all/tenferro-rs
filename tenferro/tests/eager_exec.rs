//! Coverage tests for `exec_op_on_tensors`.
//! Exercises each StdTensorOp branch in the dispatch table.

use tenferro::eager_exec::exec_op_on_tensors;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, PadConfig, SliceConfig, Tensor, TypedTensor,
};

fn f64t(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn scalar(v: f64) -> Tensor {
    f64t(vec![], vec![v])
}

fn data(t: &Tensor) -> Vec<f64> {
    match t {
        Tensor::F64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected F64"),
    }
}

fn run1(op: StdTensorOp, a: &Tensor) -> Vec<Tensor> {
    let mut b = CpuBackend::new();
    exec_op_on_tensors(&op, &[a], &mut b).unwrap()
}

fn run2(op: StdTensorOp, a: &Tensor, b: &Tensor) -> Vec<Tensor> {
    let mut ctx = CpuBackend::new();
    exec_op_on_tensors(&op, &[a, b], &mut ctx).unwrap()
}

#[test]
fn elementwise_unary() {
    let x = scalar(2.0);
    assert!((data(&run1(StdTensorOp::Neg, &x)[0])[0] - (-2.0)).abs() < 1e-12);
    assert!((data(&run1(StdTensorOp::Exp, &x)[0])[0] - 2.0_f64.exp()).abs() < 1e-10);
    assert!((data(&run1(StdTensorOp::Log, &x)[0])[0] - 2.0_f64.ln()).abs() < 1e-12);
    assert!((data(&run1(StdTensorOp::Sin, &x)[0])[0] - 2.0_f64.sin()).abs() < 1e-12);
    assert!((data(&run1(StdTensorOp::Cos, &x)[0])[0] - 2.0_f64.cos()).abs() < 1e-12);
    assert!((data(&run1(StdTensorOp::Tanh, &x)[0])[0] - 2.0_f64.tanh()).abs() < 1e-12);
    assert!((data(&run1(StdTensorOp::Sqrt, &x)[0])[0] - 2.0_f64.sqrt()).abs() < 1e-12);
    assert!((data(&run1(StdTensorOp::Rsqrt, &x)[0])[0] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
    assert!((data(&run1(StdTensorOp::Abs, &scalar(-3.0))[0])[0] - 3.0).abs() < 1e-12);
    assert!((data(&run1(StdTensorOp::Sign, &scalar(-3.0))[0])[0] - (-1.0)).abs() < 1e-12);
    assert!(
        (data(&run1(StdTensorOp::Expm1, &scalar(0.01))[0])[0] - 0.01_f64.exp_m1()).abs() < 1e-12
    );
    assert!(
        (data(&run1(StdTensorOp::Log1p, &scalar(0.01))[0])[0] - 0.01_f64.ln_1p()).abs() < 1e-12
    );
    assert!((data(&run1(StdTensorOp::Conj, &x)[0])[0] - 2.0).abs() < 1e-12);
}

#[test]
fn elementwise_binary() {
    let a = scalar(3.0);
    let b = scalar(2.0);
    assert!((data(&run2(StdTensorOp::Add, &a, &b)[0])[0] - 5.0).abs() < 1e-12);
    assert!((data(&run2(StdTensorOp::Mul, &a, &b)[0])[0] - 6.0).abs() < 1e-12);
    assert!((data(&run2(StdTensorOp::Div, &a, &b)[0])[0] - 1.5).abs() < 1e-12);
    assert!((data(&run2(StdTensorOp::Pow, &a, &b)[0])[0] - 9.0).abs() < 1e-12);
    assert!((data(&run2(StdTensorOp::Maximum, &a, &b)[0])[0] - 3.0).abs() < 1e-12);
    assert!((data(&run2(StdTensorOp::Minimum, &a, &b)[0])[0] - 2.0).abs() < 1e-12);
}

#[test]
fn compare_op() {
    let a = scalar(3.0);
    let b = scalar(2.0);
    let result = run2(StdTensorOp::Compare(CompareDir::Gt), &a, &b);
    assert!((data(&result[0])[0] - 1.0).abs() < 1e-12); // 3 > 2 = true
}

#[test]
fn structural_ops() {
    let x = f64t(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = run1(StdTensorOp::Transpose { perm: vec![1, 0] }, &x);
    assert_eq!(t[0].shape(), &[3, 2]);

    let r = run1(
        StdTensorOp::ReduceSum {
            axes: vec![0],
            input_shape: vec![DimExpr::Const(2), DimExpr::Const(3)],
        },
        &x,
    );
    assert_eq!(r[0].shape(), &[3]);

    let rev = run1(StdTensorOp::Reverse { axes: vec![0] }, &x);
    assert_eq!(rev[0].shape(), &[2, 3]);
}

#[test]
fn reduce_ops() {
    let x = f64t(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let prod = run1(
        StdTensorOp::ReduceProd {
            axes: vec![0],
            input_shape: vec![DimExpr::Const(4)],
        },
        &x,
    );
    assert!((data(&prod[0])[0] - 24.0).abs() < 1e-12);

    let max = run1(
        StdTensorOp::ReduceMax {
            axes: vec![0],
            input_shape: vec![DimExpr::Const(4)],
        },
        &x,
    );
    assert!((data(&max[0])[0] - 4.0).abs() < 1e-12);

    let min = run1(
        StdTensorOp::ReduceMin {
            axes: vec![0],
            input_shape: vec![DimExpr::Const(4)],
        },
        &x,
    );
    assert!((data(&min[0])[0] - 1.0).abs() < 1e-12);
}

#[test]
fn reshape_broadcast() {
    let x = f64t(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let r = run1(
        StdTensorOp::Reshape {
            from_shape: vec![DimExpr::Const(6)],
            to_shape: vec![DimExpr::Const(2), DimExpr::Const(3)],
        },
        &x,
    );
    assert_eq!(r[0].shape(), &[2, 3]);

    let s = scalar(5.0);
    let b = run1(
        StdTensorOp::BroadcastInDim {
            shape: vec![DimExpr::Const(3)],
            dims: vec![],
        },
        &s,
    );
    assert_eq!(b[0].shape(), &[3]);
    assert_eq!(data(&b[0]), vec![5.0, 5.0, 5.0]);
}

#[test]
fn slice_pad() {
    let x = f64t(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let sliced = run1(
        StdTensorOp::Slice(SliceConfig {
            starts: vec![1],
            limits: vec![4],
            strides: vec![1],
        }),
        &x,
    );
    assert_eq!(data(&sliced[0]), vec![2.0, 3.0, 4.0]);

    let small = f64t(vec![2], vec![1.0, 2.0]);
    let padded = run1(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![2],
            interior_padding: vec![0],
        }),
        &small,
    );
    assert_eq!(data(&padded[0]), vec![0.0, 1.0, 2.0, 0.0, 0.0]);
}

#[test]
fn diagonal_ops() {
    // 3x3 identity
    let eye = f64t(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    let diag = run1(
        StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1,
        },
        &eye,
    );
    assert_eq!(data(&diag[0]), vec![1.0, 1.0, 1.0]);

    let v = f64t(vec![3], vec![2.0, 3.0, 4.0]);
    let embedded = run1(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        &v,
    );
    assert_eq!(embedded[0].shape(), &[3, 3]);
}

#[test]
fn triangular_ops() {
    let m = f64t(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let tl = run1(StdTensorOp::Tril { k: 0 }, &m);
    assert_eq!(tl[0].shape(), &[3, 3]);
    let tu = run1(StdTensorOp::Triu { k: 0 }, &m);
    assert_eq!(tu[0].shape(), &[3, 3]);
}

#[test]
fn convert_op() {
    let x = scalar(3.14);
    let c = run1(
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::F32,
        },
        &x,
    );
    assert_eq!(c[0].dtype(), DType::F32);
}

#[test]
fn shape_of_op() {
    let x = f64t(vec![3, 5], vec![0.0; 15]);
    let s = run1(StdTensorOp::ShapeOf { axis: 1 }, &x);
    assert!((data(&s[0])[0] - 5.0).abs() < 1e-12);
}

#[test]
fn dynamic_truncate_op() {
    let x = f64t(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let size = scalar(3.0);
    let result = run2(StdTensorOp::DynamicTruncate { axis: 0 }, &x, &size);
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 3.0]);
}

#[test]
fn pad_to_match_op() {
    let x = f64t(vec![3], vec![1.0, 2.0, 3.0]);
    let reference = f64t(vec![5], vec![0.0; 5]);
    let result = run2(StdTensorOp::PadToMatch { axis: 0 }, &x, &reference);
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn svd_op() {
    // 2x2 matrix
    let a = f64t(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]); // identity (col-major)
    let result = run1(
        StdTensorOp::Svd {
            eps: 1e-12,
            input_shape: vec![DimExpr::Const(2), DimExpr::Const(2)],
        },
        &a,
    );
    assert_eq!(result.len(), 3); // U, S, Vt
    assert_eq!(result[1].shape(), &[2]); // S is vector
}

#[test]
fn cholesky_op() {
    // 2x2 positive definite (col-major: [[2,1],[1,2]])
    let a = f64t(vec![2, 2], vec![2.0, 1.0, 1.0, 2.0]);
    let result = run1(
        StdTensorOp::Cholesky {
            input_shape: vec![DimExpr::Const(2), DimExpr::Const(2)],
        },
        &a,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape(), &[2, 2]);
}

#[test]
fn dot_general_op() {
    // Matrix multiply 2x3 @ 3x2
    let a = f64t(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]); // col-major
    let b = f64t(vec![3, 2], vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    };
    let result = run2(StdTensorOp::DotGeneral(config), &a, &b);
    assert_eq!(result[0].shape(), &[2, 2]);
}
