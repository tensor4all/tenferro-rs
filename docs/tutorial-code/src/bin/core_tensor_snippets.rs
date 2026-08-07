//! Compiled documentation snippets for issue #1609.

// INVARIANT: Independent documentation examples intentionally leave some imports,
// variables, and helper mains unused when compiled as one family binary.
#![allow(dead_code, unused_imports, unused_variables, unused_mut)]

#[rustfmt::skip]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    snippet_eager_operations_1()?;

    // snippet source: docs/guides/eager-operations.md:48
    fn snippet_eager_operations_1() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_1
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TypedTensor};

let mut backend = CpuBackend::new();
        // snippet-end:eager_operations_1
        Ok(())
    }

    snippet_eager_operations_2()?;

    // snippet source: docs/guides/eager-operations.md:97
    fn snippet_eager_operations_2() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_2
use tenferro_runtime::{Tensor, TypedTensor};
use tenferro_tensor::Rank;

// Dynamic dtype (`Tensor`)
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// Static dtype (`TypedTensor`)
let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let ranked: TypedTensor<f64, Rank<2>> = match b.try_into_rank::<2>() {
    Ok(ranked) => ranked,
    Err(err) => panic!("unexpected rank mismatch: {err}"),
};
assert_eq!(ranked.shape(), &[2, 3]);
let b_bad = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
assert!(b_bad.try_into_rank::<3>().is_err());

// Convert between layers for a specific dtype.
let b_for_tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let c = Tensor::F64(b_for_tensor);
assert_eq!(c.shape(), &[2, 3]);
        // snippet-end:eager_operations_2
        Ok(())
    }

    snippet_eager_operations_3()?;

    // snippet source: docs/guides/eager-operations.md:131
    fn snippet_eager_operations_3() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_3
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{TypedTensor};
use tenferro_tensor::backend::TensorViewCanonicalization;

let tensor = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 3],
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
).unwrap();
let view = tensor.as_view().transpose_view([1, 0]).unwrap();
let mut backend = CpuBackend::new();
let compact = backend.to_contiguous(&view).unwrap();

assert_eq!(compact.shape(), &[3, 2]);
assert_eq!(compact.as_slice().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        // snippet-end:eager_operations_3
        Ok(())
    }

    snippet_eager_operations_4()?;

    // snippet source: docs/guides/eager-operations.md:154
    fn snippet_eager_operations_4() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_4
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0])?;

let sum = a.add(&b, &mut backend).unwrap();
let product = a.mul(&b, &mut backend).unwrap();
let negated = a.neg(&mut backend).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
assert_eq!(negated.as_slice::<f64>().unwrap(), &[-1.0, -2.0, -3.0]);
        // snippet-end:eager_operations_4
        Ok(())
    }

    snippet_eager_operations_5()?;

    // snippet source: docs/guides/eager-operations.md:173
    fn snippet_eager_operations_5() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_5
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_linalg::LinalgBackend;
use tenferro_runtime::{BackendSessionHost, Tensor};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3, 3], vec![
    2.0_f64, 1.0, 0.0,
    1.0, 3.0, 1.0,
    0.0, 1.0, 2.0,
])?;
let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
backend.with_backend_session(|session| {
    with_cpu_exec_session(session, |exec| {
        let svd = LinalgBackend::svd(exec, &a).unwrap();
        let qr = LinalgBackend::qr(exec, &a).unwrap();
        let chol = LinalgBackend::cholesky(exec, &a).unwrap();
        let eigh = LinalgBackend::eigh(exec, &a).unwrap();
        let x = LinalgBackend::solve(exec, &a, &b).unwrap();
        assert_eq!(svd[1].shape(), &[3]);
        assert_eq!(qr[0].shape(), &[3, 3]);
        assert_eq!(chol.shape(), &[3, 3]);
        assert_eq!(eigh[0].shape(), &[3]);
        assert_eq!(eigh[1].shape(), &[3, 3]);
        assert_eq!(x.shape(), &[3]);
    }).expect("CPU execution session is available")
});
        // snippet-end:eager_operations_5
        Ok(())
    }

    snippet_eager_operations_6()?;

    // snippet source: docs/guides/eager-operations.md:214
    fn snippet_eager_operations_6() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_6
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// Transpose
let at = a.transpose(&[1, 0], &mut backend).unwrap();
assert_eq!(at.shape(), &[3, 2]);

// Reshape
let flat = a.reshape(&[6], &mut backend).unwrap();
assert_eq!(flat.shape(), &[6]);

// Reduce
let col_sum = a.reduce_sum(&[0], &mut backend).unwrap();
assert_eq!(col_sum.shape(), &[3]);
        // snippet-end:eager_operations_6
        Ok(())
    }

    snippet_eager_operations_7()?;

    // snippet source: docs/guides/eager-operations.md:245
    fn snippet_eager_operations_7() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_7
use tenferro_runtime::Tensor;

let t = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let data: &[f64] = t.as_slice::<f64>().unwrap();
assert_eq!(data, &[1.0, 2.0, 3.0]);
        // snippet-end:eager_operations_7
        Ok(())
    }

    snippet_eager_operations_8()?;

    // snippet source: docs/guides/eager-operations.md:282
    fn snippet_eager_operations_8() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_8
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(), ctx.clone()).unwrap();
let y = EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(), ctx.clone()).unwrap();

let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[6.0, 8.0]);

x.clear_grad().unwrap();
assert!(x.grad().unwrap().is_none());

let loss = x.mul(&y).unwrap().reduce_sum(Some(&[0])).unwrap();
loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);

ctx.clear_grads().unwrap();
assert!(x.grad().unwrap().is_none());
assert!(y.grad().unwrap().is_none());
Ok(())
}
        // snippet-end:eager_operations_8
        Ok(())
    }

    snippet_eager_operations_9()?;

    // snippet source: docs/guides/eager-operations.md:316
    fn snippet_eager_operations_9() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_9
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
    ctx.clone(),
).unwrap();
let seed = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    ctx,
).unwrap();

let y = x.mul(&x).unwrap();
y.backward_with(&seed).unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[4.0, 12.0]);
Ok(())
}
        // snippet-end:eager_operations_9
        Ok(())
    }

    snippet_eager_operations_10()?;

    // snippet source: docs/guides/eager-operations.md:340
    fn snippet_eager_operations_10() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_10
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
    ctx.clone(),
).unwrap();
let y = x.mul(&x).unwrap();
let seed = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
    ctx.clone(),
).unwrap();
let tangent = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
    ctx.clone(),
).unwrap();

let vjp = ctx.vjp(&y, &x, &seed).unwrap();
let jvp = ctx.jvp(&y, &x, &tangent).unwrap();
let vjp_tensor = vjp.to_tensor().unwrap();
assert_eq!(vjp_tensor.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
let jvp_tensor = jvp.to_tensor().unwrap();
assert_eq!(jvp_tensor.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
assert!(x.grad().unwrap().is_none());
Ok(())
}
        // snippet-end:eager_operations_10
        Ok(())
    }

    snippet_eager_operations_11()?;

    // snippet source: docs/guides/eager-operations.md:372
    fn snippet_eager_operations_11() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_11
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
    ctx.clone(),
).unwrap();
let tangent = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap(),
    ctx.clone(),
).unwrap();

let loss = x.mul(&x).unwrap().mul(&x).unwrap();
let grad = ctx.grad(&loss, &x).unwrap();
let hvp = ctx.jvp(&grad, &x, &tangent).unwrap();

let grad_tensor = grad.to_tensor().unwrap();
assert_eq!(grad_tensor.as_slice::<f64>().unwrap(), &[27.0]);
let hvp_tensor = hvp.to_tensor().unwrap();
assert_eq!(hvp_tensor.as_slice::<f64>().unwrap(), &[18.0]);
Ok(())
}
        // snippet-end:eager_operations_11
        Ok(())
    }

    snippet_eager_operations_12()?;

    // snippet source: docs/guides/eager-operations.md:400
    fn snippet_eager_operations_12() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_12
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(),
    ctx.clone(),
).unwrap();
let y = {
    let _guard = ctx.no_grad();
    x.mul(&x).unwrap()
};
assert!(!y.tracks_grad());
Ok(())
}
        // snippet-end:eager_operations_12
        Ok(())
    }

    snippet_eager_operations_13()?;

    // snippet source: docs/guides/eager-operations.md:421
    fn snippet_eager_operations_13() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:eager_operations_13
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
let a = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    ctx.clone(),
).unwrap();
let x = EagerTensor::requires_grad_in(
    Tensor::from_vec_col_major(vec![2, 1], vec![5.0_f64, 6.0]).unwrap(),
    ctx.clone(),
).unwrap();

let y = a.matmul(&x).unwrap();
let y_tensor = y.to_tensor().unwrap();
assert_eq!(y_tensor.as_slice::<f64>().unwrap(), &[23.0, 34.0]);

let loss = y.mul(&y).unwrap().reduce_sum(Some(&[0, 1])).unwrap();
let loss_tensor = loss.to_tensor().unwrap();
assert_eq!(loss_tensor.as_slice::<f64>().unwrap(), &[1685.0]);

loss.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[182.0, 410.0]);
Ok(())
}
        // snippet-end:eager_operations_13
        Ok(())
    }

    snippet_tensor_operations_14()?;

    // snippet source: docs/guides/tensor-operations.md:74
    fn snippet_tensor_operations_14() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_14
use tenferro_tensor::{Rank, TypedTensor};

let mut x = TypedTensor::<f64>::from_vec_col_major(
    vec![2, 2],
    vec![1.0, 2.0, 3.0, 4.0],
)
.unwrap();
assert_eq!(x.shape(), &[2, 2]);
assert_eq!(*x.get(&[1, 0]).unwrap(), 2.0);

*x.get_mut(&[0, 1]).unwrap() = 5.0;
assert_eq!(*x.get(&[1, 1]).unwrap(), 4.0);

let sum: f64 = x.host_data().unwrap().iter().copied().sum();
assert_eq!(sum, 12.0);

let static_rank = TypedTensor::<f64, Rank<2>>::from_vec_col_major(
    [2, 2],
    vec![1.0, 2.0, 3.0, 4.0],
)
.unwrap();
assert_eq!(static_rank.rank(), 2);
        // snippet-end:tensor_operations_14
        Ok(())
    }

    snippet_tensor_operations_15()?;

    // snippet source: docs/guides/tensor-operations.md:110
    fn snippet_tensor_operations_15() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_15
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{CompareDir, TypedTensor, TypedTensorMaskOpsExt, TypedTensorOpsExt};

let mut backend = CpuBackend::new();
let x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
let y = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![4.0, 5.0, 6.0]).unwrap();

let sum = x.add(&y, &mut backend).unwrap();
let product = x.mul(&y, &mut backend).unwrap();
let total = product.reduce_sum(&[0], &mut backend).unwrap();
let mask = sum.compare(&product, CompareDir::Lt, &mut backend).unwrap();
let selected = mask.where_select(&sum, &product, &mut backend).unwrap();

assert_eq!(sum.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice().unwrap(), &[4.0, 10.0, 18.0]);
assert_eq!(total.as_slice().unwrap(), &[32.0]);
assert_eq!(mask.as_slice().unwrap(), &[false, true, true]);
assert_eq!(selected.as_slice().unwrap(), &[4.0, 7.0, 9.0]);
        // snippet-end:tensor_operations_15
        Ok(())
    }

    snippet_tensor_operations_16()?;

    // snippet source: docs/guides/tensor-operations.md:154
    fn snippet_tensor_operations_16() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_16
use tenferro_tensor::TypedTensor;

let mut x = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
for value in x.host_data_mut().unwrap() {
    *value *= 2.0;
}
assert_eq!(x.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
        // snippet-end:tensor_operations_16
        Ok(())
    }

    snippet_tensor_operations_17()?;

    // snippet source: docs/guides/tensor-operations.md:207
    fn snippet_tensor_operations_17() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_17
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0])?;

let sum = a.add(&b, &mut backend).unwrap();
let product = a.mul(&b, &mut backend).unwrap();

assert_eq!(sum.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(product.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
        // snippet-end:tensor_operations_17
        Ok(())
    }

    snippet_tensor_operations_18()?;

    // snippet source: docs/guides/tensor-operations.md:228
    fn snippet_tensor_operations_18() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_18
use tenferro_ad::{EagerRuntime, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap()).unwrap();
let y = (&x * &x)?.reduce_sum(Some(&[0])).unwrap();

y.backward().unwrap();
assert_eq!(x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);
Ok(())
}
        // snippet-end:tensor_operations_18
        Ok(())
    }

    snippet_tensor_operations_19()?;

    // snippet source: docs/guides/tensor-operations.md:246
    fn snippet_tensor_operations_19() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_19
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

let a = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
let b = TracedTensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap();
let sum = (&a + &b).unwrap();
let product = (&a * &b).unwrap();

let mut compiler = GraphCompiler::new();
let program = compiler.compile_many(&[&sum, &product]).unwrap();
let mut builder = Runtime::builder();
builder
    .register_engine(tenferro_cpu::runtime_engine_registration(&CpuBackend::new()).unwrap())
    .unwrap();
let runtime = builder.build().unwrap();
let outputs = runtime.run_compiled(&program, &[]).unwrap();

assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
assert_eq!(outputs[1].as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
        // snippet-end:tensor_operations_19
        Ok(())
    }

    snippet_tensor_operations_20()?;

    // snippet source: docs/guides/tensor-operations.md:277
    fn snippet_tensor_operations_20() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_20
use tenferro_runtime::{Error, TracedTensor};

fn add_three(
    a: &TracedTensor,
    b: &TracedTensor,
    c: &TracedTensor,
) -> Result<TracedTensor, Error> {
    let ab = (a + b)?;
    let sum = (&ab + c)?;
    let canonical = a.add(b)?.add(c)?;
    let _ = canonical;
    Ok(sum)
}
        // snippet-end:tensor_operations_20
        Ok(())
    }

    snippet_tensor_operations_21()?;

    // snippet source: docs/guides/tensor-operations.md:299
    fn snippet_tensor_operations_21() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_21
use tenferro_ad::{EagerRuntime, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let x = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![0.0_f64, 1.0, 2.0]).unwrap()).unwrap();
let y = x.exp().unwrap();

let y_tensor = y.to_tensor().unwrap();
let data = y_tensor.as_slice::<f64>().unwrap();

assert!((data[0] - 1.0).abs() < 1e-12);
assert!((data[1] - std::f64::consts::E).abs() < 1e-12);
assert!((data[2] - 7.38905609893065).abs() < 1e-12);
Ok(())
}
        // snippet-end:tensor_operations_21
        Ok(())
    }

    snippet_tensor_operations_22()?;

    // snippet source: docs/guides/tensor-operations.md:318
    fn snippet_tensor_operations_22() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_22
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
let reshaped = a.reshape(&[6], &mut backend).unwrap();
let transposed = a.transpose(&[1, 0], &mut backend).unwrap();

assert_eq!(reshaped.shape(), &[6]);
assert_eq!(reshaped.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(transposed.shape(), &[3, 2]);
assert_eq!(transposed.as_slice::<f64>().unwrap(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        // snippet-end:tensor_operations_22
        Ok(())
    }

    snippet_tensor_operations_23()?;

    // snippet source: docs/guides/tensor-operations.md:338
    fn snippet_tensor_operations_23() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_23
use tenferro_ad::{EagerRuntime, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let ctx = EagerRuntime::new()?;
let v = ctx.variable_from(Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap()).unwrap();
let repeated = v.broadcast_in_dim(&[3, 2], &[0]).unwrap();

assert_eq!(repeated.shape(), &[3, 2]);
assert_eq!(repeated.to_tensor().unwrap().as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
Ok(())
}
        // snippet-end:tensor_operations_23
        Ok(())
    }

    snippet_tensor_operations_24()?;

    // snippet source: docs/guides/tensor-operations.md:354
    fn snippet_tensor_operations_24() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:tensor_operations_24
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorOpsExt};

let mut backend = CpuBackend::new();
let a = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
)?;
// Logical matrix:
// [[1.0, 3.0, 5.0],
//  [2.0, 4.0, 6.0]]
let row_sums = a.reduce_sum(&[1], &mut backend).unwrap();
let total = a.reduce_sum(&[0, 1], &mut backend).unwrap();

assert_eq!(row_sums.shape(), &[2]);
assert_eq!(row_sums.as_slice::<f64>().unwrap(), &[9.0, 12.0]);
assert_eq!(total.shape(), &[] as &[usize]);
// Rank-0 tensors hold one scalar element; as_slice() returns a length-1 slice.
assert_eq!(total.as_slice::<f64>().unwrap(), &[21.0]);
        // snippet-end:tensor_operations_24
        Ok(())
    }

    snippet_memory_order_25()?;

    // snippet source: docs/guides/memory-order.md:30
    fn snippet_memory_order_25() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:memory_order_25
use tenferro_runtime::Tensor;

let tensor = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
)?;

assert_eq!(tensor.shape(), &[2, 3]);
assert_eq!(
    tensor.as_slice::<f64>().unwrap(),
    &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
);
        // snippet-end:memory_order_25
        Ok(())
    }

    snippet_memory_order_26()?;

    // snippet source: docs/guides/memory-order.md:52
    fn snippet_memory_order_26() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:memory_order_26
use tenferro_runtime::Tensor;

let tensor = Tensor::from_vec_col_major(
    vec![2, 3],
    vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0],
)?;

assert_eq!(
    tensor.as_slice::<f64>().unwrap(),
    &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
);
        // snippet-end:memory_order_26
        Ok(())
    }

    snippet_memory_order_27()?;

    // snippet source: docs/guides/memory-order.md:85
    fn snippet_memory_order_27() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:memory_order_27
use tenferro_runtime::Tensor;

let tensor = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])?;
let (shape, data) = tensor.into_vec_col_major::<f64>().unwrap();

assert_eq!(shape, vec![2, 2]);
assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
        // snippet-end:memory_order_27
        Ok(())
    }

    snippet_cpu_execution_28()?;

    // snippet source: docs/guides/cpu-execution.md:23
    fn snippet_cpu_execution_28() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:cpu_execution_28
use tenferro_cpu::{CpuBackend, CpuPlacement};

let backend = CpuBackend::new();
for node in backend.topology().nodes() {
    println!("OS node {}: {:?}", node.id(), node.cpus().as_usize_vec());
}

let all = backend.for_placement(CpuPlacement::AllAllowed)?;
println!("{:?}", all.execution_info());
        // snippet-end:cpu_execution_28
        Ok(())
    }

    snippet_cpu_execution_29()?;

    // snippet source: docs/guides/cpu-execution.md:55
    fn snippet_cpu_execution_29() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:cpu_execution_29
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuPlacement};

let coordinator = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer)?;
if let Some(node) = coordinator.topology().nodes().first() {
    let local = coordinator.for_placement(CpuPlacement::NumaNode(node.id()))?;
    let another_handle = local.clone();
    assert_eq!(local.resolved_placement(), another_handle.resolved_placement());
}
        // snippet-end:cpu_execution_29
        Ok(())
    }

    snippet_cpu_execution_30()?;

    // snippet source: docs/guides/cpu-execution.md:160
    fn snippet_cpu_execution_30() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:cpu_execution_30
let backend = tenferro_cpu::CpuBackend::new();
let info = backend.execution_info();
println!("kind={:?} provider={}", info.backend_kind(), info.provider_diagnostic());
println!("mode={:?} workers={}", info.execution_mode(), info.worker_count());
println!("topology={:?} requested={:?} resolved={:?}", info.topology(),
    info.requested_placement(), info.resolved_placement());
        // snippet-end:cpu_execution_30
        Ok(())
    }

    snippet_devices_and_gpu_31()?;

    // snippet source: docs/guides/devices-and-gpu.md:181
    fn snippet_devices_and_gpu_31() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:devices_and_gpu_31
use tenferro_gpu::cuda::{cuda_devices, download_tensor, gpu_available, upload_tensor, CudaBackend};
use tenferro_tensor::Tensor;

if !gpu_available() {
    return Ok(());
}
let devices = cuda_devices()?;
let Some(device) = devices.first() else { return Ok(()); };
let backend = CudaBackend::new(device.id())?;
let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
let gpu_x = upload_tensor(backend.runtime(), &x)?;
let cpu_x = download_tensor(backend.runtime(), &gpu_x)?;
assert_eq!(cpu_x.as_slice::<f64>()?, &[1.0, 2.0]);
        // snippet-end:devices_and_gpu_31
        Ok(())
    }

    #[cfg(all(feature = "apple-shared", target_os = "macos"))]
    snippet_devices_and_gpu_32()?;

    // snippet source: docs/guides/devices-and-gpu.md:225
    #[cfg(all(feature = "apple-shared", target_os = "macos"))]
    fn snippet_devices_and_gpu_32() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:devices_and_gpu_32
use tenferro_gpu::apple::AppleContext;
use tenferro_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = AppleContext::new()?;
    let host = Tensor::from_vec_col_major([4], vec![1.0_f32, 2.0, 3.0, 4.0])?;
    let managed = context.upload_tensor(&host)?;
    let mut cpu = context.cpu_backend().clone();
    let mut metal = context.metal_backend().clone();
    let _ = (&managed, &mut cpu, &mut metal);
    Ok(())
}
        // snippet-end:devices_and_gpu_32
        Ok(())
    }

    snippet_views_and_slicing_33()?;

    // snippet source: docs/guides/views-and-slicing.md:13
    fn snippet_views_and_slicing_33() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:views_and_slicing_33
use tenferro_tensor::{Rank, TypedTensor};

let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 3], vec![1.0; 6])?;
let view = tensor.as_view();
assert_eq!(view.shape(), &[2, 3]);
assert_eq!(view.strides(), &[1, 2]);
assert_eq!(view.get(&[1, 2]), Some(&1.0));
        // snippet-end:views_and_slicing_33
        Ok(())
    }

    snippet_views_and_slicing_34()?;

    // snippet source: docs/guides/views-and-slicing.md:26
    fn snippet_views_and_slicing_34() -> Result<(), Box<dyn std::error::Error>> {
use tenferro_tensor::{Rank, TypedTensor};
let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 3], vec![1.0; 6])?;
let view = tensor.as_view();
        // snippet-start:views_and_slicing_34
let transposed = view.transpose_view([1, 0])?;
let reversed = transposed.try_slice(&[
    tenferro_tensor::StridedSliceSpec::all(),
    tenferro_tensor::StridedSliceSpec::reverse(),
])?;
assert_eq!(reversed.shape(), &[3, 2]);
        // snippet-end:views_and_slicing_34
        Ok(())
    }

    snippet_views_and_slicing_35()?;

    // snippet source: docs/guides/views-and-slicing.md:45
    fn snippet_views_and_slicing_35() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:views_and_slicing_35
use tenferro_tensor::{Rank, TypedTensor};

let mut tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![0.0; 4])?;
{
    let mut view = tensor.as_view_mut();
    *view.get_mut(&[1, 0]).ok_or("missing element")? = 4.0;
}
assert_eq!(tensor.as_slice()?, &[0.0, 4.0, 0.0, 0.0]);
        // snippet-end:views_and_slicing_35
        Ok(())
    }

    snippet_views_and_slicing_36()?;

    // snippet source: docs/guides/views-and-slicing.md:66
    fn snippet_views_and_slicing_36() -> Result<(), Box<dyn std::error::Error>> {
        // snippet-start:views_and_slicing_36
use tenferro_tensor::{Rank, TypedTensor};

let tensor = TypedTensor::<f64, Rank<1>>::from_vec_col_major([3], vec![1.0, 2.0, 3.0])?;
let view = tensor.as_view();
let duplicate = view.duplicate()?;
assert_eq!(duplicate.as_slice()?, &[1.0, 2.0, 3.0]);
assert_ne!(view.as_slice()?.as_ptr(), duplicate.as_slice()?.as_ptr());
        // snippet-end:views_and_slicing_36
        Ok(())
    }

    Ok(())
}
