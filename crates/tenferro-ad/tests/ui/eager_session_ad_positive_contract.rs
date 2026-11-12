//! #2602: every eager operation that needs an engine is an `EagerSession`
//! method; the engine enters once, through `with_eager_session`.

use tenferro_ad::{EagerRuntime, EagerTensor, Gradients, Tensor};
use tenferro_runtime::ExecutionEngine;
use tenferro_tensor::backend::Engine;

fn host(values: Vec<f64>) -> Tensor {
    let n = values.len();
    Tensor::from_vec_col_major(vec![n], values).unwrap()
}

fn main() -> Result<(), tenferro_ad::Error> {
    let ctx = EagerRuntime::new()?;
    let mut engine = tenferro_cpu::CpuEngine::serial();
    // Engine-free leaf import (owned host tensors only).
    let x: EagerTensor = ctx.variable_from(Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap())?;
    let seed: EagerTensor = ctx.constant_from(Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap())?;

    ctx.with_eager_session(&mut engine, |s| {
        let c: EagerTensor = s.constant_from(host(vec![1.0]))?;
        let h: EagerTensor = s.constant_from_host(host(vec![1.0]))?;
        let v: EagerTensor = s.variable_from(host(vec![1.0]))?;
        let y = s.mul(&x, &x)?;
        let _: EagerTensor = s.grad(&y, &x)?;
        let _: Option<EagerTensor> = s.grad_optional(&y, &x)?;
        let _: EagerTensor = s.vjp(&y, &x, &seed)?;
        let _: Option<EagerTensor> = s.vjp_optional(&y, &x, &seed)?;
        let _: EagerTensor = s.jvp(&y, &x, &seed)?;
        let _: Option<EagerTensor> = s.jvp_optional(&y, &x, &seed)?;
        let _: Gradients = s.backward(&y)?;
        let _: Gradients = s.backward_with(&y, &seed)?;
        let _: EagerTensor = s.detach(&y)?;
        let _: Tensor = s.duplicate_value(&y)?;
        let engine: &mut dyn ExecutionEngine = s.engine();
        let _: &mut dyn Engine = engine;
        let _ = (c, h, v);
        Ok::<_, tenferro_ad::Error>(())
    })?;

    // Engine-free value access stays on the handles.
    let gradient = x.grad()?.expect("tracked leaf");
    let _: Tensor = gradient.to_tensor(&mut engine)?;
    let _: Option<Tensor> = seed.into_value().ok();
    Ok(())
}
