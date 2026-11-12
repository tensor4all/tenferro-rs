//! #2602: the engine-taking receivers outside `EagerSession` and the duplicate
//! value-copy spellings are gone. Each line must stay a compile error.

use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};

fn legacy(ctx: &Arc<EagerRuntime>, x: &EagerTensor, t: Tensor, u: Tensor) {
    let mut e = tenferro_cpu::CpuEngine::serial();
    let _ = x.backward(&mut e);
    let _ = x.backward_with(&mut e, x);
    let _ = ctx.grad(&mut e, x, x);
    let _ = ctx.grad_optional(&mut e, x, x);
    let _ = ctx.vjp(&mut e, x, x, x);
    let _ = ctx.vjp_optional(&mut e, x, x, x);
    let _ = ctx.jvp(&mut e, x, x, x);
    let _ = ctx.jvp_optional(&mut e, x, x, x);
    let _ = x.detach(&mut e);
    let _ = x.detach_into(&mut e, ctx);
    let _ = x.to_tensor(&mut e);
    let _ = x.duplicate_value(&mut e);
    let _ = EagerTensor::from_tensor_in(t, Arc::clone(ctx));
    let _ = EagerTensor::requires_grad_in(u, Arc::clone(ctx));
    let _ = ctx.with_eager_session(&mut e, |s| {
        let _ = s.backend_session();
        s.to_tensor(x)
    });
}

fn main() {}
