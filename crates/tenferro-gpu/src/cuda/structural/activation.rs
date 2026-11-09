//! Fused real-float activations on CUDA: one launch, one pass per element.
//!
//! Backs the optional [`tenferro_tensor::Engine::fused_activation_read`] fast path. The
//! kernel is not handwritten here: the activation is a [`FusionPlan`] over the catalogue's
//! element-wise vocabulary, which the kernel crate's fusion emitter turns into one CUDA
//! entry (`t4a_fused_elementwise`) built from the same `abi.cuh` helpers the
//! per-operation entries use. The fused kernel and the composite formulation therefore
//! evaluate the same expressions, which is why their results agree bit for bit.
//!
//! Only `softplus` has a fused form here (#2530); every other [`ActivationOp`] and
//! every non-`F32`/`F64` dtype declines with `Ok(None)` so the caller keeps the shared
//! composite.

use t4a_cuda_kernels::fusion::{FusionInstruction, FusionOp, FusionOperand, FusionPlan};
use tenferro_tensor::{ActivationOp, DType, Tensor, TensorRead};

use super::{alloc_tensor, count_i64, CudaEngine};
use crate::cuda::fusion::PreparedFusion;

/// The prepared fused-activation kernels an engine reuses across calls.
///
/// Owner: the [`CudaEngine`], for its whole lifetime, beside the prepared catalogue
/// dispatchers and with the same policy: an entry is a compiled and bound module, created
/// on the first call that needs it and dropped with the engine. Capacity is fixed by
/// construction at one slot per (fused activation, real dtype), two today, so the set
/// needs no eviction. Keeping the prepared region here, instead of re-emitting it per
/// call, avoids rebuilding and re-keying the ~42 KB emitted source on every launch.
#[derive(Default)]
pub(crate) struct FusedActivationKernels {
    softplus_f32: Option<PreparedFusion>,
    softplus_f64: Option<PreparedFusion>,
}

impl FusedActivationKernels {
    /// The prepared softplus region for `dtype`, preparing it on first use.
    fn softplus(
        &mut self,
        runtime: &crate::cuda::runtime::CudaRuntime,
        dtype: DType,
    ) -> crate::Result<&PreparedFusion> {
        let (slot, kernel) = match dtype {
            DType::F32 => (&mut self.softplus_f32, t4a_cuda_kernels::DType::F32),
            DType::F64 => (&mut self.softplus_f64, t4a_cuda_kernels::DType::F64),
            other => {
                return Err(crate::Error::unsupported(
                    "softplus",
                    format!("no fused softplus for {other:?}"),
                ))
            }
        };
        match slot {
            Some(prepared) => Ok(prepared),
            None => Ok(slot.insert(PreparedFusion::prepare(runtime, &softplus_plan(kernel))?)),
        }
    }
}

/// Softplus as one fused region over input `v0`:
///
/// ```text
/// v1 = |v0|             v4 = log1p(v3)        // tail = log1p(exp(-|x|))
/// v2 = -v1              v5 = v0 + v4          // x + tail
/// v3 = exp(v2)          v6 = max(v5, v4)      // x > 0 ? x + tail : tail
/// ```
///
/// The composite (`tenferro_runtime::composite::softplus`) computes the same `-|x|`,
/// `exp`, `log1p` and `x + tail`, then selects `x + tail` when `x > 0` and `tail`
/// otherwise. The fusion vocabulary has no compare or select, and `max(v5, v4)` makes the
/// same choice: `x + tail >= tail` exactly when `x >= 0` under monotone rounding, the two
/// coincide when they are equal, and a NaN `x` makes both operands NaN. At `x = +-inf`
/// the tail is `log1p(0) = 0`, so the result is `inf` and `0` as in the composite.
fn softplus_plan(dtype: t4a_cuda_kernels::DType) -> FusionPlan {
    use FusionOperand::{Input, Value};
    let instruction = |op, operands: &[FusionOperand]| FusionInstruction {
        op,
        operands: operands.to_vec(),
    };
    FusionPlan {
        dtype,
        inputs: 1,
        instructions: vec![
            instruction(FusionOp::Abs, &[Input(0)]),
            instruction(FusionOp::Neg, &[Value(0)]),
            instruction(FusionOp::Exp, &[Value(1)]),
            instruction(FusionOp::Log1p, &[Value(2)]),
            instruction(FusionOp::Add, &[Input(0), Value(3)]),
            instruction(FusionOp::Max, &[Value(4), Value(3)]),
        ],
        outputs: vec![Value(5)],
    }
}

impl CudaEngine {
    /// The CUDA [`tenferro_tensor::Engine::fused_activation_read`] fast path.
    ///
    /// Declines (`Ok(None)`) an activation without a fused region or a dtype other than
    /// `F32`/`F64`. A borrowed view is staged once into a compact operand by the shared
    /// `device_read`, as every element-wise entry does. The output is a fresh compact
    /// tensor allocated like the per-operation element-wise outputs.
    pub(super) fn fused_activation(
        &mut self,
        op: ActivationOp,
        input: TensorRead<'_>,
    ) -> crate::Result<Option<Tensor>> {
        if op != ActivationOp::Softplus || !matches!(input.dtype(), DType::F32 | DType::F64) {
            return Ok(None);
        }
        let name = op.label();
        let source = self.device_read(input, name)?;
        let dtype = source.dtype();
        let shape = source.shape().to_vec();
        let len = count_i64(&shape, name)?;
        let output = alloc_tensor(self.runtime(), dtype, &shape)?;
        if len == 0 {
            // Nothing to visit; the launch planner would refuse an empty grid.
            return Ok(Some(output));
        }
        let runtime = self.runtime().clone();
        {
            let mut kernels = self.fused_activations.borrow_mut();
            let prepared = kernels.softplus(&runtime, dtype)?;
            with_scalar!(dtype, |T| {
                prepared.launch(
                    len,
                    &[
                        self.allocation_of::<T>(&output, name)?,
                        self.allocation_of::<T>(&source, name)?,
                    ],
                )
            })?;
        }
        self.prove_launch(name)?;
        Ok(Some(output))
    }
}

#[cfg(test)]
#[path = "activation/tests.rs"]
mod tests;
