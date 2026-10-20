//! Pool helpers shared by the CPU engine's operations.

use num_complex::{Complex32, Complex64};
use strided_kernel::ExecContext;
use tenferro_tensor::{DType, ElementwiseReadOp};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::{Tensor, TensorRead, TensorWrite, TypedTensor};

use super::{copy_tensor_read_into, elementwise};

pub(crate) fn elementwise_read_into_fallback_with_pool(
    buffers: &mut BufferPool,
    ctx: &ExecContext,
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: TensorWrite<'_>,
) -> crate::Result<()> {
    let result = match op {
        ElementwiseReadOp::Add => {
            elementwise::add_read_with_pool(buffers, ctx, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Subtract => {
            elementwise::sub_read_with_pool(buffers, ctx, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Multiply => {
            elementwise::mul_read_with_pool(buffers, ctx, inputs[0].clone(), inputs[1].clone())?
        }
        ElementwiseReadOp::Negate => {
            elementwise::neg_read_with_pool(buffers, ctx, inputs[0].clone())?
        }
        ElementwiseReadOp::Conj => {
            elementwise::conj_read_with_pool(buffers, ctx, inputs[0].clone())?
        }
        ElementwiseReadOp::Divide => {
            elementwise::div_read_with_pool(buffers, ctx, inputs[0].clone(), inputs[1].clone())?
        }
        _ => {
            return Err(crate::Error::unsupported(
                "CpuEngine::elementwise_read_into",
                format!("CPU backend does not implement {op:?}"),
            ))
        }
    };
    let copied = copy_tensor_read_into(
        "CpuEngine::elementwise_read_into",
        TensorRead::from_tensor(&result),
        out,
    );
    // The staged result is scratch: return it to the pool for the next op.
    reclaim_tensor(buffers, result);
    copied
}

pub(crate) fn reclaim_tensor(buffers: &mut BufferPool, tensor: Tensor) {
    match tensor.dtype() {
        DType::F32 => reclaim_tensor_typed::<f32>(buffers, tensor),
        DType::F64 => reclaim_tensor_typed::<f64>(buffers, tensor),
        DType::I32 => reclaim_tensor_typed::<i32>(buffers, tensor),
        DType::I64 => reclaim_tensor_typed::<i64>(buffers, tensor),
        DType::Bool => reclaim_tensor_typed::<bool>(buffers, tensor),
        DType::C32 => reclaim_tensor_typed::<Complex32>(buffers, tensor),
        DType::C64 => reclaim_tensor_typed::<Complex64>(buffers, tensor),
        DType::External(_) => {}
    }
}

/// Hand the typed tensor back to the pool when the tag table reached the matching tag.
fn reclaim_tensor_typed<T: tenferro_cpu_basic::PoolScalar>(
    buffers: &mut BufferPool,
    tensor: Tensor,
) {
    if let Ok(typed) = tensor.into_typed::<T>() {
        reclaim_typed(buffers, typed);
    }
}

pub(crate) fn reclaim_typed<T: PoolScalar>(pool: &mut BufferPool, typed: TypedTensor<T>) {
    if typed.backend_buffer().is_some() {
        return;
    }
    if let Ok(data) = typed.into_host_vec() {
        T::pool_release(pool, data);
    }
}

#[cfg(test)]
mod tests;
