#![allow(
    clippy::multiple_bound_locations,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

//! High-level einsum with N-ary contraction tree optimization.
//!
//! This crate provides Einstein summation notation for [`Tensor`]
//! values. It supports:
//!
//! - **String notation**: `"ij,jk->ik"` (NumPy/PyTorch compatible)
//! - **Parenthesized notation**: `"ij,(jk,kl)->il"` respects user-specified
//!   contraction order via [`NestedEinsum`] (OMEinsum.jl-compatible)
//! - **Integer label notation**: omeinsum-rs compatible, using `u32` labels
//! - **N-ary contraction**: Automatic or manual optimization of pairwise
//!   contraction order via [`ContractionTree`]
//! - **Binary primitive**: Public two-input einsum APIs (`einsum_binary*`) for
//!   composing explicit contraction paths in higher layers
//! - **Accumulating variants**: [`einsum_into`], [`einsum_with_subscripts_into`],
//!   [`einsum_with_plan_into`] write into a pre-allocated output buffer with
//!   BLAS-style `alpha`/`beta` scaling, avoiding allocation in hot loops
//!
//! # Backend dispatch
//!
//! The backend is passed explicitly as a type parameter `Backend: TensorPrims<Alg>`
//! with a mutable context `&mut Backend::Context`.  This follows Rust idiom of
//! explicit ownership and mutability (no global/thread-local state).
//! The context provides access to the thread pool and plan cache.
//!
//! # Examples
//!
//! ## Common operations
//!
//! ```ignore
//! use tenferro_einsum::einsum;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, CpuContext};
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mut ctx = CpuContext::new(4);
//!
//! let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
//! let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
//! let v = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], col).unwrap();
//!
//! // Matrix multiplication: C = A @ B
//! let c = einsum::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
//!
//! // Trace: tr(A)
//! let tr = einsum::<_, _, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
//!
//! // Outer product: v_i * v_j -> M_{ij}
//! let outer = einsum::<_, _, CpuBackend>(&mut ctx, "i,j->ij", &[&v, &v], None).unwrap();
//!
//! // Dot product: v . v
//! let dot = einsum::<_, _, CpuBackend>(&mut ctx, "i,i->", &[&v, &v], None).unwrap();
//!
//! // Matrix-vector product: A @ v
//! let mv = einsum::<_, _, CpuBackend>(&mut ctx, "ij,j->i", &[&a, &v], None).unwrap();
//!
//! // Diagonal embedding: vector -> diagonal matrix
//! // v = [1, 2, 3] -> [[1,0,0],[0,2,0],[0,0,3]]
//! let diag = einsum::<_, _, CpuBackend>(&mut ctx, "i->ii", &[&v], None).unwrap();
//! assert_eq!(diag.dims(), &[3, 3]);
//!
//! // Diagonal extraction: matrix -> diagonal vector
//! let d = einsum::<_, _, CpuBackend>(&mut ctx, "ii->i", &[&a], None).unwrap();
//!
//! // Higher-order diagonal: 3D tensor with repeated index
//! // Creates T_{iii} from v_i
//! let t = einsum::<_, _, CpuBackend>(&mut ctx, "i->iii", &[&v], None).unwrap();
//! assert_eq!(t.dims(), &[3, 3, 3]);
//!
//! // Consuming variant: operands are moved (buffer reuse not yet implemented)
//! use tenferro_einsum::einsum_owned;
//! let x = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
//! let y = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
//! let z = einsum_owned::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", vec![x, y], None).unwrap();
//! ```
//!
//! ## Batch operations
//!
//! ```ignore
//! // Batched GEMM: 10 independent matrix multiplications in one call
//! // A: (batch=10, m=3, k=4), B: (batch=10, k=4, n=5) -> C: (batch=10, m=3, n=5)
//! let a = Tensor::<f64>::zeros(&[10, 3, 4], LogicalMemorySpace::MainMemory, col);
//! let b = Tensor::<f64>::zeros(&[10, 4, 5], LogicalMemorySpace::MainMemory, col);
//! let c = einsum::<_, _, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&a, &b], None).unwrap();
//! assert_eq!(c.dims(), &[10, 3, 5]);
//!
//! // Multiple batch dimensions: (batch1=2, batch2=3, m, k) x (batch1=2, batch2=3, k, n)
//! let a = Tensor::<f64>::zeros(&[2, 3, 4, 5], LogicalMemorySpace::MainMemory, col);
//! let b = Tensor::<f64>::zeros(&[2, 3, 5, 6], LogicalMemorySpace::MainMemory, col);
//! let c = einsum::<_, _, CpuBackend>(&mut ctx, "abij,abjk->abik", &[&a, &b], None).unwrap();
//! assert_eq!(c.dims(), &[2, 3, 4, 6]);
//!
//! // Broadcast batch: A has batch dim, B is shared across batch
//! // A: (batch=10, m=3, k=4), B: (k=4, n=5) -> C: (batch=10, m=3, n=5)
//! let a = Tensor::<f64>::zeros(&[10, 3, 4], LogicalMemorySpace::MainMemory, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], LogicalMemorySpace::MainMemory, col);
//! let c = einsum::<_, _, CpuBackend>(&mut ctx, "bij,jk->bik", &[&a, &b], None).unwrap();
//! assert_eq!(c.dims(), &[10, 3, 5]);
//! ```
//!
//! ## Integer label notation
//!
//! ```ignore
//! use tenferro_einsum::{einsum_with_subscripts, Subscripts};
//!
//! // Same as "ij,jk->ik" but with integer labels
//! // Useful when indices exceed 52 (a-z, A-Z) or are computed programmatically
//! let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
//! let c = einsum_with_subscripts::<_, _, CpuBackend>(&mut ctx, &subs, &[&a, &b], None).unwrap();
//! ```
//!
//! ## Contraction order control
//!
//! ```ignore
//! // Three matrices: D = A @ B @ C
//! // Parentheses specify: contract B*C first, then A*(BC)
//! let d = einsum::<_, _, CpuBackend>(&mut ctx, "ij,(jk,kl)->il", &[&a, &b, &c], None).unwrap();
//!
//! // Or use ContractionTree for programmatic control
//! use tenferro_einsum::ContractionTree;
//! let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
//! let tree = ContractionTree::from_pairs(
//!     &subs,
//!     &[&[3, 4], &[4, 5], &[5, 6]],
//!     &[(1, 2), (0, 3)],  // B*C first (avoids large intermediate)
//! ).unwrap();
//! let d = einsum_with_plan::<_, _, CpuBackend>(&mut ctx, &tree, &[&a, &b, &c], None).unwrap();
//! ```
//!
//! ## Accumulating into a pre-allocated output
//!
//! ```ignore
//! use tenferro_einsum::{einsum_with_plan_into, ContractionTree, Subscripts};
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, CpuContext};
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mut ctx = CpuContext::new(4);
//! let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
//! let tree = ContractionTree::optimize(&subs, &[&[3, 4], &[4, 5]]).unwrap();
//! let a = Tensor::<f64>::zeros(&[3, 4], LogicalMemorySpace::MainMemory, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], LogicalMemorySpace::MainMemory, col);
//! let mut c = Tensor::<f64>::zeros(&[3, 5], LogicalMemorySpace::MainMemory, col);
//!
//! // Hot loop: reuse output buffer, zero allocation per iteration
//! for _ in 0..1000 {
//!     // C = 1.0 * (A @ B) + 0.0 * C  (overwrite)
//!     einsum_with_plan_into::<_, _, CpuBackend>(
//!         &mut ctx, &tree, &[&a, &b], 1.0, 0.0, &mut c, None,
//!     ).unwrap();
//! }
//! ```
//!
//! ## GPU async chaining (deferred evaluation)
//!
//! > **Status: Not yet implemented.** GPU backends do not exist yet.
//! > The examples below are aspirational design targets, not working code.
//!
//! GPU einsum operations return immediately. The result tensor carries a
//! [`CompletionEvent`](tenferro_tensor::CompletionEvent) that tracks the
//! pending accelerator work. Passing this tensor to another einsum chains
//! via GPU stream dependencies — no CPU synchronization until data is
//! accessed from the host.
//!
//! - `wait()` — explicitly blocks until computation completes
//! - `dims()`, `strides()` — implicitly call `wait()`
//! - For CPU tensors, `event` is always `None` (zero overhead)
//!
//! ```ignore
//! use tenferro_einsum::einsum;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::CudaBackend; // future
//!
//! // In production, obtain memory spaces via BackendRegistry (future API).
//! let gpu_mem = LogicalMemorySpace::GpuMemory { device_id: 0 };
//! let col = MemoryOrder::ColumnMajor;
//! let mut gpu_ctx = /* CudaContext from BackendRegistry */;
//!
//! let a = Tensor::<f64>::zeros(&[3, 4], gpu_mem, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], gpu_mem, col);
//!
//! // Both einsum calls submit work to the GPU and return immediately.
//! // The second call detects c's pending event and chains on the stream.
//! let c = einsum::<_, _, CudaBackend>(&mut gpu_ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
//! let d = einsum::<_, _, CudaBackend>(&mut gpu_ctx, "ij,jk->ik", &[&c, &b], None).unwrap();
//!
//! // wait() blocks until GPU computation completes
//! d.wait();
//! ```
//!
//! ## Specifying a compute device
//!
//! > **Status: Not yet implemented.** See GPU note above.
//!
//! ```ignore
//! use tenferro_einsum::einsum;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::{LogicalMemorySpace, ComputeDevice};
//!
//! let col = MemoryOrder::ColumnMajor;
//! // In production, obtain memory spaces via BackendRegistry (future API).
//! let gpu_mem = LogicalMemorySpace::GpuMemory { device_id: 0 };
//!
//! let mut a = Tensor::<f64>::zeros(&[3, 4], gpu_mem, col);
//! let mut b = Tensor::<f64>::zeros(&[4, 5], gpu_mem, col);
//!
//! // Pin tensors to CUDA device 1 (overrides automatic device selection).
//! // This works when CUDA device 1 can access GpuMemory { device_id: 0 }
//! // (e.g., same physical GPU or NVLink-connected peer).
//! // If the device cannot access the memory space, einsum returns
//! // Err(NoCompatibleComputeDevice). In that case, transfer explicitly:
//! //   let a = a.to_memory_space_async(GpuMemory { device_id: 1 }).unwrap();
//! a.set_preferred_compute_device(Some(ComputeDevice::Cuda { device_id: 1 }));
//! b.set_preferred_compute_device(Some(ComputeDevice::Cuda { device_id: 1 }));
//!
//! // einsum dispatches to the specified CUDA device
//! let c = einsum::<_, _, CudaBackend>(&mut gpu_ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
//!
//! // Clear override — revert to automatic device selection
//! // a.set_preferred_compute_device(None);
//! ```

// Internal modules
pub(crate) mod ad;
pub(crate) mod api;
mod binary;
mod classify;
mod dispatch;
mod execute;
mod manual;
mod nested;
mod notation;
mod plan;
mod pool;
mod prepare;
mod subscripts;
mod tree;
mod unary;
mod util;

// Public re-exports: types
pub use nested::NestedEinsum;
pub use subscripts::Subscripts;
pub use tree::ContractionTree;

// Public re-exports: functions
pub use api::{
    einsum, einsum_into, einsum_owned, einsum_with_path, einsum_with_path_into, einsum_with_plan,
    einsum_with_plan_into, einsum_with_plan_owned, einsum_with_subscripts,
    einsum_with_subscripts_into, einsum_with_subscripts_owned,
};
pub use binary::{
    einsum_binary, einsum_binary_into, einsum_binary_with_subscripts,
    einsum_binary_with_subscripts_into,
};

pub use ad::{
    dual_einsum, einsum_frule, einsum_hvp, einsum_rrule, tracked_einsum, variable_einsum,
};

#[cfg(feature = "profile-dispatch")]
pub use dispatch::print_and_reset_profile;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use tenferro_device::{Error, LogicalMemorySpace};
    use tenferro_tensor::{MemoryOrder, Tensor};

    use crate::util::infer_memory_space;

    #[test]
    fn infer_memory_space_single_cpu() {
        let a = Tensor::<f64>::zeros(
            &[2, 3],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        let space = infer_memory_space(&[&a]).unwrap();
        assert_eq!(space, LogicalMemorySpace::MainMemory);
    }

    #[test]
    fn infer_memory_space_multiple_cpu() {
        let a = Tensor::<f64>::zeros(
            &[2, 3],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        let b = Tensor::<f64>::zeros(
            &[3, 4],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        let c = Tensor::<f64>::zeros(
            &[4, 5],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        let space = infer_memory_space(&[&a, &b, &c]).unwrap();
        assert_eq!(space, LogicalMemorySpace::MainMemory);
    }

    #[test]
    fn infer_memory_space_empty_operands_errors() {
        let operands: &[&Tensor<f64>] = &[];
        let result = infer_memory_space(operands);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "expected InvalidArgument, got: {err:?}"
        );
    }

    #[test]
    fn infer_memory_space_mixed_errors() {
        // We cannot construct GPU tensors in tests (assertion in Tensor::zeros
        // prevents GPU allocation), so we verify the logic by testing
        // the happy path (all CPU) and the error path (empty).
        // A true mixed-memory test requires GPU support which is not yet
        // available in the POC.
        //
        // This test documents the intended behaviour: calling einsum with
        // operands on different memory spaces returns
        // Error::CrossMemorySpaceOperation.
        let a = Tensor::<f64>::zeros(
            &[2, 3],
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        // Verify that identical spaces produce Ok
        let space = infer_memory_space(&[&a, &a]).unwrap();
        assert_eq!(space, LogicalMemorySpace::MainMemory);
    }
}
