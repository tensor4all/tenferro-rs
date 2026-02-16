//! Backward-mode (autodiff) implementation of [`TensorNetworkOps`] for the
//! [`Autodiff<B, C>`] backend.
//!
//! This module registers einsum as a differentiable operation in Burn's
//! autodiff graph.  When a backward pass is triggered, the VJP (vector–
//! Jacobian product) will be computed via tenferro's `einsum_rrule`.
//!
//! # Variable-arity operations and Burn's `Backward<B, N>`
//!
//! Burn's [`Backward`] trait uses a const generic `N` that fixes the number
//! of parent tensors at compile time (e.g., `Backward<B, 2>` for binary
//! ops).  Einsum accepts a variable number of inputs known only at run time.
//!
//! ## Chosen approach: macro-generated impls for N = 1..8
//!
//! We generate `Backward<B, N>` implementations for each arity via a macro.
//! Every impl delegates to the same tenferro `einsum_rrule` internally —
//! only the `[NodeRef; N]` / `[Option<NodeRef>; N]` array sizes differ.
//! At the call site, `tn_einsum` dispatches to the matching arity based on
//! `inputs.len()`.
//!
//! ```ignore
//! macro_rules! impl_einsum_backward {
//!     ($n:literal) => {
//!         impl<B: TensorNetworkOps> Backward<B, $n> for EinsumBackward {
//!             type State = (String, Vec<Shape> /*, checkpointed ids */);
//!             fn backward(self, ops: Ops<Self::State, $n>, grads: &mut Gradients, ..) {
//!                 // All N parents are real inputs — no dummy padding.
//!                 // Convert Burn grads → tenferro, call einsum_rrule,
//!                 // convert back, register per-input gradients.
//!             }
//!         }
//!     };
//! }
//! impl_einsum_backward!(1);
//! impl_einsum_backward!(2);
//! // ... up to 8
//! ```
//!
//! This keeps tenferro's N-ary einsum intact (no forced binary
//! decomposition) while fitting cleanly into Burn's compile-time–sized
//! backward infrastructure.  8 inputs covers all practical tensor network
//! contractions; the limit can be raised trivially if needed.
//!
//! For now the function body is `todo!()`, deferring implementation until
//! the AD infrastructure in tenferro (chainrules / rrule) is complete.

use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::Autodiff;
use burn::tensor::ops::FloatTensor;

use crate::TensorNetworkOps;

impl<B, C> TensorNetworkOps for Autodiff<B, C>
where
    B: TensorNetworkOps,
    C: CheckpointStrategy,
{
    /// Perform an einsum contraction, recording the operation on the autodiff
    /// tape so that gradients can be computed during the backward pass.
    ///
    /// # Future implementation plan
    ///
    /// The backward pass will invoke tenferro's `einsum_rrule` to obtain the
    /// VJP for each input tensor.  The contraction tree used in the forward
    /// pass will be cached (or re-derived) for the backward pass so that
    /// each partial derivative is itself an optimised einsum.
    fn tn_einsum(_subscripts: &str, _inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        todo!()
    }
}
