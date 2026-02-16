//! Backward-mode (autodiff) implementation of [`TensorNetworkOps`] for the
//! [`Autodiff<B, C>`] backend.
//!
//! This module registers einsum as a differentiable operation in Burn's
//! autodiff graph.  When a backward pass is triggered, the VJP (vector–
//! Jacobian product) will be computed via tenferro's `einsum_rrule`.
//!
//! # Design challenge – variable-arity operations
//!
//! Burn's `Backward` trait is parameterised by a const generic `N` that
//! fixes the number of parent tensors at compile time.  Einsum, however,
//! accepts a *variable* number of inputs that is only known at run time.
//! The final implementation will need a strategy to bridge this gap, e.g.:
//!
//! * Using a fixed upper bound `N` and padding,
//! * Decomposing the N-ary einsum into a tree of binary contractions
//!   (each of which has `N = 2` and maps directly to Burn's `Backward`),
//! * Or extending Burn with a dynamic-arity backward node.
//!
//! For now the function body is `todo!()`, deferring the choice until the
//! AD infrastructure in tenferro (chainrules / rrule) is fleshed out.

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
