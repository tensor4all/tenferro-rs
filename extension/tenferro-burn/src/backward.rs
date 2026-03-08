//! Backward-mode support for [`TensorNetworkOps`] on Burn's autodiff backend.
//!
//! The implementation currently supports unary and binary einsum calls. Larger
//! arities fail with a clear panic instead of reaching a placeholder `todo!()`.

use burn::backend::autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::Autodiff;
use burn::tensor::ops::FloatTensor;

use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext};

use crate::{burn_to_tenferro, tenferro_to_burn, TensorNetworkOps};

#[derive(Clone, Debug)]
struct EinsumState<T> {
    subscripts: String,
    inputs: Vec<T>,
}

fn rrule_grads<B: burn::tensor::backend::Backend<FloatElem = f64>>(
    subscripts: &str,
    inputs: &[FloatTensor<B>],
    cotangent: FloatTensor<B>,
) -> Vec<FloatTensor<B>> {
    let device = B::float_device(&cotangent);
    let tenferro_inputs: Vec<_> = inputs.iter().cloned().map(burn_to_tenferro::<B>).collect();
    let input_refs: Vec<_> = tenferro_inputs.iter().collect();
    let tenferro_cotangent = burn_to_tenferro::<B>(cotangent);
    let mut ctx = CpuContext::new(1);
    let grads = tenferro_einsum::einsum_rrule::<Standard<f64>, CpuBackend>(
        &mut ctx,
        subscripts,
        &input_refs,
        &tenferro_cotangent,
    )
    .expect("tenferro-burn autodiff einsum received invalid subscripts or incompatible shapes");

    grads
        .into_iter()
        .map(|grad| tenferro_to_burn::<B>(grad, &device))
        .collect()
}

fn unary_einsum<B, C>(
    subscripts: &str,
    input: FloatTensor<Autodiff<B, C>>,
) -> FloatTensor<Autodiff<B, C>>
where
    B: TensorNetworkOps,
    C: CheckpointStrategy,
{
    #[derive(Debug)]
    struct UnaryEinsum;

    impl<B: burn::tensor::backend::Backend<FloatElem = f64>> Backward<B, 1> for UnaryEinsum {
        type State = EinsumState<B::FloatTensorPrimitive>;

        fn backward(
            self,
            ops: Ops<Self::State, 1>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let mut grad_iter = rrule_grads::<B>(
                &ops.state.subscripts,
                &ops.state.inputs,
                grads.consume::<B>(&ops.node),
            )
            .into_iter();

            if let Some(node) = ops.parents[0].clone() {
                let grad = grad_iter
                    .next()
                    .expect("unary einsum rrule must return exactly one gradient");
                grads.register::<B>(node.id, grad);
            }
        }
    }

    let state = EinsumState {
        subscripts: subscripts.to_owned(),
        inputs: vec![input.primitive.clone()],
    };

    match UnaryEinsum
        .prepare::<C>([input.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => {
            prep.finish(state, B::tn_einsum(subscripts, vec![input.primitive]))
        }
        OpsKind::UnTracked(prep) => prep.finish(B::tn_einsum(subscripts, vec![input.primitive])),
    }
}

fn binary_einsum<B, C>(
    subscripts: &str,
    lhs: FloatTensor<Autodiff<B, C>>,
    rhs: FloatTensor<Autodiff<B, C>>,
) -> FloatTensor<Autodiff<B, C>>
where
    B: TensorNetworkOps,
    C: CheckpointStrategy,
{
    #[derive(Debug)]
    struct BinaryEinsum;

    impl<B: burn::tensor::backend::Backend<FloatElem = f64>> Backward<B, 2> for BinaryEinsum {
        type State = EinsumState<B::FloatTensorPrimitive>;

        fn backward(
            self,
            ops: Ops<Self::State, 2>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let mut grad_iter = rrule_grads::<B>(
                &ops.state.subscripts,
                &ops.state.inputs,
                grads.consume::<B>(&ops.node),
            )
            .into_iter();

            if let Some(node) = ops.parents[0].clone() {
                let grad = grad_iter
                    .next()
                    .expect("binary einsum rrule must return a gradient for lhs");
                grads.register::<B>(node.id, grad);
            }

            if let Some(node) = ops.parents[1].clone() {
                let grad = grad_iter
                    .next()
                    .expect("binary einsum rrule must return a gradient for rhs");
                grads.register::<B>(node.id, grad);
            }
        }
    }

    let state = EinsumState {
        subscripts: subscripts.to_owned(),
        inputs: vec![lhs.primitive.clone(), rhs.primitive.clone()],
    };

    match BinaryEinsum
        .prepare::<C>([lhs.node.clone(), rhs.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            state,
            B::tn_einsum(subscripts, vec![lhs.primitive, rhs.primitive]),
        ),
        OpsKind::UnTracked(prep) => {
            prep.finish(B::tn_einsum(subscripts, vec![lhs.primitive, rhs.primitive]))
        }
    }
}

impl<B, C> TensorNetworkOps for Autodiff<B, C>
where
    B: TensorNetworkOps,
    C: CheckpointStrategy,
{
    fn tn_einsum(subscripts: &str, mut inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        match inputs.len() {
            1 => unary_einsum::<B, C>(
                subscripts,
                inputs
                    .pop()
                    .expect("unary einsum dispatch lost its only input"),
            ),
            2 => {
                let rhs = inputs
                    .pop()
                    .expect("binary einsum dispatch lost its rhs input");
                let lhs = inputs
                    .pop()
                    .expect("binary einsum dispatch lost its lhs input");
                binary_einsum::<B, C>(subscripts, lhs, rhs)
            }
            n => panic!(
                "tenferro-burn autodiff currently supports only unary and binary einsum, got {n} inputs"
            ),
        }
    }
}
