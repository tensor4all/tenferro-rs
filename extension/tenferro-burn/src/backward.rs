//! Backward-mode support for [`TensorNetworkOps`] on Burn's autodiff backend.
//!
//! N-ary einsum calls are lowered to a sequence of unary/binary autodiff
//! nodes following tenferro's contraction tree, so the public Burn surface
//! matches the forward N-ary einsum contract.

use burn::backend::autodiff::checkpoint::{base::Checkpointer, strategy::CheckpointStrategy};
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::Autodiff;
use burn::tensor::ops::FloatTensor;
use burn::tensor::TensorMetadata;

use tenferro_algebra::Standard;
use tenferro_einsum::{ContractionTree, NestedEinsum, Subscripts};
use tenferro_prims::{CpuBackend, CpuContext};

use crate::{burn_to_tenferro, tenferro_to_burn, TensorNetworkOps};

#[derive(Clone, Debug)]
struct EinsumState<T> {
    subscripts: String,
    inputs: Vec<T>,
}

fn labels_to_notation(labels: &[u32]) -> String {
    labels
        .iter()
        .map(|&label| {
            char::from_u32(label)
                .expect("tenferro-burn received a non-Unicode einsum label in a string path")
        })
        .collect()
}

fn subscripts_to_notation(subscripts: &Subscripts) -> String {
    let inputs = subscripts
        .inputs
        .iter()
        .map(|labels| labels_to_notation(labels))
        .collect::<Vec<_>>()
        .join(",");
    format!("{inputs}->{}", labels_to_notation(&subscripts.output))
}

fn binary_step_notation(lhs: &[u32], rhs: &[u32], output: &[u32]) -> String {
    format!(
        "{},{}->{}",
        labels_to_notation(lhs),
        labels_to_notation(rhs),
        labels_to_notation(output)
    )
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

fn execute_einsum_tree<B, C>(
    subscripts: &Subscripts,
    inputs: Vec<FloatTensor<Autodiff<B, C>>>,
) -> FloatTensor<Autodiff<B, C>>
where
    B: TensorNetworkOps,
    C: CheckpointStrategy,
{
    match inputs.len() {
        0 => panic!("tenferro-burn autodiff einsum requires at least one input tensor"),
        1 => unary_einsum::<B, C>(
            &subscripts_to_notation(subscripts),
            inputs
                .into_iter()
                .next()
                .expect("unary einsum dispatch lost its only input"),
        ),
        2 => {
            let mut iter = inputs.into_iter();
            let lhs = iter
                .next()
                .expect("binary einsum dispatch lost its lhs input");
            let rhs = iter
                .next()
                .expect("binary einsum dispatch lost its rhs input");
            binary_einsum::<B, C>(&subscripts_to_notation(subscripts), lhs, rhs)
        }
        n_inputs => {
            let shapes: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape().dims).collect();
            let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
            let tree = ContractionTree::optimize(subscripts, &shape_refs).expect(
                "tenferro-burn autodiff einsum could not optimize the pairwise contraction path",
            );
            let mut slots: Vec<Option<FloatTensor<Autodiff<B, C>>>> =
                inputs.into_iter().map(Some).collect();
            slots.resize(n_inputs + tree.step_count(), None);

            for step_idx in 0..tree.step_count() {
                let (left, right) = tree
                    .step_pair(step_idx)
                    .expect("contraction tree is missing a recorded step");
                let (lhs_subs, rhs_subs, out_subs) = tree
                    .step_subscripts(step_idx)
                    .expect("contraction tree is missing step subscripts");
                let lhs = slots[left]
                    .take()
                    .expect("contraction tree referenced a consumed lhs operand");
                let rhs = slots[right]
                    .take()
                    .expect("contraction tree referenced a consumed rhs operand");
                let step_notation = binary_step_notation(lhs_subs, rhs_subs, out_subs);
                let result = binary_einsum::<B, C>(&step_notation, lhs, rhs);
                slots[n_inputs + step_idx] = Some(result);
            }

            slots
                .into_iter()
                .rev()
                .flatten()
                .next()
                .expect("contraction tree did not leave a final result")
        }
    }
}

fn execute_nested_einsum<B, C>(
    nested: &NestedEinsum,
    inputs: &[FloatTensor<Autodiff<B, C>>],
) -> FloatTensor<Autodiff<B, C>>
where
    B: TensorNetworkOps,
    C: CheckpointStrategy,
{
    match nested {
        NestedEinsum::Leaf(index) => inputs
            .get(*index)
            .cloned()
            .expect("nested einsum referenced a missing input tensor"),
        NestedEinsum::Node {
            subscripts,
            children,
        } => {
            let child_results = children
                .iter()
                .map(|child| execute_nested_einsum::<B, C>(child, inputs))
                .collect();
            execute_einsum_tree::<B, C>(subscripts, child_results)
        }
    }
}

impl<B, C> TensorNetworkOps for Autodiff<B, C>
where
    B: TensorNetworkOps,
    C: CheckpointStrategy,
{
    fn tn_einsum(subscripts: &str, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        let nested = NestedEinsum::parse(subscripts).expect(
            "tenferro-burn autodiff einsum received invalid subscripts or mismatched parentheses",
        );
        execute_nested_einsum::<B, C>(&nested, &inputs)
    }
}
