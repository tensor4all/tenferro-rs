use computegraph::fragment::FragmentBuilder;
use computegraph::GraphOp;

use tenferro_ops::semiring_ops::SemiringOps;

use super::types::{ContractionPath, Subscripts};

pub fn build_einsum_fragment<Op: GraphOp + SemiringOps>(
    _builder: &mut FragmentBuilder<Op>,
    _path: &ContractionPath,
    _input_ids: &[computegraph::LocalValId],
) -> Vec<computegraph::LocalValId> {
    todo!()
}

pub fn optimize_contraction_path(
    _subscripts: &Subscripts,
    _shapes: &[&[usize]],
) -> ContractionPath {
    todo!()
}
