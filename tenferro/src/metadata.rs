use computegraph::fragment::Fragment;
use computegraph::types::{GlobalValKey, ValRef};
use std::collections::HashMap;

use tenferro_ops::ad::context::{
    lookup_global_metadata, register_global_metadata, register_global_metadata_batch, TensorMeta,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_tensor::{DType, Tensor};

use crate::shape_infer::{infer_extension_output_meta, infer_output_dtype, infer_output_shapes};

pub(crate) fn tensor_meta(dtype: DType, shape: Vec<SymDim>) -> TensorMeta {
    TensorMeta { dtype, shape }
}

pub(crate) fn concrete_tensor_meta(dtype: DType, shape: &[usize]) -> TensorMeta {
    tensor_meta(dtype, shape.iter().copied().map(SymDim::from).collect())
}

pub(crate) fn symbolic_input_meta(dtype: DType, tensor_id: u64, rank: usize) -> TensorMeta {
    tensor_meta(
        dtype,
        (0..rank)
            .map(|axis| SymDim::tensor_axis(tensor_id, axis))
            .collect(),
    )
}

pub(crate) fn tensor_meta_from_tensor(tensor: &Tensor) -> TensorMeta {
    concrete_tensor_meta(tensor.dtype(), tensor.shape())
}

pub(crate) fn register_value_metadata(key: GlobalValKey<StdTensorOp>, meta: TensorMeta) {
    register_global_metadata(key, meta);
}

pub(crate) fn registered_meta(key: &GlobalValKey<StdTensorOp>) -> TensorMeta {
    lookup_global_metadata(key)
        .unwrap_or_else(|| panic!("metadata lookup: missing registered metadata for {:?}", key))
}

pub(crate) fn register_fragment_metadata(
    fragment: &Fragment<StdTensorOp>,
    seeded: impl IntoIterator<Item = (GlobalValKey<StdTensorOp>, TensorMeta)>,
) {
    let seeded: Vec<_> = seeded.into_iter().collect();
    // Start from just the seeded inputs. External keys not in `seeded` are
    // resolved on demand via a single-key lookup against the global
    // registry — crucially, we do NOT clone the entire global map. The
    // global registry grows monotonically across a process, so a full-map
    // snapshot per fragment construction is quadratic in the total number
    // of registered ops and dominated oracle_replay runtime.
    let mut known: HashMap<GlobalValKey<StdTensorOp>, TensorMeta> =
        seeded.iter().cloned().collect();

    let mut registrations = seeded;
    for op_node in fragment.ops() {
        let input_metas: Vec<_> = op_node
            .inputs
            .iter()
            .map(|input| {
                let key = match input {
                    ValRef::Local(local_id) => &fragment.vals()[*local_id].key,
                    ValRef::External(key) => key,
                };
                known
                    .get(key)
                    .cloned()
                    .or_else(|| lookup_global_metadata(key))
                    .unwrap_or_else(|| {
                        panic!(
                            "metadata registration: missing input metadata for {:?}",
                            key
                        )
                    })
            })
            .collect();

        let output_metas = infer_output_metas(&op_node.op, &input_metas);
        for (&output_id, meta) in op_node.outputs.iter().zip(output_metas) {
            let key = fragment.vals()[output_id].key.clone();
            known.insert(key.clone(), meta.clone());
            registrations.push((key, meta));
        }
    }

    register_global_metadata_batch(registrations);
}

fn infer_output_metas(op: &StdTensorOp, input_metas: &[TensorMeta]) -> Vec<TensorMeta> {
    let input_shape_exprs: Vec<Vec<DimExpr>> = input_metas
        .iter()
        .enumerate()
        .map(|(input_idx, meta)| DimExpr::input_shape(input_idx, meta.shape.len()))
        .collect();
    let input_shape_refs: Vec<&[DimExpr]> = input_shape_exprs.iter().map(Vec::as_slice).collect();
    let input_dtypes: Vec<DType> = input_metas.iter().map(|meta| meta.dtype).collect();

    if let StdTensorOp::Extension(ext) = op {
        let metas = infer_extension_output_meta(ext.as_ref(), &input_dtypes, &input_shape_refs);
        let resolved_inputs: Vec<&[SymDim]> = input_metas
            .iter()
            .map(|meta| meta.shape.as_slice())
            .collect();
        return metas
            .into_iter()
            .map(|(dtype, shape)| {
                tensor_meta(
                    dtype,
                    shape
                        .iter()
                        .map(|dim| SymDim::from_dim_expr(dim, &resolved_inputs))
                        .collect(),
                )
            })
            .collect();
    }

    let output_dtype = infer_output_dtype(op, &input_dtypes);
    infer_output_shapes(op, &input_shape_refs)
        .into_iter()
        .map(|shape| {
            let resolved_inputs: Vec<&[SymDim]> = input_metas
                .iter()
                .map(|meta| meta.shape.as_slice())
                .collect();
            tensor_meta(
                output_dtype,
                shape
                    .iter()
                    .map(|dim| SymDim::from_dim_expr(dim, &resolved_inputs))
                    .collect(),
            )
        })
        .collect()
}
