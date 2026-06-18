use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use computegraph::graph::Graph;
use computegraph::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, TensorMeta};
use tenferro_tensor::{DType, Tensor, TensorBackend, TypedTensor};
use tidu::eager::BackwardExecutor;
use tidu::{LinearizedGraph, PrimitiveGraph};

use crate::eager_builder::EagerPrimitiveBuilder;
use crate::eager_exec::{exec_op_on_tensors, exec_op_on_tensors_with_extension_executor};
use crate::extension_runtime::ExtensionExecutor;
use crate::metadata::{
    push_metadata_scope, register_scoped_live_graph_metadata, tensor_meta_from_tensor,
    GlobalMetadataScope,
};

use super::zero_like_tensor;

pub(crate) struct TenferroBackwardCallbacks<'a, B: TensorBackend + 'static> {
    backend: &'a mut B,
    extension_executor: Option<&'a mut ExtensionExecutor<B>>,
    metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
}

impl<'a, B: TensorBackend + 'static> TenferroBackwardCallbacks<'a, B> {
    pub(crate) fn new(
        backend: &'a mut B,
        extension_executor: Option<&'a mut ExtensionExecutor<B>>,
        metadata_scopes: Vec<Arc<GlobalMetadataScope>>,
    ) -> Self {
        Self {
            backend,
            extension_executor,
            metadata_scopes,
        }
    }
}

pub(super) fn missing_tangent_base_key(
    key: &ValueKey<StdTensorOp>,
) -> Option<ValueKey<StdTensorOp>> {
    let ValueKey::Input(tangent_key) = key else {
        return None;
    };
    let TensorInputKey::Tangent { of, .. } = tangent_key else {
        return None;
    };
    Some(ValueKey::Input((**of).clone()))
}

pub(super) fn eager_forward_input_metadata(
    key: &ValueKey<StdTensorOp>,
    initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
) -> TensorMeta {
    if let Some(value) = initial_data.get(key) {
        return tensor_meta_from_tensor(value.as_ref());
    }

    let base_key = missing_tangent_base_key(key)
        .unwrap_or_else(|| panic!("missing concrete eager value for {:?}", key));
    let base = initial_data
        .get(&base_key)
        .unwrap_or_else(|| panic!("missing base eager value for {:?}", base_key));
    tensor_meta_from_tensor(base.as_ref())
}

pub(super) fn eager_forward_value<B: TensorBackend>(
    all_values: &mut HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    key: &ValueKey<StdTensorOp>,
    initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    backend: &mut B,
) -> Arc<Tensor> {
    if let Some(value) = all_values.get(key) {
        return Arc::clone(value);
    }

    let base_key = missing_tangent_base_key(key)
        .unwrap_or_else(|| panic!("missing concrete eager value for {:?}", key));
    let base = initial_data
        .get(&base_key)
        .unwrap_or_else(|| panic!("missing base eager value for {:?}", base_key));
    // tidu's eager callback trait is infallible here; backend zero creation has
    // already been validated by the surrounding backward pass setup.
    let value =
        Arc::new(zero_like_tensor(base.as_ref(), backend).expect("eager tangent zero creation"));
    all_values.insert(key.clone(), Arc::clone(&value));
    value
}

fn live_graph_values(graph: &Graph<StdTensorOp>) -> HashSet<LocalValueId> {
    let mut producers = HashMap::new();
    for (op_index, op_node) in graph.operations().iter().enumerate() {
        for &output_id in &op_node.outputs {
            producers.insert(output_id, op_index);
        }
    }

    let mut live = HashSet::new();
    let mut stack = graph.outputs().to_vec();
    while let Some(local_id) = stack.pop() {
        if !live.insert(local_id) {
            continue;
        }
        let Some(&op_index) = producers.get(&local_id) else {
            continue;
        };
        for input in &graph.operations()[op_index].inputs {
            if let ValueRef::Local(input_id) = input {
                stack.push(*input_id);
            }
        }
    }

    live
}

fn linear_op_depends_on_tangents(mode: &OperationRole) -> bool {
    matches!(mode, OperationRole::Linearized { active_mask } if active_mask.iter().any(|is_active| *is_active))
}

fn zero_from_exact_metadata<B: TensorBackend>(
    meta: &TensorMeta,
    backend: &mut B,
) -> Option<Tensor> {
    let shape = meta
        .exact_shape()?
        .into_iter()
        .map(|dim| dim.constant_value())
        .collect::<Option<Vec<_>>>()?;
    let host = match meta.dtype {
        // The zero tensor shape comes from exact tensor metadata, so shape/product
        // overflow would indicate an earlier metadata-validation bug.
        DType::F32 => Tensor::F32(TypedTensor::zeros(shape).expect("exact metadata zero shape")),
        // The zero tensor shape comes from exact tensor metadata, so shape/product
        // overflow would indicate an earlier metadata-validation bug.
        DType::F64 => Tensor::F64(TypedTensor::zeros(shape).expect("exact metadata zero shape")),
        // The zero tensor shape comes from exact tensor metadata, so shape/product
        // overflow would indicate an earlier metadata-validation bug.
        DType::I32 => Tensor::I32(TypedTensor::zeros(shape).expect("exact metadata zero shape")),
        // The zero tensor shape comes from exact tensor metadata, so shape/product
        // overflow would indicate an earlier metadata-validation bug.
        DType::I64 => Tensor::I64(TypedTensor::zeros(shape).expect("exact metadata zero shape")),
        DType::Bool => {
            let len = shape.iter().product();
            // The boolean buffer length is computed from the same exact shape.
            Tensor::Bool(
                TypedTensor::from_vec_col_major(shape, vec![false; len])
                    .expect("exact metadata bool zero shape/data match"),
            )
        }
        // The zero tensor shape comes from exact tensor metadata, so shape/product
        // overflow would indicate an earlier metadata-validation bug.
        DType::C32 => Tensor::C32(TypedTensor::zeros(shape).expect("exact metadata zero shape")),
        // The zero tensor shape comes from exact tensor metadata, so shape/product
        // overflow would indicate an earlier metadata-validation bug.
        DType::C64 => Tensor::C64(TypedTensor::zeros(shape).expect("exact metadata zero shape")),
    };
    Some(
        backend
            .upload_host_tensor(&host)
            .unwrap_or_else(|err| panic!("eager primitive zero metadata upload failed: {err}")),
    )
}

fn prefill_missing_linear_zero_values<B: TensorBackend>(
    linear: &LinearizedGraph<StdTensorOp>,
    external_data: &mut HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ctx: &mut ShapeGuardContext,
    backend: &mut B,
) {
    for value in linear.as_graph().values() {
        if external_data.contains_key(&value.key) {
            continue;
        }
        let Some(meta) = ctx
            .metadata_if_available(&ValueRef::External(value.key.clone()))
            .cloned()
        else {
            continue;
        };
        let Some(zero) = zero_from_exact_metadata(&meta, backend) else {
            continue;
        };
        external_data.insert(value.key.clone(), Arc::new(zero));
    }
}

impl<B: TensorBackend + 'static> BackwardExecutor<StdTensorOp>
    for TenferroBackwardCallbacks<'_, B>
{
    fn execute_forward(
        &mut self,
        graph: PrimitiveGraph<'_, StdTensorOp>,
        initial_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
    ) -> HashMap<ValueKey<StdTensorOp>, Arc<Tensor>> {
        let graph = graph.as_graph();
        let mut all_values = initial_data.clone();
        let live_values = live_graph_values(graph);
        let input_metadata = graph
            .inputs()
            .iter()
            .map(|&input_id| {
                let key = graph.values()[input_id].key.clone();
                let meta = eager_forward_input_metadata(&key, initial_data);
                (key, meta)
            })
            .collect::<Vec<_>>();

        for op_node in graph.operations() {
            if linear_op_depends_on_tangents(&op_node.role) {
                continue;
            }
            if !op_node
                .outputs
                .iter()
                .any(|output_id| live_values.contains(output_id))
            {
                continue;
            }

            let resolved_values: Vec<Arc<Tensor>> = op_node
                .inputs
                .iter()
                .map(|input| match input {
                    ValueRef::Local(local_id) => {
                        let key = &graph.values()[*local_id].key;
                        eager_forward_value(&mut all_values, key, initial_data, self.backend)
                    }
                    ValueRef::External(key) => {
                        eager_forward_value(&mut all_values, key, initial_data, self.backend)
                    }
                })
                .collect();
            let resolved_inputs: Vec<&Tensor> =
                resolved_values.iter().map(|value| value.as_ref()).collect();
            let outputs =
                if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
                    exec_op_on_tensors_with_extension_executor(
                        &op_node.operation,
                        &resolved_inputs,
                        self.backend,
                        Some(extension_executor),
                    )
                } else {
                    exec_op_on_tensors(&op_node.operation, &resolved_inputs, self.backend)
                }
                .unwrap_or_else(|err| {
                    panic!(
                        "eager forward exec failed for {:?}: {}",
                        op_node.operation, err
                    )
                });

            for (output_id, output) in op_node.outputs.iter().zip(outputs) {
                let key = graph.values()[*output_id].key.clone();
                all_values.insert(key, Arc::new(output));
            }
        }

        let metadata_scope =
            register_scoped_live_graph_metadata(graph, &live_values, input_metadata)
                // The replay graph and live set were produced by the eager recorder.
                .expect("eager replay metadata registration failed");
        push_metadata_scope(&mut self.metadata_scopes, Arc::new(metadata_scope));

        all_values
    }

    fn run_transposed_linear(
        &mut self,
        linear: &LinearizedGraph<StdTensorOp>,
        cotangent_out: &[Option<Arc<Tensor>>],
        external_data: &HashMap<ValueKey<StdTensorOp>, Arc<Tensor>>,
        ctx: &mut ShapeGuardContext,
    ) -> tidu::ADRuleResult<Vec<Option<Arc<Tensor>>>> {
        let mut external_data = external_data.clone();
        ctx.refresh_global_metadata();
        prefill_missing_linear_zero_values(linear, &mut external_data, ctx, self.backend);

        let mut builder = if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
            EagerPrimitiveBuilder::with_extension_executor(self.backend, extension_executor)
        } else {
            EagerPrimitiveBuilder::new(self.backend)
        };
        builder.external_data = external_data;
        let cotangent_seed_ids = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| builder.push_tensor(Arc::clone(seed)))
            })
            .collect::<Vec<_>>();

        tidu::linear_transpose_with_builder(linear, &mut builder, &cotangent_seed_ids, ctx).map(
            |cotangent_ids| {
                cotangent_ids
                    .into_iter()
                    .map(|maybe_id| maybe_id.map(|id| builder.tensor(id)))
                    .collect()
            },
        )
    }

    fn add_operands(&mut self, a: &Arc<Tensor>, b: &Arc<Tensor>) -> Arc<Tensor> {
        Arc::new(
            self.backend
                .add(a.as_ref(), b.as_ref())
                .unwrap_or_else(|err| panic!("eager cotangent add failed: {}", err)),
        )
    }
}
