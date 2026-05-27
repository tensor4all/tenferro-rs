use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use computegraph::fragment::Fragment;
use computegraph::{GlobalValKey, LocalValId, OpMode, ValRef};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, TensorMeta};
use tenferro_tensor::{Tensor, TensorBackend};
use tidu::{BackwardCallbacks, LinearFragment};

use crate::eager_emitter::EagerEmitter;
use crate::eager_exec::{exec_op_on_tensors, exec_op_on_tensors_with_extension_executor};
use crate::extension_runtime::ExtensionExecutor;
use crate::metadata::{
    push_metadata_scope, register_scoped_live_fragment_metadata, tensor_meta_from_tensor,
    MetadataScope,
};

use super::zero_like_tensor;

pub(crate) struct TenferroBackwardCallbacks<'a, B: TensorBackend + 'static> {
    backend: &'a mut B,
    extension_executor: Option<&'a mut ExtensionExecutor<B>>,
    metadata_scopes: Vec<Arc<MetadataScope>>,
}

impl<'a, B: TensorBackend + 'static> TenferroBackwardCallbacks<'a, B> {
    pub(crate) fn new(
        backend: &'a mut B,
        extension_executor: Option<&'a mut ExtensionExecutor<B>>,
        metadata_scopes: Vec<Arc<MetadataScope>>,
    ) -> Self {
        Self {
            backend,
            extension_executor,
            metadata_scopes,
        }
    }
}

pub(super) fn missing_tangent_base_key(
    key: &GlobalValKey<StdTensorOp>,
) -> Option<GlobalValKey<StdTensorOp>> {
    let GlobalValKey::Input(tangent_key) = key else {
        return None;
    };
    let TensorInputKey::Tangent { of, .. } = tangent_key else {
        return None;
    };
    Some(GlobalValKey::Input((**of).clone()))
}

pub(super) fn eager_forward_input_metadata(
    key: &GlobalValKey<StdTensorOp>,
    initial_data: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
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
    all_values: &mut HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
    key: &GlobalValKey<StdTensorOp>,
    initial_data: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
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
    let value = Arc::new(zero_like_tensor(base.as_ref(), backend));
    all_values.insert(key.clone(), Arc::clone(&value));
    value
}

fn live_fragment_values(fragment: &Fragment<StdTensorOp>) -> HashSet<LocalValId> {
    let mut producers = HashMap::new();
    for (op_index, op_node) in fragment.ops().iter().enumerate() {
        for &output_id in &op_node.outputs {
            producers.insert(output_id, op_index);
        }
    }

    let mut live = HashSet::new();
    let mut stack = fragment.outputs().to_vec();
    while let Some(local_id) = stack.pop() {
        if !live.insert(local_id) {
            continue;
        }
        let Some(&op_index) = producers.get(&local_id) else {
            continue;
        };
        for input in &fragment.ops()[op_index].inputs {
            if let ValRef::Local(input_id) = input {
                stack.push(*input_id);
            }
        }
    }

    live
}

fn linear_op_depends_on_tangents(mode: &OpMode) -> bool {
    matches!(mode, OpMode::Linear { active_mask } if active_mask.iter().any(|is_active| *is_active))
}

impl<B: TensorBackend + 'static> BackwardCallbacks<StdTensorOp>
    for TenferroBackwardCallbacks<'_, B>
{
    fn execute_forward(
        &mut self,
        fragment: &Fragment<StdTensorOp>,
        initial_data: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
    ) -> HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>> {
        let mut all_values = initial_data.clone();
        let live_values = live_fragment_values(fragment);
        let input_metadata = fragment
            .inputs()
            .iter()
            .map(|&input_id| {
                let key = fragment.vals()[input_id].key.clone();
                let meta = eager_forward_input_metadata(&key, initial_data);
                (key, meta)
            })
            .collect::<Vec<_>>();

        for op_node in fragment.ops() {
            if linear_op_depends_on_tangents(&op_node.mode) {
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
                    ValRef::Local(local_id) => {
                        let key = &fragment.vals()[*local_id].key;
                        eager_forward_value(&mut all_values, key, initial_data, self.backend)
                    }
                    ValRef::External(key) => {
                        eager_forward_value(&mut all_values, key, initial_data, self.backend)
                    }
                })
                .collect();
            let resolved_inputs: Vec<&Tensor> =
                resolved_values.iter().map(|value| value.as_ref()).collect();
            let outputs =
                if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
                    exec_op_on_tensors_with_extension_executor(
                        &op_node.op,
                        &resolved_inputs,
                        self.backend,
                        Some(extension_executor),
                    )
                } else {
                    exec_op_on_tensors(&op_node.op, &resolved_inputs, self.backend)
                }
                .unwrap_or_else(|err| {
                    panic!("eager forward exec failed for {:?}: {}", op_node.op, err)
                });

            for (output_id, output) in op_node.outputs.iter().zip(outputs.into_iter()) {
                let key = fragment.vals()[*output_id].key.clone();
                all_values.insert(key, Arc::new(output));
            }
        }

        let metadata_scope =
            register_scoped_live_fragment_metadata(fragment, &live_values, input_metadata);
        push_metadata_scope(&mut self.metadata_scopes, Arc::new(metadata_scope));

        all_values
    }

    fn eager_transpose(
        &mut self,
        linear: &LinearFragment<StdTensorOp>,
        cotangent_out: &[Option<Arc<Tensor>>],
        external_data: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
        ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<Arc<Tensor>>> {
        let mut emitter = if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
            EagerEmitter::with_extension_executor(self.backend, extension_executor)
        } else {
            EagerEmitter::new(self.backend)
        };
        emitter.external_data = external_data.clone();
        let cotangent_seed_ids = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| emitter.push_tensor(Arc::clone(seed)))
            })
            .collect::<Vec<_>>();

        ctx.refresh_global_metadata();
        tidu::eager_transpose_fragment(linear, &mut emitter, &cotangent_seed_ids, ctx)
            .into_iter()
            .map(|maybe_id| maybe_id.map(|id| emitter.tensor(id)))
            .collect()
    }

    fn try_eager_transpose(
        &mut self,
        linear: &LinearFragment<StdTensorOp>,
        cotangent_out: &[Option<Arc<Tensor>>],
        external_data: &HashMap<GlobalValKey<StdTensorOp>, Arc<Tensor>>,
        ctx: &mut ShapeGuardContext,
    ) -> chainrules_core::ADRuleResult<Vec<Option<Arc<Tensor>>>> {
        let mut emitter = if let Some(extension_executor) = self.extension_executor.as_deref_mut() {
            EagerEmitter::with_extension_executor(self.backend, extension_executor)
        } else {
            EagerEmitter::new(self.backend)
        };
        emitter.external_data = external_data.clone();
        let cotangent_seed_ids = cotangent_out
            .iter()
            .map(|maybe_seed| {
                maybe_seed
                    .as_ref()
                    .map(|seed| emitter.push_tensor(Arc::clone(seed)))
            })
            .collect::<Vec<_>>();

        ctx.refresh_global_metadata();
        tidu::try_eager_transpose_fragment(linear, &mut emitter, &cotangent_seed_ids, ctx).map(
            |cotangent_ids| {
                cotangent_ids
                    .into_iter()
                    .map(|maybe_id| maybe_id.map(|id| emitter.tensor(id)))
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
