use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::LocalValId;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, DotGeneralConfig, Tensor, TensorBackend};

use super::compiler::{compile_to_exec, lower_to_stablehlo};
use super::engine::Engine;
use super::error::{Error, Result};
use super::exec::eval_exec_ir;

static NEXT_INPUT_ID: AtomicU64 = AtomicU64::new(0);

fn next_input_key() -> TensorInputKey {
    TensorInputKey {
        id: NEXT_INPUT_ID.fetch_add(1, Ordering::Relaxed),
    }
}

pub struct TracedTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: LocalValId,
    pub data: Option<Tensor>,
    pub(crate) inputs_map: Arc<HashMap<TensorInputKey, Tensor>>,
}

impl TracedTensor {
    pub fn from_tensor(tensor: Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        let key = next_input_key();

        let mut builder = FragmentBuilder::new();
        let val = builder.add_input(key.clone());
        builder.set_outputs(vec![val]);
        let fragment = Arc::new(builder.build());

        let mut map = HashMap::new();
        map.insert(key, tensor.clone());

        Self {
            shape,
            dtype,
            fragment,
            val,
            data: Some(tensor),
            inputs_map: Arc::new(map),
        }
    }

    pub fn eval<B: TensorBackend>(&mut self, engine: &mut Engine<B>) -> Result<&Tensor> {
        if self.data.is_some() {
            return Ok(self.data.as_ref().unwrap());
        }

        let output_key = self.fragment.vals()[self.val].key.clone();

        let view = resolve(vec![self.fragment.clone()]);
        let graph = materialize_merge(&view, &[output_key]);
        let compiled = compile(&graph);

        let stablehlo = lower_to_stablehlo(&compiled);
        let exec = compile_to_exec(&stablehlo);

        let mut input_tensors = Vec::with_capacity(graph.inputs.len());
        for key in &graph.inputs {
            match key {
                GlobalValKey::Input(k) => {
                    let tensor = self.inputs_map.get(k).ok_or_else(|| {
                        Error::MissingInput(format!("missing input data for key {:?}", k))
                    })?;
                    input_tensors.push(tensor.clone());
                }
                _ => {
                    return Err(Error::Internal(
                        "expected Input key in graph inputs".to_string(),
                    ));
                }
            }
        }

        // Use compile cache: store or retrieve the ExecProgram.
        let cached_exec = engine.get_or_compile(exec);
        let mut results = eval_exec_ir(&mut engine.backend, &cached_exec, input_tensors);
        if results.len() != 1 {
            return Err(Error::Internal(format!(
                "expected 1 output, got {}",
                results.len()
            )));
        }

        self.data = Some(results.remove(0));
        Ok(self.data.as_ref().unwrap())
    }

    pub fn grad(&self, _wrt: &TracedTensor) -> TracedTensor {
        todo!()
    }

    pub fn jvp(&self, _wrt: &TracedTensor, _tangent: &TracedTensor) -> TracedTensor {
        todo!()
    }

    pub fn traced_add(&self, other: &TracedTensor) -> TracedTensor {
        apply_binary(StdTensorOp::Add, self, other, self.shape.clone())
    }

    pub fn traced_mul(&self, other: &TracedTensor) -> TracedTensor {
        apply_binary(StdTensorOp::Mul, self, other, self.shape.clone())
    }

    pub fn traced_neg(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Neg, self, self.shape.clone())
    }

    pub fn traced_conj(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Conj, self, self.shape.clone())
    }

    pub fn traced_dot_general(
        &self,
        other: &TracedTensor,
        config: DotGeneralConfig,
    ) -> TracedTensor {
        let lhs_rank = self.shape.len();
        let rhs_rank = other.shape.len();
        let lhs_free: Vec<usize> = (0..lhs_rank)
            .filter(|d| {
                !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d)
            })
            .collect();
        let rhs_free: Vec<usize> = (0..rhs_rank)
            .filter(|d| {
                !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d)
            })
            .collect();
        let mut out_shape = Vec::new();
        for &d in &config.lhs_batch_dims {
            out_shape.push(self.shape[d]);
        }
        for &d in &lhs_free {
            out_shape.push(self.shape[d]);
        }
        for &d in &rhs_free {
            out_shape.push(other.shape[d]);
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        apply_binary(StdTensorOp::DotGeneral(config), self, other, out_shape)
    }

    pub fn traced_reduce_sum(&self, axes: &[usize]) -> TracedTensor {
        let out_shape: Vec<usize> = (0..self.shape.len())
            .filter(|d| !axes.contains(d))
            .map(|d| self.shape[d])
            .collect();
        let out_shape = if out_shape.is_empty() {
            vec![1]
        } else {
            out_shape
        };
        apply_unary(
            StdTensorOp::ReduceSum {
                axes: axes.to_vec(),
                input_shape: self.shape.clone(),
            },
            self,
            out_shape,
        )
    }

    pub fn traced_reshape(&self, shape: &[usize]) -> TracedTensor {
        apply_unary(
            StdTensorOp::Reshape {
                from_shape: self.shape.clone(),
                to_shape: shape.to_vec(),
            },
            self,
            shape.to_vec(),
        )
    }

    pub fn traced_broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> TracedTensor {
        apply_unary(
            StdTensorOp::BroadcastInDim {
                shape: shape.to_vec(),
                dims: dims.to_vec(),
            },
            self,
            shape.to_vec(),
        )
    }

    pub fn traced_transpose(&self, perm: &[usize]) -> TracedTensor {
        let out_shape: Vec<usize> = perm.iter().map(|&p| self.shape[p]).collect();
        apply_unary(
            StdTensorOp::Transpose {
                perm: perm.to_vec(),
            },
            self,
            out_shape,
        )
    }
}

fn apply_unary(op: StdTensorOp, input: &TracedTensor, out_shape: Vec<usize>) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    builder.add_parent(input.fragment.clone());
    let input_ref = ValRef::External(input.fragment.vals()[input.val].key.clone());
    let outputs = builder.add_op(op, vec![input_ref], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());

    TracedTensor {
        shape: out_shape,
        dtype: input.dtype,
        fragment,
        val: outputs[0],
        data: None,
        inputs_map: input.inputs_map.clone(),
    }
}

fn apply_binary(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    out_shape: Vec<usize>,
) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    builder.add_parent(lhs.fragment.clone());
    builder.add_parent(rhs.fragment.clone());
    let lhs_ref = ValRef::External(lhs.fragment.vals()[lhs.val].key.clone());
    let rhs_ref = ValRef::External(rhs.fragment.vals()[rhs.val].key.clone());
    let outputs = builder.add_op(op, vec![lhs_ref, rhs_ref], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());

    let mut merged = (*lhs.inputs_map).clone();
    merged.extend(rhs.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));

    TracedTensor {
        shape: out_shape,
        dtype: lhs.dtype,
        fragment,
        val: outputs[0],
        data: None,
        inputs_map: Arc::new(merged),
    }
}

pub fn eval_all<B: TensorBackend>(
    engine: &mut Engine<B>,
    outputs: &mut [&mut TracedTensor],
) -> Result<Vec<Tensor>> {
    let mut all_fragments = Vec::new();
    let mut output_keys = Vec::new();
    let mut all_inputs: HashMap<TensorInputKey, Tensor> = HashMap::new();

    for tt in outputs.iter() {
        all_fragments.push(tt.fragment.clone());
        output_keys.push(tt.fragment.vals()[tt.val].key.clone());
        all_inputs.extend(tt.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));
    }

    let view = resolve(all_fragments);
    let graph = materialize_merge(&view, &output_keys);
    let compiled = compile(&graph);
    let stablehlo = lower_to_stablehlo(&compiled);
    let exec = compile_to_exec(&stablehlo);

    let mut input_tensors = Vec::with_capacity(graph.inputs.len());
    for key in &graph.inputs {
        match key {
            GlobalValKey::Input(k) => {
                let tensor = all_inputs.get(k).ok_or_else(|| {
                    Error::MissingInput(format!("missing input data for key {:?}", k))
                })?;
                input_tensors.push(tensor.clone());
            }
            _ => {
                return Err(Error::Internal(
                    "expected Input key in graph inputs".to_string(),
                ));
            }
        }
    }

    let cached_exec = engine.get_or_compile(exec);
    let results = eval_exec_ir(&mut engine.backend, &cached_exec, input_tensors);

    for (tt, result) in outputs.iter_mut().zip(results.iter()) {
        tt.data = Some(result.clone());
    }

    Ok(results)
}
