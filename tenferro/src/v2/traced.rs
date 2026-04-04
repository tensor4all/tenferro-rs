use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::LocalValId;
use tenferro_einsum::v2::builder::build_einsum_fragment;
use tenferro_einsum::v2::types::Subscripts;
use tenferro_ops::config::DotGeneralConfig;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::v2::{DType, Tensor};

use super::backend::eval_exec_ir_host;
use super::compiler::{compile_to_exec, lower_to_stablehlo};
use super::engine::Engine;

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

    pub fn eval(&mut self, _engine: &mut Engine) -> &Tensor {
        if self.data.is_some() {
            return self.data.as_ref().unwrap();
        }

        let output_key = self.fragment.vals()[self.val].key.clone();

        let view = resolve(vec![self.fragment.clone()]);
        let graph = materialize_merge(&view, &[output_key]);
        let compiled = compile(&graph);

        let stablehlo = lower_to_stablehlo(&compiled);
        let exec = compile_to_exec(&stablehlo);

        let input_tensors: Vec<Tensor> = graph
            .inputs
            .iter()
            .map(|key| match key {
                GlobalValKey::Input(k) => self
                    .inputs_map
                    .get(k)
                    .expect("missing input data for key")
                    .clone(),
                _ => panic!("expected Input key in graph inputs"),
            })
            .collect();

        let mut results = eval_exec_ir_host(&exec, input_tensors);
        assert_eq!(results.len(), 1);

        self.data = Some(results.remove(0));
        self.data.as_ref().unwrap()
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
            },
            self,
            out_shape,
        )
    }

    pub fn traced_reshape(&self, shape: &[usize]) -> TracedTensor {
        apply_unary(
            StdTensorOp::Reshape {
                shape: shape.to_vec(),
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

    pub fn traced_einsum(subscript_str: &str, inputs: &[&TracedTensor]) -> TracedTensor {
        let subscripts = Subscripts::parse(subscript_str);

        let shapes: Vec<Vec<usize>> = inputs.iter().map(|t| t.shape.clone()).collect();

        // Compute output shape from subscripts and input shapes
        let out_shape = compute_einsum_output_shape(&subscripts, &shapes);

        let mut builder = FragmentBuilder::new();

        // Add parents and create ValRef for each input
        let mut input_vals = Vec::new();
        for input in inputs {
            builder.add_parent(input.fragment.clone());
            let val_ref = ValRef::External(input.fragment.vals()[input.val].key.clone());
            input_vals.push(val_ref);
        }

        let result_ref = build_einsum_fragment(&mut builder, &subscripts, &input_vals, &shapes);

        match result_ref {
            ValRef::Local(result_local) => {
                builder.set_outputs(vec![result_local]);
                let fragment = Arc::new(builder.build());

                let mut merged = HashMap::new();
                for input in inputs {
                    merged.extend(input.inputs_map.iter().map(|(k, v)| (k.clone(), v.clone())));
                }

                TracedTensor {
                    shape: out_shape,
                    dtype: inputs[0].dtype,
                    fragment,
                    val: result_local,
                    data: None,
                    inputs_map: Arc::new(merged),
                }
            }
            ValRef::External(_) => {
                // Identity pass-through: the einsum doesn't add any ops.
                // Find which input was returned and clone its TracedTensor.
                for (i, iv) in input_vals.iter().enumerate() {
                    if *iv == result_ref {
                        return TracedTensor {
                            shape: out_shape,
                            dtype: inputs[i].dtype,
                            fragment: inputs[i].fragment.clone(),
                            val: inputs[i].val,
                            data: inputs[i].data.clone(),
                            inputs_map: inputs[i].inputs_map.clone(),
                        };
                    }
                }
                panic!("build_einsum_fragment returned unrecognized external ref");
            }
        }
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

fn compute_einsum_output_shape(subscripts: &Subscripts, shapes: &[Vec<usize>]) -> Vec<usize> {
    // Build label -> size mapping from inputs
    let mut label_to_size = std::collections::HashMap::new();
    for (labels, shape) in subscripts.input_labels.iter().zip(shapes.iter()) {
        for (&label, &size) in labels.iter().zip(shape.iter()) {
            label_to_size.insert(label, size);
        }
    }
    // Output shape is determined by output_labels
    let out_shape: Vec<usize> = subscripts
        .output_labels
        .iter()
        .map(|l| {
            *label_to_size
                .get(l)
                .expect("output label not found in inputs")
        })
        .collect();
    if out_shape.is_empty() {
        vec![1]
    } else {
        out_shape
    }
}

pub fn eval_all(engine: &mut Engine, outputs: &mut [&mut TracedTensor]) -> Vec<Tensor> {
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

    let input_tensors: Vec<Tensor> = graph
        .inputs
        .iter()
        .map(|key| match key {
            GlobalValKey::Input(k) => all_inputs.get(k).expect("missing input").clone(),
            _ => panic!("expected Input key"),
        })
        .collect();

    let results = eval_exec_ir_host(&exec, input_tensors);

    for (tt, result) in outputs.iter_mut().zip(results.iter()) {
        tt.data = Some(result.clone());
    }

    let _ = engine;
    results
}
