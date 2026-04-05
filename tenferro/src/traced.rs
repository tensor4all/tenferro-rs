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
use tenferro_tensor::{DType, DotGeneralConfig, Tensor, TensorBackend, TypedTensor};
use tidu::{differentiate, transpose};

use super::compiler::{compile_to_exec, lower_to_stablehlo};
use super::engine::Engine;
use super::error::{Error, Result};
use super::exec::eval_exec_ir;

static NEXT_INPUT_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_DIFF_PASS_ID: AtomicU64 = AtomicU64::new(0);

fn next_input_key() -> TensorInputKey {
    TensorInputKey::User {
        id: NEXT_INPUT_ID.fetch_add(1, Ordering::Relaxed),
    }
}

fn next_pass_id() -> u64 {
    NEXT_DIFF_PASS_ID.fetch_add(1, Ordering::Relaxed)
}

pub struct TracedTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: LocalValId,
    pub data: Option<Tensor>,
    pub(crate) inputs_map: Arc<HashMap<TensorInputKey, Tensor>>,
    pub(crate) extra_roots: Vec<Arc<Fragment<StdTensorOp>>>,
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
            extra_roots: Vec::new(),
        }
    }

    pub fn eval<B: TensorBackend>(&mut self, engine: &mut Engine<B>) -> Result<&Tensor> {
        if self.data.is_some() {
            return Ok(self.data.as_ref().unwrap());
        }

        let output_key = self.fragment.vals()[self.val].key.clone();

        let view = resolve(self.resolve_roots());
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

    pub fn grad(&self, wrt: &TracedTensor) -> Result<TracedTensor> {
        let n_elements: usize = self.shape.iter().product();
        if n_elements != 1 {
            return Err(Error::NonScalarGrad {
                shape: self.shape.clone(),
            });
        }

        let ones = ones_tensor(self.dtype, self.shape.clone());
        let seed = TracedTensor::from_tensor(ones);
        Ok(self.vjp(wrt, &seed))
    }

    pub fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor {
        let wrt_input_key = leaf_input_key(wrt);
        let output_key = self.fragment.vals()[self.val].key.clone();
        let view = resolve(vec![self.fragment.clone()]);
        let linear = differentiate(
            &view,
            std::slice::from_ref(&output_key),
            std::slice::from_ref(&wrt_input_key),
            next_pass_id(),
        );
        let tangent_output = linear.tangent_outputs[0]
            .unwrap_or_else(|| panic!("jvp output is inactive for {:?}", wrt_input_key));
        let tangent_input_key = linear_input_key(&linear.fragment, linear.tangent_inputs[0].1);

        let mut inputs_map = (*self.inputs_map).clone();
        inputs_map.insert(
            tangent_input_key,
            tangent
                .data
                .clone()
                .unwrap_or_else(|| panic!("jvp tangent must have concrete tensor data")),
        );

        TracedTensor {
            shape: self.shape.clone(),
            dtype: self.dtype,
            fragment: Arc::new(linear.fragment),
            val: tangent_output,
            data: None,
            inputs_map: Arc::new(inputs_map),
            extra_roots: vec![self.fragment.clone()],
        }
    }

    pub fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> TracedTensor {
        let wrt_input_key = leaf_input_key(wrt);
        let output_key = self.fragment.vals()[self.val].key.clone();
        let view = resolve(vec![self.fragment.clone()]);
        let linear = differentiate(
            &view,
            std::slice::from_ref(&output_key),
            std::slice::from_ref(&wrt_input_key),
            next_pass_id(),
        );
        let transposed = transpose(&linear);
        let linear_fragment = Arc::new(linear.fragment);
        let cotangent_output = transposed.tangent_outputs[0]
            .unwrap_or_else(|| panic!("vjp output is inactive for {:?}", wrt_input_key));
        let cotangent_input_key =
            linear_input_key(&transposed.fragment, transposed.tangent_inputs[0].1);

        let mut inputs_map = (*self.inputs_map).clone();
        inputs_map.insert(
            cotangent_input_key,
            cotangent
                .data
                .clone()
                .unwrap_or_else(|| panic!("vjp cotangent must have concrete tensor data")),
        );

        TracedTensor {
            shape: wrt.shape.clone(),
            dtype: wrt.dtype,
            fragment: Arc::new(transposed.fragment),
            val: cotangent_output,
            data: None,
            inputs_map: Arc::new(inputs_map),
            extra_roots: vec![self.fragment.clone(), linear_fragment],
        }
    }

    pub fn traced_add(&self, other: &TracedTensor) -> TracedTensor {
        apply_binary(StdTensorOp::Add, self, other, self.shape.clone())
    }

    pub fn traced_mul(&self, other: &TracedTensor) -> TracedTensor {
        apply_binary(StdTensorOp::Mul, self, other, self.shape.clone())
    }

    pub fn traced_div(&self, other: &TracedTensor) -> TracedTensor {
        apply_binary(StdTensorOp::Div, self, other, self.shape.clone())
    }

    pub fn traced_neg(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Neg, self, self.shape.clone())
    }

    pub fn traced_conj(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Conj, self, self.shape.clone())
    }

    pub fn traced_abs(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Abs, self, self.shape.clone())
    }

    pub fn traced_sign(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sign, self, self.shape.clone())
    }

    pub fn traced_exp(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Exp, self, self.shape.clone())
    }

    pub fn traced_log(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Log, self, self.shape.clone())
    }

    pub fn traced_sin(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sin, self, self.shape.clone())
    }

    pub fn traced_cos(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Cos, self, self.shape.clone())
    }

    pub fn traced_tanh(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Tanh, self, self.shape.clone())
    }

    pub fn traced_sqrt(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sqrt, self, self.shape.clone())
    }

    pub fn traced_rsqrt(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Rsqrt, self, self.shape.clone())
    }

    pub fn traced_pow(&self, other: &TracedTensor) -> TracedTensor {
        apply_binary(StdTensorOp::Pow, self, other, self.shape.clone())
    }

    pub fn traced_expm1(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Expm1, self, self.shape.clone())
    }

    pub fn traced_log1p(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Log1p, self, self.shape.clone())
    }

    pub fn traced_dot_general(
        &self,
        other: &TracedTensor,
        config: DotGeneralConfig,
    ) -> TracedTensor {
        let lhs_free: Vec<usize> = (0..config.lhs_rank)
            .filter(|d| {
                !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d)
            })
            .collect();
        let rhs_free: Vec<usize> = (0..config.rhs_rank)
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

        apply_binary(StdTensorOp::DotGeneral(config), self, other, out_shape)
    }

    pub fn traced_reduce_sum(&self, axes: &[usize]) -> TracedTensor {
        let out_shape: Vec<usize> = (0..self.shape.len())
            .filter(|d| !axes.contains(d))
            .map(|d| self.shape[d])
            .collect();
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

    pub fn traced_extract_diag(&self, axis_a: usize, axis_b: usize) -> TracedTensor {
        assert!(
            axis_a < self.shape.len() && axis_b < self.shape.len() && axis_a != axis_b,
            "extract_diag: invalid axes"
        );
        let out_shape = self
            .shape
            .iter()
            .enumerate()
            .filter_map(|(axis, &dim)| (axis != axis_b).then_some(dim))
            .collect();
        apply_unary(StdTensorOp::ExtractDiag { axis_a, axis_b }, self, out_shape)
    }

    pub fn traced_embed_diag(&self, axis_a: usize, axis_b: usize) -> TracedTensor {
        assert!(
            axis_a < self.shape.len() && axis_b <= self.shape.len(),
            "embed_diag: invalid axes"
        );
        let mut out_shape = self.shape.clone();
        out_shape.insert(axis_b, self.shape[axis_a]);
        apply_unary(StdTensorOp::EmbedDiag { axis_a, axis_b }, self, out_shape)
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
        extra_roots: Vec::new(),
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
        extra_roots: Vec::new(),
    }
}

impl TracedTensor {
    fn resolve_roots(&self) -> Vec<Arc<Fragment<StdTensorOp>>> {
        let mut roots = Vec::with_capacity(1 + self.extra_roots.len());
        roots.push(self.fragment.clone());
        roots.extend(self.extra_roots.iter().cloned());
        roots
    }
}

fn leaf_input_key(tt: &TracedTensor) -> TensorInputKey {
    match &tt.fragment.vals()[tt.val].key {
        GlobalValKey::Input(key) => key.clone(),
        other => panic!("expected traced leaf input, got {:?}", other),
    }
}

fn linear_input_key(fragment: &Fragment<StdTensorOp>, local_id: LocalValId) -> TensorInputKey {
    match &fragment.vals()[local_id].key {
        GlobalValKey::Input(key) => key.clone(),
        other => panic!("expected linear fragment input, got {:?}", other),
    }
}

fn ones_tensor(dtype: DType, shape: Vec<usize>) -> Tensor {
    match dtype {
        DType::F32 => Tensor::F32(TypedTensor::ones(shape)),
        DType::F64 => Tensor::F64(TypedTensor::ones(shape)),
        DType::C32 => Tensor::C32(TypedTensor::ones(shape)),
        DType::C64 => Tensor::C64(TypedTensor::ones(shape)),
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
        all_fragments.extend(tt.resolve_roots());
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
