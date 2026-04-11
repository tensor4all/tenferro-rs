use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::LocalValId;
use num_complex::{Complex32, Complex64};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, DotGeneralConfig, Tensor, TensorBackend, TensorScalar, TypedTensor};
use tidu::{differentiate, transpose};

use super::compiler::{compile_to_exec, lower_to_stablehlo};
use super::engine::Engine;
use super::error::{Error, Result};
use super::exec::eval_exec_ir;
use super::sym_dim::SymDim;

static NEXT_INPUT_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_DIFF_PASS_ID: AtomicU64 = AtomicU64::new(0);
static NEXT_TRACED_ID: AtomicU64 = AtomicU64::new(0);

pub type TracedTensorId = u64;

fn next_input_key() -> TensorInputKey {
    TensorInputKey::User {
        id: NEXT_INPUT_ID.fetch_add(1, Ordering::Relaxed),
    }
}

fn next_pass_id() -> u64 {
    NEXT_DIFF_PASS_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn next_traced_id() -> TracedTensorId {
    NEXT_TRACED_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone)]
pub struct TracedTensor {
    pub id: TracedTensorId,
    pub rank: usize,
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: LocalValId,
    pub data: Option<Tensor>,
    pub(crate) shape_hint: Option<Vec<usize>>,
    pub(crate) inputs_map: Arc<HashMap<TensorInputKey, Tensor>>,
    pub(crate) extra_roots: Vec<Arc<Fragment<StdTensorOp>>>,
}

/// Compute a broadcast output shape following NumPy rules.
///
/// Returns `None` when the two shapes are incompatible.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut result = Vec::with_capacity(rank);
    for index in 0..rank {
        let a_dim = if index < rank - a.len() {
            1
        } else {
            a[index - (rank - a.len())]
        };
        let b_dim = if index < rank - b.len() {
            1
        } else {
            b[index - (rank - b.len())]
        };
        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return None;
        }
    }
    Some(result)
}

fn known_shape(tensor: &TracedTensor) -> &[usize] {
    tensor.shape_hint.as_deref().unwrap_or_else(|| {
        panic!(
            "missing concrete shape hint for traced tensor {}",
            tensor.id
        )
    })
}

fn error_shape_hint(tensor: &TracedTensor) -> Vec<usize> {
    tensor
        .shape_hint
        .clone()
        .unwrap_or_else(|| vec![0; tensor.rank])
}

/// Broadcast a traced tensor to `target_shape`.
///
/// Expanding singleton axes are first reshaped away so the existing
/// `BroadcastInDim` transpose rule reduces them correctly during VJP.
fn broadcast_to(tensor: &TracedTensor, target_shape: &[usize]) -> TracedTensor {
    if known_shape(tensor) == target_shape {
        return tensor.clone();
    }

    assert!(
        tensor.rank <= target_shape.len(),
        "cannot broadcast higher-rank shape {:?} to {:?}",
        known_shape(tensor),
        target_shape
    );

    let tensor_shape = known_shape(tensor);
    let rank_diff = target_shape.len() - tensor.rank;
    let mut source_shape = Vec::with_capacity(tensor.rank);
    let mut dims = Vec::with_capacity(tensor.rank);
    for (src_axis, &src_dim) in tensor_shape.iter().enumerate() {
        let dst_axis = src_axis + rank_diff;
        let dst_dim = target_shape[dst_axis];
        assert!(
            src_dim == dst_dim || src_dim == 1,
            "cannot broadcast shape {:?} to {:?}",
            tensor_shape,
            target_shape
        );
        if src_dim == 1 && dst_dim != 1 {
            continue;
        }
        source_shape.push(src_dim);
        dims.push(dst_axis);
    }

    let source = if source_shape == tensor_shape {
        tensor.clone()
    } else {
        tensor.reshape(&source_shape)
    };
    source.broadcast_in_dim(target_shape, &dims)
}

/// Broadcast two tensors to a common shape.
fn broadcast_binary(a: &TracedTensor, b: &TracedTensor) -> (TracedTensor, TracedTensor) {
    if a.shape_hint == b.shape_hint && a.rank == b.rank {
        return (a.clone(), b.clone());
    }
    let a_shape = known_shape(a);
    let b_shape = known_shape(b);
    let target = broadcast_shape(a_shape, b_shape).unwrap_or_else(|| {
        panic!(
            "incompatible shapes for broadcast: {:?} and {:?}",
            a_shape, b_shape
        )
    });
    (broadcast_to(a, &target), broadcast_to(b, &target))
}

fn scale_with_constant(input: &TracedTensor, op: StdTensorOp) -> TracedTensor {
    let scalar = apply_nullary(op, 0, input.dtype, Some(vec![]));
    let factor = broadcast_to(&scalar, known_shape(input));
    apply_binary(
        StdTensorOp::Mul,
        input,
        &factor,
        input.rank,
        input.shape_hint.clone(),
    )
}

impl std::ops::Add for &TracedTensor {
    type Output = TracedTensor;

    fn add(self, rhs: &TracedTensor) -> TracedTensor {
        TracedTensor::add(self, rhs)
    }
}

impl std::ops::Mul for &TracedTensor {
    type Output = TracedTensor;

    fn mul(self, rhs: &TracedTensor) -> TracedTensor {
        TracedTensor::mul(self, rhs)
    }
}

impl std::ops::Mul<f64> for &TracedTensor {
    type Output = TracedTensor;

    fn mul(self, rhs: f64) -> TracedTensor {
        self.scale_real(rhs)
    }
}

impl std::ops::Mul<&TracedTensor> for f64 {
    type Output = TracedTensor;

    fn mul(self, rhs: &TracedTensor) -> TracedTensor {
        rhs.scale_real(self)
    }
}

impl std::ops::Neg for &TracedTensor {
    type Output = TracedTensor;

    fn neg(self) -> TracedTensor {
        TracedTensor::neg(self)
    }
}

impl std::ops::Div for &TracedTensor {
    type Output = TracedTensor;

    fn div(self, rhs: &TracedTensor) -> TracedTensor {
        TracedTensor::div(self, rhs)
    }
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
            id: next_traced_id(),
            rank: shape.len(),
            dtype,
            fragment,
            val,
            data: Some(tensor),
            shape_hint: Some(shape),
            inputs_map: Arc::new(map),
            extra_roots: Vec::new(),
        }
    }

    /// Create a traced tensor from shape and data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::TracedTensor;
    ///
    /// let a = TracedTensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(a.rank, 2);
    /// ```
    pub fn new<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
        Self::from_tensor(T::into_tensor(shape, data))
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
        let mut results = eval_exec_ir(&mut engine.backend, &cached_exec, input_tensors)?;
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
        if self.rank != 0 {
            return Err(Error::NonScalarGrad {
                shape: error_shape_hint(self),
            });
        }

        let ones = ones_tensor(self.dtype, vec![]);
        let seed = TracedTensor::from_tensor(ones);
        Ok(self.vjp(wrt, &seed))
    }

    /// Like [`grad`](Self::grad) but returns `None` when the scalar output does
    /// not depend on `wrt`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let maybe_dx = loss.try_grad(&x)?;
    /// ```
    pub fn try_grad(&self, wrt: &TracedTensor) -> Result<Option<TracedTensor>> {
        if self.rank != 0 {
            return Err(Error::NonScalarGrad {
                shape: error_shape_hint(self),
            });
        }

        let ones = ones_tensor(self.dtype, vec![]);
        let seed = TracedTensor::from_tensor(ones);
        Ok(self.try_vjp(wrt, &seed))
    }

    pub fn jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> TracedTensor {
        self.try_jvp(wrt, tangent)
            .unwrap_or_else(|| panic!("jvp output is inactive for {:?}", leaf_input_key(wrt)))
    }

    /// Like [`jvp`](Self::jvp) but returns `None` when the output does not
    /// depend on `wrt` (i.e. the tangent is structurally zero).
    pub fn try_jvp(&self, wrt: &TracedTensor, tangent: &TracedTensor) -> Option<TracedTensor> {
        let wrt_input_key = leaf_input_key(wrt);
        let output_key = self.fragment.vals()[self.val].key.clone();
        let view = resolve(self.resolve_roots());
        let linear = differentiate(
            &view,
            std::slice::from_ref(&output_key),
            std::slice::from_ref(&wrt_input_key),
            next_pass_id(),
        );
        let tangent_output = linear.tangent_outputs[0]?;
        let tangent_input_key = linear_input_key(&linear.fragment, linear.tangent_inputs[0].1);

        let mut inputs_map = (*self.inputs_map).clone();
        inputs_map.insert(
            tangent_input_key,
            tangent
                .data
                .clone()
                .unwrap_or_else(|| panic!("jvp tangent must have concrete tensor data")),
        );

        let mut extra_roots = vec![self.fragment.clone()];
        extra_roots.extend(self.extra_roots.iter().cloned());

        Some(TracedTensor {
            id: next_traced_id(),
            rank: self.rank,
            dtype: self.dtype,
            fragment: Arc::new(linear.fragment),
            val: tangent_output,
            data: None,
            shape_hint: self.shape_hint.clone(),
            inputs_map: Arc::new(inputs_map),
            extra_roots,
        })
    }

    pub fn vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> TracedTensor {
        self.try_vjp(wrt, cotangent)
            .unwrap_or_else(|| panic!("vjp output is inactive for {:?}", leaf_input_key(wrt)))
    }

    fn try_vjp(&self, wrt: &TracedTensor, cotangent: &TracedTensor) -> Option<TracedTensor> {
        let wrt_input_key = leaf_input_key(wrt);
        let output_key = self.fragment.vals()[self.val].key.clone();
        let view = resolve(self.resolve_roots());
        let linear = differentiate(
            &view,
            std::slice::from_ref(&output_key),
            std::slice::from_ref(&wrt_input_key),
            next_pass_id(),
        );
        let linear_tangent_input_ids: Vec<LocalValId> = linear
            .tangent_inputs
            .iter()
            .map(|(_, local_id)| *local_id)
            .collect();
        let transposed = transpose(&linear);
        let linear_fragment = Arc::new(linear.fragment);
        let cotangent_output = transposed.tangent_outputs[0]?;
        let cotangent_input_key =
            linear_input_key(&transposed.fragment, transposed.tangent_inputs[0].1);

        let mut inputs_map = (*self.inputs_map).clone();
        inputs_map.insert(
            cotangent_input_key.clone(),
            cotangent
                .data
                .clone()
                .unwrap_or_else(|| panic!("vjp cotangent must have concrete tensor data")),
        );
        let zero_tangent = zeros_tensor(
            wrt.dtype,
            wrt.shape_hint.clone().unwrap_or_else(|| vec![0; wrt.rank]),
        );
        for (_, local_id) in &transposed.tangent_inputs {
            let tangent_input_key = linear_input_key(&transposed.fragment, *local_id);
            if tangent_input_key != cotangent_input_key {
                inputs_map.insert(tangent_input_key, zero_tangent.clone());
            }
        }
        for local_id in linear_tangent_input_ids {
            let tangent_input_key = linear_input_key(&linear_fragment, local_id);
            inputs_map.insert(tangent_input_key, zero_tangent.clone());
        }

        let mut extra_roots = vec![self.fragment.clone(), linear_fragment];
        extra_roots.extend(self.extra_roots.iter().cloned());

        Some(TracedTensor {
            id: next_traced_id(),
            rank: wrt.rank,
            dtype: wrt.dtype,
            fragment: Arc::new(transposed.fragment),
            val: cotangent_output,
            data: None,
            shape_hint: wrt.shape_hint.clone(),
            inputs_map: Arc::new(inputs_map),
            extra_roots,
        })
    }

    /// Elementwise addition with NumPy-style broadcasting.
    ///
    /// Prefer using the `+` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.add(&z);
    /// let y2 = &x + &z;
    /// ```
    pub fn add(&self, other: &TracedTensor) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
        apply_binary(
            StdTensorOp::Add,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise multiplication with NumPy-style broadcasting.
    ///
    /// Prefer using the `*` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.mul(&z);
    /// let y2 = &x * &z;
    /// ```
    pub fn mul(&self, other: &TracedTensor) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
        apply_binary(
            StdTensorOp::Mul,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise division with NumPy-style broadcasting.
    ///
    /// Prefer using the `/` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.div(&z);
    /// let y2 = &x / &z;
    /// ```
    pub fn div(&self, other: &TracedTensor) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
        apply_binary(
            StdTensorOp::Div,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise negation.
    ///
    /// Prefer using the unary `-` operator when it reads naturally.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.neg();
    /// let y2 = -&x;
    /// ```
    pub fn neg(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Neg, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise complex conjugate.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.conj();
    /// ```
    pub fn conj(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Conj, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise absolute value.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.abs();
    /// ```
    pub fn abs(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Abs, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise sign.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.sign();
    /// ```
    pub fn sign(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sign, self, self.rank, self.shape_hint.clone())
    }

    /// Scale by a real scalar: `y = factor * x`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.scale_real(2.0);
    /// ```
    pub fn scale_real(&self, factor: f64) -> TracedTensor {
        let op = match self.dtype {
            DType::F64 => StdTensorOp::constant_f64(factor),
            DType::F32 => StdTensorOp::constant_f32(factor as f32),
            DType::C64 => StdTensorOp::constant_c64(Complex64::new(factor, 0.0)),
            DType::C32 => StdTensorOp::constant_c32(Complex32::new(factor as f32, 0.0)),
        };
        scale_with_constant(self, op)
    }

    /// Scale by a complex scalar: `y = factor * x`.
    ///
    /// This currently supports complex tensors only. For real scaling, prefer
    /// [`scale_real`](Self::scale_real).
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use num_complex::Complex64;
    /// let y = x.scale_complex(Complex64::new(0.0, 1.0)); // multiply by i
    /// ```
    pub fn scale_complex(&self, factor: Complex64) -> TracedTensor {
        match self.dtype {
            DType::C64 => scale_with_constant(self, StdTensorOp::constant_c64(factor)),
            DType::C32 => scale_with_constant(
                self,
                StdTensorOp::constant_c32(Complex32::new(factor.re as f32, factor.im as f32)),
            ),
            DType::F32 | DType::F64 => {
                panic!(
                    "scale_complex only supports complex tensors; use scale_real for real tensors"
                )
            }
        }
    }

    /// Elementwise exponential.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.exp();
    /// ```
    pub fn exp(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Exp, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise natural logarithm.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.log();
    /// ```
    pub fn log(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Log, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise sine.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.sin();
    /// ```
    pub fn sin(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sin, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise cosine.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.cos();
    /// ```
    pub fn cos(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Cos, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise hyperbolic tangent.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.tanh();
    /// ```
    pub fn tanh(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Tanh, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise square root.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.sqrt();
    /// ```
    pub fn sqrt(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Sqrt, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise reciprocal square root.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.rsqrt();
    /// ```
    pub fn rsqrt(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Rsqrt, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise power with NumPy-style broadcasting.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = base.pow(&exp);
    /// ```
    pub fn pow(&self, other: &TracedTensor) -> TracedTensor {
        let (lhs, rhs) = broadcast_binary(self, other);
        apply_binary(
            StdTensorOp::Pow,
            &lhs,
            &rhs,
            lhs.rank,
            lhs.shape_hint.clone(),
        )
    }

    /// Elementwise `exp(x) - 1`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.expm1();
    /// ```
    pub fn expm1(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Expm1, self, self.rank, self.shape_hint.clone())
    }

    /// Elementwise `log(1 + x)`.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.log1p();
    /// ```
    pub fn log1p(&self) -> TracedTensor {
        apply_unary(StdTensorOp::Log1p, self, self.rank, self.shape_hint.clone())
    }

    /// Convert the tensor to a different dtype.
    ///
    /// For real-to-complex conversions this embeds the real values as
    /// `x + 0i`. For complex-to-real conversions this extracts the real part.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use tenferro::DType;
    ///
    /// let y = x.convert(DType::C64);
    /// ```
    pub fn convert(&self, to: DType) -> TracedTensor {
        if self.dtype == to {
            return self.clone();
        }

        apply_unary_with_dtype(
            StdTensorOp::Convert {
                from: self.dtype,
                to,
            },
            self,
            self.rank,
            self.shape_hint.clone(),
            to,
        )
    }

    /// Generalized tensor contraction.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = a.dot_general(&b, config);
    /// ```
    pub fn dot_general(&self, other: &TracedTensor, config: DotGeneralConfig) -> TracedTensor {
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
        let out_rank = config.lhs_batch_dims.len() + lhs_free.len() + rhs_free.len();
        let out_shape_hint = match (&self.shape_hint, &other.shape_hint) {
            (Some(lhs_shape), Some(rhs_shape)) => {
                let mut out_shape = Vec::with_capacity(out_rank);
                for &d in &lhs_free {
                    out_shape.push(lhs_shape[d]);
                }
                for &d in &rhs_free {
                    out_shape.push(rhs_shape[d]);
                }
                for &d in &config.lhs_batch_dims {
                    out_shape.push(lhs_shape[d]);
                }
                Some(out_shape)
            }
            _ => None,
        };

        apply_binary(
            StdTensorOp::DotGeneral(config),
            self,
            other,
            out_rank,
            out_shape_hint,
        )
    }

    /// Sum over the given axes.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.reduce_sum(&[0]);
    /// let y2 = x.sum(&[0]);
    /// ```
    pub fn reduce_sum(&self, axes: &[usize]) -> TracedTensor {
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            (0..shape.len())
                .filter(|d| !axes.contains(d))
                .map(|d| shape[d])
                .collect()
        });
        apply_unary(
            StdTensorOp::ReduceSum {
                axes: axes.to_vec(),
                input_shape: DimExpr::input_shape(0, self.rank),
            },
            self,
            self.rank - axes.len(),
            out_shape_hint,
        )
    }

    /// Reshape without changing element order.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.reshape(&[2, 2]);
    /// ```
    pub fn reshape(&self, shape: &[usize]) -> TracedTensor {
        apply_unary(
            StdTensorOp::Reshape {
                from_shape: DimExpr::input_shape(0, self.rank),
                to_shape: DimExpr::from_concrete(shape),
            },
            self,
            shape.len(),
            Some(shape.to_vec()),
        )
    }

    /// Return a symbolic expression for the size of one axis.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let rows = x.sym_size(0);
    /// let cols = x.sym_size(1);
    /// let y = x.reshape_sym(&[rows * cols])?;
    /// ```
    pub fn sym_size(&self, axis: usize) -> SymDim {
        assert!(
            axis < self.rank,
            "axis {axis} out of bounds for rank {}",
            self.rank
        );
        SymDim::tensor_axis(self.id, axis)
    }

    /// Reshape using symbolic dimensions derived from traced tensor axes.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let rows = x.sym_size(0);
    /// let cols = x.sym_size(1);
    /// let y = x.reshape_sym(&[rows * cols])?;
    /// ```
    pub fn reshape_sym(&self, shape: &[SymDim]) -> Result<TracedTensor> {
        let tensor_map = [(self.id, 0usize)];
        let to_shape = shape
            .iter()
            .map(|dim| dim.to_dim_expr(&tensor_map).map_err(Error::Internal))
            .collect::<Result<Vec<_>>>()?;
        let out_shape_hint = shape
            .iter()
            .map(SymDim::constant_value)
            .collect::<Option<Vec<_>>>();
        Ok(apply_unary(
            StdTensorOp::Reshape {
                from_shape: DimExpr::input_shape(0, self.rank),
                to_shape,
            },
            self,
            shape.len(),
            out_shape_hint,
        ))
    }

    /// Broadcast into a larger shape with explicit dimension placement.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.broadcast_in_dim(&[2, 3], &[1]);
    /// let y2 = x.broadcast(&[2, 3], &[1]);
    /// ```
    pub fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> TracedTensor {
        apply_unary(
            StdTensorOp::BroadcastInDim {
                shape: DimExpr::from_concrete(shape),
                dims: dims.to_vec(),
            },
            self,
            shape.len(),
            Some(shape.to_vec()),
        )
    }

    /// Permute tensor axes.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.transpose(&[1, 0]);
    /// ```
    pub fn transpose(&self, perm: &[usize]) -> TracedTensor {
        let out_shape_hint = self
            .shape_hint
            .as_ref()
            .map(|shape| perm.iter().map(|&p| shape[p]).collect());
        apply_unary(
            StdTensorOp::Transpose {
                perm: perm.to_vec(),
            },
            self,
            self.rank,
            out_shape_hint,
        )
    }

    /// Extract the diagonal along two axes.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.extract_diag(0, 1);
    /// ```
    pub fn extract_diag(&self, axis_a: usize, axis_b: usize) -> TracedTensor {
        assert!(
            axis_a < self.rank && axis_b < self.rank && axis_a != axis_b,
            "extract_diag: invalid axes"
        );
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            shape
                .iter()
                .enumerate()
                .filter_map(|(axis, &dim)| (axis != axis_b).then_some(dim))
                .collect()
        });
        apply_unary(
            StdTensorOp::ExtractDiag { axis_a, axis_b },
            self,
            self.rank - 1,
            out_shape_hint,
        )
    }

    /// Embed a vector or lower-rank tensor along a diagonal.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.embed_diag(0, 1);
    /// ```
    pub fn embed_diag(&self, axis_a: usize, axis_b: usize) -> TracedTensor {
        assert!(
            axis_a < self.rank && axis_b <= self.rank,
            "embed_diag: invalid axes"
        );
        let out_shape_hint = self.shape_hint.as_ref().map(|shape| {
            let mut out_shape = shape.clone();
            out_shape.insert(axis_b, shape[axis_a]);
            out_shape
        });
        apply_unary(
            StdTensorOp::EmbedDiag { axis_a, axis_b },
            self,
            self.rank + 1,
            out_shape_hint,
        )
    }

    /// Alias for [`Self::reduce_sum`].
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.sum(&[0]);
    /// ```
    pub fn sum(&self, axes: &[usize]) -> TracedTensor {
        self.reduce_sum(axes)
    }

    /// Alias for [`Self::broadcast_in_dim`].
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let y = x.broadcast(&[2, 3], &[1]);
    /// ```
    pub fn broadcast(&self, shape: &[usize], dims: &[usize]) -> TracedTensor {
        self.broadcast_in_dim(shape, dims)
    }
}

pub(crate) fn apply_unary(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<usize>>,
) -> TracedTensor {
    apply_unary_with_dtype(op, input, out_rank, out_shape_hint, input.dtype)
}

pub(crate) fn apply_unary_with_dtype(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<usize>>,
    out_dtype: DType,
) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    builder.add_parent(input.fragment.clone());
    let input_ref = ValRef::External(input.fragment.vals()[input.val].key.clone());
    let outputs = builder.add_op(op, vec![input_ref], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());

    TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: out_dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: input.inputs_map.clone(),
        extra_roots: input.extra_roots.clone(),
    }
}

pub(crate) fn apply_nullary(
    op: StdTensorOp,
    rank: usize,
    dtype: DType,
    shape_hint: Option<Vec<usize>>,
) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    let outputs = builder.add_op(op, vec![], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());

    TracedTensor {
        id: next_traced_id(),
        rank,
        dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint,
        inputs_map: Arc::new(HashMap::new()),
        extra_roots: Vec::new(),
    }
}

pub(crate) fn apply_binary(
    op: StdTensorOp,
    lhs: &TracedTensor,
    rhs: &TracedTensor,
    out_rank: usize,
    out_shape_hint: Option<Vec<usize>>,
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
    let mut extra_roots = lhs.extra_roots.clone();
    extra_roots.extend(rhs.extra_roots.iter().cloned());

    TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: lhs.dtype,
        fragment,
        val: outputs[0],
        data: None,
        shape_hint: out_shape_hint,
        inputs_map: Arc::new(merged),
        extra_roots,
    }
}

pub(crate) fn apply_multi_output(
    op: StdTensorOp,
    input: &TracedTensor,
    output_shapes: Vec<Vec<usize>>,
) -> Vec<TracedTensor> {
    let mut builder = FragmentBuilder::new();
    builder.add_parent(input.fragment.clone());
    let input_ref = ValRef::External(input.fragment.vals()[input.val].key.clone());
    let outputs = builder.add_op(op, vec![input_ref], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());
    assert_eq!(
        outputs.len(),
        output_shapes.len(),
        "apply_multi_output: output count must match output_shapes"
    );

    outputs
        .iter()
        .zip(output_shapes)
        .map(|(&val, shape)| TracedTensor {
            id: next_traced_id(),
            rank: shape.len(),
            dtype: input.dtype,
            fragment: fragment.clone(),
            val,
            data: None,
            shape_hint: Some(shape),
            inputs_map: input.inputs_map.clone(),
            extra_roots: input.extra_roots.clone(),
        })
        .collect()
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

fn zeros_tensor(dtype: DType, shape: Vec<usize>) -> Tensor {
    match dtype {
        DType::F32 => Tensor::F32(TypedTensor::zeros(shape)),
        DType::F64 => Tensor::F64(TypedTensor::zeros(shape)),
        DType::C32 => Tensor::C32(TypedTensor::zeros(shape)),
        DType::C64 => Tensor::C64(TypedTensor::zeros(shape)),
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
    let results: Vec<Tensor> = eval_exec_ir(&mut engine.backend, &cached_exec, input_tensors)?;

    for (tt, result) in outputs.iter_mut().zip(results.iter()) {
        tt.data = Some(result.clone());
    }

    Ok(results)
}
