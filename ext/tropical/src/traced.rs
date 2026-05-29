//! Traced tropical composition helpers.
//!
//! These helpers expose both compositional tropical graph operations and fused
//! extension-backed tropical einsum operations.
//!
//! # Examples
//!
//! ```
//! use tenferro_cpu::CpuBackend;
//! use tenferro_ext_tropical::traced::tropical_reduce_sum;
//! use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
//!
//! let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 5.0, 2.0]);
//! let y = tropical_reduce_sum(&x, &[0]);
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let mut executor = GraphExecutor::new(CpuBackend::new());
//! let out = executor.run(&program).unwrap();
//! assert_eq!(out.as_slice::<f64>().unwrap(), &[5.0]);
//! ```

use std::sync::Arc;

use tenferro_einsum::Subscripts;
use tenferro_runtime::{extension, Error, Result, TracedTensor};

use crate::extension::TropicalEinsumOp;
use crate::TropicalKind;

fn require_rank2(tensor: &TracedTensor, label: &str) {
    assert_eq!(
        tensor.rank, 2,
        "{label}: tropical matrix composition requires rank-2 input, got rank {}",
        tensor.rank
    );
}

fn assert_contracting_axes_compatible(a: &TracedTensor, b: &TracedTensor, label: &str) {
    if let (Some(a_shape), Some(b_shape)) = (a.try_concrete_shape(), b.try_concrete_shape()) {
        assert_eq!(
            a_shape[1], b_shape[0],
            "{label}: contracting axes must match, got a[1]={} and b[0]={}",
            a_shape[1], b_shape[0]
        );
    }
}

fn tropical_dot_general_impl(
    a: &TracedTensor,
    b: &TracedTensor,
    reduce: impl FnOnce(&TracedTensor) -> TracedTensor,
    label: &str,
) -> TracedTensor {
    require_rank2(a, &format!("{label}.a"));
    require_rank2(b, &format!("{label}.b"));
    assert_contracting_axes_compatible(a, b, label);

    let m = a.axis_sym_dim(0);
    let k = a.axis_sym_dim(1);
    let n = b.axis_sym_dim(1);
    let target_shape = [m, k, n];

    let a_b = a.broadcast_in_dim_sym(&target_shape, &[0, 1], &[b]);
    let b_b = b.broadcast_in_dim_sym(&target_shape, &[1, 2], &[a]);
    let sum = a_b.add(&b_b);
    reduce(&sum)
}

/// Max-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = max_k(a[i, k] + b[k, j])` by composing
/// `BroadcastInDim`, `Add`, and `ReduceMax` graph operations.
///
/// # Panics
///
/// Panics if either input is not rank 2. If both input shapes are concrete,
/// also panics when the contracting axes differ.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::tropical_dot_general;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
/// let c = tropical_dot_general(&a, &b);
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&c).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[23.0, 24.0, 43.0, 44.0]);
/// ```
#[must_use]
pub fn tropical_dot_general(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    tropical_dot_general_impl(a, b, |sum| sum.reduce_max(&[1]), "tropical_dot_general")
}

/// Min-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = min_k(a[i, k] + b[k, j])` by composing
/// `BroadcastInDim`, `Add`, and `ReduceMin` graph operations.
///
/// # Panics
///
/// Panics if either input is not rank 2. If both input shapes are concrete,
/// also panics when the contracting axes differ.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::min_plus_dot_general;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
/// let c = min_plus_dot_general(&a, &b);
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&c).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[6.0, 7.0, 8.0, 9.0]);
/// ```
#[must_use]
pub fn min_plus_dot_general(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    tropical_dot_general_impl(a, b, |sum| sum.reduce_min(&[1]), "min_plus_dot_general")
}

/// Fused binary tropical einsum over traced tensors.
///
/// The operation is carried as a runtime extension and uses the tropical
/// value-plus-argmax executor during forward execution.
///
/// # Errors
///
/// Returns an error when the notation is invalid or outside the supported
/// two-input tropical contraction surface.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::{traced::tropical_einsum, TropicalKind};
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
/// let out = tropical_einsum(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik").unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&out).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_ext_tropical::register_runtime).unwrap();
/// let value = executor.run(&program).unwrap();
/// assert_eq!(value.as_slice::<f64>().unwrap(), &[3.0]);
/// ```
pub fn tropical_einsum(
    kind: TropicalKind,
    inputs: &[&TracedTensor],
    notation: &str,
) -> Result<TracedTensor> {
    let subscripts = Subscripts::parse(notation).map_err(|err| {
        Error::InvalidSubscripts(format!("invalid tropical einsum notation: {err}"))
    })?;
    tropical_einsum_subscripts(kind, inputs, &subscripts)
}

/// Fused binary tropical einsum over traced tensors with parsed subscripts.
///
/// This variant is useful when the caller already owns
/// [`tenferro_einsum::Subscripts`] and wants to avoid a string roundtrip.
///
/// # Errors
///
/// Returns an error when the parsed subscripts are outside the supported
/// two-input tropical contraction surface.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::Subscripts;
/// use tenferro_ext_tropical::{traced::tropical_einsum_subscripts, TropicalKind};
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]);
/// let subscripts = Subscripts::parse("ij,jk->ik").unwrap();
/// let out = tropical_einsum_subscripts(TropicalKind::MaxPlus, &[&a, &b], &subscripts).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&out).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_ext_tropical::register_runtime).unwrap();
/// let value = executor.run(&program).unwrap();
/// assert_eq!(value.as_slice::<f64>().unwrap(), &[3.0]);
/// ```
pub fn tropical_einsum_subscripts(
    kind: TropicalKind,
    inputs: &[&TracedTensor],
    subscripts: &Subscripts,
) -> Result<TracedTensor> {
    validate_tropical_einsum_inputs(inputs, subscripts)?;
    let outputs = extension::apply(
        Arc::new(TropicalEinsumOp::new(kind, subscripts.clone())),
        inputs,
    );
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| Error::Internal("tropical einsum extension produced no output".to_string()))
}

/// Fused max-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = max_k(a[i, k] + b[k, j])` through the tropical
/// extension executor rather than a broadcast/reduce composition.
///
/// # Panics
///
/// Panics if either input is not rank 2 or the concrete contracting dimensions
/// are incompatible.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::tropical_dot_general_fused;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]);
/// let out = tropical_dot_general_fused(&a, &b);
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&out).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_ext_tropical::register_runtime).unwrap();
/// let value = executor.run(&program).unwrap();
/// assert_eq!(value.as_slice::<f64>().unwrap(), &[23.0, 24.0, 43.0, 44.0]);
/// ```
#[must_use]
pub fn tropical_dot_general_fused(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    fused_dot_general_impl(TropicalKind::MaxPlus, a, b, "tropical_dot_general_fused")
}

/// Fused min-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = min_k(a[i, k] + b[k, j])` through the tropical
/// extension executor rather than a broadcast/reduce composition.
///
/// # Panics
///
/// Panics if either input is not rank 2 or the concrete contracting dimensions
/// are incompatible.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::min_plus_dot_general_fused;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]);
/// let out = min_plus_dot_general_fused(&a, &b);
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&out).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_ext_tropical::register_runtime).unwrap();
/// let value = executor.run(&program).unwrap();
/// assert_eq!(value.as_slice::<f64>().unwrap(), &[6.0, 7.0, 8.0, 9.0]);
/// ```
#[must_use]
pub fn min_plus_dot_general_fused(a: &TracedTensor, b: &TracedTensor) -> TracedTensor {
    fused_dot_general_impl(TropicalKind::MinPlus, a, b, "min_plus_dot_general_fused")
}

fn fused_dot_general_impl(
    kind: TropicalKind,
    a: &TracedTensor,
    b: &TracedTensor,
    label: &str,
) -> TracedTensor {
    require_rank2(a, &format!("{label}.a"));
    require_rank2(b, &format!("{label}.b"));
    assert_contracting_axes_compatible(a, b, label);
    let subscripts = Subscripts::new(
        &[&[b'i' as u32, b'j' as u32], &[b'j' as u32, b'k' as u32]],
        &[b'i' as u32, b'k' as u32],
    );
    tropical_einsum_subscripts(kind, &[a, b], &subscripts).unwrap_or_else(|err| panic!("{err}"))
}

fn validate_tropical_einsum_inputs(
    inputs: &[&TracedTensor],
    subscripts: &Subscripts,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::ContractionError(format!(
            "tropical einsum supports exactly two inputs, got {}",
            inputs.len()
        )));
    }
    if subscripts.inputs.len() != 2 {
        return Err(Error::ContractionError(format!(
            "tropical einsum subscripts describe {} inputs, expected 2",
            subscripts.inputs.len()
        )));
    }
    for (input_idx, (labels, tensor)) in subscripts.inputs.iter().zip(inputs).enumerate() {
        if labels.len() != tensor.rank {
            return Err(Error::ContractionError(format!(
                "tropical einsum input {input_idx} rank mismatch: labels={}, rank={}",
                labels.len(),
                tensor.rank
            )));
        }
        if labels
            .iter()
            .enumerate()
            .any(|(idx, label)| labels[..idx].contains(label))
        {
            return Err(Error::ContractionError(format!(
                "tropical einsum input {input_idx} repeated labels are not supported"
            )));
        }
    }
    if subscripts
        .output
        .iter()
        .enumerate()
        .any(|(idx, label)| subscripts.output[..idx].contains(label))
    {
        return Err(Error::ContractionError(
            "tropical einsum repeated output labels are not supported".to_string(),
        ));
    }
    let has_contracted = subscripts.inputs[0]
        .iter()
        .any(|label| subscripts.inputs[1].contains(label) && !subscripts.output.contains(label));
    if !has_contracted {
        return Err(Error::ContractionError(
            "tropical einsum requires at least one contracted label".to_string(),
        ));
    }
    Ok(())
}

/// Tropical max-plus reduction over the given axes.
///
/// In max-plus algebra, reduction under semiring addition is ordinary maximum.
/// This wrapper exists so callers can express tropical intent while still
/// lowering to tenferro's core `ReduceMax` operation.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::tropical_reduce_sum;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 5.0, 2.0]);
/// let y = tropical_reduce_sum(&x, &[0]);
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[5.0]);
/// ```
#[must_use]
pub fn tropical_reduce_sum(a: &TracedTensor, axes: &[usize]) -> TracedTensor {
    a.reduce_max(axes)
}
