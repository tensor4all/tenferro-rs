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
//! let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 5.0, 2.0]).unwrap();
//! let y = tropical_reduce_sum(&x, &[0]).unwrap();
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let mut executor = GraphExecutor::new(CpuBackend::new());
//! let out = executor.run(&program).unwrap();
//! assert_eq!(out.as_slice::<f64>().unwrap(), &[5.0]);
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tenferro_einsum::Subscripts;
use tenferro_runtime::{extension, DType, Error, ErrorPhase, Result, TracedTensor};
use tenferro_tensor::{ShapeMismatch, ShapeVec, ValidationError};

use crate::extension::TropicalEinsumOp;
use crate::TropicalKind;

fn try_require_rank2(tensor: &TracedTensor, op: &'static str) -> Result<()> {
    if tensor.rank != 2 {
        return Err(Error::validation(
            op,
            ErrorPhase::GraphBuild,
            ValidationError::RankMismatch {
                expected: 2,
                actual: tensor.rank,
            },
        ));
    }
    Ok(())
}

fn try_require_contracting_axes_compatible(
    a: &TracedTensor,
    b: &TracedTensor,
    op: &'static str,
) -> Result<()> {
    if let (Some(a_shape), Some(b_shape)) = (a.try_concrete_shape(), b.try_concrete_shape()) {
        if a_shape[1] != b_shape[0] {
            return Err(Error::validation(
                op,
                ErrorPhase::GraphBuild,
                ShapeMismatch::ContractedDimensions {
                    lhs_axis: 1,
                    lhs_size: a_shape[1],
                    rhs_axis: 0,
                    rhs_size: b_shape[0],
                }
                .into(),
            ));
        }
    }
    Ok(())
}

fn checked_tropical_dot_general_impl(
    a: &TracedTensor,
    b: &TracedTensor,
    reduce: impl FnOnce(&TracedTensor) -> Result<TracedTensor>,
    op: &'static str,
) -> Result<TracedTensor> {
    try_require_rank2(a, op)?;
    try_require_rank2(b, op)?;
    try_require_contracting_axes_compatible(a, b, op)?;

    let m = a.axis_sym_dim(0)?;
    let k = a.axis_sym_dim(1)?;
    let n = b.axis_sym_dim(1)?;
    let target_shape = [m, k, n];

    let a_b = a.broadcast_in_dim_sym(&target_shape, &[0, 1], &[b])?;
    let b_b = b.broadcast_in_dim_sym(&target_shape, &[1, 2], &[a])?;
    let sum = a_b.add(&b_b)?;
    reduce(&sum)
}

/// Max-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = max_k(a[i, k] + b[k, j])` by composing
/// `BroadcastInDim`, `Add`, and `ReduceMax` graph operations.
///
/// # Errors
///
/// Returns `Error::Validation` if either input is not rank 2 or if known
/// contracting dimensions are incompatible, `Error::Extension` with an
/// unsupported-dtype source for non-floating inputs, and `Error::RuntimeState`
/// if the tropical runtime extension is unavailable.
///
/// # Deferred errors
///
/// Symbolic contracting dimensions are checked during compilation or
/// execution and can produce `ShapeConstraintViolation` or
/// `ShapeConstraintEvaluation`.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::tropical_dot_general;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
/// let c = tropical_dot_general(&a, &b).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&c).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[23.0, 24.0, 43.0, 44.0]);
/// ```
pub fn tropical_dot_general(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    checked_tropical_dot_general_impl(a, b, |sum| sum.reduce_max(&[1]), "tropical_dot_general")
}

/// Min-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = min_k(a[i, k] + b[k, j])` by composing
/// `BroadcastInDim`, `Add`, and `ReduceMin` graph operations.
///
/// # Errors
///
/// Returns `Error::Validation` if either input is not rank 2 or if known
/// contracting dimensions are incompatible, `Error::Extension` with an
/// unsupported-dtype source for non-floating inputs, and `Error::RuntimeState`
/// if the tropical runtime extension is unavailable.
///
/// # Deferred errors
///
/// Symbolic contracting dimensions are checked during compilation or
/// execution and can produce `ShapeConstraintViolation` or
/// `ShapeConstraintEvaluation`.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::min_plus_dot_general;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
/// let c = min_plus_dot_general(&a, &b).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&c).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[6.0, 7.0, 8.0, 9.0]);
/// ```
pub fn min_plus_dot_general(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    checked_tropical_dot_general_impl(a, b, |sum| sum.reduce_min(&[1]), "min_plus_dot_general")
}

/// Fused binary tropical einsum over traced tensors.
///
/// The operation is carried as a runtime extension and uses the tropical
/// value-plus-argmax executor during forward execution.
///
/// # Errors
///
/// Returns `Error::Validation` when `subscripts` describe anything other than
/// two inputs with compatible contracting axes, `Error::Extension` with an
/// unsupported-dtype source for non-floating inputs, or `Error::Internal` if
/// the extension returns no output.
///
/// # Deferred errors
///
/// Symbolic dimension equalities are checked during compilation or execution
/// and can produce `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::{traced::tropical_einsum, TropicalKind};
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]).unwrap();
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
    let subscripts = Subscripts::parse(notation)
        .map_err(|err| crate::error::from_einsum_error("tropical_einsum", err))?;
    tropical_einsum_subscripts(kind, inputs, &subscripts)
}

/// Fused binary tropical einsum over traced tensors with parsed subscripts.
///
/// This variant is useful when the caller already owns
/// [`tenferro_einsum::Subscripts`] and wants to avoid a string roundtrip.
///
/// # Errors
///
/// Returns `Error::Validation` when `subscripts` describe anything other than
/// two inputs with compatible contracting axes, `Error::Extension` with an
/// unsupported-dtype source for non-floating inputs, or `Error::Internal` if
/// the extension returns no output.
///
/// # Deferred errors
///
/// Symbolic dimension equalities are checked during compilation or execution
/// and can produce `ShapeConstraintViolation` or `ShapeConstraintEvaluation`.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::Subscripts;
/// use tenferro_ext_tropical::{traced::tropical_einsum_subscripts, TropicalKind};
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 1.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 2.0]).unwrap();
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
    )?;
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
/// # Errors
///
/// Returns `Error::Validation` if either input is not rank 2 or if known
/// contracting dimensions are incompatible, `Error::Extension` with an
/// unsupported-dtype source for non-floating inputs, and `Error::RuntimeState`
/// if the tropical runtime extension is unavailable.
///
/// # Deferred errors
///
/// Symbolic contracting dimensions are checked during compilation or
/// execution and can produce `ShapeConstraintViolation` or
/// `ShapeConstraintEvaluation`.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::tropical_dot_general_fused;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![10.0_f64, 20.0, 30.0, 40.0]).unwrap();
/// let out = tropical_dot_general_fused(&a, &b).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&out).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_ext_tropical::register_runtime).unwrap();
/// let value = executor.run(&program).unwrap();
/// assert_eq!(value.as_slice::<f64>().unwrap(), &[23.0, 24.0, 43.0, 44.0]);
/// ```
pub fn tropical_dot_general_fused(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    try_fused_dot_general_impl(TropicalKind::MaxPlus, a, b, "tropical_dot_general_fused")
}

/// Fused min-plus matrix multiplication on rank-2 traced tensors.
///
/// Computes `out[i, j] = min_k(a[i, k] + b[k, j])` through the tropical
/// extension executor rather than a broadcast/reduce composition.
///
/// # Errors
///
/// Returns `Error::Validation` if either input is not rank 2 or if known
/// contracting dimensions are incompatible, `Error::Extension` with an
/// unsupported-dtype source for non-floating inputs, and `Error::RuntimeState`
/// if the tropical runtime extension is unavailable.
///
/// # Deferred errors
///
/// Symbolic contracting dimensions are checked during compilation or
/// execution and can produce `ShapeConstraintViolation` or
/// `ShapeConstraintEvaluation`.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ext_tropical::traced::min_plus_dot_general_fused;
/// use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
///
/// let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
/// let b = TracedTensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();
/// let out = min_plus_dot_general_fused(&a, &b).unwrap();
///
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&out).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// executor.register_extension(tenferro_ext_tropical::register_runtime).unwrap();
/// let value = executor.run(&program).unwrap();
/// assert_eq!(value.as_slice::<f64>().unwrap(), &[6.0, 7.0, 8.0, 9.0]);
/// ```
pub fn min_plus_dot_general_fused(a: &TracedTensor, b: &TracedTensor) -> Result<TracedTensor> {
    try_fused_dot_general_impl(TropicalKind::MinPlus, a, b, "min_plus_dot_general_fused")
}

fn try_fused_dot_general_impl(
    kind: TropicalKind,
    a: &TracedTensor,
    b: &TracedTensor,
    op: &'static str,
) -> Result<TracedTensor> {
    try_require_rank2(a, op)?;
    try_require_rank2(b, op)?;
    try_require_contracting_axes_compatible(a, b, op)?;
    let subscripts = Subscripts::new(
        &[&[b'i' as u32, b'j' as u32], &[b'j' as u32, b'k' as u32]],
        &[b'i' as u32, b'k' as u32],
    );
    tropical_einsum_subscripts(kind, &[a, b], &subscripts)
}

fn validate_tropical_einsum_inputs(
    inputs: &[&TracedTensor],
    subscripts: &Subscripts,
) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::invalid_argument(
            "tropical_einsum_subscripts",
            ErrorPhase::GraphBuild,
            "inputs",
            format!("tropical einsum supports exactly two inputs, got {}", inputs.len()),
        ));
    }
    if subscripts.inputs.len() != 2 {
        return Err(Error::invalid_argument(
            "tropical_einsum_subscripts",
            ErrorPhase::GraphBuild,
            "subscripts",
            format!(
                "tropical einsum subscripts describe {} inputs, expected 2",
                subscripts.inputs.len()
            ),
        ));
    }
    for tensor in inputs.iter() {
        if !matches!(tensor.dtype, DType::F32 | DType::F64) {
            return Err(Error::TensorRuntime(crate::error::unsupported_dtype(
                "tropical_einsum_subscripts",
                tensor.dtype,
            )));
        }
    }
    if inputs[0].dtype != inputs[1].dtype {
        return Err(Error::dtype_mismatch(
            "tropical_einsum_subscripts",
            ErrorPhase::GraphBuild,
            inputs[0].dtype,
            inputs[1].dtype,
        ));
    }

    let mut labels_seen = HashSet::new();
    let mut concrete_label_dims = HashMap::new();
    for (labels, tensor) in subscripts.inputs.iter().zip(inputs) {
        if labels.len() != tensor.rank {
            return Err(Error::validation(
                "tropical_einsum_subscripts",
                ErrorPhase::GraphBuild,
                ValidationError::RankMismatch {
                    expected: labels.len(),
                    actual: tensor.rank,
                },
            ));
        }
        if labels
            .iter()
            .enumerate()
            .any(|(idx, label)| labels[..idx].contains(label))
        {
            return Err(Error::validation(
                "tropical_einsum_subscripts",
                ErrorPhase::GraphBuild,
                ValidationError::DuplicateAxis {
                    axis: 0,
                    role: "input label",
                },
            ));
        }
        labels_seen.extend(labels.iter().copied());
        if let Some(shape) = tensor.try_concrete_shape() {
            for (&label, &extent) in labels.iter().zip(shape.iter()) {
                if let Some(previous) = concrete_label_dims.insert(label, extent) {
                    if previous != extent {
                        return Err(Error::validation(
                            "tropical_einsum_subscripts",
                            ErrorPhase::GraphBuild,
                            ShapeMismatch::ExpectedActual {
                                expected: ShapeVec::from_vec(vec![previous]),
                                actual: ShapeVec::from_vec(vec![extent]),
                            }
                            .into(),
                        ));
                    }
                }
            }
        }
    }
    if subscripts
        .output
        .iter()
        .enumerate()
        .any(|(idx, label)| subscripts.output[..idx].contains(label))
    {
        return Err(Error::validation(
            "tropical_einsum_subscripts",
            ErrorPhase::GraphBuild,
            ValidationError::DuplicateAxis {
                axis: 0,
                role: "output label",
            },
        ));
    }
    for &label in &subscripts.output {
        if !labels_seen.contains(&label) {
            return Err(Error::invalid_argument(
                "tropical_einsum_subscripts",
                ErrorPhase::GraphBuild,
                "output",
                format!("output label {label} is not present in any input"),
            ));
        }
    }
    let has_contracted = subscripts.inputs[0]
        .iter()
        .any(|label| subscripts.inputs[1].contains(label) && !subscripts.output.contains(label));
    if !has_contracted {
        return Err(Error::invalid_argument(
            "tropical_einsum_subscripts",
            ErrorPhase::GraphBuild,
            "subscripts",
            "tropical einsum requires at least one contracted label",
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
/// let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 5.0, 2.0]).unwrap();
/// let y = tropical_reduce_sum(&x, &[0]).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&y).unwrap();
/// let mut executor = GraphExecutor::new(CpuBackend::new());
/// let out = executor.run(&program).unwrap();
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[5.0]);
/// ```
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::Validation`] with an
/// `AxisOutOfBounds` or `DuplicateAxis` payload when `axes` is invalid, or
/// [`tenferro_runtime::Error::Internal`] if output metadata registration
/// fails.
pub fn tropical_reduce_sum(a: &TracedTensor, axes: &[usize]) -> Result<TracedTensor> {
    a.reduce_max(axes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compositional_dot_general_try_api_rejects_rank_mismatch() {
        let lhs = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
        let rhs = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

        let err = tropical_dot_general(&lhs, &rhs).unwrap_err();
        assert!(matches!(
            err,
            Error::Validation {
                op: "tropical_dot_general",
                phase: ErrorPhase::GraphBuild,
                source: ValidationError::RankMismatch {
                    expected: 2,
                    actual: 1,
                },
            }
        ));
    }

    #[test]
    fn compositional_dot_general_try_api_rejects_contracting_dim_mismatch() {
        let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
        let rhs = TracedTensor::from_vec_col_major(vec![4, 2], vec![1.0_f64; 8]).unwrap();

        let err = min_plus_dot_general(&lhs, &rhs).unwrap_err();
        assert!(matches!(
            err,
            Error::Validation {
                op: "min_plus_dot_general",
                phase: ErrorPhase::GraphBuild,
                source: ValidationError::ShapeMismatch(payload),
            } if matches!(
                payload.as_ref(),
                ShapeMismatch::ContractedDimensions {
                    lhs_axis: 1,
                    lhs_size: 3,
                    rhs_axis: 0,
                    rhs_size: 4,
                }
            )
        ));
    }
}
