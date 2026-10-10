use std::sync::Arc;

use computegraph::GraphOperation;
use tenferro_ops::broadcast::{
    broadcast_error_to_validation, broadcast_in_dim_extent_error, broadcast_input_plan,
    broadcast_shape, broadcast_shapes,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{SliceConfig, Tensor, TensorValue};

use crate::eager::{
    eager_capture_active, eager_grad_recording_enabled, eager_op_profile_start,
    maybe_print_eager_op_profile, profile_eager_op_section, record_eager_op_profile,
    record_eager_outputs_in_session, record_eager_value_outputs_in_session, EagerSession,
    EagerTensor,
};
use crate::eager_exec::{
    exec_standard_op_on_tensor_reads_with_session, exec_standard_op_on_tensors_with_session,
};
use crate::error::{Error, Result};

/// Whether an untracked fast path may skip AD recording for `tensors`.
///
/// Mirrors the eligibility test in [`EagerTensor::nary_op_in_session`]: a fused
/// kernel may run directly only when no gradient or semantic-trace recording
/// would be lost. Tracked operands, or active semantic capture, keep the
/// recorded primitive path.
pub(crate) fn untracked_fast_path_allowed(tensors: &[&EagerTensor]) -> bool {
    !eager_grad_recording_enabled()
        || (!eager_capture_active() && !tensors.iter().any(|tensor| tensor.requires_grad))
}

pub(crate) fn broadcast_binary_in_session(
    op: &'static str,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    session: &mut EagerSession<'_>,
) -> Result<(EagerTensor, EagerTensor)> {
    ensure_same_context(lhs, rhs)?;
    let shape =
        broadcast_shape(lhs.shape(), rhs.shape()).map_err(|err| broadcast_error(op, err))?;
    Ok((
        broadcast_to_in_session(op, lhs, &shape, session)?,
        broadcast_to_in_session(op, rhs, &shape, session)?,
    ))
}

pub(crate) fn broadcast_ternary_in_session(
    op: &'static str,
    first: &EagerTensor,
    second: &EagerTensor,
    third: &EagerTensor,
    session: &mut EagerSession<'_>,
) -> Result<(EagerTensor, EagerTensor, EagerTensor)> {
    ensure_same_context(first, second)?;
    ensure_same_context(first, third)?;
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(|err| broadcast_error(op, err))?;
    Ok((
        broadcast_to_in_session(op, first, &shape, session)?,
        broadcast_to_in_session(op, second, &shape, session)?,
        broadcast_to_in_session(op, third, &shape, session)?,
    ))
}

fn broadcast_to_in_session(
    op: &'static str,
    input: &EagerTensor,
    shape: &[usize],
    session: &mut EagerSession<'_>,
) -> Result<EagerTensor> {
    if input.shape() == shape {
        return Ok(input.clone());
    }
    let plan =
        broadcast_input_plan(input.shape(), shape).map_err(|err| broadcast_error(op, err))?;
    let source = if plan.source_shape == input.shape() {
        input.clone()
    } else {
        session.reshape(input, &plan.source_shape)?
    };
    session.broadcast_in_dim(&source, shape, &plan.dims)
}

fn broadcast_error(op: &'static str, err: tenferro_ops::broadcast::BroadcastError) -> Error {
    tenferro_tensor::Error::validation(op, broadcast_error_to_validation(err)).into()
}

fn ensure_same_context(lhs: &EagerTensor, rhs: &EagerTensor) -> Result<()> {
    if !lhs.same_context(rhs) {
        return Err(Error::ContextMismatch {
            lhs: lhs.ctx_id(),
            rhs: rhs.ctx_id(),
        });
    }
    Ok(())
}

impl EagerTensor {
    pub(crate) fn reduce_sum_in_session(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn tenferro_tensor::BackendSession,
    ) -> Result<Self> {
        let axes = axes.map_or_else(|| (0..self.shape().len()).collect(), <[usize]>::to_vec);
        validate_eager_axes("EagerTensor::reduce_sum", self.shape().len(), &axes)?;
        Self::nary_op_in_session(&[self], StdTensorOp::ReduceSum { axes }, session)
    }

    pub(crate) fn transpose_in_session(
        &self,
        perm: &[usize],
        session: &mut dyn tenferro_tensor::BackendSession,
    ) -> Result<Self> {
        let base = self.duplicate_value_in_session(session)?;
        let value = TensorValue::from_tensor(base)
            .transpose_view(perm)
            .map_err(Error::TensorRuntime)?;
        Self::nary_value_op_in_session(
            &[self],
            StdTensorOp::Transpose {
                perm: perm.to_vec(),
            },
            value,
            session,
        )
    }

    pub(crate) fn reshape_in_session(
        &self,
        shape: &[usize],
        session: &mut dyn tenferro_tensor::BackendSession,
    ) -> Result<Self> {
        let op = StdTensorOp::Reshape {
            to_shape: DimExpr::from_concrete(shape),
        };
        let base = self.duplicate_value_in_session(session)?;
        if let Ok(value) = TensorValue::from_tensor(base).reshape_view(shape) {
            return Self::nary_value_op_in_session(&[self], op, value, session);
        }
        Self::nary_op_in_session(&[self], op, session)
    }

    pub(crate) fn slice_in_session(
        &self,
        config: SliceConfig,
        session: &mut dyn tenferro_tensor::BackendSession,
    ) -> Result<Self> {
        let base = self.duplicate_value_in_session(session)?;
        let value = TensorValue::from_tensor(base)
            .slice_view(&config)
            .map_err(Error::TensorRuntime)?;
        Self::nary_value_op_in_session(&[self], StdTensorOp::Slice(config), value, session)
    }

    pub(crate) fn broadcast_in_dim_in_session(
        &self,
        shape: &[usize],
        dims: &[usize],
        session: &mut dyn tenferro_tensor::BackendSession,
    ) -> Result<Self> {
        if let Some(error) = broadcast_in_dim_extent_error(self.shape(), shape, dims) {
            return Err(broadcast_error("EagerTensor::broadcast_in_dim", error));
        }
        let op = StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(shape),
            dims: dims.to_vec(),
        };
        // INVARIANT: output descriptors cannot borrow the input's move-only
        // allocation group, so this explicit duplicate owns the view's root.
        let base = self.duplicate_value_in_session(session)?;
        let value = TensorValue::from_tensor(base)
            .broadcast_in_dim_view(shape, dims)
            .map_err(Error::TensorRuntime)?;
        Self::nary_value_op_in_session(&[self], op, value, session)
    }

    pub(crate) fn nary_value_op_in_session(
        tensors: &[&Self],
        op: StdTensorOp,
        value: TensorValue,
        session: &mut dyn tenferro_tensor::BackendSession,
    ) -> Result<Self> {
        let Some(first) = tensors.first() else {
            return Err(empty_nary_input_error(&op));
        };

        let ctx = Arc::clone(&first.ctx);
        for tensor in tensors.iter().skip(1) {
            if !first.same_context(tensor) {
                return Err(Error::ContextMismatch {
                    lhs: first.ctx_id(),
                    rhs: tensor.ctx_id(),
                });
            }
        }

        if !eager_grad_recording_enabled()
            || (!eager_capture_active() && !tensors.iter().any(|tensor| tensor.requires_grad))
        {
            return Self::new_untracked_value_result(ctx, value);
        }

        let output_ref = &value;
        let mut recorded =
            record_eager_value_outputs_in_session(&op, &[output_ref], tensors, session)?;
        let trace = recorded.traces.pop().ok_or_else(|| {
            Error::Internal(format!("expected one eager trace for {:?}, got 0", op))
        })?;
        let semantic_trace = recorded.semantic_traces.pop().flatten();

        let result = Self::new_result_value(
            ctx,
            trace.key,
            value,
            trace.requires_grad,
            trace.trace,
            semantic_trace,
        )?;
        crate::eager::finish_residuals(&op, tensors, &[&result])?;
        Ok(result)
    }

    pub(crate) fn nary_op_in_session(
        tensors: &[&Self],
        op: StdTensorOp,
        session: &mut dyn tenferro_tensor::BackendSession,
    ) -> Result<Self> {
        let total_started = eager_op_profile_start();
        let Some(first) = tensors.first() else {
            return Err(empty_nary_input_error(&op));
        };
        let expected = op.input_count();
        if tensors.len() != expected {
            return Err(wrong_nary_input_count_error(&op, expected, tensors.len()));
        }

        let ctx = Arc::clone(&first.ctx);
        profile_eager_op_section("nary_op.context_check", || -> Result<()> {
            for tensor in tensors.iter().skip(1) {
                if !first.same_context(tensor) {
                    return Err(Error::ContextMismatch {
                        lhs: first.ctx_id(),
                        rhs: tensor.ctx_id(),
                    });
                }
            }
            Ok(())
        })?;

        let any_requires_grad = profile_eager_op_section("nary_op.requires_grad_scan", || {
            eager_grad_recording_enabled()
                && (eager_capture_active() || tensors.iter().any(|tensor| tensor.requires_grad))
        });
        if !any_requires_grad {
            let input_reads = profile_eager_op_section("nary_op.collect_input_reads", || {
                tensors
                    .iter()
                    .map(|tensor| tensor.tensor_read())
                    .collect::<Vec<_>>()
            });
            let output = profile_eager_op_section("nary_op.exec_single_output_read", || {
                single_session_output(
                    &op,
                    exec_standard_op_on_tensor_reads_with_session(&op, &input_reads, session)?,
                )
            })?;
            let result = profile_eager_op_section("nary_op.new_untracked_result", || {
                Self::new_untracked_result(ctx, output)
            });
            if let Some(total_started) = total_started {
                record_eager_op_profile("nary_op.total", total_started.elapsed());
                maybe_print_eager_op_profile();
            }
            return result;
        }

        let input_arcs = profile_eager_op_section("nary_op.materialize_inputs", || {
            tensors
                .iter()
                .map(|tensor| tensor.duplicate_value_in_session(session).map(Arc::new))
                .collect::<Result<Vec<_>>>()
        })?;
        let inputs: Vec<&Tensor> = profile_eager_op_section("nary_op.collect_inputs", || {
            input_arcs.iter().map(|tensor| tensor.as_ref()).collect()
        });
        let output = profile_eager_op_section("nary_op.exec_single_output", || {
            single_session_output(
                &op,
                exec_standard_op_on_tensors_with_session(&op, &inputs, session)?,
            )
        })?;

        let outputs = vec![&output];
        let mut recorded = profile_eager_op_section("nary_op.record_outputs", || {
            record_eager_outputs_in_session(&op, &outputs, tensors, session)
        })?;
        let trace = recorded.traces.pop().ok_or_else(|| {
            Error::Internal(format!("expected one eager trace for {:?}, got 0", op))
        })?;
        let semantic_trace = recorded.semantic_traces.pop().flatten();

        let result = profile_eager_op_section("nary_op.new_tracked_result", || {
            Self::new_result_with_semantic_trace(
                ctx,
                trace.key,
                output,
                trace.requires_grad,
                trace.trace,
                semantic_trace,
            )
        })?;
        crate::eager::finish_residuals(&op, tensors, &[&result])?;
        if let Some(total_started) = total_started {
            record_eager_op_profile("nary_op.total", total_started.elapsed());
            maybe_print_eager_op_profile();
        }
        Ok(result)
    }
}

fn single_session_output(op: &StdTensorOp, outputs: Vec<Tensor>) -> Result<Tensor> {
    let [output] = outputs.try_into().map_err(|outputs: Vec<Tensor>| {
        Error::Internal(format!(
            "expected one eager output for {:?}, got {}",
            op,
            outputs.len()
        ))
    })?;
    Ok(output)
}

pub(crate) fn validate_eager_axes(op: &'static str, rank: usize, axes: &[usize]) -> Result<()> {
    tenferro_tensor::validate::validate_unique_axes(op, "axis", rank, axes)
        .map_err(Error::TensorRuntime)
}

fn empty_nary_input_error(op: &StdTensorOp) -> Error {
    Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        eager_validation_op_name(op),
        "inputs",
        "operation requires at least one input tensor",
    ))
}

fn wrong_nary_input_count_error(op: &StdTensorOp, expected: usize, actual: usize) -> Error {
    Error::TensorRuntime(tenferro_tensor::Error::invalid_argument(
        eager_validation_op_name(op),
        "inputs",
        format!("operation expects {expected} inputs, got {actual}"),
    ))
}

fn eager_validation_op_name(op: &StdTensorOp) -> &'static str {
    match op {
        StdTensorOp::Concatenate { .. } => "concatenate",
        _ => "eager_nary_op",
    }
}
