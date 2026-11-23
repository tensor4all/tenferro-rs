//! Recording a Rust closure as a closed, shape-generic sub-program.
//!
//! See `docs/design/subprograms.md`.

use std::fmt;
use std::sync::Arc;

use tenferro_tensor::DType;

use crate::checkpoint::RetainedValue;
use crate::error::{Error, ErrorPhase, Result};
use crate::program::Subprogram;
use crate::trace::TraceContext;
use crate::traced::TracedTensor;

const RECORD_OP: &str = "TraceContext::record";

/// A recorded closure: a closed [`Subprogram`] and the tensors it captured.
///
/// The sub-program's inputs are the declared parameters followed by one input
/// per capture, in [`captures`](Self::captures) order. An operation that
/// carries the sub-program receives the captures as ordinary operands.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{DType, TraceContext};
///
/// let recorded = TraceContext::record(&[(DType::F64, 1)], |trace, inputs| {
///     let weight = trace.from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?;
///     Ok(vec![inputs[0].mul(&weight)?])
/// })?;
///
/// let program = recorded.subprogram();
/// assert_eq!(program.parameter_count(), 1);
/// assert_eq!(program.capture_count(), 1);
/// assert_eq!(recorded.captures().len(), 1);
/// assert_eq!(recorded.captures().next().map(|capture| capture.shape()), Some(&[2][..]));
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
#[derive(Clone)]
pub struct RecordedProgram {
    subprogram: Subprogram,
    captures: Arc<[Arc<RetainedValue>]>,
}

impl RecordedProgram {
    /// Borrow the closed sub-program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, TraceContext};
    ///
    /// let recorded = TraceContext::record(&[(DType::F64, 2)], |_, inputs| {
    ///     Ok(vec![inputs[0].tanh()?])
    /// })?;
    /// assert_eq!(recorded.subprogram().program().outputs().len(), 1);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    pub fn subprogram(&self) -> &Subprogram {
        &self.subprogram
    }

    /// Iterate over the captured tensors in trailing-input order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, TraceContext};
    ///
    /// let recorded = TraceContext::record(&[(DType::F64, 0)], |trace, inputs| {
    ///     let offset = trace.from_vec_col_major(vec![], vec![1.5_f64])?;
    ///     Ok(vec![inputs[0].add(&offset)?])
    /// })?;
    /// let captured: Vec<_> = recorded.captures().map(|capture| capture.dtype()).collect();
    /// assert_eq!(captured, [DType::F64]);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    pub fn captures(&self) -> impl ExactSizeIterator<Item = &RetainedValue> + '_ {
        self.captures.iter().map(AsRef::as_ref)
    }
}

impl fmt::Debug for RecordedProgram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RecordedProgram")
            .field("subprogram", &self.subprogram)
            .field("captures", &self.captures.len())
            .finish()
    }
}

impl TraceContext {
    /// Trace `f` once on a fresh context and close the result as a sub-program.
    ///
    /// Each `(dtype, rank)` in `inputs` becomes one parameter with symbolic
    /// extents, so the recording is shape-generic: it is traced once and later
    /// specialized for each concrete shape it is run at. Every declared
    /// parameter stays an input, in order, even if `f` does not use it.
    ///
    /// A tensor that `f` binds from data (with
    /// [`from_vec_col_major`](Self::from_vec_col_major) or
    /// [`from_tensor_concrete_shape`](Self::from_tensor_concrete_shape)) is a
    /// *capture*: it is hoisted to a trailing input and returned by
    /// [`RecordedProgram::captures`], so the sub-program's identity does not
    /// depend on its values. A value built by a constant constructor such as
    /// `full`, `zeros`, `ones` or `scalar` is a `Constant` operation instead:
    /// its value stays in the program and is part of the identity, so two
    /// recordings that differ only in such a value are different
    /// sub-programs. Bind a weight from data when recordings should share one
    /// identity across values. Values of any other trace context cannot enter
    /// the recording.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, TraceContext};
    ///
    /// // y = a * x + b, elementwise over vectors of any (shared) length.
    /// let affine = TraceContext::record(&[(DType::F64, 1); 3], |_, p| {
    ///     Ok(vec![p[0].mul(&p[1])?.add(&p[2])?])
    /// })?;
    /// assert_eq!(affine.subprogram().parameter_count(), 3);
    ///
    /// // Inside the closure extents are symbolic.
    /// let error = TraceContext::record(&[(DType::F64, 1)], |_, p| {
    ///     p[0].concrete_shape()?;
    ///     Ok(vec![p[0].clone()])
    /// });
    /// assert!(error.is_err());
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns any error `f` returns. Returns [`Error::Validation`] with
    /// `InvalidArgument` when an output belongs to another trace context, or
    /// when an output depends on a placeholder input (one without data) that
    /// `f` added itself instead of declaring it in `inputs`. A placeholder no
    /// output depends on is pruned like any other dead value. Returns [`Error::RuntimeStateSource`] when the trace
    /// cannot accept the inputs or the program cannot be frozen.
    pub fn record<F>(inputs: &[(DType, usize)], f: F) -> Result<RecordedProgram>
    where
        F: FnOnce(&TraceContext, &[TracedTensor]) -> Result<Vec<TracedTensor>>,
    {
        let trace = TraceContext::new();
        let parameters = declare_parameters(&trace, inputs)?;
        let outputs = f(&trace, &parameters)?;
        close_recording(&trace, parameters.len(), &outputs)
    }
}

/// Add one symbolic-extent parameter per `(dtype, rank)`, in order.
fn declare_parameters(
    trace: &TraceContext,
    inputs: &[(DType, usize)],
) -> Result<Vec<TracedTensor>> {
    inputs
        .iter()
        .map(|&(dtype, rank)| trace.input_symbolic_shape(dtype, rank))
        .collect()
}

/// Freeze a finished recording and split its inputs into parameters and
/// captures.
///
/// Kept out of the generic [`TraceContext::record`] so it is compiled once,
/// not once per closure type.
fn close_recording(
    trace: &TraceContext,
    parameters: usize,
    outputs: &[TracedTensor],
) -> Result<RecordedProgram> {
    let mut values = Vec::with_capacity(outputs.len());
    for (index, output) in outputs.iter().enumerate() {
        if !output.context().same_context(trace) {
            return Err(Error::invalid_argument(
                RECORD_OP,
                ErrorPhase::GraphBuild,
                "outputs",
                format!(
                    "output {index} belongs to another trace context; a recorded \
                     program may return only values created inside its closure"
                ),
            ));
        }
        values.push(output.trace_value());
    }

    let frozen = trace
        .snapshot_keeping_leading_inputs(&values, parameters)
        .map_err(|source| Error::runtime_state_source(RECORD_OP, ErrorPhase::GraphBuild, source))?;
    // INVARIANT: the snapshot keeps the leading parameters first and in
    // order, so every later input is something the closure itself added.
    let mut captures = Vec::new();
    for input in frozen.program.inputs().iter().skip(parameters) {
        let Some(capture) = frozen.bindings.tensor_for_input(*input) else {
            return Err(Error::invalid_argument(
                RECORD_OP,
                ErrorPhase::GraphBuild,
                "f",
                "the closure added a placeholder input that is not a declared \
                 parameter; declare it in `inputs` instead",
            ));
        };
        captures.push(capture);
    }
    Ok(RecordedProgram {
        subprogram: Subprogram::new(frozen.program, parameters),
        captures: captures.into(),
    })
}

#[cfg(test)]
mod tests;
