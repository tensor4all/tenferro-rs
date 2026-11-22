//! Closed, shape-generic programs that an operation can carry.
//!
//! See `docs/design/subprograms.md` for the representation and identity
//! contract.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::Arc;

use super::{ProgramValueMetadata, SemanticFingerprint, SemanticProgram};

/// An immutable, closed, shape-generic semantic program.
///
/// A sub-program has no tensor bindings. Its inputs are the `parameter_count`
/// declared parameters, in declaration order, followed by
/// [`capture_count`](Self::capture_count) trailing inputs. Captures are
/// tensors the recorded closure referenced; they travel beside the program as
/// operands, so the identity never depends on tensor data.
///
/// Equality and hashing are structural: two recordings of the same closure are
/// equal even though they are different allocations, so caches keyed by a
/// sub-program hit across recordings.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{DType, TraceContext};
///
/// let square = || {
///     TraceContext::record(&[(DType::F64, 1)], |_, inputs| {
///         Ok(vec![inputs[0].mul(&inputs[0])?])
///     })
/// };
/// let first = square()?;
/// let second = square()?;
///
/// let program = first.subprogram();
/// assert_eq!(program.parameter_count(), 1);
/// assert_eq!(program.capture_count(), 0);
/// assert_eq!(program, second.subprogram());
/// assert_eq!(
///     program.semantic_fingerprint(),
///     second.subprogram().semantic_fingerprint(),
/// );
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
#[derive(Clone)]
pub struct Subprogram {
    program: Arc<SemanticProgram>,
    parameters: usize,
    fingerprint: SemanticFingerprint,
}

impl Subprogram {
    /// Wrap a frozen program whose first `parameters` inputs are parameters
    /// and whose remaining inputs are captures.
    pub(crate) fn new(program: Arc<SemanticProgram>, parameters: usize) -> Self {
        debug_assert!(parameters <= program.inputs().len());
        let fingerprint = program
            .semantic_fingerprint()
            .with_parameter_split(parameters);
        Self {
            program,
            parameters,
            fingerprint,
        }
    }

    /// Borrow the immutable semantic program.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, TraceContext};
    ///
    /// let recorded = TraceContext::record(&[(DType::F64, 2)], |_, inputs| {
    ///     Ok(vec![inputs[0].neg()?])
    /// })?;
    /// let program = recorded.subprogram().program();
    /// assert_eq!(program.inputs().len(), 1);
    /// assert_eq!(program.outputs().len(), 1);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    pub fn program(&self) -> &SemanticProgram {
        &self.program
    }

    /// Return the number of declared parameters, the leading program inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, TraceContext};
    ///
    /// let recorded = TraceContext::record(&[(DType::F64, 1), (DType::F64, 1)], |_, inputs| {
    ///     Ok(vec![inputs[0].add(&inputs[1])?])
    /// })?;
    /// assert_eq!(recorded.subprogram().parameter_count(), 2);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    pub fn parameter_count(&self) -> usize {
        self.parameters
    }

    /// Return the number of trailing capture inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, TraceContext};
    ///
    /// let recorded = TraceContext::record(&[(DType::F64, 1)], |trace, inputs| {
    ///     let weight = trace.from_vec_col_major(vec![1], vec![3.0_f64])?;
    ///     Ok(vec![inputs[0].mul(&weight)?])
    /// })?;
    /// assert_eq!(recorded.subprogram().capture_count(), 1);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    pub fn capture_count(&self) -> usize {
        // `new` keeps `parameters <= inputs`; saturate rather than wrap if a
        // future producer breaks that.
        self.program.inputs().len().saturating_sub(self.parameters)
    }

    /// Return the cached structural fingerprint of this sub-program.
    ///
    /// It covers the program structure and the parameter/capture split, the
    /// same facts equality compares, so unequal sub-programs differ here
    /// except by hash collision. It differs from
    /// [`program().semantic_fingerprint()`](SemanticProgram::semantic_fingerprint),
    /// which ignores the split.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{DType, TraceContext};
    ///
    /// let recorded = TraceContext::record(&[(DType::F64, 1)], |_, inputs| {
    ///     Ok(vec![inputs[0].exp()?])
    /// })?;
    /// assert_eq!(recorded.subprogram().semantic_fingerprint().as_bytes().len(), 16);
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    pub fn semantic_fingerprint(&self) -> SemanticFingerprint {
        self.fingerprint
    }
}

impl PartialEq for Subprogram {
    fn eq(&self, other: &Self) -> bool {
        self.parameters == other.parameters
            && (Arc::ptr_eq(&self.program, &other.program)
                || self.program.semantic_eq(&other.program))
    }
}

impl Eq for Subprogram {}

impl Hash for Subprogram {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
    }
}

impl fmt::Debug for Subprogram {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Subprogram")
            .field("parameters", &self.parameters)
            .field("captures", &self.capture_count())
            .field("operations", &self.program.operations.len())
            .field("semantic_fingerprint", &self.semantic_fingerprint())
            .finish()
    }
}

/// One use of a sub-program inside an operation.
///
/// `input_meta` states the callee's inputs in the *operation-local* symbols of
/// the carrying operation: an `InputDim` names one of that operation's own
/// inputs, as operation payloads do. Snapshots and imports renumber program
/// inputs but never an operation's own inputs, so a call is copied through them
/// unchanged.
#[derive(Clone)]
pub(crate) struct SubprogramCall {
    pub(crate) program: Subprogram,
    pub(crate) input_meta: Box<[ProgramValueMetadata]>,
}

// INVARIANT: no public builder entry attaches a call yet; the validated entry
// (`add_extension_with_subprograms`, #2691 step 2) is the next PR. Identity,
// snapshot and import already carry calls so that entry adds no identity work.
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "constructed by the #2691 step 2 builder entry")
)]
impl SubprogramCall {
    pub(crate) fn new(program: Subprogram, input_meta: Box<[ProgramValueMetadata]>) -> Self {
        Self {
            program,
            input_meta,
        }
    }
}

impl SubprogramCall {
    pub(crate) fn exact_eq(&self, other: &Self) -> bool {
        self.program == other.program && self.input_meta == other.input_meta
    }

    /// Bytes this call retains beyond its slot, including the callee.
    ///
    /// The callee is an `Arc` that several calls may share; it is counted at
    /// every call, so the total over-estimates shared callees and never
    /// under-counts what a program keeps alive.
    pub(crate) fn logical_retained_bytes(&self) -> Option<usize> {
        let metadata = self
            .input_meta
            .len()
            .checked_mul(size_of::<ProgramValueMetadata>())?;
        let metadata = self.input_meta.iter().try_fold(metadata, |sum, meta| {
            sum.checked_add(meta.logical_retained_bytes()?)
        })?;
        metadata.checked_add(self.program.program.logical_retained_bytes()?)
    }
}

#[cfg(test)]
mod tests;
