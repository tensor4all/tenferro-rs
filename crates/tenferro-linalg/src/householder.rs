use std::fmt;
use std::ops::Range;

use tenferro_tensor::{BackendSession, Tensor};

use crate::backend::CompactQrResult;
use crate::QrOptions;

/// Opaque compact Householder QR state.
///
/// The packed reflector tensors remain private so callers cannot accidentally
/// treat provider state as ordinary tensor values.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_linalg::{QrOptions, TensorLinalgExt};
/// use tenferro_tensor::{BackendSessionHost, Tensor};
///
/// let a = Tensor::from_vec_col_major(
///     vec![3, 2],
///     vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0],
/// )?;
/// let mut host = CpuBackend::new();
/// let r = host.with_backend_session(|session| {
///     let qr = a.householder_qr(session)?;
///     qr.r(QrOptions::default(), session)
/// })?;
/// assert_eq!(r.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone)]
pub struct HouseholderQr<T> {
    pub(crate) packed: T,
    pub(crate) coeff: T,
}

impl<T> fmt::Debug for HouseholderQr<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HouseholderQr")
            .finish_non_exhaustive()
    }
}

impl HouseholderQr<Tensor> {
    pub(crate) fn from_backend(state: CompactQrResult) -> Self {
        Self {
            packed: state.packed,
            coeff: state.coeff,
        }
    }

    /// Construct compact state for the product of compatible factors `Q * R`.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_tensor::Error::Validation` for incompatible rank,
    /// shape, dtype, placement, or a non-trapezoidal R factor;
    /// `tenferro_tensor::Error::Unsupported` when the provider lacks compact
    /// QR; or `tenferro_tensor::Error::BackendSource` for provider failures.
    pub fn from_factors(
        q: &Tensor,
        r: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Self> {
        crate::tensor_ext::with_linalg_backend(session, "householder_qr_from_factors", |backend| {
            backend
                .householder_qr_from_factors(q, r)
                .map(Self::from_backend)
        })
    }

    /// Append a column block without refactorizing existing columns.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_tensor::Error::Validation` for incompatible shape,
    /// dtype, placement, or malformed state; `tenferro_tensor::Error::Unsupported`
    /// when append is unavailable; or `tenferro_tensor::Error::BackendSource`
    /// for reflector or factorization failures.
    pub fn append_columns(
        &self,
        block: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Self> {
        crate::tensor_ext::with_linalg_backend(session, "householder_qr_append", |backend| {
            backend
                .householder_qr_append(&self.packed, &self.coeff, block)
                .map(Self::from_backend)
        })
    }

    /// Extract the thin upper-trapezoidal factor.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_tensor::Error::Validation` for malformed state,
    /// `tenferro_tensor::Error::Unsupported` for an unavailable provider path,
    /// or `tenferro_tensor::Error::BackendSource` for extraction failures.
    pub fn r(
        &self,
        options: QrOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        crate::tensor_ext::with_linalg_backend(session, "householder_qr_r", |backend| {
            backend.householder_qr_r(&self.packed, &self.coeff, options)
        })
    }

    /// Materialize a contiguous range of thin-Q columns.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_tensor::Error::Validation` when the range is outside
    /// the thin-Q width or state metadata is malformed,
    /// `tenferro_tensor::Error::Unsupported` for an unavailable provider path,
    /// or `tenferro_tensor::Error::BackendSource` for execution failures.
    pub fn q_columns(
        &self,
        columns: Range<usize>,
        options: QrOptions,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor> {
        crate::tensor_ext::with_linalg_backend(session, "householder_qr_q_columns", |backend| {
            backend.householder_qr_q_columns(&self.packed, &self.coeff, columns, options)
        })
    }
}

#[cfg(feature = "autodiff")]
impl HouseholderQr<tenferro_ad::EagerTensor> {
    pub(crate) fn from_eager_outputs(
        packed: tenferro_ad::EagerTensor,
        coeff: tenferro_ad::EagerTensor,
    ) -> Self {
        Self { packed, coeff }
    }

    /// Construct compact state from compatible eager factors.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_ad::Error::Validation` for known invalid metadata,
    /// `tenferro_ad::Error::Extension` for unsupported or provider failures,
    /// or `tenferro_ad::Error::RuntimeState` when eager execution is unavailable.
    pub fn from_factors(
        q: &tenferro_ad::EagerTensor,
        r: &tenferro_ad::EagerTensor,
    ) -> tenferro_ad::Result<Self> {
        eager_state(crate::eager_ext::apply_linalg_eager(
            crate::extension::LinalgOp::HouseholderQrFromFactors,
            &[q, r],
        )?)
    }

    /// Append an eager column block functionally.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_ad::Error::Validation` for known invalid metadata,
    /// `tenferro_ad::Error::Extension` for unsupported or provider failures,
    /// or `tenferro_ad::Error::RuntimeState` when eager execution is unavailable.
    pub fn append_columns(&self, block: &tenferro_ad::EagerTensor) -> tenferro_ad::Result<Self> {
        eager_state(crate::eager_ext::apply_linalg_eager(
            crate::extension::LinalgOp::HouseholderQrAppend,
            &[&self.packed, &self.coeff, block],
        )?)
    }

    /// Extract eager R.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_ad::Error::Validation` for known invalid metadata,
    /// `tenferro_ad::Error::Extension` for unsupported or provider failures,
    /// or `tenferro_ad::Error::RuntimeState` when eager execution is unavailable.
    pub fn r(&self, options: QrOptions) -> tenferro_ad::Result<tenferro_ad::EagerTensor> {
        eager_one(
            crate::eager_ext::apply_linalg_eager(
                crate::extension::LinalgOp::HouseholderQrR {
                    gauge: options.gauge,
                },
                &[&self.packed, &self.coeff],
            )?,
            "householder_qr_r",
        )
    }

    /// Materialize eager thin-Q columns.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_ad::Error::Validation` for known invalid metadata,
    /// `tenferro_ad::Error::Extension` for unsupported or provider failures,
    /// or `tenferro_ad::Error::RuntimeState` when eager execution is unavailable.
    pub fn q_columns(
        &self,
        columns: Range<usize>,
        options: QrOptions,
    ) -> tenferro_ad::Result<tenferro_ad::EagerTensor> {
        eager_one(
            crate::eager_ext::apply_linalg_eager(
                crate::extension::LinalgOp::HouseholderQrQColumns {
                    start: columns.start,
                    end: columns.end,
                    gauge: options.gauge,
                },
                &[&self.packed, &self.coeff],
            )?,
            "householder_qr_q_columns",
        )
    }
}

#[cfg(feature = "autodiff")]
fn eager_state(
    outputs: Vec<tenferro_ad::EagerTensor>,
) -> tenferro_ad::Result<HouseholderQr<tenferro_ad::EagerTensor>> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(packed), Some(coeff), None) => Ok(HouseholderQr::from_eager_outputs(packed, coeff)),
        _ => Err(tenferro_ad::Error::Internal(
            "compact Householder QR returned an unexpected output count".into(),
        )),
    }
}

#[cfg(feature = "autodiff")]
fn eager_one(
    outputs: Vec<tenferro_ad::EagerTensor>,
    op: &'static str,
) -> tenferro_ad::Result<tenferro_ad::EagerTensor> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(tenferro_ad::Error::Internal(format!(
            "{op} returned an unexpected output count"
        ))),
    }
}

impl HouseholderQr<tenferro_runtime::TracedTensor> {
    pub(crate) fn from_traced_outputs(
        packed: tenferro_runtime::TracedTensor,
        coeff: tenferro_runtime::TracedTensor,
    ) -> Self {
        Self { packed, coeff }
    }

    /// Construct compact state from compatible traced factors.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_runtime::Error::Validation` for known invalid metadata
    /// or `tenferro_runtime::Error::Extension` for unsupported operation state.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints and backend provider failures may be reported
    /// during compile or execution.
    pub fn from_factors(
        q: &tenferro_runtime::TracedTensor,
        r: &tenferro_runtime::TracedTensor,
    ) -> tenferro_runtime::Result<Self> {
        crate::validation::ensure_float_or_complex("householder_qr_from_factors", q.dtype())?;
        crate::validation::ensure_float_or_complex("householder_qr_from_factors", r.dtype())?;
        traced_state(
            crate::extension::LinalgOp::HouseholderQrFromFactors,
            &[q, r],
        )
    }

    /// Append a traced column block functionally.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_runtime::Error::Validation` for known invalid metadata
    /// or `tenferro_runtime::Error::Extension` for unsupported operation state.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints and backend provider failures may be reported
    /// during compile or execution.
    pub fn append_columns(
        &self,
        block: &tenferro_runtime::TracedTensor,
    ) -> tenferro_runtime::Result<Self> {
        crate::validation::ensure_float_or_complex("householder_qr_append", block.dtype())?;
        traced_state(
            crate::extension::LinalgOp::HouseholderQrAppend,
            &[&self.packed, &self.coeff, block],
        )
    }

    /// Extract traced R.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_runtime::Error::Validation` for known invalid metadata
    /// or `tenferro_runtime::Error::Extension` for unsupported operation state.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints and backend provider failures may be reported
    /// during compile or execution.
    pub fn r(
        &self,
        options: QrOptions,
    ) -> tenferro_runtime::Result<tenferro_runtime::TracedTensor> {
        traced_one(
            crate::extension::LinalgOp::HouseholderQrR {
                gauge: options.gauge,
            },
            &[&self.packed, &self.coeff],
            "householder_qr_r",
        )
    }

    /// Materialize traced thin-Q columns.
    ///
    /// # Errors
    ///
    /// Returns `tenferro_runtime::Error::Validation` for known invalid metadata
    /// or `tenferro_runtime::Error::Extension` for unsupported operation state.
    ///
    /// # Deferred errors
    ///
    /// Symbolic shape constraints and backend provider failures may be reported
    /// during compile or execution.
    pub fn q_columns(
        &self,
        columns: Range<usize>,
        options: QrOptions,
    ) -> tenferro_runtime::Result<tenferro_runtime::TracedTensor> {
        traced_one(
            crate::extension::LinalgOp::HouseholderQrQColumns {
                start: columns.start,
                end: columns.end,
                gauge: options.gauge,
            },
            &[&self.packed, &self.coeff],
            "householder_qr_q_columns",
        )
    }
}

fn traced_outputs(
    op: crate::extension::LinalgOp,
    inputs: &[&tenferro_runtime::TracedTensor],
) -> tenferro_runtime::Result<Vec<tenferro_runtime::TracedTensor>> {
    tenferro_runtime::extension::apply(
        std::sync::Arc::new(crate::extension::LinalgExtensionOp::new(op)),
        inputs,
    )
}

fn traced_state(
    op: crate::extension::LinalgOp,
    inputs: &[&tenferro_runtime::TracedTensor],
) -> tenferro_runtime::Result<HouseholderQr<tenferro_runtime::TracedTensor>> {
    let mut outputs = traced_outputs(op, inputs)?.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(packed), Some(coeff), None) => Ok(HouseholderQr::from_traced_outputs(packed, coeff)),
        _ => Err(tenferro_runtime::Error::Internal(
            "compact Householder QR returned an unexpected output count".into(),
        )),
    }
}

fn traced_one(
    op: crate::extension::LinalgOp,
    inputs: &[&tenferro_runtime::TracedTensor],
    name: &'static str,
) -> tenferro_runtime::Result<tenferro_runtime::TracedTensor> {
    let mut outputs = traced_outputs(op, inputs)?.into_iter();
    match (outputs.next(), outputs.next()) {
        (Some(output), None) => Ok(output),
        _ => Err(tenferro_runtime::Error::Internal(format!(
            "{name} returned an unexpected output count"
        ))),
    }
}
