use super::super::super::*;
use super::common::dispatch_linalg_ad_runtime;
use num_complex::Complex;

use crate::DynTensorTyped;

/// Builder for AD eigen decomposition.
/// # Examples
///
/// ```text
/// // Construct `EigenAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EigenAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> EigenAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Executes AD eigen decomposition.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<TypedEigenResult<T>> {
        let operands = [self.tensor];
        ensure_dense_linalg_inputs("eigen", &operands)?;
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
            "eigen_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "eigen_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                "eigen_ad",
                |ctx| {
                    tenferro_linalg::eigen_frule::<T, _>(ctx, &input_primal, &dt)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                    "eigen_ad",
                    |ctx| {
                        tenferro_linalg::eigen::<T, _>(ctx, &input_primal).map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dvalues, dvectors) = if let Some(d) = tangent {
            (Some(d.values), Some(d.vectors))
        } else {
            (None, None)
        };

        let out_values =
            wrap_same_type_dense_ad_output("eigen_ad", &operands, primal.values, dvalues)?;
        let out_vectors =
            wrap_same_type_dense_ad_output("eigen_ad", &operands, primal.vectors, dvectors)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let Some((node, tape)) = out_values.reverse_handle() {
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                tape::register_rule::<T>(
                    &tape,
                    node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                            "eigen_ad_pullback_values",
                            |ctx| {
                                tenferro_linalg::eigen_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigenCotangent {
                                        values: Some(cotangent.payload().clone()),
                                        vectors: None,
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "eigen_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let Some((node, tape)) = out_vectors.reverse_handle() {
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                tape::register_rule::<T>(
                    &tape,
                    node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                            "eigen_ad_pullback_vectors",
                            |ctx| {
                                tenferro_linalg::eigen_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigenCotangent {
                                        values: None,
                                        vectors: Some(cotangent.payload().clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "eigen_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(TypedEigenResult {
            values: out_values,
            vectors: out_vectors,
        })
    }
}

/// Creates an AD eigen builder.
/// # Examples
///
/// ```ignore
/// let _ = eigen_ad(/* ... */);
/// ```
pub fn eigen_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> EigenAdBuilder<'a, T> {
    EigenAdBuilder { tensor }
}

/// Builder for AD eig.
/// # Examples
///
/// ```text
/// // Construct `EigAdBuilder` via its corresponding operation constructor.
/// ```
pub struct EigAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> EigAdBuilder<'a, T>
where
    T: ComplexLinalgRuntimeValue,
    Complex<T>: DynTensorTyped,
{
    /// Executes AD eig.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<TypedEigResult<T>> {
        let operands = [self.tensor];
        ensure_dense_linalg_inputs("eig", &operands)?;
        if has_reverse(&operands) {
            return Err(Error::UnsupportedAdOp { op: "eig_ad" });
        }
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Eig,
            "eig_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "eig_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::Eig,
                "eig_ad",
                |ctx| {
                    tenferro_linalg::eig_frule::<T, _>(ctx, &input_primal, &dt).map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::Eig,
                    "eig_ad",
                    |ctx| { tenferro_linalg::eig::<T, _>(ctx, &input_primal).map_err(Error::from) }
                )?,
                None,
            )
        };

        let (dvalues, dvectors) = if let Some(d) = tangent {
            (Some(d.values), Some(d.vectors))
        } else {
            (None, None)
        };

        let out_values = wrap_dense_ad_output("eig_ad", &operands, primal.values, dvalues)?;
        let out_vectors = wrap_dense_ad_output("eig_ad", &operands, primal.vectors, dvectors)?;

        Ok(TypedEigResult {
            values: out_values,
            vectors: out_vectors,
        })
    }
}

/// Creates an AD eig builder.
/// # Examples
///
/// ```ignore
/// let _ = eig_ad(/* ... */);
/// ```
pub fn eig_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> EigAdBuilder<'a, T> {
    EigAdBuilder { tensor }
}
