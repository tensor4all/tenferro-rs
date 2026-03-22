use super::super::super::*;
use super::common::dispatch_linalg_ad_runtime;
use crate::DynTensorTyped;

fn wrap_mixed_dense_linalg_output<TIn, TOut>(
    op_name: &'static str,
    inputs: &[&AdTensor<TIn>],
    primal: Tensor<TOut>,
    tangent: Option<Tensor<TOut>>,
) -> Result<AdTensor<TOut>>
where
    TIn: Scalar + DynTensorTyped,
    TOut: Scalar + DynTensorTyped,
{
    let tangent = tangent
        .map(|tangent| normalize_output_tangent_shape(tangent, primal.dims(), op_name))
        .transpose()?;

    if has_reverse(inputs) {
        let tape = derive_reverse_tape_handle(inputs)?.ok_or_else(|| Error::InvalidAdTensor {
            message: "reverse-mode output requested but no reverse tape found".to_string(),
        })?;
        return AdTensor::new_reverse_output(
            crate::structured::StructuredTensor::from_dense(primal),
            &tape,
            tangent.map(crate::structured::StructuredTensor::from_dense),
        );
    }

    if has_forward(inputs) {
        let tangent = tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: "forward-mode inputs must provide tangent output".to_string(),
        })?;
        return AdTensor::new_forward(
            crate::structured::StructuredTensor::from_dense(primal),
            crate::structured::StructuredTensor::from_dense(tangent),
        );
    }

    Ok(AdTensor::new_primal(
        crate::structured::StructuredTensor::from_dense(primal),
    ))
}

/// Builder for AD SVD.
/// # Examples
///
/// ```text
/// // Construct `SvdAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SvdAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    options: Option<&'a SvdOptions>,
}

impl<'a, T> SvdAdBuilder<'a, T>
where
    T: LinalgRuntimeValue,
    T::Real: DynTensorTyped + tenferro_tensor::KeepCountScalar,
{
    /// Sets optional SVD options.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.options(&options);
    /// ```
    #[allow(dead_code)]
    pub fn options(mut self, options: &'a SvdOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Executes AD SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<TypedSvdResult<T, T::Real>> {
        let operands = [self.tensor];
        ensure_dense_linalg_inputs("svd", &operands)?;
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
            "svd_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "svd_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                "svd_ad",
                |ctx| {
                    tenferro_linalg::svd_frule::<T, _>(ctx, &input_primal, &dt, self.options)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                    "svd_ad",
                    |ctx| {
                        tenferro_linalg::svd::<T, _>(ctx, &input_primal, self.options)
                            .map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (du, ds, dvt) = if let Some(d) = tangent {
            (Some(d.u), Some(d.s), Some(d.vt))
        } else {
            (None, None, None)
        };

        let out_u = wrap_same_type_dense_ad_output("svd_ad", &operands, primal.u, du)?;
        let out_s = wrap_mixed_dense_linalg_output("svd_ad", &operands, primal.s, ds)?;
        let out_vt = wrap_same_type_dense_ad_output("svd_ad", &operands, primal.vt, dvt)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            let a_primal = input_primal.clone();
            let options = self.options.cloned();

            if let Some((node, tape)) = out_u.reverse_handle() {
                let spec = spec.clone();
                let options = options.clone();
                tape::register_rule::<T>(
                    &tape,
                    node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_u",
                            |ctx| {
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: Some(cotangent.payload().clone()),
                                        s: None,
                                        vt: None,
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "svd_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let Some((node, tape)) = out_s.reverse_handle() {
                let spec = spec.clone();
                let options = options.clone();
                let a_primal = input_primal.clone();
                tape::register_mixed_rule::<T::Real, T>(
                    &tape,
                    node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_s",
                            |ctx| {
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: None,
                                        s: Some(cotangent.payload().clone()),
                                        vt: None,
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "svd_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let Some((node, tape)) = out_vt.reverse_handle() {
                let spec = spec.clone();
                let options = options.clone();
                let a_primal = input_primal.clone();
                tape::register_rule::<T>(
                    &tape,
                    node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_vt",
                            |ctx| {
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: None,
                                        s: None,
                                        vt: Some(cotangent.payload().clone()),
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "svd_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(TypedSvdResult {
            u: out_u,
            s: out_s,
            vt: out_vt,
        })
    }
}

/// Creates an AD SVD builder.
/// # Examples
///
/// ```ignore
/// let _ = svd_ad(/* ... */);
/// ```
pub fn svd_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> SvdAdBuilder<'a, T> {
    SvdAdBuilder {
        tensor,
        options: None,
    }
}

/// Builder for AD QR.
/// # Examples
///
/// ```text
/// // Construct `QrAdBuilder` via its corresponding operation constructor.
/// ```
pub struct QrAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> QrAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Executes AD QR.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<TypedQrResult<T>> {
        let operands = [self.tensor];
        ensure_dense_linalg_inputs("qr", &operands)?;
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
            "qr_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "qr_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                "qr_ad",
                |ctx| {
                    tenferro_linalg::qr_frule::<T, _>(ctx, &input_primal, &dt).map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                    "qr_ad",
                    |ctx| { tenferro_linalg::qr::<T, _>(ctx, &input_primal).map_err(Error::from) }
                )?,
                None,
            )
        };

        let (dq, dr) = if let Some(d) = tangent {
            (Some(d.q), Some(d.r))
        } else {
            (None, None)
        };

        let out_q = wrap_same_type_dense_ad_output("qr_ad", &operands, primal.q, dq)?;
        let out_r = wrap_same_type_dense_ad_output("qr_ad", &operands, primal.r, dr)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let Some((node, tape)) = out_q.reverse_handle() {
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                tape::register_rule::<T>(
                    &tape,
                    node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                            "qr_ad_pullback_q",
                            |ctx| {
                                tenferro_linalg::qr_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::QrCotangent {
                                        q: Some(cotangent.payload().clone()),
                                        r: None,
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "qr_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let Some((node, tape)) = out_r.reverse_handle() {
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                tape::register_rule::<T>(
                    &tape,
                    node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                            "qr_ad_pullback_r",
                            |ctx| {
                                tenferro_linalg::qr_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::QrCotangent {
                                        q: None,
                                        r: Some(cotangent.payload().clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "qr_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(TypedQrResult { q: out_q, r: out_r })
    }
}

/// Creates an AD QR builder.
/// # Examples
///
/// ```ignore
/// let _ = qr_ad(/* ... */);
/// ```
pub fn qr_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> QrAdBuilder<'a, T> {
    QrAdBuilder { tensor }
}
