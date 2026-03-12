use super::super::super::*;
use super::super::common::*;

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
    T: RealLinalgRuntimeValue,
{
    /// Sets optional SVD options.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.options(&options);
    /// ```
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
    pub fn run(self) -> Result<AdSvdResult<T>> {
        let operands = [self.tensor];
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
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
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
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
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

        let out_u = wrap_dense_ad_output("svd_ad", &operands, primal.u, du, 1)?;
        let out_s = wrap_dense_ad_output("svd_ad", &operands, primal.s, ds, 2)?;
        let out_vt = wrap_dense_ad_output("svd_ad", &operands, primal.vt, dvt, 3)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            let a_primal = input_primal.clone();
            let options = self.options.cloned();

            if let AdValue::Reverse { node, tape, .. } = out_u.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let options = options.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_u",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: Some(cotangent.clone()),
                                        s: None,
                                        vt: None,
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("svd_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let AdValue::Reverse { node, tape, .. } = out_s.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let options = options.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_s",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: None,
                                        s: Some(cotangent.clone()),
                                        vt: None,
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("svd_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let AdValue::Reverse { node, tape, .. } = out_vt.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let options = options.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
                            "svd_ad_pullback_vt",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::svd_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SvdCotangent {
                                        u: None,
                                        s: None,
                                        vt: Some(cotangent.clone()),
                                    },
                                    options.as_ref(),
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("svd_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(AdSvdResult {
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
    pub fn run(self) -> Result<AdQrResult<T>> {
        let operands = [self.tensor];
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
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
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
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::qr::<T, _>(ctx, &input_primal).map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dq, dr) = if let Some(d) = tangent {
            (Some(d.q), Some(d.r))
        } else {
            (None, None)
        };

        let out_q = wrap_dense_ad_output("qr_ad", &operands, primal.q, dq, 1)?;
        let out_r = wrap_dense_ad_output("qr_ad", &operands, primal.r, dr, 2)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let AdValue::Reverse { node, tape, .. } = out_q.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                            "qr_ad_pullback_q",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::qr_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::QrCotangent {
                                        q: Some(cotangent.clone()),
                                        r: None,
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("qr_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let AdValue::Reverse { node, tape, .. } = out_r.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_rule::<T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
                            "qr_ad_pullback_r",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::qr_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::QrCotangent {
                                        q: None,
                                        r: Some(cotangent.clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("qr_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(AdQrResult { q: out_q, r: out_r })
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
