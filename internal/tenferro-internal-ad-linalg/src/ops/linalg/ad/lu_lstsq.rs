use super::super::super::*;
use super::common::dispatch_linalg_ad_runtime;

/// Builder for AD LU.
/// # Examples
///
/// ```text
/// // Construct `LuAdBuilder` via its corresponding operation constructor.
/// ```
pub struct LuAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    pivot: LuPivot,
}

impl<'a, T> LuAdBuilder<'a, T>
where
    T: crate::runtime::dispatch::RealLuLinalgDispatchValue + DynAdTensorTyped,
{
    /// Sets LU pivot policy.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivot(pivot);
    /// ```
    #[allow(dead_code)]
    pub fn pivot(mut self, pivot: LuPivot) -> Self {
        self.pivot = pivot;
        self
    }

    /// Executes AD LU.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<DynLuResult> {
        let operands = [self.tensor];
        ensure_dense_linalg_inputs("lu", &operands)?;
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
            "lu_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "lu_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
                "lu_ad",
                |ctx| {
                    tenferro_linalg::lu_frule::<T, _>(ctx, &input_primal, &dt, self.pivot)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
                    "lu_ad",
                    |ctx| {
                        tenferro_linalg::lu::<T, _>(ctx, &input_primal, self.pivot)
                            .map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dl, du) = if let Some(d) = tangent {
            (Some(d.l), Some(d.u))
        } else {
            (None, None)
        };

        let out_l = wrap_same_type_dense_ad_output("lu_ad", &operands, primal.l, dl)?;
        let out_u = wrap_same_type_dense_ad_output("lu_ad", &operands, primal.u, du)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let Some((node, tape)) = out_l.reverse_handle() {
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                let pivot = self.pivot;
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    vec![spec.node],
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
                            "lu_ad_pullback_l",
                            |ctx| {
                                tenferro_linalg::lu_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::LuCotangent {
                                        l: Some(cotangent.payload().clone()),
                                        u: None,
                                    },
                                    pivot,
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "lu_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let Some((node, tape)) = out_u.reverse_handle() {
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                let pivot = self.pivot;
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    vec![spec.node],
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
                            "lu_ad_pullback_u",
                            |ctx| {
                                tenferro_linalg::lu_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::LuCotangent {
                                        l: None,
                                        u: Some(cotangent.payload().clone()),
                                    },
                                    pivot,
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "lu_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(DynLuResult {
            p: AdTensor::try_from(tenferro_internal_ad_core::AdTensorSnapshot::Primal(
                primal.p.into(),
            ))?
            .into(),
            l: out_l.into(),
            u: out_u.into(),
        })
    }
}

/// Creates an AD LU builder.
/// # Examples
///
/// ```ignore
/// let _ = lu_ad(/* ... */);
/// ```
pub fn lu_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> LuAdBuilder<'a, T> {
    LuAdBuilder {
        tensor,
        pivot: LuPivot::Partial,
    }
}

/// Builder for AD least squares.
/// # Examples
///
/// ```text
/// // Construct `LstsqAdBuilder` via its corresponding operation constructor.
/// ```
pub struct LstsqAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
}

impl<'a, T> LstsqAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    /// Executes AD least squares.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<DynLstsqResult> {
        let operands = [self.a, self.b];
        ensure_dense_linalg_inputs("lstsq", &operands)?;
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let ((a_primal, a_tangent), (b_primal, b_tangent)) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
            "lstsq_ad",
            |ctx, Backend| {
                Ok((
                    dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.a, needs_tangent)?,
                    dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.b, needs_tangent)?,
                ))
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let da = a_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "lstsq_ad missing materialized lhs tangent".to_string(),
            })?;
            let db = b_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "lstsq_ad missing materialized rhs tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
                "lstsq_ad",
                |ctx| {
                    tenferro_linalg::lstsq_frule::<T, _>(ctx, &a_primal, &b_primal, &da, &db)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
                    "lstsq_ad",
                    |ctx| {
                        tenferro_linalg::lstsq::<T, _>(ctx, &a_primal, &b_primal)
                            .map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dx, dresidual) = if let Some(d) = tangent {
            (Some(d.x), Some(d.residual))
        } else {
            (None, None)
        };

        let out_x = wrap_same_type_dense_ad_output("lstsq_ad", &operands, primal.x, dx)?;
        let out_residual =
            wrap_same_type_dense_ad_output("lstsq_ad", &operands, primal.residual, dresidual)?;

        let reverse_specs = collect_reverse_input_specs(&operands);
        if has_reverse(&operands) {
            if let Some((node, tape)) = out_x.reverse_handle() {
                let reverse_specs = reverse_specs.clone();
                let a_primal = a_primal.clone();
                let b_primal = b_primal.clone();
                let input_node_ids: Vec<_> = reverse_specs
                    .iter()
                    .filter_map(|s| s.as_ref().map(|s| s.node))
                    .collect();
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    input_node_ids,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
                            "lstsq_ad_pullback_x",
                            |ctx| {
                                tenferro_linalg::lstsq_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &b_primal,
                                    cotangent.payload(),
                                )
                                .map_err(Error::from)
                            }
                        )?;

                        let mut input_grads = Vec::new();
                        if let Some(spec) = &reverse_specs[0] {
                            let grad_a = spec.layout.with_payload_like(compress_pullback_like(
                                "lstsq_ad",
                                grad.a,
                                &spec.layout,
                            )?)?;
                            input_grads.push((spec.node, grad_a));
                        }
                        if let Some(spec) = &reverse_specs[1] {
                            let grad_b = spec.layout.with_payload_like(compress_pullback_like(
                                "lstsq_ad",
                                grad.b,
                                &spec.layout,
                            )?)?;
                            input_grads.push((spec.node, grad_b));
                        }
                        Ok(input_grads)
                    }),
                );
            }

            if let Some((node, tape)) = out_residual.reverse_handle() {
                let reverse_specs = reverse_specs.clone();
                let zero_a = zero_like(self.a.structured_primal().payload())?;
                let zero_b = zero_like(self.b.structured_primal().payload())?;
                let input_node_ids: Vec<_> = reverse_specs
                    .iter()
                    .filter_map(|s| s.as_ref().map(|s| s.node))
                    .collect();
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    input_node_ids,
                    Box::new(move |_cotangent| {
                        let mut input_grads = Vec::new();
                        if let Some(spec) = &reverse_specs[0] {
                            input_grads
                                .push((spec.node, spec.layout.with_payload_like(zero_a.clone())?));
                        }
                        if let Some(spec) = &reverse_specs[1] {
                            input_grads
                                .push((spec.node, spec.layout.with_payload_like(zero_b.clone())?));
                        }
                        Ok(input_grads)
                    }),
                );
            }
        }

        Ok(DynLstsqResult {
            x: out_x.into(),
            residual: out_residual.into(),
        })
    }
}

/// Creates an AD lstsq builder.
/// # Examples
///
/// ```ignore
/// let _ = lstsq_ad(/* ... */);
/// ```
pub fn lstsq_ad<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> LstsqAdBuilder<'a, T> {
    LstsqAdBuilder { a, b }
}
