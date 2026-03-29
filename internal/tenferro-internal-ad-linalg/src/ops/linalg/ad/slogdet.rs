use super::super::super::*;
use super::common::dispatch_linalg_ad_runtime;

/// Builder for AD slogdet.
/// # Examples
///
/// ```text
/// // Construct `SlogdetAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SlogdetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> SlogdetAdBuilder<'a, T>
where
    T: crate::runtime::dispatch::SlogdetLinalgDispatchValue,
{
    /// Executes AD slogdet.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<TypedSlogdetResult<T>> {
        let operands = [self.tensor];
        ensure_dense_linalg_inputs("slogdet", &operands)?;
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
            "slogdet_ad",
            |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, T>(ctx, self.tensor, needs_tangent)
            }
        )?;

        let (primal, tangent) = if needs_tangent {
            let dt = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: "slogdet_ad missing materialized tangent".to_string(),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                T,
                tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
                "slogdet_ad",
                |ctx| {
                    tenferro_linalg::slogdet_frule::<T, _>(ctx, &input_primal, &dt)
                        .map_err(Error::from)
                }
            )?;
            (p, Some(d))
        } else {
            (
                dispatch_linalg_ad_runtime!(
                    T,
                    tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
                    "slogdet_ad",
                    |ctx| {
                        tenferro_linalg::slogdet::<T, _>(ctx, &input_primal).map_err(Error::from)
                    }
                )?,
                None,
            )
        };

        let (dsign, dlogabsdet) = if let Some(d) = tangent {
            (Some(d.sign), Some(d.logabsdet))
        } else {
            (None, None)
        };

        let out_sign = wrap_same_type_dense_ad_output("slogdet_ad", &operands, primal.sign, dsign)?;
        let out_logabsdet =
            wrap_same_type_dense_ad_output("slogdet_ad", &operands, primal.logabsdet, dlogabsdet)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let Some((node, tape)) = out_sign.reverse_handle() {
                let spec = spec.clone();
                let zero = zero_like(spec.layout.payload())?;
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    vec![spec.node],
                    Box::new(move |_cotangent| {
                        Ok(vec![(
                            spec.node,
                            spec.layout.with_payload_like(zero.clone())?,
                        )])
                    }),
                );
            }

            if let Some((node, tape)) = out_logabsdet.reverse_handle() {
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    vec![spec.node],
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
                            "slogdet_ad_pullback_logabsdet",
                            |ctx| {
                                tenferro_linalg::slogdet_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::SlogdetCotangent {
                                        logabsdet: Some(cotangent.payload().clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = spec.layout.with_payload_like(compress_pullback_like(
                            "slogdet_ad",
                            grad,
                            &spec.layout,
                        )?)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(TypedSlogdetResult {
            sign: out_sign,
            logabsdet: out_logabsdet,
        })
    }
}

/// Creates an AD slogdet builder.
/// # Examples
///
/// ```ignore
/// let _ = slogdet_ad(/* ... */);
/// ```
pub fn slogdet_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> SlogdetAdBuilder<'a, T> {
    SlogdetAdBuilder { tensor }
}
