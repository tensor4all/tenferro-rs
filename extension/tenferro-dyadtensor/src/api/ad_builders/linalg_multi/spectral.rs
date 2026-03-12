use super::super::super::*;
use super::super::common::*;

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
    pub fn run(self) -> Result<AdEigenResult<T>> {
        let operands = [self.tensor];
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
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
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
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
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

        let out_values = wrap_dense_ad_output("eigen_ad", &operands, primal.values, dvalues, 1)?;
        let out_vectors = wrap_dense_ad_output("eigen_ad", &operands, primal.vectors, dvectors, 2)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let AdValue::Reverse { node, tape, .. } = out_values.as_value() {
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
                            tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                            "eigen_ad_pullback_values",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::eigen_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigenCotangent {
                                        values: Some(cotangent.clone()),
                                        vectors: None,
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("eigen_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let AdValue::Reverse { node, tape, .. } = out_vectors.as_value() {
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
                            tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
                            "eigen_ad_pullback_vectors",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::eigen_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigenCotangent {
                                        values: None,
                                        vectors: Some(cotangent.clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("eigen_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(AdEigenResult {
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
{
    /// Executes AD eig.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdEigResult<T>> {
        let operands = [self.tensor];
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
                |ctx, Backend| {
                    let _ = std::marker::PhantomData::<Backend>;
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
                    |ctx, Backend| {
                        let _ = std::marker::PhantomData::<Backend>;
                        tenferro_linalg::eig::<T, _>(ctx, &input_primal).map_err(Error::from)
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

        let out_values = wrap_dense_ad_output("eig_ad", &operands, primal.values, dvalues, 1)?;
        let out_vectors = wrap_dense_ad_output("eig_ad", &operands, primal.vectors, dvectors, 2)?;

        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        if let Some(spec) = input_spec {
            if let AdValue::Reverse { node, tape, .. } = out_values.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_bridge_rule::<Complex<T>, T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Eig,
                            "eig_ad_pullback_values",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::eig_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigCotangent {
                                        values: Some(cotangent.clone()),
                                        vectors: None,
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("eig_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }

            if let AdValue::Reverse { node, tape, .. } = out_vectors.as_value() {
                let output_node = *node;
                let tape_id = *tape;
                let spec = spec.clone();
                let a_primal = input_primal.clone();
                reverse_tape::register_bridge_rule::<Complex<T>, T>(
                    tape_id,
                    output_node,
                    Box::new(move |cotangent| {
                        let grad = dispatch_linalg_ad_runtime!(
                            T,
                            tenferro_linalg::backend::LinalgCapabilityOp::Eig,
                            "eig_ad_pullback_vectors",
                            |ctx, Backend| {
                                let _ = std::marker::PhantomData::<Backend>;
                                tenferro_linalg::eig_rrule::<T, _>(
                                    ctx,
                                    &a_primal,
                                    &tenferro_linalg::EigCotangent {
                                        values: None,
                                        vectors: Some(cotangent.clone()),
                                    },
                                )
                                .map_err(Error::from)
                            }
                        )?;
                        let grad = compress_pullback_like("eig_ad", grad, &spec.layout)?;
                        Ok(vec![(spec.node, grad)])
                    }),
                );
            }
        }

        Ok(AdEigResult {
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
