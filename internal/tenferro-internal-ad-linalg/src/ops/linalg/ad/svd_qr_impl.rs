use super::super::super::super::*;
use super::super::common::dispatch_linalg_ad_runtime;
use std::marker::PhantomData;
use tenferro_internal_frontend_core::DynTensorTyped;
use tidu::{AdResult, AutodiffError, Op, Schema, SlotSchema, Value};

fn ad_invalid_argument(err: impl std::fmt::Display) -> AutodiffError {
    AutodiffError::InvalidArgument(err.to_string())
}

struct EdgeQrSaved<T: Scalar> {
    input_layout: crate::structured::StructuredTensor<T>,
    input_primal: tenferro_tensor::Tensor<T>,
}

struct EdgeSvdSaved<T: Scalar> {
    input_layout: crate::structured::StructuredTensor<T>,
    input_primal: tenferro_tensor::Tensor<T>,
    options: Option<SvdOptions>,
}

#[derive(Clone, Copy)]
struct EdgeQrOp<T>(PhantomData<T>);

impl<T> Op<crate::structured::StructuredTensor<T>> for EdgeQrOp<T>
where
    T: RealLinalgRuntimeValue,
{
    type SavedBackward = EdgeQrSaved<T>;
    type SavedJvp = EdgeQrSaved<T>;

    fn primal(
        &self,
        inputs: &[&crate::structured::StructuredTensor<T>],
    ) -> AdResult<Vec<crate::structured::StructuredTensor<T>>> {
        let input_dense = inputs[0].to_dense().map_err(ad_invalid_argument)?;
        let primal = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
            "edge_qr_primal",
            |ctx| { tenferro_linalg::qr::<T, _>(ctx, &input_dense).map_err(Error::from) }
        )
        .map_err(ad_invalid_argument)?;
        Ok(vec![
            crate::structured::StructuredTensor::from(primal.q),
            crate::structured::StructuredTensor::from(primal.r),
        ])
    }

    fn input_schema(
        &self,
        _inputs: &[&crate::structured::StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(
        &self,
        _inputs: &[&crate::structured::StructuredTensor<T>],
        _outputs: &[crate::structured::StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
            ],
        })
    }

    fn save_for_backward(
        &self,
        inputs: &[&crate::structured::StructuredTensor<T>],
        _outputs: &[crate::structured::StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(EdgeQrSaved {
            input_layout: inputs[0].clone(),
            input_primal: inputs[0].to_dense().map_err(ad_invalid_argument)?,
        })
    }

    fn save_for_jvp(
        &self,
        inputs: &[&crate::structured::StructuredTensor<T>],
        _outputs: &[crate::structured::StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(EdgeQrSaved {
            input_layout: inputs[0].clone(),
            input_primal: inputs[0].to_dense().map_err(ad_invalid_argument)?,
        })
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<crate::structured::StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<crate::structured::StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let grad_q = grad_outputs[0]
            .as_ref()
            .map(|value| value.to_dense().map_err(ad_invalid_argument))
            .transpose()?;
        let grad_r = grad_outputs[1]
            .as_ref()
            .map(|value| value.to_dense().map_err(ad_invalid_argument))
            .transpose()?;
        if grad_q.is_none() && grad_r.is_none() {
            return Ok(vec![None]);
        }
        let grad_dense = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
            "edge_qr_pullback",
            |ctx| {
                tenferro_linalg::qr_rrule::<T, _>(
                    ctx,
                    &saved.input_primal,
                    &tenferro_linalg::QrCotangent {
                        q: grad_q.clone(),
                        r: grad_r.clone(),
                    },
                )
                .map_err(Error::from)
            }
        )
        .map_err(ad_invalid_argument)?;
        let grad =
            compress_structured_pullback_like("edge_qr_pullback", grad_dense, &saved.input_layout)
                .map_err(ad_invalid_argument)?;
        Ok(vec![Some(grad)])
    }

    fn jvp(
        &self,
        saved: &Self::SavedJvp,
        tangents: &[Option<crate::structured::StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<crate::structured::StructuredTensor<T>>>> {
        let Some(tangent) = tangents[0].as_ref() else {
            return Ok(vec![None, None]);
        };
        let tangent_dense = tangent.to_dense().map_err(ad_invalid_argument)?;
        let (_primal, tangent_out) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Qr,
            "edge_qr_jvp",
            |ctx| {
                tenferro_linalg::qr_frule::<T, _>(ctx, &saved.input_primal, &tangent_dense)
                    .map_err(Error::from)
            }
        )
        .map_err(ad_invalid_argument)?;
        Ok(vec![
            Some(crate::structured::StructuredTensor::from(tangent_out.q)),
            Some(crate::structured::StructuredTensor::from(tangent_out.r)),
        ])
    }
}

#[derive(Clone)]
struct EdgeSvdOp<T> {
    options: Option<SvdOptions>,
    _marker: PhantomData<T>,
}

impl<T> Op<crate::structured::StructuredTensor<T>> for EdgeSvdOp<T>
where
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    type SavedBackward = EdgeSvdSaved<T>;
    type SavedJvp = EdgeSvdSaved<T>;

    fn primal(
        &self,
        inputs: &[&crate::structured::StructuredTensor<T>],
    ) -> AdResult<Vec<crate::structured::StructuredTensor<T>>> {
        let input_dense = inputs[0].to_dense().map_err(ad_invalid_argument)?;
        let primal = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
            "edge_svd_primal",
            |ctx| {
                tenferro_linalg::svd::<T, _>(ctx, &input_dense, self.options.as_ref())
                    .map_err(Error::from)
            }
        )
        .map_err(ad_invalid_argument)?;
        Ok(vec![
            crate::structured::StructuredTensor::from(primal.u),
            crate::structured::StructuredTensor::from(primal.s),
            crate::structured::StructuredTensor::from(primal.vt),
        ])
    }

    fn input_schema(
        &self,
        _inputs: &[&crate::structured::StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![SlotSchema {
                differentiable: true,
                auxiliary: false,
            }],
        })
    }

    fn output_schema(
        &self,
        _inputs: &[&crate::structured::StructuredTensor<T>],
        _outputs: &[crate::structured::StructuredTensor<T>],
    ) -> AdResult<Schema> {
        Ok(Schema {
            slots: vec![
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
                SlotSchema {
                    differentiable: true,
                    auxiliary: false,
                },
            ],
        })
    }

    fn save_for_backward(
        &self,
        inputs: &[&crate::structured::StructuredTensor<T>],
        _outputs: &[crate::structured::StructuredTensor<T>],
    ) -> AdResult<Self::SavedBackward> {
        Ok(EdgeSvdSaved {
            input_layout: inputs[0].clone(),
            input_primal: inputs[0].to_dense().map_err(ad_invalid_argument)?,
            options: self.options.clone(),
        })
    }

    fn save_for_jvp(
        &self,
        inputs: &[&crate::structured::StructuredTensor<T>],
        _outputs: &[crate::structured::StructuredTensor<T>],
    ) -> AdResult<Self::SavedJvp> {
        Ok(EdgeSvdSaved {
            input_layout: inputs[0].clone(),
            input_primal: inputs[0].to_dense().map_err(ad_invalid_argument)?,
            options: self.options.clone(),
        })
    }

    fn backward(
        &self,
        saved: &Self::SavedBackward,
        grad_outputs: &[Option<crate::structured::StructuredTensor<T>>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<crate::structured::StructuredTensor<T>>>> {
        if !input_grad_mask[0] {
            return Ok(vec![None]);
        }
        let grad_u = grad_outputs[0]
            .as_ref()
            .map(|value| value.to_dense().map_err(ad_invalid_argument))
            .transpose()?;
        let grad_s = grad_outputs[1]
            .as_ref()
            .map(|value| value.to_dense().map_err(ad_invalid_argument))
            .transpose()?;
        let grad_vt = grad_outputs[2]
            .as_ref()
            .map(|value| value.to_dense().map_err(ad_invalid_argument))
            .transpose()?;
        if grad_u.is_none() && grad_s.is_none() && grad_vt.is_none() {
            return Ok(vec![None]);
        }
        let grad_dense = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
            "edge_svd_pullback",
            |ctx| {
                tenferro_linalg::svd_rrule::<T, _>(
                    ctx,
                    &saved.input_primal,
                    &tenferro_linalg::SvdCotangent {
                        u: grad_u.clone(),
                        s: grad_s.clone(),
                        vt: grad_vt.clone(),
                    },
                    saved.options.as_ref(),
                )
                .map_err(Error::from)
            }
        )
        .map_err(ad_invalid_argument)?;
        let grad =
            compress_structured_pullback_like("edge_svd_pullback", grad_dense, &saved.input_layout)
                .map_err(ad_invalid_argument)?;
        Ok(vec![Some(grad)])
    }

    fn jvp(
        &self,
        saved: &Self::SavedJvp,
        tangents: &[Option<crate::structured::StructuredTensor<T>>],
    ) -> AdResult<Vec<Option<crate::structured::StructuredTensor<T>>>> {
        let Some(tangent) = tangents[0].as_ref() else {
            return Ok(vec![None, None, None]);
        };
        let tangent_dense = tangent.to_dense().map_err(ad_invalid_argument)?;
        let (_primal, tangent_out) = dispatch_linalg_ad_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
            "edge_svd_jvp",
            |ctx| {
                tenferro_linalg::svd_frule::<T, _>(
                    ctx,
                    &saved.input_primal,
                    &tangent_dense,
                    saved.options.as_ref(),
                )
                .map_err(Error::from)
            }
        )
        .map_err(ad_invalid_argument)?;
        Ok(vec![
            Some(crate::structured::StructuredTensor::from(tangent_out.u)),
            Some(crate::structured::StructuredTensor::from(tangent_out.s)),
            Some(crate::structured::StructuredTensor::from(tangent_out.vt)),
        ])
    }
}

fn can_use_edge_qr_reverse<T>(tensor: &AdTensor<T>) -> bool
where
    T: RealLinalgRuntimeValue,
{
    tensor.is_dense()
        && tensor.structured_tangent().is_none()
        && tensor.reverse_edge_value().is_some()
}

fn can_use_edge_svd_reverse<T>(tensor: &AdTensor<T>) -> bool
where
    T: RealLinalgRuntimeValue,
{
    tensor.is_dense()
        && tensor.structured_tangent().is_none()
        && tensor.reverse_edge_value().is_some()
}

fn edge_output_to_ad<T>(
    output: Value<crate::structured::StructuredTensor<T>>,
) -> Result<AdTensor<T>>
where
    T: Scalar + DynTensorTyped,
{
    AdTensor::from_reverse_edge_value(output)
}

pub(crate) fn edge_qr<T>(tensor: &AdTensor<T>) -> Result<DynQrResult>
where
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    let input = tensor
        .reverse_edge_value()
        .ok_or(Error::UnsupportedAdOp { op: "edge_qr" })?;
    let mut outputs = EdgeQrOp::<T>(PhantomData)
        .apply(&[input.as_ref()])
        .map_err(Error::from)?;
    let r = edge_output_to_ad(
        outputs
            .pop()
            .ok_or_else(|| ad_invalid_argument("qr output r missing"))?,
    )?;
    let q = edge_output_to_ad(
        outputs
            .pop()
            .ok_or_else(|| ad_invalid_argument("qr output q missing"))?,
    )?;
    Ok(DynQrResult {
        q: q.into(),
        r: r.into(),
    })
}

pub(crate) fn edge_svd_real<T>(
    tensor: &AdTensor<T>,
    options: Option<&SvdOptions>,
) -> Result<TypedSvdResult>
where
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    let input = tensor
        .reverse_edge_value()
        .ok_or(Error::UnsupportedAdOp { op: "edge_svd" })?;
    let op = EdgeSvdOp::<T> {
        options: options.cloned(),
        _marker: PhantomData,
    };
    let mut outputs = op.apply(&[input.as_ref()]).map_err(Error::from)?;
    let vt = edge_output_to_ad(
        outputs
            .pop()
            .ok_or_else(|| ad_invalid_argument("svd output vt missing"))?,
    )?;
    let s = edge_output_to_ad(
        outputs
            .pop()
            .ok_or_else(|| ad_invalid_argument("svd output s missing"))?,
    )?;
    let u = edge_output_to_ad(
        outputs
            .pop()
            .ok_or_else(|| ad_invalid_argument("svd output u missing"))?,
    )?;
    Ok(TypedSvdResult {
        u: T::into_dyn_ad(u),
        s: T::into_dyn_ad(s),
        vt: T::into_dyn_ad(vt),
    })
}

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
        return AdTensor::from_reverse_output(
            crate::structured::StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(
                primal,
            )),
            &tape,
            tangent.map(|value| {
                crate::structured::StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(
                    value,
                ))
            }),
        );
    }

    if has_forward(inputs) {
        let tangent = tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: "forward-mode inputs must provide tangent output".to_string(),
        })?;
        return AdTensor::try_from(tenferro_internal_ad_core::AdTensorSnapshot::Forward {
            primal: crate::structured::StructuredTensor(
                tenferro_tensor::StructuredTensor::from_dense(primal),
            ),
            tangent: crate::structured::StructuredTensor(
                tenferro_tensor::StructuredTensor::from_dense(tangent),
            ),
        });
    }

    Ok(AdTensor::try_from(
        tenferro_internal_ad_core::AdTensorSnapshot::Primal(crate::structured::StructuredTensor(
            tenferro_tensor::StructuredTensor::from_dense(primal),
        )),
    )?)
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
    T: DynAdTensorTyped,
    T::Real: DynTensorTyped + DynAdTensorTyped + tenferro_tensor::KeepCountScalar,
{
    /// Executes AD SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<TypedSvdResult> {
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
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    vec![spec.node],
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
                    vec![spec.node],
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
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    vec![spec.node],
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
            u: T::into_dyn_ad(out_u),
            s: T::Real::into_dyn_ad(out_s),
            vt: T::into_dyn_ad(out_vt),
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

pub(crate) fn can_use_edge_svd_real_reverse<T>(tensor: &AdTensor<T>) -> bool
where
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    can_use_edge_svd_reverse(tensor)
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
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    /// Executes AD QR.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<DynQrResult> {
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
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    vec![spec.node],
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
                tape::register_closure_rule::<T>(
                    &tape,
                    node,
                    vec![spec.node],
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

        Ok(DynQrResult {
            q: out_q.into(),
            r: out_r.into(),
        })
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

pub(crate) fn can_use_edge_qr_real_reverse<T>(tensor: &AdTensor<T>) -> bool
where
    T: RealLinalgRuntimeValue,
{
    can_use_edge_qr_reverse(tensor)
}
