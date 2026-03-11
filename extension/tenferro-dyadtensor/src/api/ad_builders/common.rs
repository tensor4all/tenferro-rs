macro_rules! dispatch_linalg_ad_runtime {
    ($ty:ty, $capability:expr, $op:literal, |$ctx:ident, $backend:ident| $body:expr) => {{
        with_linalg_runtime::<$ty, _>(
            $op,
            $capability,
            |$ctx| {
                type $backend = CpuBackend;
                $body
            },
            |$ctx| {
                type $backend = tenferro_prims::CudaBackend;
                $body
            },
            |$ctx| {
                type $backend = tenferro_prims::RocmBackend;
                $body
            },
        )
    }};
}

macro_rules! run_unary_tensor_ad {
    (
        ty = $ty:ty,
        capability = $capability:expr,
        op = $op_name:literal,
        pullback = $pullback_op_name:literal,
        input = $input:expr,
        primal = |$primal_ctx:ident, $primal_backend:ident, $primal_tensor:ident| $primal_body:expr,
        frule = |$frule_ctx:ident, $frule_backend:ident, $frule_tensor:ident, $frule_tangent:ident| $frule_body:expr,
        rrule = |$rrule_ctx:ident, $rrule_backend:ident, $rrule_tensor:ident, $rrule_cotangent:ident| $rrule_body:expr $(,)?
    ) => {{
        let operands = [$input];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) =
            dispatch_linalg_ad_runtime!($ty, $capability, $op_name, |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, $ty>(ctx, $input, needs_tangent)
            })?;

        let (primal, tangent) = if needs_tangent {
            let input_tangent = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: format!("{} missing materialized tangent", $op_name),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$frule_ctx, $frule_backend| {
                    let $frule_tensor = &input_primal;
                    let $frule_tangent = &input_tangent;
                    $frule_body
                }
            )?;
            (p, Some(d))
        } else {
            let primal = dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$primal_ctx, $primal_backend| {
                    let $primal_tensor = &input_primal;
                    $primal_body
                }
            )?;
            (primal, None)
        };

        let out = wrap_dense_ad_output($op_name, &operands, primal, tangent, 0)?;

        if let AdValue::Reverse { node, tape, .. } = out.as_value() {
            let input_spec = collect_reverse_input_specs(&operands)
                .into_iter()
                .next()
                .flatten();
            let output_node = *node;
            let tape_id = *tape;

            reverse_tape::register_rule::<$ty>(
                tape_id,
                output_node,
                Box::new(move |cotangent| {
                    let grad = dispatch_linalg_ad_runtime!(
                        $ty,
                        $capability,
                        $pullback_op_name,
                        |$rrule_ctx, $rrule_backend| {
                            let $rrule_tensor = &input_primal;
                            let $rrule_cotangent = cotangent;
                            $rrule_body
                        }
                    )?;

                    let Some(spec) = &input_spec else {
                        return Ok(Vec::new());
                    };
                    let grad = compress_pullback_like($op_name, grad, &spec.layout)?;
                    Ok(vec![(spec.node, grad)])
                }),
            )?;
        }

        Ok(out)
    }};
}

macro_rules! run_binary_tensor_ad {
    (
        ty = $ty:ty,
        capability = $capability:expr,
        op = $op_name:literal,
        pullback = $pullback_op_name:literal,
        lhs = $lhs:expr,
        rhs = $rhs:expr,
        primal = |$primal_ctx:ident, $primal_backend:ident, $lhs_primal:ident, $rhs_primal:ident| $primal_body:expr,
        frule = |$frule_ctx:ident, $frule_backend:ident, $frule_lhs:ident, $frule_rhs:ident, $lhs_tangent:ident, $rhs_tangent:ident| $frule_body:expr,
        rrule = |$rrule_ctx:ident, $rrule_backend:ident, $rrule_lhs:ident, $rrule_rhs:ident, $rrule_cotangent:ident| $rrule_body:expr $(,)?
    ) => {{
        let operands = [$lhs, $rhs];
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let ((lhs_primal, lhs_tangent), (rhs_primal, rhs_tangent)) =
            dispatch_linalg_ad_runtime!($ty, $capability, $op_name, |ctx, Backend| {
                Ok((
                    dense_input_snapshot_in_backend::<Backend, _, $ty>(ctx, $lhs, needs_tangent)?,
                    dense_input_snapshot_in_backend::<Backend, _, $ty>(ctx, $rhs, needs_tangent)?,
                ))
            })?;

        let (primal, tangent) = if needs_tangent {
            let lhs_tangent = lhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: format!("{} missing materialized lhs tangent", $op_name),
            })?;
            let rhs_tangent = rhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: format!("{} missing materialized rhs tangent", $op_name),
            })?;
            let (p, d) = dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$frule_ctx, $frule_backend| {
                    let $frule_lhs = &lhs_primal;
                    let $frule_rhs = &rhs_primal;
                    let $lhs_tangent = &lhs_tangent;
                    let $rhs_tangent = &rhs_tangent;
                    $frule_body
                }
            )?;
            (p, Some(d))
        } else {
            let primal = dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$primal_ctx, $primal_backend| {
                    let $lhs_primal = &lhs_primal;
                    let $rhs_primal = &rhs_primal;
                    $primal_body
                }
            )?;
            (primal, None)
        };

        let out = wrap_dense_ad_output($op_name, &operands, primal, tangent, 0)?;

        if let AdValue::Reverse { node, tape, .. } = out.as_value() {
            let reverse_specs = collect_reverse_input_specs(&operands);
            let output_node = *node;
            let tape_id = *tape;

            reverse_tape::register_rule::<$ty>(
                tape_id,
                output_node,
                Box::new(move |cotangent| {
                    let (grad_lhs, grad_rhs) = dispatch_linalg_ad_runtime!(
                        $ty,
                        $capability,
                        $pullback_op_name,
                        |$rrule_ctx, $rrule_backend| {
                            let $rrule_lhs = &lhs_primal;
                            let $rrule_rhs = &rhs_primal;
                            let $rrule_cotangent = cotangent;
                            $rrule_body
                        }
                    )?;

                    let mut input_grads = Vec::new();
                    if let Some(spec) = &reverse_specs[0] {
                        let grad_lhs = compress_pullback_like($op_name, grad_lhs, &spec.layout)?;
                        input_grads.push((spec.node, grad_lhs));
                    }
                    if let Some(spec) = &reverse_specs[1] {
                        let grad_rhs = compress_pullback_like($op_name, grad_rhs, &spec.layout)?;
                        input_grads.push((spec.node, grad_rhs));
                    }
                    Ok(input_grads)
                }),
            )?;
        }

        Ok(out)
    }};
}

pub(super) use dispatch_linalg_ad_runtime;
pub(super) use run_binary_tensor_ad;
pub(super) use run_unary_tensor_ad;
