macro_rules! dispatch_linalg_ad_runtime {
    ($ty:ty, $capability:expr, $op:literal, |$ctx:ident| $body:expr) => {{
        with_linalg_runtime::<$ty, _>($op, $capability, |$ctx| $body, |$ctx| $body, |$ctx| $body)
    }};
    ($ty:ty, $capability:expr, $op:literal, |$ctx:ident, $backend:ident| $body:expr) => {{
        with_linalg_runtime::<$ty, _>(
            $op,
            $capability,
            |$ctx| {
                type $backend = tenferro_prims::CpuBackend;
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
        primal = |$primal_ctx:ident, $primal_tensor:ident| $primal_body:expr,
        frule = |$frule_ctx:ident, $frule_tensor:ident, $frule_tangent:ident| $frule_body:expr,
        rrule = |$rrule_ctx:ident, $rrule_tensor:ident, $rrule_cotangent:ident| $rrule_body:expr $(,)?
    ) => {{
        let operands = [$input];
        ensure_dense_linalg_inputs($op_name, &operands)?;
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let (input_primal, input_tangent) =
            $crate::ops::linalg::ad::common::dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |ctx, Backend| {
                dense_input_snapshot_in_backend::<Backend, _, $ty>(ctx, $input, needs_tangent)
            })?;

        let (primal, tangent) = if needs_tangent {
            let input_tangent = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
                message: format!("{} missing materialized tangent", $op_name),
            })?;
            let (p, d) = $crate::ops::linalg::ad::common::dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$frule_ctx| {
                let $frule_tensor = &input_primal;
                let $frule_tangent = &input_tangent;
                $frule_body
            })?;
            (p, Some(d))
        } else {
            let primal = $crate::ops::linalg::ad::common::dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$primal_ctx| {
                let $primal_tensor = &input_primal;
                $primal_body
            })?;
            (primal, None)
        };

        let out = wrap_same_type_dense_ad_output($op_name, &operands, primal, tangent)?;

        if let Some((node, tape)) = out.reverse_handle() {
            let input_spec = collect_reverse_input_specs(&operands)
                .into_iter()
                .next()
                .flatten();

            let input_node_ids: Vec<_> = input_spec.iter().map(|s| s.node).collect();
            tape::register_closure_rule::<$ty>(
                &tape,
                node,
                input_node_ids,
                Box::new(move |cotangent| {
                    let grad = $crate::ops::linalg::ad::common::dispatch_linalg_ad_runtime!(
                        $ty,
                        $capability,
                        $pullback_op_name,
                        |$rrule_ctx| {
                            let $rrule_tensor = &input_primal;
                            let $rrule_cotangent = cotangent.payload();
                            $rrule_body
                        }
                    )?;

                    let Some(spec) = &input_spec else {
                        return Ok(Vec::new());
                    };
                    let grad = spec.layout.with_payload_like(compress_pullback_like(
                        $op_name,
                        grad,
                        &spec.layout,
                    )?)?;
                    Ok(vec![(spec.node, grad)])
                }),
            );
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
        primal = |$primal_ctx:ident, $lhs_primal:ident, $rhs_primal:ident| $primal_body:expr,
        frule = |$frule_ctx:ident, $frule_lhs:ident, $frule_rhs:ident, $lhs_tangent:ident, $rhs_tangent:ident| $frule_body:expr,
        rrule = |$rrule_ctx:ident, $rrule_lhs:ident, $rrule_rhs:ident, $rrule_cotangent:ident| $rrule_body:expr $(,)?
    ) => {{
        let operands = [$lhs, $rhs];
        ensure_dense_linalg_inputs($op_name, &operands)?;
        let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
        let ((lhs_primal, lhs_tangent), (rhs_primal, rhs_tangent)) =
            $crate::ops::linalg::ad::common::dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |ctx, Backend| {
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
            let (p, d) = $crate::ops::linalg::ad::common::dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$frule_ctx| {
                let $frule_lhs = &lhs_primal;
                let $frule_rhs = &rhs_primal;
                let $lhs_tangent = &lhs_tangent;
                let $rhs_tangent = &rhs_tangent;
                $frule_body
            })?;
            (p, Some(d))
        } else {
            let primal = $crate::ops::linalg::ad::common::dispatch_linalg_ad_runtime!(
                $ty,
                $capability,
                $op_name,
                |$primal_ctx| {
                let $lhs_primal = &lhs_primal;
                let $rhs_primal = &rhs_primal;
                $primal_body
            })?;
            (primal, None)
        };

        let out = wrap_same_type_dense_ad_output($op_name, &operands, primal, tangent)?;

        if let Some((node, tape)) = out.reverse_handle() {
            let reverse_specs = collect_reverse_input_specs(&operands);

            let input_node_ids: Vec<_> = reverse_specs
                .iter()
                .filter_map(|s| s.as_ref().map(|s| s.node))
                .collect();
            tape::register_closure_rule::<$ty>(
                &tape,
                node,
                input_node_ids,
                Box::new(move |cotangent| {
                    let (grad_lhs, grad_rhs) =
                        $crate::ops::linalg::ad::common::dispatch_linalg_ad_runtime!(
                        $ty,
                        $capability,
                        $pullback_op_name,
                        |$rrule_ctx| {
                            let $rrule_lhs = &lhs_primal;
                            let $rrule_rhs = &rhs_primal;
                            let $rrule_cotangent = cotangent.payload();
                            $rrule_body
                        }
                    )?;

                    let mut input_grads = Vec::new();
                    if let Some(spec) = &reverse_specs[0] {
                        let grad_lhs = spec.layout.with_payload_like(compress_pullback_like(
                            $op_name,
                            grad_lhs,
                            &spec.layout,
                        )?)?;
                        input_grads.push((spec.node, grad_lhs));
                    }
                    if let Some(spec) = &reverse_specs[1] {
                        let grad_rhs = spec.layout.with_payload_like(compress_pullback_like(
                            $op_name,
                            grad_rhs,
                            &spec.layout,
                        )?)?;
                        input_grads.push((spec.node, grad_rhs));
                    }
                    Ok(input_grads)
                }),
            );
        }

        Ok(out)
    }};
}

pub(crate) use dispatch_linalg_ad_runtime;
pub(crate) use run_binary_tensor_ad;
pub(crate) use run_unary_tensor_ad;
