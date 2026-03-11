use std::sync::atomic::{AtomicU64, Ordering};

use chainrules_scalarops as scalarops;

use crate::reverse_tape;

use super::super::core::{AdValue, NodeId};

pub(super) static NEXT_AD_SCALAR_NODE_ID: AtomicU64 = AtomicU64::new(1_u64 << 62);

pub(super) fn fresh_ad_scalar_node_id() -> NodeId {
    NodeId(NEXT_AD_SCALAR_NODE_ID.fetch_add(1, Ordering::Relaxed))
}

pub(crate) fn map_ad_value_same_type_linear<T, M>(
    value: AdValue<T>,
    op_name: &'static str,
    map: M,
) -> AdValue<T>
where
    T: scalarops::ScalarAd + 'static,
    M: Fn(T) -> T + Copy + 'static,
{
    match value {
        AdValue::Primal(primal) => AdValue::Primal(map(primal)),
        AdValue::Forward { primal, tangent } => AdValue::Forward {
            primal: map(primal),
            tangent: map(tangent),
        },
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = map(primal);
            let output_tangent = tangent.map(map);
            let output_node = fresh_ad_scalar_node_id();
            reverse_tape::register_scalar_rule(
                tape,
                output_node,
                Box::new(move |cotangent| Ok(vec![(input_node, map(*cotangent))])),
            )
            .unwrap_or_else(|e| panic!("{op_name}: {e}"));
            AdValue::Reverse {
                primal: output_primal,
                node: output_node,
                tape,
                tangent: output_tangent,
            }
        }
    }
}

pub(crate) fn map_ad_value_mixed_linear<TIn, TOut, P, R>(
    value: AdValue<TIn>,
    op_name: &'static str,
    primal_map: P,
    reverse_map: R,
) -> AdValue<TOut>
where
    TIn: scalarops::ScalarAd + 'static,
    TOut: scalarops::ScalarAd + 'static,
    P: Fn(TIn) -> TOut + Copy,
    R: Fn(TOut) -> TIn + Copy + 'static,
{
    match value {
        AdValue::Primal(primal) => AdValue::Primal(primal_map(primal)),
        AdValue::Forward { primal, tangent } => AdValue::Forward {
            primal: primal_map(primal),
            tangent: primal_map(tangent),
        },
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = primal_map(primal);
            let output_tangent = tangent.map(primal_map);
            let output_node = fresh_ad_scalar_node_id();
            reverse_tape::register_scalar_mixed_rule(
                tape,
                output_node,
                Box::new(move |cotangent| Ok(vec![(input_node, reverse_map(*cotangent))])),
            )
            .unwrap_or_else(|e| panic!("{op_name}: {e}"));
            AdValue::Reverse {
                primal: output_primal,
                node: output_node,
                tape,
                tangent: output_tangent,
            }
        }
    }
}
