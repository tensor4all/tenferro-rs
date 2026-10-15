//! #2143 item 1: the session softmax must reach its broadcast as a metadata view
//! rather than through the materializing `broadcast_in_dim_read` route.
#![cfg(feature = "native")]

use tenferro_cpu::CpuBackend;
use tenferro_runtime::composite::{
    set_broadcast_route_observation, take_broadcast_routes, BroadcastRoute,
};
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
use tenferro_tensor::BackendSessionHost;

#[test]
fn the_session_softmax_broadcasts_as_a_view() -> tenferro_runtime::Result<()> {
    // The broadcast that subtracts the clamped row maximum is the one the
    // acceptance names; at the reported [512, 512, 8] reproducer it would be a
    // full-size tensor.
    let x = TypedTensor::<f32>::from_vec_col_major(
        vec![2, 4, 2],
        (0..16).map(|index| index as f32 * 0.5 - 3.0).collect(),
    )?;
    // A right-aligned mask that the composite has to broadcast explicitly.
    let mask = TypedTensor::<bool>::from_vec_col_major(vec![1, 2], vec![true, false])?;
    let mut backend = CpuBackend::with_threads(1).expect("a CPU backend with one thread");

    set_broadcast_route_observation(true);
    let (softmax_routes, y) = backend.with_backend_session(|session| {
        let y = x.softmax(1, session)?;
        Ok::<_, tenferro_runtime::Error>((take_broadcast_routes(), y))
    })??;
    let (masked_routes, masked) = backend.with_backend_session(|session| {
        let masked = x.masked_softmax(&mask, 1, session)?;
        Ok::<_, tenferro_runtime::Error>((take_broadcast_routes(), masked))
    })??;
    set_broadcast_route_observation(false);

    assert_eq!(y.shape(), &[2, 4, 2]);
    assert_eq!(masked.shape(), &[2, 4, 2]);
    assert!(
        !softmax_routes.is_empty(),
        "the softmax composite broadcasts the reduced maximum and ratio back"
    );
    assert!(
        softmax_routes
            .iter()
            .all(|route| *route == BroadcastRoute::View),
        "every softmax broadcast must be a metadata view, got {softmax_routes:?}"
    );
    // The right-aligned mask is the deliberate materializing broadcast (the
    // composite only holds a borrow of it), which also shows that the counter
    // would see a regression on the softmax route.
    assert_eq!(
        masked_routes
            .iter()
            .filter(|route| **route == BroadcastRoute::Materializing)
            .count(),
        1,
        "the mask broadcast is the one materializing route, got {masked_routes:?}"
    );
    assert!(
        masked_routes.contains(&BroadcastRoute::View),
        "the masked form still takes the view route for its reduced operands"
    );
    Ok(())
}
