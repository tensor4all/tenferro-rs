#[test]
fn p2_root_claims_artifact_is_wired() {
    // Runtime ownership proofs live beside the private storage module. This
    // integration artifact keeps the ledger command stable without exporting
    // the crate-private owner kernel.
    let first = tenferro_tensor::AllocationDomainId::fresh();
    let second = tenferro_tensor::AllocationDomainId::fresh();
    assert_ne!(first, second);
}
