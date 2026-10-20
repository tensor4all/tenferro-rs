use std::num::NonZeroU64;

use tenferro_tensor::{DType, Tensor};

use crate::exec::{ExecInstruction, ExecOp, ExecSlot};
use crate::runtime::execution::{retain_instruction_results, LocatedExecSlot};
use crate::runtime::schedule::ExecutionLocation;
use crate::runtime::{
    EngineId, EventDomainId, ProviderDeviceIdentity, ProviderId, RegistrationIdentity,
    RuntimeEpoch, RuntimeId, StorageClass,
};

fn test_provider_device_identity(target: &str) -> ProviderDeviceIdentity {
    ProviderDeviceIdentity::new(
        ProviderId::new("tenferro.test.execution").expect("provider id"),
        target,
    )
    .expect("provider target")
}

fn qualified_domain(ordinal: u64) -> EventDomainId {
    EventDomainId::runtime_created_for_test(
        RuntimeId::from_nonzero(NonZeroU64::new(1).expect("runtime id")),
        RuntimeEpoch::from_nonzero(NonZeroU64::new(1).expect("runtime epoch")),
        RegistrationIdentity::new(
            NonZeroU64::new(1).expect("registration issuer"),
            NonZeroU64::new(ordinal).expect("registration ordinal"),
        ),
    )
}

#[test]
fn result_retention_preserves_output_that_reuses_last_input_slot() {
    let location = ExecutionLocation::new(
        EngineId::new("tenferro-test.same-slot-engine").expect("engine id"),
        test_provider_device_identity("same-slot"),
        qualified_domain(1),
        StorageClass::new("tenferro-test.same-slot-storage").expect("storage class"),
    );
    let instruction = ExecInstruction {
        op: ExecOp::Negate,
        semantic_operation_index: None,
        input_slots: vec![0],
        output_slots: vec![0],
        dtype: DType::F64,
        output_shapes: Default::default(),
        output_extents: Default::default(),
        last_use: vec![true],
    };
    let output = Tensor::from_vec_col_major(vec![2], vec![-1.0_f64, -2.0]).expect("output tensor");
    let mut staged = vec![Some(ExecSlot::Owned(output))];
    let mut located: Vec<Vec<LocatedExecSlot<'_>>> = vec![Vec::new()];

    retain_instruction_results(&instruction, &location, &mut located, &mut staged)
        .expect("retain output");

    assert!(staged[0].is_none());
    assert_eq!(located[0].len(), 1);
    assert_eq!(located[0][0].location, location);
    let ExecSlot::Owned(output) = &located[0][0].value else {
        panic!("same-slot output must remain owned");
    };
    assert_eq!(output.as_slice::<f64>().expect("f64 output"), &[-1.0, -2.0]);
}
