//! A host-backed tensor cannot be relabelled: `set_placement` lives on the `Dynamic` union only.
//!
//! The union is where the metadata is allowed to disagree with the payload (that is what lets a test
//! fabricate a device label on host memory and check that an entry point refuses). A `Host`-typed
//! value is evidence, so the label must not be writable on it, or an entry point that takes `Host`
//! and skips its runtime check would be trusting a field anyone can overwrite.
use tenferro_tensor::{Host, MemoryKind, Placement, TypedTensor};

fn main() {
    let mut host: TypedTensor<f64, _, Host> =
        TypedTensor::from_host_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    host.set_placement(Placement {
        memory_kind: MemoryKind::Device,
        device: None,
        cpu_affinity: None,
    });
}
