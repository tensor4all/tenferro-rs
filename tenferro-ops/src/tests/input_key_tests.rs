use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use chainrules_core::{ADKey, DiffPassId};

use crate::input_key::TensorInputKey;

#[test]
fn tensor_input_key_clone_eq_and_hash_are_stable() {
    let lhs = TensorInputKey { id: 7 };
    let rhs = lhs.clone();

    let mut lhs_hasher = DefaultHasher::new();
    lhs.hash(&mut lhs_hasher);
    let mut rhs_hasher = DefaultHasher::new();
    rhs.hash(&mut rhs_hasher);

    assert_eq!(lhs, rhs);
    assert_eq!(lhs_hasher.finish(), rhs_hasher.finish());
    assert!(format!("{lhs:?}").contains("TensorInputKey"));
}

#[test]
#[should_panic]
fn tensor_input_key_tangent_of_is_explicitly_unimplemented() {
    let key = TensorInputKey { id: 3 };
    let _ = key.tangent_of(42 as DiffPassId);
}
