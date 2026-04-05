use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use chainrules_core::{ADKey, DiffPassId};

use crate::input_key::TensorInputKey;

#[test]
fn tensor_input_key_clone_eq_and_hash_are_stable() {
    let lhs = TensorInputKey::User { id: 7 };
    let rhs = lhs.clone();

    let mut lhs_hasher = DefaultHasher::new();
    lhs.hash(&mut lhs_hasher);
    let mut rhs_hasher = DefaultHasher::new();
    rhs.hash(&mut rhs_hasher);

    assert_eq!(lhs, rhs);
    assert_eq!(lhs_hasher.finish(), rhs_hasher.finish());
    assert_eq!(format!("{lhs:?}"), "User { id: 7 }");
}

#[test]
fn tensor_input_key_tangent_of_derives_a_unique_key() {
    let key = TensorInputKey::User { id: 3 };
    let tangent = key.tangent_of(42 as DiffPassId);
    assert_eq!(
        tangent,
        TensorInputKey::Tangent {
            of: Box::new(TensorInputKey::User { id: 3 }),
            pass: 42,
        }
    );
}
