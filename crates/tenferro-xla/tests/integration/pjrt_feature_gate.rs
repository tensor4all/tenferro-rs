#![cfg(not(feature = "pjrt"))]

use tenferro_xla::{Error, XlaExecutor};

#[test]
fn from_env_requires_pjrt_feature() {
    let err = XlaExecutor::from_env().unwrap_err();

    assert!(matches!(err, Error::PjrtFeatureDisabled));
}
