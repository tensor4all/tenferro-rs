use crate::{BackendId, DType, Error};

#[test]
fn unsupported_capability_preserves_structured_context_and_display_contract() {
    let error = Error::unsupported_capability(
        "prepare_svd",
        BackendId::Cpu,
        "faer",
        DType::C64,
        "prepared compact SVD",
    );

    assert_eq!(
        error,
        Error::UnsupportedCapability {
            op: "prepare_svd",
            backend: BackendId::Cpu,
            provider: "faer",
            dtype: DType::C64,
            capability: "prepared compact SVD",
        }
    );
    assert!(matches!(
        error,
        Error::UnsupportedCapability {
            op: "prepare_svd",
            backend: BackendId::Cpu,
            provider: "faer",
            dtype: DType::C64,
            capability: "prepared compact SVD",
        }
    ));
    assert_eq!(
        error.to_string(),
        "cpu backend provider faer does not support prepared compact SVD for prepare_svd with dtype C64"
    );
}
