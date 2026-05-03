use crate::CubeclKernelError;

#[test]
fn unimplemented_launch_returns_invalid_strategy_error() {
    let err = super::unimplemented_launch().unwrap_err();

    assert_eq!(
        err,
        CubeclKernelError::InvalidStrategy {
            reason: "reduction kernels are not implemented yet".to_owned(),
        }
    );
}
