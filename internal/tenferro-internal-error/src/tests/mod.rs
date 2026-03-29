use chainrules_core::AutodiffError;
use tenferro_device::Error as DeviceError;

use crate::{Error, Result};

#[test]
fn runtime_not_configured_display_text_is_stable() {
    assert_eq!(
        Error::RuntimeNotConfigured.to_string(),
        "default runtime is not configured; call `set_default_runtime(...)` first"
    );
}

#[test]
fn device_errors_convert_transparently() {
    let err: Error = DeviceError::InvalidArgument("backend".into()).into();
    assert!(matches!(
        err,
        Error::Backend(DeviceError::InvalidArgument(message)) if message == "backend"
    ));
}

#[test]
fn autodiff_errors_convert_transparently() {
    let err: Error = AutodiffError::InvalidArgument("autodiff".into()).into();
    assert!(matches!(
        err,
        Error::Autodiff(AutodiffError::InvalidArgument(message)) if message == "autodiff"
    ));

    let result: Result<()> = Err(AutodiffError::HvpNotSupported.into());
    assert!(matches!(
        result,
        Err(Error::Autodiff(AutodiffError::HvpNotSupported))
    ));
}
