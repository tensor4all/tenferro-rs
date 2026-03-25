use tenferro_device::{Error, Result};

pub(crate) fn unsupported<T>(op: &str) -> Result<T> {
    Err(Error::DeviceError(format!(
        "HIP linalg backend operation {op} is not yet implemented"
    )))
}
