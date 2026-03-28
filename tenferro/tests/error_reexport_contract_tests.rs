use tenferro::{Error, Result};
use tenferro_internal_error::{Error as InternalError, Result as InternalResult};

#[test]
fn public_error_surface_reexports_the_internal_error_types() {
    let public = Error::RuntimeNotConfigured;
    let _: InternalError = public;

    let internal = InternalError::RuntimeNotConfigured;
    let _: Error = internal;

    let public_result: Result<()> = Err(Error::RuntimeNotConfigured);
    let _: InternalResult<()> = public_result;
}
