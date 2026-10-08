//! The public session-surface `pass` contract, compiled as integration tests.
//!
//! These cases were trybuild `pass` fixtures. trybuild generates a manifest that
//! only forwards the crate's own feature names (`<crate>/<feature>`), which
//! cannot carry the CPU backend feature this surface needs; a generated manifest
//! therefore builds `tenferro-cpu` with no backend at all. An ordinary
//! integration test is the same compilation environment — a separate crate built
//! against the public API with this package's dev-dependencies — without the
//! generated manifest.
//!
//! The `fail` side of the contract is rustdoc `compile_fail` examples in the
//! library documentation.

#[path = "ui/session_surface/pass/session_cached_route.rs"]
mod session_cached_route;
#[path = "ui/session_surface/pass/session_dyn_erased.rs"]
mod session_dyn_erased;
#[path = "ui/session_surface/pass/session_read_ops.rs"]
mod session_read_ops;
#[path = "ui/session_surface/pass/session_scope_nesting.rs"]
mod session_scope_nesting;
#[path = "ui/session_surface/pass/session_tensor_extension.rs"]
mod session_tensor_extension;
