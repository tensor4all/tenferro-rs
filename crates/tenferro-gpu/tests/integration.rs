#[path = "integration/backend_read_contract.rs"]
mod backend_read_contract;
#[cfg(feature = "cuda")]
#[path = "integration/cuda_refused_allocation.rs"]
mod cuda_refused_allocation;
#[path = "integration/engine_run.rs"]
mod engine_run;
#[path = "integration/public_surface_contract.rs"]
mod public_surface_contract;
#[path = "integration/session_contract.rs"]
mod session_contract;
#[path = "integration/session_surface_parity.rs"]
mod session_surface_parity;
