#[path = "integration/ad_support_manifest.rs"]
mod ad_support_manifest;
#[cfg(all(feature = "autodiff", feature = "webgpu"))]
#[path = "integration/apple_shared.rs"]
mod apple_shared;
#[path = "integration/backend_errors.rs"]
mod backend_errors;
#[path = "integration/concrete_surface.rs"]
mod concrete_surface;
#[path = "integration/cpu_linalg_source_contract.rs"]
mod cpu_linalg_source_contract;
#[path = "integration/eager_surface_parity.rs"]
mod eager_surface_parity;
#[path = "integration/eager_tensor.rs"]
mod eager_tensor;
#[path = "integration/full_piv_lu.rs"]
mod full_piv_lu;
#[path = "integration/full_svd_lstsq.rs"]
mod full_svd_lstsq;
#[path = "integration/gpu_linalg.rs"]
mod gpu_linalg;
#[path = "integration/gpu_linalg_source_contract.rs"]
mod gpu_linalg_source_contract;
#[path = "integration/householder_qr.rs"]
mod householder_qr;
#[path = "integration/incremental_qr_performance_contract.rs"]
mod incremental_qr_performance_contract;
#[path = "integration/inject_dual_abi_tests.rs"]
mod inject_dual_abi_tests;
#[path = "integration/inject_tests.rs"]
mod inject_tests;
#[path = "integration/linalg_internal_path_contract.rs"]
mod linalg_internal_path_contract;
#[cfg(feature = "autodiff")]
#[path = "integration/oracle_replay.rs"]
mod oracle_replay;
#[path = "integration/small_append_performance_contract.rs"]
mod small_append_performance_contract;
#[path = "integration/support.rs"]
mod support;
#[path = "integration/traced_ad_explicit.rs"]
mod traced_ad_explicit;
#[path = "integration/traced_correctness.rs"]
mod traced_correctness;
#[path = "integration/traced_extension.rs"]
mod traced_extension;
