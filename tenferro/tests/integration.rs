#[path = "support/mod.rs"]
mod support;

#[path = "cases/autograd_surface_error_tests.rs"]
mod autograd_surface_error_tests;
#[path = "cases/dyn_tensor_shape_ops_tests.rs"]
mod dyn_tensor_shape_ops_tests;
#[path = "cases/dynamic_tensor_placement_dispatch_tests.rs"]
mod dynamic_tensor_placement_dispatch_tests;
#[path = "cases/dynamic_wrapper_coverage_tests.rs"]
mod dynamic_wrapper_coverage_tests;
#[path = "cases/einsum_label_order_regression.rs"]
mod einsum_label_order_regression;
#[path = "cases/error_reexport_contract_tests.rs"]
mod error_reexport_contract_tests;
#[path = "cases/homogeneous_tape_tests.rs"]
mod homogeneous_tape_tests;
#[path = "cases/hvp_tests.rs"]
mod hvp_tests;
#[path = "cases/linalg_frontend_gap_tests.rs"]
mod linalg_frontend_gap_tests;
#[path = "cases/linalg_memory_order_tests.rs"]
mod linalg_memory_order_tests;
#[path = "cases/mixed_complex_real_scalar_tests.rs"]
mod mixed_complex_real_scalar_tests;
#[path = "cases/mixed_primitives_forward_tests.rs"]
mod mixed_primitives_forward_tests;
#[path = "cases/mixed_primitives_reverse_tests.rs"]
mod mixed_primitives_reverse_tests;
#[path = "cases/projection_reverse_tests.rs"]
mod projection_reverse_tests;
#[path = "cases/public_surface_tests.rs"]
mod public_surface_tests;
#[path = "cases/rank0_ad_tensor_tests.rs"]
mod rank0_ad_tensor_tests;
#[path = "cases/runtime_reexport_contract_tests.rs"]
mod runtime_reexport_contract_tests;
#[path = "cases/snapshot_surface_tests.rs"]
mod snapshot_surface_tests;
#[path = "cases/structured_layout_validation_tests.rs"]
mod structured_layout_validation_tests;
#[path = "cases/structured_linalg_fallback_tests.rs"]
mod structured_linalg_fallback_tests;
#[path = "cases/structured_reverse_tests.rs"]
mod structured_reverse_tests;
#[path = "cases/structured_tensor_root_tests.rs"]
mod structured_tensor_root_tests;
#[path = "cases/tensor_mode_and_conj_tests.rs"]
mod tensor_mode_and_conj_tests;
#[path = "cases/tensor_permute_tests.rs"]
mod tensor_permute_tests;
#[path = "cases/tensor_placement_surface_tests.rs"]
mod tensor_placement_surface_tests;
#[path = "cases/tensor_shape_surface_tests.rs"]
mod tensor_shape_surface_tests;
#[path = "cases/tensor_ui_contract_tests.rs"]
mod tensor_ui_contract_tests;
#[path = "cases/thread_safety_tests.rs"]
mod thread_safety_tests;
#[path = "cases/workspace_taxonomy_contract_tests.rs"]
mod workspace_taxonomy_contract_tests;
