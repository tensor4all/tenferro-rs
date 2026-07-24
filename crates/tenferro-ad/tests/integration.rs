#[path = "integration/support/mod.rs"]
mod support;

#[path = "integration/ad_api_naming_contract.rs"]
mod ad_api_naming_contract;
#[path = "integration/ad_optimizer.rs"]
mod ad_optimizer;
#[path = "integration/ad_structural_primitives.rs"]
mod ad_structural_primitives;
#[path = "integration/binding_validation.rs"]
mod binding_validation;
#[path = "integration/broadcast_error_parity.rs"]
mod broadcast_error_parity;
#[path = "integration/cache_management.rs"]
mod cache_management;
#[path = "integration/checkpoint.rs"]
mod checkpoint;
#[path = "integration/checkpoint_truncate_integration.rs"]
mod checkpoint_truncate_integration;
#[path = "integration/convenience_api.rs"]
mod convenience_api;
#[path = "integration/cpu_backend.rs"]
mod cpu_backend;
#[path = "integration/dot_general_validation.rs"]
mod dot_general_validation;
#[path = "integration/dtype_propagation.rs"]
mod dtype_propagation;
#[path = "integration/dynamic_truncate.rs"]
mod dynamic_truncate;
#[path = "integration/eager_device_placement_contract.rs"]
mod eager_device_placement_contract;
#[path = "integration/eager_fixed_pivot_cross.rs"]
mod eager_fixed_pivot_cross;
#[path = "integration/eager_runtime_api.rs"]
mod eager_runtime_api;
#[path = "integration/eager_tensor.rs"]
mod eager_tensor;
#[path = "integration/engine_eval.rs"]
mod engine_eval;
#[path = "integration/fallible_api.rs"]
mod fallible_api;
#[path = "integration/gpu_ad_tests.rs"]
mod gpu_ad_tests;
#[path = "integration/gpu_f32_fusion.rs"]
mod gpu_f32_fusion;
#[path = "integration/graph_compile.rs"]
mod graph_compile;
#[path = "integration/graph_executor.rs"]
mod graph_executor;
#[path = "integration/hvp.rs"]
mod hvp;
#[path = "integration/iterative_ad.rs"]
mod iterative_ad;
#[path = "integration/memory_order_api.rs"]
mod memory_order_api;
#[path = "integration/numpy_api.rs"]
mod numpy_api;
#[path = "integration/placement_bound_eager.rs"]
mod placement_bound_eager;
#[path = "integration/primitive_ops.rs"]
mod primitive_ops;
#[path = "integration/runtime_snapshot_bridge.rs"]
mod runtime_snapshot_bridge;
#[path = "integration/semantic_extension.rs"]
mod semantic_extension;
#[path = "integration/semantic_transform.rs"]
mod semantic_transform;
#[path = "integration/shape_inference.rs"]
mod shape_inference;
#[path = "integration/shape_of.rs"]
mod shape_of;
#[path = "integration/staging_surface_contract.rs"]
mod staging_surface_contract;
#[path = "integration/sym_dim.rs"]
mod sym_dim;
#[path = "integration/symbolic_grad.rs"]
mod symbolic_grad;
#[path = "integration/symbolic_input.rs"]
mod symbolic_input;
