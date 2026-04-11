pub use tenferro::{
    set_default_runtime, with_default_runtime, DefaultRuntimeGuard, Error, Result, RuntimeContext,
    Tensor,
};
pub use tenferro_internal_ad_core::{AdMode, AdTensor, NodeId};
pub use tenferro_internal_frontend_core::DynTensor;

mod core {
    pub use tenferro_internal_ad_surface::core::*;
    pub use tenferro_internal_frontend_core::DynTensor;
}

mod runtime {
    pub use tenferro::{set_default_runtime, with_default_runtime, RuntimeContext};
    pub use tenferro_internal_runtime::{with_runtime, DefaultRuntimeGuard};

    pub mod contracts {
        pub use tenferro_internal_runtime::contracts::*;
    }

    pub mod dispatch {
        pub use tenferro_internal_runtime::dispatch::*;
    }
}

mod structured {
    pub use tenferro_internal_frontend_core::{
        accumulate_tangent, compress_dense_to_layout_in_ctx, einsum_with_subscripts_in_ctx,
        first_duplicate_pair, normalize_payload_for_roots, plan_axis_classes_for_subscripts,
        reverse_subscripts, to_dense_in_ctx, unique_ids_first_appearance, usize_vec_to_u32,
        AxisClassPlanError, OperandAxisClasses, StructuredTensor,
    };
    pub use tenferro_tensor::structured_tensor::canonicalize_axis_classes;

    pub use tenferro_internal_frontend_core::accumulate_tangent as accumulate_structured_tangent;

    pub mod meta {
        pub use tenferro_internal_frontend_core::{
            plan_axis_classes_for_subscripts, AxisClassPlanError, OperandAxisClasses,
        };
    }
}

mod tape {
    pub use tenferro_internal_ad_core::{
        pullback, register_closure_rule, register_mixed_rule, register_rule,
    };
}

mod ops {
    pub use tenferro_internal_ad_core::ops::*;
    pub use tenferro_internal_ad_linalg::__typed_ad::*;
    pub use tenferro_internal_ad_linalg::__typed_results::*;
    pub use tenferro_internal_ad_ops::__typed_einsum::*;
    pub use tenferro_internal_ad_ops::__typed_reduction::*;
    pub use tenferro_internal_ad_ops::__typed_scalar::*;
    pub use tenferro_internal_ad_surface::__typed_linalg_primal::*;

    pub mod ad {
        pub use tenferro_internal_ad_linalg::__typed_eager::*;
        pub use tenferro_internal_ad_ops::__typed_ad::*;
    }

    pub mod scalar {
        pub mod primal {
            pub use tenferro_internal_ad_ops::__typed_scalar::primal::*;
        }
    }

    pub mod tests {
        pub(crate) use crate::ops_unit_port::support::{
            as_slice, assert_primal_mode, reverse_leaf_f64, with_cpu_runtime,
        };

        pub mod support {
            pub(crate) use crate::ops_unit_port::support::*;
        }
    }
}

#[path = "support/mod.rs"]
mod support;

#[path = "integration/autograd_surface_error_tests.rs"]
mod autograd_surface_error_tests;
#[path = "integration/core_value_surface_regressions.rs"]
mod core_value_surface_regressions;
// TODO: re-enable once dyn_tensor_shape_ops_tests.rs is implemented
// #[path = "integration/dyn_tensor_shape_ops_tests.rs"]
// mod dyn_tensor_shape_ops_tests;
// TODO: re-enable once dynamic_tensor_placement_dispatch_tests.rs is implemented
// #[path = "integration/dynamic_tensor_placement_dispatch_tests.rs"]
// mod dynamic_tensor_placement_dispatch_tests;
#[path = "integration/dynamic_wrapper_coverage_tests.rs"]
mod dynamic_wrapper_coverage_tests;
#[path = "integration/einsum_label_order_regression.rs"]
mod einsum_label_order_regression;
// TODO: re-enable once error_reexport_contract_tests.rs is implemented
// #[path = "integration/error_reexport_contract_tests.rs"]
// mod error_reexport_contract_tests;
// TODO: re-enable once homogeneous_tape_tests.rs is implemented
// #[path = "integration/homogeneous_tape_tests.rs"]
// mod homogeneous_tape_tests;
#[path = "integration/hvp_tests.rs"]
mod hvp_tests;
// TODO: re-enable once linalg_frontend_gap_tests.rs is implemented
// #[path = "integration/linalg_frontend_gap_tests.rs"]
// mod linalg_frontend_gap_tests;
// TODO: re-enable once linalg_memory_order_tests.rs is implemented
// #[path = "integration/linalg_memory_order_tests.rs"]
// mod linalg_memory_order_tests;
#[path = "integration/mixed_complex_real_scalar_tests.rs"]
mod mixed_complex_real_scalar_tests;
// TODO: re-enable once mixed_primitives_forward_tests.rs is implemented
// #[path = "integration/mixed_primitives_forward_tests.rs"]
// mod mixed_primitives_forward_tests;
// TODO: re-enable once mixed_primitives_reverse_tests.rs is implemented
// #[path = "integration/mixed_primitives_reverse_tests.rs"]
// mod mixed_primitives_reverse_tests;
#[path = "integration/ops_ad_unit_port.rs"]
mod ops_ad_unit_port;
#[path = "integration/ops_unit_port.rs"]
mod ops_unit_port;
// TODO: re-enable once projection_reverse_tests.rs is implemented
// #[path = "integration/projection_reverse_tests.rs"]
// mod projection_reverse_tests;
#[path = "integration/autograd_surface_tests.rs"]
mod autograd_surface_tests;
#[path = "integration/linalg_surface_tests.rs"]
mod linalg_surface_tests;
#[path = "integration/migration_organization.rs"]
mod migration_organization;
#[path = "integration/public_jvp.rs"]
mod public_jvp;
#[path = "integration/public_surface_tests.rs"]
mod public_surface_tests;
