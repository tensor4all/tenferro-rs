use t4a_cuda_kernels::fusion::{emit_fusion, FusionOperand};

use super::softplus_plan;

#[test]
fn softplus_plan_is_one_input_one_output_region_the_emitter_accepts() {
    for dtype in [t4a_cuda_kernels::DType::F32, t4a_cuda_kernels::DType::F64] {
        let plan = softplus_plan(dtype);
        assert_eq!(plan.inputs, 1);
        assert_eq!(plan.outputs, vec![FusionOperand::Value(5)]);
        let emitted = emit_fusion(&plan).expect("the catalogue emitter accepts softplus");
        let manifest = emitted.manifest();
        assert_eq!((manifest.inputs, manifest.outputs), (1, 1));
        assert!(emitted.source().contains("t4a_cuda_log1p("));
        assert!(emitted.source().contains("t4a_cuda_max("));
    }
}
