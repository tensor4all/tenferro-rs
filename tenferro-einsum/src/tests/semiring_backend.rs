use tenferro_algebra::Standard;
use tenferro_device::Result;
use tenferro_prims::{
    CpuBackend, CpuContext, SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore,
    TensorSemiringFastPath,
};
use tenferro_tensor::Tensor;

pub(crate) struct SemiringOnlyCpuBackend;

impl TensorSemiringCore<Standard<f64>> for SemiringOnlyCpuBackend {
    type Plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::Plan;
    type Context = CpuContext;

    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(ctx, desc, shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: f64,
        inputs: &[&Tensor<f64>],
        beta: f64,
        output: &mut Tensor<f64>,
    ) -> Result<()> {
        <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
            ctx, plan, alpha, inputs, beta, output,
        )
    }
}

impl TensorSemiringFastPath<Standard<f64>> for SemiringOnlyCpuBackend {
    type Plan = <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::Plan;
    type Context = CpuContext;

    fn plan(
        ctx: &mut Self::Context,
        desc: &SemiringFastPathDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::plan(ctx, desc, shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: f64,
        inputs: &[&Tensor<f64>],
        beta: f64,
        output: &mut Tensor<f64>,
    ) -> Result<()> {
        <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::execute(
            ctx, plan, alpha, inputs, beta, output,
        )
    }

    fn has_fast_path(desc: SemiringFastPathDescriptor) -> bool {
        <CpuBackend as TensorSemiringFastPath<Standard<f64>>>::has_fast_path(desc)
    }
}
