use std::any::TypeId;

use num_traits::Float;
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::{Algebra, Scalar, Semiring};
use tenferro_device::{Error, Result};
use tenferro_prims::{
    CpuBackend, CpuContext, SemiringCoreDescriptor, SemiringFastPathDescriptor, TensorSemiringCore,
    TensorSemiringFastPath,
};
use tenferro_tensor::Tensor;

use super::execute::{execute_batched_gemm_optimized, tropical_execute, TropicalGemmDispatch};
use super::plan::{tropical_plan, TropicalPlan};
use super::view::{tensor_to_view, tensor_to_view_mut};
use crate::algebra::{MaxMulAlgebra, MaxPlusAlgebra, MinPlusAlgebra};
use crate::scalar::{MaxMul, MaxPlus, MinPlus};

/// Try to dispatch BatchedGemm to the SIMD-optimized path for a concrete type.
///
/// SAFETY: The TypeId check guarantees T == $concrete_ty before transmuting.
/// All tropical scalar types are #[repr(transparent)] over their inner type,
/// so the transmute is sound.
macro_rules! try_simd_dispatch {
    ($T:ty, [$($concrete_ty:ty),+ $(,)?], $inputs:expr, $alpha:expr, $beta:expr,
     $output:expr, $batch_dims:expr, $m:expr, $n:expr, $k:expr) => {{
        let tid = TypeId::of::<$T>();
        $(
            if tid == TypeId::of::<$concrete_ty>() {
                let a = unsafe {
                    &*($inputs[0] as *const StridedView<$T> as *const StridedView<$concrete_ty>)
                };
                let b = unsafe {
                    &*($inputs[1] as *const StridedView<$T> as *const StridedView<$concrete_ty>)
                };
                let out = unsafe {
                    &mut *(
                        $output as *mut StridedViewMut<$T>
                            as *mut StridedViewMut<$concrete_ty>
                    )
                };
                let alpha = unsafe { *(&$alpha as *const $T as *const $concrete_ty) };
                let beta = unsafe { *(&$beta as *const $T as *const $concrete_ty) };
                return execute_batched_gemm_optimized(
                    alpha,
                    &[a, b],
                    beta,
                    out,
                    $batch_dims,
                    $m,
                    $n,
                    $k,
                );
            }
        )+
    }};
}

macro_rules! impl_tropical_prims {
    ($marker:ident, $wrapper:ident) => {
        impl<S: Scalar + Float> TensorSemiringCore<$marker<S>> for CpuBackend
        where
            $marker<S>: Algebra<Scalar = $wrapper<S>> + Semiring<Scalar = $wrapper<S>>,
            $wrapper<S>: Scalar + TropicalGemmDispatch,
        {
            type Plan = TropicalPlan<$wrapper<S>>;
            type Context = CpuContext;

            fn plan(
                _ctx: &mut CpuContext,
                desc: &SemiringCoreDescriptor,
                shapes: &[&[usize]],
            ) -> Result<TropicalPlan<$wrapper<S>>> {
                tropical_plan(desc, shapes)
            }

            fn execute(
                _ctx: &mut CpuContext,
                plan: &TropicalPlan<$wrapper<S>>,
                alpha: $wrapper<S>,
                inputs: &[&Tensor<$wrapper<S>>],
                beta: $wrapper<S>,
                output: &mut Tensor<$wrapper<S>>,
            ) -> Result<()> {
                let views: Vec<StridedView<$wrapper<S>>> = inputs
                    .iter()
                    .map(|t| tensor_to_view(t))
                    .collect::<Result<_>>()?;
                let view_refs: Vec<&StridedView<$wrapper<S>>> = views.iter().collect();
                let mut out_view = tensor_to_view_mut(output)?;

                if let TropicalPlan::BatchedGemm {
                    batch_dims,
                    m,
                    n,
                    k,
                    ..
                } = plan
                {
                    if views.len() != 2 {
                        return Err(Error::InvalidArgument(
                            "BatchedGemm execute requires 2 input tensors".into(),
                        ));
                    }
                    try_simd_dispatch!(
                        $wrapper<S>,
                        [$wrapper<f64>, $wrapper<f32>],
                        &view_refs,
                        alpha,
                        beta,
                        &mut out_view,
                        batch_dims,
                        *m,
                        *n,
                        *k
                    );
                }
                tropical_execute(plan, alpha, &view_refs, beta, &mut out_view)
            }
        }

        impl<S: Scalar + Float> TensorSemiringFastPath<$marker<S>> for CpuBackend
        where
            $marker<S>: Algebra<Scalar = $wrapper<S>> + Semiring<Scalar = $wrapper<S>>,
            $wrapper<S>: Scalar,
        {
            type Plan = TropicalPlan<$wrapper<S>>;
            type Context = CpuContext;

            fn plan(
                _ctx: &mut CpuContext,
                _desc: &SemiringFastPathDescriptor,
                _shapes: &[&[usize]],
            ) -> Result<TropicalPlan<$wrapper<S>>> {
                Err(Error::InvalidArgument(
                    "tropical algebras do not support semiring fast paths".into(),
                ))
            }

            fn execute(
                _ctx: &mut CpuContext,
                _plan: &TropicalPlan<$wrapper<S>>,
                _alpha: $wrapper<S>,
                _inputs: &[&Tensor<$wrapper<S>>],
                _beta: $wrapper<S>,
                _output: &mut Tensor<$wrapper<S>>,
            ) -> Result<()> {
                Err(Error::InvalidArgument(
                    "tropical algebras do not support semiring fast paths".into(),
                ))
            }

            fn has_fast_path(_desc: SemiringFastPathDescriptor) -> bool {
                false
            }
        }
    };
}

impl_tropical_prims!(MaxPlusAlgebra, MaxPlus);
impl_tropical_prims!(MinPlusAlgebra, MinPlus);
impl_tropical_prims!(MaxMulAlgebra, MaxMul);
