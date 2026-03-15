use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_algebra::Conjugate;
use tenferro_capi::{tfe_status_t, TfeTensorF64};
use tenferro_einsum::{einsum, EinsumBackend};
use tenferro_prims::{CpuBackend, CpuContext, TensorSemiringCore, TensorSemiringFastPath};
use tenferro_tensor::Tensor;
use tenferro_tropical::ad::{
    extract_inner, promote_to_tropical, tropical_einsum_frule, tropical_einsum_rrule,
    TropicalScalar,
};
use tenferro_tropical::{MaxMul, MaxMulAlgebra, MaxPlus, MaxPlusAlgebra, MinPlus, MinPlusAlgebra};

use crate::ffi_utils::{
    collect_operand_handles, collect_optional_tangent_handles, cpu_context, parse_subscripts,
};
use crate::handle::{handle_to_ref, tensor_to_handle};
use crate::status::{finalize_ptr, finalize_void, map_device_error};

unsafe fn tropical_einsum_impl<T, Alg>(
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
) -> std::result::Result<*mut TfeTensorF64, tfe_status_t>
where
    T: TropicalScalar<Inner = f64> + Conjugate + tenferro_algebra::HasAlgebra<Algebra = Alg>,
    Alg: tenferro_algebra::Semiring<Scalar = T>,
    CpuBackend: EinsumBackend<Alg>
        + TensorSemiringCore<Alg, Context = CpuContext>
        + TensorSemiringFastPath<
            Alg,
            Context = CpuContext,
            Plan = <CpuBackend as TensorSemiringCore<Alg>>::Plan,
        >,
{
    let subscripts = parse_subscripts(subscripts)?;
    let operands = collect_operand_handles(operands, num_operands)?;
    let tropical_operands: Vec<Tensor<T>> = operands
        .iter()
        .map(|tensor| promote_to_tropical::<T>(*tensor).map_err(|err| map_device_error(&err)))
        .collect::<std::result::Result<_, _>>()?;
    let tropical_refs: Vec<&Tensor<T>> = tropical_operands.iter().collect();
    let mut ctx = cpu_context()?;
    let output = einsum::<Alg, CpuBackend>(&mut ctx, subscripts, &tropical_refs, None)
        .map_err(|err| map_device_error(&err))?;
    let output = extract_inner::<T>(&output).map_err(|err| map_device_error(&err))?;
    Ok(tensor_to_handle(output))
}

unsafe fn tropical_einsum_rrule_impl<T, Alg>(
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    cotangent: *const TfeTensorF64,
    grads_out: *mut *mut TfeTensorF64,
) -> std::result::Result<(), tfe_status_t>
where
    T: TropicalScalar<Inner = f64> + Conjugate + tenferro_algebra::HasAlgebra<Algebra = Alg>,
    Alg: tenferro_algebra::Semiring<Scalar = T>,
    CpuBackend: EinsumBackend<Alg>
        + TensorSemiringCore<Alg, Context = CpuContext>
        + TensorSemiringFastPath<
            Alg,
            Context = CpuContext,
            Plan = <CpuBackend as TensorSemiringCore<Alg>>::Plan,
        >,
{
    if cotangent.is_null() || grads_out.is_null() {
        return Err(tenferro_capi::TFE_INVALID_ARGUMENT);
    }

    let subscripts = parse_subscripts(subscripts)?;
    let operands = collect_operand_handles(operands, num_operands)?;
    let tropical_operands: Vec<Tensor<T>> = operands
        .iter()
        .map(|tensor| promote_to_tropical::<T>(*tensor).map_err(|err| map_device_error(&err)))
        .collect::<std::result::Result<_, _>>()?;
    let tropical_refs: Vec<&Tensor<T>> = tropical_operands.iter().collect();
    let cotangent = handle_to_ref(cotangent);
    let mut ctx = cpu_context()?;
    let grads = tropical_einsum_rrule::<T, Alg, CpuBackend>(
        &mut ctx,
        subscripts,
        &tropical_refs,
        cotangent,
    )
    .map_err(|err| map_device_error(&err))?;

    if grads.len() != num_operands {
        return Err(tenferro_capi::TFE_INTERNAL_ERROR);
    }

    let out = std::slice::from_raw_parts_mut(grads_out, num_operands);
    for (slot, grad) in out.iter_mut().zip(grads.into_iter()) {
        *slot = tensor_to_handle(grad);
    }
    Ok(())
}

unsafe fn tropical_einsum_frule_impl<T, Alg>(
    subscripts: *const c_char,
    primals: *const *const TfeTensorF64,
    num_operands: usize,
    tangents: *const *const TfeTensorF64,
) -> std::result::Result<*mut TfeTensorF64, tfe_status_t>
where
    T: TropicalScalar<Inner = f64> + Conjugate + tenferro_algebra::HasAlgebra<Algebra = Alg>,
    Alg: tenferro_algebra::Semiring<Scalar = T>,
    CpuBackend: EinsumBackend<Alg>
        + TensorSemiringCore<Alg, Context = CpuContext>
        + TensorSemiringFastPath<
            Alg,
            Context = CpuContext,
            Plan = <CpuBackend as TensorSemiringCore<Alg>>::Plan,
        >,
{
    let subscripts = parse_subscripts(subscripts)?;
    let primals = collect_operand_handles(primals, num_operands)?;
    let tangents = collect_optional_tangent_handles(tangents, num_operands)?;
    let tropical_primals: Vec<Tensor<T>> = primals
        .iter()
        .map(|tensor| promote_to_tropical::<T>(*tensor).map_err(|err| map_device_error(&err)))
        .collect::<std::result::Result<_, _>>()?;
    let tropical_refs: Vec<&Tensor<T>> = tropical_primals.iter().collect();
    let mut ctx = cpu_context()?;
    let output = tropical_einsum_frule::<T, Alg, CpuBackend>(
        &mut ctx,
        subscripts,
        &tropical_refs,
        &tangents,
    )
    .map_err(|err| map_device_error(&err))?;
    Ok(tensor_to_handle(output))
}

macro_rules! define_tropical_entrypoints {
    (
        algebra = $algebra_name:literal,
        combine = $combine:literal,
        multiply = $multiply:literal,
        tropical_ty = $tropical_ty:ty,
        algebra_ty = $algebra_ty:ty,
        einsum_fn = $einsum_fn:ident,
        rrule_fn = $rrule_fn:ident,
        frule_fn = $frule_fn:ident
    ) => {
        #[doc = concat!(
            "Execute tropical einsum under ", $algebra_name, " algebra (⊕=", $combine, ", ⊗=", $multiply, ").\n\n",
            "Accepts standard `TfeTensorF64` handles and interprets them as tropical scalars internally.\n",
            "Returns a new tensor that the caller must release with `tfe_tensor_f64_release`.\n\n",
            "# Safety\n\n",
            "- `subscripts` must be a valid null-terminated C string.\n",
            "- `operands` must point to an array of `num_operands` valid tensor pointers.\n",
            "- `status` must be a valid, non-null pointer.\n\n",
            "# Examples (C)\n\n",
            "```c\n",
            "const tfe_tensor_f64 *ops[] = {a, b};\n",
            "tfe_status_t status;\n",
            "tfe_tensor_f64 *c = ", stringify!($einsum_fn), "(\"ij,jk->ik\", ops, 2, &status);\n",
            "tfe_tensor_f64_release(c);\n",
            "```"
        )]
        #[no_mangle]
        pub unsafe extern "C" fn $einsum_fn(
            subscripts: *const c_char,
            operands: *const *const TfeTensorF64,
            num_operands: usize,
            status: *mut tfe_status_t,
        ) -> *mut TfeTensorF64 {
            let result = catch_unwind(AssertUnwindSafe(
                || -> std::result::Result<*mut TfeTensorF64, tfe_status_t> {
                    tropical_einsum_impl::<$tropical_ty, $algebra_ty>(
                        subscripts,
                        operands,
                        num_operands,
                    )
                },
            ));
            finalize_ptr(result, status)
        }

        #[doc = concat!(
            "Reverse-mode rule (VJP) for ", $algebra_name, " tropical einsum.\n\n",
            "Computes one gradient tensor per input operand given the output cotangent.\n\n",
            "# Safety\n\n",
            "- `subscripts` must be a valid null-terminated C string.\n",
            "- `operands` must point to an array of `num_operands` valid tensor pointers.\n",
            "- `cotangent` must be a valid, non-null tensor pointer.\n",
            "- `grads_out` must point to a caller-allocated array of `num_operands` mutable output pointers.\n",
            "- `status` must be a valid, non-null pointer.\n\n",
            "# Examples (C)\n\n",
            "```c\n",
            "tfe_tensor_f64 *grads[2];\n",
            "tfe_status_t status;\n",
            "const tfe_tensor_f64 *ops[] = {a, b};\n",
            stringify!($rrule_fn), "(\"ij,jk->ik\", ops, 2, grad_c, grads, &status);\n",
            "tfe_tensor_f64_release(grads[0]);\n",
            "tfe_tensor_f64_release(grads[1]);\n",
            "```"
        )]
        #[no_mangle]
        pub unsafe extern "C" fn $rrule_fn(
            subscripts: *const c_char,
            operands: *const *const TfeTensorF64,
            num_operands: usize,
            cotangent: *const TfeTensorF64,
            grads_out: *mut *mut TfeTensorF64,
            status: *mut tfe_status_t,
        ) {
            let result = catch_unwind(AssertUnwindSafe(
                || -> std::result::Result<(), tfe_status_t> {
                    tropical_einsum_rrule_impl::<$tropical_ty, $algebra_ty>(
                        subscripts,
                        operands,
                        num_operands,
                        cotangent,
                        grads_out,
                    )
                },
            ));
            finalize_void(result, status)
        }

        #[doc = concat!(
            "Forward-mode rule (JVP) for ", $algebra_name, " tropical einsum.\n\n",
            "Returns the output tangent. Elements of `tangents` may be null to denote zero tangents.\n\n",
            "# Safety\n\n",
            "- `subscripts` must be a valid null-terminated C string.\n",
            "- `primals` must point to an array of `num_operands` valid tensor pointers.\n",
            "- `tangents` must point to an array of `num_operands` tensor pointers (elements may be null).\n",
            "- `status` must be a valid, non-null pointer.\n\n",
            "# Examples (C)\n\n",
            "```c\n",
            "const tfe_tensor_f64 *primals[] = {a, b};\n",
            "const tfe_tensor_f64 *tangents[] = {da, NULL};\n",
            "tfe_status_t status;\n",
            "tfe_tensor_f64 *dc = ", stringify!($frule_fn), "(\"ij,jk->ik\", primals, 2, tangents, &status);\n",
            "tfe_tensor_f64_release(dc);\n",
            "```"
        )]
        #[no_mangle]
        pub unsafe extern "C" fn $frule_fn(
            subscripts: *const c_char,
            primals: *const *const TfeTensorF64,
            num_operands: usize,
            tangents: *const *const TfeTensorF64,
            status: *mut tfe_status_t,
        ) -> *mut TfeTensorF64 {
            let result = catch_unwind(AssertUnwindSafe(
                || -> std::result::Result<*mut TfeTensorF64, tfe_status_t> {
                    tropical_einsum_frule_impl::<$tropical_ty, $algebra_ty>(
                        subscripts,
                        primals,
                        num_operands,
                        tangents,
                    )
                },
            ));
            finalize_ptr(result, status)
        }
    };
}

define_tropical_entrypoints!(
    algebra = "MaxPlus",
    combine = "max",
    multiply = "+",
    tropical_ty = MaxPlus<f64>,
    algebra_ty = MaxPlusAlgebra<f64>,
    einsum_fn = tfe_tropical_einsum_maxplus_f64,
    rrule_fn = tfe_tropical_einsum_rrule_maxplus_f64,
    frule_fn = tfe_tropical_einsum_frule_maxplus_f64
);

define_tropical_entrypoints!(
    algebra = "MinPlus",
    combine = "min",
    multiply = "+",
    tropical_ty = MinPlus<f64>,
    algebra_ty = MinPlusAlgebra<f64>,
    einsum_fn = tfe_tropical_einsum_minplus_f64,
    rrule_fn = tfe_tropical_einsum_rrule_minplus_f64,
    frule_fn = tfe_tropical_einsum_frule_minplus_f64
);

define_tropical_entrypoints!(
    algebra = "MaxMul",
    combine = "max",
    multiply = "×",
    tropical_ty = MaxMul<f64>,
    algebra_ty = MaxMulAlgebra<f64>,
    einsum_fn = tfe_tropical_einsum_maxmul_f64,
    rrule_fn = tfe_tropical_einsum_rrule_maxmul_f64,
    frule_fn = tfe_tropical_einsum_frule_maxmul_f64
);
