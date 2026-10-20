//! Typed-result tests for the `ConcreteEinsumPlan::execute*` engine surfaces.
//!
//! The typed surface must reject a backend-returned wrong dtype through
//! `into_typed_result`.

use tenferro_cpu::CpuEngine;
use tenferro_einsum::{ConcreteEinsumPlan, Error};
use tenferro_runtime::Tensor;
use tenferro_tensor::backend::{
    SessionCachedDot, TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, ElementwiseReadOp, Engine, GatherConfig, PadConfig,
    ScatterConfig, SliceConfig, TensorRead, TensorWrite,
};

type TensorResult = tenferro_tensor::Result<Tensor>;

/// Test backend whose `dot_general` returns an `F64` tensor regardless of the
/// requested dtype, so the typed surface must reject the output through
/// `into_typed_result`.
struct WrongDTypeSessionBackend;

macro_rules! panic_backend_methods {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                let _ = ($($arg),*);
                panic!(concat!(stringify!($name), " should not be called in this test"))
            }
        )+
    };
}

/// Panic-only implementations of every op trait except `TensorDot`,
/// `TensorAnalytic`, and `TensorReduction`, which test backends override with
/// real or wrong-dtype ops.
macro_rules! test_backend_impls {
    ($ty:ident) => {

        impl TensorStructural for $ty {
            fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult {
                { let __s = &mut CpuEngine::serial();  __s.to_contiguous_read(input) }
            }

            fn copy_read_into(
                &mut self,
                src: TensorRead<'_>,
                dst: TensorWrite<'_>,
            ) -> tenferro_tensor::Result<()> {
                { let __s = &mut CpuEngine::serial();  __s.copy_read_into(src, dst) }
            }

            panic_backend_methods! {
                cast(input: &Tensor, to: DType) -> TensorResult;
                convert(input: &Tensor, to: DType) -> TensorResult;
                extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult;
                embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult;
                tril(input: &Tensor, k: i64) -> TensorResult;
                triu(input: &Tensor, k: i64) -> TensorResult;
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly rather than
            // forwarding a view, which would widen the accepted input surface.
            fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> TensorResult {
                                let _ = tenferro_tensor::backend::read_owned_tensor("transpose", input)?;
                    let _ = perm;
                    panic!("transpose should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly rather than
            // forwarding a view, which would widen the accepted input surface.
            fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> TensorResult {
                                let _ = tenferro_tensor::backend::read_owned_tensor("reshape", input)?;
                    let _ = shape;
                    panic!("reshape should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly rather than
            // forwarding a view, which would widen the accepted input surface.
            fn broadcast_in_dim_read(&mut self, input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> TensorResult {
                                let _ = tenferro_tensor::backend::read_owned_tensor("broadcast_in_dim", input)?;
                    let _ = (shape, dims);
                    panic!("broadcast_in_dim should not be called in this test")
            }
        }

        impl TensorIndexing for $ty {
            panic_backend_methods! {
                gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> TensorResult;
                scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> TensorResult;
                slice(input: &Tensor, config: &SliceConfig) -> TensorResult;
                dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> TensorResult;
                dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> TensorResult;
                pad(input: &Tensor, config: &PadConfig) -> TensorResult;
                concatenate(inputs: &[&Tensor], axis: usize) -> TensorResult;
                reverse(input: &Tensor, axes: &[usize]) -> TensorResult;
            }
        }

        impl TensorFusion for $ty {}
        impl TensorBuffer for $ty {}

        impl TensorDeviceTransfer for $ty {
            fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
                Err(tenferro_tensor::Error::unsupported(
                    concat!(stringify!($ty), "::download_to_host"),
                    "test backend does not transfer tensors",
                ))
            }

            fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
                Err(tenferro_tensor::Error::unsupported(
                    concat!(stringify!($ty), "::upload_host_tensor"),
                    "test backend does not transfer tensors",
                ))
            }
        }

        impl SessionCachedDot for $ty {}

        impl Engine for $ty {

        }

    };
}

macro_rules! panic_elementwise {
    ($ty:ident) => {
        impl TensorElementwise for $ty {
            fn elementwise_read_into(
                &mut self,
                op: ElementwiseReadOp,
                inputs: &[TensorRead<'_>],
                out: TensorWrite<'_>,
            ) -> tenferro_tensor::Result<()> {
                let _ = (op, inputs, out);
                panic!("elementwise_read_into should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("add", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("add", rhs)?;
                panic!("add should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn sub_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("sub", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("sub", rhs)?;
                panic!("sub should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("mul", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("mul", rhs)?;
                panic!("mul should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn neg_read(&mut self, input: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("neg", input)?;
                panic!("neg should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn conj_read(&mut self, input: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("conj", input)?;
                panic!("conj should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("div", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("div", rhs)?;
                panic!("div should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn abs_read(&mut self, input: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("abs", input)?;
                panic!("abs should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn sign_read(&mut self, input: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("sign", input)?;
                panic!("sign should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("maximum", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("maximum", rhs)?;
                panic!("maximum should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("minimum", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("minimum", rhs)?;
                panic!("minimum should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn compare_read(
                &mut self,
                lhs: TensorRead<'_>,
                rhs: TensorRead<'_>,
                dir: &CompareDir,
            ) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("compare", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("compare", rhs)?;
                let _ = dir;
                panic!("compare should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn select_read(
                &mut self,
                pred: TensorRead<'_>,
                on_true: TensorRead<'_>,
                on_false: TensorRead<'_>,
            ) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("select", pred)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("select", on_true)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("select", on_false)?;
                panic!("select should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn clamp_read(
                &mut self,
                input: TensorRead<'_>,
                lower: TensorRead<'_>,
                upper: TensorRead<'_>,
            ) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("clamp", input)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("clamp", lower)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("clamp", upper)?;
                panic!("clamp should not be called in this test")
            }
        }
    };
}

macro_rules! panic_analytic {
    ($ty:ident) => {
        impl TensorAnalytic for $ty {
            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn exp_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("exp", input)?;
                panic!("exp should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn log_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("log", input)?;
                panic!("log should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn sin_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("sin", input)?;
                panic!("sin should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn cos_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("cos", input)?;
                panic!("cos should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn tanh_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("tanh", input)?;
                panic!("tanh should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn sqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("sqrt", input)?;
                panic!("sqrt should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn rsqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("rsqrt", input)?;
                panic!("rsqrt should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn pow_read(
                &mut self,
                lhs: tenferro_tensor::TensorRead<'_>,
                rhs: tenferro_tensor::TensorRead<'_>,
            ) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("pow", lhs)?;
                let _ = tenferro_tensor::backend::read_owned_tensor("pow", rhs)?;
                panic!("pow should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn expm1_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("expm1", input)?;
                panic!("expm1 should not be called in this test")
            }

            // Reproduce the previous read-half default: delegate an owned tensor and
            // reject a borrowed view.
            fn log1p_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("log1p", input)?;
                panic!("log1p should not be called in this test")
            }

            fn erf_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("erf", input)?;
                panic!("erf should not be called in this test")
            }
        }
    };
}

macro_rules! panic_reduction {
    ($ty:ident) => {
        impl TensorReduction for $ty {
            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly.
            fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("reduce_sum", input)?;
                let _ = axes;
                panic!("reduce_sum should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly.
            fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("reduce_prod", input)?;
                let _ = axes;
                panic!("reduce_prod should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly.
            fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("reduce_max", input)?;
                let _ = axes;
                panic!("reduce_max should not be called in this test")
            }

            // The previous read-half default delegated owned tensors to the one-shot
            // method and rejected borrowed views. Reproduce it explicitly.
            fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult {
                let _ = tenferro_tensor::backend::read_owned_tensor("reduce_min", input)?;
                let _ = axes;
                panic!("reduce_min should not be called in this test")
            }
        }
    };
}

test_backend_impls!(WrongDTypeSessionBackend);

impl TensorDot for WrongDTypeSessionBackend {
    // The previous read-half default delegated an owned pair to the one-shot
    // method and materialized borrowed views through to_contiguous_read before
    // contracting. Reproduce that exactly rather than forwarding a view.
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        _config: &DotGeneralConfig,
    ) -> TensorResult {
        const RESULT: [f64; 4] = [1.0; 4];
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(_), Some(_)) => Ok(Tensor::from_typed::<f64>(
                tenferro_tensor::TypedTensor::from_vec_col_major(vec![2, 2], RESULT.to_vec())
                    .unwrap(),
            )),
            _ => {
                let _ = self.to_contiguous_read(lhs)?;
                let _ = self.to_contiguous_read(rhs)?;
                Ok(Tensor::from_typed::<f64>(
                    tenferro_tensor::TypedTensor::from_vec_col_major(vec![2, 2], RESULT.to_vec())
                        .unwrap(),
                ))
            }
        }
    }
}

panic_elementwise!(WrongDTypeSessionBackend);
panic_analytic!(WrongDTypeSessionBackend);
panic_reduction!(WrongDTypeSessionBackend);

// ---------------------------------------------------------------------------
// Single-session-entry proof
// ---------------------------------------------------------------------------

#[test]
fn einsum_plan_typed_execute_rejects_wrong_backend_dtype() {
    let typed_lhs =
        tenferro_tensor::TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0_f32; 4])
            .unwrap();
    let typed_rhs =
        tenferro_tensor::TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0_f32; 4])
            .unwrap();
    let plan = ConcreteEinsumPlan::prepare_typed([&typed_lhs, &typed_rhs], "ij,jk->ik").unwrap();
    let mut backend = WrongDTypeSessionBackend;

    // The backend returns an F64 tensor regardless of the requested dtype, so
    // the typed session surface must reject it through into_typed_result.
    let in_session = {
        let session = &mut backend;
        plan.execute_typed([&typed_lhs, &typed_rhs], session)
    }
    .unwrap_err();
    assert!(matches!(
        in_session,
        Error::Tensor(tenferro_tensor::Error::Validation {
            op: "ConcreteEinsumPlan::execute",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
}
