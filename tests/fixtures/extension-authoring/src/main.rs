use std::any::Any;
use std::error::Error as StdError;
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::{
    apply, define_extension_runtime, ExtensionAliasDeclaration, ExtensionEffectDeclaration,
    ExtensionExecutionContext, ExtensionOp, ExtensionShapeContext, SymDim,
};
use tenferro_runtime::{
    CoreCapabilityKind, DType, ErasedExecutionContext, Error, ExecutionContextIdentity,
    ExecutionContextMismatch, GraphCompiler, PrepareError, ProviderContractError, Runtime, Tensor,
    TracedTensor, UnsupportedReason,
};
use tenferro_tensor::{TensorBackend, TensorRead};

const IDENTITY_FAMILY: &str = "fixture.identity.v1";
const SECOND_FAMILY: &str = "fixture.second.v1";

macro_rules! identity_op {
    ($name:ident, $family:expr) => {
        #[derive(Clone, Debug)]
        struct $name;

        impl ExtensionOp for $name {
            fn family_id(&self) -> &'static str {
                $family
            }

            fn payload_hash(&self, state: &mut dyn std::hash::Hasher) {
                state.write_u8(0);
            }

            fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
                other.as_any().is::<Self>()
            }

            fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
                Arc::new(self.clone())
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn input_count(&self) -> usize {
                1
            }

            fn output_count(&self) -> usize {
                1
            }

            fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
                ExtensionEffectDeclaration::Declared(&[])
            }

            fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
                ExtensionAliasDeclaration::AllFresh
            }

            fn infer_output_meta(
                &self,
                context: &mut ExtensionShapeContext<'_>,
            ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
                Ok(vec![(
                    context.input_dtype(0)?,
                    context.input_shape(0)?.to_vec(),
                )])
            }
        }
    };
}

identity_op!(IdentityOp, IDENTITY_FAMILY);
identity_op!(SecondOp, SECOND_FAMILY);
// Reuse the registered family ID with a different payload type so dispatch
// reaches the family's engine and returns its typed WrongOperationFamily error.
identity_op!(WrongFamilyOp, IDENTITY_FAMILY);

fn execute_identity<B: TensorBackend + 'static>(
    _op: &IdentityOp,
    inputs: &[TensorRead<'_>],
    context: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    Ok(vec![context
        .backend_mut()
        .to_contiguous_read(inputs[0].clone())?])
}

fn execute_second<B: TensorBackend + 'static>(
    _op: &SecondOp,
    inputs: &[TensorRead<'_>],
    context: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    Ok(vec![context
        .backend_mut()
        .to_contiguous_read(inputs[0].clone())?])
}

mod identity_runtime {
    use super::*;

    define_extension_runtime! {
        runtime = IdentityRuntime,
        family_id = IDENTITY_FAMILY,
        op_type = IdentityOp,
        execute_reads = execute_identity,
    }
}

mod second_runtime {
    use super::*;

    define_extension_runtime! {
        runtime = SecondRuntime,
        family_id = SECOND_FAMILY,
        op_type = SecondOp,
        execute_reads = execute_second,
    }
}

fn compile_op(
    op: Arc<dyn ExtensionOp>,
) -> tenferro_runtime::Result<tenferro_runtime::CompiledGraph> {
    let input = TracedTensor::input_concrete_shape(DType::F64, &[2])?;
    let output = apply(op, &[&input])?.remove(0);
    let program =
        GraphCompiler::new().compile_with_input_specs(&output, &[(&input, DType::F64, &[2])])?;
    Ok(program)
}

fn has_missing_family(error: &(dyn StdError + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(source) = current {
        if matches!(
            source.downcast_ref::<PrepareError>(),
            Some(PrepareError::Unsupported {
                reason: UnsupportedReason::Operation {
                    operation: IDENTITY_FAMILY,
                },
            })
        ) {
            return true;
        }
        current = source.source();
    }
    false
}

fn has_wrong_family(error: &(dyn StdError + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(source) = current {
        if matches!(
            source.downcast_ref::<PrepareError>(),
            Some(PrepareError::ProviderContract {
                source: ProviderContractError::WrongOperationFamily {
                    expected: CoreCapabilityKind::Elementwise,
                    operation: IDENTITY_FAMILY,
                },
            })
        ) {
            return true;
        }
        current = source.source();
    }
    false
}

fn main() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let engine_id = tenferro_cpu::runtime_engine_id()?;
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    builder.install_extension_module(identity_runtime::extension_module::<CpuBackend>(
        engine_id.clone(),
    )?)?;
    builder.install_extension_module(second_runtime::extension_module::<CpuBackend>(engine_id)?)?;
    let runtime = builder.build()?;

    let program = compile_op(Arc::new(IdentityOp))?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let output = runtime.run_compiled(&program, &[&input])?.remove(0);
    assert_eq!(output.as_slice::<f64>()?, &[1.0, 2.0]);

    let mut missing_builder = Runtime::builder();
    missing_builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    let missing = missing_builder
        .build()?
        .run_compiled(&program, &[&input])
        .expect_err("an unregistered extension family must fail");
    assert!(matches!(
        missing,
        Error::RuntimeStateSource { .. } | Error::Extension { .. }
    ));
    assert!(has_missing_family(&missing), "{missing:?}");

    let wrong_family_program = compile_op(Arc::new(WrongFamilyOp))?;
    let wrong_family = runtime
        .run_compiled(&wrong_family_program, &[&input])
        .expect_err("an engine receiving the wrong concrete family must fail");
    assert!(has_wrong_family(&wrong_family), "{wrong_family:?}");

    let mut erased_value = 7_u32;
    let mismatch = ErasedExecutionContext::new(&mut erased_value)
        .downcast_mut::<u64>(ExecutionContextIdentity::of::<u64>())
        .expect_err("a wrong execution context must stay typed");
    assert_eq!(
        mismatch,
        ExecutionContextMismatch {
            expected: ExecutionContextIdentity::of::<u64>(),
            actual: ExecutionContextIdentity::of::<u32>(),
        }
    );

    Ok(())
}
