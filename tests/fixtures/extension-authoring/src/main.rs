use std::alloc::{GlobalAlloc, Layout, System};
use std::any::Any;
use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::{
    apply, define_extension_runtime, ExtensionAliasDeclaration, ExtensionCacheStore,
    ExtensionEffectDeclaration, ExtensionExecutionContext, ExtensionOp, ExtensionShapeContext,
    SymDim,
};
use tenferro_runtime::{
    CoreCapabilityKind, DType, ErasedExecutionContext, Error, ExecutionContextIdentity,
    ExecutionContextMismatch, GraphCompiler, PrepareError, ProviderContractError, Runtime, Tensor,
    TracedTensor, UnsupportedReason,
};
use tenferro_tensor::{
    BackendStorageHandle, Placement, StorageBuffer, TensorBackend, TensorRead, TensorView,
    TypedTensor, TypedTensorView,
};

const IDENTITY_FAMILY: &str = "fixture.identity.v1";
const SECOND_FAMILY: &str = "fixture.second.v1";
const BORROWED_FAMILY: &str = "fixture.borrowed.v1";

struct CountingAllocator;

static COUNT_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static BORROWED_CALLBACKS: AtomicUsize = AtomicUsize::new(0);
static MATERIALIZATION_CALLBACKS: AtomicUsize = AtomicUsize::new(0);
static BORROWED_POINTERS: [AtomicUsize; 4] = [const { AtomicUsize::new(0) }; 4];

// SAFETY: this allocator forwards every operation to the standard allocator and
// only records allocation sizes for the single-threaded fixture measurements.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNT_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: the layout is forwarded unchanged to the standard allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from the corresponding allocator.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

macro_rules! identity_op {
    ($name:ident, $family:expr, $inputs:expr) => {
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
                $inputs
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

identity_op!(IdentityOp, IDENTITY_FAMILY, 1);
identity_op!(SecondOp, SECOND_FAMILY, 1);
identity_op!(BorrowedOp, BORROWED_FAMILY, 4);
// Reuse the registered family ID with a different payload type so dispatch
// reaches the family's engine and returns its typed WrongOperationFamily error.
identity_op!(WrongFamilyOp, IDENTITY_FAMILY, 1);

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
    MATERIALIZATION_CALLBACKS.fetch_add(1, Ordering::SeqCst);
    let materialized = context
        .backend_mut()
        .to_contiguous_read(inputs[0].clone())?;
    let _values = materialized.as_slice::<f64>()?;
    Ok(vec![materialized])
}

fn sum_four(slices: [&[f64]; 4], shape: &[usize]) -> tenferro_tensor::Result<Tensor> {
    let data = (0..slices[0].len())
        .map(|index| slices[0][index] + slices[1][index] + slices[2][index] + slices[3][index])
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data)
}

fn execute_borrowed<B: TensorBackend + 'static>(
    _op: &BorrowedOp,
    inputs: &[TensorRead<'_>],
    _context: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    BORROWED_CALLBACKS.fetch_add(1, Ordering::SeqCst);
    let slices = [
        inputs[0].as_slice::<f64>()?,
        inputs[1].as_slice::<f64>()?,
        inputs[2].as_slice::<f64>()?,
        inputs[3].as_slice::<f64>()?,
    ];
    for (slot, values) in slices.iter().enumerate() {
        BORROWED_POINTERS[slot].store(values.as_ptr() as usize, Ordering::SeqCst);
    }
    Ok(vec![sum_four(slices, inputs[0].shape())?])
}

fn execute_host_only<B: TensorBackend + 'static>(
    _op: &IdentityOp,
    inputs: &[TensorRead<'_>],
    _context: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let values = inputs[0].as_slice::<f64>()?;
    Ok(vec![Tensor::from_vec_col_major(
        inputs[0].shape().to_vec(),
        values.to_vec(),
    )?])
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

mod borrowed_runtime {
    use super::*;

    define_extension_runtime! {
        runtime = BorrowedRuntime,
        family_id = BORROWED_FAMILY,
        op_type = BorrowedOp,
        execute_reads = execute_borrowed,
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

fn compile_borrowed_op() -> tenferro_runtime::Result<tenferro_runtime::CompiledGraph> {
    let inputs = (0..4)
        .map(|_| TracedTensor::input_concrete_shape(DType::F64, &[1024]))
        .collect::<tenferro_runtime::Result<Vec<_>>>()?;
    let output = apply(
        Arc::new(BorrowedOp),
        &[&inputs[0], &inputs[1], &inputs[2], &inputs[3]],
    )?
    .remove(0);
    let specs = inputs
        .iter()
        .map(|input| (input, DType::F64, &[1024][..]))
        .collect::<Vec<_>>();
    GraphCompiler::new().compile_with_input_specs(&output, &specs)
}

fn measure_allocated_bytes(call: impl FnOnce()) -> usize {
    ALLOCATED_BYTES.store(0, Ordering::SeqCst);
    COUNT_ALLOCATIONS.store(true, Ordering::SeqCst);
    call();
    COUNT_ALLOCATIONS.store(false, Ordering::SeqCst);
    ALLOCATED_BYTES.load(Ordering::SeqCst)
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
    let mut backend = CpuBackend::new();
    let engine_id = tenferro_cpu::runtime_engine_id()?;
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    builder.install_extension_module(identity_runtime::extension_module::<CpuBackend>(
        engine_id.clone(),
    )?)?;
    builder.install_extension_module(second_runtime::extension_module::<CpuBackend>(
        engine_id.clone(),
    )?)?;
    builder
        .install_extension_module(borrowed_runtime::extension_module::<CpuBackend>(engine_id)?)?;
    let runtime = builder.build()?;

    let program = compile_op(Arc::new(IdentityOp))?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    let output = runtime.run_compiled(&program, &[&input])?.remove(0);
    assert_eq!(output.as_slice::<f64>()?, &[1.0, 2.0]);

    let borrowed_program = compile_borrowed_op()?;
    let borrowed_inputs = (0..4)
        .map(|input| {
            Tensor::from_vec_col_major(
                vec![1024],
                (0..1024)
                    .map(|index| (input + index) as f64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<tenferro_tensor::Result<Vec<_>>>()?;
    let borrowed_refs = borrowed_inputs.iter().collect::<Vec<_>>();
    runtime.run_compiled(&borrowed_program, &borrowed_refs)?;
    let caller_pointers = borrowed_inputs
        .iter()
        .map(|input| input.as_slice::<f64>().unwrap().as_ptr() as usize)
        .collect::<Vec<_>>();
    let mut borrowed_backend = CpuBackend::new();
    let mut borrowed_caches = ExtensionCacheStore::new();
    let mut borrowed_context =
        ExtensionExecutionContext::new(&mut borrowed_backend, &mut borrowed_caches);
    let borrowed_reads = borrowed_inputs
        .iter()
        .map(TensorRead::from_tensor)
        .collect::<Vec<_>>();
    execute_borrowed(&BorrowedOp, &borrowed_reads, &mut borrowed_context)?;
    let direct_slices = [
        borrowed_inputs[0].as_slice::<f64>()?,
        borrowed_inputs[1].as_slice::<f64>()?,
        borrowed_inputs[2].as_slice::<f64>()?,
        borrowed_inputs[3].as_slice::<f64>()?,
    ];
    let output_baseline_bytes = measure_allocated_bytes(|| {
        let output = sum_four(direct_slices, &[1024]).expect("baseline output should build");
        assert_eq!(output.as_slice::<f64>().unwrap()[0], 6.0);
    });
    let borrowed_bytes = measure_allocated_bytes(|| {
        let output = execute_borrowed(&BorrowedOp, &borrowed_reads, &mut borrowed_context)
            .expect("borrowed extension should execute");
        assert_eq!(output[0].as_slice::<f64>().unwrap()[0], 6.0);
    });
    assert_eq!(BORROWED_CALLBACKS.load(Ordering::SeqCst), 3);
    assert_eq!(MATERIALIZATION_CALLBACKS.load(Ordering::SeqCst), 0);
    for (slot, pointer) in caller_pointers.iter().copied().enumerate() {
        assert_eq!(BORROWED_POINTERS[slot].load(Ordering::SeqCst), pointer);
    }
    let input_bytes = 1024 * std::mem::size_of::<f64>();
    assert!(
        borrowed_bytes < output_baseline_bytes + input_bytes,
        "borrowed path allocated {borrowed_bytes} bytes versus {output_baseline_bytes} output-baseline bytes for {input_bytes}-byte inputs"
    );

    let mut materialize_backend = CpuBackend::new();
    let mut materialize_caches = ExtensionCacheStore::new();
    let noncompact_data = vec![1.0_f64; 2048];
    let noncompact_view = TensorView::F64(TypedTensorView::from_slice(
        [1024],
        [2],
        0,
        &noncompact_data,
    )?);
    let noncompact_read = TensorRead::from_view(noncompact_view);
    let mut materialize_context =
        ExtensionExecutionContext::new(&mut materialize_backend, &mut materialize_caches);
    MATERIALIZATION_CALLBACKS.store(0, Ordering::SeqCst);
    let materialized_bytes = measure_allocated_bytes(|| {
        execute_second(
            &SecondOp,
            std::slice::from_ref(&noncompact_read),
            &mut materialize_context,
        )
        .expect("explicit materialization should execute");
    });
    assert_eq!(MATERIALIZATION_CALLBACKS.load(Ordering::SeqCst), 1);
    assert!(
        materialized_bytes >= input_bytes,
        "explicit materialization allocated {materialized_bytes} bytes for {input_bytes}-byte input"
    );

    let backend_tensor = Tensor::F64(TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(BackendStorageHandle::<f64>::new_with_len(1709, 2))),
        Placement::default(),
    )?);
    let mut host_caches = ExtensionCacheStore::new();
    let mut host_context = ExtensionExecutionContext::new(&mut backend, &mut host_caches);
    let host_error = execute_host_only(
        &IdentityOp,
        &[TensorRead::from_tensor(&backend_tensor)],
        &mut host_context,
    )
    .expect_err("host-only typed access must reject backend-owned input");
    assert!(matches!(
        host_error,
        tenferro_tensor::Error::RuntimeState { .. }
    ));
    assert!(host_error.to_string().contains("download explicitly first"));

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
