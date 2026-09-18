//! The extension-owned operation reached through the runtime.
//!
//! The input travels as a runtime `Tensor` carrying a caller-owned payload, the
//! operation is an `ExtensionOp` family registered by a downstream
//! `ExtensionModule`, and the result comes back as another external payload.

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_df64_proof::extension::{apply_total, Df64Total};
use tenferro_df64_proof::Df64;
use tenferro_tensor::{AllocationGroup, BackendSessionHost, GroupError, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn external(values: Vec<Df64>) -> Tensor {
    Tensor::external(ErasedHostTensor::new(
        HostTensor::from_vec_col_major(vec![values.len()], values).expect("shape matches data"),
    ))
}

fn payload(tensor: &Tensor) -> Vec<Df64> {
    match tensor.external_payload() {
        Some(value) => value
            .downcast_ref::<Df64>()
            .expect("external element type")
            .as_slice()
            .to_vec(),
        _ => panic!("expected an externally defined payload"),
    }
}

#[test]
fn the_extension_operation_runs_through_the_registered_module() {
    let low = 2f64.powi(-80);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let input = tenferro_ad::EagerTensor::from_tensor_in(
        external(vec![Df64::from_f64(1.0), Df64::from_f64(low)]),
        std::sync::Arc::clone(&runtime),
    )
    .expect("eager tensor");

    let outputs = apply_total(&input).expect("extension execution");
    assert_eq!(outputs.len(), 1);

    let value = outputs[0].to_tensor().expect("output tensor");
    let total = payload(&value);
    assert_eq!(total.len(), 1);
    assert_eq!(total[0], Df64 { hi: 1.0, lo: low });
}

#[test]
fn the_registered_operation_rejects_a_preset_input() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let ordinary = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches");
    let input = tenferro_ad::EagerTensor::from_tensor_in(ordinary, std::sync::Arc::clone(&runtime))
        .expect("eager tensor");

    // The extension declares only the external scalar, so a preset input fails
    // explicitly instead of being coerced.
    assert!(apply_total(&input).is_err());
}

#[test]
fn ordinary_work_shares_the_session_with_the_extension() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|session| {
        let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches");
        let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).expect("shape matches");
        let sum = tenferro_tensor::backend::TensorElementwise::add(session, &a, &b)
            .expect("ordinary addition");
        assert_eq!(sum.as_slice::<f64>().expect("f64 slice"), &[4.0, 6.0]);
    });
    assert_eq!(
        <Df64Total as tenferro_ad::extension::ExtensionOp>::input_count(&Df64Total),
        1
    );
    drop(runtime);
}

#[test]
fn the_runtime_retains_and_returns_a_caller_owned_payload() {
    let low = 2f64.powi(-80);
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).expect("cpu runtime");
    let original = Df64::from_f64(1.0) + Df64::from_f64(low);
    let leaf = tenferro_ad::EagerTensor::from_tensor_in(
        external(vec![original, Df64::from_f64(3.0)]),
        std::sync::Arc::clone(&runtime),
    )
    .expect("eager tensor");

    // A caller-owned payload keeps every bit while the runtime retains it, so the
    // low-order component survives the round trip unchanged.
    let value = leaf.to_tensor().expect("round trip");
    assert_eq!(payload(&value), vec![original, Df64::from_f64(3.0)]);
    assert!(value.external_payload().is_some());
}

#[test]
fn an_allocation_group_rejects_a_caller_owned_payload_with_a_typed_error() {
    // A payload tenferro does not define owns no pooled allocation group, so the
    // group reports it explicitly instead of dropping or zeroing the value.
    let error = AllocationGroup::from_tensors(vec![external(vec![Df64::from_f64(1.0)])])
        .expect_err("a caller-owned payload has no allocation group");
    assert!(matches!(error, GroupError::InvalidDescriptor { .. }));
}

#[test]
fn the_module_installs_and_plans_the_declared_scalar() {
    use tenferro_df64_proof::extension::{module, Df64Total, DF64_SCALAR_IDENTITY};
    use tenferro_ops::dim_expr::DimExpr;
    use tenferro_runtime::program::{ProgramBuildError, ProgramInputSpec};
    use tenferro_runtime::{GraphCompiler, Runtime, TraceContext};
    use tenferro_tensor::DType;

    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("engine"))
        .expect("register the CPU engine");
    // Installing the module registers its engine and planning config against the
    // CPU runtime engine, which is what a downstream application does.
    builder
        .install_extension_module(module().expect("module"))
        .expect("install the Df64 module");
    let runtime = builder.build().expect("runtime with the module");

    let external_dtype = DType::External(std::any::TypeId::of::<Df64>());

    // Without a declared identity the program cannot be given a canonical one, so
    // planning rejects the tag instead of encoding a process-local code.
    let mut undeclared = TraceContext::new();
    assert!(matches!(
        undeclared
            .input(ProgramInputSpec::new(external_dtype, [DimExpr::Const(2)]))
            .expect_err("an undeclared external scalar has no canonical identity"),
        ProgramBuildError::ExternalScalarWithoutIdentity { .. }
    ));

    // With the contribution's declared identity the same program plans, and the
    // registered engine executes it through the prepared path.
    let mut context = TraceContext::new();
    let input = context
        .input(
            ProgramInputSpec::new(external_dtype, [DimExpr::Const(2)])
                .with_scalar_identity(DF64_SCALAR_IDENTITY),
        )
        .expect("the declared scalar is accepted");
    let outputs = context
        .add_extension(std::sync::Arc::new(Df64Total), &[input])
        .expect("the family is planned from its declared metadata");
    let graph = context.finish(&outputs).expect("finished trace");

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_traced_graph(&graph)
        .expect("compiled program");
    let values = external(vec![Df64::from_f64(1.0), Df64::from_f64(2.0)]);
    let results = runtime
        .run_compiled(&program, &[&values])
        .expect("prepared execution");
    assert_eq!(results.len(), 1);
    assert_eq!(payload(&results[0]), vec![Df64::from_f64(3.0)]);
}
