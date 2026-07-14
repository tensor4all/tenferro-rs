use tenferro_runtime::{GraphCompiler, TracedTensor};
use tenferro_tensor::Tensor;
use tenferro_xla::{
    Error, StableHloModule, StableHloModuleFingerprint, XlaExecutor, XlaExecutorOptions,
};

#[test]
fn stablehlo_module_fingerprint_is_deterministic_and_hex_encoded() {
    let text = "module {\n  func.func @main() { return }\n}\n";
    let module = StableHloModule::new(text.to_string());

    assert_eq!(module.as_str(), text);
    assert_eq!(module.clone(), module);
    assert!(format!("{module:?}").contains("StableHloModule"));

    let fingerprint = module.fingerprint();
    assert_eq!(fingerprint, StableHloModuleFingerprint::from_text(text));
    assert_ne!(
        fingerprint,
        StableHloModuleFingerprint::from_text("module {}")
    );
    assert_eq!(fingerprint.as_bytes().len(), 32);

    let hex = fingerprint.to_hex();
    assert_eq!(hex.len(), 64);
    assert!(hex
        .chars()
        .all(|ch| ch.is_ascii_digit() || ('a'..='f').contains(&ch)));
}

#[test]
fn xla_executor_options_debug_and_lowering_are_stable() {
    let options = XlaExecutorOptions::default();
    let executor = XlaExecutor::new(options);

    assert_eq!(format!("{options:?}"), "XlaExecutorOptions");
    assert_eq!(executor.options(), options);
    assert!(!executor.has_loaded_pjrt_plugin());
    let debug = format!("{executor:?}");
    assert!(debug.contains("XlaExecutor"));
    assert!(debug.contains("has_loaded_pjrt_plugin"));
    assert_eq!(XlaExecutor::default().options(), options);

    let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&x.neg().unwrap()).unwrap();
    let module = executor.lower_to_stablehlo(&program).unwrap();
    assert!(module.as_str().contains("stablehlo.negate"));
}

#[test]
fn xla_executor_run_with_inputs_requires_pjrt_plugin_boundary() {
    let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&x.neg().unwrap()).unwrap();
    let input = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();

    let err = XlaExecutor::default()
        .run_with_inputs(&program, &[&input])
        .unwrap_err();

    assert!(matches!(
        err,
        Error::PjrtFeatureDisabled | Error::PjrtPluginNotLoaded
    ));
}
