use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tenferro_einsum::GraphCompilerEinsumExt;

use tenferro_runtime::{DType, DotGeneralConfig, GraphCompiler, TracedTensor};
use tenferro_xla::lower_to_stablehlo;

const RUN_HLO_MODULE_ENV: &str = "TENFERRO_XLA_RUN_HLO_MODULE";
const RUN_HLO_PLATFORM_ENV: &str = "TENFERRO_XLA_RUN_HLO_PLATFORM";

#[test]
fn generated_stablehlo_executes_with_xla_run_hlo_module_when_configured() {
    let Some(config) = XlaToolConfig::from_env() else {
        return;
    };
    let module = stablehlo_dot_add_reduce_module();

    run_hlo_module(&config, "tenferro-xla-dot-add-reduce", &module);
}

#[test]
fn nary_einsum_extension_stablehlo_executes_with_xla_run_hlo_module_when_configured() {
    let Some(config) = XlaToolConfig::from_env() else {
        return;
    };
    let module = stablehlo_nary_einsum_module();
    let text = module.as_str();
    assert_eq!(text.matches("stablehlo.dot_general").count(), 2);
    assert!(!text.contains("tenferro.einsum"));

    run_hlo_module(&config, "tenferro-xla-nary-einsum", &module);
}

#[test]
fn phase_one_elementwise_stablehlo_executes_with_xla_run_hlo_module_when_configured() {
    let Some(config) = XlaToolConfig::from_env() else {
        return;
    };
    let module = stablehlo_phase_one_elementwise_module();
    let text = module.as_str();
    for op in [
        "stablehlo.abs",
        "stablehlo.exponential",
        "stablehlo.log",
        "stablehlo.sine",
        "stablehlo.cosine",
        "stablehlo.tanh",
        "stablehlo.sqrt",
        "stablehlo.rsqrt",
        "stablehlo.exponential_minus_one",
        "stablehlo.log_plus_one",
        "stablehlo.divide",
        "stablehlo.power",
    ] {
        assert!(text.contains(op), "StableHLO did not contain {op}:\n{text}");
    }

    run_hlo_module(&config, "tenferro-xla-phase-one-elementwise", &module);
}

struct XlaToolConfig {
    run_hlo_module: PathBuf,
    platform: String,
}

impl XlaToolConfig {
    fn from_env() -> Option<Self> {
        let Some(run_hlo_module) = std::env::var_os(RUN_HLO_MODULE_ENV).map(PathBuf::from) else {
            eprintln!("skipping XLA execution check; set {RUN_HLO_MODULE_ENV} to run_hlo_module");
            return None;
        };
        let platform = std::env::var(RUN_HLO_PLATFORM_ENV).unwrap_or_else(|_| "Host".to_string());
        Some(Self {
            run_hlo_module,
            platform,
        })
    }
}

fn run_hlo_module(config: &XlaToolConfig, prefix: &str, module: &tenferro_xla::StableHloModule) {
    let module_path = write_temp_stablehlo(prefix, module.as_str());

    let output = Command::new(&config.run_hlo_module)
        .arg(&module_path)
        .arg("--input_format=stablehlo")
        .arg(format!("--platform={}", config.platform))
        .arg("--iterations=1")
        .output()
        .unwrap_or_else(|err| panic!("failed to run {:?}: {err}", config.run_hlo_module));

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "run_hlo_module failed with status {:?}\nstdout:\n{}\nstderr:\n{}\nStableHLO:\n{}",
        output.status,
        stdout,
        stderr,
        module.as_str()
    );
    assert!(
        stderr.contains("Results on") || stderr.contains("Skipping reference runner"),
        "run_hlo_module did not report execution success\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

fn stablehlo_dot_add_reduce_module() -> tenferro_xla::StableHloModule {
    let lhs = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let dot = lhs
        .dot_general(
            &rhs,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    let y = (&dot + &dot).unwrap().reduce_sum(&[0]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &y,
            &[(&lhs, DType::F32, &[2, 3]), (&rhs, DType::F32, &[3, 4])],
        )
        .unwrap();
    lower_to_stablehlo(&program).unwrap()
}

fn stablehlo_nary_einsum_module() -> tenferro_xla::StableHloModule {
    let lhs = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let mid = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let mut compiler = GraphCompiler::new();
    let product = compiler
        .einsum(&[&lhs, &mid, &rhs], "ij,jk,kl->il")
        .unwrap();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[
                (&lhs, DType::F32, &[2, 3]),
                (&mid, DType::F32, &[3, 4]),
                (&rhs, DType::F32, &[4, 2]),
            ],
        )
        .unwrap();
    lower_to_stablehlo(&program).unwrap()
}

fn stablehlo_phase_one_elementwise_module() -> tenferro_xla::StableHloModule {
    let x = TracedTensor::input_symbolic_shape(DType::F32, 1).unwrap();
    let positive = x.abs().unwrap().exp().unwrap();
    let analytic = positive
        .log()
        .unwrap()
        .sqrt()
        .unwrap()
        .rsqrt()
        .unwrap()
        .expm1()
        .unwrap()
        .log1p()
        .unwrap();
    let trig = positive.sin().unwrap().cos().unwrap().tanh().unwrap();
    let combined = (&analytic + &trig).unwrap();
    let divided = combined.div(&positive).unwrap();
    let powered = divided.abs().unwrap().pow(&positive).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&powered, &[(&x, DType::F32, &[4])])
        .unwrap();
    lower_to_stablehlo(&program).unwrap()
}

fn write_temp_stablehlo(prefix: &str, text: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time must be after UNIX_EPOCH")
        .as_nanos();
    path.push(format!("{prefix}-{}-{timestamp}.mlir", std::process::id()));
    std::fs::write(&path, text).unwrap_or_else(|err| {
        panic!(
            "failed to write StableHLO module to {}: {err}",
            display_path(&path)
        )
    });
    path
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}
