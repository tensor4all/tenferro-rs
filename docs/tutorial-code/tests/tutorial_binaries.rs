use std::process::Command;

fn run_tutorial(name: &str, path: &str) {
    let output = Command::new(path)
        .output()
        .unwrap_or_else(|err| panic!("failed to run tutorial binary {name}: {err}"));
    assert!(
        output.status.success(),
        "tutorial binary {name} failed\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn tutorial_binaries_run_successfully() {
    run_tutorial(
        "typed_tensor_non_ad",
        env!("CARGO_BIN_EXE_typed_tensor_non_ad"),
    );
    run_tutorial(
        "direct_linalg_quickstart",
        env!("CARGO_BIN_EXE_direct_linalg_quickstart"),
    );
    run_tutorial(
        "eager_autodiff_pytorch_style",
        env!("CARGO_BIN_EXE_eager_autodiff_pytorch_style"),
    );
    run_tutorial(
        "traced_autodiff_jax_style",
        env!("CARGO_BIN_EXE_traced_autodiff_jax_style"),
    );
    run_tutorial(
        "xla_einsum_backend",
        env!("CARGO_BIN_EXE_xla_einsum_backend"),
    );
    run_tutorial(
        "xla_pjrt_execution",
        env!("CARGO_BIN_EXE_xla_pjrt_execution"),
    );
    run_tutorial(
        "complex_ad_convention",
        env!("CARGO_BIN_EXE_complex_ad_convention"),
    );
    run_tutorial(
        "einsum_subscripts_to_gradients",
        env!("CARGO_BIN_EXE_einsum_subscripts_to_gradients"),
    );
    run_tutorial(
        "dynamic_shape_truncated_svd",
        env!("CARGO_BIN_EXE_dynamic_shape_truncated_svd"),
    );
}
