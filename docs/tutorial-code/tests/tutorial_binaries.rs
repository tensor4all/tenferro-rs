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
        "eager_autodiff_pytorch_style",
        env!("CARGO_BIN_EXE_eager_autodiff_pytorch_style"),
    );
    run_tutorial(
        "traced_autodiff_jax_style",
        env!("CARGO_BIN_EXE_traced_autodiff_jax_style"),
    );
}
