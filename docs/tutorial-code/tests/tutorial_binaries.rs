use std::process::Command;

const TUTORIAL_SKIP_MARKER: &str = "TENFERRO_TUTORIAL_SKIP:";

fn run_tutorial(name: &str, path: &str) {
    let output = Command::new(path)
        .output()
        .unwrap_or_else(|err| panic!("failed to run tutorial binary {name}: {err}"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(skip) = stdout
        .lines()
        .chain(stderr.lines())
        .find(|line| line.starts_with(TUTORIAL_SKIP_MARKER))
    {
        assert_eq!(
            name, "cuda_fft",
            "unexpected tutorial skip marker from {name}: {skip}"
        );
        assert!(
            output.status.success(),
            "tutorial binary {name} reported a skip but failed\nstatus: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        eprintln!("tutorial binary {name} skipped: {skip}");
        return;
    }
    assert!(
        output.status.success(),
        "tutorial binary {name} failed\nstatus: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
}

#[test]
fn tutorial_binaries_run_successfully() {
    run_tutorial(
        "typed_tensor_non_ad",
        env!("CARGO_BIN_EXE_typed_tensor_non_ad"),
    );
    run_tutorial(
        "storage_element_access",
        env!("CARGO_BIN_EXE_storage_element_access"),
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
    run_tutorial(
        "core_tensor_snippets",
        env!("CARGO_BIN_EXE_core_tensor_snippets"),
    );
    run_tutorial("math_snippets", env!("CARGO_BIN_EXE_math_snippets"));
    run_tutorial(
        "execution_snippets",
        env!("CARGO_BIN_EXE_execution_snippets"),
    );
    run_tutorial(
        "extension_snippets",
        env!("CARGO_BIN_EXE_extension_snippets"),
    );
    run_tutorial(
        "tenferro_compute_skill",
        env!("CARGO_BIN_EXE_tenferro_compute_skill"),
    );
    #[cfg(feature = "apple-shared")]
    {
        run_tutorial("apple_shared_fft", env!("CARGO_BIN_EXE_apple_shared_fft"));
        run_tutorial(
            "apple_shared_cholesky",
            env!("CARGO_BIN_EXE_apple_shared_cholesky"),
        );
    }
    #[cfg(feature = "cuda")]
    run_tutorial("cuda_fft", env!("CARGO_BIN_EXE_cuda_fft"));
}
