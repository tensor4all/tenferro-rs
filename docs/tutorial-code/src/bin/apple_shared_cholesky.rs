//! Explicit CPU Cholesky over an Apple managed tensor without a download.

#[cfg(target_os = "macos")]
use tenferro_gpu::AppleContext;
#[cfg(target_os = "macos")]
use tenferro_linalg::LinalgBackend;
#[cfg(target_os = "macos")]
use tenferro_tensor::{Buffer, Tensor};

#[cfg(target_os = "macos")]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = AppleContext::new()?;

    // Column-major representation of the SPD matrix [[4, 2], [2, 3]].
    let host = Tensor::from_vec_col_major([2, 2], vec![4.0_f32, 2.0, 2.0, 3.0])?;
    let managed = context.upload_tensor(&host)?;
    let Tensor::F32(managed_typed) = &managed else {
        panic!("expected F32 input")
    };
    let input_domain = managed_typed.allocation_domain().expect("managed domain");
    let input_allocation = managed_typed.allocation_id().expect("managed allocation");
    let after_creation = context.transfer_stats();

    // Backend selection is explicit. Cholesky is the initial mapped CPU linalg
    // operation; this is not an automatic or general linalg fallback.
    let mut cpu = context.cpu_backend().clone();
    let factor = cpu.cholesky(&managed)?;

    assert_eq!(managed_typed.allocation_domain(), Some(input_domain));
    assert_eq!(managed_typed.allocation_id(), Some(input_allocation));
    let Tensor::F32(factor) = factor else {
        panic!("expected F32 factor")
    };
    assert_eq!(factor.allocation_domain(), Some(context.domain_id()));
    assert_ne!(factor.allocation_id(), Some(input_allocation));
    let Buffer::Backend(buffer) = factor.buffer() else {
        panic!("expected Apple managed output")
    };
    let l = buffer.map_read()?;
    let reconstructed = [
        l[0] * l[0] + l[2] * l[2],
        l[1] * l[0] + l[3] * l[2],
        l[0] * l[1] + l[2] * l[3],
        l[1] * l[1] + l[3] * l[3],
    ];
    let expected = [4.0_f32, 2.0, 2.0, 3.0];
    let residual = reconstructed
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(residual <= 1.0e-5, "reconstruction residual: {residual}");
    assert_eq!(context.transfer_stats(), after_creation);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "macos")]
    run()?;
    #[cfg(not(target_os = "macos"))]
    eprintln!("Apple shared Cholesky example is runtime-gated to macOS");
    Ok(())
}
