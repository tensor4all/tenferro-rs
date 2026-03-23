use super::*;
use tenferro_device::{Generator, LogicalMemorySpace};

const CPU: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

#[cfg(feature = "cuda")]
const GPU0: LogicalMemorySpace = LogicalMemorySpace::GpuMemory { device_id: 0 };

#[cfg(feature = "cuda")]
fn cuda_device_zero_is_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

fn host_f64(tensor: &Tensor<f64>) -> Tensor<f64> {
    tensor.to_memory_space_async(CPU).unwrap()
}

fn host_i32(tensor: &Tensor<i32>) -> Tensor<i32> {
    tensor.to_memory_space_async(CPU).unwrap()
}

#[test]
fn cpu_rand_and_randn_seeded_replay_match() {
    let mut lhs = Generator::cpu(1234);
    let mut rhs = Generator::cpu(1234);

    let lhs_rand =
        Tensor::<f64>::rand(&[64], CPU, MemoryOrder::ColumnMajor, Some(&mut lhs)).unwrap();
    let rhs_rand =
        Tensor::<f64>::rand(&[64], CPU, MemoryOrder::ColumnMajor, Some(&mut rhs)).unwrap();
    assert_eq!(
        host_f64(&lhs_rand).buffer().as_slice().unwrap(),
        host_f64(&rhs_rand).buffer().as_slice().unwrap()
    );

    let mut lhs = Generator::cpu(777);
    let mut rhs = Generator::cpu(777);
    let lhs_randn =
        Tensor::<f64>::randn(&[64], CPU, MemoryOrder::ColumnMajor, Some(&mut lhs)).unwrap();
    let rhs_randn =
        Tensor::<f64>::randn(&[64], CPU, MemoryOrder::ColumnMajor, Some(&mut rhs)).unwrap();
    assert_eq!(
        host_f64(&lhs_randn).buffer().as_slice().unwrap(),
        host_f64(&rhs_randn).buffer().as_slice().unwrap()
    );
}

#[test]
fn cpu_randint_and_like_constructors_preserve_shape_dtype_and_device() {
    let mut generator = Generator::cpu(99);
    let base = Tensor::<f64>::zeros(&[2, 3], CPU, MemoryOrder::RowMajor).unwrap();

    let randint = Tensor::<i32>::randint(
        -3,
        7,
        &[2, 3],
        CPU,
        MemoryOrder::ColumnMajor,
        Some(&mut generator),
    )
    .unwrap();
    assert_eq!(randint.dims(), &[2, 3]);
    assert_eq!(randint.logical_memory_space(), CPU);
    for &value in host_i32(&randint).buffer().as_slice().unwrap() {
        assert!(
            (-3..7).contains(&value),
            "randint sample {value} escaped range"
        );
    }

    let rand_like = Tensor::<f64>::rand_like(&base, Some(&mut generator)).unwrap();
    assert_eq!(rand_like.dims(), base.dims());
    assert_eq!(
        rand_like.logical_memory_space(),
        base.logical_memory_space()
    );
    assert_eq!(rand_like.strides(), base.strides());

    let randn_like = Tensor::<f64>::randn_like(&base, Some(&mut generator)).unwrap();
    assert_eq!(randn_like.dims(), base.dims());
    assert_eq!(
        randn_like.logical_memory_space(),
        base.logical_memory_space()
    );
    assert_eq!(randn_like.strides(), base.strides());

    let randint_like = Tensor::<i32>::randint_like(&randint, -3, 7, Some(&mut generator)).unwrap();
    assert_eq!(randint_like.dims(), randint.dims());
    assert_eq!(
        randint_like.logical_memory_space(),
        randint.logical_memory_space()
    );
}

#[test]
fn cpu_randn_has_basic_statistical_sanity() {
    let mut generator = Generator::cpu(31415);
    let samples =
        Tensor::<f64>::randn(&[8192], CPU, MemoryOrder::ColumnMajor, Some(&mut generator)).unwrap();
    let host = host_f64(&samples);
    let data = host.buffer().as_slice().unwrap();
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    let var = data
        .iter()
        .map(|&x| {
            let centered = x - mean;
            centered * centered
        })
        .sum::<f64>()
        / data.len() as f64;
    assert!(mean.abs() < 0.1, "randn mean drifted to {mean}");
    assert!((var - 1.0).abs() < 0.2, "randn variance drifted to {var}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rand_and_randn_seeded_replay_match() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let mut lhs = Generator::cuda(0, 1234).unwrap();
    let mut rhs = Generator::cuda(0, 1234).unwrap();
    let lhs_rand =
        Tensor::<f64>::rand(&[64], GPU0, MemoryOrder::ColumnMajor, Some(&mut lhs)).unwrap();
    let rhs_rand =
        Tensor::<f64>::rand(&[64], GPU0, MemoryOrder::ColumnMajor, Some(&mut rhs)).unwrap();
    assert_eq!(
        host_f64(&lhs_rand).buffer().as_slice().unwrap(),
        host_f64(&rhs_rand).buffer().as_slice().unwrap()
    );

    let mut lhs = Generator::cuda(0, 777).unwrap();
    let mut rhs = Generator::cuda(0, 777).unwrap();
    let lhs_randn =
        Tensor::<f64>::randn(&[64], GPU0, MemoryOrder::ColumnMajor, Some(&mut lhs)).unwrap();
    let rhs_randn =
        Tensor::<f64>::randn(&[64], GPU0, MemoryOrder::ColumnMajor, Some(&mut rhs)).unwrap();
    assert_eq!(
        host_f64(&lhs_randn).buffer().as_slice().unwrap(),
        host_f64(&rhs_randn).buffer().as_slice().unwrap()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_rng_constructors_require_cuda_generators_for_gpu_memory() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let gpu_base_f64 = Tensor::<f64>::zeros(&[2, 3], GPU0, MemoryOrder::ColumnMajor).unwrap();
    let gpu_base_i32 = Tensor::<i32>::zeros(&[2, 3], GPU0, MemoryOrder::ColumnMajor).unwrap();

    let mut cpu_uniform = Generator::cpu(111);
    assert!(Tensor::<f64>::rand(
        &[2, 3],
        GPU0,
        MemoryOrder::ColumnMajor,
        Some(&mut cpu_uniform)
    )
    .is_err());

    let mut cpu_normal = Generator::cpu(222);
    assert!(Tensor::<f64>::randn(
        &[2, 3],
        GPU0,
        MemoryOrder::ColumnMajor,
        Some(&mut cpu_normal)
    )
    .is_err());

    let mut cpu_integer = Generator::cpu(333);
    assert!(Tensor::<i32>::randint(
        -4,
        5,
        &[2, 3],
        GPU0,
        MemoryOrder::ColumnMajor,
        Some(&mut cpu_integer)
    )
    .is_err());

    let mut cpu_like_uniform = Generator::cpu(444);
    assert!(Tensor::<f64>::rand_like(&gpu_base_f64, Some(&mut cpu_like_uniform)).is_err());

    let mut cpu_like_normal = Generator::cpu(555);
    assert!(Tensor::<f64>::randn_like(&gpu_base_f64, Some(&mut cpu_like_normal)).is_err());

    let mut cpu_like_integer = Generator::cpu(666);
    assert!(
        Tensor::<i32>::randint_like(&gpu_base_i32, -4, 5, Some(&mut cpu_like_integer)).is_err()
    );
}
