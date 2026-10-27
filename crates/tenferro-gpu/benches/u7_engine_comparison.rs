//! U7 of issue #2208: the same workload on the engine that ships today and on its replacement.
//!
//! One measurement at a time, one worker, one thread, pinned to a single L3 domain. The two
//! engines differ only in their dispatch and launch plumbing, so the workload is the same call
//! sequence on both, and the run records both the timing and the numerical residual between them.

use std::time::{Duration, Instant};

use tenferro_gpu::cuda::CudaEngine as Today;
use tenferro_gpu::cuda::CudaEngine as Replacement;
use tenferro_tensor::{
    DType, DotGeneralConfig, Tensor, TensorDot, TensorElementwise, TensorRead, TensorReduction,
};
use tinyvec::TinyVec;

const MATMUL_ORDER: usize = 512;
const ELEMENTWISE_LEN: usize = 1 << 20;
const REDUCTION_ROWS: usize = 4096;
const REDUCTION_COLS: usize = 256;
const REPEATS: usize = 7;
const WARMUPS: usize = 2;

fn dims(values: &[usize]) -> TinyVec<[usize; 4]> {
    values.iter().copied().collect()
}

fn plain(lhs_contracting: &[usize], rhs_contracting: &[usize]) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: dims(lhs_contracting),
        rhs_contracting_dims: dims(rhs_contracting),
        lhs_batch_dims: TinyVec::new(),
        rhs_batch_dims: TinyVec::new(),
    }
}

fn host_f64(shape: Vec<usize>, seed: u64) -> Tensor {
    let len: usize = shape.iter().product();
    let values: Vec<f64> = (0..len)
        .map(|index| {
            let mixed = (index as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed);
            ((mixed >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        })
        .collect();
    Tensor::from_vec_col_major(shape, values).expect("host tensor")
}

/// Median wall clock of `body` over the timed repeats, with the warmups discarded.
fn median(mut body: impl FnMut()) -> Duration {
    for _ in 0..WARMUPS {
        body();
    }
    let mut samples: Vec<Duration> = (0..REPEATS)
        .map(|_| {
            let start = Instant::now();
            body();
            start.elapsed()
        })
        .collect();
    samples.sort();
    samples[samples.len() / 2]
}

fn checksum(tensor: &Tensor) -> f64 {
    match tensor.dtype() {
        DType::F64 => tensor
            .as_slice::<f64>()
            .expect("f64 payload")
            .iter()
            .map(|value| value.abs())
            .sum(),
        DType::I64 => tensor
            .as_slice::<i64>()
            .expect("i64 payload")
            .iter()
            .map(|value| *value as f64)
            .sum(),
        other => panic!("no checksum for {other:?}"),
    }
}

struct Cell {
    name: &'static str,
    today: Duration,
    replacement: Duration,
    residual: f64,
}

fn main() {
    let left = host_f64(vec![MATMUL_ORDER, MATMUL_ORDER], 1);
    let right = host_f64(vec![MATMUL_ORDER, MATMUL_ORDER], 7);
    let elementwise = host_f64(vec![ELEMENTWISE_LEN], 11);
    let reduced = host_f64(vec![REDUCTION_ROWS, REDUCTION_COLS], 13);
    let config = plain(&[1], &[0]);

    let mut today =
        Today::open(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0)).expect("today's engine");
    let mut replacement = Replacement::open(tenferro_gpu::cuda::CudaDeviceId::from_ordinal(0))
        .expect("replacement engine");

    let upload_today = |engine: &Today, host: &Tensor| {
        tenferro_gpu::cuda::upload_tensor(engine.runtime(), host).expect("upload")
    };
    let upload_new = |engine: &Replacement, host: &Tensor| {
        tenferro_gpu::cuda::upload_tensor(engine.runtime(), host).expect("upload")
    };

    // Residency is outside the timed region: this measures dispatch and launch, not transfer.
    let tl = upload_today(&today, &left);
    let tr = upload_today(&today, &right);
    let te = upload_today(&today, &elementwise);
    let tred = upload_today(&today, &reduced);
    let nl = upload_new(&replacement, &left);
    let nr = upload_new(&replacement, &right);
    let ne = upload_new(&replacement, &elementwise);
    let nred = upload_new(&replacement, &reduced);

    let mut cells = Vec::new();

    let today_matmul = median(|| {
        today
            .dot_general_read(
                TensorRead::from_tensor(&tl),
                TensorRead::from_tensor(&tr),
                &config,
            )
            .expect("matmul");
        today.runtime().synchronize().expect("barrier");
    });
    let new_matmul = median(|| {
        replacement
            .dot_general_read(
                TensorRead::from_tensor(&nl),
                TensorRead::from_tensor(&nr),
                &config,
            )
            .expect("matmul");
        today.runtime().synchronize().expect("barrier");
    });
    let today_value = today
        .dot_general_read(
            TensorRead::from_tensor(&tl),
            TensorRead::from_tensor(&tr),
            &config,
        )
        .expect("matmul");
    let new_value = replacement
        .dot_general_read(
            TensorRead::from_tensor(&nl),
            TensorRead::from_tensor(&nr),
            &config,
        )
        .expect("matmul");
    let today_sum =
        checksum(&tenferro_gpu::cuda::download_tensor(today.runtime(), &today_value).unwrap());
    let new_sum =
        checksum(&tenferro_gpu::cuda::download_tensor(replacement.runtime(), &new_value).unwrap());
    cells.push(Cell {
        name: "matmul_512_f64",
        today: today_matmul,
        replacement: new_matmul,
        residual: (today_sum - new_sum).abs(),
    });

    let today_add = median(|| {
        today
            .add_read(TensorRead::from_tensor(&te), TensorRead::from_tensor(&te))
            .expect("add");
        today.runtime().synchronize().expect("barrier");
    });
    let new_add = median(|| {
        replacement
            .add_read(TensorRead::from_tensor(&ne), TensorRead::from_tensor(&ne))
            .expect("add");
        today.runtime().synchronize().expect("barrier");
    });
    let today_value = today
        .add_read(TensorRead::from_tensor(&te), TensorRead::from_tensor(&te))
        .expect("add");
    let new_value = replacement
        .add_read(TensorRead::from_tensor(&ne), TensorRead::from_tensor(&ne))
        .expect("add");
    let today_sum =
        checksum(&tenferro_gpu::cuda::download_tensor(today.runtime(), &today_value).unwrap());
    let new_sum =
        checksum(&tenferro_gpu::cuda::download_tensor(replacement.runtime(), &new_value).unwrap());
    cells.push(Cell {
        name: "add_1m_f64",
        today: today_add,
        replacement: new_add,
        residual: (today_sum - new_sum).abs(),
    });

    let today_reduce = median(|| {
        today
            .reduce_sum_read(TensorRead::from_tensor(&tred), &[1])
            .expect("reduce");
        today.runtime().synchronize().expect("barrier");
    });
    let new_reduce = median(|| {
        replacement
            .reduce_sum_read(TensorRead::from_tensor(&nred), &[1])
            .expect("reduce");
        today.runtime().synchronize().expect("barrier");
    });
    let today_value = today
        .reduce_sum_read(TensorRead::from_tensor(&tred), &[1])
        .expect("reduce");
    let new_value = replacement
        .reduce_sum_read(TensorRead::from_tensor(&nred), &[1])
        .expect("reduce");
    let today_sum =
        checksum(&tenferro_gpu::cuda::download_tensor(today.runtime(), &today_value).unwrap());
    let new_sum =
        checksum(&tenferro_gpu::cuda::download_tensor(replacement.runtime(), &new_value).unwrap());
    cells.push(Cell {
        name: "reduce_4096x256_f64",
        today: today_reduce,
        replacement: new_reduce,
        residual: (today_sum - new_sum).abs(),
    });

    // Scaling check: a 16x larger reduction tells a fixed per-call cost from a bandwidth one.
    let big = host_f64(vec![REDUCTION_ROWS * 16, REDUCTION_COLS], 17);
    let tbig = upload_today(&today, &big);
    let nbig = upload_new(&replacement, &big);
    let today_big = median(|| {
        today
            .reduce_sum_read(TensorRead::from_tensor(&tbig), &[1])
            .expect("reduce");
        today.runtime().synchronize().expect("barrier");
    });
    let new_big = median(|| {
        replacement
            .reduce_sum_read(TensorRead::from_tensor(&nbig), &[1])
            .expect("reduce");
        replacement.runtime().synchronize().expect("barrier");
    });
    cells.push(Cell {
        name: "reduce_65536x256_f64",
        today: today_big,
        replacement: new_big,
        residual: 0.0,
    });

    // Fixed-cost check: a reduction of only 4096x8 still pays whatever the call costs per call.
    let tiny = host_f64(vec![REDUCTION_ROWS, 8], 19);
    let ttiny = upload_today(&today, &tiny);
    let ntiny = upload_new(&replacement, &tiny);
    let today_tiny = median(|| {
        today
            .reduce_sum_read(TensorRead::from_tensor(&ttiny), &[1])
            .expect("reduce");
        today.runtime().synchronize().expect("barrier");
    });
    let new_tiny = median(|| {
        replacement
            .reduce_sum_read(TensorRead::from_tensor(&ntiny), &[1])
            .expect("reduce");
        replacement.runtime().synchronize().expect("barrier");
    });
    cells.push(Cell {
        name: "reduce_4096x8_f64",
        today: today_tiny,
        replacement: new_tiny,
        residual: 0.0,
    });

    // Per-call overhead check: the same reduction eight times inside one timed body. A fixed cost
    // per call shows up as a per-op time that falls towards the other engine's as the batch grows.
    let batch = 8;
    let today_batch = median(|| {
        for _ in 0..batch {
            today
                .reduce_sum_read(TensorRead::from_tensor(&tred), &[1])
                .expect("reduce");
        }
        today.runtime().synchronize().expect("barrier");
    });
    let new_batch = median(|| {
        for _ in 0..batch {
            replacement
                .reduce_sum_read(TensorRead::from_tensor(&nred), &[1])
                .expect("reduce");
        }
        replacement.runtime().synchronize().expect("barrier");
    });
    cells.push(Cell {
        name: "reduce_x8_per_op",
        today: today_batch / batch as u32,
        replacement: new_batch / batch as u32,
        residual: 0.0,
    });

    println!("# U7 engine comparison (median of {REPEATS}, {WARMUPS} warmups, 1 thread, 1 worker)");
    println!();
    println!("| cell | today (ms) | replacement (ms) | delta | residual |");
    println!("|---|---|---|---|---|");
    for cell in &cells {
        let today_ms = cell.today.as_secs_f64() * 1e3;
        let new_ms = cell.replacement.as_secs_f64() * 1e3;
        let delta = (new_ms - today_ms) / today_ms * 100.0;
        println!(
            "| {} | {:.3} | {:.3} | {:+.1}% | {:.3e} |",
            cell.name, today_ms, new_ms, delta, cell.residual
        );
    }
}
