//! Which driver a large c64 product takes on this host, per thread budget.
//!
//! tcontract's `FaerLimit` sends c64 above `m*n*k = 1 << 17` to its packed
//! driver, and `tenferro-cpu` lifts that bound only for `x86_64-avx2` at a
//! thread budget of one (tcontract-rs#156, tensor4all-rs#2415). This example
//! prints the plan the engine actually prepared, so a host can be checked
//! without reading the code:
//!
//! ```text
//! cargo run --release -p tenferro-cpu --example route_faerlimit \
//!     --features route-report
//! ```
//!
//! One line per budget for `64x64 * 64x64` c64 (volume 262144, above the
//! bound):
//!
//! * `x86_64-avx2`: `budget=1 route=faer reason=Fused`, then
//!   `budget=4 route=packed reason=AboveFaerLimit { .. }`.
//! * `aarch64`, `x86_64-avx512`, `x86_64-scalar` and `generic`: packed at both
//!   budgets, `reason=AboveFaerLimit { volume: 262144, limit: 131072 }`.

use tenferro_cpu::{take_last_plan_route, CpuEngine, Pool};
use tenferro_tensor::{Complex64, DotGeneralConfig, Tensor, TensorDot, TensorRead};

fn config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn data(len: usize, seed: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            let x = (i as u64 * 2654435761 + seed as u64 * 40503) % 1009;
            let y = (x * 7 + i as u64) % 211;
            Complex64::new(x as f64 / 503.0 - 1.0, y as f64 / 105.0 - 1.0)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (m, k, n) = (64, 64, 64);
    let a = Tensor::from_vec_col_major(vec![m, k], data(m * k, 1))?;
    let b = Tensor::from_vec_col_major(vec![k, n], data(k * n, 2))?;
    let pool = Pool::builder().threads(4).build()?;
    println!(
        "host={} volume={}",
        threadteam::HostClass::detect().name(),
        m * k * n
    );
    // A fresh engine has a private, empty plan cache, so each budget prepares
    // its own plan and reports it once.
    for mut engine in [CpuEngine::serial(), CpuEngine::new(pool)] {
        let budget = engine.num_threads();
        engine.dot_general_read(
            TensorRead::from_tensor(&a),
            TensorRead::from_tensor(&b),
            &config(),
        )?;
        match take_last_plan_route() {
            Some((algorithm, reason)) => {
                println!(
                    "budget={budget} route={} reason={reason:?}",
                    algorithm.name()
                )
            }
            None => {
                return Err(format!(
                    "budget={budget}: no plan was prepared, so the host was not checked"
                )
                .into())
            }
        }
    }
    Ok(())
}
