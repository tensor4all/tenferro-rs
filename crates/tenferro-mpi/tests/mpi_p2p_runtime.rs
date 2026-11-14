//! Two-rank scenarios on the runtime-loading (`runtime`) backend of `tenferro-mpi`.
//!
//! This is the same scenario list as `mpi_p2p.rs`, included from the same package so
//! both backends are held to one protocol contract. Needs `MPI_RT_LIB` and
//! `scripts/run-mpi.sh rt`.

#[path = "scenarios/mod.rs"]
mod scenarios;

use tenferro_mpi::mpi::traits::*;
use tenferro_mpi::Rt;

fn main() {
    // A plain `cargo test --workspace` must not need MPIwrapper: without the loaded
    // MPI ABI library there is nothing to initialize, so the binary reports the skip.
    if std::env::var_os("MPI_RT_LIB").is_none() {
        println!("MPI_RT_LIB is not set; skipping the runtime-loading scenarios");
        return;
    }
    let universe = tenferro_mpi::mpi::initialize().expect("MPI_Init");
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    if size < 2 {
        println!("rank {rank}: needs two ranks, skipping (run scripts/run-mpi.sh rt)");
        return;
    }
    assert_eq!(size, 2, "these scenarios pair exactly two ranks");

    let context = scenarios::Context {
        backend: &Rt,
        rank,
        other: world.process_at_rank((rank + 1) % 2),
        me: world.this_process(),
    };
    if let Err(error) = scenarios::run(&context) {
        eprintln!("rank {rank}: {error}");
        std::process::exit(1);
    }
    println!("rank {rank}: all scenarios passed");
}
