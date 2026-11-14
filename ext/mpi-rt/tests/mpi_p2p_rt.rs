//! Two-rank scenarios on the runtime-loading rsmpi-rt backend.
//!
//! The same scenario list as `ext/mpi/tests/mpi_p2p.rs`, included from that package so
//! both backends are held to one protocol contract. Needs `MPI_RT_LIB` and
//! `scripts/run-mpi.sh rt`.

#[path = "../../mpi/tests/scenarios/mod.rs"]
mod scenarios;

use tenferro_mpi_rt::mpi::traits::*;
use tenferro_mpi_rt::Rt;

fn main() {
    let universe = tenferro_mpi_rt::mpi::initialize().expect("MPI_Init");
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
