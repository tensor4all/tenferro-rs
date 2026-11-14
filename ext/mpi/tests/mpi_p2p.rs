//! Two-rank scenarios on the crates.io `mpi` backend.
//!
//! `harness = false`: the binary runs one fixed scenario list per rank, which needs a
//! real `mpiexec -n 2` (see `scripts/run-mpi.sh`) rather than a self-spawning test
//! harness. A single-process run reports the skip and succeeds.

#[path = "scenarios/mod.rs"]
mod scenarios;

use tenferro_mpi::mpi::traits::*;
use tenferro_mpi::Upstream;

fn main() {
    let universe = tenferro_mpi::mpi::initialize().expect("MPI_Init");
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    if size < 2 {
        println!("rank {rank}: needs two ranks, skipping (run scripts/run-mpi.sh)");
        return;
    }
    assert_eq!(size, 2, "these scenarios pair exactly two ranks");

    let context = scenarios::Context {
        backend: &Upstream,
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
