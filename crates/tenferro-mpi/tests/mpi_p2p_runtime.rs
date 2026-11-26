//! Two-rank scenarios on the runtime-loading (`runtime`) backend of `tenferro-mpi`.
//!
//! This is the same scenario list as `mpi_p2p.rs`, included from the same package so
//! both backends are held to one protocol contract. Run it with
//! `scripts/run-mpi.sh rt`; it finds the site MPI in the process or by soname
//! (`MPI_NATIVE_LIB` overrides the search).

#[path = "scenarios/mod.rs"]
mod scenarios;

use tenferro_mpi::mpi::ffi;
use tenferro_mpi::mpi::traits::*;
use tenferro_mpi::Rt;

fn main() {
    // Skip only when no MPI library can be found at all. Every other loader error (an
    // unsupported family, a mismatch, an ambiguous process) is a real failure and must not
    // hide as a skip; `scripts/run-mpi.sh` also requires each rank's success line.
    let library = match ffi::loader::standalone() {
        Ok(library) => library,
        Err(error) if error.is_not_found() => {
            println!("no MPI library found ({error}); skipping the runtime-loading scenarios");
            return;
        }
        Err(error) => panic!("{error}"),
    };
    println!("MPI library: {} ({})", library.path, library.family.name());
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
    // A tenferro path that reached a panic stub would fail a user's session; fail here.
    assert_eq!(ffi::stub_hits(), 0, "the scenarios reached an unsupported MPI call");
    println!("rank {rank}: all scenarios passed");
}
