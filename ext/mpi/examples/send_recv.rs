//! Send a tensor from rank 0 to rank 1, then back again.
//!
//! ```sh
//! mpiexec -n 2 -- cargo run --example send_recv
//! ```
//!
//! The example owns the MPI lifecycle: the adapter borrows a communicator and never
//! initializes or finalizes MPI itself.

use tenferro_mpi::mpi::traits::*;
use tenferro_mpi::{recv, recv_into, send};
use tenferro_tensor::{DynRank, Host, Tensor, TypedTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe = tenferro_mpi::mpi::initialize().expect("MPI_Init");
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();
    if size < 2 {
        println!("rank {rank}: run this example with at least two ranks");
        return Ok(());
    }
    let peer = world.process_at_rank((rank + 1) % size);
    let next = world.process_at_rank((rank + 1) % size);
    let previous = world.process_at_rank((rank + size - 1) % size);

    // A statically typed tensor.
    let typed = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(
        vec![2, 3],
        vec![41.0, 42.0, 43.0, 44.0, 45.0, 46.0 + f64::from(rank)],
    )?;
    if rank == 0 {
        send(next, 0, &typed)?;
        let back: TypedTensor<f64, DynRank, Host> = recv(previous, 0)?;
        println!("rank 0 received {:?}", back.as_slice());
    } else {
        let mut mine =
            TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2, 3], vec![0.0; 6])?;
        recv_into(previous, 0, &mut mine)?;
        send(next, 0, &mine)?;
        println!("rank {rank} passed on {:?}", mine.as_slice());
    }

    // The dtype-erased tensor, on its own tag.
    if rank == 0 {
        let erased = Tensor::from_typed(
            TypedTensor::<i64, DynRank, Host>::from_host_vec_col_major(vec![3], vec![7, 8, 9])?
                .into_dynamic(),
        );
        send(peer, 1, &erased)?;
    } else {
        let erased: Tensor = recv(peer, 1)?;
        println!("rank {rank} received {:?}", erased.as_slice::<i64>()?);
    }
    Ok(())
}
