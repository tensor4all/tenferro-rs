//! Holding one CPU session across a stage, and lending its pool to a phase driver.
//!
//! Run with `cargo run -p tenferro-cpu --example held_session_driver`.
//!
//! Both entry shapes shown here reach the same session routes. The callback-scoped entry
//! (`BackendSessionHost::with_backend_session`) is the right shape for one operation or one
//! unrelated burst; a *held* session is the right shape for a stage that keeps its execution
//! domain open - a fit loop, a sweep, a batched driver - because the caller pays the session
//! entry once instead of once per operation. `with_session` hands out the same borrowed view
//! the scoped entry hands out, so routes do not change with the entry shape.
//!
//! Measured on AMD EPYC 7713P, rustc 1.97.1, one worker pinned to one CPU: an empty
//! scoped entry costs 537.7 ns against 321.1 ns for opening and closing a held session, and a
//! 95x95 by 95x1 f64 contraction costs 13.52 us scoped against 12.70 us held. See
//! `docs/design/held-cpu-session-1945-u2-concrete.md` for the full table and
//! `docs/guides/session-entry-cost.md` for when the difference matters.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use tenferro_cpu::CpuBackend;
use tenferro_tensor::{Tensor, TensorRead};

fn vector(values: &[f64]) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![values.len()],
        values.to_vec(),
    )?)
}

fn payload(tensor: &Tensor) -> Result<Vec<f64>, tenferro_tensor::Error> {
    Ok(tensor.as_slice::<f64>()?.to_vec())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = CpuBackend::with_threads(2)?;
    let mut session = backend.open_session()?;

    // A stage of dependent steps: each step reads the value the stage keeps in the session.
    let mut state = vector(&[1.0, 2.0, 3.0, 4.0])?;
    for _ in 0..4 {
        let doubled = session.with_session(|view| {
            view.add_read(
                TensorRead::from_tensor(&state),
                TensorRead::from_tensor(&state),
            )
        })?;
        state = doubled;
    }
    assert_eq!(payload(&state)?, vec![16.0, 32.0, 48.0, 64.0]);
    println!("held stage: 4 doublings left {:?}", payload(&state)?);

    // A phase lends the session's own pool: each lane pulls work from the caller's queue and
    // publishes a partial result. The queue here is the caller's own; the session only lends
    // its workers and its routes.
    let units = 8;
    let next = AtomicUsize::new(0);
    let unit = vector(&[1.0, 1.0, 2.0, 3.0])?;
    let sums = Mutex::new(Vec::new());
    let (next_in_lane, unit_in_lane, sums_in_lane) = (&next, &unit, &sums);
    let phase = session.phase(move |phase| {
        phase.run(move |_index, lane| -> Result<(), tenferro_tensor::Error> {
            let mut partial = 0.0;
            while next_in_lane.fetch_add(1, Ordering::SeqCst) < units {
                let total = lane
                    .session()
                    .reduce_sum_read(TensorRead::from_tensor(unit_in_lane), &[0])?;
                partial += payload(&total)?[0];
            }
            sums_in_lane.lock().expect("sums lock").push(partial);
            Ok(())
        })
    })?;
    phase?;

    let sums = sums.into_inner().expect("sums lock");
    let total: f64 = sums.iter().sum();
    assert_eq!(sums.len(), 2, "one partial per lane of the two-worker pool");
    assert_eq!(total, 8.0 * 7.0, "every queue unit ran exactly once");
    println!("phase: {} lanes produced partials {:?}", sums.len(), sums);

    // Closing restores the opening thread's affinity and releases the held resources.
    session.close()?;
    println!("held session closed");
    Ok(())
}
