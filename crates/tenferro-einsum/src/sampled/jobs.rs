//! Job list of a gather-free pair step (sampled einsum v1, #2675).
//!
//! A pair `p` multiplies row `lr[p]` of the left slot by row `rr[p]` of the
//! right slot. With the configuration axis last, a row is the contiguous block
//! `[r * blk, (r + 1) * blk)`, so a job addresses the slots in place.

use cpueinsum::GroupedGemmJob;

/// One geometry for every pair of a step: `batch` GEMMs of `(m, k, n)` per
/// pair; a row block is `m*k*batch` (left), `k*n*batch` (right) and
/// `m*n*batch` (output) elements.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PairGeometry {
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) batch: usize,
}

impl PairGeometry {
    pub(crate) fn left_block(&self) -> usize {
        self.m * self.k * self.batch
    }
    pub(crate) fn right_block(&self) -> usize {
        self.k * self.n * self.batch
    }
    pub(crate) fn out_block(&self) -> usize {
        self.m * self.n * self.batch
    }
}

/// Jobs for the pairs `pairs` (a range into `lr`/`rr`).
///
/// **Offset contract.** `GroupedPlan::execute_uninit` requires the jobs to
/// cover the output slice it is given exactly once, from element 0, without
/// gaps (`GroupedError::FreshCoverage`, checked before the first write). So:
/// `lhs_offset`/`rhs_offset` stay in the operand slots' own coordinates (the
/// slot is the buffer; nothing was copied), while `out_offset` **subtracts the
/// chunk head** (`pairs.start * out_block`), and the caller passes
/// `out[pairs.start * out_block .. pairs.end * out_block]`.
pub(crate) fn pair_jobs(
    geometry: PairGeometry,
    lr: &[usize],
    rr: &[usize],
    pairs: std::ops::Range<usize>,
) -> Vec<GroupedGemmJob> {
    let PairGeometry { m, k, n, batch } = geometry;
    let head = pairs.start;
    let mut jobs = Vec::with_capacity(pairs.len() * batch);
    for p in pairs {
        for b in 0..batch {
            jobs.push(GroupedGemmJob::new(
                lr[p] * geometry.left_block() + b * m * k,
                rr[p] * geometry.right_block() + b * k * n,
                (p - head) * geometry.out_block() + b * m * n,
                m,
                k,
                n,
            ));
        }
    }
    jobs
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::MaybeUninit;
    use cpueinsum::{Exec, GroupedError, GroupedPlan};

    fn fill(len: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) % 17) as f64 - 8.0
            })
            .collect()
    }

    /// Dense reference: out[(p, b)] = left[lr[p], b] * right[rr[p], b],
    /// column-major `(m,k)`, `(k,n)`, `(m,n)` blocks.
    fn dense(g: PairGeometry, left: &[f64], right: &[f64], lr: &[usize], rr: &[usize]) -> Vec<f64> {
        let mut out = vec![0.0; lr.len() * g.out_block()];
        for p in 0..lr.len() {
            for b in 0..g.batch {
                for j in 0..g.n {
                    for i in 0..g.m {
                        let mut acc = 0.0;
                        for t in 0..g.k {
                            acc += left[lr[p] * g.left_block() + b * g.m * g.k + t * g.m + i]
                                * right[rr[p] * g.right_block() + b * g.k * g.n + j * g.k + t];
                        }
                        out[p * g.out_block() + b * g.m * g.n + j * g.m + i] = acc;
                    }
                }
            }
        }
        out
    }

    /// The plan's 5.1 case: three pairs split 2 + 1 at batch = 2.
    fn setup() -> (PairGeometry, Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
        let g = PairGeometry {
            m: 2,
            k: 3,
            n: 2,
            batch: 2,
        };
        let (lr, rr) = (vec![3, 0, 3], vec![1, 2, 0]); // repeated left row
        let left = fill(4 * g.left_block(), 1);
        let right = fill(3 * g.right_block(), 2);
        (g, left, right, lr, rr)
    }

    fn run(
        g: PairGeometry,
        left: &[f64],
        right: &[f64],
        jobs: &[GroupedGemmJob],
        len: usize,
    ) -> cpueinsum::Result<Vec<f64>> {
        let plan = GroupedPlan::<f64>::new(jobs, &Default::default())?;
        let mut out = vec![MaybeUninit::<f64>::uninit(); len];
        let (done, _) =
            plan.execute_uninit(&mut Exec::serial(), 1.0, left, right, 0.0, &mut out)?;
        let _ = g;
        Ok(done.to_vec())
    }

    #[test]
    fn three_pairs_split_two_plus_one_at_batch_two_match_dense() {
        let (g, left, right, lr, rr) = setup();
        let want = dense(g, &left, &right, &lr, &rr);
        let ob = g.out_block();
        let mut got = Vec::new();
        for pairs in [0..2, 2..3] {
            let jobs = pair_jobs(g, &lr, &rr, pairs.clone());
            // The second chunk's head is non-zero: its first job must write at 0.
            assert_eq!(jobs[0].out_offset, 0);
            got.extend(run(g, &left, &right, &jobs, pairs.len() * ob).unwrap());
        }
        assert_eq!(got, want);
        // One chunk of all three agrees too (the v0-shaped single call).
        let jobs = pair_jobs(g, &lr, &rr, 0..3);
        assert_eq!(run(g, &left, &right, &jobs, 3 * ob).unwrap(), want);
    }

    #[test]
    fn a_missing_chunk_head_subtraction_is_refused_before_any_write() {
        let (g, left, right, lr, rr) = setup();
        let ob = g.out_block();
        // Same jobs for the remainder chunk but with global output offsets.
        let wrong: Vec<_> = pair_jobs(g, &lr, &rr, 2..3)
            .into_iter()
            .map(|mut j| {
                j.out_offset += 2 * ob;
                j
            })
            .collect();
        let err = run(g, &left, &right, &wrong, ob).unwrap_err();
        assert!(
            matches!(&err, cpueinsum::Error::Grouped(GroupedError::FreshCoverage))
                || err.to_string().to_lowercase().contains("coverage")
                || err.to_string().to_lowercase().contains("range"),
            "{err}"
        );
    }
}
