use std::collections::HashSet;

/// Classify contraction modes into batch, left-only, right-only, and summed.
///
/// - batch: modes in A ∩ B ∩ C (preserved in both inputs and output)
/// - lo (left-only): modes in (A ∩ C) \ B (free modes of A)
/// - ro (right-only): modes in (B ∩ C) \ A (free modes of B)
/// - sum: modes in (A ∩ B) \ C (contracted/summed over)
///
/// Each category preserves the order in which modes first appear in subs_a
/// (for batch, lo, sum) or subs_b (for ro).
pub(crate) fn classify_modes(
    subs_a: &[u32],
    subs_b: &[u32],
    subs_c: &[u32],
) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let set_a: HashSet<u32> = subs_a.iter().copied().collect();
    let set_b: HashSet<u32> = subs_b.iter().copied().collect();
    let set_c: HashSet<u32> = subs_c.iter().copied().collect();

    let mut batch = Vec::new();
    let mut lo = Vec::new();
    let mut sum = Vec::new();
    let mut seen = HashSet::new();

    // Scan A modes: classify as batch, lo, or sum
    for &m in subs_a {
        if !seen.insert(m) {
            continue;
        }
        if set_b.contains(&m) && set_c.contains(&m) {
            batch.push(m);
        } else if set_c.contains(&m) && !set_b.contains(&m) {
            lo.push(m);
        } else if set_b.contains(&m) && !set_c.contains(&m) {
            sum.push(m);
        }
        // mode only in A and not in B or C: ignored (won't appear in output)
    }

    // Scan B modes for right-only
    let mut ro = Vec::new();
    let mut seen_b = HashSet::new();
    for &m in subs_b {
        if !seen_b.insert(m) {
            continue;
        }
        if set_c.contains(&m) && !set_a.contains(&m) {
            ro.push(m);
        }
    }

    (batch, lo, ro, sum)
}
