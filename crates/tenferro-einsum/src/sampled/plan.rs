//! Planning of a sampled einsum along a fixed contraction path.
//!
//! Provenance: ported from `src/plan.rs` of Fanerst/sampled-einsum at commit
//! c482b79a0f1e407390ef7822a89e7706220021d1 (Feng Pan, Apache-2.0), itself a
//! port of `contraction_scheme_sparse` in artensor's `artensor/contraction.py`
//! (Fanerst/artensor 58a93ae7e0199dc73f3f5dfbee15a98782aded7c, Apache-2.0).
//! The step kinds, the row bookkeeping (`rank_unique`, the right-row-major
//! pair order, the full-product test, `outer_select_ratio`), the chunk size
//! rule and the peak accounting are the prototype's, line for line where the
//! types allow. What differs is written next to the code and listed in the
//! module documentation of `crate::sampled`:
//!
//! - the network is tenferro's [`Subscripts`] plus a size dictionary, and the
//!   path comes from the resolved contraction tree;
//! - errors are tenferro's typed [`Error`]s;
//! - a gathered-pairs step is one [`Instr::Pairs`] instruction that the two
//!   executors realise differently (windows of a preallocated result on the
//!   concrete layer, pieces and one concatenation on the eager layer), so the
//!   plan carries a peak for each;
//! - the plan records each contraction's subscripts and shapes, so the
//!   concrete layer prepares one [`crate::ConcreteEinsumPlan`] per contraction
//!   (two at most per chunked step) before anything runs.
//!
//! Every node of the contraction tree is stored as a dense tensor over its
//! non-sampled labels plus one *configuration axis*. Row `p` of that axis is
//! one distinct assignment `P_v[p]` of the sampled labels below the node. A
//! sample's assignment restricted to `O_l ∪ O_r` is fixed by its rows in the
//! two children, so the rows of a node are the distinct pairs `(row in left,
//! row in right)` over the samples, sorted right-row-major. With that order
//! the full product `P_l × P_r` is exactly the column-major merge of the two
//! child axes, which is what the no-gather step relies on.

use std::collections::{HashMap, HashSet};

use crate::{Error, Result, Subscripts};

use super::{SampleOptions, SampledStepKind, SampledStepReport, Samples, OP};

/// One primitive operation of a plan. Slots `0..n_inputs` are the operands;
/// every other slot is a temporary owned by the executor.
#[derive(Clone, Debug)]
pub(crate) enum Instr {
    /// `out = einsum(inputs)`; `contraction` indexes [`Program::contractions`].
    Einsum {
        inputs: Vec<usize>,
        contraction: usize,
        out: usize,
    },
    /// `out = index_select(input, axis, positions)`.
    IndexSelect {
        input: usize,
        axis: usize,
        positions: Vec<usize>,
        out: usize,
    },
    /// `out = reshape(input, shape)` (column-major).
    Reshape {
        input: usize,
        shape: Vec<usize>,
        out: usize,
    },
    /// A gathered-pairs step in chunks.
    Pairs(Box<PairStep>),
    /// Drop a temporary that is no longer needed.
    Free { slot: usize },
}

/// `out[..., p, ...] = Σ left[..., left_rows[p], ...] right[..., right_rows[p], ...]`
/// over the pairs `p`, in chunks of `chunk` pairs.
#[derive(Clone, Debug)]
pub(crate) struct PairStep {
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) axis_left: usize,
    pub(crate) axis_right: usize,
    pub(crate) left_rows: Vec<usize>,
    pub(crate) right_rows: Vec<usize>,
    /// Pairs per chunk; the last chunk may be shorter.
    pub(crate) chunk: usize,
    /// The contraction of a full chunk, and of a shorter last chunk if any.
    pub(crate) full: usize,
    pub(crate) tail: Option<usize>,
    pub(crate) out: usize,
    /// Axis of `out` that carries the pairs.
    pub(crate) out_axis: usize,
    pub(crate) out_shape: Vec<usize>,
}

impl PairStep {
    pub(crate) fn rows(&self) -> usize {
        self.left_rows.len()
    }

    pub(crate) fn chunks(&self) -> usize {
        self.rows().div_ceil(self.chunk)
    }

    /// Pair range of chunk `id`.
    pub(crate) fn range(&self, id: usize) -> std::ops::Range<usize> {
        let start = id * self.chunk;
        start..(start + self.chunk).min(self.rows())
    }

    /// Contraction of chunk `id`.
    pub(crate) fn contraction(&self, id: usize) -> usize {
        match self.tail {
            Some(tail) if self.range(id).len() != self.chunk => tail,
            _ => self.full,
        }
    }
}

/// One binary or unary contraction of a plan: integer subscripts and the
/// shapes of its inputs.
#[derive(Clone, Debug)]
pub(crate) struct Contraction {
    pub(crate) subscripts: Subscripts,
    pub(crate) input_shapes: Vec<Vec<usize>>,
}

/// A planned program and its numbers.
#[derive(Clone, Debug)]
pub(crate) struct Program {
    pub(crate) n_slots: usize,
    pub(crate) instrs: Vec<Instr>,
    pub(crate) contractions: Vec<Contraction>,
    pub(crate) result: usize,
    pub(crate) output_shape: Vec<usize>,
    pub(crate) leaf_rows: Vec<usize>,
    pub(crate) steps: Vec<SampledStepReport>,
    /// Planned shape of every slot.
    pub(crate) slot_shapes: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
struct Cfg {
    label: u32,
    rows: usize,
    sample_rows: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Node {
    slot: usize,
    labels: Vec<u32>,
    cfg: Option<Cfg>,
}

struct Builder {
    instrs: Vec<Instr>,
    contractions: Vec<Contraction>,
    slot_shapes: Vec<Vec<usize>>,
    dims: HashMap<u32, usize>,
    next_label: u32,
    n_inputs: usize,
}

impl Builder {
    fn new_slot(&mut self, shape: Vec<usize>) -> usize {
        self.slot_shapes.push(shape);
        self.slot_shapes.len() - 1
    }

    fn fresh_label(&mut self, extent: usize) -> u32 {
        let label = self.next_label;
        self.next_label += 1;
        self.dims.insert(label, extent);
        label
    }

    fn shape(&self, labels: &[u32]) -> Vec<usize> {
        labels.iter().map(|l| self.dims[l]).collect()
    }

    fn elems(&self, labels: &[u32]) -> usize {
        labels.iter().map(|l| self.dims[l]).product()
    }

    fn free_if_temp(&mut self, slot: usize) {
        if slot >= self.n_inputs {
            self.instrs.push(Instr::Free { slot });
        }
    }

    fn contraction(
        &mut self,
        input_labels: Vec<Vec<u32>>,
        output_labels: Vec<u32>,
        input_shapes: Vec<Vec<usize>>,
    ) -> usize {
        self.contractions.push(Contraction {
            subscripts: Subscripts::new(
                &input_labels.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                &output_labels,
            ),
            input_shapes,
        });
        self.contractions.len() - 1
    }

    fn einsum(
        &mut self,
        inputs: Vec<usize>,
        input_labels: Vec<Vec<u32>>,
        output_labels: Vec<u32>,
    ) -> usize {
        let input_shapes = inputs
            .iter()
            .map(|&s| self.slot_shapes[s].clone())
            .collect();
        let out = self.new_slot(self.shape(&output_labels));
        let contraction = self.contraction(input_labels, output_labels, input_shapes);
        self.instrs.push(Instr::Einsum {
            inputs,
            contraction,
            out,
        });
        out
    }

    fn index_select(&mut self, input: usize, axis: usize, positions: Vec<usize>) -> usize {
        let mut shape = self.slot_shapes[input].clone();
        shape[axis] = positions.len();
        let out = self.new_slot(shape);
        self.instrs.push(Instr::IndexSelect {
            input,
            axis,
            positions,
            out,
        });
        out
    }

    fn reshape(&mut self, input: usize, shape: Vec<usize>) -> usize {
        let out = self.new_slot(shape.clone());
        self.instrs.push(Instr::Reshape { input, shape, out });
        out
    }

    /// Multiply-adds of a binary contraction: the product of the extents of every label involved.
    fn madds(&self, lhs: &[u32], rhs: &[u32]) -> f64 {
        let mut seen = HashSet::new();
        lhs.iter()
            .chain(rhs)
            .filter(|l| seen.insert(**l))
            .map(|l| self.dims[l] as f64)
            .product()
    }
}

/// Sorted distinct keys, and each key's rank among them.
fn rank_unique(keys: &[u64]) -> (Vec<u64>, Vec<u32>) {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_unstable_by_key(|&i| keys[i]);
    let mut uniq: Vec<u64> = Vec::new();
    let mut rank = vec![0u32; keys.len()];
    for i in order {
        if uniq.last() != Some(&keys[i]) {
            uniq.push(keys[i]);
        }
        rank[i] = (uniq.len() - 1) as u32;
    }
    (uniq, rank)
}

/// Output labels in tenferro's canonical binary order: left-free, right-free, then batch.
fn canonical_out(lhs: &[u32], rhs: &[u32], keep: impl Fn(u32) -> bool) -> Vec<u32> {
    let in_lhs: HashSet<u32> = lhs.iter().copied().collect();
    let in_rhs: HashSet<u32> = rhs.iter().copied().collect();
    let mut out = Vec::new();
    out.extend(
        lhs.iter()
            .copied()
            .filter(|&x| keep(x) && !in_rhs.contains(&x)),
    );
    out.extend(
        rhs.iter()
            .copied()
            .filter(|&x| keep(x) && !in_lhs.contains(&x)),
    );
    out.extend(
        lhs.iter()
            .copied()
            .filter(|&x| keep(x) && in_rhs.contains(&x)),
    );
    out
}

fn invalid(argument: &'static str, message: impl Into<String>) -> Error {
    Error::invalid_argument(OP, argument, message.into())
}

/// Check the network and the samples (the path is checked by the tree that produced it).
pub(crate) fn validate(
    subscripts: &Subscripts,
    shapes: &[&[usize]],
    sizes: &HashMap<u32, usize>,
    samples: &Samples,
) -> Result<()> {
    for (i, (labels, shape)) in subscripts.inputs.iter().zip(shapes).enumerate() {
        let mut seen = HashSet::new();
        for (l, &extent) in labels.iter().zip(shape.iter()) {
            if !seen.insert(l) {
                return Err(invalid(
                    "spec",
                    format!("label {l} repeats within operand {i}; v0 takes no diagonals"),
                ));
            }
            if extent == 0 {
                return Err(invalid(
                    "inputs",
                    format!("label {l} has extent 0 on operand {i}; v0 needs positive extents"),
                ));
            }
            if extent != sizes[l] {
                return Err(invalid(
                    "inputs",
                    format!(
                        "label {l} has extent {extent} on operand {i} and {} elsewhere; v0 takes no broadcast labels",
                        sizes[l]
                    ),
                ));
            }
        }
    }
    let mut seen = HashSet::new();
    for l in &subscripts.output {
        if !seen.insert(l) {
            return Err(invalid("spec", format!("output label {l} repeats")));
        }
        if !subscripts.inputs.iter().any(|ls| ls.contains(l)) {
            return Err(invalid(
                "spec",
                format!("output label {l} is on no operand"),
            ));
        }
    }
    if samples.labels.is_empty() {
        return Err(invalid(
            "samples",
            "at least one output label must be sampled",
        ));
    }
    let mut seen = HashSet::new();
    for l in &samples.labels {
        if !seen.insert(l) {
            return Err(invalid("samples", format!("sampled label {l} repeats")));
        }
        if !subscripts.output.contains(l) {
            return Err(invalid(
                "samples",
                format!("sampled label {l} is not an output label"),
            ));
        }
    }
    let expected = samples.n.checked_mul(samples.labels.len());
    if expected != Some(samples.values.len()) {
        return Err(invalid(
            "samples",
            format!(
                "values hold {} entries, expected n * labels = {} * {}",
                samples.values.len(),
                samples.n,
                samples.labels.len()
            ),
        ));
    }
    if samples.n > u32::MAX as usize {
        return Err(invalid("samples", "more than 2^32 - 1 samples"));
    }
    for (j, l) in samples.labels.iter().enumerate() {
        let d = sizes[l];
        if let Some(v) = samples.values[j * samples.n..(j + 1) * samples.n]
            .iter()
            .find(|&&v| v >= d)
        {
            return Err(invalid(
                "samples",
                format!("sample value {v} is out of range for label {l} of extent {d}"),
            ));
        }
    }
    Ok(())
}

/// Plan a sampled einsum along an SSA contraction path (`pairs[k] = (a, b)`
/// contracts nodes `a` and `b` into node `n_inputs + k`). The inputs must
/// have passed [`validate`].
pub(crate) fn plan(
    subscripts: &Subscripts,
    sizes: &HashMap<u32, usize>,
    pairs: &[(usize, usize)],
    samples: &Samples,
    opts: &SampleOptions,
) -> Result<Program> {
    let n_inputs = subscripts.inputs.len();
    let n = samples.n;
    // With no samples, plan one dummy sample and select none of it at the end.
    let (n_eff, values): (usize, Vec<usize>) = if n == 0 {
        (1, vec![0; samples.labels.len()])
    } else {
        (n, samples.values.clone())
    };
    let column: HashMap<u32, usize> = samples
        .labels
        .iter()
        .enumerate()
        .map(|(j, &l)| (l, j))
        .collect();
    let rest: Vec<u32> = subscripts
        .output
        .iter()
        .copied()
        .filter(|l| !column.contains_key(l))
        .collect();
    let rest_set: HashSet<u32> = rest.iter().copied().collect();
    let max_label = subscripts
        .inputs
        .iter()
        .flatten()
        .chain(&subscripts.output)
        .chain(sizes.keys())
        .copied()
        .max()
        .unwrap_or(0);
    // Fresh labels for configuration axes: at most one per leaf and step.
    let fresh_needed = (2 * n_inputs) as u32;
    if max_label.checked_add(fresh_needed + 1).is_none() {
        return Err(invalid(
            "spec",
            "labels near u32::MAX leave no room for configuration labels",
        ));
    }
    let mut b = Builder {
        instrs: Vec::new(),
        contractions: Vec::new(),
        slot_shapes: Vec::new(),
        dims: sizes.clone(),
        next_label: max_label + 1,
        n_inputs,
    };
    let mut leaf_rows = Vec::with_capacity(n_inputs);
    let mut steps = Vec::with_capacity(pairs.len());
    for labels in &subscripts.inputs {
        let shape = b.shape(labels);
        b.new_slot(shape);
    }

    // How many live nodes carry each dense (non-sampled) label.
    let mut count: HashMap<u32, usize> = HashMap::new();
    let mut nodes: Vec<Option<Node>> = Vec::new();
    for (i, labels) in subscripts.inputs.iter().enumerate() {
        for l in labels.iter().filter(|l| !column.contains_key(l)) {
            *count.entry(*l).or_default() += 1;
        }
        let q: Vec<u32> = labels
            .iter()
            .copied()
            .filter(|l| column.contains_key(l))
            .collect();
        if q.is_empty() {
            leaf_rows.push(1);
            nodes.push(Some(Node {
                slot: i,
                labels: labels.clone(),
                cfg: None,
            }));
            continue;
        }
        // Linear index of each sample over this operand's sampled labels, first label fastest.
        let mut lin = vec![0u64; n_eff];
        let mut stride = 1u64;
        // contraction-kernel-ok: integer linearisation of sample indices (host bookkeeping, no tensor data)
        for l in &q {
            let col = column[l];
            for (s, v) in lin.iter_mut().enumerate() {
                *v += stride * values[s + n_eff * col] as u64;
            }
            stride = stride
                .checked_mul(sizes[l] as u64)
                .ok_or_else(|| invalid("samples", "a leaf's sampled extents overflow u64"))?;
        }
        let (positions, sample_rows) = rank_unique(&lin);
        let rows = positions.len();
        leaf_rows.push(rows);
        let positions: Vec<usize> = positions.into_iter().map(|p| p as usize).collect();
        let (source, mut node_labels, axis) = if q.len() == 1 {
            let axis = labels
                .iter()
                .position(|&x| x == q[0])
                .expect("sampled label is on this operand");
            (i, labels.clone(), axis)
        } else {
            // Move the sampled labels to the end (in `q` order) and merge them into one axis.
            let mut permuted: Vec<u32> =
                labels.iter().copied().filter(|l| !q.contains(l)).collect();
            let kept = permuted.len();
            permuted.extend(&q);
            let t1 = b.einsum(vec![i], vec![labels.clone()], permuted.clone());
            let mut shape: Vec<usize> = permuted[..kept].iter().map(|l| b.dims[l]).collect();
            shape.push(stride as usize);
            let t2 = b.reshape(t1, shape);
            b.free_if_temp(t1);
            let mut merged_labels = permuted[..kept].to_vec();
            merged_labels.push(u32::MAX); // replaced by the configuration label below
            (t2, merged_labels, kept)
        };
        let c = b.fresh_label(rows);
        node_labels[axis] = c;
        let slot = b.index_select(source, axis, positions);
        if source != i {
            b.free_if_temp(source);
        }
        nodes.push(Some(Node {
            slot,
            labels: node_labels,
            cfg: Some(Cfg {
                label: c,
                rows,
                sample_rows,
            }),
        }));
    }

    for &(ia, ib) in pairs {
        let l = nodes[ia].take().expect("validated path");
        let r = nodes[ib].take().expect("validated path");
        let cfg_labels: HashSet<u32> = l.cfg.iter().chain(&r.cfg).map(|c| c.label).collect();
        for x in l
            .labels
            .iter()
            .chain(&r.labels)
            .filter(|x| !cfg_labels.contains(x))
        {
            *count.get_mut(x).expect("counted") -= 1;
        }
        let survive = |x: u32| rest_set.contains(&x) || count.get(&x).copied().unwrap_or(0) > 0;
        let rows_left = l.cfg.as_ref().map_or(1, |c| c.rows);
        let rows_right = r.cfg.as_ref().map_or(1, |c| c.rows);
        let (node, kind, madds) = match (&l.cfg, &r.cfg) {
            (None, None) => {
                let out = canonical_out(&l.labels, &r.labels, survive);
                let madds = b.madds(&l.labels, &r.labels);
                let slot = b.einsum(
                    vec![l.slot, r.slot],
                    vec![l.labels.clone(), r.labels.clone()],
                    out.clone(),
                );
                (
                    Node {
                        slot,
                        labels: out,
                        cfg: None,
                    },
                    SampledStepKind::Dense,
                    madds,
                )
            }
            (Some(c), None) | (None, Some(c)) => {
                let out = canonical_out(&l.labels, &r.labels, |x| x == c.label || survive(x));
                let madds = b.madds(&l.labels, &r.labels);
                let slot = b.einsum(
                    vec![l.slot, r.slot],
                    vec![l.labels.clone(), r.labels.clone()],
                    out.clone(),
                );
                (
                    Node {
                        slot,
                        labels: out,
                        cfg: Some(c.clone()),
                    },
                    SampledStepKind::OneSided,
                    madds,
                )
            }
            (Some(cl), Some(cr)) => {
                let keys: Vec<u64> = (0..n_eff)
                    .map(|s| ((cr.sample_rows[s] as u64) << 32) | cl.sample_rows[s] as u64)
                    .collect();
                let (pair_keys, sample_rows) = rank_unique(&keys);
                let rows = pair_keys.len();
                let product = cl.rows * cr.rows;
                let dense = canonical_out(&l.labels, &r.labels, |x| {
                    !cfg_labels.contains(&x) && survive(x)
                });
                let dense_elems = b.elems(&dense);
                let outer = rows == product
                    || (opts.outer_select_ratio > 0.0
                        && product as f64 <= opts.outer_select_ratio * rows as f64
                        && product.saturating_mul(dense_elems) <= opts.max_chunk_elements);
                if outer {
                    let mut out = dense.clone();
                    out.push(cl.label);
                    out.push(cr.label);
                    let madds = b.madds(&l.labels, &r.labels);
                    let t = b.einsum(
                        vec![l.slot, r.slot],
                        vec![l.labels.clone(), r.labels.clone()],
                        out,
                    );
                    let mut shape: Vec<usize> = dense.iter().map(|x| b.dims[x]).collect();
                    shape.push(product);
                    let merged = b.reshape(t, shape);
                    b.free_if_temp(t);
                    let c = b.fresh_label(rows);
                    let mut labels = dense.clone();
                    labels.push(c);
                    let (slot, kind) = if rows == product {
                        (merged, SampledStepKind::FullProduct)
                    } else {
                        // Row (a, b) of the product sits at a + rows_left * b.
                        let positions: Vec<usize> = pair_keys
                            .iter()
                            .map(|&k| (k & 0xffff_ffff) as usize + cl.rows * (k >> 32) as usize)
                            .collect();
                        let slot = b.index_select(merged, dense.len(), positions);
                        b.free_if_temp(merged);
                        (slot, SampledStepKind::OuterSelect)
                    };
                    (
                        Node {
                            slot,
                            labels,
                            cfg: Some(Cfg {
                                label: c,
                                rows,
                                sample_rows,
                            }),
                        },
                        kind,
                        madds,
                    )
                } else {
                    let p = b.fresh_label(rows);
                    let lhs: Vec<u32> = l
                        .labels
                        .iter()
                        .map(|&x| if x == cl.label { p } else { x })
                        .collect();
                    let rhs: Vec<u32> = r
                        .labels
                        .iter()
                        .map(|&x| if x == cr.label { p } else { x })
                        .collect();
                    let out = canonical_out(&lhs, &rhs, |x| x == p || survive(x));
                    let axis_left = l
                        .labels
                        .iter()
                        .position(|&x| x == cl.label)
                        .expect("cfg axis");
                    let axis_right = r
                        .labels
                        .iter()
                        .position(|&x| x == cr.label)
                        .expect("cfg axis");
                    let out_axis = out
                        .iter()
                        .position(|&x| x == p)
                        .expect("pair axis survives");
                    let per_row = (b.elems(&out) / rows)
                        .max(b.elems(&l.labels) / cl.rows)
                        .max(b.elems(&r.labels) / cr.rows)
                        .max(1);
                    let chunk = (opts.max_chunk_elements / per_row).clamp(1, rows);
                    let left_rows: Vec<usize> = pair_keys
                        .iter()
                        .map(|&k| (k & 0xffff_ffff) as usize)
                        .collect();
                    let right_rows: Vec<usize> =
                        pair_keys.iter().map(|&k| (k >> 32) as usize).collect();
                    let madds = b.madds(&lhs, &rhs);
                    let chunk_shapes = |b: &Builder, m: usize| {
                        let with = |labels: &[u32]| -> Vec<usize> {
                            labels
                                .iter()
                                .map(|x| if *x == p { m } else { b.dims[x] })
                                .collect()
                        };
                        vec![with(&lhs), with(&rhs)]
                    };
                    let full_shapes = chunk_shapes(&b, chunk);
                    let full =
                        b.contraction(vec![lhs.clone(), rhs.clone()], out.clone(), full_shapes);
                    let tail = match rows % chunk {
                        0 => None,
                        m => {
                            let tail_shapes = chunk_shapes(&b, m);
                            Some(b.contraction(
                                vec![lhs.clone(), rhs.clone()],
                                out.clone(),
                                tail_shapes,
                            ))
                        }
                    };
                    let out_shape = b.shape(&out);
                    let slot = b.new_slot(out_shape.clone());
                    let step = PairStep {
                        left: l.slot,
                        right: r.slot,
                        axis_left,
                        axis_right,
                        left_rows,
                        right_rows,
                        chunk,
                        full,
                        tail,
                        out: slot,
                        out_axis,
                        out_shape,
                    };
                    let chunks = step.chunks();
                    b.instrs.push(Instr::Pairs(Box::new(step)));
                    (
                        Node {
                            slot,
                            labels: out,
                            cfg: Some(Cfg {
                                label: p,
                                rows,
                                sample_rows,
                            }),
                        },
                        SampledStepKind::Pairs { chunks },
                        madds,
                    )
                }
            }
        };
        b.free_if_temp(l.slot);
        b.free_if_temp(r.slot);
        for x in node
            .labels
            .iter()
            .filter(|x| Some(**x) != node.cfg.as_ref().map(|c| c.label))
        {
            *count.entry(*x).or_default() += 1;
        }
        let rows = node.cfg.as_ref().map_or(1, |c| c.rows);
        steps.push(SampledStepReport {
            kind,
            rows_left,
            rows_right,
            rows,
            out_elements: b.elems(&node.labels),
            flops: madds,
        });
        nodes.push(Some(node));
    }

    let root = nodes
        .iter_mut()
        .rev()
        .find_map(Option::take)
        .expect("one root");
    let cfg = root
        .cfg
        .as_ref()
        .expect("every sampled label is on some operand");
    // Bring the root into [rest..., configurations]; this also sums any label that is not an output.
    let mut target = rest.clone();
    target.push(cfg.label);
    let mut slot = root.slot;
    if root.labels != target {
        slot = b.einsum(vec![root.slot], vec![root.labels.clone()], target.clone());
        b.free_if_temp(root.slot);
    }
    // Put the rows in sample order (duplicates allowed).
    let positions: Vec<usize> = if n == 0 {
        Vec::new()
    } else {
        cfg.sample_rows.iter().map(|&r| r as usize).collect()
    };
    let identity =
        positions.len() == cfg.rows && positions.iter().enumerate().all(|(i, &p)| i == p);
    if !identity {
        let selected = b.index_select(slot, rest.len(), positions);
        b.free_if_temp(slot);
        slot = selected;
    }
    let mut output_shape: Vec<usize> = rest.iter().map(|l| sizes[l]).collect();
    output_shape.push(n);

    Ok(Program {
        n_slots: b.slot_shapes.len(),
        instrs: b.instrs,
        contractions: b.contractions,
        result: slot,
        output_shape,
        leaf_rows,
        steps,
        slot_shapes: b.slot_shapes,
    })
}

/// The two executors' peaks of live temporaries, from the planned sizes.
///
/// Counted: every slot an instruction writes, from that instruction until
/// its `Free`, plus, inside a gathered-pairs step, the gathered rows of the
/// chunk being contracted. The concrete executor writes each chunk into a
/// window of the preallocated step result; the eager executor keeps each
/// chunk's result as a piece and concatenates the pieces at the end (the
/// prototype's program), so its peak also holds the pieces and the
/// concatenation.
pub(crate) fn peaks(program: &Program) -> (usize, usize) {
    let e: Vec<usize> = program
        .slot_shapes
        .iter()
        .map(|shape| shape.iter().product())
        .collect();
    let mut concrete = (0usize, 0usize); // (live, peak)
    let mut eager = (0usize, 0usize);
    let bump = |state: &mut (usize, usize), add: usize| {
        state.0 += add;
        state.1 = state.1.max(state.0);
    };
    for instr in &program.instrs {
        match instr {
            Instr::Free { slot } => {
                concrete.0 -= e[*slot];
                eager.0 -= e[*slot];
            }
            Instr::Einsum { out, .. }
            | Instr::IndexSelect { out, .. }
            | Instr::Reshape { out, .. } => {
                bump(&mut concrete, e[*out]);
                bump(&mut eager, e[*out]);
            }
            Instr::Pairs(step) => {
                let per_row =
                    |slot: usize, axis: usize| e[slot] / program.slot_shapes[slot][axis].max(1);
                let left_per_row = per_row(step.left, step.axis_left);
                let right_per_row = per_row(step.right, step.axis_right);
                let out_per_row = per_row(step.out, step.out_axis);
                // Concrete: the result, then each chunk's gathered rows on top.
                bump(&mut concrete, e[step.out]);
                let widest = step.range(0).len();
                let held = (left_per_row + right_per_row) * widest;
                concrete.1 = concrete.1.max(concrete.0 + held);
                // Eager: per chunk the gathered rows and the piece, then the pieces stay.
                let chunks = step.chunks();
                for id in 0..chunks {
                    let m = step.range(id).len();
                    bump(&mut eager, (left_per_row + right_per_row + out_per_row) * m);
                    eager.0 -= (left_per_row + right_per_row) * m;
                }
                if chunks > 1 {
                    bump(&mut eager, e[step.out]);
                    eager.0 -= e[step.out];
                }
            }
        }
    }
    (concrete.1, eager.1)
}
