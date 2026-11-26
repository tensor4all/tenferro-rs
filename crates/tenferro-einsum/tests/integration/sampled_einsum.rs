//! Sampled einsum (#2675) against a dense einsum followed by host-side
//! selection.
//!
//! Provenance: ported from `tests/dense_reference.rs` and `tests/common/mod.rs`
//! of Fanerst/sampled-einsum at commit c482b79a0f1e407390ef7822a89e7706220021d1
//! (Feng Pan, Apache-2.0): the random-network generator (1-6 operands, labels
//! of extent 1-3, a random output and a random non-empty sampled subset,
//! random samples with duplicates or every combination in shuffled order, a
//! random SSA path), the three planner settings, the 400 f64 and 150 c64
//! cases, the measured-equals-planned peak assertion, the step-kind coverage
//! check and the hand-written cases. The prototype draws with `rand::StdRng`;
//! tenferro has no `rand` dev-dependency, so the same generator draws from the
//! LCG of the sliced einsum tests: the distribution is the prototype's, the
//! individual networks are not.

use std::collections::HashMap;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{
    ContractionTree, EinsumOps, EinsumOptimize, EinsumOptions, EinsumSubscripts, SampleOptions,
    SampledEinsumPlan, SampledStepKind, Samples, Subscripts,
};
use tenferro_tensor::{DType, ErrorKind, Tensor, ValidationKind};

/// A small deterministic generator (no `rand` dev-dependency).
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    /// Uniform in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() % (hi - lo + 1) as u64) as usize
    }

    pub(crate) fn unit(&mut self) -> f64 {
        (self.next() % 2_000_001) as f64 / 1_000_000.0 - 1.0
    }

    fn chance(&mut self, p: f64) -> bool {
        (self.next() % 1_000_000) as f64 / 1_000_000.0 < p
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.range(0, i);
            items.swap(i, j);
        }
    }
}

/// An einsum with integer labels (the prototype's `Network`).
#[derive(Clone, Debug)]
pub(crate) struct Network {
    pub(crate) inputs: Vec<Vec<u32>>,
    pub(crate) output: Vec<u32>,
    pub(crate) sizes: HashMap<u32, usize>,
}

impl Network {
    pub(crate) fn subscripts(&self) -> EinsumSubscripts {
        let labels: Vec<&[u32]> = self.inputs.iter().map(Vec::as_slice).collect();
        EinsumSubscripts::new(&labels, &self.output)
    }

    pub(crate) fn shapes(&self) -> Vec<Vec<usize>> {
        self.inputs
            .iter()
            .map(|ls| ls.iter().map(|l| self.sizes[l]).collect())
            .collect()
    }

    /// Options that plan along exactly `path` (SSA pairs).
    pub(crate) fn along(&self, path: &[(usize, usize)], options: SampleOptions) -> SampleOptions {
        let shapes = self.shapes();
        let refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        let tree = ContractionTree::from_pairs(&Subscripts::from(&self.subscripts()), &refs, path)
            .expect("a valid SSA path");
        options.einsum(EinsumOptions::default().optimize(EinsumOptimize::Tree(tree)))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Case {
    pub(crate) net: Network,
    pub(crate) path: Vec<(usize, usize)>,
    pub(crate) samples: Samples,
}

/// The three planner settings every random case runs under (the prototype's).
pub(crate) fn settings() -> Vec<(&'static str, SampleOptions)> {
    vec![
        ("default", SampleOptions::default()),
        (
            "pairs, tiny chunks",
            SampleOptions::default()
                .max_chunk_elements(3)
                .outer_select_ratio(0.0),
        ),
        (
            "outer select",
            SampleOptions::default()
                .max_chunk_elements(usize::MAX)
                .outer_select_ratio(1e9),
        ),
    ]
}

/// A random einsum: 1-6 operands, labels of extent 1-3 (some shared by three
/// or more operands), a random output, a random non-empty sampled subset of
/// it, and a random SSA path.
pub(crate) fn random_case(rng: &mut Lcg) -> Case {
    let n_ops = rng.range(1, 6);
    let n_labels = rng.range(2, 7) as u32;
    let sizes: HashMap<u32, usize> = (0..n_labels).map(|l| (l, rng.range(1, 3))).collect();
    let mut inputs = Vec::new();
    for _ in 0..n_ops {
        let rank = rng.range(1, 4.min(n_labels as usize));
        let mut pool: Vec<u32> = (0..n_labels).collect();
        rng.shuffle(&mut pool);
        inputs.push(pool[..rank].to_vec());
    }
    let mut used: Vec<u32> = inputs.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    rng.shuffle(&mut used);
    let n_out = rng.range(1, used.len().min(4));
    let output = used[..n_out].to_vec();
    let mut sampled = output.clone();
    rng.shuffle(&mut sampled);
    sampled.truncate(rng.range(1, output.len()));
    let net = Network {
        inputs,
        output,
        sizes,
    };
    let samples = random_samples(rng, &net, &sampled);
    let path = random_path(rng, n_ops);
    Case { net, path, samples }
}

/// Random samples (with duplicates), or every combination when there are few.
fn random_samples(rng: &mut Lcg, net: &Network, labels: &[u32]) -> Samples {
    let dims: Vec<usize> = labels.iter().map(|l| net.sizes[l]).collect();
    let combos: usize = dims.iter().product();
    if combos <= 24 && rng.chance(0.3) {
        let mut order: Vec<usize> = (0..combos).collect();
        rng.shuffle(&mut order);
        let n = combos;
        let mut values = vec![0; n * labels.len()];
        for (s, &c) in order.iter().enumerate() {
            let mut rem = c;
            for (j, &d) in dims.iter().enumerate() {
                values[s + n * j] = rem % d;
                rem /= d;
            }
        }
        return Samples::new(labels.iter().copied(), n, values);
    }
    let n = rng.range(1, 12);
    let mut values = vec![0; n * labels.len()];
    for (j, &d) in dims.iter().enumerate() {
        for s in 0..n {
            values[s + n * j] = rng.range(0, d - 1);
        }
    }
    Samples::new(labels.iter().copied(), n, values)
}

/// A random SSA contraction path over `n_ops` leaves.
fn random_path(rng: &mut Lcg, n_ops: usize) -> Vec<(usize, usize)> {
    let mut live: Vec<usize> = (0..n_ops).collect();
    let mut path = Vec::new();
    for k in 0..n_ops.saturating_sub(1) {
        let a = live.swap_remove(rng.range(0, live.len() - 1));
        let b = live.swap_remove(rng.range(0, live.len() - 1));
        path.push((a, b));
        live.push(n_ops + k);
    }
    path
}

/// Random operands of `dtype` for `net`.
pub(crate) fn operands(rng: &mut Lcg, net: &Network, dtype: DType) -> Vec<Tensor> {
    net.shapes()
        .into_iter()
        .map(|shape| {
            let len: usize = shape.iter().product();
            match dtype {
                DType::F64 => Tensor::from_vec_col_major(
                    shape,
                    (0..len).map(|_| rng.unit()).collect::<Vec<f64>>(),
                ),
                DType::F32 => Tensor::from_vec_col_major(
                    shape,
                    (0..len).map(|_| rng.unit() as f32).collect::<Vec<f32>>(),
                ),
                DType::C64 => Tensor::from_vec_col_major(
                    shape,
                    (0..len)
                        .map(|_| Complex64::new(rng.unit(), rng.unit()))
                        .collect::<Vec<_>>(),
                ),
                DType::C32 => Tensor::from_vec_col_major(
                    shape,
                    (0..len)
                        .map(|_| Complex32::new(rng.unit() as f32, rng.unit() as f32))
                        .collect::<Vec<_>>(),
                ),
                _ => unreachable!("floating dtypes only"),
            }
            .unwrap()
        })
        .collect()
}

/// `Y[rest..., s]` read from a dense column-major output over `net.output`.
pub(crate) fn select_dense<T: Copy>(
    net: &Network,
    samples: &Samples,
    dense: &[T],
) -> (Vec<usize>, Vec<T>) {
    let out_dims: Vec<usize> = net.output.iter().map(|l| net.sizes[l]).collect();
    let column: HashMap<u32, usize> = samples
        .labels
        .iter()
        .enumerate()
        .map(|(j, &l)| (l, j))
        .collect();
    let rest: Vec<usize> = (0..net.output.len())
        .filter(|&k| !column.contains_key(&net.output[k]))
        .collect();
    let rest_dims: Vec<usize> = rest.iter().map(|&k| out_dims[k]).collect();
    let rest_size: usize = rest_dims.iter().product();
    let n = samples.n;
    let mut y = Vec::with_capacity(rest_size * n);
    for s in 0..n {
        for r in 0..rest_size {
            let mut idx = vec![0usize; out_dims.len()];
            let mut rem = r;
            for (&k, &d) in rest.iter().zip(&rest_dims) {
                idx[k] = rem % d;
                rem /= d;
            }
            for (k, l) in net.output.iter().enumerate() {
                if let Some(&j) = column.get(l) {
                    idx[k] = samples.value(s, j).expect("in range");
                }
            }
            let mut lin = 0;
            let mut stride = 1;
            for (k, &d) in out_dims.iter().enumerate() {
                lin += idx[k] * stride;
                stride *= d;
            }
            y.push(dense[lin]);
        }
    }
    let mut shape = rest_dims;
    shape.push(n);
    (shape, y)
}

trait Close: Copy {
    fn distance(self, other: Self) -> f64;
    fn magnitude(self) -> f64;
}
impl Close for f64 {
    fn distance(self, o: Self) -> f64 {
        (self - o).abs()
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
}
impl Close for f32 {
    fn distance(self, o: Self) -> f64 {
        f64::from((self - o).abs())
    }
    fn magnitude(self) -> f64 {
        f64::from(self.abs())
    }
}
impl Close for Complex64 {
    fn distance(self, o: Self) -> f64 {
        (self - o).norm()
    }
    fn magnitude(self) -> f64 {
        self.norm()
    }
}
impl Close for Complex32 {
    fn distance(self, o: Self) -> f64 {
        f64::from((self - o).norm())
    }
    fn magnitude(self) -> f64 {
        f64::from(self.norm())
    }
}

fn kind_name(kind: &SampledStepKind) -> String {
    match kind {
        SampledStepKind::Pairs { chunks } if *chunks > 1 => "Pairs (chunked)".to_string(),
        k => format!("{k:?}"),
    }
}

/// Run one case of `dtype` under every planner setting against the dense
/// reference; assert measured peak == planned peak; count the step kinds.
fn check<T: Close + tenferro_tensor::TensorScalar>(
    case: &Case,
    dtype: DType,
    tolerance: f64,
    rng: &mut Lcg,
    seen: &mut HashMap<String, usize>,
) {
    let mut engine = CpuEngine::serial();
    let tensors = operands(rng, &case.net, dtype);
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let full = refs.einsum(&case.net.subscripts(), &mut engine).unwrap();
    let (shape, want) = select_dense(&case.net, &case.samples, full.as_slice::<T>().unwrap());
    for (name, options) in settings() {
        let options = case.net.along(&case.path, options);
        let plan =
            SampledEinsumPlan::prepare(&refs, &case.net.subscripts(), &case.samples, &options)
                .unwrap_or_else(|e| panic!("{name}: {e}\n{case:?}"));
        assert_eq!(
            plan.report().pairs,
            case.path,
            "{name}: the plan follows the given path"
        );
        let (y, stats) = plan.execute_with_stats(&refs, &mut engine).unwrap();
        assert_eq!(y.shape(), shape.as_slice(), "{name}: {case:?}");
        for (k, (g, w)) in y.as_slice::<T>().unwrap().iter().zip(&want).enumerate() {
            assert!(
                g.distance(*w) <= tolerance * (1.0 + w.magnitude()),
                "{name}: entry {k}\n{case:?}"
            );
        }
        assert_eq!(
            stats.peak_elements,
            plan.report().peak_elements,
            "{name}: measured vs planned peak\n{case:?}"
        );
        assert_eq!(plan.output_shape(), shape.as_slice());
        for step in &plan.report().steps {
            *seen.entry(kind_name(&step.kind)).or_default() += 1;
        }
    }
}

#[test]
fn random_networks_f64_match_dense() {
    let mut rng = Lcg(2675);
    let mut seen = HashMap::new();
    for _ in 0..400 {
        let case = random_case(&mut rng);
        check::<f64>(&case, DType::F64, 1e-12, &mut rng, &mut seen);
    }
    for kind in [
        "Dense",
        "OneSided",
        "FullProduct",
        "OuterSelect",
        "Pairs { chunks: 1 }",
        "Pairs (chunked)",
    ] {
        assert!(
            seen.get(kind).copied().unwrap_or(0) > 0,
            "step kind {kind} never ran: {seen:?}"
        );
    }
}

#[test]
fn random_networks_c64_match_dense() {
    let mut rng = Lcg(1867);
    let mut seen = HashMap::new();
    for _ in 0..150 {
        let case = random_case(&mut rng);
        check::<Complex64>(&case, DType::C64, 1e-12, &mut rng, &mut seen);
    }
}

#[test]
fn random_networks_f32_and_c32_match_dense() {
    let mut rng = Lcg(32);
    let mut seen = HashMap::new();
    for _ in 0..80 {
        let case = random_case(&mut rng);
        check::<f32>(&case, DType::F32, 1e-5, &mut rng, &mut seen);
        let case = random_case(&mut rng);
        check::<Complex32>(&case, DType::C32, 1e-5, &mut rng, &mut seen);
    }
}

fn run_case(net: &Network, path: &[(usize, usize)], samples: &Samples, seed: u64) {
    let case = Case {
        net: net.clone(),
        path: path.to_vec(),
        samples: samples.clone(),
    };
    check::<f64>(
        &case,
        DType::F64,
        1e-12,
        &mut Lcg(seed),
        &mut HashMap::new(),
    );
}

pub(crate) fn sizes(pairs: &[(u32, usize)]) -> HashMap<u32, usize> {
    pairs.iter().copied().collect()
}

#[test]
fn sddmm_and_duplicates() {
    // ij,jk->ik at (i, k) = (1, 2), (0, 0), (1, 2): a duplicate sample.
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        sizes: sizes(&[(0, 3), (1, 4), (2, 5)]),
    };
    let samples = Samples::new([0, 2], 3, vec![1, 0, 1, 2, 0, 2]);
    run_case(&net, &[(0, 1)], &samples, 1);
}

#[test]
fn one_sample_and_no_samples() {
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        output: vec![0, 3],
        sizes: sizes(&[(0, 2), (1, 3), (2, 2), (3, 3)]),
    };
    run_case(&net, &[(0, 1), (3, 2)], &Samples::new([3], 1, vec![2]), 2);
    run_case(&net, &[(1, 2), (0, 3)], &Samples::new([3, 0], 0, vec![]), 3);
}

#[test]
fn sampled_label_on_several_operands() {
    // b is sampled and sits on all three operands; d is sampled too; a and c stay dense.
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2, 4], vec![1, 3, 4]],
        output: vec![0, 1, 2, 3],
        sizes: sizes(&[(0, 2), (1, 3), (2, 2), (3, 3), (4, 2)]),
    };
    let samples = Samples::new([1, 3], 5, vec![0, 2, 2, 1, 0, 1, 0, 1, 2, 1]);
    run_case(&net, &[(1, 2), (0, 3)], &samples, 4);
    run_case(&net, &[(0, 1), (3, 2)], &samples, 5);
}

#[test]
fn single_operand_with_a_summed_label() {
    // abc->ca sampled on a: c stays dense, b is summed.
    let net = Network {
        inputs: vec![vec![0, 1, 2]],
        output: vec![2, 0],
        sizes: sizes(&[(0, 4), (1, 3), (2, 2)]),
    };
    run_case(&net, &[], &Samples::new([0], 6, vec![3, 0, 0, 2, 1, 3]), 6);
}

#[test]
fn several_sampled_labels_on_one_leaf() {
    // The first operand carries both sampled labels, so its leaf merges two axes before the gather.
    let net = Network {
        inputs: vec![vec![0, 1, 2], vec![2, 3]],
        output: vec![3, 1, 0],
        sizes: sizes(&[(0, 3), (1, 2), (2, 4), (3, 2)]),
    };
    run_case(
        &net,
        &[(0, 1)],
        &Samples::new([1, 0], 4, vec![1, 0, 1, 0, 2, 2, 0, 1]),
        7,
    );
}

#[test]
fn the_full_product_needs_no_gather_and_a_tiny_cap_forces_chunks() {
    // Every (i, k) pair: the step is the full product.
    let mut rng = Lcg(11);
    let a = operands(
        &mut rng,
        &Network {
            inputs: vec![vec![0, 1]],
            output: vec![0],
            sizes: sizes(&[(0, 3), (1, 4)]),
        },
        DType::F64,
    )
    .remove(0);
    let b = operands(
        &mut rng,
        &Network {
            inputs: vec![vec![1, 2]],
            output: vec![2],
            sizes: sizes(&[(1, 4), (2, 2)]),
        },
        DType::F64,
    )
    .remove(0);
    let mut engine = CpuEngine::serial();
    let all = Samples::new(
        [u32::from(b'i'), u32::from(b'k')],
        6,
        vec![0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 1, 1],
    );
    let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &all, &SampleOptions::default())
        .unwrap();
    assert_eq!(plan.report().steps[0].kind, SampledStepKind::FullProduct);
    let dense = [&a, &b].einsum("ij,jk->ik", &mut engine).unwrap();
    let y = plan.execute(&[&a, &b], &mut engine).unwrap();
    assert_eq!(
        y.as_slice::<f64>().unwrap(),
        dense.as_slice::<f64>().unwrap()
    );

    // Three of six pairs with a cap of one row per chunk: three chunks.
    let some = Samples::new(
        [u32::from(b'i'), u32::from(b'k')],
        3,
        vec![2, 0, 1, 1, 0, 0],
    );
    let options = SampleOptions::default()
        .max_chunk_elements(1)
        .outer_select_ratio(0.0);
    let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &some, &options).unwrap();
    assert_eq!(
        plan.report().steps[0].kind,
        SampledStepKind::Pairs { chunks: 3 }
    );
    let (y, stats) = plan.execute_with_stats(&[&a, &b], &mut engine).unwrap();
    assert_eq!(stats.chunks, 3);
    let d = dense.as_slice::<f64>().unwrap();
    let got = y.as_slice::<f64>().unwrap();
    for (s, want) in [d[2 + 3], d[0], d[1]].iter().enumerate() {
        assert!((got[s] - want).abs() <= 1e-12 * (1.0 + want.abs()));
    }
}

#[test]
fn a_warm_rerun_prepares_no_new_plans() {
    // Pairs in five chunks of two and a shorter last one, on two operands
    // (the engine's binary contraction, whose plan cache keys on the
    // destination offset: rebased windows keep one plan per chunk size).
    let net = Network {
        inputs: vec![vec![0, 1, 3], vec![1, 2, 3]],
        output: vec![0, 2, 3],
        sizes: sizes(&[(0, 6), (1, 5), (2, 7), (3, 3)]),
    };
    let mut rng = Lcg(41);
    let tensors = operands(&mut rng, &net, DType::F64);
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let n = 11;
    let values: Vec<usize> = (0..n)
        .map(|s| s % 6)
        .chain((0..n).map(|s| (3 * s + 1) % 7))
        .collect();
    let samples = Samples::new([0, 2], n, values);
    let options = net.along(
        &[(0, 1)],
        SampleOptions::default()
            .max_chunk_elements(2 * 5 * 3)
            .outer_select_ratio(0.0),
    );
    let plan = SampledEinsumPlan::prepare(&refs, &net.subscripts(), &samples, &options).unwrap();
    assert_eq!(
        plan.report().steps[0].kind,
        SampledStepKind::Pairs { chunks: 6 }
    );
    let mut engine = CpuEngine::serial();
    let first = plan.execute(&refs, &mut engine).unwrap();
    // Engine plans follow the distinct shapes (a full and a shorter last
    // chunk), not the windows: six chunks miss as often as three.
    let cold_misses = |max_chunk_elements: usize| {
        let options = net.along(
            &[(0, 1)],
            SampleOptions::default()
                .max_chunk_elements(max_chunk_elements)
                .outer_select_ratio(0.0),
        );
        let plan =
            SampledEinsumPlan::prepare(&refs, &net.subscripts(), &samples, &options).unwrap();
        let mut engine = CpuEngine::serial();
        plan.execute(&refs, &mut engine).unwrap();
        (
            plan.report().steps[0].kind.clone(),
            engine.plan_cache().stats().misses,
        )
    };
    let (six, six_misses) = cold_misses(2 * 5 * 3);
    let (three, three_misses) = cold_misses(4 * 5 * 3);
    assert_eq!(
        (six, three),
        (
            SampledStepKind::Pairs { chunks: 6 },
            SampledStepKind::Pairs { chunks: 3 }
        )
    );
    assert_eq!(six_misses, three_misses);
    let cpu_misses = plan.cpu_cache_stats().unwrap().misses;
    let engine_misses = engine.plan_cache().stats().misses;
    let second = plan.execute(&refs, &mut engine).unwrap();
    assert_eq!(plan.cpu_cache_stats().unwrap().misses, cpu_misses);
    assert_eq!(engine.plan_cache().stats().misses, engine_misses);
    assert_eq!(
        first.as_slice::<f64>().unwrap(),
        second.as_slice::<f64>().unwrap()
    );
    let full = refs.einsum(&net.subscripts(), &mut engine).unwrap();
    let (_, want) = select_dense(&net, &samples, full.as_slice::<f64>().unwrap());
    for (g, w) in first.as_slice::<f64>().unwrap().iter().zip(&want) {
        assert!((g - w).abs() <= 1e-12 * (1.0 + w.abs()));
    }
}

#[test]
fn einsum_at_is_prepare_then_execute() {
    let mut rng = Lcg(5);
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2], vec![2, 3]],
        output: vec![0, 3],
        sizes: sizes(&[(0, 3), (1, 4), (2, 2), (3, 5)]),
    };
    let tensors = operands(&mut rng, &net, DType::F64);
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let samples = Samples::new([3, 0], 4, vec![4, 0, 4, 1, 2, 2, 2, 0]);
    let mut engine = CpuEngine::serial();
    let one_shot = refs
        .einsum_at(&net.subscripts(), &samples, &mut engine)
        .unwrap();
    let plan = SampledEinsumPlan::prepare(
        &refs,
        &net.subscripts(),
        &samples,
        &SampleOptions::default(),
    )
    .unwrap();
    let planned = plan.execute(&refs, &mut engine).unwrap();
    assert_eq!(
        one_shot.as_slice::<f64>().unwrap(),
        planned.as_slice::<f64>().unwrap()
    );
    let with = refs
        .einsum_at_with(
            &net.subscripts(),
            &samples,
            &SampleOptions::default().max_chunk_elements(1),
            &mut engine,
        )
        .unwrap();
    for (g, w) in with
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(planned.as_slice::<f64>().unwrap())
    {
        assert!((g - w).abs() <= 1e-12 * (1.0 + w.abs()));
    }
}

fn invalid_argument(error: &tenferro_tensor::Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    )
}

#[test]
fn invalid_inputs_are_rejected() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let plan = |spec: &str, samples: Samples| {
        SampledEinsumPlan::prepare(&[&a, &a], spec, &samples, &SampleOptions::default())
    };
    let i = u32::from(b'i');
    let j = u32::from(b'j');
    let k = u32::from(b'k');
    assert!(plan("ij,jk->ik", Samples::new([i], 1, vec![1])).is_ok());
    for (spec, samples) in [
        ("ij,jk->ik", Samples::new([j], 1, vec![0])), // not an output label
        ("ij,jk->ik", Samples::new([k], 1, vec![2])), // out of range
        ("ij,jk->ik", Samples::new([], 1, vec![])),   // nothing sampled
        ("ij,jk->ik", Samples::new([i, i], 1, vec![0, 0])), // repeated label
        ("ij,jk->ik", Samples::new([i], 2, vec![0])), // wrong value count
        ("ii,ik->ik", Samples::new([i], 1, vec![0])), // diagonal
    ] {
        let error = plan(spec, samples.clone()).expect_err(spec);
        assert!(invalid_argument(&error), "{spec} {samples:?}: {error}");
    }
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32; 4]).unwrap();
    assert!(SampledEinsumPlan::prepare(
        &[&a, &b],
        "ij,jk->ik",
        &Samples::new([i], 1, vec![0]),
        &SampleOptions::default()
    )
    .is_err());
    let plan = SampledEinsumPlan::prepare(
        &[&a, &a],
        "ij,jk->ik",
        &Samples::new([i], 1, vec![0]),
        &SampleOptions::default(),
    )
    .unwrap();
    let c = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    assert!(plan.execute(&[&a, &c], &mut CpuEngine::serial()).is_err());
}

#[test]
fn strided_operand_views_are_gathered_at_full_rank() {
    // A transposed view cannot be relabeled as [prefix, extent, suffix]; the
    // gather falls back to the full-rank configuration.
    let a = tenferro_tensor::TypedTensor::<f64>::from_vec_col_major(
        vec![3, 4],
        (0..12).map(f64::from).collect(),
    )
    .unwrap();
    let at = a.as_view().transpose_view([1, 0]).unwrap(); // shape [4, 3], at[j, i] = a[i, j]
    let samples = Samples::new([u32::from(b'i')], 3, vec![2, 0, 2]);
    let plan = SampledEinsumPlan::prepare(
        std::slice::from_ref(&at),
        "ji->ji",
        &samples,
        &SampleOptions::default(),
    )
    .unwrap();
    let y = plan.execute(&[at], &mut CpuEngine::serial()).unwrap();
    let a = a.host_data().unwrap();
    let want: Vec<f64> = [2usize, 0, 2]
        .iter()
        .flat_map(|&i| (0..4).map(move |j| (i, j)))
        .map(|(i, j)| a[i + 3 * j])
        .collect();
    assert_eq!(y.host_data().unwrap(), want.as_slice());
}

#[test]
fn a_zero_extent_is_rejected() {
    let a = Tensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap();
    let error = SampledEinsumPlan::prepare(
        &[&a],
        "i->i",
        &Samples::new([u32::from(b'i')], 0, vec![]),
        &SampleOptions::default(),
    )
    .expect_err("extent 0");
    assert!(invalid_argument(&error), "{error}");
}
