//! Sliced einsum (#2676): planning under a memory target and the slice loop,
//! checked against the unsliced contraction.

use std::error::Error as _;

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{
    ConjMask, ContractionTree, EinsumOps, EinsumOptimize, EinsumOptions, EinsumSubscripts,
    NestedEinsum, SliceOptions, SlicedEinsumPlan, Subscripts,
};
use tenferro_tensor::{Conj, DType, Dest, ErrorKind, Tensor, TypedTensor, ValidationKind};

/// A small deterministic generator (no `rand` dev-dependency).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit(&mut self) -> f64 {
        (self.next() % 2_000_001) as f64 / 1_000_000.0 - 1.0
    }
}

/// A random network: every label on at least one operand, some on three or
/// more (hyperedges), and an output of a few labels.
struct Network {
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    extents: Vec<usize>,
}

impl Network {
    fn random(rng: &mut Lcg) -> Self {
        let labels = 6 + rng.below(3);
        let operands = 4 + rng.below(3);
        let extents: Vec<usize> = (0..labels).map(|_| 2 + rng.below(3)).collect();
        let mut inputs: Vec<Vec<u32>> = vec![Vec::new(); operands];
        // A hyperedge: label 0 on three operands.
        for input in inputs.iter_mut().take(3) {
            input.push(0);
        }
        for label in 1..labels as u32 {
            let carriers = 1 + rng.below(3);
            for _ in 0..carriers {
                let k = rng.below(operands);
                if !inputs[k].contains(&label) {
                    inputs[k].push(label);
                }
            }
        }
        for input in &mut inputs {
            if input.is_empty() {
                input.push(1);
            }
        }
        let output: Vec<u32> = (1..labels as u32)
            .filter(|_| rng.below(4) == 0)
            .take(2)
            .collect();
        Self {
            inputs,
            output,
            extents,
        }
    }

    fn subscripts(&self) -> EinsumSubscripts {
        let inputs: Vec<&[u32]> = self.inputs.iter().map(Vec::as_slice).collect();
        EinsumSubscripts::new(&inputs, &self.output)
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        self.inputs
            .iter()
            .map(|labels| labels.iter().map(|&l| self.extents[l as usize]).collect())
            .collect()
    }

    fn contracted(&self) -> Vec<u32> {
        let mut labels: Vec<u32> = self.inputs.iter().flatten().copied().collect();
        labels.sort_unstable();
        labels.dedup();
        labels.retain(|l| !self.output.contains(l));
        labels
    }

    fn tensors(&self, dtype: DType, rng: &mut Lcg) -> Vec<Tensor> {
        self.shapes()
            .into_iter()
            .map(|shape| random_tensor(dtype, shape, rng))
            .collect()
    }
}

fn random_tensor(dtype: DType, shape: Vec<usize>, rng: &mut Lcg) -> Tensor {
    let n: usize = shape.iter().product();
    match dtype {
        DType::F64 => Tensor::from_vec_col_major(shape, (0..n).map(|_| rng.unit()).collect()),
        DType::F32 => Tensor::from_vec_col_major(
            shape,
            (0..n).map(|_| rng.unit() as f32).collect::<Vec<f32>>(),
        ),
        DType::C64 => Tensor::from_vec_col_major(
            shape,
            (0..n)
                .map(|_| Complex64::new(rng.unit(), rng.unit()))
                .collect::<Vec<_>>(),
        ),
        DType::C32 => Tensor::from_vec_col_major(
            shape,
            (0..n)
                .map(|_| Complex32::new(rng.unit() as f32, rng.unit() as f32))
                .collect::<Vec<_>>(),
        ),
        other => panic!("no generator for {other:?}"),
    }
    .expect("random operand")
}

/// Elements of any float or complex tensor as `Complex64`.
fn values(tensor: &Tensor) -> Vec<Complex64> {
    match tensor.dtype() {
        DType::F64 => tensor
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect(),
        DType::F32 => tensor
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&x| Complex64::new(f64::from(x), 0.0))
            .collect(),
        DType::C64 => tensor.as_slice::<Complex64>().unwrap().to_vec(),
        DType::C32 => tensor
            .as_slice::<Complex32>()
            .unwrap()
            .iter()
            .map(|x| Complex64::new(f64::from(x.re), f64::from(x.im)))
            .collect(),
        other => panic!("no reader for {other:?}"),
    }
}

/// `max |a - b| / max(1, max |b|)`.
fn relative_error(actual: &Tensor, expected: &Tensor) -> f64 {
    let (a, b) = (values(actual), values(expected));
    assert_eq!(a.len(), b.len(), "result sizes differ");
    let scale = b.iter().map(|x| x.norm()).fold(1.0, f64::max);
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
        / scale
}

fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F64 | DType::C64 => 1e-12,
        _ => 1e-5,
    }
}

fn label(c: char) -> u32 {
    u32::from(c)
}

fn einsum_error(error: &tenferro_tensor::Error) -> &tenferro_einsum::Error {
    error
        .source()
        .and_then(|source| source.downcast_ref::<tenferro_einsum::Error>())
        .expect("the einsum error is the typed source")
}

/// The tetrahedron `abc,cde,efa,bdf->`: any pairwise tree has an
/// intermediate of four labels (8^4 = 2^12 elements at extent 8).
fn tetrahedron(rng: &mut Lcg) -> Vec<Tensor> {
    (0..4)
        .map(|_| random_tensor(DType::F64, vec![8, 8, 8], rng))
        .collect()
}

const TETRAHEDRON: &str = "abc,cde,efa,bdf->";

#[test]
fn a_target_below_the_dense_intermediate_slices_and_matches_the_dense_plan() {
    let tensors = tetrahedron(&mut Lcg(7));
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let mut engine = CpuEngine::serial();
    let options = EinsumOptions::default();

    // A target the tree already meets: one slice, the dense contraction.
    let dense_plan = SlicedEinsumPlan::prepare(
        &inputs,
        TETRAHEDRON,
        &options,
        &SliceOptions::default().target_log2_elements(30.0),
    )
    .unwrap();
    assert_eq!(dense_plan.slice_count(), 1);
    assert!(dense_plan.report().sliced_labels.is_empty());
    let dense = dense_plan.execute(&inputs, &mut engine).unwrap();

    // 2^8 elements is far below the 2^12 the dense tree needs.
    let plan = SlicedEinsumPlan::prepare(
        &inputs,
        TETRAHEDRON,
        &options,
        &SliceOptions::default().target_log2_elements(8.0),
    )
    .unwrap();
    let report = plan.report();
    assert!(report.unsliced_largest_intermediate_elements > 256.0);
    assert!(report.slice_largest_intermediate_elements <= 256.0);
    assert!(report.slice_count > 1);
    assert_eq!(
        report.slice_count,
        report.slice_extents.iter().product::<usize>()
    );
    for l in &report.sliced_labels {
        assert!("abcdef".chars().map(label).any(|x| x == *l));
    }
    assert!(report.total_flops >= report.unsliced_flops);

    let sliced = plan.execute(&inputs, &mut engine).unwrap();
    let error = relative_error(&sliced, &dense);
    assert!(error <= 1e-12, "sliced vs dense: {error:e}");
    let reference = inputs.einsum(TETRAHEDRON, &mut engine).unwrap();
    assert!(relative_error(&sliced, &reference) <= 1e-12);

    // The executor's arena for one slice is below the dense one.
    let sliced_arena = plan.cpu_arena_elements().expect("an N-ary CPU slice ran");
    let dense_arena = dense_plan.cpu_arena_elements().expect("an N-ary CPU run");
    assert!(
        sliced_arena < dense_arena,
        "arena {sliced_arena} vs {dense_arena}"
    );
    assert!(sliced_arena as f64 <= report.slice_peak_intermediate_elements);
}

#[test]
fn an_unreachable_target_fails_before_execution() {
    let tensors = tetrahedron(&mut Lcg(3));
    let inputs: Vec<&Tensor> = tensors.iter().collect();

    // The output of `abc,cde,efa,bdf->ab` has 2^6 elements; only contracted
    // labels can be sliced, so a target of 2^4 cannot be met.
    let error = SlicedEinsumPlan::prepare(
        &inputs,
        "abc,cde,efa,bdf->ab",
        &EinsumOptions::default(),
        &SliceOptions::default().target_log2_elements(4.0),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    let tenferro_einsum::Error::MemoryTargetUnmet {
        target_log2_elements,
        reached_log2_elements,
        ..
    } = einsum_error(&error)
    else {
        panic!("expected MemoryTargetUnmet, got {error}");
    };
    assert_eq!(*target_log2_elements, 4.0);
    assert!((reached_log2_elements - 6.0).abs() < 1e-9);

    // A target below even a scalar output fails with an explicit order too.
    let path =
        EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(0, 1), (0, 1), (0, 1)]));
    let error = SlicedEinsumPlan::prepare(
        &inputs,
        TETRAHEDRON,
        &path,
        &SliceOptions::default().target_log2_elements(-1.0),
    )
    .unwrap_err();
    assert!(matches!(
        einsum_error(&error),
        tenferro_einsum::Error::MemoryTargetUnmet { .. }
    ));
}

#[test]
fn random_hyperedge_networks_match_the_unsliced_einsum_for_every_dtype() {
    let mut rng = Lcg(2676);
    let mut engine = CpuEngine::serial();
    let mut checked = 0;
    for _ in 0..12 {
        let network = Network::random(&mut rng);
        let subscripts = network.subscripts();
        let contracted = network.contracted();
        if contracted.is_empty() {
            continue;
        }
        for dtype in [DType::F64, DType::F32, DType::C64, DType::C32] {
            let tensors = network.tensors(dtype, &mut rng);
            let inputs: Vec<&Tensor> = tensors.iter().collect();
            let expected = inputs.einsum(&subscripts, &mut engine).unwrap();

            let one = vec![contracted[rng.below(contracted.len())]];
            let choices = [
                SliceOptions::default().labels(one),
                SliceOptions::default().labels(contracted.clone()),
            ];
            for slicing in &choices {
                let plan = SlicedEinsumPlan::prepare(
                    &inputs,
                    &subscripts,
                    &EinsumOptions::default(),
                    slicing,
                )
                .unwrap();
                let actual = plan.execute(&inputs, &mut engine).unwrap();
                let error = relative_error(&actual, &expected);
                assert!(
                    error <= tolerance(dtype),
                    "{dtype:?} {:?} labels {:?}: {error:e}",
                    network.inputs,
                    slicing.labels
                );
                checked += 1;
            }

            // The slicer under a target two bits below the unsliced tree.
            let probe = SlicedEinsumPlan::prepare(
                &inputs,
                &subscripts,
                &EinsumOptions::default(),
                &SliceOptions::default(),
            )
            .unwrap();
            let unsliced = probe.report().unsliced_largest_intermediate_elements;
            let output: f64 = network
                .output
                .iter()
                .map(|&l| network.extents[l as usize] as f64)
                .product();
            let target = unsliced.log2() - 2.0;
            if target >= output.log2() {
                match SlicedEinsumPlan::prepare(
                    &inputs,
                    &subscripts,
                    &EinsumOptions::default(),
                    &SliceOptions::default().target_log2_elements(target),
                ) {
                    Ok(plan) => {
                        assert!(
                            plan.report().slice_largest_intermediate_elements.log2()
                                <= target + 1e-9
                        );
                        let actual = plan.execute(&inputs, &mut engine).unwrap();
                        assert!(relative_error(&actual, &expected) <= tolerance(dtype));
                        checked += 1;
                    }
                    // A typed planning error is an allowed outcome; running
                    // past the target is not.
                    Err(error) => assert!(matches!(
                        einsum_error(&error),
                        tenferro_einsum::Error::MemoryTargetUnmet { .. }
                    )),
                }
            }
        }
    }
    assert!(checked >= 60, "only {checked} sliced plans checked");
}

#[test]
fn conjugation_is_applied_to_sliced_operands() {
    let mut rng = Lcg(11);
    let mut engine = CpuEngine::serial();
    let tensors: Vec<Tensor> = [vec![3, 4], vec![4, 5], vec![5, 3]]
        .into_iter()
        .map(|shape| random_tensor(DType::C64, shape, &mut rng))
        .collect();
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let options = EinsumOptions::default().conj(ConjMask::new([Conj::Yes, Conj::No, Conj::Yes]));
    let expected = inputs
        .einsum_with("ij,jk,ki->", &options, &mut engine)
        .unwrap();
    let plan = SlicedEinsumPlan::prepare(
        &inputs,
        "ij,jk,ki->",
        &options,
        &SliceOptions::default().labels([label('j'), label('i')]),
    )
    .unwrap();
    assert_eq!(plan.slice_count(), 12);
    let actual = plan.execute(&inputs, &mut engine).unwrap();
    assert!(relative_error(&actual, &expected) <= 1e-12);
}

#[test]
fn a_label_of_extent_one_can_be_sliced() {
    let mut rng = Lcg(5);
    let mut engine = CpuEngine::serial();
    // `j` has extent 1.
    let a = random_tensor(DType::F64, vec![3, 1, 4], &mut rng);
    let b = random_tensor(DType::F64, vec![1, 4, 2], &mut rng);
    let c = random_tensor(DType::F64, vec![4, 2], &mut rng);
    let inputs = [&a, &b, &c];
    let spec = "ijk,jkl,kl->i";
    let expected = inputs.einsum(spec, &mut engine).unwrap();
    for labels in [
        vec![label('j')],
        vec![label('k')],
        vec![label('j'), label('k'), label('l')],
    ] {
        let plan = SlicedEinsumPlan::prepare(
            &inputs,
            spec,
            &EinsumOptions::default(),
            &SliceOptions::default().labels(labels.clone()),
        )
        .unwrap();
        let actual = plan.execute(&inputs, &mut engine).unwrap();
        assert!(
            relative_error(&actual, &expected) <= 1e-12,
            "labels {labels:?}"
        );
    }
}

#[test]
fn an_explicit_order_is_kept_and_sliced_under_a_target() {
    let tensors = tetrahedron(&mut Lcg(17));
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let mut engine = CpuEngine::serial();
    let expected = inputs.einsum(TETRAHEDRON, &mut engine).unwrap();
    let subscripts = Subscripts::parse(TETRAHEDRON).unwrap();
    let shapes = [&[8, 8, 8][..], &[8, 8, 8], &[8, 8, 8], &[8, 8, 8]];
    let tree =
        ContractionTree::from_pairs(&subscripts, &shapes, &[(0, 1), (2, 3), (4, 5)]).unwrap();
    let orders = [
        EinsumOptimize::Path(vec![(0, 1), (0, 1), (0, 1)]),
        EinsumOptimize::Nested(NestedEinsum::parse("((abc,cde),(efa,bdf))->").unwrap()),
        EinsumOptimize::Tree(tree),
        EinsumOptimize::False,
    ];
    for order in orders {
        let options = EinsumOptions::default().optimize(order.clone());
        let unsliced =
            SlicedEinsumPlan::prepare(&inputs, TETRAHEDRON, &options, &SliceOptions::default())
                .unwrap();
        let plan = SlicedEinsumPlan::prepare(
            &inputs,
            TETRAHEDRON,
            &options,
            &SliceOptions::default().target_log2_elements(9.0),
        )
        .unwrap();
        let report = plan.report();
        // The order before slicing is the given one.
        assert_eq!(
            report.unsliced_flops,
            unsliced.report().unsliced_flops,
            "{order:?}"
        );
        assert!(
            report.slice_largest_intermediate_elements <= 512.0,
            "{order:?}"
        );
        let actual = plan.execute(&inputs, &mut engine).unwrap();
        assert!(relative_error(&actual, &expected) <= 1e-12, "{order:?}");

        // Explicit labels together with the explicit order.
        let plan = SlicedEinsumPlan::prepare(
            &inputs,
            TETRAHEDRON,
            &options,
            &SliceOptions::default().labels([label('c'), label('f')]),
        )
        .unwrap();
        assert_eq!(plan.slice_count(), 64);
        let actual = plan.execute(&inputs, &mut engine).unwrap();
        assert!(relative_error(&actual, &expected) <= 1e-12, "{order:?}");
    }
}

#[test]
fn invalid_sliced_labels_are_rejected() {
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3, 0], Vec::<f64>::new()).unwrap();
    let options = EinsumOptions::default();
    let cases: [(&str, Vec<u32>); 4] = [
        ("ij,jk->ik", vec![label('i')]),
        ("ij,jk->ik", vec![label('z')]),
        ("ij,jk->i", vec![label('j'), label('j')]),
        ("ij,jk->i", vec![label('k')]),
    ];
    for (spec, labels) in cases {
        let error = SlicedEinsumPlan::prepare(
            &[&a, &b],
            spec,
            &options,
            &SliceOptions::default().labels(labels.clone()),
        )
        .unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::Validation(ValidationKind::InvalidArgument),
            "{spec} {labels:?}: {error}"
        );
    }
    let error = SlicedEinsumPlan::prepare(
        &[&a, &a],
        "ij,ij->",
        &options,
        &SliceOptions::default().target_log2_elements(f64::NAN),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
}

#[test]
fn disjoint_ranges_sum_to_the_full_result_and_dest_modes_hold() {
    let tensors = tetrahedron(&mut Lcg(23));
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let mut engine = CpuEngine::serial();
    let plan = SlicedEinsumPlan::prepare(
        &inputs,
        "abc,cde,efa,bdf->a",
        &EinsumOptions::default(),
        &SliceOptions::default().labels([label('c'), label('d')]),
    )
    .unwrap();
    let count = plan.slice_count();
    assert_eq!(count, 64);
    let full = plan.execute(&inputs, &mut engine).unwrap();

    // Ranges into separate accumulators, summed.
    let bounds = [0, 5, 31, 32, count];
    let mut total = vec![0.0; 8];
    for window in bounds.windows(2) {
        let mut part = Tensor::zeros(DType::F64, vec![8]).unwrap();
        plan.execute_range_into(
            &inputs,
            window[0]..window[1],
            &mut engine,
            Dest::accumulate(&mut part),
        )
        .unwrap();
        for (t, p) in total.iter_mut().zip(part.as_slice::<f64>().unwrap()) {
            *t += p;
        }
    }
    let sum = Tensor::from_vec_col_major(vec![8], total).unwrap();
    assert!(relative_error(&sum, &full) <= 1e-12);

    // Ascending ranges into one destination are bit-identical to execute_into.
    let mut whole = Tensor::zeros(DType::F64, vec![8]).unwrap();
    plan.execute_into(&inputs, &mut engine, Dest::overwrite(&mut whole))
        .unwrap();
    let mut stepped = Tensor::from_vec_col_major(vec![8], vec![f64::NAN; 8]).unwrap();
    plan.execute_range_into(&inputs, 0..5, &mut engine, Dest::overwrite(&mut stepped))
        .unwrap();
    plan.execute_range_into(
        &inputs,
        5..count,
        &mut engine,
        Dest::accumulate(&mut stepped),
    )
    .unwrap();
    assert_eq!(
        stepped.as_slice::<f64>().unwrap(),
        whole.as_slice::<f64>().unwrap()
    );
    assert_eq!(
        whole.as_slice::<f64>().unwrap(),
        full.as_slice::<f64>().unwrap()
    );

    // Scaled: out = 2 r + 3 out.
    let mut scaled = Tensor::from_vec_col_major(vec![8], vec![1.0; 8]).unwrap();
    plan.execute_into(&inputs, &mut engine, Dest::scaled(&mut scaled, 2.0, 3.0))
        .unwrap();
    for (s, f) in scaled
        .as_slice::<f64>()
        .unwrap()
        .iter()
        .zip(full.as_slice::<f64>().unwrap())
    {
        assert!((s - (2.0 * f + 3.0)).abs() <= 1e-12 * f.abs().max(1.0));
    }

    // Range and destination validation happen before any write.
    let mut untouched = Tensor::from_vec_col_major(vec![8], vec![4.0; 8]).unwrap();
    for range in [0..count + 1, 3..3] {
        assert!(plan
            .execute_range_into(&inputs, range, &mut engine, Dest::overwrite(&mut untouched))
            .is_err());
    }
    plan.execute_range_into(&inputs, 3..3, &mut engine, Dest::accumulate(&mut untouched))
        .unwrap();
    let mut wrong = Tensor::zeros(DType::F64, vec![7]).unwrap();
    assert!(plan
        .execute_into(&inputs, &mut engine, Dest::overwrite(&mut wrong))
        .is_err());
    assert_eq!(untouched.as_slice::<f64>().unwrap(), &[4.0; 8]);
}

#[test]
fn a_warm_slice_loop_prepares_no_new_plans() {
    let tensors = tetrahedron(&mut Lcg(29));
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let mut engine = CpuEngine::serial();
    let plan = SlicedEinsumPlan::prepare(
        &inputs,
        TETRAHEDRON,
        &EinsumOptions::default(),
        &SliceOptions::default().labels([label('a'), label('c')]),
    )
    .unwrap();
    let slices = plan.slice_count() as u64;
    plan.execute(&inputs, &mut engine).unwrap();
    let stats = plan.cpu_cache_stats().unwrap();
    assert_eq!((stats.misses, stats.hits), (1, slices - 1));
    let engine_misses = engine.plan_cache().stats().misses;
    plan.execute(&inputs, &mut engine).unwrap();
    let stats = plan.cpu_cache_stats().unwrap();
    assert_eq!((stats.misses, stats.hits), (1, 2 * slices - 1));
    assert_eq!(engine.plan_cache().stats().misses, engine_misses);

    // Two operands take the engine's binary contraction, whose plan cache
    // keys on the view offset: rebased slice views keep one plan.
    let mut rng = Lcg(31);
    let a = random_tensor(DType::F64, vec![6, 16, 5], &mut rng);
    let b = random_tensor(DType::F64, vec![16, 5, 7], &mut rng);
    let plan = SlicedEinsumPlan::prepare(
        &[&a, &b],
        "ijk,jkl->il",
        &EinsumOptions::default(),
        &SliceOptions::default().labels([label('j')]),
    )
    .unwrap();
    let before = engine.plan_cache().stats().misses;
    let sliced = plan.execute(&[&a, &b], &mut engine).unwrap();
    let first = engine.plan_cache().stats().misses - before;
    assert!(first <= 2, "{first} plan misses over 16 slices");
    plan.execute(&[&a, &b], &mut engine).unwrap();
    assert_eq!(engine.plan_cache().stats().misses - before, first);
    let expected = [&a, &b].einsum("ijk,jkl->il", &mut engine).unwrap();
    assert!(relative_error(&sliced, &expected) <= 1e-12);
}

#[test]
fn slicing_is_bit_identical_where_summation_order_does_not_matter() {
    // Integer-valued data: every partial sum is exact in f64.
    let mut rng = Lcg(37);
    let tensors: Vec<Tensor> = (0..4)
        .map(|_| {
            let data = (0..512).map(|_| (rng.below(7) as f64) - 3.0).collect();
            Tensor::from_vec_col_major(vec![8, 8, 8], data).unwrap()
        })
        .collect();
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let mut engine = CpuEngine::serial();
    let dense = inputs.einsum("abc,cde,efa,bdf->b", &mut engine).unwrap();
    let plan = SlicedEinsumPlan::prepare(
        &inputs,
        "abc,cde,efa,bdf->b",
        &EinsumOptions::default(),
        &SliceOptions::default().target_log2_elements(7.0),
    )
    .unwrap();
    assert!(plan.slice_count() > 1);
    let sliced = plan.execute(&inputs, &mut engine).unwrap();
    assert_eq!(
        sliced.as_slice::<f64>().unwrap(),
        dense.as_slice::<f64>().unwrap()
    );

    // Random data: repeated runs of one plan are bit-identical.
    let tensors = tetrahedron(&mut Lcg(41));
    let inputs: Vec<&Tensor> = tensors.iter().collect();
    let plan = SlicedEinsumPlan::prepare(
        &inputs,
        TETRAHEDRON,
        &EinsumOptions::default(),
        &SliceOptions::default().labels([label('b'), label('e')]),
    )
    .unwrap();
    let first = plan.execute(&inputs, &mut engine).unwrap();
    let second = plan.execute(&inputs, &mut engine).unwrap();
    assert_eq!(
        first.as_slice::<f64>().unwrap()[0].to_bits(),
        second.as_slice::<f64>().unwrap()[0].to_bits()
    );
}

#[test]
fn typed_inputs_give_a_typed_result() {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0; 6]).unwrap();
    let plan = SlicedEinsumPlan::prepare(
        &[&a, &b],
        "ij,jk->ik",
        &EinsumOptions::default(),
        &SliceOptions::default().labels([label('j')]),
    )
    .unwrap();
    let out: TypedTensor<f64> = plan.execute(&[&a, &b], &mut CpuEngine::serial()).unwrap();
    assert_eq!(out.host_data().unwrap(), &[9.0, 12.0, 9.0, 12.0]);
}

#[test]
fn a_slice_count_that_overflows_is_a_planning_error() {
    // Four labels of extent 2^16: 2^64 slices.
    let v = Tensor::from_vec_col_major(vec![1 << 16], vec![1.0_f64; 1 << 16]).unwrap();
    let inputs = [&v, &v, &v, &v, &v, &v, &v, &v];
    let error = SlicedEinsumPlan::prepare(
        &inputs,
        "a,a,b,b,c,c,d,d->",
        &EinsumOptions::default(),
        &SliceOptions::default().labels("abcd".chars().map(u32::from)),
    )
    .unwrap_err();
    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument),
        "{error}"
    );
}
