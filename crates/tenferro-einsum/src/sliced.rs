//! Sliced einsum (#2676): a memory target that planning honours, and a slice
//! loop that runs one prepared contraction tree per assignment of the sliced
//! labels.
//!
//! Slicing fixes a few *contracted* labels, contracts the rest of the network
//! once for every assignment of them, and sums the results. It trades extra
//! flops for a bound on the largest intermediate of one slice.
//!
//! - **Planning** ([`SlicedEinsumPlan::prepare`]) resolves the contraction
//!   order from [`EinsumOptions::optimize`], then, when a memory target is set
//!   and the tree misses it, runs omeco's `slice_code` with
//!   `TreeSASlicer::fast().with_sc_target(target)`. With the automatic order
//!   the slicer may also reshape the tree; an explicit `Tree`, `Nested`,
//!   `Path` or `False` order is kept exactly and only labels are chosen.
//!   omeco 0.2.6 cannot exclude labels, so when its slicer picks an output
//!   label the slicer is rerun on the network without output labels, with
//!   the target lowered by log2 of the output's size (a tensor holds at most
//!   every output label). A plan that still misses the target is a
//!   [`Error::MemoryTargetUnmet`](crate::Error::MemoryTargetUnmet) before
//!   anything runs.
//! - **Execution** takes, per slice, metadata-only views of the operands that
//!   carry a sliced label (the sliced axis kept with extent 1), runs the one
//!   prepared [`ConcreteEinsumPlan`] for one slice, and accumulates into the
//!   destination. Host views are rebased so every slice presents the same
//!   layout (offset 0) to the engine, and the plan's CPU numerical plan is
//!   prepared once for all slices.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use omeco::{EinCode as OmecoEinCode, NestedEinsum as OmecoNested, TreeSASlicer};
use tenferro_tensor::{
    CacheStats, Conj, ContractionScalar, DType, Dest, DestMode, Engine, Repr, StridedSliceSpec,
    Tensor, TensorRead, TensorView, TensorWrite, TypedTensorView,
};

use crate::api::concrete::{check_mask, conjugate, resolve, substituted, tensor_err};
use crate::api::{ConjMask, EinsumInputs, EinsumOptions, EinsumSpec};
use crate::chunked::{run_chunks, ChunkSource};
use crate::concrete::read_input_specs;
use crate::{ConcreteEinsumPlan, ContractionTree, EinsumOptimize, Error, Subscripts};

type Res<T> = tenferro_tensor::Result<T>;

const OP: &str = "SlicedEinsumPlan";

/// Tolerance on log2 element counts when comparing against the target, so a
/// target of exactly `log2(n)` admits an intermediate of `n` elements.
const LOG2_TOLERANCE: f64 = 1e-9;

/// How a [`SlicedEinsumPlan`] chooses its sliced labels.
///
/// The memory target is **log2 of an element count**, the unit of omeco's
/// `sc_target`: it bounds the largest intermediate (any step result,
/// including the per-slice result) of one slice. Elements rather than bytes
/// because the bound is a property of the contraction tree, independent of
/// dtype, and because it is what the slicer optimizes; for a byte budget `b`
/// and element size `s`, use `(b / s).log2()`. The operands themselves are
/// caller-owned and not counted.
///
/// With neither a target nor labels the plan has one slice and runs the
/// network unsliced.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::SliceOptions;
///
/// let options = SliceOptions::default()
///     .target_log2_elements(20.0)
///     .labels([u32::from(b'k')]);
/// assert_eq!(options.target_log2_elements, Some(20.0));
/// assert_eq!(options.labels, vec![u32::from(b'k')]);
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct SliceOptions {
    /// log2 of the most elements one intermediate of one slice may hold.
    /// `None` keeps only [`Self::labels`].
    pub target_log2_elements: Option<f64>,
    /// Labels that are always sliced. A text label `'k'` is `u32::from(b'k')`
    /// (`'k' as u32`). Only contracted labels can be sliced.
    pub labels: Vec<u32>,
}

impl SliceOptions {
    /// Return options with the given memory target (log2 elements).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::SliceOptions;
    ///
    /// let o = SliceOptions::default().target_log2_elements(10.0);
    /// assert_eq!(o.target_log2_elements, Some(10.0));
    /// ```
    #[must_use]
    pub fn target_log2_elements(mut self, target_log2_elements: f64) -> Self {
        self.target_log2_elements = Some(target_log2_elements);
        self
    }

    /// Return options that always slice `labels`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::SliceOptions;
    ///
    /// let o = SliceOptions::default().labels("jk".chars().map(u32::from));
    /// assert_eq!(o.labels, vec![u32::from('j'), u32::from('k')]);
    /// ```
    #[must_use]
    pub fn labels(mut self, labels: impl IntoIterator<Item = u32>) -> Self {
        self.labels = labels.into_iter().collect();
        self
    }
}

/// What a [`SlicedEinsumPlan`] will do, known before anything runs.
///
/// Counts are `f64` because the unsliced counts of a network worth slicing
/// can exceed `usize`. "Intermediates" are the results of the pairwise steps
/// (the last one is the per-slice result); operands are not counted.
/// "Peak" is the most intermediate elements alive at once when the steps run
/// in order and each intermediate is released after its consumer.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![4, 8], vec![1.0_f64; 32])?;
/// let b = Tensor::from_vec_col_major(vec![8, 4], vec![1.0_f64; 32])?;
/// let plan = SlicedEinsumPlan::prepare(
///     &[&a, &b],
///     "ij,jk->ik",
///     &EinsumOptions::default(),
///     &SliceOptions::default().labels([u32::from(b'j')]),
/// )?;
/// let report = plan.report();
/// assert_eq!(report.sliced_labels, vec![u32::from(b'j')]);
/// assert_eq!(report.slice_count, 8);
/// assert_eq!(report.slice_flops, 16.0);
/// assert_eq!(report.total_flops, report.unsliced_flops);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SlicedEinsumReport {
    /// The sliced labels; slice ids count through them with the first fastest.
    pub sliced_labels: Vec<u32>,
    /// Extent of each sliced label.
    pub slice_extents: Vec<usize>,
    /// Number of slices: the product of [`Self::slice_extents`].
    pub slice_count: usize,
    /// The memory target the plan was checked against, if any.
    pub target_log2_elements: Option<f64>,
    /// Elements of the largest intermediate of one slice.
    pub slice_largest_intermediate_elements: f64,
    /// Peak intermediate elements alive at once during one slice.
    pub slice_peak_intermediate_elements: f64,
    /// Multiply-adds of one slice (per step, the product of the extents of
    /// every label of its two operands).
    pub slice_flops: f64,
    /// `slice_flops * slice_count`.
    pub total_flops: f64,
    /// Elements of the largest intermediate of the tree chosen before slicing.
    pub unsliced_largest_intermediate_elements: f64,
    /// Peak intermediate elements of the tree chosen before slicing.
    pub unsliced_peak_intermediate_elements: f64,
    /// Multiply-adds of the tree chosen before slicing, run unsliced.
    pub unsliced_flops: f64,
}

/// A concrete einsum planned under a memory target: the labels to slice, one
/// contraction tree for every slice, and that tree prepared once.
///
/// Opt-in: an ordinary [`EinsumOps::einsum`](crate::EinsumOps) call never
/// slices. Execution follows the concrete plan grammar (`execute`,
/// `execute_into` with every [`Dest`] mode) and adds
/// [`Self::execute_range_into`], which runs a range of slice ids into a
/// caller-owned destination so that disjoint ranges can be partitioned across
/// callers and summed.
///
/// # Numerics
///
/// The sum over the sliced labels moves to the outermost loop, so the result
/// equals the unsliced contraction up to floating-point summation order; it
/// is bit-identical only where that order does not matter (for example exact
/// integer-valued data). Running the same plan twice, or the slices one range
/// at a time into the same destination in ascending order, is bit-identical
/// to [`Self::execute_into`].
///
/// # Determinism
///
/// Explicit [`SliceOptions::labels`] with any order, or a target the tree
/// already meets, give the same plan every time. When omeco's slicer chooses
/// the labels, omeco 0.2.6 breaks ties between equally useful labels in
/// `HashMap` order, so two plannings may slice different labels; every such
/// plan is checked against the target.
///
/// # Scope (v0)
///
/// Concrete layer, any engine; only contracted labels are sliced. Traced and
/// eager (AD) execution are not provided.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuEngine;
/// use tenferro_einsum::{EinsumOps, EinsumOptions, SliceOptions, SlicedEinsumPlan};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![4, 8], (0..32).map(f64::from).collect())?;
/// let b = Tensor::from_vec_col_major(vec![8, 8], (0..64).map(f64::from).collect())?;
/// let c = Tensor::from_vec_col_major(vec![8, 4], (0..32).map(f64::from).collect())?;
/// let inputs = [&a, &b, &c];
/// let mut engine = CpuEngine::serial();
///
/// // The unsliced tree keeps a 4 x 8 intermediate (2^5 elements); ask for 2^2.
/// let plan = SlicedEinsumPlan::prepare(
///     &inputs,
///     "ij,jk,kl->il",
///     &EinsumOptions::default(),
///     &SliceOptions::default().target_log2_elements(4.0),
/// )?;
/// assert!(plan.report().slice_largest_intermediate_elements <= 16.0);
/// assert!(plan.slice_count() > 1);
///
/// let sliced = plan.execute(&inputs, &mut engine)?;
/// let dense = inputs.einsum("ij,jk,kl->il", &mut engine)?;
/// assert_eq!(sliced.as_slice::<f64>()?, dense.as_slice::<f64>()?);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct SlicedEinsumPlan {
    /// The full operands' dtypes and shapes.
    inputs: Vec<(DType, Vec<usize>)>,
    output_shape: Vec<usize>,
    dtype: DType,
    conj: ConjMask,
    /// Extent of each sliced label.
    extents: Vec<usize>,
    /// Per operand: `(axis, sliced label index)` of each axis that carries a
    /// sliced label with its full extent (a broadcast axis of extent 1 is
    /// left alone).
    fixes: Vec<Vec<(usize, usize)>>,
    /// The plan for one slice, prepared once.
    inner: ConcreteEinsumPlan,
    report: SlicedEinsumReport,
}

impl SlicedEinsumPlan {
    /// Plan a sliced einsum from the inputs' dtypes and shapes; element data
    /// is not read.
    ///
    /// The order comes from `options.optimize`. With a target, a tree that
    /// misses it is sliced by omeco's slicer (fast preset), which may also
    /// reshape an automatic tree; an explicit order is kept as given.
    /// `options.conj` is recorded for every execution.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumOptimize, EinsumOptions, SliceOptions, SlicedEinsumPlan};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 6], vec![1.0_f64; 12])?;
    /// let b = Tensor::from_vec_col_major(vec![6, 6], vec![1.0_f64; 36])?;
    /// let c = Tensor::from_vec_col_major(vec![6, 2], vec![1.0_f64; 12])?;
    /// // An explicit order together with explicit labels.
    /// let options = EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(1, 2), (0, 1)]));
    /// let plan = SlicedEinsumPlan::prepare(
    ///     &[&a, &b, &c],
    ///     "ij,jk,kl->il",
    ///     &options,
    ///     &SliceOptions::default().labels([u32::from(b'j')]),
    /// )?;
    /// assert_eq!(plan.slice_count(), 6);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// - `Validation` for a malformed spec, mismatched ranks or extents, an
    ///   invalid order, a conjugation mask longer than the inputs, a NaN
    ///   target, a sliced label that is unknown, repeated, an output label
    ///   or of extent 0, or a slice count that overflows `usize`;
    /// - [`Error::MemoryTargetUnmet`](crate::Error::MemoryTargetUnmet) (kind
    ///   `Validation(InvalidArgument)`) when no plan meets the target: the
    ///   output alone exceeds it, the slicer chose an output label, or the
    ///   sliced tree still exceeds it.
    pub fn prepare<I: EinsumInputs + ?Sized>(
        inputs: &I,
        spec: &(impl EinsumSpec + ?Sized),
        options: &EinsumOptions,
        slicing: &SliceOptions,
    ) -> Res<Self> {
        let reads = inputs.reads();
        check_mask(&options.conj, reads.len())?;
        let subscripts = resolve(spec.spec(), &reads).map_err(tensor_err)?;
        Self::plan(&reads, &subscripts, options, slicing).map_err(tensor_err)
    }

    fn plan(
        reads: &[TensorRead<'_>],
        subscripts: &Subscripts,
        options: &EinsumOptions,
        slicing: &SliceOptions,
    ) -> crate::Result<Self> {
        if reads.is_empty() {
            return Err(Error::invalid_argument(
                OP,
                "inputs",
                "einsum requires at least one input tensor",
            ));
        }
        let shapes: Vec<&[usize]> = reads.iter().map(TensorRead::shape).collect();
        let (_, tree) = crate::optimize::resolve_einsum_strategy_with_spec(
            options.optimize.clone(),
            subscripts,
            &shapes,
        )?;
        let sizes = tree.size_dict.clone();
        let original_pairs = crate::optimize::tree_pairs(&tree);
        let reshapes_tree = matches!(options.optimize, EinsumOptimize::Auto(_));

        let mut labels = validated_labels(&slicing.labels, subscripts, &sizes)?;
        let target = slicing.target_log2_elements;
        if target.is_some_and(f64::is_nan) {
            return Err(Error::invalid_argument(
                OP,
                "target_log2_elements",
                "the memory target must not be NaN",
            ));
        }

        let unsliced = tree_cost(&tree, |label| sizes[&label] as f64);
        let mut pairs = original_pairs.clone();
        if let Some(target) = target {
            let output_elements = elements(&subscripts.output, |label| sizes[&label] as f64);
            if output_elements.log2() > target + LOG2_TOLERANCE {
                return Err(target_unmet(
                    target,
                    output_elements,
                    &labels,
                    "the output alone exceeds it, and only contracted labels can be sliced",
                ));
            }
            let fixed = sliced_size(&sizes, &labels);
            let with_fixed = tree_cost(&tree, |label| fixed[&label] as f64);
            if with_fixed.largest.log2() > target + LOG2_TOLERANCE && reads.len() > 1 {
                let output = &subscripts.output;
                let mut sliced =
                    run_slicer(&tree, subscripts, &fixed, target, reshapes_tree, false);
                if sliced
                    .as_ref()
                    .is_some_and(|s| s.slicing.iter().any(|l| output.contains(l)))
                {
                    // omeco 0.2.6 cannot exclude labels from slicing. Hide the
                    // output labels from it instead, and lower the target by
                    // the output's size: a tensor holds at most every output
                    // label, so its true size is at most its size without
                    // them times the output's size.
                    let stripped_target = target - output_elements.log2();
                    sliced = run_slicer(
                        &tree,
                        subscripts,
                        &fixed,
                        stripped_target,
                        reshapes_tree,
                        true,
                    );
                }
                let sliced = sliced.ok_or_else(|| {
                    target_unmet(
                        target,
                        with_fixed.largest,
                        &labels,
                        "omeco's slicer returned no plan",
                    )
                })?;
                for label in sliced.slicing {
                    if labels.contains(&label) || sizes.get(&label).is_none_or(|&e| e <= 1) {
                        continue;
                    }
                    labels.push(label);
                    if output.contains(&label) {
                        return Err(target_unmet(
                            target,
                            with_fixed.largest,
                            &labels,
                            "omeco's slicer chose an output label, and only contracted labels can be sliced",
                        ));
                    }
                }
                if reshapes_tree {
                    pairs = omeco_pairs(&sliced.eins, subscripts.inputs.len())?;
                }
            }
        }

        let per_slice_sizes = sliced_size(&sizes, &labels);
        let per_slice_shapes: Vec<Vec<usize>> = subscripts
            .inputs
            .iter()
            .zip(&shapes)
            .map(|(axes, shape)| {
                axes.iter()
                    .zip(shape.iter())
                    .map(
                        |(label, &extent)| {
                            if labels.contains(label) {
                                1
                            } else {
                                extent
                            }
                        },
                    )
                    .collect()
            })
            .collect();
        let shape_refs: Vec<&[usize]> = per_slice_shapes.iter().map(Vec::as_slice).collect();
        let slice_tree = ContractionTree::from_pairs(subscripts, &shape_refs, &pairs)?;
        let per_slice = tree_cost(&slice_tree, |label| per_slice_sizes[&label] as f64);
        if let Some(target) = target {
            if per_slice.largest.log2() > target + LOG2_TOLERANCE {
                return Err(target_unmet(
                    target,
                    per_slice.largest,
                    &labels,
                    "the sliced tree still exceeds it",
                ));
            }
        }

        let extents: Vec<usize> = labels.iter().map(|label| sizes[label]).collect();
        let slice_count = extents
            .iter()
            .try_fold(1_usize, |count, &extent| count.checked_mul(extent))
            .ok_or_else(|| {
                Error::invalid_argument(
                    OP,
                    "labels",
                    format!("the slice count, the product of extents {extents:?}, overflows usize"),
                )
            })?;
        let fixes: Vec<Vec<(usize, usize)>> = subscripts
            .inputs
            .iter()
            .zip(&shapes)
            .map(|(axes, shape)| {
                axes.iter()
                    .enumerate()
                    .filter_map(|(axis, label)| {
                        let index = labels.iter().position(|l| l == label)?;
                        (shape[axis] > 1).then_some((axis, index))
                    })
                    .collect()
            })
            .collect();

        // Slice 0 has the dtypes and shapes of every slice.
        let first = slice_reads(reads, &fixes, &vec![0; labels.len()])?;
        let inner = ConcreteEinsumPlan::prepare_planned(
            read_input_specs(&first),
            subscripts,
            &EinsumOptimize::Tree(slice_tree),
            ConjMask::default(),
        )?;
        let output_shape = subscripts.output.iter().map(|label| sizes[label]).collect();
        let report = SlicedEinsumReport {
            sliced_labels: labels,
            slice_extents: extents.clone(),
            slice_count,
            target_log2_elements: target,
            slice_largest_intermediate_elements: per_slice.largest,
            slice_peak_intermediate_elements: per_slice.peak,
            slice_flops: per_slice.flops,
            total_flops: per_slice.flops * slice_count as f64,
            unsliced_largest_intermediate_elements: unsliced.largest,
            unsliced_peak_intermediate_elements: unsliced.peak,
            unsliced_flops: unsliced.flops,
        };
        Ok(Self {
            inputs: reads
                .iter()
                .map(|read| (read.dtype(), read.shape().to_vec()))
                .collect(),
            output_shape,
            dtype: reads[0].dtype(),
            conj: options.conj.clone(),
            extents,
            fixes,
            inner,
            report,
        })
    }

    /// What the plan does: sliced labels, slice count, per-slice bound and
    /// flops, against the tree before slicing.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3])?;
    /// let plan = SlicedEinsumPlan::prepare(&[&a, &a], "i,i->", &EinsumOptions::default(), &SliceOptions::default())?;
    /// assert!(plan.report().sliced_labels.is_empty());
    /// assert_eq!(plan.report().slice_count, 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn report(&self) -> &SlicedEinsumReport {
        &self.report
    }

    /// Number of slices; slice ids are `0..slice_count()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3])?;
    /// let options = SliceOptions::default().labels([u32::from(b'i')]);
    /// let plan = SlicedEinsumPlan::prepare(&[&a, &a], "i,i->", &EinsumOptions::default(), &options)?;
    /// assert_eq!(plan.slice_count(), 3);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn slice_count(&self) -> usize {
        self.report.slice_count
    }

    /// Run every slice and return the summed result, which takes the inputs'
    /// representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0])?;
    /// let options = SliceOptions::default().labels([u32::from(b'i')]);
    /// let plan = SlicedEinsumPlan::prepare(&[&a, &a], "i,i->", &EinsumOptions::default(), &options)?;
    /// let dot: TypedTensor<f64> = plan.execute(&[&a, &a], &mut CpuEngine::serial())?;
    /// assert_eq!(dot.host_data()?, &[14.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `Validation` when the inputs' count, dtypes or shapes differ from the
    /// prepared ones; `Unsupported` for an input without a borrowed view;
    /// backend failures otherwise.
    pub fn execute<I: EinsumInputs + ?Sized>(
        &self,
        inputs: &I,
        exec: &mut dyn Engine,
    ) -> Res<<I::Repr as Repr>::Value> {
        let reads = inputs.reads();
        self.check_inputs(&reads)?;
        let conj = conjugate(&reads, &self.conj, exec)?;
        let reads = substituted(&reads, &conj);
        let first =
            slice_reads(&reads, &self.fixes, &vec![0; self.extents.len()]).map_err(tensor_err)?;
        let mut out: Tensor = self.inner.run_reads(&first[..], exec).map_err(tensor_err)?;
        let add = tenferro_tensor::DotGeneralAccumulation::add_to(self.dtype)?;
        let source = Slices {
            plan: self,
            reads: &reads,
        };
        run_chunks(
            &source,
            1..self.slice_count(),
            exec,
            &mut TensorWrite::from_tensor(&mut out),
            add,
            add,
        )
        .map_err(tensor_err)?;
        <I::Repr as Repr>::wrap_value(out)
    }

    /// Run every slice into `dest`: `Overwrite`, `Accumulate` (`out += r`) or
    /// `Scaled` (`out = alpha r + beta out`), where `r` is the sum over all
    /// slices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
    /// use tenferro_tensor::{Dest, Tensor};
    ///
    /// let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    /// let options = SliceOptions::default().labels([u32::from(b'i')]);
    /// let plan = SlicedEinsumPlan::prepare(&[&a, &a], "i,i->", &EinsumOptions::default(), &options)?;
    /// let mut out = Tensor::from_vec_col_major(Vec::<usize>::new(), vec![1.0_f64])?;
    /// plan.execute_into(&[&a, &a], &mut CpuEngine::serial(), Dest::scaled(&mut out, 2.0, 1.0))?;
    /// assert_eq!(out.as_slice::<f64>()?, &[29.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Self::execute_range_into`] over all slices.
    pub fn execute_into<I: EinsumInputs + ?Sized>(
        &self,
        inputs: &I,
        exec: &mut dyn Engine,
        dest: Dest<<I::Repr as Repr>::Write<'_>>,
    ) -> Res<()> {
        self.execute_range_into(inputs, 0..self.slice_count(), exec, dest)
    }

    /// Run the slices with ids in `slices` into `dest`, with the semantics of
    /// [`Self::execute_into`] for the sum over those slices only.
    ///
    /// This is the partitioning hook: disjoint ranges run with
    /// `Dest::accumulate` into separate destinations sum to the full result.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
    /// use tenferro_tensor::{Dest, Tensor};
    ///
    /// let a = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    /// let options = SliceOptions::default().labels([u32::from(b'i')]);
    /// let plan = SlicedEinsumPlan::prepare(&[&a, &a], "i,i->", &EinsumOptions::default(), &options)?;
    /// let mut engine = CpuEngine::serial();
    /// let mut low = Tensor::zeros(tenferro_tensor::DType::F64, Vec::<usize>::new())?;
    /// let mut high = Tensor::zeros(tenferro_tensor::DType::F64, Vec::<usize>::new())?;
    /// plan.execute_range_into(&[&a, &a], 0..2, &mut engine, Dest::accumulate(&mut low))?;
    /// plan.execute_range_into(&[&a, &a], 2..4, &mut engine, Dest::accumulate(&mut high))?;
    /// assert_eq!(low.as_slice::<f64>()?, &[5.0]);
    /// assert_eq!(high.as_slice::<f64>()?, &[25.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `Validation` when the inputs or the destination differ in count,
    /// dtype or shape from the plan, `alpha`/`beta` have another dtype, the
    /// range is not within `0..slice_count()`, or an empty range is given with
    /// `Overwrite` or `Scaled` (which would leave `dest` undefined); checked
    /// before any write. `Unsupported` for an input without a borrowed view;
    /// backend failures otherwise.
    pub fn execute_range_into<I: EinsumInputs + ?Sized>(
        &self,
        inputs: &I,
        slices: Range<usize>,
        exec: &mut dyn Engine,
        dest: Dest<<I::Repr as Repr>::Write<'_>>,
    ) -> Res<()> {
        let reads = inputs.reads();
        self.check_inputs(&reads)?;
        if slices.start > slices.end || slices.end > self.slice_count() {
            return Err(tensor_err(Error::invalid_argument(
                OP,
                "slices",
                format!(
                    "slice range {slices:?} is not within 0..{}",
                    self.slice_count()
                ),
            )));
        }
        let mode = dest.mode();
        let (out, first) = dest.into_accumulation(self.dtype, Conj::No, Conj::No)?;
        let mut out = <I::Repr as Repr>::erase_write(out);
        if out.dtype() != self.dtype {
            return Err(tensor_err(Error::dtype_mismatch(
                OP,
                self.dtype,
                out.dtype(),
            )));
        }
        if out.shape() != self.output_shape.as_slice() {
            return Err(tensor_err(Error::shape_mismatch(
                OP,
                self.output_shape.clone(),
                out.shape().to_vec(),
            )));
        }
        if slices.is_empty() {
            if mode == DestMode::Accumulate {
                return Ok(());
            }
            return Err(tensor_err(Error::invalid_argument(
                OP,
                "slices",
                "an empty slice range with Overwrite or Scaled would leave the destination undefined",
            )));
        }
        let conj = conjugate(&reads, &self.conj, exec)?;
        let reads = substituted(&reads, &conj);
        let rest = tenferro_tensor::DotGeneralAccumulation {
            beta: ContractionScalar::one(self.dtype)?,
            ..first
        };
        let source = Slices {
            plan: self,
            reads: &reads,
        };
        run_chunks(&source, slices, exec, &mut out, first, rest).map_err(tensor_err)?;
        Ok(())
    }

    /// Statistics of the bounded CPU numerical-plan cache of the per-slice
    /// plan. A warm slice loop adds hits and no misses: every slice presents
    /// the same layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
    /// let options = SliceOptions::default().labels([u32::from(b'j')]);
    /// let plan = SlicedEinsumPlan::prepare(&[&a, &a, &a], "ij,jk,kl->il", &EinsumOptions::default(), &options)?;
    /// plan.execute(&[&a, &a, &a], &mut CpuEngine::serial())?;
    /// let stats = plan.cpu_cache_stats()?;
    /// assert_eq!((stats.misses, stats.hits), (1, 1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `RuntimeState` when the cache's metadata lock is poisoned.
    pub fn cpu_cache_stats(&self) -> Res<CacheStats> {
        self.inner.cpu_cache_stats().map_err(tensor_err)
    }

    /// Elements of the intermediate arena the CPU executor allocated for one
    /// slice (every intermediate of one slice lives in it), once a slice of a
    /// network of three or more operands has run on a CPU engine; `None`
    /// before that or on another route. This is the achieved counterpart of
    /// [`SlicedEinsumReport::slice_peak_intermediate_elements`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{EinsumOptions, SliceOptions, SlicedEinsumPlan};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
    /// let options = SliceOptions::default().labels([u32::from(b'j')]);
    /// let plan = SlicedEinsumPlan::prepare(&[&a, &a, &a], "ij,jk,kl->il", &EinsumOptions::default(), &options)?;
    /// assert_eq!(plan.cpu_arena_elements(), None);
    /// plan.execute(&[&a, &a, &a], &mut CpuEngine::serial())?;
    /// assert!(plan.cpu_arena_elements().is_some());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn cpu_arena_elements(&self) -> Option<usize> {
        self.inner.cpu_arena_elements()
    }

    fn check_inputs(&self, reads: &[TensorRead<'_>]) -> Res<()> {
        if reads.len() != self.inputs.len() {
            return Err(tensor_err(Error::invalid_argument(
                OP,
                "inputs",
                format!(
                    "the plan expects {} inputs, got {}",
                    self.inputs.len(),
                    reads.len()
                ),
            )));
        }
        for (read, (dtype, shape)) in reads.iter().zip(&self.inputs) {
            if read.dtype() != *dtype {
                return Err(tensor_err(Error::dtype_mismatch(OP, *dtype, read.dtype())));
            }
            if read.shape() != shape.as_slice() {
                return Err(tensor_err(Error::shape_mismatch(
                    OP,
                    shape.clone(),
                    read.shape().to_vec(),
                )));
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ planning

fn target_unmet(target: f64, reached: f64, labels: &[u32], reason: &str) -> Error {
    Error::MemoryTargetUnmet {
        target_log2_elements: target,
        reached_log2_elements: reached.log2(),
        sliced_labels: labels.to_vec(),
        reason: reason.to_string(),
    }
}

fn validated_labels(
    labels: &[u32],
    subscripts: &Subscripts,
    sizes: &HashMap<u32, usize>,
) -> crate::Result<Vec<u32>> {
    let mut seen = HashSet::new();
    for label in labels {
        let reject = |message: String| Err(Error::invalid_argument(OP, "labels", message));
        if !seen.insert(*label) {
            return reject(format!("label {label} is sliced twice"));
        }
        let Some(&extent) = sizes.get(label) else {
            return reject(format!("label {label} is on no operand"));
        };
        if subscripts.output.contains(label) {
            return reject(format!(
                "label {label} is an output label; only contracted labels can be sliced"
            ));
        }
        if extent == 0 {
            return reject(format!("label {label} has extent 0"));
        }
    }
    Ok(labels.to_vec())
}

/// `sizes` with every label of `labels` at extent 1.
fn sliced_size(sizes: &HashMap<u32, usize>, labels: &[u32]) -> HashMap<u32, usize> {
    let mut sliced = sizes.clone();
    for label in labels {
        sliced.insert(*label, 1);
    }
    sliced
}

fn elements(labels: &[u32], size: impl Fn(u32) -> f64) -> f64 {
    labels.iter().map(|&label| size(label)).product()
}

#[derive(Clone, Copy, Debug)]
struct TreeCost {
    flops: f64,
    largest: f64,
    peak: f64,
}

/// Flops, largest intermediate and peak live intermediates of `tree` at the
/// extents `size` gives.
fn tree_cost(tree: &ContractionTree, size: impl Fn(u32) -> f64) -> TreeCost {
    let n = tree.subscripts.inputs.len();
    if tree.step_count() == 0 {
        let out = elements(&tree.subscripts.output, &size);
        return TreeCost {
            flops: elements(&tree.subscripts.inputs[0], &size),
            largest: out,
            peak: out,
        };
    }
    let mut results = Vec::with_capacity(tree.step_count());
    let (mut flops, mut largest, mut live, mut peak) = (0.0, 0.0_f64, 0.0, 0.0_f64);
    for step in 0..tree.step_count() {
        let (left, right) = tree.step_pair(step).expect("step within step_count");
        let (lhs, rhs, out) = tree.step_subscripts(step).expect("step within step_count");
        let mut joined: Vec<u32> = lhs.iter().chain(rhs).copied().collect();
        joined.sort_unstable();
        joined.dedup();
        flops += elements(&joined, &size);
        let result = elements(out, &size);
        results.push(result);
        largest = largest.max(result);
        live += result;
        peak = peak.max(live);
        for operand in [left, right] {
            if operand >= n {
                live -= results[operand - n];
            }
        }
    }
    TreeCost {
        flops,
        largest,
        peak,
    }
}

/// Run omeco's slicer (fast preset) on `tree` at the extents `sizes`. Without
/// `reshape` the slicer anneals nothing and only picks labels. With
/// `strip_output` the output labels are removed from the network it sees.
fn run_slicer(
    tree: &ContractionTree,
    subscripts: &Subscripts,
    sizes: &HashMap<u32, usize>,
    target: f64,
    reshape: bool,
    strip_output: bool,
) -> Option<omeco::SlicedEinsum<u32>> {
    let mut slicer = TreeSASlicer::fast().with_sc_target(target);
    if !reshape {
        slicer = slicer.with_niters(0);
    }
    let keep = |label: &u32| !strip_output || !subscripts.output.contains(label);
    let ixs: Vec<Vec<u32>> = subscripts
        .inputs
        .iter()
        .map(|axes| axes.iter().copied().filter(keep).collect())
        .collect();
    omeco::slice_code(&to_omeco(tree, keep), sizes, &slicer, &ixs)
}

/// `tree` in omeco's nested form, each step one binary node, keeping the
/// labels `keep` accepts.
fn to_omeco(tree: &ContractionTree, keep: impl Fn(&u32) -> bool) -> OmecoNested<u32> {
    let n = tree.subscripts.inputs.len();
    let mut nodes: Vec<Option<OmecoNested<u32>>> =
        (0..n).map(|i| Some(OmecoNested::leaf(i))).collect();
    for step in 0..tree.step_count() {
        let (left, right) = tree.step_pair(step).expect("step within step_count");
        let (lhs, rhs, out) = tree.step_subscripts(step).expect("step within step_count");
        let filtered = |labels: &[u32]| labels.iter().copied().filter(&keep).collect::<Vec<_>>();
        let eins = OmecoEinCode::new(vec![filtered(lhs), filtered(rhs)], filtered(out));
        let args = vec![
            nodes[left]
                .take()
                .expect("a valid tree uses each operand once"),
            nodes[right]
                .take()
                .expect("a valid tree uses each operand once"),
        ];
        nodes.push(Some(OmecoNested::node(args, eins)));
    }
    nodes
        .pop()
        .flatten()
        .expect("a tree over at least one operand has a root")
}

/// omeco's nested form as fixed-id pairs (operands `0..n`, the k-th pair
/// makes `n + k`); a node with more than two arguments folds left to right.
fn omeco_pairs(tree: &OmecoNested<u32>, n: usize) -> crate::Result<Vec<(usize, usize)>> {
    fn walk(
        tree: &OmecoNested<u32>,
        n: usize,
        pairs: &mut Vec<(usize, usize)>,
    ) -> crate::Result<usize> {
        match tree {
            OmecoNested::Leaf { tensor_index } if *tensor_index < n => Ok(*tensor_index),
            OmecoNested::Leaf { tensor_index } => Err(Error::planning(format!(
                "omeco's slicer returned leaf {tensor_index} outside 0..{n}"
            ))),
            OmecoNested::Node { args, .. } => {
                let mut acc: Option<usize> = None;
                for arg in args {
                    let id = walk(arg, n, pairs)?;
                    acc = Some(match acc {
                        None => id,
                        Some(previous) => {
                            pairs.push((previous, id));
                            n + pairs.len() - 1
                        }
                    });
                }
                acc.ok_or_else(|| Error::planning("omeco's slicer returned an empty node"))
            }
        }
    }
    let mut pairs = Vec::with_capacity(n.saturating_sub(1));
    walk(tree, n, &mut pairs)?;
    Ok(pairs)
}

// ----------------------------------------------------------------- execution

/// Slice id `id` as one value per sliced label, the first label fastest.
fn decode(mut id: usize, extents: &[usize], assignment: &mut [usize]) {
    for (value, &extent) in assignment.iter_mut().zip(extents) {
        *value = id % extent;
        id /= extent;
    }
}

/// The slices of a plan as chunks of the shared chunk loop: chunk `id` reads
/// metadata-only views of the operands with the sliced axes fixed to slice
/// `id`'s assignment, and accumulates into the whole destination.
struct Slices<'p, 'a> {
    plan: &'p SlicedEinsumPlan,
    reads: &'p [TensorRead<'a>],
}

impl ChunkSource for Slices<'_, '_> {
    type Held = Vec<usize>;

    fn hold(&self, id: usize, _exec: &mut dyn Engine) -> Res<Vec<usize>> {
        let mut assignment = vec![0; self.plan.extents.len()];
        decode(id, &self.plan.extents, &mut assignment);
        Ok(assignment)
    }

    fn reads<'s>(&'s self, assignment: &'s Vec<usize>) -> Res<Vec<TensorRead<'s>>> {
        slice_reads(self.reads, &self.plan.fixes, assignment).map_err(tensor_err)
    }

    fn plan(&self, _id: usize) -> &ConcreteEinsumPlan {
        &self.plan.inner
    }

    fn window(&self, _id: usize) -> Option<(usize, Range<usize>)> {
        None
    }
}

/// Each operand as one slice reads it: untouched without a sliced label, and
/// otherwise a metadata-only view with each sliced axis narrowed to the
/// slice's value (extent 1).
fn slice_reads<'a>(
    reads: &[TensorRead<'a>],
    fixes: &[Vec<(usize, usize)>],
    assignment: &[usize],
) -> crate::Result<Vec<TensorRead<'a>>> {
    reads
        .iter()
        .zip(fixes)
        .map(|(read, fixes)| {
            if fixes.is_empty() {
                return Ok(read.clone());
            }
            let at: Vec<(usize, usize)> = fixes
                .iter()
                .map(|&(axis, label)| (axis, assignment[label]))
                .collect();
            let dtype = read.dtype();
            let view = read.clone().tensor_view().ok_or_else(|| {
                Error::Tensor(tenferro_tensor::Error::unsupported_dtype(
                    OP,
                    dtype,
                    "an operand without a borrowed view cannot be sliced",
                ))
            })?;
            let view = match view {
                TensorView::F32(v) => TensorView::F32(fix_axes(v, &at)?),
                TensorView::F64(v) => TensorView::F64(fix_axes(v, &at)?),
                TensorView::I32(v) => TensorView::I32(fix_axes(v, &at)?),
                TensorView::I64(v) => TensorView::I64(fix_axes(v, &at)?),
                TensorView::Bool(v) => TensorView::Bool(fix_axes(v, &at)?),
                TensorView::C32(v) => TensorView::C32(fix_axes(v, &at)?),
                TensorView::C64(v) => TensorView::C64(fix_axes(v, &at)?),
            };
            Ok(TensorRead::from_view(view))
        })
        .collect()
}

/// `view` with each `(axis, value)` narrowed to `value..value + 1`.
///
/// A host view with non-negative strides is rebased onto its storage from the
/// new first element, so every slice has offset 0 and the same layout: an
/// engine whose plan cache keys on the offset (the CPU binary contraction
/// cache) then reuses one plan for all slices. Other views keep their offset.
fn fix_axes<'a, T: 'static>(
    view: TypedTensorView<'a, T>,
    at: &[(usize, usize)],
) -> crate::Result<TypedTensorView<'a, T>> {
    let strides = view.strides().to_vec();
    let mut shape = view.shape().to_vec();
    let mut offset = view.offset();
    for &(axis, value) in at {
        offset += value as isize * strides[axis];
        shape[axis] = 1;
    }
    // Rebasing builds a default-placement view, so only views that already
    // have the default placement (no pinned memory, no CPU affinity) take it.
    if let Ok(storage) = view.host_storage() {
        if view.placement() == &tenferro_tensor::Placement::default()
            && strides.iter().all(|&stride| stride >= 0)
        {
            if let Some(rest) = usize::try_from(offset).ok().and_then(|o| storage.get(o..)) {
                return Ok(TypedTensorView::from_slice(&shape, &strides, 0, rest)?);
            }
        }
    }
    let specs: Vec<StridedSliceSpec> = (0..shape.len())
        .map(|axis| match at.iter().find(|&&(a, _)| a == axis) {
            Some(&(_, value)) => StridedSliceSpec::new(value as isize, Some(value as isize + 1), 1),
            None => StridedSliceSpec::all(),
        })
        .collect();
    Ok(view.slice_view(&specs)?)
}

#[cfg(test)]
mod tests;
