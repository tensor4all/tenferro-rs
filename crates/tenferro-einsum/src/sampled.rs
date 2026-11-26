//! Sampled einsum (#2675): a contraction evaluated only at a list of output
//! index tuples, without forming the output.
//!
//! ```text
//! Y[r..., s] = T[r..., c_s],   T = einsum(spec, operands),   s = 0, ..., n - 1
//! ```
//!
//! `c_s` fixes a value of every *sampled* output label; the other output
//! labels `r` stay dense. Each node `v` of the contraction tree keeps one row
//! per **distinct** assignment `P_v` of the sampled labels below it, so samples
//! that agree on a subtree's labels share its work (the sparse-state method of
//! F. Pan, K. Chen and P. Zhang, PRL **129**, 090502 (2022)).
//!
//! # Provenance
//!
//! The planner is ported from Feng Pan's prototype
//! [Fanerst/sampled-einsum](https://github.com/Fanerst/sampled-einsum) at
//! commit `c482b79a0f1e407390ef7822a89e7706220021d1` (`src/plan.rs`,
//! `src/exec.rs`; Apache-2.0), which ports `contraction_scheme_sparse` of
//! [artensor](https://github.com/Fanerst/artensor) at
//! `58a93ae7e0199dc73f3f5dfbee15a98782aded7c` (Apache-2.0). The code is
//! ported, not re-derived; each file names what it took.
//!
//! # Execution (v0: composition)
//!
//! Each pairwise step is one of the prototype's kinds ([`SampledStepKind`]):
//! an ordinary contraction, a configuration axis carried as a free axis, the
//! full product `P_l × P_r` (a contraction and a reshape, no gather), the full
//! product followed by a row selection, or a list of row pairs gathered with
//! `index_select` and contracted as a batch, in chunks under
//! [`SampleOptions::max_chunk_elements`]. A final `index_select` returns the
//! rows in the caller's sample order; duplicate samples are allowed. Only
//! existing operations run (contractions, gathers, reshapes), so the eager
//! layer differentiates through them with no new rules. The gather-free v1
//! of the issue is not part of this.
//!
//! The chunked pairs run through the chunk loop the sliced einsum uses
//! (`SlicedEinsumPlan`, #2676): one prepared contraction for every chunk (a
//! second one for a shorter last chunk), each chunk written into a
//! metadata-only window of the step's preallocated result. The eager layer
//! has no in-place window write on its tape, so there each chunk's result is
//! kept and the pieces are concatenated, as the prototype does everywhere.
//!
//! # Gradients
//!
//! Differentiating an eager sampled einsum gives the gradient of a function
//! of the **sampled outputs** `Y` only: that is what variational Monte Carlo
//! needs (`ψ(σ)` and `∂ψ(σ)/∂θ` for a batch of `σ`). It is **not** the
//! gradient of a function of the full contraction `T`: the unsampled entries
//! of `T` are never computed, so their derivative cannot appear.
//!
//! # Layers
//!
//! Concrete (any engine) and eager (with the `autodiff` feature). A traced
//! spelling is deferred: `|P_v|` depends on the values of the samples, so a
//! traced sampled einsum needs execution-time planning; it is to follow the
//! one-call grammar of #2552 and lower through the sub-program machinery of
//! #2707.

use std::collections::HashMap;

use tenferro_tensor::{DType, Engine, Repr, TensorRead};

use crate::api::concrete::{check_mask, resolve, tensor_err};
use crate::api::{ConjMask, EinsumInputs, EinsumOptions, EinsumSpec};
use crate::concrete::ConcreteEinsumInputSpec;
use crate::{ConcreteEinsumPlan, EinsumOptimize, Error, Subscripts};

mod exec;
mod plan;

#[cfg(feature = "autodiff")]
pub(crate) use exec::execute_eager;
pub(crate) use exec::{execute_concrete, ExecStats};

type Res<T> = tenferro_tensor::Result<T>;

pub(crate) const OP: &str = "SampledEinsumPlan";

/// The output configurations at which a sampled einsum is evaluated.
///
/// `labels` names the sampled output labels by label id (for a text spec,
/// `u32::from(c)` of the letter, as for [`crate::SliceOptions::labels`]);
/// they may be any non-empty subset of the output labels, in any order, and a
/// label may sit on several operands. `values` is column-major `[n,
/// labels.len()]`: entry `(s, j)` at `s + n * j` is the value of `labels[j]`
/// in sample `s`. Samples may repeat; each result column is one sample, in
/// the order given.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::Samples;
///
/// // (i, k) = (1, 2) and (0, 0) for `ij,jk->ik`.
/// let samples = Samples::new([u32::from(b'i'), u32::from(b'k')], 2, vec![1, 0, 2, 0]);
/// assert_eq!(samples.n, 2);
/// assert_eq!(samples.value(0, 1), Some(2));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Samples {
    /// Sampled output labels.
    pub labels: Vec<u32>,
    /// Number of samples. Duplicates are allowed.
    pub n: usize,
    /// Column-major `[n, labels.len()]` values.
    pub values: Vec<usize>,
}

impl Samples {
    /// Samples of `labels` with column-major `[n, labels.len()]` values;
    /// checked when a plan is prepared.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::Samples;
    ///
    /// let samples = Samples::new([u32::from(b'k')], 3, vec![2, 0, 2]);
    /// assert_eq!(samples.labels, vec![u32::from(b'k')]);
    /// ```
    #[must_use]
    pub fn new(labels: impl IntoIterator<Item = u32>, n: usize, values: Vec<usize>) -> Self {
        Self {
            labels: labels.into_iter().collect(),
            n,
            values,
        }
    }

    /// The value of `labels[label]` in sample `sample`, or `None` when either
    /// index is out of range or `values` is shorter than `n * labels.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::Samples;
    ///
    /// let samples = Samples::new([0, 1], 2, vec![3, 4, 5, 6]);
    /// assert_eq!(samples.value(1, 0), Some(4));
    /// assert_eq!(samples.value(2, 0), None);
    /// ```
    #[must_use]
    pub fn value(&self, sample: usize, label: usize) -> Option<usize> {
        if sample >= self.n || label >= self.labels.len() {
            return None;
        }
        let at = self.n.checked_mul(label)?.checked_add(sample)?;
        self.values.get(at).copied()
    }
}

/// How a sampled einsum is planned: the contraction order and the two
/// planner settings of the prototype.
///
/// - `einsum`: the contraction order (`optimize`) and conjugation (`conj`).
///   An explicit order (`EinsumOptimize::Path`, `Nested`, `Tree` or `False`)
///   is planned exactly; `Auto` asks omeco for a tree (dense cost, not
///   sparsity-aware).
/// - `max_chunk_elements`: the largest element count of one chunk of a
///   gathered-pairs step (its result rows and each gathered operand); the
///   step's result itself is not split, it is the next node. Elements, like
///   the target of [`crate::SliceOptions`], but a count rather than log2: it
///   is the prototype's `max_chunk_elems`.
/// - `outer_select_ratio`: when both children carry configurations, compute
///   the full product `P_l × P_r` and select the needed rows if the product
///   is at most this many times the rows needed (and fits one chunk); `0.0`
///   disables it.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{EinsumOptimize, EinsumOptions, SampleOptions};
///
/// let o = SampleOptions::default()
///     .einsum(EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(0, 1)])))
///     .max_chunk_elements(1 << 20)
///     .outer_select_ratio(0.0);
/// assert_eq!(o.max_chunk_elements, 1 << 20);
/// assert_eq!(SampleOptions::default().max_chunk_elements, 1 << 26);
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SampleOptions {
    /// Contraction order and conjugation.
    pub einsum: EinsumOptions,
    /// Largest element count of one chunk of a gathered-pairs step.
    pub max_chunk_elements: usize,
    /// Full-product-then-select threshold; `0.0` disables it.
    pub outer_select_ratio: f64,
}

impl Default for SampleOptions {
    fn default() -> Self {
        Self {
            einsum: EinsumOptions::default(),
            max_chunk_elements: 1 << 26,
            outer_select_ratio: 2.0,
        }
    }
}

impl SampleOptions {
    /// Return options with the given contraction order and conjugation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumOptimize, EinsumOptions, SampleOptions};
    ///
    /// let o = SampleOptions::default().einsum(EinsumOptions::default().optimize(EinsumOptimize::False));
    /// assert!(matches!(o.einsum.optimize, EinsumOptimize::False));
    /// ```
    #[must_use]
    pub fn einsum(mut self, einsum: EinsumOptions) -> Self {
        self.einsum = einsum;
        self
    }

    /// Return options with the given chunk cap (elements).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::SampleOptions;
    ///
    /// assert_eq!(SampleOptions::default().max_chunk_elements(8).max_chunk_elements, 8);
    /// ```
    #[must_use]
    pub fn max_chunk_elements(mut self, max_chunk_elements: usize) -> Self {
        self.max_chunk_elements = max_chunk_elements;
        self
    }

    /// Return options with the given full-product-then-select threshold.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::SampleOptions;
    ///
    /// assert_eq!(SampleOptions::default().outer_select_ratio(0.0).outer_select_ratio, 0.0);
    /// ```
    #[must_use]
    pub fn outer_select_ratio(mut self, outer_select_ratio: f64) -> Self {
        self.outer_select_ratio = outer_select_ratio;
        self
    }
}

/// How one pairwise step of a sampled einsum runs.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::SampledStepKind;
///
/// assert_ne!(SampledStepKind::Dense, SampledStepKind::Pairs { chunks: 1 });
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampledStepKind {
    /// Neither child carries sampled labels: one ordinary contraction.
    Dense,
    /// One child carries sampled labels: its configuration axis is a free axis.
    OneSided,
    /// Every pair of child rows is needed: one contraction and a reshape, no gather.
    FullProduct,
    /// The full product is computed and the needed rows are selected.
    OuterSelect,
    /// Rows are gathered per pair and contracted as a batch, in `chunks` chunks.
    Pairs {
        /// Number of chunks.
        chunks: usize,
    },
}

/// The planning record of one pairwise step: `rows` is `|P_v|` of the node it
/// produces.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, Samples};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
/// let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
/// let samples = Samples::new([u32::from(b'i'), u32::from(b'k')], 2, vec![0, 1, 1, 2]);
/// let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &samples, &SampleOptions::default())?;
/// let step = &plan.report().steps[0];
/// assert_eq!((step.rows_left, step.rows_right, step.rows), (2, 2, 2));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SampledStepReport {
    /// How the step runs.
    pub kind: SampledStepKind,
    /// Configuration rows of the left child (1 if it has none).
    pub rows_left: usize,
    /// Configuration rows of the right child (1 if it has none).
    pub rows_right: usize,
    /// Configuration rows of the result, `|P_v|` (1 if it has none).
    pub rows: usize,
    /// Elements of the result.
    pub out_elements: usize,
    /// Multiply-adds of the step (the product of the extents of every label
    /// of its two operands, the configuration axis counted once), as in
    /// [`crate::SlicedEinsumReport::slice_flops`].
    pub flops: f64,
}

/// What a sampled einsum plan does, known before anything runs.
///
/// Every node's `|P_v|` is here: the leaves in [`Self::leaf_rows`], each step's
/// result in [`SampledStepReport::rows`]. Peaks count **logical elements of
/// live temporaries**: every tensor an instruction produces, from that
/// instruction until the plan drops it, and the gathered rows of the chunk
/// being contracted. They exclude the caller's operands, the returned
/// result's storage outside the program, backend scratch (packing buffers,
/// a CPU contraction that cannot write into the window computes the chunk
/// and then copies it), and, on the eager layer, what the tape keeps for the
/// backward pass. The executors measure the same quantity on the tensors
/// they actually hold ([`SampledEinsumPlan::execute_with_stats`]).
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, Samples};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
/// let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
/// let samples = Samples::new([u32::from(b'i'), u32::from(b'k')], 2, vec![0, 1, 1, 2]);
/// let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &samples, &SampleOptions::default())?;
/// let report = plan.report();
/// assert_eq!(report.leaf_rows, vec![2, 2]);
/// assert_eq!(report.pairs, vec![(0, 1)]);
/// assert!(report.peak_elements > 0);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SampledEinsumReport {
    /// The contraction path the plan follows, as SSA pairs: step `k`
    /// contracts nodes `a` and `b` into node `n_inputs + k`.
    pub pairs: Vec<(usize, usize)>,
    /// Configuration rows of each leaf (1 for a leaf without sampled labels).
    pub leaf_rows: Vec<usize>,
    /// One record per pairwise step, in execution order.
    pub steps: Vec<SampledStepReport>,
    /// Peak live-temporary elements of the concrete executor.
    pub peak_elements: usize,
    /// Peak live-temporary elements of the eager executor.
    pub eager_peak_elements: usize,
    /// Sum of the steps' multiply-adds.
    pub total_flops: f64,
}

/// A sampled einsum planned for given operand shapes and samples.
///
/// This is the primary entry: it plans along the contraction order of
/// [`SampleOptions::einsum`], exactly as given when the order is explicit,
/// which is what makes [`Self::report`] (per-node `|P_v|`, flops, peak) a
/// statement about the run. The one-call `einsum_at` (concrete
/// [`crate::EinsumOps`], eager `EagerEinsumOps`) is derived from it: prepare
/// with the given options, then execute.
///
/// A plan depends on the sample values, not only on shapes, so it is never
/// cached by tenferro; hold it to run the same samples on new operand values
/// (for example each step of an optimisation).
///
/// # Gradients
///
/// [`Self::execute_eager`] records ordinary operations, so reverse and forward
/// mode follow. They give the derivative of a function of the sampled outputs
/// `Y` only: right for VMC (`ψ(σ)`, `∂ψ(σ)/∂θ`), **not** the gradient of a
/// function of the full contraction, because the unsampled entries are never
/// computed and their derivative cannot appear.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuEngine;
/// use tenferro_einsum::{EinsumOps, SampleOptions, SampledEinsumPlan, Samples};
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2, 3], (0..6).map(f64::from).collect())?;
/// let b = Tensor::from_vec_col_major(vec![3, 4], (0..12).map(f64::from).collect())?;
/// let mut engine = CpuEngine::serial();
///
/// // `ij,jk->ik` at (i, k) = (1, 2), (0, 0), (1, 2): SDDMM with a duplicate.
/// let samples = Samples::new([u32::from(b'i'), u32::from(b'k')], 3, vec![1, 0, 1, 2, 0, 2]);
/// let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &samples, &SampleOptions::default())?;
/// let y = plan.execute(&[&a, &b], &mut engine)?;
/// assert_eq!(y.shape(), &[3]);
///
/// let dense = [&a, &b].einsum("ij,jk->ik", &mut engine)?;
/// let d = dense.as_slice::<f64>()?;
/// assert_eq!(y.as_slice::<f64>()?, &[d[1 + 2 * 2], d[0], d[1 + 2 * 2]]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct SampledEinsumPlan {
    inputs: Vec<(DType, Vec<usize>)>,
    conj: ConjMask,
    program: plan::Program,
    /// One prepared plan per contraction of the program.
    plans: Vec<ConcreteEinsumPlan>,
    report: SampledEinsumReport,
}

impl SampledEinsumPlan {
    /// Plan a sampled einsum from the inputs' dtypes and shapes and the
    /// sample values; element data is not read.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumOptimize, EinsumOptions, SampleOptions, SampledEinsumPlan, Samples};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
    /// let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6])?;
    /// let c = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
    /// // An explicit path: (b, c) first, then a.
    /// let options = SampleOptions::default()
    ///     .einsum(EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(1, 2), (0, 1)])));
    /// let samples = Samples::new([u32::from(b'i')], 2, vec![1, 0]);
    /// let plan = SampledEinsumPlan::prepare(&[&a, &b, &c], "ij,jk,kl->il", &samples, &options)?;
    /// assert_eq!(plan.report().pairs, vec![(1, 2), (0, 3)]);
    /// assert_eq!(plan.output_shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `Validation` for a malformed spec or order, operands of different
    /// dtypes, a label repeated within one operand or broadcast between
    /// operands (v0 takes neither), an output label on no operand, a
    /// conjugation mask longer than the inputs, no sampled label, a sampled
    /// label that is not an output label or repeats, sample values of the
    /// wrong count or out of range, or more than `2^32 - 1` samples.
    pub fn prepare<I: EinsumInputs + ?Sized>(
        inputs: &I,
        spec: &(impl EinsumSpec + ?Sized),
        samples: &Samples,
        options: &SampleOptions,
    ) -> Res<Self> {
        let reads = inputs.reads();
        check_mask(&options.einsum.conj, reads.len())?;
        let subscripts = resolve(spec.spec(), &reads).map_err(tensor_err)?;
        let metadata: Vec<(DType, Vec<usize>)> = reads
            .iter()
            .map(|read| (read.dtype(), read.shape().to_vec()))
            .collect();
        Self::plan(metadata, &subscripts, samples, options).map_err(tensor_err)
    }

    pub(crate) fn plan(
        inputs: Vec<(DType, Vec<usize>)>,
        subscripts: &Subscripts,
        samples: &Samples,
        options: &SampleOptions,
    ) -> crate::Result<Self> {
        let Some(&(dtype, _)) = inputs.first() else {
            return Err(Error::invalid_argument(
                OP,
                "inputs",
                "a sampled einsum needs at least one input",
            ));
        };
        if options.einsum.conj.len() > inputs.len() {
            return Err(Error::invalid_argument(
                OP,
                "conj",
                format!(
                    "the conjugation mask has {} entries for {} inputs",
                    options.einsum.conj.len(),
                    inputs.len()
                ),
            ));
        }
        if let Some(&(other, _)) = inputs.iter().find(|(d, _)| *d != dtype) {
            return Err(Error::dtype_mismatch(OP, dtype, other));
        }
        let shapes: Vec<&[usize]> = inputs.iter().map(|(_, s)| s.as_slice()).collect();
        let (_, tree) = crate::optimize::resolve_einsum_strategy_with_spec(
            options.einsum.optimize.clone(),
            subscripts,
            &shapes,
        )?;
        let sizes: HashMap<u32, usize> = tree.size_dict.clone();
        plan::validate(subscripts, &shapes, &sizes, samples)?;
        let pairs = crate::optimize::tree_pairs(&tree);
        let program = plan::plan(subscripts, &sizes, &pairs, samples, options)?;
        let plans = program
            .contractions
            .iter()
            .map(|c| {
                let specs = c
                    .input_shapes
                    .iter()
                    .map(|shape| ConcreteEinsumInputSpec::new(dtype, shape.clone()))
                    .collect();
                ConcreteEinsumPlan::prepare_planned(
                    specs,
                    &c.subscripts,
                    &EinsumOptimize::False,
                    ConjMask::default(),
                )
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let (peak_elements, eager_peak_elements) = plan::peaks(&program);
        let report = SampledEinsumReport {
            pairs,
            leaf_rows: program.leaf_rows.clone(),
            total_flops: program.steps.iter().map(|s| s.flops).sum(),
            steps: program.steps.clone(),
            peak_elements,
            eager_peak_elements,
        };
        Ok(Self {
            inputs,
            conj: options.einsum.conj.clone(),
            program,
            plans,
            report,
        })
    }

    /// What the plan does: the path, every node's `|P_v|`, per-step kind and
    /// flops, and the peak live-temporary elements of each executor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, SampledStepKind, Samples};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
    /// let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6])?;
    /// // Every (i, k) combination: the full product, no gather.
    /// let samples = Samples::new([u32::from(b'i'), u32::from(b'k')], 4, vec![0, 1, 0, 1, 0, 0, 1, 1]);
    /// let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &samples, &SampleOptions::default())?;
    /// assert_eq!(plan.report().steps[0].kind, SampledStepKind::FullProduct);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn report(&self) -> &SampledEinsumReport {
        &self.report
    }

    /// Shape of the result: the non-sampled output labels in output order,
    /// then the samples.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, Samples};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
    /// let samples = Samples::new([u32::from(b'j')], 5, vec![0, 1, 2, 1, 0]);
    /// let plan = SampledEinsumPlan::prepare(&[&a], "ij->ij", &samples, &SampleOptions::default())?;
    /// assert_eq!(plan.output_shape(), &[2, 5]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    #[must_use]
    pub fn output_shape(&self) -> &[usize] {
        &self.program.output_shape
    }

    /// Run the plan on concrete operands (any engine) and return the
    /// `[rest..., n]` result in the inputs' representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, Samples};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let samples = Samples::new([u32::from(b'i'), u32::from(b'j')], 2, vec![1, 0, 1, 0]);
    /// let plan = SampledEinsumPlan::prepare(&[&a], "ij->ij", &samples, &SampleOptions::default())?;
    /// let y: TypedTensor<f64> = plan.execute(&[&a], &mut CpuEngine::serial())?;
    /// assert_eq!(y.host_data()?, &[4.0, 1.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `Validation` when the inputs' count, dtypes or shapes differ from the
    /// prepared ones; `Unsupported` for an input without a tensor read;
    /// backend failures otherwise.
    pub fn execute<I: EinsumInputs + ?Sized>(
        &self,
        inputs: &I,
        exec: &mut dyn Engine,
    ) -> Res<<I::Repr as Repr>::Value> {
        let (out, _) = self.execute_with_stats(inputs, exec)?;
        <I::Repr as Repr>::wrap_value(out)
    }

    /// [`Self::execute`] returning an erased tensor and what the run measured:
    /// the peak live-temporary elements on the tensors actually held, the
    /// quantity [`SampledEinsumReport::peak_elements`] plans.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, Samples};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
    /// let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
    /// let samples = Samples::new([u32::from(b'i'), u32::from(b'k')], 2, vec![0, 1, 1, 2]);
    /// let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &samples, &SampleOptions::default())?;
    /// let (y, stats) = plan.execute_with_stats(&[&a, &b], &mut CpuEngine::serial())?;
    /// assert_eq!(y.as_slice::<f64>()?, &[3.0, 3.0]);
    /// assert_eq!(stats.peak_elements, plan.report().peak_elements);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// As [`Self::execute`].
    pub fn execute_with_stats<I: EinsumInputs + ?Sized>(
        &self,
        inputs: &I,
        exec: &mut dyn Engine,
    ) -> Res<(tenferro_tensor::Tensor, SampledExecStats)> {
        let reads = inputs.reads();
        self.check_inputs(&reads)?;
        let conj = crate::api::concrete::conjugate(&reads, &self.conj, exec)?;
        let reads = crate::api::concrete::substituted(&reads, &conj);
        let (out, stats) =
            execute_concrete(&self.program, &self.plans, &reads, exec).map_err(tensor_err)?;
        Ok((out, stats.into()))
    }

    /// Run the plan on eager tensors of a session; the operations are recorded
    /// for reverse mode and differentiate as described under *Gradients*.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, Samples};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
    /// let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
    /// let samples = Samples::new([u32::from(b'i'), u32::from(b'k')], 2, vec![0, 1, 1, 2]);
    /// let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &samples, &SampleOptions::default())?;
    /// let runtime = EagerRuntime::new()?;
    /// let y = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    ///     let x = s.constant_from(a.duplicate()?)?;
    ///     let w = s.constant_from(b.duplicate()?)?;
    ///     plan.execute_eager(s, &[&x, &w])
    /// })?;
    /// assert_eq!(y.value()?.as_slice::<f64>()?, &[3.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`tenferro_ad::Error::Extension`] with a `Validation` kind when the
    /// inputs' count, dtypes or shapes differ from the prepared ones; the
    /// session's [`tenferro_ad::Error`] otherwise (for example
    /// [`tenferro_ad::Error::TensorRuntime`] for a backend failure).
    #[cfg(feature = "autodiff")]
    #[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
    pub fn execute_eager(
        &self,
        session: &mut tenferro_ad::EagerSession<'_>,
        inputs: &[&tenferro_ad::EagerTensor],
    ) -> tenferro_ad::Result<tenferro_ad::EagerTensor> {
        Ok(self.execute_eager_with_stats(session, inputs)?.0)
    }

    /// [`Self::execute_eager`] with what the run measured on the eager
    /// tensors it held; the peak is the quantity
    /// [`SampledEinsumReport::eager_peak_elements`] plans.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ad::EagerRuntime;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, Samples};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?;
    /// let b = Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?;
    /// let samples = Samples::new([u32::from(b'i'), u32::from(b'k')], 2, vec![0, 1, 1, 2]);
    /// let plan = SampledEinsumPlan::prepare(&[&a, &b], "ij,jk->ik", &samples, &SampleOptions::default())?;
    /// let runtime = EagerRuntime::new()?;
    /// let peak = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
    ///     let x = s.constant_from(a.duplicate()?)?;
    ///     let w = s.constant_from(b.duplicate()?)?;
    ///     Ok::<_, tenferro_ad::Error>(plan.execute_eager_with_stats(s, &[&x, &w])?.1.peak_elements)
    /// })?;
    /// assert_eq!(peak, plan.report().eager_peak_elements);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`tenferro_ad::Error::Extension`] with a `Validation` kind when the
    /// inputs' count, dtypes or shapes differ from the prepared ones; the
    /// session's [`tenferro_ad::Error`] otherwise (for example
    /// [`tenferro_ad::Error::TensorRuntime`] for a backend failure).
    #[cfg(feature = "autodiff")]
    #[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
    pub fn execute_eager_with_stats(
        &self,
        session: &mut tenferro_ad::EagerSession<'_>,
        inputs: &[&tenferro_ad::EagerTensor],
    ) -> tenferro_ad::Result<(tenferro_ad::EagerTensor, SampledExecStats)> {
        let runtime_err = |e: Error| crate::api::traced::runtime_err(e, OP);
        self.check_metadata(inputs.iter().map(|t| (t.dtype(), t.shape())))
            .map_err(runtime_err)?;
        let mut conjugated = Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            conjugated.push(if self.conj.get(i).is_yes() {
                Some(session.conj(input)?)
            } else {
                None
            });
        }
        let operands: Vec<&tenferro_ad::EagerTensor> = inputs
            .iter()
            .zip(&conjugated)
            .map(|(input, c)| c.as_ref().unwrap_or(input))
            .collect();
        let (out, stats) = execute_eager(&self.program, session, &operands)?;
        Ok((out, stats.into()))
    }

    /// Statistics of the prepared contractions' CPU numerical-plan caches,
    /// summed. A warm rerun adds hits and no misses: every chunk of a
    /// gathered-pairs step presents the same layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_einsum::{SampleOptions, SampledEinsumPlan, Samples};
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?;
    /// let samples = Samples::new([u32::from(b'i')], 1, vec![1]);
    /// let plan = SampledEinsumPlan::prepare(&[&a, &a, &a], "ij,jk,kl->il", &samples, &SampleOptions::default())?;
    /// plan.execute(&[&a, &a, &a], &mut CpuEngine::serial())?;
    /// let cold = plan.cpu_cache_stats()?;
    /// plan.execute(&[&a, &a, &a], &mut CpuEngine::serial())?;
    /// assert_eq!(plan.cpu_cache_stats()?.misses, cold.misses);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// `RuntimeState` when a cache's metadata lock is poisoned.
    pub fn cpu_cache_stats(&self) -> Res<tenferro_tensor::CacheStats> {
        let mut total = tenferro_tensor::CacheStats::default();
        for plan in &self.plans {
            let s = plan.cpu_cache_stats().map_err(tensor_err)?;
            total.hits += s.hits;
            total.misses += s.misses;
        }
        Ok(total)
    }

    fn check_inputs(&self, reads: &[TensorRead<'_>]) -> Res<()> {
        self.check_metadata(reads.iter().map(|r| (r.dtype(), r.shape())))
            .map_err(tensor_err)
    }

    fn check_metadata<'a>(
        &self,
        actual: impl ExactSizeIterator<Item = (DType, &'a [usize])>,
    ) -> crate::Result<()> {
        if actual.len() != self.inputs.len() {
            return Err(Error::invalid_argument(
                OP,
                "inputs",
                format!(
                    "the plan expects {} inputs, got {}",
                    self.inputs.len(),
                    actual.len()
                ),
            ));
        }
        for ((dtype, shape), (want_dtype, want_shape)) in actual.zip(&self.inputs) {
            if dtype != *want_dtype {
                return Err(Error::dtype_mismatch(OP, *want_dtype, dtype));
            }
            if shape != want_shape.as_slice() {
                return Err(Error::shape_mismatch(
                    OP,
                    want_shape.clone(),
                    shape.to_vec(),
                ));
            }
        }
        Ok(())
    }
}

/// What one execution of a [`SampledEinsumPlan`] measured.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::SampledExecStats;
///
/// let stats = SampledExecStats::default();
/// assert_eq!(stats.peak_elements, 0);
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SampledExecStats {
    /// Peak live-temporary elements, counted on the tensors the executor held.
    pub peak_elements: usize,
    /// Chunks of gathered-pairs steps that ran.
    pub chunks: usize,
}

impl From<ExecStats> for SampledExecStats {
    fn from(stats: ExecStats) -> Self {
        Self {
            peak_elements: stats.peak_elements,
            chunks: stats.chunks,
        }
    }
}
