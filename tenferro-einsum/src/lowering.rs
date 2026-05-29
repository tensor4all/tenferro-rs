//! Read-only einsum lowering plans.
//!
//! These wrappers expose shape and mode metadata computed by the ordinary
//! einsum planner without exposing graph-building or arithmetic execution.
//!
//! # Examples
//!
//! ```
//! use tenferro_einsum::{ContractionTree, Subscripts};
//!
//! let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
//! let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
//! let gemm = tree.step_plan(0).unwrap().gemm();
//!
//! assert_eq!(gemm.contracted_modes(), &[1]);
//! ```

use crate::planning::plan::{
    DiagPlan as InnerDiagPlan, DiagStage as InnerDiagStage, GemmPlan as InnerGemmPlan,
    ReducePlan as InnerReducePlan, StepPlan as InnerStepPlan,
};

/// Read-only lowering data for one pairwise contraction step.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
/// let step = tree.step_plan(0).unwrap();
///
/// assert_eq!(step.gemm().contracted_modes(), &[1]);
/// ```
#[derive(Clone, Copy)]
pub struct PairwiseStepPlan<'a> {
    inner: &'a InnerStepPlan,
}

impl<'a> PairwiseStepPlan<'a> {
    pub(crate) fn new(inner: &'a InnerStepPlan) -> Self {
        Self { inner }
    }

    /// Return the left operand diagonal extraction plan, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    ///
    /// assert!(tree.step_plan(0).unwrap().lhs_diag().is_none());
    /// ```
    #[must_use]
    pub fn lhs_diag(&self) -> Option<DiagPlan<'a>> {
        let inner: &'a InnerStepPlan = self.inner;
        inner.diag_a.as_ref().map(DiagPlan::new)
    }

    /// Return the right operand diagonal extraction plan, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    ///
    /// assert!(tree.step_plan(0).unwrap().rhs_diag().is_none());
    /// ```
    #[must_use]
    pub fn rhs_diag(&self) -> Option<DiagPlan<'a>> {
        let inner: &'a InnerStepPlan = self.inner;
        inner.diag_b.as_ref().map(DiagPlan::new)
    }

    /// Return the left operand pre-reduction plan, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let reduce = tree.step_plan(0).unwrap().lhs_reduce().unwrap();
    ///
    /// assert_eq!(reduce.kept_subs(), &[1]);
    /// ```
    #[must_use]
    pub fn lhs_reduce(&self) -> Option<ReducePlan<'a>> {
        let inner: &'a InnerStepPlan = self.inner;
        inner.gemm.reduce_a.as_ref().map(ReducePlan::new)
    }

    /// Return the right operand pre-reduction plan, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let reduce = tree.step_plan(0).unwrap().rhs_reduce().unwrap();
    ///
    /// assert_eq!(reduce.kept_subs(), &[1]);
    /// ```
    #[must_use]
    pub fn rhs_reduce(&self) -> Option<ReducePlan<'a>> {
        let inner: &'a InnerStepPlan = self.inner;
        inner.gemm.reduce_b.as_ref().map(ReducePlan::new)
    }

    /// Return the GEMM decomposition for this pairwise step.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.output_gemm_shape(), &[2, 4]);
    /// ```
    #[must_use]
    pub fn gemm(&self) -> GemmPlan<'a> {
        let inner: &'a InnerStepPlan = self.inner;
        GemmPlan::new(&inner.gemm)
    }
}

/// Read-only diagonal extraction plan for one operand.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let subs = Subscripts::new(&[&[0, 0], &[0]], &[0]);
/// let tree = ContractionTree::from_pairs(&subs, &[&[3, 3], &[3]], &[(0, 1)]).unwrap();
/// let diag = tree.step_plan(0).unwrap().lhs_diag().unwrap();
///
/// assert_eq!(diag.result_subs(), &[0]);
/// ```
#[derive(Clone, Copy)]
pub struct DiagPlan<'a> {
    inner: &'a InnerDiagPlan,
}

impl<'a> DiagPlan<'a> {
    fn new(inner: &'a InnerDiagPlan) -> Self {
        Self { inner }
    }

    /// Return the sequential diagonal extraction stages.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 0], &[0]], &[0]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[3, 3], &[3]], &[(0, 1)]).unwrap();
    /// let diag = tree.step_plan(0).unwrap().lhs_diag().unwrap();
    /// let mut stages = diag.stages();
    ///
    /// assert_eq!(stages.next().unwrap().axis_pairs(), &[(0, 1)]);
    /// assert!(stages.next().is_none());
    /// ```
    pub fn stages(&self) -> impl Iterator<Item = DiagStage<'a>> + '_ {
        let inner: &'a InnerDiagPlan = self.inner;
        inner.stages.iter().map(DiagStage::new)
    }

    /// Return final subscripts after all diagonal extraction stages.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 0], &[0]], &[0]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[3, 3], &[3]], &[(0, 1)]).unwrap();
    /// let diag = tree.step_plan(0).unwrap().lhs_diag().unwrap();
    ///
    /// assert_eq!(diag.result_subs(), &[0]);
    /// ```
    #[must_use]
    pub fn result_subs(&self) -> &'a [u32] {
        let inner: &'a InnerDiagPlan = self.inner;
        inner.result_subs.as_slice()
    }
}

/// Read-only metadata for one diagonal extraction stage.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let subs = Subscripts::new(&[&[0, 0], &[0]], &[0]);
/// let tree = ContractionTree::from_pairs(&subs, &[&[3, 3], &[3]], &[(0, 1)]).unwrap();
/// let stage = tree.step_plan(0).unwrap().lhs_diag().unwrap().stages().next().unwrap();
///
/// assert_eq!(stage.axis_pairs(), &[(0, 1)]);
/// ```
#[derive(Clone, Copy)]
pub struct DiagStage<'a> {
    inner: &'a InnerDiagStage,
}

impl<'a> DiagStage<'a> {
    fn new(inner: &'a InnerDiagStage) -> Self {
        Self { inner }
    }

    /// Return axis pairs extracted by this diagonal stage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 0], &[0]], &[0]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[3, 3], &[3]], &[(0, 1)]).unwrap();
    /// let stage = tree.step_plan(0).unwrap().lhs_diag().unwrap().stages().next().unwrap();
    ///
    /// assert_eq!(stage.axis_pairs(), &[(0, 1)]);
    /// ```
    #[must_use]
    pub fn axis_pairs(&self) -> &'a [(usize, usize)] {
        let inner: &'a InnerDiagStage = self.inner;
        inner.axis_pairs.as_slice()
    }

    /// Return subscripts after this diagonal stage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 0], &[0]], &[0]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[3, 3], &[3]], &[(0, 1)]).unwrap();
    /// let stage = tree.step_plan(0).unwrap().lhs_diag().unwrap().stages().next().unwrap();
    ///
    /// assert_eq!(stage.result_subs(), &[0]);
    /// ```
    #[must_use]
    pub fn result_subs(&self) -> &'a [u32] {
        let inner: &'a InnerDiagStage = self.inner;
        inner.result_subs.as_slice()
    }
}

/// Read-only pre-reduction plan for axes unique to one operand.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2]);
/// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
/// let reduce = tree.step_plan(0).unwrap().lhs_reduce().unwrap();
///
/// assert_eq!(reduce.out_shape(), &[3]);
/// ```
#[derive(Clone, Copy)]
pub struct ReducePlan<'a> {
    inner: &'a InnerReducePlan,
}

impl<'a> ReducePlan<'a> {
    fn new(inner: &'a InnerReducePlan) -> Self {
        Self { inner }
    }

    /// Return subscripts before pre-reduction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let reduce = tree.step_plan(0).unwrap().lhs_reduce().unwrap();
    ///
    /// assert_eq!(reduce.original_subs(), &[0, 1]);
    /// ```
    #[must_use]
    pub fn original_subs(&self) -> &'a [u32] {
        let inner: &'a InnerReducePlan = self.inner;
        inner.original_subs.as_slice()
    }

    /// Return subscripts kept after pre-reduction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let reduce = tree.step_plan(0).unwrap().lhs_reduce().unwrap();
    ///
    /// assert_eq!(reduce.kept_subs(), &[1]);
    /// ```
    #[must_use]
    pub fn kept_subs(&self) -> &'a [u32] {
        let inner: &'a InnerReducePlan = self.inner;
        inner.kept_subs.as_slice()
    }

    /// Return the shape after pre-reduction.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let reduce = tree.step_plan(0).unwrap().lhs_reduce().unwrap();
    ///
    /// assert_eq!(reduce.out_shape(), &[3]);
    /// ```
    #[must_use]
    pub fn out_shape(&self) -> &'a [usize] {
        let inner: &'a InnerReducePlan = self.inner;
        inner.out_shape.as_slice()
    }
}

/// Read-only GEMM decomposition plan for a pairwise contraction step.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{ContractionTree, Subscripts};
///
/// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
/// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
/// let gemm = tree.step_plan(0).unwrap().gemm();
///
/// assert_eq!(gemm.lhs_gemm_shape(), &[2, 3]);
/// ```
#[derive(Clone, Copy)]
pub struct GemmPlan<'a> {
    inner: &'a InnerGemmPlan,
}

impl<'a> GemmPlan<'a> {
    fn new(inner: &'a InnerGemmPlan) -> Self {
        Self { inner }
    }

    /// Return modes present only on the left operand and output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.left_only_modes(), &[0]);
    /// ```
    #[must_use]
    pub fn left_only_modes(&self) -> &'a [u32] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.lo_modes.as_slice()
    }

    /// Return modes present only on the right operand and output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.right_only_modes(), &[2]);
    /// ```
    #[must_use]
    pub fn right_only_modes(&self) -> &'a [u32] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.ro_modes.as_slice()
    }

    /// Return modes contracted between the pairwise operands.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.contracted_modes(), &[1]);
    /// ```
    #[must_use]
    pub fn contracted_modes(&self) -> &'a [u32] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.sum_modes.as_slice()
    }

    /// Return modes shared by both operands and preserved in the output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[3, 0, 1], &[1, 2, 3]], &[3, 0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[5, 2, 3], &[3, 4, 5]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.batch_modes(), &[3]);
    /// ```
    #[must_use]
    pub fn batch_modes(&self) -> &'a [u32] {
        let inner: &'a InnerGemmPlan = self.inner;
        let batch_start = inner.lo_modes.len() + inner.ro_modes.len();
        &inner.canonical_modes[batch_start..]
    }

    /// Return target mode order for preparing the left GEMM operand.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.lhs_target_modes(), &[0, 1]);
    /// ```
    #[must_use]
    pub fn lhs_target_modes(&self) -> &'a [u32] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.target_a.as_slice()
    }

    /// Return target mode order for preparing the right GEMM operand.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.rhs_target_modes(), &[1, 2]);
    /// ```
    #[must_use]
    pub fn rhs_target_modes(&self) -> &'a [u32] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.target_b.as_slice()
    }

    /// Return canonical output mode order before any final permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.canonical_output_modes(), &[0, 2]);
    /// ```
    #[must_use]
    pub fn canonical_output_modes(&self) -> &'a [u32] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.canonical_modes.as_slice()
    }

    /// Return fused left-only dimension size.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.m(), 2);
    /// ```
    #[must_use]
    pub fn m(&self) -> usize {
        self.inner.m
    }

    /// Return fused right-only dimension size.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.n(), 4);
    /// ```
    #[must_use]
    pub fn n(&self) -> usize {
        self.inner.n
    }

    /// Return fused contracted dimension size.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.k(), 3);
    /// ```
    #[must_use]
    pub fn k(&self) -> usize {
        self.inner.k
    }

    /// Return prepared left operand GEMM shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.lhs_gemm_shape(), &[2, 3]);
    /// ```
    #[must_use]
    pub fn lhs_gemm_shape(&self) -> &'a [usize] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.a_gemm_shape.as_slice()
    }

    /// Return prepared right operand GEMM shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.rhs_gemm_shape(), &[3, 4]);
    /// ```
    #[must_use]
    pub fn rhs_gemm_shape(&self) -> &'a [usize] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.b_gemm_shape.as_slice()
    }

    /// Return GEMM output shape before expanding fused dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.output_gemm_shape(), &[2, 4]);
    /// ```
    #[must_use]
    pub fn output_gemm_shape(&self) -> &'a [usize] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.c_gemm_shape.as_slice()
    }

    /// Return expanded output shape in canonical output mode order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert_eq!(gemm.expanded_output_shape(), &[2, 4]);
    /// ```
    #[must_use]
    pub fn expanded_output_shape(&self) -> &'a [usize] {
        let inner: &'a InnerGemmPlan = self.inner;
        inner.expanded_shape.as_slice()
    }

    /// Return whether canonical output requires a final permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ContractionTree, Subscripts};
    ///
    /// let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[2, 0]);
    /// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
    /// let gemm = tree.step_plan(0).unwrap().gemm();
    ///
    /// assert!(gemm.needs_final_permute());
    /// ```
    #[must_use]
    pub fn needs_final_permute(&self) -> bool {
        self.inner.needs_final_permute
    }
}
