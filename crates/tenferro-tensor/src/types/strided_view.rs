/// One-axis slice specification for typed view slicing.
///
/// `start` and `end` always name the **ascending** half-open interval
/// `[start, end)` on the axis, with negative bounds counting from the end of
/// the axis; a negative `step` then walks that interval in reverse. The
/// endpoint roles therefore do not depend on the sign of `step`, and this is
/// *not* ndarray's range-with-step model, where a negative `step` makes
/// `start` the high endpoint and `end` the low one. Full-axis reversal is
/// [`StridedSliceSpec::reverse`] (`new(0, None, -1)`); under the ndarray model
/// that spelling would select only the first element.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{StridedSliceSpec, TypedTensorView};
///
/// let data = [0_i32, 1, 2, 3, 4, 5];
/// let view = TypedTensorView::from_slice(vec![6], vec![1], 0, &data)?;
///
/// // Ascending selection `[0, 3)` walked backwards with step -2: 2, 0.
/// let reversed = view.slice_view(&[StridedSliceSpec::new(0, Some(3), -2)])?;
/// assert_eq!(reversed.shape(), &[2]);
/// assert_eq!(reversed.get(&[0]), Some(&2));
/// assert_eq!(reversed.get(&[1]), Some(&0));
///
/// // `start` is the low endpoint even for a negative step, so a range that
/// // looks descending under ndarray's model is empty here.
/// let empty = view.slice_view(&[StridedSliceSpec::new(4, Some(1), -1)])?;
/// assert_eq!(empty.shape(), &[0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StridedSliceSpec {
    start: isize,
    end: Option<isize>,
    step: isize,
}

impl StridedSliceSpec {
    /// Create a slice from a signed start, optional exclusive end, and step.
    ///
    /// `start` and `end` are the low and high endpoints of the ascending
    /// half-open interval `[start, end)`; a negative `step` reverses that
    /// interval instead of re-reading the endpoints as high and low. See the
    /// [type documentation](Self) for the full model.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// let slice = StridedSliceSpec::new(1, Some(-1), 1);
    /// assert_eq!(slice.start(), 1);
    /// assert_eq!(slice.end(), Some(-1));
    /// ```
    pub fn new(start: isize, end: Option<isize>, step: isize) -> Self {
        Self { start, end, step }
    }

    /// Select the full axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::all(), StridedSliceSpec::new(0, None, 1));
    /// ```
    pub fn all() -> Self {
        Self::new(0, None, 1)
    }

    /// Select the full axis in reverse order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::reverse(), StridedSliceSpec::new(0, None, -1));
    /// ```
    pub fn reverse() -> Self {
        Self::new(0, None, -1)
    }

    /// Return the signed start bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::new(-3, None, 1).start(), -3);
    /// ```
    pub fn start(&self) -> isize {
        self.start
    }

    /// Return the optional signed exclusive end bound.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::new(0, Some(2), 1).end(), Some(2));
    /// ```
    pub fn end(&self) -> Option<isize> {
        self.end
    }

    /// Return the signed step.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::StridedSliceSpec;
    ///
    /// assert_eq!(StridedSliceSpec::reverse().step(), -1);
    /// ```
    pub fn step(&self) -> isize {
        self.step
    }
}
