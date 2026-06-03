/// One-axis slice specification for typed view slicing.
///
/// This follows ndarray's range-with-step model: negative `start` or `end`
/// values count from the end of the axis, `end` is exclusive, and negative
/// `step` reverses the selected range before stepping.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::StridedSliceSpec;
///
/// let every_other_reversed = StridedSliceSpec::new(0, None, -2);
/// assert_eq!(every_other_reversed.step(), -2);
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
