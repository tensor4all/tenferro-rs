//! Shape extent metadata with exactness guarantees.

/// A dimension expression plus the guarantee it provides.
///
/// `Exact` means the expression is the runtime size. `UpperBound` means the
/// runtime size is no larger than the expression. `Unknown` preserves rank
/// when no useful bound is available.
///
/// # Examples
///
/// ```
/// use tenferro_ops::shape_extent::ShapeExtent;
///
/// let extent = ShapeExtent::exact(4usize);
/// assert_eq!(extent.as_exact(), Some(&4));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShapeExtent<D> {
    /// The dimension expression is exact.
    Exact(D),
    /// The dimension expression is an upper bound.
    UpperBound(D),
    /// The dimension is rank-known but otherwise unknown.
    Unknown,
}

impl<D> ShapeExtent<D> {
    /// Construct an exact extent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::shape_extent::ShapeExtent;
    ///
    /// let extent = ShapeExtent::exact(3usize);
    /// assert!(extent.is_exact());
    /// ```
    pub fn exact(dim: D) -> Self {
        Self::Exact(dim)
    }

    /// Construct an upper-bound extent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::shape_extent::ShapeExtent;
    ///
    /// let extent = ShapeExtent::upper_bound(3usize);
    /// assert!(!extent.is_exact());
    /// ```
    pub fn upper_bound(dim: D) -> Self {
        Self::UpperBound(dim)
    }

    /// Construct an unknown extent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::shape_extent::ShapeExtent;
    ///
    /// let extent: ShapeExtent<usize> = ShapeExtent::unknown();
    /// assert_eq!(extent.bound_expr(), None);
    /// ```
    pub fn unknown() -> Self {
        Self::Unknown
    }

    /// Return true when this extent is exact.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::shape_extent::ShapeExtent;
    ///
    /// assert!(ShapeExtent::exact(2usize).is_exact());
    /// assert!(!ShapeExtent::upper_bound(2usize).is_exact());
    /// ```
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }

    /// Return the exact dimension expression, if this extent is exact.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::shape_extent::ShapeExtent;
    ///
    /// assert_eq!(ShapeExtent::exact(5usize).as_exact(), Some(&5));
    /// assert_eq!(ShapeExtent::upper_bound(5usize).as_exact(), None);
    /// ```
    pub fn as_exact(&self) -> Option<&D> {
        match self {
            Self::Exact(dim) => Some(dim),
            Self::UpperBound(_) | Self::Unknown => None,
        }
    }

    /// Return the known bound expression, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::shape_extent::ShapeExtent;
    ///
    /// assert_eq!(ShapeExtent::upper_bound(5usize).bound_expr(), Some(&5));
    /// ```
    pub fn bound_expr(&self) -> Option<&D> {
        match self {
            Self::Exact(dim) | Self::UpperBound(dim) => Some(dim),
            Self::Unknown => None,
        }
    }

    /// Map the contained dimension expression while preserving exactness.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_ops::shape_extent::ShapeExtent;
    ///
    /// let extent = ShapeExtent::upper_bound(5usize).map(|dim| dim + 1);
    /// assert_eq!(extent.bound_expr(), Some(&6));
    /// ```
    pub fn map<E>(self, f: impl FnOnce(D) -> E) -> ShapeExtent<E> {
        match self {
            Self::Exact(dim) => ShapeExtent::Exact(f(dim)),
            Self::UpperBound(dim) => ShapeExtent::UpperBound(f(dim)),
            Self::Unknown => ShapeExtent::Unknown,
        }
    }
}
