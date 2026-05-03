//! Shape extent metadata with exactness guarantees.

/// A dimension expression plus the guarantee it provides.
///
/// `Exact` means the expression is the runtime size. `UpperBound` means the
/// runtime size is no larger than the expression. `Unknown` preserves rank
/// when no useful bound is available.
///
/// # Examples
///
/// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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

/// Rank-exact shape metadata.
///
/// The rank is known from the number of extents even when some dimensions are
/// only upper-bounded or unknown.
///
/// # Examples
///
/// ```ignore
/// use tenferro_ops::shape_extent::ShapeMeta;
///
/// let meta = ShapeMeta::exact(vec![2usize, 3]);
/// assert_eq!(meta.rank(), 2);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShapeMeta<D> {
    extents: Vec<ShapeExtent<D>>,
}

impl<D> ShapeMeta<D> {
    /// Construct shape metadata from per-axis extents.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_ops::shape_extent::{ShapeExtent, ShapeMeta};
    ///
    /// let meta = ShapeMeta::new(vec![ShapeExtent::upper_bound(4usize)]);
    /// assert_eq!(meta.rank(), 1);
    /// ```
    pub fn new(extents: Vec<ShapeExtent<D>>) -> Self {
        Self { extents }
    }

    /// Construct shape metadata whose every axis is exact.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_ops::shape_extent::ShapeMeta;
    ///
    /// let meta = ShapeMeta::exact(vec![2usize, 3]);
    /// assert_eq!(meta.exact_shape(), Some(vec![2usize, 3]));
    /// ```
    pub fn exact(shape: Vec<D>) -> Self {
        Self::new(shape.into_iter().map(ShapeExtent::Exact).collect())
    }

    /// Return the rank represented by this metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_ops::shape_extent::ShapeMeta;
    ///
    /// assert_eq!(ShapeMeta::exact(vec![2usize, 3]).rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    /// Return the per-axis extents.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_ops::shape_extent::ShapeMeta;
    ///
    /// let meta = ShapeMeta::exact(vec![2usize]);
    /// assert_eq!(meta.extents().len(), 1);
    /// ```
    pub fn extents(&self) -> &[ShapeExtent<D>] {
        &self.extents
    }
}

impl<D: Clone> ShapeMeta<D> {
    /// Return the exact shape only when every axis is exact.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_ops::shape_extent::{ShapeExtent, ShapeMeta};
    ///
    /// let meta = ShapeMeta::new(vec![ShapeExtent::upper_bound(4usize)]);
    /// assert_eq!(meta.exact_shape(), None);
    /// ```
    pub fn exact_shape(&self) -> Option<Vec<D>> {
        self.extents
            .iter()
            .map(|extent| extent.as_exact().cloned())
            .collect()
    }

    /// Return the bound shape only when every axis has a bound.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_ops::shape_extent::{ShapeExtent, ShapeMeta};
    ///
    /// let meta = ShapeMeta::new(vec![ShapeExtent::upper_bound(4usize)]);
    /// assert_eq!(meta.bound_shape(), Some(vec![4usize]));
    /// ```
    pub fn bound_shape(&self) -> Option<Vec<D>> {
        self.extents
            .iter()
            .map(|extent| extent.bound_expr().cloned())
            .collect()
    }
}
