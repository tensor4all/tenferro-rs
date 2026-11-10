//! Einsum spec, options and the concrete input-list adapter (#2552 §6.2).

use tenferro_tensor::{AsView, Conj, Repr, TensorRead};

use crate::{EinsumNotation, EinsumOptimize, EinsumSubscripts};

/// An einsum specification: text, parsed notation or integer subscripts.
///
/// Construction never fails; the text is parsed and every form is validated
/// inside the operation, which reports a malformed spec as an
/// `InvalidSubscripts` cause with kind `Validation(InvalidArgument)`.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{EinsumNotation, EinsumSubscripts, Spec};
///
/// let text: Spec<'_> = "ij,jk->ik".into();
/// assert!(matches!(text, Spec::Text("ij,jk->ik")));
///
/// let notation = EinsumNotation::parse("ij->ji")?;
/// assert!(matches!(Spec::from(&notation), Spec::Notation(_)));
///
/// let subscripts = EinsumSubscripts::new(&[&[0, 1]], &[1, 0]);
/// assert!(matches!(Spec::from(&subscripts), Spec::Subscripts(_)));
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub enum Spec<'a> {
    /// NumPy-style text such as `"ij,jk->ik"`, parsed inside the operation.
    Text(&'a str),
    /// Parsed notation, possibly with ellipses.
    Notation(&'a EinsumNotation),
    /// Integer subscripts without ellipses.
    Subscripts(&'a EinsumSubscripts),
}

impl<'a> From<&'a str> for Spec<'a> {
    fn from(text: &'a str) -> Self {
        Self::Text(text)
    }
}

impl<'a> From<&'a String> for Spec<'a> {
    fn from(text: &'a String) -> Self {
        Self::Text(text)
    }
}

impl<'a> From<&'a EinsumNotation> for Spec<'a> {
    fn from(notation: &'a EinsumNotation) -> Self {
        Self::Notation(notation)
    }
}

impl<'a> From<&'a EinsumSubscripts> for Spec<'a> {
    fn from(subscripts: &'a EinsumSubscripts) -> Self {
        Self::Subscripts(subscripts)
    }
}

/// Anything an einsum accepts as its spec argument: `str`, `String`,
/// [`EinsumNotation`], [`EinsumSubscripts`] or a [`Spec`]. Operations take it
/// by reference, so `"ij,jk->ik"`, `&notation` and `&subscripts` all work.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{EinsumNotation, EinsumSpec, Spec};
///
/// assert!(matches!("ij->j".spec(), Spec::Text("ij->j")));
/// let notation = EinsumNotation::parse("ij->j")?;
/// assert!(matches!(notation.spec(), Spec::Notation(_)));
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
pub trait EinsumSpec {
    /// The spec this value denotes; never fails and never parses.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumSpec, Spec};
    ///
    /// let owned = String::from("ij,jk->ik");
    /// assert!(matches!(owned.spec(), Spec::Text("ij,jk->ik")));
    /// ```
    fn spec(&self) -> Spec<'_>;
}

impl EinsumSpec for str {
    fn spec(&self) -> Spec<'_> {
        Spec::Text(self)
    }
}

impl EinsumSpec for String {
    fn spec(&self) -> Spec<'_> {
        Spec::Text(self)
    }
}

impl EinsumSpec for EinsumNotation {
    fn spec(&self) -> Spec<'_> {
        Spec::Notation(self)
    }
}

impl EinsumSpec for EinsumSubscripts {
    fn spec(&self) -> Spec<'_> {
        Spec::Subscripts(self)
    }
}

impl EinsumSpec for Spec<'_> {
    fn spec(&self) -> Spec<'_> {
        *self
    }
}

/// Per-operand conjugation of an einsum: operand `i` is conjugated when entry
/// `i` is [`Conj::Yes`]. Missing entries are [`Conj::No`]; the default
/// conjugates nothing.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::ConjMask;
/// use tenferro_tensor::Conj;
///
/// let mask = ConjMask::new([Conj::Yes, Conj::No]);
/// assert_eq!(mask.get(0), Conj::Yes);
/// assert_eq!(mask.get(5), Conj::No);
/// assert!(ConjMask::default().is_none());
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConjMask {
    operands: Vec<Conj>,
}

impl ConjMask {
    /// A mask with one entry per leading operand.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::ConjMask;
    /// use tenferro_tensor::Conj;
    ///
    /// assert_eq!(ConjMask::new([Conj::No, Conj::Yes]).len(), 2);
    /// ```
    pub fn new(operands: impl IntoIterator<Item = Conj>) -> Self {
        Self {
            operands: operands.into_iter().collect(),
        }
    }

    /// Conjugation of operand `index` ([`Conj::No`] past the end).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::ConjMask;
    /// use tenferro_tensor::Conj;
    ///
    /// assert_eq!(ConjMask::new([Conj::Yes]).get(0), Conj::Yes);
    /// ```
    #[must_use]
    pub fn get(&self, index: usize) -> Conj {
        self.operands.get(index).copied().unwrap_or_default()
    }

    /// Number of explicit entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::ConjMask;
    ///
    /// assert_eq!(ConjMask::default().len(), 0);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.operands.len()
    }

    /// `true` when there are no explicit entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::ConjMask;
    ///
    /// assert!(ConjMask::default().is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operands.is_empty()
    }

    /// `true` when no operand is conjugated.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::ConjMask;
    /// use tenferro_tensor::Conj;
    ///
    /// assert!(ConjMask::new([Conj::No]).is_none());
    /// ```
    #[must_use]
    pub fn is_none(&self) -> bool {
        self.operands.iter().all(|c| !c.is_yes())
    }
}

/// Options of [`einsum`](crate::EinsumOps::einsum_with) on every layer.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::{ConjMask, EinsumOptimize, EinsumOptions};
/// use tenferro_tensor::Conj;
///
/// let options = EinsumOptions::default()
///     .optimize(EinsumOptimize::Path(vec![(1, 2), (0, 1)]))
///     .conj(ConjMask::new([Conj::Yes]));
/// assert!(matches!(options.optimize, EinsumOptimize::Path(_)));
/// assert_eq!(options.conj.get(0), Conj::Yes);
/// ```
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct EinsumOptions {
    /// Contraction order: automatic (default), left to right, or an explicit
    /// path, nested grouping or tree. An explicit order is always honoured.
    pub optimize: EinsumOptimize,
    /// Per-operand conjugation; the default conjugates nothing.
    pub conj: ConjMask,
}

impl EinsumOptions {
    /// Return options with the given contraction order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{EinsumOptimize, EinsumOptions};
    ///
    /// let o = EinsumOptions::default().optimize(EinsumOptimize::False);
    /// assert!(matches!(o.optimize, EinsumOptimize::False));
    /// ```
    pub fn optimize(mut self, optimize: EinsumOptimize) -> Self {
        self.optimize = optimize;
        self
    }

    /// Return options with the given per-operand conjugation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::{ConjMask, EinsumOptions};
    /// use tenferro_tensor::Conj;
    ///
    /// let o = EinsumOptions::default().conj(ConjMask::new([Conj::No, Conj::Yes]));
    /// assert_eq!(o.conj.get(1), Conj::Yes);
    /// ```
    pub fn conj(mut self, conj: ConjMask) -> Self {
        self.conj = conj;
        self
    }

    /// `true` when the options select the automatic order without annealing,
    /// which plans exactly as the default, and conjugate nothing.
    pub(crate) fn is_plain(&self) -> bool {
        self.conj.is_none()
            && matches!(&self.optimize, crate::EinsumOptimize::Auto(o) if !o.anneals())
    }
}

/// Options of [`tensordot`](crate::TensordotOps::tensordot_with). There is
/// nothing to choose yet; the type exists so every operation has an options
/// argument.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::TensordotOptions;
///
/// let _ = TensordotOptions::default();
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct TensordotOptions {}

mod sealed {
    pub trait Sealed {}
}

/// The input list of a concrete einsum: a slice or array of values that
/// implement [`AsView`] with one representation. Mix owned tensors and views
/// by borrowing every element (`[a.as_read(), b]`); that copies no data.
///
/// # Examples
///
/// ```
/// use tenferro_einsum::EinsumInputs;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
/// let inputs = [&a, &a];
/// assert_eq!(inputs.reads().len(), 2);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait EinsumInputs: sealed::Sealed {
    /// Representation of the inputs, which the result takes.
    type Repr: Repr;

    /// Borrow every input; metadata only, never copies element data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_einsum::EinsumInputs;
    /// use tenferro_tensor::Tensor;
    ///
    /// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])?;
    /// let inputs = vec![a, b];
    /// let reads = inputs.reads();
    /// assert_eq!(reads[0].shape(), &[2]);
    /// assert_eq!(reads[1].shape(), &[3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn reads(&self) -> Vec<TensorRead<'_>>;
}

impl<X: AsView> sealed::Sealed for [X] {}
impl<X: AsView, const N: usize> sealed::Sealed for [X; N] {}
impl<X: AsView> sealed::Sealed for Vec<X> {}

impl<X: AsView> EinsumInputs for [X] {
    type Repr = X::Repr;

    fn reads(&self) -> Vec<TensorRead<'_>> {
        self.iter().map(AsView::as_read).collect()
    }
}

impl<X: AsView, const N: usize> EinsumInputs for [X; N] {
    type Repr = X::Repr;

    fn reads(&self) -> Vec<TensorRead<'_>> {
        self.iter().map(AsView::as_read).collect()
    }
}

impl<X: AsView> EinsumInputs for Vec<X> {
    type Repr = X::Repr;

    fn reads(&self) -> Vec<TensorRead<'_>> {
        self.iter().map(AsView::as_read).collect()
    }
}
