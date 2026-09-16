//! Closed scalar sets.
//!
//! A scalar set is the closed list of scalar types one tensor value type can
//! carry. tenferro declares its own set once and gets the value enum, the tag,
//! and the membership query from that single declaration. A crate that needs a
//! different set declares it with [`define_scalar_set!`] in its own crate, and
//! its values implement this trait without touching tenferro's set.

/// A closed set of scalar types carried by one tensor value type.
///
/// The set is represented by the value enum itself: each of its variants holds a
/// host tensor of one member type, and [`ScalarSet::tag`] reports which member a
/// value currently holds.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{DefaultScalars, ScalarSet};
///
/// let value = DefaultScalars::F64(tenferro_tensor_core::HostTensor::from_vec_col_major(vec![1], vec![1.0_f64])?);
/// assert_eq!(value.tag(), tenferro_tensor_core::DType::F64);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
pub trait ScalarSet: Clone + core::fmt::Debug + 'static {
    /// Tag identifying one member of this set.
    type Tag: Copy + Eq + core::fmt::Debug + 'static;

    /// Tags of every member, in declaration order.
    const TAGS: &'static [Self::Tag];

    /// Tag of the member this value currently holds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DType, HostTensor, ScalarSet};
    ///
    /// let value = tenferro_tensor_core::DefaultScalars::F64(
    ///     HostTensor::from_vec_col_major(vec![1], vec![1.0_f64])?,
    /// );
    /// assert_eq!(value.tag(), DType::F64);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    fn tag(&self) -> Self::Tag;

    /// Promote two members of this set to the member that represents both.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{DType, ScalarSet};
    ///
    /// assert_eq!(
    ///     <tenferro_tensor_core::DefaultScalars as ScalarSet>::promote(DType::I32, DType::F32),
    ///     DType::F64
    /// );
    /// ```
    fn promote(lhs: Self::Tag, rhs: Self::Tag) -> Self::Tag;
}

/// Kind of arithmetic a set member belongs to.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::MemberKind;
///
/// assert_ne!(MemberKind::Integer, MemberKind::Float);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemberKind {
    /// Boolean.
    Boolean,
    /// Signed integer.
    Integer,
    /// Real floating point.
    Float,
    /// Complex floating point.
    Complex,
    /// A scalar tenferro does not declare, so its facts are unknown.
    External,
}

/// Promotion-relevant facts about one set member.
///
/// `level` orders members within a kind and `width` is the component width in
/// bits. `level` is what orders two members of the same kind, so a set can rank
/// an extended-precision member above a standard one without claiming a wider
/// exponent range.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{MemberKind, MemberSpec};
///
/// let widened = tenferro_tensor_core::promote_specs(
///     MemberSpec::new(MemberKind::Float, 0, 32),
///     MemberSpec::new(MemberKind::Float, 1, 64),
/// );
/// assert_eq!(widened, MemberSpec::new(MemberKind::Float, 1, 64));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemberSpec {
    /// Arithmetic kind of the member.
    pub kind: MemberKind,
    /// Rank within the kind.
    pub level: u32,
    /// Component width in bits.
    pub width: u32,
}

impl MemberSpec {
    /// Build a member fact.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor_core::{MemberKind, MemberSpec};
    ///
    /// let spec = MemberSpec::new(MemberKind::Float, 1, 64);
    /// assert_eq!(spec.level, 1);
    /// ```
    #[must_use]
    pub const fn new(kind: MemberKind, level: u32, width: u32) -> Self {
        Self { kind, level, width }
    }
}

/// Combine two member facts under the ordinary numeric promotion rules.
///
/// A boolean yields to anything. Two members of one kind keep the higher level.
/// An integer with a float or complex yields to that kind's widest member,
/// because an integer is widened rather than mixed. A float with a complex takes
/// the narrowest complex that can still hold both, which keeps `f32 + c32` in
/// `c32` while `f64 + c32` becomes `c64`.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{promote_specs, MemberKind, MemberSpec};
///
/// let f32_ = MemberSpec::new(MemberKind::Float, 0, 32);
/// let c32 = MemberSpec::new(MemberKind::Complex, 0, 32);
/// assert_eq!(promote_specs(f32_, c32), c32);
/// ```
#[must_use]
pub const fn promote_specs(lhs: MemberSpec, rhs: MemberSpec) -> MemberSpec {
    use MemberKind::{Boolean, Complex, External, Float, Integer};
    match (lhs.kind, rhs.kind) {
        (Boolean, _) => rhs,
        (_, Boolean) => lhs,
        (External, _) => lhs,
        (_, External) => rhs,
        (Integer, Float) | (Float, Integer) => MemberSpec::new(Float, u32::MAX, u32::MAX),
        (Integer, Complex) | (Complex, Integer) => MemberSpec::new(Complex, u32::MAX, u32::MAX),
        (Integer, Integer) | (Float, Float) | (Complex, Complex) => {
            if lhs.level >= rhs.level {
                lhs
            } else {
                rhs
            }
        }
        (Float, Complex) | (Complex, Float) => {
            let width = if lhs.width >= rhs.width {
                lhs.width
            } else {
                rhs.width
            };
            MemberSpec::new(Complex, 0, width)
        }
    }
}

/// Define a closed scalar set: its tag type, its value enum, and its membership.
///
/// The declaration lists each member once. The macro emits the tag enum, the
/// value enum whose variants hold a
/// [`HostTensor`](crate::HostTensor) of the member type, and the
/// [`ScalarSet`] implementation. A downstream crate invokes this in its own
/// crate, so tenferro never needs to know the set.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor_core::{define_scalar_set, HostTensor, ScalarSet};
///
/// define_scalar_set! {
///     /// Tag for a two-member set.
///     pub enum PairTag {
///         /// Double precision.
///         F64 => f64 : Float 1 64,
///         /// Single precision.
///         F32 => f32 : Float 0 32,
///     }
///     /// Value enum for a two-member set.
///     pub enum Pair;
/// }
///
/// let value = Pair::F32(HostTensor::from_vec_col_major(vec![1], vec![1.0_f32])?);
/// assert_eq!(value.tag(), PairTag::F32);
/// assert_eq!(<Pair as ScalarSet>::TAGS, &[PairTag::F64, PairTag::F32]);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
#[macro_export]
macro_rules! define_scalar_set {
    (
        $(#[$tag_meta:meta])*
        $tag_vis:vis enum $tag:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $ty:ty : $kind:ident $level:literal $width:literal
            ),+ $(,)?
        }
        $(#[$set_meta:meta])*
        $set_vis:vis enum $set:ident;
        $( external $ext_variant:ident($ext_ty:ty); )?
    ) => {
        $(#[$tag_meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        $tag_vis enum $tag {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
            $( $ext_variant($ext_ty), )?
        }

        $(#[$set_meta])*
        #[derive(Clone, Debug, PartialEq)]
        $set_vis enum $set {
            $(
                $(#[$variant_meta])*
                $variant($crate::HostTensor<$ty>),
            )+
        }

        impl $tag {
            /// Promotion facts of this member.
            ///
            /// # Examples
            ///
            /// ```rust
            /// use tenferro_tensor_core::{MemberKind, DType};
            ///
            /// assert_eq!(DType::F64.spec().kind, MemberKind::Float);
            /// ```
            #[must_use]
            pub const fn spec(self) -> $crate::MemberSpec {
                match self {
                    $(
                        $tag::$variant => $crate::MemberSpec::new(
                            $crate::MemberKind::$kind,
                            $level,
                            $width,
                        ),
                    )+
                    $( $tag::$ext_variant(_) => {
                        $crate::MemberSpec::new($crate::MemberKind::External, 0, 0)
                    } )?
                }
            }

            /// Every member tag, in declaration order.
            pub const TAGS: &'static [Self] = &[
                $(
                    $tag::$variant,
                )+
            ];

            /// Promotion facts of every member, in declaration order.
            pub const SPECS: &'static [$crate::MemberSpec] = &[
                $(
                    $crate::MemberSpec::new($crate::MemberKind::$kind, $level, $width),
                )+
            ];
        }

        impl $crate::ScalarSet for $set {
            type Tag = $tag;

            const TAGS: &'static [Self::Tag] = &[
                $(
                    $tag::$variant,
                )+
            ];

            fn tag(&self) -> Self::Tag {
                match self {
                    $(
                        $set::$variant(_) => $tag::$variant,
                    )+
                }
            }

            fn promote(lhs: Self::Tag, rhs: Self::Tag) -> Self::Tag {
                $(
                    if matches!(lhs, $tag::$ext_variant(_)) {
                        return lhs;
                    }
                    if matches!(rhs, $tag::$ext_variant(_)) {
                        return rhs;
                    }
                )?
                let target = $crate::promote_specs(lhs.spec(), rhs.spec());
                for (index, spec) in <$tag>::SPECS.iter().enumerate() {
                    if *spec == target {
                        return <$tag>::TAGS[index];
                    }
                }
                let mut chosen: Option<(usize, $crate::MemberSpec)> = None;
                for (index, spec) in <$tag>::SPECS.iter().enumerate() {
                    if spec.kind != target.kind {
                        continue;
                    }
                    let better = match chosen {
                        None => true,
                        Some((_, current)) => {
                            let current_wide_enough = current.width >= target.width;
                            let candidate_wide_enough = spec.width >= target.width;
                            match (current_wide_enough, candidate_wide_enough) {
                                (false, true) => true,
                                (true, false) => false,
                                (true, true) => {
                                    spec.width < current.width
                                        || (spec.width == current.width
                                            && spec.level < current.level)
                                }
                                (false, false) => {
                                    spec.width > current.width
                                        || (spec.width == current.width
                                            && spec.level < current.level)
                                }
                            }
                        }
                    };
                    if better {
                        chosen = Some((index, *spec));
                    }
                }
                match chosen {
                    Some((index, _)) => <$tag>::TAGS[index],
                    None => lhs,
                }
            }
        }
    };
}

#[cfg(test)]
mod tests;
