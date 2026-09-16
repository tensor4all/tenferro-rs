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
    fn tag(&self) -> Self::Tag;
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
///         F64 => f64,
///         /// Single precision.
///         F32 => f32,
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
                $variant:ident => $ty:ty
            ),+ $(,)?
        }
        $(#[$set_meta:meta])*
        $set_vis:vis enum $set:ident;
    ) => {
        $(#[$tag_meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        $tag_vis enum $tag {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        $(#[$set_meta])*
        #[derive(Clone, Debug, PartialEq)]
        $set_vis enum $set {
            $(
                $(#[$variant_meta])*
                $variant($crate::HostTensor<$ty>),
            )+
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
        }
    };
}
