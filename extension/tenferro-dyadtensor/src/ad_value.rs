use chainrules_scalarops as scalarops;
use core::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::atomic::{AtomicU64, Ordering};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

use crate::structured::StructuredTensor;
use crate::{reverse_tape, Error, Result};

/// Automatic differentiation mode.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::AdMode;
///
/// assert_eq!(AdMode::Primal, AdMode::Primal);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdMode {
    /// Plain evaluation without derivative propagation.
    Primal,
    /// Forward-mode value carrying tangent information.
    Forward,
    /// Reverse-mode value carrying graph metadata.
    Reverse,
}

/// Opaque identifier of a reverse-mode graph node.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::NodeId;
///
/// let node = NodeId(7);
/// assert_eq!(node.0, 7);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Opaque identifier of a tape instance.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::TapeId;
///
/// let tape = TapeId(2);
/// assert_eq!(tape.0, 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TapeId(pub u64);

/// Generic AD value that can wrap any user-defined payload type `T`.
///
/// This is the primary extension point of the crate.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdValue, NodeId, TapeId};
///
/// let primal = AdValue::primal(3.0_f64);
/// assert_eq!(primal.mode(), AdMode::Primal);
///
/// let dual = AdValue::forward(3.0_f64, 1.0_f64);
/// assert_eq!(dual.mode(), AdMode::Forward);
///
/// let tracked = AdValue::reverse(3.0_f64, NodeId(1), TapeId(9), None);
/// assert_eq!(tracked.mode(), AdMode::Reverse);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum AdValue<T> {
    /// Primal-only value.
    Primal(T),
    /// Forward-mode value and tangent.
    Forward { primal: T, tangent: T },
    /// Reverse-mode value with graph metadata.
    Reverse {
        primal: T,
        node: NodeId,
        tape: TapeId,
        tangent: Option<T>,
    },
}

impl<T> AdValue<T> {
    /// Creates a primal-only value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::primal(2_i32);
    /// assert!(matches!(x, AdValue::Primal(2)));
    /// ```
    pub fn primal(value: T) -> Self {
        Self::Primal(value)
    }

    /// Creates a forward-mode value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::forward(2.0_f64, 1.0_f64);
    /// assert!(matches!(x, AdValue::Forward { .. }));
    /// ```
    pub fn forward(primal: T, tangent: T) -> Self {
        Self::Forward { primal, tangent }
    }

    /// Creates a reverse-mode value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(2.0_f64, NodeId(3), TapeId(5), Some(0.1));
    /// assert!(matches!(x, AdValue::Reverse { .. }));
    /// ```
    pub fn reverse(primal: T, node: NodeId, tape: TapeId, tangent: Option<T>) -> Self {
        Self::Reverse {
            primal,
            node,
            tape,
            tangent,
        }
    }

    /// Returns the AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdValue};
    ///
    /// let x = AdValue::forward(1.0_f64, 1.0_f64);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn mode(&self) -> AdMode {
        match self {
            Self::Primal(_) => AdMode::Primal,
            Self::Forward { .. } => AdMode::Forward,
            Self::Reverse { .. } => AdMode::Reverse,
        }
    }

    /// Returns a reference to the primal payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::forward(10_i32, 1_i32);
    /// assert_eq!(x.primal_ref(), &10);
    /// ```
    pub fn primal_ref(&self) -> &T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
            Self::Reverse { primal, .. } => primal,
        }
    }

    /// Returns a mutable reference to the primal payload.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let mut x = AdValue::primal(1_i32);
    /// *x.primal_mut() = 7;
    /// assert_eq!(x.primal_ref(), &7);
    /// ```
    pub fn primal_mut(&mut self) -> &mut T {
        match self {
            Self::Primal(value) => value,
            Self::Forward { primal, .. } => primal,
            Self::Reverse { primal, .. } => primal,
        }
    }

    /// Returns a reference to tangent payload when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::forward(2.0_f64, 3.0_f64);
    /// assert_eq!(x.tangent_ref(), Some(&3.0));
    /// ```
    pub fn tangent_ref(&self) -> Option<&T> {
        match self {
            Self::Primal(_) => None,
            Self::Forward { tangent, .. } => Some(tangent),
            Self::Reverse { tangent, .. } => tangent.as_ref(),
        }
    }

    /// Returns reverse-mode node id when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(1.0_f64, NodeId(4), TapeId(6), None);
    /// assert_eq!(x.node_id(), Some(NodeId(4)));
    /// ```
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            Self::Reverse { node, .. } => Some(*node),
            _ => None,
        }
    }

    /// Returns reverse-mode tape id when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdValue, NodeId, TapeId};
    ///
    /// let x = AdValue::reverse(1.0_f64, NodeId(4), TapeId(6), None);
    /// assert_eq!(x.tape_id(), Some(TapeId(6)));
    /// ```
    pub fn tape_id(&self) -> Option<TapeId> {
        match self {
            Self::Reverse { tape, .. } => Some(*tape),
            _ => None,
        }
    }

    /// Maps the payload type while preserving AD metadata.
    ///
    /// In reverse mode this preserves the existing `node` / `tape` metadata, so
    /// it is only appropriate for identity-same-cotangent-space transforms.
    /// Dtype-changing or otherwise graph-changing reverse transforms must
    /// register explicit pullback rules instead of using this helper directly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdValue;
    ///
    /// let x = AdValue::forward(2_i32, 3_i32);
    /// let y = x.map_preserving_metadata(|v| v as f64);
    /// assert_eq!(y.primal_ref(), &2.0_f64);
    /// assert_eq!(y.tangent_ref(), Some(&3.0_f64));
    /// ```
    pub fn map_preserving_metadata<U>(self, mut f: impl FnMut(T) -> U) -> AdValue<U> {
        match self {
            Self::Primal(value) => AdValue::Primal(f(value)),
            Self::Forward { primal, tangent } => AdValue::Forward {
                primal: f(primal),
                tangent: f(tangent),
            },
            Self::Reverse {
                primal,
                node,
                tape,
                tangent,
            } => AdValue::Reverse {
                primal: f(primal),
                node,
                tape,
                tangent: tangent.map(f),
            },
        }
    }
}

impl<T> From<T> for AdValue<T> {
    fn from(value: T) -> Self {
        Self::Primal(value)
    }
}

/// Scalar newtype carrying AD mode information.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdScalar};
///
/// let x: AdScalar<f64> = 2.0_f64.into();
/// assert_eq!(x.mode(), AdMode::Primal);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AdScalar<T>(pub AdValue<T>);

impl<T> AdScalar<T> {
    /// Creates a primal scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdScalar};
    ///
    /// let x = AdScalar::new_primal(1.5_f64);
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn new_primal(value: T) -> Self {
        Self(AdValue::primal(value))
    }

    /// Creates a forward-mode scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdScalar};
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn new_forward(primal: T, tangent: T) -> Self {
        Self(AdValue::forward(primal, tangent))
    }

    /// Creates a reverse-mode scalar.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdScalar, NodeId, TapeId};
    ///
    /// let x = AdScalar::new_reverse(2.0_f64, NodeId(1), TapeId(2), Some(0.4));
    /// assert_eq!(x.mode(), AdMode::Reverse);
    /// ```
    pub fn new_reverse(primal: T, node: NodeId, tape: TapeId, tangent: Option<T>) -> Self {
        Self(AdValue::reverse(primal, node, tape, tangent))
    }

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdScalar};
    ///
    /// let x = AdScalar::new_primal(2.0_f64);
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn mode(&self) -> AdMode {
        self.0.mode()
    }

    /// Returns reference to underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdScalar, AdValue};
    ///
    /// let x = AdScalar::new_primal(2.0_f64);
    /// assert!(matches!(x.as_value(), AdValue::Primal(_)));
    /// ```
    pub fn as_value(&self) -> &AdValue<T> {
        &self.0
    }

    /// Consumes wrapper and returns the underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdScalar, AdValue};
    ///
    /// let x = AdScalar::new_primal(2.0_f64).into_value();
    /// assert!(matches!(x, AdValue::Primal(_)));
    /// ```
    pub fn into_value(self) -> AdValue<T> {
        self.0
    }

    /// Consumes this wrapper and returns only the primal scalar value.
    ///
    /// This is an explicit AD-drop API for intentional metadata discard.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdScalar;
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.into_primal(), 2.0);
    /// ```
    pub fn into_primal(self) -> T {
        match self.0 {
            AdValue::Primal(primal) => primal,
            AdValue::Forward { primal, .. } => primal,
            AdValue::Reverse { primal, .. } => primal,
        }
    }

    /// Returns primal scalar reference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdScalar;
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.primal(), &2.0);
    /// ```
    pub fn primal(&self) -> &T {
        self.0.primal_ref()
    }

    /// Returns tangent scalar reference when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdScalar;
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// assert_eq!(x.tangent(), Some(&1.0));
    /// ```
    pub fn tangent(&self) -> Option<&T> {
        self.0.tangent_ref()
    }

    /// Applies scalar conjugation with AD propagation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use tenferro_dyadtensor::AdScalar;
    ///
    /// let x = AdScalar::new_forward(Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0));
    /// let y = x.conj();
    /// assert_eq!(*y.primal(), Complex64::new(1.0, -2.0));
    /// assert_eq!(*y.tangent().unwrap(), Complex64::new(3.0, 4.0));
    /// ```
    pub fn conj(self) -> Self
    where
        T: scalarops::ScalarAd + 'static,
    {
        unary_ad_scalar_op(
            self,
            "conj",
            scalarops::conj,
            scalarops::conj_frule,
            |_input, _output, cotangent| scalarops::conj_rrule(cotangent),
        )
    }

    /// Applies scalar square-root with AD propagation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdScalar;
    ///
    /// let x = AdScalar::new_forward(9.0_f64, 1.0_f64);
    /// let y = x.sqrt();
    /// assert!((*y.primal() - 3.0).abs() < 1e-12);
    /// assert!((*y.tangent().unwrap() - 1.0 / 6.0).abs() < 1e-12);
    /// ```
    pub fn sqrt(self) -> Self
    where
        T: scalarops::ScalarAd + 'static,
    {
        unary_ad_scalar_op(
            self,
            "sqrt",
            scalarops::sqrt,
            scalarops::sqrt_frule,
            |_input, output, cotangent| scalarops::sqrt_rrule(output, cotangent),
        )
    }

    /// Applies scalar real-exponent power with AD propagation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdScalar;
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// let y = x.powf(3.0);
    /// assert_eq!(*y.primal(), 8.0);
    /// assert_eq!(*y.tangent().unwrap(), 12.0);
    /// ```
    pub fn powf(self, exponent: <T as scalarops::ScalarAd>::Real) -> Self
    where
        T: scalarops::ScalarAd + 'static,
        <T as scalarops::ScalarAd>::Real: 'static,
    {
        unary_ad_scalar_op(
            self,
            "powf",
            move |primal| scalarops::powf(primal, exponent),
            move |primal, tangent| scalarops::powf_frule(primal, exponent, tangent),
            move |input, _output, cotangent| scalarops::powf_rrule(input, exponent, cotangent),
        )
    }

    /// Applies scalar integer-exponent power with AD propagation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdScalar;
    ///
    /// let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
    /// let y = x.powi(4);
    /// assert_eq!(*y.primal(), 16.0);
    /// assert_eq!(*y.tangent().unwrap(), 32.0);
    /// ```
    pub fn powi(self, exponent: i32) -> Self
    where
        T: scalarops::ScalarAd + 'static,
    {
        unary_ad_scalar_op(
            self,
            "powi",
            move |primal| scalarops::powi(primal, exponent),
            move |primal, tangent| scalarops::powi_frule(primal, exponent, tangent),
            move |input, _output, cotangent| scalarops::powi_rrule(input, exponent, cotangent),
        )
    }
}

impl<T> From<T> for AdScalar<T> {
    fn from(value: T) -> Self {
        Self(AdValue::Primal(value))
    }
}

impl<T> From<AdValue<T>> for AdScalar<T> {
    fn from(value: AdValue<T>) -> Self {
        Self(value)
    }
}

impl<T> From<AdScalar<T>> for AdValue<T> {
    fn from(value: AdScalar<T>) -> Self {
        value.0
    }
}

static NEXT_AD_SCALAR_NODE_ID: AtomicU64 = AtomicU64::new(1_u64 << 62);

fn unary_ad_scalar_op<T, P, F, R>(
    value: AdScalar<T>,
    op_name: &'static str,
    primal_rule: P,
    frule: F,
    rrule: R,
) -> AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
    P: Fn(T) -> T,
    F: Fn(T, T) -> (T, T),
    R: Fn(T, T, T) -> T + 'static,
{
    match value.into_value() {
        AdValue::Primal(primal) => AdScalar::new_primal(primal_rule(primal)),
        AdValue::Forward { primal, tangent } => {
            let (primal, tangent) = frule(primal, tangent);
            AdScalar::new_forward(primal, tangent)
        }
        AdValue::Reverse {
            primal: input_primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let (output_primal, tangent) = match tangent {
                Some(tangent) => {
                    let (primal, tangent) = frule(input_primal, tangent);
                    (primal, Some(tangent))
                }
                None => (primal_rule(input_primal), None),
            };
            let output_node = NodeId(NEXT_AD_SCALAR_NODE_ID.fetch_add(1, Ordering::Relaxed));
            reverse_tape::register_scalar_rule(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    Ok(vec![(
                        input_node,
                        rrule(input_primal, output_primal, *cotangent),
                    )])
                }),
            )
            .unwrap_or_else(|e| panic!("{op_name}: {e}"));
            AdScalar::new_reverse(output_primal, output_node, tape, tangent)
        }
    }
}

pub(crate) fn map_ad_value_same_type_linear<T, M>(
    value: AdValue<T>,
    op_name: &'static str,
    map: M,
) -> AdValue<T>
where
    T: scalarops::ScalarAd + 'static,
    M: Fn(T) -> T + Copy + 'static,
{
    match value {
        AdValue::Primal(primal) => AdValue::Primal(map(primal)),
        AdValue::Forward { primal, tangent } => AdValue::Forward {
            primal: map(primal),
            tangent: map(tangent),
        },
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = map(primal);
            let output_tangent = tangent.map(map);
            let output_node = NodeId(NEXT_AD_SCALAR_NODE_ID.fetch_add(1, Ordering::Relaxed));
            reverse_tape::register_scalar_rule(
                tape,
                output_node,
                Box::new(move |cotangent| Ok(vec![(input_node, map(*cotangent))])),
            )
            .unwrap_or_else(|e| panic!("{op_name}: {e}"));
            AdValue::Reverse {
                primal: output_primal,
                node: output_node,
                tape,
                tangent: output_tangent,
            }
        }
    }
}

pub(crate) fn map_ad_value_mixed_linear<TIn, TOut, P, R>(
    value: AdValue<TIn>,
    op_name: &'static str,
    primal_map: P,
    reverse_map: R,
) -> AdValue<TOut>
where
    TIn: scalarops::ScalarAd + 'static,
    TOut: scalarops::ScalarAd + 'static,
    P: Fn(TIn) -> TOut + Copy,
    R: Fn(TOut) -> TIn + Copy + 'static,
{
    match value {
        AdValue::Primal(primal) => AdValue::Primal(primal_map(primal)),
        AdValue::Forward { primal, tangent } => AdValue::Forward {
            primal: primal_map(primal),
            tangent: primal_map(tangent),
        },
        AdValue::Reverse {
            primal,
            node: input_node,
            tape,
            tangent,
        } => {
            let output_primal = primal_map(primal);
            let output_tangent = tangent.map(primal_map);
            let output_node = NodeId(NEXT_AD_SCALAR_NODE_ID.fetch_add(1, Ordering::Relaxed));
            reverse_tape::register_scalar_mixed_rule(
                tape,
                output_node,
                Box::new(move |cotangent| Ok(vec![(input_node, reverse_map(*cotangent))])),
            )
            .unwrap_or_else(|e| panic!("{op_name}: {e}"));
            AdValue::Reverse {
                primal: output_primal,
                node: output_node,
                tape,
                tangent: output_tangent,
            }
        }
    }
}

#[derive(Clone, Copy)]
struct BinaryScalarState<T> {
    primal: T,
    tangent: Option<T>,
    reverse: Option<(NodeId, TapeId)>,
}

fn split_binary_state<T: scalarops::ScalarAd>(value: AdScalar<T>) -> BinaryScalarState<T> {
    match value.into_value() {
        AdValue::Primal(primal) => BinaryScalarState {
            primal,
            tangent: None,
            reverse: None,
        },
        AdValue::Forward { primal, tangent } => BinaryScalarState {
            primal,
            tangent: Some(tangent),
            reverse: None,
        },
        AdValue::Reverse {
            primal,
            node,
            tape,
            tangent,
        } => BinaryScalarState {
            primal,
            tangent,
            reverse: Some((node, tape)),
        },
    }
}

fn add_rrule_wrapped<T: scalarops::ScalarAd>(_lhs: T, _rhs: T, cotangent: T) -> (T, T) {
    scalarops::add_rrule(cotangent)
}

fn sub_rrule_wrapped<T: scalarops::ScalarAd>(_lhs: T, _rhs: T, cotangent: T) -> (T, T) {
    scalarops::sub_rrule(cotangent)
}

fn mul_rrule_wrapped<T: scalarops::ScalarAd>(lhs: T, rhs: T, cotangent: T) -> (T, T) {
    scalarops::mul_rrule(lhs, rhs, cotangent)
}

fn div_rrule_wrapped<T: scalarops::ScalarAd>(lhs: T, rhs: T, cotangent: T) -> (T, T) {
    scalarops::div_rrule(lhs, rhs, cotangent)
}

fn binary_ad_scalar_try_op<T: scalarops::ScalarAd + 'static>(
    lhs: AdScalar<T>,
    rhs: AdScalar<T>,
    _op_name: &'static str,
    primal_rule: fn(T, T) -> T,
    frule: fn(T, T, T, T) -> (T, T),
    rrule: fn(T, T, T) -> (T, T),
) -> Result<AdScalar<T>> {
    let lhs_state = split_binary_state(lhs);
    let rhs_state = split_binary_state(rhs);

    let primal = primal_rule(lhs_state.primal, rhs_state.primal);
    let has_tangent = lhs_state.tangent.is_some() || rhs_state.tangent.is_some();
    let tangent = if has_tangent {
        let dx = lhs_state.tangent.unwrap_or_else(|| T::from_i32(0));
        let dy = rhs_state.tangent.unwrap_or_else(|| T::from_i32(0));
        let (_, tangent) = frule(lhs_state.primal, rhs_state.primal, dx, dy);
        Some(tangent)
    } else {
        None
    };

    match (lhs_state.reverse, rhs_state.reverse) {
        (None, None) => {
            if let Some(tangent) = tangent {
                Ok(AdScalar::new_forward(primal, tangent))
            } else {
                Ok(AdScalar::new_primal(primal))
            }
        }
        (lhs_reverse, rhs_reverse) => {
            let tape = match (lhs_reverse, rhs_reverse) {
                (Some((_, lhs_tape)), Some((_, rhs_tape))) if lhs_tape != rhs_tape => {
                    return Err(Error::MixedReverseTape {
                        expected: lhs_tape.0,
                        found: rhs_tape.0,
                    })
                }
                (Some((_, lhs_tape)), Some(_)) => lhs_tape,
                (Some((_, lhs_tape)), None) => lhs_tape,
                (None, Some((_, rhs_tape))) => rhs_tape,
                (None, None) => unreachable!("non-reverse case handled above"),
            };

            let output_node = NodeId(NEXT_AD_SCALAR_NODE_ID.fetch_add(1, Ordering::Relaxed));
            let lhs_primal = lhs_state.primal;
            let rhs_primal = rhs_state.primal;
            let lhs_node = lhs_reverse.map(|(node, _)| node);
            let rhs_node = rhs_reverse.map(|(node, _)| node);

            reverse_tape::register_scalar_rule(
                tape,
                output_node,
                Box::new(move |cotangent| {
                    let (dlhs, drhs) = rrule(lhs_primal, rhs_primal, *cotangent);
                    let mut grads = Vec::new();
                    if let Some(node) = lhs_node {
                        grads.push((node, dlhs));
                    }
                    if let Some(node) = rhs_node {
                        grads.push((node, drhs));
                    }
                    Ok(grads)
                }),
            )?;

            Ok(AdScalar::new_reverse(primal, output_node, tape, tangent))
        }
    }
}

fn binary_ad_scalar_op<T: scalarops::ScalarAd + 'static>(
    lhs: AdScalar<T>,
    rhs: AdScalar<T>,
    op_name: &'static str,
    primal_rule: fn(T, T) -> T,
    frule: fn(T, T, T, T) -> (T, T),
    rrule: fn(T, T, T) -> (T, T),
) -> AdScalar<T> {
    binary_ad_scalar_try_op(lhs, rhs, op_name, primal_rule, frule, rrule)
        .unwrap_or_else(|e| panic!("{op_name}: {e}"))
}

impl<T> AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    /// Checked addition for AD scalars.
    pub fn try_add(self, rhs: Self) -> Result<Self> {
        binary_ad_scalar_try_op(
            self,
            rhs,
            "add",
            scalarops::add,
            scalarops::add_frule,
            add_rrule_wrapped,
        )
    }

    /// Checked subtraction for AD scalars.
    pub fn try_sub(self, rhs: Self) -> Result<Self> {
        binary_ad_scalar_try_op(
            self,
            rhs,
            "sub",
            scalarops::sub,
            scalarops::sub_frule,
            sub_rrule_wrapped,
        )
    }

    /// Checked multiplication for AD scalars.
    pub fn try_mul(self, rhs: Self) -> Result<Self> {
        binary_ad_scalar_try_op(
            self,
            rhs,
            "mul",
            scalarops::mul,
            scalarops::mul_frule,
            mul_rrule_wrapped,
        )
    }

    /// Checked division for AD scalars.
    pub fn try_div(self, rhs: Self) -> Result<Self> {
        binary_ad_scalar_try_op(
            self,
            rhs,
            "div",
            scalarops::div,
            scalarops::div_frule,
            div_rrule_wrapped,
        )
    }
}

impl<T> Add for AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        binary_ad_scalar_op(
            self,
            rhs,
            "add",
            scalarops::add,
            scalarops::add_frule,
            add_rrule_wrapped,
        )
    }
}

impl<T> Sub for AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_ad_scalar_op(
            self,
            rhs,
            "sub",
            scalarops::sub,
            scalarops::sub_frule,
            sub_rrule_wrapped,
        )
    }
}

impl<T> Mul for AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_ad_scalar_op(
            self,
            rhs,
            "mul",
            scalarops::mul,
            scalarops::mul_frule,
            mul_rrule_wrapped,
        )
    }
}

impl<T> Div for AdScalar<T>
where
    T: scalarops::ScalarAd + 'static,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        binary_ad_scalar_op(
            self,
            rhs,
            "div",
            scalarops::div,
            scalarops::div_frule,
            div_rrule_wrapped,
        )
    }
}

impl<T> Neg for AdScalar<T>
where
    T: scalarops::ScalarAd + Neg<Output = T> + 'static,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        unary_ad_scalar_op(
            self,
            "neg",
            |primal| -primal,
            |primal, tangent| (-primal, -tangent),
            |_input, _output, cotangent| -cotangent,
        )
    }
}

/// Tensor newtype carrying AD mode information.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{AdMode, AdTensor};
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
/// let x: AdTensor<f64> = t.into();
/// assert_eq!(x.mode(), AdMode::Primal);
/// ```
#[derive(Clone)]
pub struct AdTensor<T: Scalar>(pub AdValue<StructuredTensor<T>>);

impl<T: Scalar> AdTensor<T> {
    /// Creates a primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.dims(), &[1]);
    /// ```
    pub fn new_primal(tensor: impl Into<StructuredTensor<T>>) -> Self {
        Self(AdValue::primal(tensor.into()))
    }

    /// Creates a forward-mode tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let tangent = Tensor::<f64>::from_slice(&[0.1], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_forward(primal, tangent);
    /// assert_eq!(x.mode(), AdMode::Forward);
    /// ```
    pub fn new_forward(
        primal: impl Into<StructuredTensor<T>>,
        tangent: impl Into<StructuredTensor<T>>,
    ) -> Self {
        Self(AdValue::forward(primal.into(), tangent.into()))
    }

    /// Creates a reverse-mode tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdTensor, NodeId, TapeId};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_reverse(primal, NodeId(8), TapeId(3), None);
    /// assert_eq!(x.mode(), AdMode::Reverse);
    /// ```
    pub fn new_reverse(
        primal: impl Into<StructuredTensor<T>>,
        node: NodeId,
        tape: TapeId,
        tangent: Option<impl Into<StructuredTensor<T>>>,
    ) -> Self {
        Self(AdValue::reverse(primal.into(), node, tape, tangent.map(Into::into)))
    }

    /// Returns AD mode.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdMode, AdTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.mode(), AdMode::Primal);
    /// ```
    pub fn mode(&self) -> AdMode {
        self.0.mode()
    }

    /// Returns reference to underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, AdValue};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert!(matches!(x.as_value(), AdValue::Primal(_)));
    /// ```
    pub fn as_value(&self) -> &AdValue<StructuredTensor<T>> {
        &self.0
    }

    /// Consumes wrapper and returns the underlying [`AdValue`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, AdValue};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t).into_value();
    /// assert!(matches!(x, AdValue::Primal(_)));
    /// ```
    pub fn into_value(self) -> AdValue<StructuredTensor<T>> {
        self.0
    }

    /// Returns structured primal payload reference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.structured_primal().logical_dims(), &[2]);
    /// ```
    pub fn structured_primal(&self) -> &StructuredTensor<T> {
        self.0.primal_ref()
    }

    /// Returns compressed primal payload tensor reference.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.primal().dims(), &[2]);
    /// ```
    pub fn primal(&self) -> &Tensor<T> {
        self.structured_primal().payload()
    }

    /// Returns structured tangent reference when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let tangent = Tensor::<f64>::from_slice(&[0.5], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_forward(primal, tangent);
    /// assert_eq!(x.structured_tangent().unwrap().logical_dims(), &[1]);
    /// ```
    pub fn structured_tangent(&self) -> Option<&StructuredTensor<T>> {
        self.0.tangent_ref()
    }

    /// Returns compressed tangent payload tensor reference when available.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let primal = Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let tangent = Tensor::<f64>::from_slice(&[0.5], &[1], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_forward(primal, tangent);
    /// assert_eq!(x.tangent().unwrap().dims(), &[1]);
    /// ```
    pub fn tangent(&self) -> Option<&Tensor<T>> {
        self.structured_tangent().map(StructuredTensor::payload)
    }

    /// Returns dimensions of the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.dims(), &[2]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        self.structured_primal().logical_dims()
    }

    /// Returns number of dimensions of the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.ndim(), 1);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims().len()
    }

    /// Returns total number of elements in the primal tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert_eq!(x.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }

    /// Returns true when primal tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[], &[0], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert!(x.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns axis classes of the structured primal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, StructuredTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(StructuredTensor::from_diagonal_vector(payload, 2).unwrap());
    /// assert_eq!(x.axis_classes(), &[0, 0]);
    /// ```
    pub fn axis_classes(&self) -> &[usize] {
        self.structured_primal().axis_classes()
    }

    /// Returns `true` when the structured primal is dense.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::AdTensor;
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(t);
    /// assert!(x.is_dense());
    /// ```
    pub fn is_dense(&self) -> bool {
        self.structured_primal().is_dense()
    }

    /// Returns `true` when the structured primal is diagonal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_dyadtensor::{AdTensor, StructuredTensor};
    /// use tenferro_tensor::{MemoryOrder, Tensor};
    ///
    /// let payload =
    ///     Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    /// let x = AdTensor::new_primal(StructuredTensor::from_diagonal_vector(payload, 2).unwrap());
    /// assert!(x.is_diag());
    /// ```
    pub fn is_diag(&self) -> bool {
        self.structured_primal().is_diag()
    }
}

impl<T: Scalar> From<Tensor<T>> for AdTensor<T> {
    fn from(value: Tensor<T>) -> Self {
        Self(AdValue::Primal(StructuredTensor::from_dense(value)))
    }
}

impl<T: Scalar> From<StructuredTensor<T>> for AdTensor<T> {
    fn from(value: StructuredTensor<T>) -> Self {
        Self(AdValue::Primal(value))
    }
}

impl<T: Scalar> From<AdValue<StructuredTensor<T>>> for AdTensor<T> {
    fn from(value: AdValue<StructuredTensor<T>>) -> Self {
        Self(value)
    }
}

impl<T: Scalar> From<AdValue<Tensor<T>>> for AdTensor<T> {
    fn from(value: AdValue<Tensor<T>>) -> Self {
        let mapped = match value {
            AdValue::Primal(primal) => AdValue::Primal(StructuredTensor::from_dense(primal)),
            AdValue::Forward { primal, tangent } => AdValue::Forward {
                primal: StructuredTensor::from_dense(primal),
                tangent: StructuredTensor::from_dense(tangent),
            },
            AdValue::Reverse {
                primal,
                node,
                tape,
                tangent,
            } => AdValue::Reverse {
                primal: StructuredTensor::from_dense(primal),
                node,
                tape,
                tangent: tangent.map(StructuredTensor::from_dense),
            },
        };
        Self(mapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use tenferro_tensor::MemoryOrder;

    #[test]
    fn ad_value_map_preserving_metadata_preserves_mode() {
        let x = AdValue::forward(2_i32, 3_i32);
        let y = x.map_preserving_metadata(|v| v as f64);
        assert_eq!(y.mode(), AdMode::Forward);
        assert_eq!(y.primal_ref(), &2.0_f64);
        assert_eq!(y.tangent_ref(), Some(&3.0_f64));
    }

    #[test]
    fn ad_tensor_metadata() {
        let tensor =
            Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        let ad = AdTensor::new_primal(tensor);
        assert_eq!(ad.mode(), AdMode::Primal);
        assert_eq!(ad.dims(), &[2]);
        assert_eq!(ad.ndim(), 1);
        assert_eq!(ad.len(), 2);
    }

    #[test]
    fn ad_scalar_into_primal_drops_metadata() {
        let x = AdScalar::new_forward(2.5_f64, 0.1_f64);
        assert_eq!(x.into_primal(), 2.5_f64);
    }

    #[test]
    fn ad_scalar_sqrt_forward_propagates_tangent() {
        let x = AdScalar::new_forward(9.0_f64, 1.0_f64);
        let y = x.sqrt();
        assert!((*y.primal() - 3.0).abs() < 1e-12);
        assert!((*y.tangent().unwrap() - (1.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn ad_scalar_powi_forward_propagates_tangent() {
        let x = AdScalar::new_forward(2.0_f64, 1.0_f64);
        let y = x.powi(4);
        assert_eq!(*y.primal(), 16.0);
        assert_eq!(*y.tangent().unwrap(), 32.0);
    }

    #[test]
    fn ad_scalar_mul_forward_propagates_tangent() {
        let x = AdScalar::new_forward(2.0_f64, 0.5_f64);
        let y = AdScalar::new_forward(4.0_f64, 0.25_f64);
        let z = x * y;
        assert_eq!(*z.primal(), 8.0_f64);
        assert_eq!(*z.tangent().unwrap(), 2.5_f64);
    }

    #[test]
    fn ad_scalar_div_forward_propagates_tangent() {
        let x = AdScalar::new_forward(8.0_f64, 0.5_f64);
        let y = AdScalar::new_forward(2.0_f64, 0.25_f64);
        let z = x / y;
        assert_eq!(*z.primal(), 4.0_f64);
        assert_eq!(*z.tangent().unwrap(), -0.25_f64);
    }

    #[test]
    fn ad_scalar_conj_reverse_allocates_fresh_output_node() {
        let x = AdScalar::new_reverse(
            Complex64::new(1.0, 2.0),
            NodeId(11),
            TapeId(7),
            Some(Complex64::new(-1.0, 0.5)),
        );
        let y = x.conj();
        assert_eq!(y.mode(), AdMode::Reverse);
        assert_eq!(y.as_value().tape_id(), Some(TapeId(7)));
        assert_ne!(y.as_value().node_id(), Some(NodeId(11)));
        assert_eq!(*y.primal(), Complex64::new(1.0, -2.0));
        assert_eq!(*y.tangent().unwrap(), Complex64::new(-1.0, -0.5));
    }

    #[test]
    fn ad_scalar_sqrt_reverse_registers_pullback_chain() {
        let x = AdScalar::new_reverse(4.0_f64, NodeId(21), TapeId(17), None);
        let y = x.sqrt();
        let grads = crate::reverse_tape::pullback_scalar::<f64>(
            TapeId(17),
            y.as_value().node_id().unwrap(),
            &3.0_f64,
        )
        .unwrap();
        assert_eq!(grads.get(&NodeId(21)).copied(), Some(0.75));
    }

    #[test]
    #[should_panic(expected = "reverse-mode operands must share one tape")]
    fn ad_scalar_binary_op_panics_on_mixed_reverse_tapes() {
        let x = AdScalar::new_reverse(2.0_f64, NodeId(1), TapeId(7), None);
        let y = AdScalar::new_reverse(3.0_f64, NodeId(2), TapeId(8), None);
        let _ = x * y;
    }

    #[test]
    fn ad_scalar_try_binary_op_returns_error_on_mixed_reverse_tapes() {
        let x = AdScalar::new_reverse(2.0_f64, NodeId(1), TapeId(7), None);
        let y = AdScalar::new_reverse(3.0_f64, NodeId(2), TapeId(8), None);
        let err = x.try_mul(y).unwrap_err();
        assert!(matches!(
            err,
            Error::MixedReverseTape {
                expected: 7,
                found: 8
            }
        ));
    }
}
