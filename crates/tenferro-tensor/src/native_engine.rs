//! Lifetime-bound native session capability tokens.
//!
//! A backend leaf crate (CPU, CUDA, WebGPU) exposes its concrete execution
//! session to higher-level operation crates through a [`NativeSessionRef`]
//! returned by [`Engine::native_engine`](crate::Engine::native_engine).
//! The token is opaque: its fields are private and its only constructor is
//! `unsafe`, so safe code can neither fabricate one nor retarget it at another
//! value. A backend leaf recovers its own session with a safe visitor that
//! checks the token's marker against a marker type only that leaf can name.
//!
//! This narrows the unsafe boundary to each leaf's constructor and visitor; it
//! does not remove it.
//!
//! # Examples
//!
//! A session without native services returns no token, which is the default:
//!
//! ```rust
//! use tenferro_tensor::{Engine, NativeSessionRef};
//!
//! fn has_native_services(session: &mut dyn Engine) -> bool {
//!     session.native_engine().is_some()
//! }
//! ```
//!
//! The token borrows its session exclusively and cannot outlive that borrow:
//!
//! ```compile_fail
//! use tenferro_tensor::{Engine, NativeSessionRef};
//!
//! fn escape(session: &mut dyn Engine) -> NativeSessionRef<'static> {
//!     session.native_engine().unwrap()
//! }
//! ```
//!
//! It cannot be built from safe code, even with a private marker type:
//!
//! ```compile_fail,E0133
//! use tenferro_tensor::NativeSessionRef;
//!
//! struct Marker;
//! let mut value = 0_u8;
//! let _token = NativeSessionRef::new::<Marker, u8>(&mut value);
//! ```
//!
//! It is neither `Send` nor `Clone`, so it cannot leave its thread or be
//! duplicated into a second exclusive borrow:
//!
//! ```compile_fail,E0277
//! fn require_send<T: Send>(_: T) {}
//! fn check(token: tenferro_tensor::NativeSessionRef<'_>) {
//!     require_send(token);
//! }
//! ```
//!
//! ```compile_fail,E0599
//! fn duplicate(token: tenferro_tensor::NativeSessionRef<'_>) {
//!     let _second = token.clone();
//!     let _first = token;
//! }
//! ```

use std::any::TypeId;
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// An exclusively borrowed backend-leaf execution session.
///
/// The token names one concrete session by a backend-leaf marker type and
/// borrows it for `'s`. It is not `Clone`, `Copy`, `Send` or `Sync`. Only the
/// backend leaf that owns the marker can recover the session, through its own
/// safe visitor; custom sessions either return no token or forward the token of
/// a standard delegate they own.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Engine, NativeSessionRef};
///
/// struct Marker;
///
/// fn is_marked(token: &NativeSessionRef<'_>) -> bool {
///     token.has_marker::<Marker>()
/// }
///
/// fn inspect(session: &mut dyn Engine) -> bool {
///     session.native_engine().is_some_and(|token| is_marked(&token))
/// }
/// ```
pub struct NativeSessionRef<'s> {
    marker: TypeId,
    session: NonNull<()>,
    _borrow: PhantomData<&'s mut ()>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'s> NativeSessionRef<'s> {
    /// Create a token that names `session` by the backend-leaf marker `M`.
    ///
    /// # Safety
    ///
    /// `M` must be a marker type private to the calling backend leaf crate,
    /// and every token that crate creates with `M` must point to a value of the
    /// single concrete session type its visitor recovers for `M`. `session`
    /// is exclusively borrowed for `'s`, which the signature enforces; the
    /// visitor relies on the marker/type correspondence for its cast.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::NativeSessionRef;
    ///
    /// struct LeafSession(u32);
    /// struct LeafMarker;
    ///
    /// let mut session = LeafSession(7);
    /// // SAFETY: `LeafMarker` is private to this example and names `LeafSession`.
    /// let token = unsafe { NativeSessionRef::new::<LeafMarker, _>(&mut session) };
    /// let pointer = token.into_marked_ptr::<LeafMarker>().expect("same marker");
    /// // SAFETY: the marker proved the pointee is `LeafSession`, and the
    /// // exclusive borrow of `session` is still held here.
    /// assert_eq!(unsafe { pointer.cast::<LeafSession>().as_ref().0 }, 7);
    /// ```
    pub unsafe fn new<M: 'static, S>(session: &'s mut S) -> Self {
        Self {
            marker: TypeId::of::<M>(),
            session: NonNull::from(session).cast(),
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Whether this token was created with the backend-leaf marker `M`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::NativeSessionRef;
    ///
    /// struct Marker;
    /// struct Other;
    /// let mut value = 0_u8;
    /// // SAFETY: `Marker` is private to this example and names `u8`.
    /// let token = unsafe { NativeSessionRef::new::<Marker, _>(&mut value) };
    /// assert!(token.has_marker::<Marker>());
    /// assert!(!token.has_marker::<Other>());
    /// ```
    #[must_use]
    pub fn has_marker<M: 'static>(&self) -> bool {
        self.marker == TypeId::of::<M>()
    }

    /// Consume the token and return its session pointer when it carries `M`.
    ///
    /// Returning the pointer is safe; dereferencing it is the backend leaf's
    /// audited step. The leaf may only do so while the borrow the token was
    /// created from is still held, which its visitor guarantees by holding the
    /// `&mut` session for the whole visit.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::NativeSessionRef;
    ///
    /// struct Marker;
    /// struct Other;
    /// let mut value = 3_u8;
    /// // SAFETY: `Marker` is private to this example and names `u8`.
    /// let token = unsafe { NativeSessionRef::new::<Marker, _>(&mut value) };
    /// assert!(token.into_marked_ptr::<Other>().is_none());
    /// ```
    #[must_use]
    pub fn into_marked_ptr<M: 'static>(self) -> Option<NonNull<()>> {
        self.has_marker::<M>().then_some(self.session)
    }
}

impl fmt::Debug for NativeSessionRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeSessionRef").finish_non_exhaustive()
    }
}
