use std::any::{type_name, Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::{Error, Result};

thread_local! {
    static GLOBAL_CONTEXTS: RefCell<HashMap<TypeId, Box<dyn Any>>> = RefCell::new(HashMap::new());
}

/// Guard returned by [`set_global_context`].
///
/// When dropped, the previously installed context value (if any) is restored.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{set_global_context, with_global_context};
///
/// let _guard = set_global_context::<u32>(7);
/// let v = with_global_context::<u32, _>(|ctx| Ok(*ctx)).unwrap();
/// assert_eq!(v, 7);
/// ```
pub struct GlobalContextGuard<C: 'static> {
    previous: Option<Box<dyn Any>>,
    _marker: PhantomData<C>,
}

impl<C: 'static> Drop for GlobalContextGuard<C> {
    fn drop(&mut self) {
        GLOBAL_CONTEXTS.with(|contexts| {
            let mut contexts = contexts.borrow_mut();
            let key = TypeId::of::<C>();
            contexts.remove(&key);
            if let Some(previous) = self.previous.take() {
                contexts.insert(key, previous);
            }
        });
    }
}

/// Sets a thread-local global context for type `C`.
///
/// Returns a guard that restores the previous context on drop.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{set_global_context, with_global_context};
///
/// let guard = set_global_context::<usize>(123);
/// let value = with_global_context::<usize, _>(|ctx| Ok(*ctx)).unwrap();
/// assert_eq!(value, 123);
/// drop(guard);
/// ```
pub fn set_global_context<C: 'static>(ctx: C) -> GlobalContextGuard<C> {
    let previous = GLOBAL_CONTEXTS.with(|contexts| {
        contexts
            .borrow_mut()
            .insert(TypeId::of::<C>(), Box::new(ctx) as Box<dyn Any>)
    });

    GlobalContextGuard {
        previous,
        _marker: PhantomData,
    }
}

/// Runs `f` with a mutable reference to thread-local global context `C`.
///
/// Returns [`Error::MissingGlobalContext`] if no context is registered.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{set_global_context, with_global_context};
///
/// let _guard = set_global_context::<usize>(11);
/// let value = with_global_context::<usize, _>(|ctx| {
///     *ctx += 1;
///     Ok(*ctx)
/// })
/// .unwrap();
/// assert_eq!(value, 12);
/// ```
pub fn with_global_context<C: 'static, R>(f: impl FnOnce(&mut C) -> Result<R>) -> Result<R> {
    GLOBAL_CONTEXTS.with(|contexts| {
        let mut contexts = contexts.borrow_mut();
        let erased =
            contexts
                .get_mut(&TypeId::of::<C>())
                .ok_or_else(|| Error::MissingGlobalContext {
                    type_name: type_name::<C>(),
                })?;
        let typed = erased
            .downcast_mut::<C>()
            .ok_or_else(|| Error::ContextTypeMismatch {
                expected: type_name::<C>(),
            })?;
        f(typed)
    })
}

/// Like [`with_global_context`] but returns `Ok(None)` when context is missing.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::try_with_global_context;
///
/// let value = try_with_global_context::<u64, _>(|ctx| Ok(*ctx)).unwrap();
/// assert_eq!(value, None);
/// ```
pub fn try_with_global_context<C: 'static, R>(
    f: impl FnOnce(&mut C) -> Result<R>,
) -> Result<Option<R>> {
    GLOBAL_CONTEXTS.with(|contexts| {
        let mut contexts = contexts.borrow_mut();
        let Some(erased) = contexts.get_mut(&TypeId::of::<C>()) else {
            return Ok(None);
        };
        let typed = erased
            .downcast_mut::<C>()
            .ok_or_else(|| Error::ContextTypeMismatch {
                expected: type_name::<C>(),
            })?;
        f(typed).map(Some)
    })
}

#[cfg(test)]
mod tests;
