//! Owner-local initialized N-ary workspaces, separate from numerical plan caches.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::Mutex,
};

struct Entry {
    scratch: Box<dyn Any + Send>,
    bytes: usize,
}

/// Retained workspaces borrowed from an already-admitted CPU resource owner.
///
/// # Examples
/// ```
/// use tenferro_cpu::ContractionWorkspaces;
/// let workspaces = ContractionWorkspaces::default();
/// let len = workspaces.with_scratch::<f64, _>(4, 1024, |scratch| Ok(scratch.len()))?;
/// assert_eq!(len, 4);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[doc(hidden)]
#[derive(Default)]
pub struct ContractionWorkspaces {
    entries: Mutex<HashMap<TypeId, Entry>>,
}

/// Reports how many typed scratch entries this owner retains and their bytes,
/// without exposing the payloads; the engine's own `Debug` needs it.
impl std::fmt::Debug for ContractionWorkspaces {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.entries.try_lock() {
            Ok(entries) => f
                .debug_struct("ContractionWorkspaces")
                .field("entries", &entries.len())
                .field(
                    "bytes",
                    &entries.values().map(|entry| entry.bytes).sum::<usize>(),
                )
                .finish(),
            Err(_) => f
                .debug_struct("ContractionWorkspaces")
                .field("entries", &"borrowed")
                .finish(),
        }
    }
}

impl ContractionWorkspaces {
    pub(crate) fn lock(&self) -> crate::Result<WorkspaceLease<'_>> {
        self.entries
            .try_lock()
            .map(WorkspaceLease)
            .map_err(|error| {
                crate::Error::backend_source("CPU N-ary workspace", WorkspaceError::from(error))
            })
    }
    /// Borrow initialized scratch for a lower N-ary operation.
    ///
    /// `Scalar` is sealed by cpueinsum: this store has at most four typed slots.
    /// No input/output tensor, executor, or numerical plan is stored here.
    ///
    /// # Examples
    /// ```
    /// use tenferro_cpu::ContractionWorkspaces;
    /// let workspaces = ContractionWorkspaces::default();
    /// workspaces.with_scratch::<f32, _>(8, 0, |scratch| {
    ///     assert_eq!(scratch.len(), 8);
    ///     Ok(())
    /// })?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::BackendSource`] when the store is already
    /// borrowed by a concurrent or recursive execution, and otherwise returns the
    /// error produced by `op`.
    pub fn with_scratch<T: cpueinsum::Scalar, R>(
        &self,
        len: usize,
        max_retained_bytes: usize,
        op: impl FnOnce(&mut cpueinsum::Scratch<T>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        // Resource admission normally serializes this owner. Never wait for an
        // unexpected concurrent or recursive workspace borrow.
        let mut entries = self.entries.try_lock().map_err(|error| {
            crate::Error::backend_source("CPU N-ary workspace", WorkspaceError::from(error))
        })?;
        let cleanup = RetentionCleanup {
            entries: &mut entries,
            max_bytes: max_retained_bytes,
        };
        let entry = cleanup
            .entries
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Entry {
                scratch: Box::new(cpueinsum::Scratch::<T>::new()),
                bytes: 0,
            });
        let scratch = entry
            .scratch
            .downcast_mut::<cpueinsum::Scratch<T>>()
            .ok_or_else(|| {
                crate::Error::runtime_state("CPU N-ary workspace", "incompatible scratch type")
            })?;
        if scratch.len() < len {
            // with_len uses an exactly-sized initialized Vec. Do not let resize
            // growth hide retained capacity from the owner accounting.
            *scratch = cpueinsum::Scratch::with_len(len);
        }
        let before = scratch.len();
        let result = op(scratch);
        if scratch.len() != before {
            *scratch = cpueinsum::Scratch::with_len(scratch.len());
        }
        entry.bytes = scratch.len().saturating_mul(std::mem::size_of::<T>());
        result
    }
}

pub(crate) struct WorkspaceLease<'a>(std::sync::MutexGuard<'a, HashMap<TypeId, Entry>>);
impl WorkspaceLease<'_> {
    pub(crate) fn stats(&self) -> (usize, usize) {
        (
            self.0.len(),
            self.0
                .values()
                .fold(0usize, |sum, entry| sum.saturating_add(entry.bytes)),
        )
    }
    pub(crate) fn clear(&mut self) {
        self.0.clear();
        self.0.shrink_to_fit();
    }
    pub(crate) fn trim(&mut self, max_bytes: usize) {
        if self.stats().1 > max_bytes {
            self.clear();
        }
    }
}

struct RetentionCleanup<'a> {
    entries: &'a mut HashMap<TypeId, Entry>,
    max_bytes: usize,
}
impl Drop for RetentionCleanup<'_> {
    fn drop(&mut self) {
        let retained = self
            .entries
            .values()
            .fold(0usize, |sum, entry| sum.saturating_add(entry.bytes));
        if std::thread::panicking() || retained > self.max_bytes {
            self.entries.clear();
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum WorkspaceError {
    #[error("CPU N-ary workspace is already borrowed")]
    Contended,
    #[error("CPU N-ary workspace lock is poisoned")]
    Poisoned,
}
impl<T> From<std::sync::TryLockError<T>> for WorkspaceError {
    fn from(error: std::sync::TryLockError<T>) -> Self {
        match error {
            std::sync::TryLockError::WouldBlock => Self::Contended,
            std::sync::TryLockError::Poisoned(_) => Self::Poisoned,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scratch_reuses_initialized_storage_and_obeys_retention_limit() {
        let workspaces = ContractionWorkspaces::default();
        let first = workspaces
            .with_scratch::<f64, _>(16, 1024, |scratch| {
                assert_eq!(scratch.len(), 16);
                Ok(std::ptr::from_ref(scratch) as usize)
            })
            .unwrap();
        let second = workspaces
            .with_scratch::<f64, _>(8, 1024, |scratch| {
                assert_eq!(scratch.len(), 16);
                Ok(std::ptr::from_ref(scratch) as usize)
            })
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            workspaces
                .entries
                .lock()
                .unwrap()
                .values()
                .map(|e| e.bytes)
                .sum::<usize>(),
            128
        );
        workspaces
            .with_scratch::<f64, _>(16, 0, |_| Ok(()))
            .unwrap();
        assert!(workspaces.entries.lock().unwrap().is_empty());
    }

    #[test]
    fn unwind_drops_workspace_storage() {
        let workspaces = ContractionWorkspaces::default();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = workspaces.with_scratch::<f64, ()>(32, 1024, |_| panic!("workspace test"));
        }));
        assert!(panic.is_err());
        match workspaces.entries.lock() {
            Err(poison) => assert!(poison.into_inner().is_empty()),
            Ok(_) => panic!("unwound workspace lock should be poisoned"),
        };
    }

    #[test]
    fn recursive_scratch_borrow_returns_instead_of_waiting() {
        let workspaces = ContractionWorkspaces::default();
        workspaces
            .with_scratch::<f32, _>(4, 1024, |_| {
                assert!(workspaces
                    .with_scratch::<f32, _>(4, 1024, |_| Ok(()))
                    .is_err());
                Ok(())
            })
            .unwrap();
    }
}
