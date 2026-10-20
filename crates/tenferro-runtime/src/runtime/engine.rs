//! The runtime-facing side of a caller-held execution engine.
//!
//! A runtime keeps only what was prepared (graphs, compiled and prepared
//! programs); everything execution writes travels with the engine the caller
//! passes to [`crate::Runtime::run`]. This module defines what the runtime needs
//! from such an engine: an immutable target description used at prepare time,
//! the engine-owned extension caches, its event-domain driver, and the hand-off
//! point for unfinished device work.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use tenferro_tensor::{AllocationDomainId, Engine, Tensor, TensorRead};

use super::{
    EngineId, EventDomainDriver, EventDomainRun, ExecutionContextIdentity,
    ImmediateEventDomainDriver,
};
use crate::extension_cache::ExtensionCacheStore;

/// Immutable description of one execution engine, used to prepare a program.
///
/// A target names the registered engine kind (`engine_id`), the concrete
/// engine type (`context_identity`) and the memory the engine executes on
/// (`allocation_domain`, `None` for host memory). It holds no threads, streams
/// or memory. [`crate::Runtime::prepare`] places each part of a program on one
/// of the targets it is given, and [`crate::Runtime::run`] checks that the
/// engines it receives report the same targets.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{EngineId, EngineTarget, ExecutionContextIdentity};
///
/// struct ExampleEngine;
/// let target = EngineTarget::new(
///     EngineId::new("example.engine.v1").unwrap(),
///     ExecutionContextIdentity::of::<ExampleEngine>(),
///     None,
/// );
/// assert_eq!(target.engine_id().as_str(), "example.engine.v1");
/// assert!(target.allocation_domain().is_none());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EngineTarget {
    engine_id: EngineId,
    context_identity: ExecutionContextIdentity,
    allocation_domain: Option<AllocationDomainId>,
}

impl EngineTarget {
    /// Describe an engine of the registered kind `engine_id`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, EngineTarget, ExecutionContextIdentity};
    ///
    /// struct ExampleEngine;
    /// let target = EngineTarget::new(
    ///     EngineId::new("example.engine.v1").unwrap(),
    ///     ExecutionContextIdentity::of::<ExampleEngine>(),
    ///     None,
    /// );
    /// assert_eq!(target.context_identity(), ExecutionContextIdentity::of::<ExampleEngine>());
    /// ```
    pub fn new(
        engine_id: EngineId,
        context_identity: ExecutionContextIdentity,
        allocation_domain: Option<AllocationDomainId>,
    ) -> Self {
        Self {
            engine_id,
            context_identity,
            allocation_domain,
        }
    }

    /// The registered engine kind this target runs as.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, EngineTarget, ExecutionContextIdentity};
    /// struct E;
    /// let id = EngineId::new("example.engine.v1").unwrap();
    /// let target = EngineTarget::new(id.clone(), ExecutionContextIdentity::of::<E>(), None);
    /// assert_eq!(target.engine_id(), &id);
    /// ```
    pub fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    /// The concrete engine type, checked against the registration.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, EngineTarget, ExecutionContextIdentity};
    /// struct E;
    /// let target = EngineTarget::new(
    ///     EngineId::new("example.engine.v1").unwrap(),
    ///     ExecutionContextIdentity::of::<E>(),
    ///     None,
    /// );
    /// assert_eq!(target.context_identity(), ExecutionContextIdentity::of::<E>());
    /// ```
    pub fn context_identity(&self) -> ExecutionContextIdentity {
        self.context_identity
    }

    /// The memory domain the engine allocates in, `None` for plain host memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{EngineId, EngineTarget, ExecutionContextIdentity};
    /// struct E;
    /// let target = EngineTarget::new(
    ///     EngineId::new("example.engine.v1").unwrap(),
    ///     ExecutionContextIdentity::of::<E>(),
    ///     None,
    /// );
    /// assert_eq!(target.allocation_domain(), None);
    /// ```
    pub fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.allocation_domain
    }
}

/// An engine the runtime can run prepared programs on.
///
/// Every operation runs on the [`Engine`] the caller passes in; this trait adds
/// what the scheduler needs from that engine: its target description, its own
/// extension caches, its event-domain driver and the hand-off point for device
/// work that is still running when a call returns. A CPU engine finishes its
/// work before returning, so the defaults (no caches, immediate events, drain
/// on hand-off) already describe it.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{EngineTarget, ExecutionEngine};
///
/// fn describe(engine: &dyn ExecutionEngine) -> EngineTarget {
///     engine.target()
/// }
/// ```
pub trait ExecutionEngine: Engine {
    /// Immutable description used by [`crate::Runtime::prepare`].
    fn target(&self) -> EngineTarget;

    /// Move this engine's extension caches out for the duration of one run.
    ///
    /// The runtime calls [`Self::restore_extension_caches`] with the same store
    /// when the run ends, on success and on error. The default keeps no caches.
    fn take_extension_caches(&mut self) -> ExtensionCacheStore {
        ExtensionCacheStore::new()
    }

    /// Return the extension caches taken by [`Self::take_extension_caches`].
    fn restore_extension_caches(&mut self, caches: ExtensionCacheStore) {
        drop(caches);
    }

    /// The driver that opens this engine's event-domain runs.
    ///
    /// Host engines complete each launch before it returns, which the default
    /// immediate driver describes.
    fn event_domain_driver(&self) -> Arc<dyn EventDomainDriver> {
        Arc::new(ImmediateEventDomainDriver::new())
    }

    /// Take ownership of an event-domain run whose launches may still be running.
    ///
    /// The runtime calls this after a run's outputs have been collected. An
    /// engine that executes asynchronously keeps the run (and with it the
    /// memory its unfinished work uses) and retires it later; the default
    /// waits for it now.
    ///
    /// # Errors
    ///
    /// Returns the error the run reports while it is retired.
    fn retire_run(&mut self, mut run: Box<dyn EventDomainRun>) -> crate::Result<()> {
        run.drain()
    }

    /// Typed access for prepared operations that execute against the concrete
    /// engine type instead of the [`Engine`] operations.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// One output of [`crate::Runtime::run`].
///
/// A value the program computed is returned as a new tensor with a compact
/// layout. An output that is just one of the inputs, or a view of one, is
/// returned as that input's view instead of a copy; Rust's borrow rules keep
/// the input alive for as long as the output is used.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::Output;
/// use tenferro_tensor::{Tensor, TensorRead};
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let output = Output::Borrowed(TensorRead::from_tensor(&tensor));
/// assert_eq!(output.read().shape(), &[2]);
/// let owned = Output::Owned(Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap());
/// assert!(owned.into_owned().is_some());
/// ```
// The borrowed variant carries a `TensorRead` by value so the output keeps
// the input's layout and ownership without an allocation on the common path;
// the size difference is the accepted cost (phase decision, T1).
#[allow(clippy::large_enum_variant)]
pub enum Output<'a> {
    /// A tensor the program computed.
    Owned(Tensor),
    /// One of the inputs, or a view of one, returned without a copy.
    Borrowed(TensorRead<'a>),
}

impl<'a> Output<'a> {
    /// Read the output, owned or borrowed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::Output;
    /// use tenferro_tensor::Tensor;
    /// let output = Output::Owned(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap());
    /// assert_eq!(output.read().shape(), &[1]);
    /// ```
    pub fn read(&self) -> TensorRead<'_> {
        match self {
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
            Self::Borrowed(read) => read.clone(),
        }
    }

    /// The owned tensor, or `None` for a borrowed output.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::Output;
    /// use tenferro_tensor::{Tensor, TensorRead};
    /// let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// assert!(Output::Borrowed(TensorRead::from_tensor(&tensor)).into_owned().is_none());
    /// ```
    pub fn into_owned(self) -> Option<Tensor> {
        match self {
            Self::Owned(tensor) => Some(tensor),
            Self::Borrowed(_) => None,
        }
    }

    /// The output as an owned tensor, copying a borrowed output on `engine`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::Output;
    /// use tenferro_tensor::{Engine, Tensor};
    ///
    /// fn owned(output: Output<'_>, engine: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
    ///     output.into_tensor(engine)
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns the engine's error when copying a borrowed output fails.
    pub fn into_tensor(self, engine: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
        match self {
            Self::Owned(tensor) => Ok(tensor),
            Self::Borrowed(read) => engine.to_contiguous_read(read),
        }
    }
}

impl fmt::Debug for Output<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Owned(tensor) => formatter
                .debug_struct("Output::Owned")
                .field("dtype", &tensor.dtype())
                .field("shape", &tensor.shape())
                .finish(),
            Self::Borrowed(read) => formatter
                .debug_struct("Output::Borrowed")
                .field("dtype", &read.dtype())
                .field("shape", &read.shape())
                .finish(),
        }
    }
}
