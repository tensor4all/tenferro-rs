//! Runtime FFTW loading and lane plans.
//!
//! Every FFTW plan creation and destruction in the process passes through the
//! planner hooks installed here, which serialize them under one lock. A tenferro
//! plan sets FFTW's process-wide planner thread count and restores it inside
//! that lock, so no other FFTW caller plans with tenferro's count.
use std::any::Any;
use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::ffi::{c_int, c_uint, c_void};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use libloading::Library;
use num_complex::Complex;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExtensionCacheKey, ExtensionCacheStore};

use crate::{FftOperation, FFT_EXTENSION_FAMILY_ID};

#[cfg(test)]
mod tests;

// Runtimes that installed tenferro's planner hooks in their FFTW installation
// or initialized its threads; the flag records the latter. FFTW keeps
// process-wide pointers into these libraries (the threading library's solver
// hooks), so they stay loaded after their last plan is evicted.
static LOADED: Mutex<Vec<(Arc<Api>, bool)>> = Mutex::new(Vec::new());
const UNALIGNED: c_uint = 1 << 1;

/// State of the planner lock that tenferro's hooks take for every FFTW caller.
struct Planner {
    busy: bool,
    /// Planner thread count to restore when the current holder leaves.
    restore: Option<(SetThreads, c_int)>,
}

static PLANNER: Mutex<Planner> = Mutex::new(Planner {
    busy: false,
    restore: None,
});
static PLANNER_FREE: Condvar = Condvar::new();

// Both cells are `Copy` with a const initializer, so they have no destructor and
// stay accessible on every thread for its whole life, including while
// thread-local destructors run; FFTW may call the hooks from any thread.
thread_local! {
    /// Planner thread count the next plan created on this thread must use.
    static REQUEST: Cell<Option<(ThreadApi, c_int)>> = const { Cell::new(None) };
    /// Whether this thread holds the planner lock.
    static HOLDS_PLANNER: Cell<bool> = const { Cell::new(false) };
}

/// FFTW calls this before it creates or destroys any plan in the process.
extern "C" fn before_planning() {
    let request = REQUEST.with(Cell::take);
    let mut planner = PLANNER.lock().unwrap_or_else(PoisonError::into_inner);
    while planner.busy {
        planner = PLANNER_FREE
            .wait(planner)
            .unwrap_or_else(PoisonError::into_inner);
    }
    planner.busy = true;
    HOLDS_PLANNER.with(|holds| holds.set(true));
    planner.restore = request.and_then(|(threads, count)| {
        // SAFETY: the planner lock is held, so no FFTW planner reads or writes
        // the count concurrently. The count differs from the default of one
        // only after `plan_with_nthreads`, which initializes FFTW's threads, so
        // a count that needs changing never triggers that initialization.
        let previous = unsafe { (threads.get)() };
        (previous != count).then(|| {
            // SAFETY: as above.
            unsafe { (threads.set)(count) };
            (threads.set, previous)
        })
    });
}

/// FFTW calls this after it creates or destroys any plan in the process.
extern "C" fn after_planning() {
    // A plan that was already inside FFTW's planner when the hooks were
    // installed returns through this hook without holding the lock.
    if !HOLDS_PLANNER.with(|holds| holds.replace(false)) {
        return;
    }
    let mut planner = PLANNER.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((set, previous)) = planner.restore.take() {
        // SAFETY: the planner lock is still held.
        unsafe { set(previous) };
    }
    planner.busy = false;
    drop(planner);
    PLANNER_FREE.notify_one();
}

type PlanComplex =
    unsafe extern "C" fn(c_int, *mut c_void, *mut c_void, c_int, c_uint) -> *mut c_void;
type PlanReal = unsafe extern "C" fn(c_int, *mut c_void, *mut c_void, c_uint) -> *mut c_void;
type Execute = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
type Destroy = unsafe extern "C" fn(*mut c_void);
type InitThreads = unsafe extern "C" fn() -> c_int;
type SetThreads = unsafe extern "C" fn(c_int);
type GetThreads = unsafe extern "C" fn() -> c_int;
type MakePlannerThreadSafe = unsafe extern "C" fn();
type SetPlannerHooks = unsafe extern "C" fn(extern "C" fn(), extern "C" fn());

/// FFTW 3.3.9+ threading entry points, from the core or the companion library.
#[derive(Clone, Copy, Debug)]
struct ThreadApi {
    init: InitThreads,
    set: SetThreads,
    get: GetThreads,
    make_planner_thread_safe: MakePlannerThreadSafe,
}

/// Describes an FFTW runtime or planning failure.
///
/// # Examples
/// ```
/// use tenferro_fft::FftwError;
/// let error = FftwError::ThreadCount(usize::MAX);
/// assert!(error.to_string().contains("thread count"));
/// ```
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FftwError {
    /// A required FFTW shared library could not be loaded.
    #[error("could not load FFTW library {library:?}: {source}")]
    Load {
        /// Library name or explicit path.
        library: PathBuf,
        /// Dynamic loader error.
        #[source]
        source: libloading::Error,
    },
    /// A required FFTW symbol was missing.
    #[error("FFTW symbol {symbol} is unavailable in {library:?}: {source}")]
    Symbol {
        /// Library searched for the symbol.
        library: PathBuf,
        /// Name of the symbol.
        symbol: String,
        /// Dynamic loader error.
        #[source]
        source: libloading::Error,
    },
    /// The threading library belongs to a different FFTW installation.
    #[error("FFTW threading library {0:?} does not use the selected core library")]
    ThreadLibraryMismatch(PathBuf),
    /// Thread count exceeds the FFTW C API limit.
    #[error("FFTW thread count {0} does not fit in a C int")]
    ThreadCount(usize),
    /// FFTW returned zero from thread initialization.
    #[error("FFTW could not initialize its threads")]
    ThreadInitialization,
    /// Transform length exceeds the FFTW C API limit.
    #[error("FFTW transform length {0} does not fit in a C int")]
    Length(usize),
    /// FFTW did not produce a plan.
    #[error("FFTW could not plan {operation:?} of length {len}")]
    Plan {
        /// Requested operation.
        operation: FftOperation,
        /// FFT length.
        len: usize,
    },
}

/// FFTW planner effort for newly cached plans.
///
/// # Examples
/// ```
/// use tenferro_fft::FftwPlanning;
/// assert_eq!(FftwPlanning::default(), FftwPlanning::Estimate);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum FftwPlanning {
    /// Select a plan without timing candidate plans.
    #[default]
    Estimate,
    /// Time candidate plans with private planning buffers.
    Measure,
}

/// Explicit CPU FFTW provider loaded from the system or a specified library directory.
///
/// # Examples
/// ```
/// use std::num::NonZeroUsize;
/// use tenferro_cpu::CpuEngine;
/// use tenferro_fft::{Fftw, FftwPlanning};
/// let mut cpu = CpuEngine::serial();
/// if let Ok(fftw) = Fftw::load(FftwPlanning::Estimate, NonZeroUsize::MIN) {
///     fftw.install(&mut cpu);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Fftw(Arc<Runtime>);

impl Fftw {
    /// Load both FFTW precisions using the platform's shared library search path.
    ///
    /// # Errors
    /// Returns [`FftwError::Load`] if a required library cannot be opened or
    /// [`FftwError::Symbol`] if a required symbol is missing.
    /// Returns [`FftwError::ThreadCount`] if the count exceeds the C API limit,
    /// [`FftwError::ThreadLibraryMismatch`] if the companion library uses a
    /// different FFTW installation, or [`FftwError::ThreadInitialization`]
    /// if FFTW fails to initialize threads.
    ///
    /// A thread count above one requires FFTW 3.3.9 or later, the first release
    /// that exports `fftw_planner_nthreads`.
    ///
    /// Loading installs planner hooks (`fftw_set_planner_hooks`) in each FFTW
    /// library. They serialize every plan creation and destruction in the
    /// process, other libraries' included, like `fftw_make_planner_thread_safe`,
    /// whose hooks they replace. Inside that lock a tenferro plan sets FFTW's
    /// process-wide planner thread count to the provider's count, or to one for
    /// a plan that runs inside a parallel lane, and restores it afterward. A
    /// single-threaded provider can do this only when the installation exports
    /// FFTW's thread API, in the core library or in `fftw3_threads`; without it,
    /// the provider plans with whatever count it finds.
    ///
    /// `fftw_plan_with_nthreads` itself takes no lock, so other code that calls
    /// it while any thread is planning races with that planning; FFTW leaves
    /// serializing it to the caller. Installing the hooks is not atomic either:
    /// load the provider before other threads start planning with FFTW, and do
    /// not let other code call `fftw_set_planner_hooks` afterward.
    ///
    /// # Examples
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro_fft::{Fftw, FftwPlanning};
    /// let _ = Fftw::load(FftwPlanning::Estimate, NonZeroUsize::MIN);
    /// ```
    pub fn load(planning: FftwPlanning, threads: NonZeroUsize) -> Result<Self, FftwError> {
        Self::load_impl(None, planning, threads)
    }

    /// Load both FFTW precisions from the given library directory.
    /// The directory must contain a trusted FFTW 3 installation because loading
    /// a shared library can execute native code.
    ///
    /// # Errors
    /// Returns [`FftwError::Load`] if a required library cannot be opened or
    /// [`FftwError::Symbol`] if a required symbol is missing.
    /// Returns [`FftwError::ThreadCount`] if the count exceeds the C API limit,
    /// [`FftwError::ThreadLibraryMismatch`] if the companion library uses a
    /// different FFTW installation, or [`FftwError::ThreadInitialization`]
    /// if FFTW fails to initialize threads.
    ///
    /// A thread count above one requires FFTW 3.3.9 or later, the first release
    /// that exports `fftw_planner_nthreads`.
    ///
    /// Loading installs planner hooks (`fftw_set_planner_hooks`) in each FFTW
    /// library. They serialize every plan creation and destruction in the
    /// process, other libraries' included, like `fftw_make_planner_thread_safe`,
    /// whose hooks they replace. Inside that lock a tenferro plan sets FFTW's
    /// process-wide planner thread count to the provider's count, or to one for
    /// a plan that runs inside a parallel lane, and restores it afterward. A
    /// single-threaded provider can do this only when the installation exports
    /// FFTW's thread API, in the core library or in `fftw3_threads`; without it,
    /// the provider plans with whatever count it finds.
    ///
    /// `fftw_plan_with_nthreads` itself takes no lock, so other code that calls
    /// it while any thread is planning races with that planning; FFTW leaves
    /// serializing it to the caller. Installing the hooks is not atomic either:
    /// load the provider before other threads start planning with FFTW, and do
    /// not let other code call `fftw_set_planner_hooks` afterward.
    ///
    /// # Examples
    /// ```
    /// use std::{num::NonZeroUsize, path::Path};
    /// use tenferro_fft::{Fftw, FftwPlanning};
    /// let _ = Fftw::load_from(Path::new("/usr/local/lib"), FftwPlanning::Estimate, NonZeroUsize::MIN);
    /// ```
    pub fn load_from(
        directory: &Path,
        planning: FftwPlanning,
        threads: NonZeroUsize,
    ) -> Result<Self, FftwError> {
        Self::load_impl(Some(directory), planning, threads)
    }

    fn load_impl(
        directory: Option<&Path>,
        planning: FftwPlanning,
        threads: NonZeroUsize,
    ) -> Result<Self, FftwError> {
        let threads =
            c_int::try_from(threads.get()).map_err(|_| FftwError::ThreadCount(threads.get()))?;
        let flags = UNALIGNED
            | match planning {
                FftwPlanning::Estimate => 1 << 6,
                FftwPlanning::Measure => 0,
            };
        let f64 = Arc::new(Api::load("fftw", "fftw3", directory, threads > 1)?);
        let f32 = Arc::new(Api::load("fftwf", "fftw3f", directory, threads > 1)?);
        f64.install(threads > 1)?;
        f32.install(threads > 1)?;
        Ok(Self(Arc::new(Runtime {
            f64,
            f32,
            flags,
            threads,
        })))
    }

    /// Select this provider for FFT calls on the given engine.
    ///
    /// # Examples
    /// ```
    /// use std::num::NonZeroUsize;
    /// use tenferro_cpu::CpuEngine;
    /// use tenferro_fft::{Fftw, FftwPlanning};
    /// let mut cpu = CpuEngine::serial();
    /// if let Ok(fftw) = Fftw::load(FftwPlanning::Measure, NonZeroUsize::MIN) {
    ///     fftw.install(&mut cpu);
    /// }
    /// ```
    pub fn install(&self, engine: &mut CpuEngine) {
        engine.set_fft_provider(self.0.clone());
    }

    /// The FFTW provider stored by [`Self::install`], or `None` for any other value.
    pub(crate) fn from_provider(provider: &Arc<dyn Any + Send + Sync>) -> Option<Self> {
        Arc::clone(provider).downcast::<Runtime>().ok().map(Self)
    }

    pub(crate) fn threaded(&self) -> bool {
        self.0.threads > 1
    }

    /// Cached plan for `operation` of length `len`. `threaded` plans with the
    /// runtime's FFTW thread count; otherwise the plan is single-threaded.
    pub(crate) fn plan<T: crate::cache::CachedFftPlanScalar>(
        &self,
        store: &mut ExtensionCacheStore,
        len: usize,
        operation: FftOperation,
        threaded: bool,
    ) -> tenferro_tensor::Result<Arc<LanePlan<T>>> {
        let operation = if operation == FftOperation::R2cFull {
            FftOperation::R2cOnesided
        } else {
            operation
        };
        let threads = if threaded { self.0.threads } else { 1 };
        let identity = (Arc::as_ptr(&self.0), len, operation, threads);
        let mut hasher = DefaultHasher::new();
        identity.hash(&mut hasher);
        std::any::TypeId::of::<T>().hash(&mut hasher);
        let key = ExtensionCacheKey::new(FFT_EXTENSION_FAMILY_ID, "fftw-plans", hasher.finish());
        if let Some(plan) = store.get::<Arc<LanePlan<T>>>(&key) {
            if (
                Arc::as_ptr(&plan.runtime),
                plan.len,
                plan.operation,
                plan.threads,
            ) == identity
            {
                return Ok(plan.clone());
            }
        }
        let plan = Arc::new(
            LanePlan::new(self.0.clone(), len, operation, threads)
                .map_err(|source| tenferro_tensor::Error::backend_source("fft", source))?,
        );
        store.put(key, plan.clone(), std::mem::size_of::<LanePlan<T>>());
        Ok(plan)
    }
}

#[derive(Debug)]
struct Runtime {
    f64: Arc<Api>,
    f32: Arc<Api>,
    flags: c_uint,
    threads: c_int,
}

impl Runtime {
    fn api<T>(&self) -> &Arc<Api> {
        // LanePlan is constructed only for the sealed crate-local f32 and f64 implementations.
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
            &self.f32
        } else {
            &self.f64
        }
    }
}

#[derive(Debug)]
struct Api {
    complex: PlanComplex,
    r2c: PlanReal,
    c2r: PlanReal,
    execute_complex: Execute,
    execute_r2c: Execute,
    execute_c2r: Execute,
    destroy: Destroy,
    set_planner_hooks: SetPlannerHooks,
    /// Required for a threaded runtime; otherwise present when the installation exports it.
    threads: Option<ThreadApi>,
    _library: Library,
    _thread_library: Option<Library>,
}

fn library_path(directory: Option<&Path>, stem: &str) -> PathBuf {
    let name = if cfg!(target_os = "macos") {
        format!("lib{stem}.3.dylib")
    } else if cfg!(windows) {
        format!("lib{stem}-3.dll")
    } else {
        format!("lib{stem}.so.3")
    };
    directory.map_or_else(|| PathBuf::from(&name), |path| path.join(&name))
}

fn load_library(path: &Path) -> Result<Library, FftwError> {
    // SAFETY: the caller selects a trusted FFTW installation, or the system loader finds it.
    #[cfg(unix)]
    let loaded = unsafe {
        use libloading::os::unix::{Library as UnixLibrary, RTLD_LOCAL, RTLD_NOW};
        UnixLibrary::open(Some(path), RTLD_LOCAL | RTLD_NOW).map(Library::from)
    };
    // SAFETY: as above, the library comes from the selected FFTW installation.
    #[cfg(not(unix))]
    let loaded = unsafe { Library::new(path) };
    loaded.map_err(|source| FftwError::Load {
        library: path.into(),
        source,
    })
}

impl Api {
    /// Install tenferro's planner hooks in this FFTW installation and, for a
    /// threaded runtime, initialize FFTW's threads. Each happens once per
    /// installation in the process.
    fn install(self: &Arc<Self>, threaded: bool) -> Result<(), FftwError> {
        let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
        let mut hooked = false;
        let mut threads_ready = false;
        for (other, initialized) in loaded.iter() {
            if std::ptr::fn_addr_eq(other.set_planner_hooks, self.set_planner_hooks) {
                hooked = true;
                threads_ready |= *initialized;
            }
        }
        let initialize = threaded && !threads_ready;
        if initialize {
            let threads = self.threads.ok_or(FftwError::ThreadInitialization)?;
            // SAFETY: the installation exports this signature. LOADED serializes
            // tenferro's initialization; FFTW requires it before threaded planning.
            if unsafe { (threads.init)() } == 0 {
                return Err(FftwError::ThreadInitialization);
            }
        }
        if !hooked {
            if let Some(threads) = self.threads {
                // FFTW installs its own planner hooks at most once per
                // installation. Installing them first turns later calls of
                // `fftw_make_planner_thread_safe` by other code into no-ops,
                // so they cannot replace tenferro's hooks.
                // SAFETY: the installation exports this signature.
                unsafe { (threads.make_planner_thread_safe)() };
            }
            // SAFETY: the hooks are `extern "C"` functions with FFTW's
            // `void (*)(void)` signature that live for the whole process.
            unsafe { (self.set_planner_hooks)(before_planning, after_planning) };
        }
        if !hooked || initialize {
            loaded.push((self.clone(), initialize));
        }
        Ok(())
    }

    fn load(
        prefix: &str,
        stem: &str,
        directory: Option<&Path>,
        threaded: bool,
    ) -> Result<Self, FftwError> {
        let path = library_path(directory, stem);
        let library = load_library(&path)?;
        fn symbol<T: Copy>(library: &Library, path: &Path, name: String) -> Result<T, FftwError> {
            // SAFETY: the callers use the FFTW C ABI signature of each named symbol.
            unsafe { library.get::<T>(name.as_bytes()) }
                .map(|value| *value)
                .map_err(|source| FftwError::Symbol {
                    library: path.into(),
                    symbol: name,
                    source,
                })
        }
        let complex_name = format!("{prefix}_plan_dft_1d");
        let complex: PlanComplex = symbol(&library, &path, complex_name.clone())?;
        let load_threads = || -> Result<(ThreadApi, Option<Library>), FftwError> {
            // Threading lives in the core library when it exports
            // `init_threads`, and otherwise in the companion library.
            let init_name = format!("{prefix}_init_threads");
            let companion = if symbol::<InitThreads>(&library, &path, init_name.clone()).is_ok() {
                None
            } else {
                let path = library_path(directory, &format!("{stem}_threads"));
                let library = load_library(&path)?;
                let companion: PlanComplex = symbol(&library, &path, complex_name.clone())?;
                if !std::ptr::fn_addr_eq(companion, complex) {
                    return Err(FftwError::ThreadLibraryMismatch(path));
                }
                Some((library, path))
            };
            let (thread_library, thread_path) = companion
                .as_ref()
                .map_or((&library, path.as_path()), |(library, path)| {
                    (library, path.as_path())
                });
            let threads = ThreadApi {
                init: symbol(thread_library, thread_path, init_name)?,
                set: symbol(
                    thread_library,
                    thread_path,
                    format!("{prefix}_plan_with_nthreads"),
                )?,
                get: symbol(
                    thread_library,
                    thread_path,
                    format!("{prefix}_planner_nthreads"),
                )?,
                make_planner_thread_safe: symbol(
                    thread_library,
                    thread_path,
                    format!("{prefix}_make_planner_thread_safe"),
                )?,
            };
            Ok((threads, companion.map(|(library, _)| library)))
        };
        // A single-threaded runtime uses the thread API, when the installation
        // has one, only to plan with one thread whatever count other code set.
        let (threads, thread_library) = match load_threads() {
            Ok((threads, library)) => (Some(threads), library),
            Err(error @ FftwError::ThreadLibraryMismatch(_)) => return Err(error),
            Err(error) if threaded => return Err(error),
            Err(_) => (None, None),
        };
        Ok(Self {
            complex,
            r2c: symbol(&library, &path, format!("{prefix}_plan_dft_r2c_1d"))?,
            c2r: symbol(&library, &path, format!("{prefix}_plan_dft_c2r_1d"))?,
            execute_complex: symbol(&library, &path, format!("{prefix}_execute_dft"))?,
            execute_r2c: symbol(&library, &path, format!("{prefix}_execute_dft_r2c"))?,
            execute_c2r: symbol(&library, &path, format!("{prefix}_execute_dft_c2r"))?,
            destroy: symbol(&library, &path, format!("{prefix}_destroy_plan"))?,
            set_planner_hooks: symbol(&library, &path, format!("{prefix}_set_planner_hooks"))?,
            threads,
            _library: library,
            _thread_library: thread_library,
        })
    }
}

pub(crate) struct LanePlan<T> {
    handle: NonNull<c_void>,
    runtime: Arc<Runtime>,
    len: usize,
    operation: FftOperation,
    /// FFTW planner thread count the plan was created with.
    threads: c_int,
    scalar: PhantomData<T>,
}

// SAFETY: FFTW's new-array execute routines permit concurrent execution of
// the same plan with disjoint arrays. The planner hooks serialize destruction.
unsafe impl<T: Send> Send for LanePlan<T> {}
// SAFETY: shared methods execute against caller-provided, exclusively borrowed arrays.
unsafe impl<T: Sync> Sync for LanePlan<T> {}

impl<T: crate::cache::CachedFftPlanScalar> LanePlan<T> {
    fn new(
        runtime: Arc<Runtime>,
        len: usize,
        operation: FftOperation,
        plan_threads: c_int,
    ) -> Result<Self, FftwError> {
        let n = c_int::try_from(len).map_err(|_| FftwError::Length(len))?;
        let api = runtime.api::<T>();
        let complex = matches!(
            operation,
            FftOperation::C2cForward | FftOperation::C2cInverse
        );
        let mut spectrum =
            vec![Complex::new(T::zero(), T::zero()); if complex { len } else { len / 2 + 1 }];
        let mut real = if complex {
            Vec::new()
        } else {
            vec![T::zero(); len]
        };
        let s = spectrum.as_mut_ptr().cast();
        let r = real.as_mut_ptr().cast();
        // The planner hooks set the thread count once they hold the planner lock.
        REQUEST.with(|request| request.set(api.threads.map(|threads| (threads, plan_threads))));
        // SAFETY: the pointers have FFTW's prescribed lengths and element layout.
        // The complex plan is in place. Real plans receive disjoint input and output.
        // Planning may overwrite these private allocations. The planner hooks
        // serialize the C calls with every other FFTW plan in the process.
        let raw = unsafe {
            match operation {
                FftOperation::C2cForward => (api.complex)(n, s, s, -1, runtime.flags),
                FftOperation::C2cInverse => (api.complex)(n, s, s, 1, runtime.flags),
                FftOperation::R2cFull | FftOperation::R2cOnesided => {
                    (api.r2c)(n, r, s, runtime.flags)
                }
                FftOperation::C2r => (api.c2r)(n, s, r, runtime.flags),
            }
        };
        // FFTW can reject arguments before it enters its planner; a request left
        // here would apply to the next FFTW plan created on this thread.
        REQUEST.with(|request| request.set(None));
        Ok(Self {
            handle: NonNull::new(raw).ok_or(FftwError::Plan { operation, len })?,
            runtime,
            len,
            operation,
            threads: plan_threads,
            scalar: PhantomData,
        })
    }

    pub(crate) fn complex(&self, values: &mut [Complex<T>]) {
        assert!(matches!(
            self.operation,
            FftOperation::C2cForward | FftOperation::C2cInverse
        ));
        assert_eq!(values.len(), self.len);
        let p = values.as_mut_ptr().cast();
        // SAFETY: validated in-place array of the plan's exact length and precision.
        unsafe { (self.runtime.api::<T>().execute_complex)(self.handle.as_ptr(), p, p) };
    }

    pub(crate) fn real(&self, real: &mut [T], spectrum: &mut [Complex<T>]) {
        assert_eq!(real.len(), self.len);
        assert_eq!(spectrum.len(), self.len / 2 + 1);
        let api = self.runtime.api::<T>();
        let r = real.as_mut_ptr().cast();
        let s = spectrum.as_mut_ptr().cast();
        // SAFETY: the operation determines direction and the checked mutable
        // slices are disjoint and match the original plan. UNALIGNED permits
        // different alignments from the original allocations.
        unsafe {
            match self.operation {
                FftOperation::R2cFull | FftOperation::R2cOnesided => {
                    (api.execute_r2c)(self.handle.as_ptr(), r, s)
                }
                FftOperation::C2r => (api.execute_c2r)(self.handle.as_ptr(), s, r),
                _ => unreachable!("real FFTW execution requires a real plan"),
            }
        }
    }
}

impl<T> Drop for LanePlan<T> {
    fn drop(&mut self) {
        // SAFETY: this plan is uniquely owned at destruction and the library is
        // still loaded. The planner hooks serialize destruction with planning.
        unsafe { (self.runtime.api::<T>().destroy)(self.handle.as_ptr()) };
    }
}
