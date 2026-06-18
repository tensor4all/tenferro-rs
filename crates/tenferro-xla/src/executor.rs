use std::fmt;

use tenferro_runtime::GraphProgram;

#[cfg(not(feature = "pjrt"))]
use crate::Error;
use crate::{lower_to_stablehlo, Result, StableHloModule};

/// Options for the experimental XLA executor.
///
/// # Examples
///
/// ```
/// use tenferro_xla::XlaExecutorOptions;
///
/// let options = XlaExecutorOptions::default();
/// assert_eq!(options, XlaExecutorOptions::default());
/// ```
#[derive(Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct XlaExecutorOptions {
    _private: (),
}

impl fmt::Debug for XlaExecutorOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XlaExecutorOptions").finish()
    }
}

/// Experimental peer executor for XLA/PJRT.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{GraphCompiler, TracedTensor};
/// use tenferro_xla::XlaExecutor;
///
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&x.neg()).unwrap();
/// let module = XlaExecutor::default().lower_to_stablehlo(&program).unwrap();
/// assert!(module.as_str().contains("stablehlo.negate"));
/// ```
pub struct XlaExecutor {
    options: XlaExecutorOptions,
    #[cfg(feature = "pjrt")]
    plugin: Option<crate::pjrt::PjrtPlugin>,
}

impl fmt::Debug for XlaExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XlaExecutor")
            .field("options", &self.options)
            .field("has_loaded_pjrt_plugin", &self.has_loaded_pjrt_plugin())
            .finish()
    }
}

impl XlaExecutor {
    /// Create an executor with explicit options.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::{XlaExecutor, XlaExecutorOptions};
    ///
    /// let executor = XlaExecutor::new(XlaExecutorOptions::default());
    /// assert_eq!(executor.options(), XlaExecutorOptions::default());
    /// ```
    pub fn new(options: XlaExecutorOptions) -> Self {
        Self {
            options,
            #[cfg(feature = "pjrt")]
            plugin: None,
        }
    }

    /// Create an executor by loading PJRT configuration from environment variables.
    ///
    /// Without the `pjrt` feature this returns [`Error::PjrtFeatureDisabled`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::{Error, XlaExecutor};
    ///
    /// if let Err(err) = XlaExecutor::from_env() {
    ///     assert!(matches!(err, Error::PjrtFeatureDisabled | Error::MissingEnv { .. } | Error::PluginLoad { .. }));
    /// }
    /// ```
    #[cfg(not(feature = "pjrt"))]
    pub fn from_env() -> Result<Self> {
        Err(Error::PjrtFeatureDisabled)
    }

    /// Create an executor by loading PJRT configuration from environment variables.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::XlaExecutor;
    ///
    /// let _ = XlaExecutor::from_env();
    /// ```
    #[cfg(feature = "pjrt")]
    pub fn from_env() -> Result<Self> {
        Self::from_env_var(crate::TENFERRO_PJRT_PLUGIN_ENV)
    }

    /// Create an executor by loading a PJRT plugin path from a specific
    /// environment variable.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::XlaExecutor;
    ///
    /// let _ = XlaExecutor::from_env_var("__TENFERRO_XLA_DOCS_UNSET");
    /// ```
    #[cfg(feature = "pjrt")]
    pub fn from_env_var(var: &'static str) -> Result<Self> {
        let plugin = crate::pjrt::PjrtPlugin::load_from_env(var)?;
        Ok(Self {
            options: XlaExecutorOptions::default(),
            plugin: Some(plugin),
        })
    }

    /// Return the executor options.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::XlaExecutor;
    ///
    /// assert_eq!(XlaExecutor::default().options(), Default::default());
    /// ```
    pub fn options(&self) -> XlaExecutorOptions {
        self.options
    }

    /// Return whether this executor owns a loaded PJRT plugin.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::XlaExecutor;
    ///
    /// assert!(!XlaExecutor::default().has_loaded_pjrt_plugin());
    /// ```
    pub fn has_loaded_pjrt_plugin(&self) -> bool {
        #[cfg(feature = "pjrt")]
        {
            self.plugin.is_some()
        }
        #[cfg(not(feature = "pjrt"))]
        {
            false
        }
    }

    /// Lower a graph program to StableHLO without executing it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    /// use tenferro_xla::XlaExecutor;
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x.neg()).unwrap();
    /// let module = XlaExecutor::default().lower_to_stablehlo(&program).unwrap();
    /// assert!(module.as_str().contains("stablehlo.negate"));
    /// ```
    pub fn lower_to_stablehlo(&self, program: &GraphProgram) -> Result<StableHloModule> {
        lower_to_stablehlo(program)
    }
}

impl Default for XlaExecutor {
    fn default() -> Self {
        Self::new(XlaExecutorOptions::default())
    }
}
