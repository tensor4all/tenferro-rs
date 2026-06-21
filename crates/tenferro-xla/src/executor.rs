use std::fmt;

use tenferro_runtime::GraphProgram;
use tenferro_tensor::Tensor;

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

    /// Execute a graph program through a loaded PJRT plugin and return all outputs.
    ///
    /// Inputs must match [`GraphProgram::input_specs`] exactly. This
    /// experimental execution path supports the same exact-static-shape,
    /// `F32`/`F64` subset as StableHLO lowering.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    /// use tenferro_tensor::Tensor;
    /// use tenferro_xla::{Error, XlaExecutor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x.neg()).unwrap();
    /// let input = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let err = XlaExecutor::default()
    ///     .run_many_with_inputs(&program, &[&input])
    ///     .unwrap_err();
    /// assert!(matches!(err, Error::PjrtFeatureDisabled | Error::PjrtPluginNotLoaded));
    /// ```
    pub fn run_many_with_inputs(
        &self,
        program: &GraphProgram,
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        #[cfg(feature = "pjrt")]
        {
            let Some(plugin) = self.plugin.as_ref() else {
                return Err(Error::PjrtPluginNotLoaded);
            };
            crate::pjrt::run_many_with_inputs(plugin, program, inputs)
        }
        #[cfg(not(feature = "pjrt"))]
        {
            let _ = (program, inputs);
            Err(Error::PjrtFeatureDisabled)
        }
    }

    /// Execute a single-output graph program through a loaded PJRT plugin.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{GraphCompiler, TracedTensor};
    /// use tenferro_tensor::Tensor;
    /// use tenferro_xla::{Error, XlaExecutor};
    ///
    /// let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let mut compiler = GraphCompiler::new();
    /// let program = compiler.compile(&x.neg()).unwrap();
    /// let input = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let err = XlaExecutor::default().run_with_inputs(&program, &[&input]).unwrap_err();
    /// assert!(matches!(err, Error::PjrtFeatureDisabled | Error::PjrtPluginNotLoaded));
    /// ```
    pub fn run_with_inputs(&self, program: &GraphProgram, inputs: &[&Tensor]) -> Result<Tensor> {
        let mut outputs = self.run_many_with_inputs(program, inputs)?;
        if outputs.len() != 1 {
            return Err(crate::Error::InvalidProgram {
                message: format!(
                    "PJRT single-output execution expected 1 output, got {}",
                    outputs.len()
                ),
            });
        }
        Ok(outputs.remove(0))
    }
}

impl Default for XlaExecutor {
    fn default() -> Self {
        Self::new(XlaExecutorOptions::default())
    }
}
