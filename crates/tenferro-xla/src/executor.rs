use std::fmt;

use tenferro_runtime::program::SemanticProgram;
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
/// let y = x.neg().unwrap();
/// let program = compiler.compile(&y).unwrap();
/// let module = XlaExecutor::default()
///     .lower_to_stablehlo(program.semantic_program())
///     .unwrap();
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
    ///
    /// # Errors
    ///
    /// Without `pjrt`, returns `Error::PjrtFeatureDisabled`. With `pjrt`,
    /// returns `Error::MissingEnv` for an unset plugin path or
    /// `Error::PluginLoad` with the typed dynamic-library source.
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
    ///
    /// # Errors
    ///
    /// Returns `Error::MissingEnv` when the configured plugin-path variable is
    /// unset, or `Error::PluginLoad` with the typed dynamic-library source
    /// when the path cannot be loaded.
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
    ///
    /// # Errors
    ///
    /// Returns `Error::MissingEnv` when `var` is unset, or `Error::PluginLoad`
    /// with the typed dynamic-library source when its value cannot be loaded.
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
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let module = XlaExecutor::default().lower_to_stablehlo(program.semantic_program()).unwrap();
    /// assert!(module.as_str().contains("stablehlo.negate"));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Error::UnsupportedDType`, `Error::UnsupportedOp`, or
    /// `Error::NonStaticShape` for unsupported graph content, and
    /// `Error::InvalidProgram` for inconsistent graph metadata.
    pub fn lower_to_stablehlo(&self, program: &SemanticProgram) -> Result<StableHloModule> {
        lower_to_stablehlo(program)
    }

    /// Execute a graph program through a loaded PJRT plugin and return all outputs.
    ///
    /// Inputs must match the ordered semantic-program input metadata exactly. This
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
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let input = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let err = XlaExecutor::default()
    ///     .run_many_with_inputs(program.semantic_program(), &[&input])
    ///     .unwrap_err();
    /// assert!(matches!(err, Error::PjrtFeatureDisabled | Error::PjrtPluginNotLoaded));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Error::PjrtFeatureDisabled` or `Error::PjrtPluginNotLoaded`
    /// when no PJRT executor is available, `Error::InvalidProgram` for input
    /// count/dtype/shape mismatches, and `Error::PjrtCall` for vendor status
    /// failures.
    pub fn run_many_with_inputs(
        &self,
        program: &SemanticProgram,
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
    /// let y = x.neg().unwrap();
    /// let program = compiler.compile(&y).unwrap();
    /// let input = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let err = XlaExecutor::default().run_with_inputs(program.semantic_program(), &[&input]).unwrap_err();
    /// assert!(matches!(err, Error::PjrtFeatureDisabled | Error::PjrtPluginNotLoaded));
    /// ```
    ///
    /// # Errors
    ///
    /// Propagates the `run_many_with_inputs` errors and returns
    /// `Error::InvalidProgram` if the program does not have exactly one
    /// output.
    pub fn run_with_inputs(&self, program: &SemanticProgram, inputs: &[&Tensor]) -> Result<Tensor> {
        single_output_tensor(self.run_many_with_inputs(program, inputs)?)
    }
}

impl Default for XlaExecutor {
    fn default() -> Self {
        Self::new(XlaExecutorOptions::default())
    }
}

fn single_output_tensor(mut outputs: Vec<Tensor>) -> Result<Tensor> {
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

#[cfg(test)]
mod tests {
    use super::{single_output_tensor, XlaExecutor, XlaExecutorOptions};
    use crate::Error;
    use tenferro_runtime::{GraphCompiler, TracedTensor};
    use tenferro_tensor::Tensor;

    #[test]
    fn executor_options_and_debug_are_directly_covered() {
        let options = XlaExecutorOptions::default();
        let executor = XlaExecutor::new(options);

        assert_eq!(executor.options(), options);
        assert!(!executor.has_loaded_pjrt_plugin());
        assert_eq!(format!("{options:?}"), "XlaExecutorOptions");
        assert!(format!("{executor:?}").contains("has_loaded_pjrt_plugin"));
    }

    #[test]
    fn default_executor_reports_missing_pjrt_before_dispatch() {
        let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
        let mut compiler = GraphCompiler::new();
        let program = compiler.compile(&x.neg().unwrap()).unwrap();
        let input = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();

        let err = XlaExecutor::default()
            .run_many_with_inputs(program.semantic_program(), &[&input])
            .unwrap_err();
        assert!(matches!(
            err,
            Error::PjrtFeatureDisabled | Error::PjrtPluginNotLoaded
        ));
        let err = XlaExecutor::default()
            .run_with_inputs(program.semantic_program(), &[&input])
            .unwrap_err();
        assert!(matches!(
            err,
            Error::PjrtFeatureDisabled | Error::PjrtPluginNotLoaded
        ));
    }

    #[test]
    fn single_output_tensor_rejects_zero_or_multiple_outputs() {
        let tensor = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
        assert_eq!(
            single_output_tensor(vec![tensor.clone()])
                .unwrap()
                .as_slice::<f64>()
                .unwrap(),
            &[1.0]
        );

        let err = single_output_tensor(Vec::new()).unwrap_err();
        assert!(
            err.to_string().contains("got 0"),
            "expected zero-output error, got {err:?}"
        );

        let err = single_output_tensor(vec![tensor.clone(), tensor]).unwrap_err();
        assert!(
            err.to_string().contains("got 2"),
            "expected multi-output error, got {err:?}"
        );
    }
}
