use std::fmt;
use std::path::{Path, PathBuf};

use libloading::Library;

use crate::{Error, Result};

use super::sys::{GetPjrtApiFn, PJRT_Api};

/// Dynamically loaded PJRT plugin.
///
/// # Examples
///
/// ```
/// use tenferro_xla::{Error, PjrtPlugin};
///
/// let err = PjrtPlugin::load_from_env("__TENFERRO_XLA_DOCS_UNSET").unwrap_err();
/// assert!(matches!(err, Error::MissingEnv { .. }));
/// ```
pub struct PjrtPlugin {
    path: PathBuf,
    _api: *const PJRT_Api,
    _library: Library,
}

impl fmt::Debug for PjrtPlugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PjrtPlugin")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

impl PjrtPlugin {
    /// Load a PJRT plugin path from an environment variable.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::{Error, PjrtPlugin};
    ///
    /// let err = PjrtPlugin::load_from_env("__TENFERRO_XLA_DOCS_UNSET").unwrap_err();
    /// assert!(matches!(err, Error::MissingEnv { .. }));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Error::MissingEnv` when `var` is unset, or `Error::PluginLoad`
    /// with the typed dynamic-library source when loading or symbol lookup
    /// fails.
    pub fn load_from_env(var: &'static str) -> Result<Self> {
        let path = super::plugin_path_from_env(var)?;
        Self::load_path(path)
    }

    /// Load a PJRT plugin from an explicit dynamic-library path.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_xla::{Error, PjrtPlugin};
    ///
    /// let err = PjrtPlugin::load_path("/definitely/missing/pjrt.so").unwrap_err();
    /// assert!(matches!(err, Error::PluginLoad { .. }));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `Error::PluginLoad` with the original dynamic-library error as
    /// its source when the file cannot be opened or `GetPjrtApi` is missing,
    /// or `Error::PjrtCall` when the plugin returns a null API table.
    pub fn load_path(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        // SAFETY: Loading a dynamic library is inherently unsafe. The library is
        // retained in `Self` for at least as long as the API pointer is exposed.
        let library = unsafe { Library::new(&path) }.map_err(|source| Error::PluginLoad {
            path: path.clone(),
            source: Box::new(source),
        })?;
        // SAFETY: OpenXLA PJRT plugins export `GetPjrtApi` with this signature.
        // The returned table is owned by the plugin and remains valid while the
        // dynamic library is loaded.
        let api = unsafe {
            let symbol: libloading::Symbol<'_, GetPjrtApiFn> =
                library
                    .get(b"GetPjrtApi")
                    .map_err(|source| Error::PluginLoad {
                        path: path.clone(),
                        source: Box::new(source),
                    })?;
            symbol()
        };
        if api.is_null() {
            return Err(Error::PjrtCall {
                call: "GetPjrtApi",
                message: format!("plugin at {path:?} returned a null API table"),
            });
        }
        Ok(Self {
            path,
            _api: api,
            _library: library,
        })
    }

    /// Return the path used to load the plugin.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use tenferro_xla::PjrtPlugin;
    ///
    /// let _path_type = std::any::type_name::<&Path>();
    /// let _method: fn(&PjrtPlugin) -> &Path = PjrtPlugin::path;
    /// ```
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn api(&self) -> *const PJRT_Api {
        self._api
    }
}
