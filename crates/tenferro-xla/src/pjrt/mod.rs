use std::path::PathBuf;

use crate::{Error, Result};

mod plugin;
mod sys;

pub use plugin::PjrtPlugin;

pub(crate) fn plugin_path_from_env(var: &'static str) -> Result<PathBuf> {
    std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or(Error::MissingEnv { var })
}
