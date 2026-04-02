use std::path::{Path, PathBuf};
use std::{env, fs};

use sha2::{Digest, Sha256};
use tenferro_device::{Error, Result};

fn module_key(
    tag: &[u8],
    source: &str,
    sm_major: i32,
    sm_minor: i32,
    driver_version: i32,
    compile_options: &[String],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(tag);
    hasher.update(source.as_bytes());
    hasher.update(sm_major.to_le_bytes());
    hasher.update(sm_minor.to_le_bytes());
    hasher.update(driver_version.to_le_bytes());
    for option in compile_options {
        hasher.update(option.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub(super) fn pointwise_real_module_key(
    source: &str,
    sm_major: i32,
    sm_minor: i32,
    driver_version: i32,
    compile_options: &[String],
) -> String {
    module_key(
        b"tenferro-prims-cuda-pointwise-real-v1",
        source,
        sm_major,
        sm_minor,
        driver_version,
        compile_options,
    )
}

pub(super) fn pointwise_complex_module_key(
    source: &str,
    sm_major: i32,
    sm_minor: i32,
    driver_version: i32,
    compile_options: &[String],
) -> String {
    module_key(
        b"tenferro-prims-cuda-pointwise-complex-v1",
        source,
        sm_major,
        sm_minor,
        driver_version,
        compile_options,
    )
}

pub(super) fn load_ptx(cache_key: &str) -> Result<Option<String>> {
    let path = artifact_path(cache_key)?;
    if !path.exists() {
        return Ok(None);
    }
    fs::read_to_string(&path)
        .map(Some)
        .map_err(|err| Error::DeviceError(format!("Failed to read cached PTX {path:?}: {err}")))
}

pub(super) fn store_ptx(cache_key: &str, ptx: &str) -> Result<()> {
    let path = artifact_path(cache_key)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            Error::DeviceError(format!(
                "Failed to create CUDA cache directory {parent:?}: {err}"
            ))
        })?;
    }
    fs::write(&path, ptx)
        .map_err(|err| Error::DeviceError(format!("Failed to write cached PTX {path:?}: {err}")))
}

fn artifact_path(cache_key: &str) -> Result<PathBuf> {
    Ok(cache_root()?.join(format!("{cache_key}.ptx")))
}

fn cache_root() -> Result<PathBuf> {
    if let Some(dir) = env::var_os("TENFERRO_CACHE_DIR") {
        return Ok(PathBuf::from(dir).join("cuda"));
    }
    if let Some(dir) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(dir).join("tenferro").join("cuda"));
    }
    let Some(home) = env::var_os("HOME") else {
        return Err(Error::DeviceError(
            "Unable to resolve CUDA cache directory: HOME is unset".into(),
        ));
    };
    Ok(Path::new(&home)
        .join(".cache")
        .join("tenferro")
        .join("cuda"))
}
