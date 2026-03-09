use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct DbTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub order: String,
    pub data: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Comparison {
    pub kind: String,
    #[serde(default)]
    pub rtol: f64,
    #[serde(default)]
    pub atol: f64,
    #[serde(default)]
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Observable {
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PytorchRef {
    pub jvp: BTreeMap<String, DbTensor>,
    pub vjp: BTreeMap<String, DbTensor>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct FdRef {
    pub method: String,
    pub stencil_order: usize,
    pub step: f64,
    pub jvp: BTreeMap<String, DbTensor>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ProbeRecord {
    pub probe_id: String,
    pub direction: BTreeMap<String, DbTensor>,
    pub cotangent: BTreeMap<String, DbTensor>,
    pub pytorch_ref: PytorchRef,
    pub fd_ref: FdRef,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaseRecord {
    pub case_id: String,
    pub op: String,
    pub dtype: String,
    pub family: String,
    pub expected_behavior: String,
    pub inputs: BTreeMap<String, DbTensor>,
    pub observable: Observable,
    pub comparison: Comparison,
    pub probes: Vec<ProbeRecord>,
}

pub fn default_oracle_db_root() -> Option<PathBuf> {
    if let Some(root) = env::var_os("TENSOR_AD_ORACLES_ROOT") {
        let path = PathBuf::from(root);
        if path.is_dir() {
            return Some(path);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let sibling = manifest_dir.parent()?.parent()?.join("tensor-ad-oracles");
    sibling.is_dir().then_some(sibling)
}

pub fn case_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let cases_root = root.join("cases");
    let mut files = Vec::new();
    let entries = fs::read_dir(&cases_root)
        .map_err(|err| format!("failed to read {}: {err}", cases_root.display()))?;
    for op_entry in entries {
        let op_entry = op_entry
            .map_err(|err| format!("failed to read entry in {}: {err}", cases_root.display()))?;
        let op_path = op_entry.path();
        if !op_path.is_dir() {
            continue;
        }
        let op_entries = fs::read_dir(&op_path)
            .map_err(|err| format!("failed to read {}: {err}", op_path.display()))?;
        for case_entry in op_entries {
            let case_entry = case_entry
                .map_err(|err| format!("failed to read entry in {}: {err}", op_path.display()))?;
            let case_path = case_entry.path();
            if case_path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(case_path);
            }
        }
    }
    files.sort();
    Ok(files)
}

pub fn load_case_records(path: &Path) -> Result<Vec<CaseRecord>, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<CaseRecord>(line)
                .map_err(|err| format!("failed to parse {}: {err}", path.display()))
        })
        .collect()
}
