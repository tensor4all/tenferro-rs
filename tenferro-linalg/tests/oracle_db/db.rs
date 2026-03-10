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
pub struct ComparisonTolerance {
    pub kind: String,
    pub rtol: f64,
    pub atol: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ErrorComparison {
    pub kind: String,
    pub reason_code: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum Comparison {
    Success {
        first_order: ComparisonTolerance,
        second_order: ComparisonTolerance,
    },
    Error(ErrorComparison),
}

impl Comparison {
    pub fn first_order(&self) -> Option<&ComparisonTolerance> {
        match self {
            Comparison::Success { first_order, .. } => Some(first_order),
            Comparison::Error(_) => None,
        }
    }

    pub fn second_order(&self) -> Option<&ComparisonTolerance> {
        match self {
            Comparison::Success { second_order, .. } => Some(second_order),
            Comparison::Error(_) => None,
        }
    }

    pub fn error(&self) -> Option<&ErrorComparison> {
        match self {
            Comparison::Success { .. } => None,
            Comparison::Error(error) => Some(error),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Observable {
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct PytorchRef {
    pub jvp: BTreeMap<String, DbTensor>,
    #[serde(default)]
    pub hvp: Option<BTreeMap<String, DbTensor>>,
    pub vjp: BTreeMap<String, DbTensor>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct FdRef {
    pub method: String,
    pub stencil_order: usize,
    pub step: f64,
    pub jvp: BTreeMap<String, DbTensor>,
    #[serde(default)]
    pub hvp: Option<BTreeMap<String, DbTensor>>,
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
#[allow(dead_code)]
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

impl ProbeRecord {
    fn validate(&self) -> Result<(), String> {
        match (&self.pytorch_ref.hvp, &self.fd_ref.hvp) {
            (Some(_), Some(_)) | (None, None) => Ok(()),
            (Some(_), None) | (None, Some(_)) => Err(format!(
                "probe {} has half-present HVP payloads",
                self.probe_id
            )),
        }
    }
}

impl CaseRecord {
    fn validate(self) -> Result<Self, String> {
        match self.expected_behavior.as_str() {
            "success" => {
                if self.comparison.first_order().is_none() || self.comparison.second_order().is_none()
                {
                    return Err(format!(
                        "success case {} must use success comparison schema",
                        self.case_id
                    ));
                }
            }
            "error" => {
                if self.comparison.error().is_none() {
                    return Err(format!(
                        "error case {} must use expect_error comparison schema",
                        self.case_id
                    ));
                }
            }
            other => {
                return Err(format!(
                    "case {} has unsupported expected_behavior {}",
                    self.case_id, other
                ));
            }
        }

        for probe in &self.probes {
            probe.validate()?;
        }

        Ok(self)
    }
}

pub fn parse_case_record_value(value: Value) -> Result<CaseRecord, String> {
    let record: CaseRecord =
        serde_json::from_value(value).map_err(|err| format!("failed to parse case record: {err}"))?;
    record.validate()
}

fn parse_case_record_str(raw: &str) -> Result<CaseRecord, String> {
    let record: CaseRecord =
        serde_json::from_str(raw).map_err(|err| format!("failed to parse case record: {err}"))?;
    record.validate()
}

pub fn default_oracle_db_root() -> Option<PathBuf> {
    if let Some(root) = env::var_os("TENSOR_AD_ORACLES_ROOT") {
        let path = PathBuf::from(root);
        if path.is_dir() {
            return Some(path);
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vendored = manifest_dir.parent()?.join("third_party/tensor-ad-oracles");
    vendored.is_dir().then_some(vendored)
}

pub fn case_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let cases_root = root.join("cases");
    let mut files = Vec::new();
    let op_dirs = fs::read_dir(&cases_root)
        .map_err(|err| format!("failed to read {}: {err}", cases_root.display()))?;
    for op_dir in op_dirs {
        let op_dir = op_dir
            .map_err(|err| format!("failed to read entry in {}: {err}", cases_root.display()))?;
        let op_path = op_dir.path();
        if !op_path.is_dir() {
            continue;
        }
        let family_entries = fs::read_dir(&op_path)
            .map_err(|err| format!("failed to read {}: {err}", op_path.display()))?;
        for family_entry in family_entries {
            let family_entry = family_entry
                .map_err(|err| format!("failed to read entry in {}: {err}", op_path.display()))?;
            let family_path = family_entry.path();
            if family_path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(family_path);
            }
        }
    }
    files.sort();
    Ok(files)
}

#[allow(dead_code)]
pub fn load_case_records(path: &Path) -> Result<Vec<CaseRecord>, String> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            parse_case_record_str(line)
                .map_err(|err| format!("failed to parse {}: {err}", path.display()))
        })
        .collect()
}
