#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use crate::decode::CaseRecord;

pub fn oracle_cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../third_party/tensor-ad-oracles/cases")
}

pub fn discover_case_files(root: &Path) -> Vec<(String, PathBuf)> {
    match try_discover_case_files(root) {
        Ok(files) => files,
        Err(_) => Vec::new(),
    }
}

pub fn try_discover_case_files(root: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut files = Vec::new();
    let entries = fs::read_dir(root)
        .map_err(|err| format!("failed to read oracle case dir {}: {err}", root.display()))?;

    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read case dir entry: {err}"))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let op_name = entry.file_name().to_string_lossy().into_owned();
        let op_entries = fs::read_dir(&path)
            .map_err(|err| format!("failed to read op dir {}: {err}", path.display()))?;
        for op_entry in op_entries {
            let op_entry =
                op_entry.map_err(|err| format!("failed to read op file entry: {err}"))?;
            let file_path = op_entry.path();
            if file_path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                files.push((op_name.clone(), file_path));
            }
        }
    }

    files.sort_unstable_by(|lhs, rhs| lhs.cmp(rhs));
    Ok(files)
}

pub fn load_cases(path: &Path) -> Vec<CaseRecord> {
    match try_load_cases(path) {
        Ok(cases) => cases,
        Err(_) => Vec::new(),
    }
}

pub fn try_load_cases(path: &Path) -> Result<Vec<CaseRecord>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|err| format!("failed to read oracle file {}: {err}", path.display()))?;
    let mut cases = Vec::new();

    for (line_no, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let sanitized = sanitize_non_finite_tokens(line);
        let case = serde_json::from_str::<CaseRecord>(&sanitized).map_err(|err| {
            format!(
                "failed to parse {} line {}: {err}",
                path.display(),
                line_no + 1
            )
        })?;
        cases.push(case);
    }

    Ok(cases)
}

fn sanitize_non_finite_tokens(line: &str) -> String {
    let mut sanitized = String::with_capacity(line.len());
    let mut in_string = false;
    let mut escaped = false;
    let bytes = line.as_bytes();
    let mut index = 0usize;

    while index < bytes.len() {
        let ch = bytes[index] as char;
        if in_string {
            sanitized.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            index += 1;
            continue;
        }

        if ch == '"' {
            in_string = true;
            sanitized.push(ch);
            index += 1;
            continue;
        }

        let tail = &line[index..];
        if tail.starts_with("-Infinity") {
            sanitized.push_str("-1.7976931348623157e+308");
            index += "-Infinity".len();
            continue;
        }
        if tail.starts_with("Infinity") {
            sanitized.push_str("1.7976931348623157e+308");
            index += "Infinity".len();
            continue;
        }
        if tail.starts_with("NaN") {
            sanitized.push_str("0.0");
            index += "NaN".len();
            continue;
        }

        sanitized.push(ch);
        index += 1;
    }

    sanitized
}
