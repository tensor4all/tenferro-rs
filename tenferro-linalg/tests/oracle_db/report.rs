use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::db::{case_files, load_case_records};
use crate::support::{classify_record, RecordSupport};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FamilyKey {
    op: String,
    family: String,
    observable: String,
}

#[derive(Clone, Debug, Default)]
struct Totals {
    total_records: usize,
    supported_records: usize,
    supported_hvp_records: usize,
    expected_error_records: usize,
    unsupported_records: usize,
}

pub fn checked_in_report_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("docs/generated/tensor-ad-oracles-support.md")
}

pub fn generate_support_report(root: &Path) -> Result<String, String> {
    let files = case_files(root)?;
    let mut totals = Totals::default();
    let mut supported = BTreeMap::<FamilyKey, usize>::new();
    let mut expected_error = BTreeMap::<FamilyKey, usize>::new();
    let mut unsupported = BTreeMap::<(FamilyKey, &'static str), usize>::new();

    for path in files {
        let records = load_case_records(&path)?;
        for record in records {
            totals.total_records += 1;
            let key = FamilyKey {
                op: record.op.clone(),
                family: record.family.clone(),
                observable: record.observable.kind.clone(),
            };
            match classify_record(&record) {
                RecordSupport::Supported(_) => {
                    totals.supported_records += 1;
                    if record
                        .probes
                        .iter()
                        .any(|probe| probe.pytorch_ref.hvp.is_some())
                    {
                        totals.supported_hvp_records += 1;
                    }
                    *supported.entry(key).or_default() += 1;
                }
                RecordSupport::ExpectedError(_) => {
                    totals.expected_error_records += 1;
                    *expected_error.entry(key).or_default() += 1;
                }
                RecordSupport::Unsupported { reason } => {
                    totals.unsupported_records += 1;
                    *unsupported.entry((key, reason)).or_default() += 1;
                }
                RecordSupport::Unknown => {
                    return Err(format!(
                        "unclassified oracle family {}/{}/{}",
                        key.op, key.family, key.observable
                    ));
                }
            }
        }
    }

    let mut out = String::new();
    out.push_str("# Tensor AD Oracles Support Coverage\n\n");
    out.push_str(
        "This file is generated from the vendored `third_party/tensor-ad-oracles` subtree and the local oracle replay support registry.\n\n",
    );
    out.push_str("## Summary\n\n");
    out.push_str(&format!(
        "- Total published records: {}\n",
        totals.total_records
    ));
    out.push_str(&format!(
        "- Supported success records: {}\n",
        totals.supported_records
    ));
    out.push_str(&format!(
        "- Supported success records with HVP payloads: {}\n",
        totals.supported_hvp_records
    ));
    out.push_str(&format!(
        "- Expected error records: {}\n",
        totals.expected_error_records
    ));
    out.push_str(&format!(
        "- Unsupported success records: {}\n\n",
        totals.unsupported_records
    ));

    out.push_str("## Supported\n\n");
    out.push_str("| op | family | observable | sample count |\n");
    out.push_str("| --- | --- | --- | ---: |\n");
    for (key, count) in supported {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            key.op, key.family, key.observable, count
        ));
    }
    out.push('\n');

    out.push_str("## Expected Errors\n\n");
    out.push_str("| op | family | observable | sample count |\n");
    out.push_str("| --- | --- | --- | ---: |\n");
    for (key, count) in expected_error {
        out.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            key.op, key.family, key.observable, count
        ));
    }
    out.push('\n');

    out.push_str("## Unsupported\n\n");
    out.push_str("| op | family | observable | sample count | reason |\n");
    out.push_str("| --- | --- | --- | ---: | --- |\n");
    for ((key, reason), count) in unsupported {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            key.op, key.family, key.observable, count, reason
        ));
    }

    Ok(out)
}
