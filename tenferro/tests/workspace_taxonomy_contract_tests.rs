use std::fs;
use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tenferro manifest should have a parent repo root")
        .join(path)
}

fn manifest_array_entries(manifest: &str, key: &str) -> Vec<String> {
    let mut in_array = false;
    let mut values = Vec::new();

    for line in manifest.lines() {
        let trimmed = line.trim();
        if !in_array {
            if trimmed.starts_with(&format!("{key} = [")) {
                in_array = true;
            }
            continue;
        }

        if trimmed == "]" {
            break;
        }

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let value = trimmed.trim_end_matches(',').trim().trim_matches('"');
        if !value.is_empty() {
            values.push(value.to_string());
        }
    }

    values
}

#[test]
fn workspace_and_docs_use_the_taxonomy_vocabulary() {
    let cargo = fs::read_to_string(repo_path("Cargo.toml")).unwrap();
    let default_members = manifest_array_entries(&cargo, "default-members");
    let workspace_members = manifest_array_entries(&cargo, "members");

    assert!(
        !default_members.is_empty(),
        "workspace root should define default-members"
    );
    assert!(
        default_members
            .iter()
            .any(|entry| entry == "tenferro-tensor-compute"),
        "default-members should keep the typed compute facade in the default build set"
    );
    assert!(
        default_members
            .iter()
            .any(|entry| entry == "tenferro-tensor"),
        "default-members should keep the tensor data crate in the default build set"
    );

    let readme = fs::read_to_string(repo_path("README.md")).unwrap();
    assert!(
        readme.contains("end-user public") && readme.contains("protocol public"),
        "README should explain the public crate taxonomy"
    );
    assert!(
        readme.contains("### Public Crate Choices"),
        "README should keep public crate guidance in a dedicated section"
    );
    assert!(
        readme.contains("### Internal Crate Policy"),
        "README should keep internal crate policy in a dedicated section"
    );
    assert!(
        readme.contains("### Public Crate Choices") && readme.contains("### Internal Crate Policy"),
        "README should keep separate sections for public crate choices and internal crate policy"
    );
    assert!(
        readme.contains("tenferro-internal-"),
        "README should name the internal crate prefix"
    );
    assert!(
        readme.contains("tenferro_ext_tropical"),
        "README should use the Rust crate name derived from tenferro-ext-tropical"
    );
    assert!(
        !readme.contains("tenferro_tropical"),
        "README should not use the pre-rename tropical Rust crate path"
    );

    let architecture = fs::read_to_string(repo_path("docs/design/architecture.md")).unwrap();
    assert!(
        architecture.contains("public / internal crate taxonomy")
            || architecture.contains("crate taxonomy"),
        "architecture doc should introduce the taxonomy explicitly"
    );
    assert!(
        architecture.contains("tenferro-internal-"),
        "architecture doc should use the internal crate naming rule"
    );

    let api_index = fs::read_to_string(repo_path("docs/api_index.md")).unwrap();
    assert!(
        api_index.contains("end-user public") && api_index.contains("internal implementation"),
        "API index should classify crates by end-user/public/internal intent"
    );
    assert!(
        api_index.contains("tenferro-ext-burn")
            && api_index.contains("tenferro-ext-mdarray")
            && api_index.contains("tenferro-ext-ndarray")
            && api_index.contains("tenferro-ext-tropical")
            && api_index.contains("tenferro-ext-tropical-capi"),
        "API index should list every renamed extension crate in the taxonomy summary"
    );

    let burn_doc = fs::read_to_string(repo_path("docs/design/integrations/burn.md")).unwrap();
    assert!(
        burn_doc.contains("tenferro_ext_burn"),
        "Burn integration doc should use the Rust crate name derived from tenferro-ext-burn"
    );
    assert!(
        !burn_doc.contains("tenferro_burn"),
        "Burn integration doc should not use the pre-rename burn Rust crate path"
    );

    for member in workspace_members {
        let manifest_path = repo_path(&format!("{member}/Cargo.toml"));
        let manifest = fs::read_to_string(&manifest_path).unwrap();
        if member.starts_with("extension/") {
            let relative = manifest_path
                .strip_prefix(repo_path(""))
                .expect("manifest should live inside the repository");
            assert!(
                manifest.contains("name = \"tenferro-ext-"),
                "extension crate manifests must use the tenferro-ext- prefix: {}",
                relative.display()
            );
        }
        if !manifest.contains("name = \"tenferro-internal-") {
            continue;
        }

        let relative = manifest_path
            .strip_prefix(repo_path(""))
            .expect("manifest should live inside the repository");
        assert!(
            relative.starts_with("internal"),
            "internal crate manifests must live under internal/, found {}",
            relative.display()
        );
        assert!(
            manifest.contains("publish = false"),
            "internal crate manifests must set publish = false explicitly: {}",
            relative.display()
        );
    }
}
