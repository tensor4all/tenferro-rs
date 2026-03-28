use std::fs;
use std::path::PathBuf;

fn repo_path(path: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tenferro manifest should have a parent repo root")
        .join(path)
}

fn collect_test_files(root: &std::path::Path, out: &mut Vec<PathBuf>) {
    if !root.exists() {
        return;
    }
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let metadata = entry.metadata().unwrap();
        if metadata.is_dir() {
            collect_test_files(&path, out);
            continue;
        }
        if path
            .components()
            .any(|component| component.as_os_str() == "tests")
        {
            out.push(path);
        }
    }
}

fn collect_rust_files(root: &std::path::Path, out: &mut Vec<PathBuf>) {
    if !root.exists() {
        return;
    }
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let metadata = entry.metadata().unwrap();
        if metadata.is_dir() {
            collect_rust_files(&path, out);
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
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

fn manifest_package_name(manifest: &str) -> Option<String> {
    let mut in_package = false;

    for line in manifest.lines() {
        let trimmed = line.trim();
        if !in_package {
            in_package = trimmed == "[package]";
            continue;
        }

        if trimmed.starts_with('[') {
            break;
        }
        if trimmed.starts_with("name = ") {
            return Some(
                trimmed
                    .trim_start_matches("name = ")
                    .trim()
                    .trim_matches('"')
                    .to_string(),
            );
        }
    }

    None
}

fn manifest_lib_test_disabled(manifest: &str) -> bool {
    let mut in_lib = false;

    for line in manifest.lines() {
        let trimmed = line.trim();
        if !in_lib {
            in_lib = trimmed == "[lib]";
            continue;
        }

        if trimmed.starts_with('[') {
            break;
        }
        if trimmed.starts_with("test = ") {
            return trimmed
                .trim_start_matches("test = ")
                .trim()
                .trim_matches('"')
                == "false";
        }
    }

    false
}

const DYNAMIC_SPLIT_MEMBERS: &[&str] = &[
    "tenferro-dynamic-compute",
    "internal/tenferro-internal-frontend-core",
    "internal/tenferro-internal-ad-core",
    "internal/tenferro-internal-ad-ops",
    "internal/tenferro-internal-ad-linalg",
    "internal/tenferro-internal-ad-surface",
];

const DYNAMIC_SPLIT_PUBLIC_CRATES: &[&str] = &[
    "tenferro-tensor",
    "tenferro-tensor-compute",
    "tenferro-dynamic-compute",
    "tenferro",
];

#[test]
fn dynamic_split_members_are_declared_in_workspace() {
    let cargo = fs::read_to_string(repo_path("Cargo.toml")).unwrap();
    let workspace_members = manifest_array_entries(&cargo, "members");

    for member in DYNAMIC_SPLIT_MEMBERS {
        assert!(
            workspace_members.iter().any(|entry| entry == member),
            "workspace members should include {member}"
        );

        let manifest_path = repo_path(&format!("{member}/Cargo.toml"));
        let manifest = fs::read_to_string(&manifest_path)
            .unwrap_or_else(|error| panic!("expected manifest for {member}: {error}"));
        let package_name =
            manifest_package_name(&manifest).expect("member manifest should define package.name");
        let basename = manifest_path
            .parent()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .expect("member path should have a UTF-8 directory basename");

        assert_eq!(
            basename, package_name,
            "directory basename should match package.name for {member}"
        );

        if member.starts_with("internal/") {
            assert!(
                manifest.contains("publish = false"),
                "internal dynamic split members must set publish = false: {member}"
            );
        }
    }
}

#[test]
fn public_crate_docs_list_dynamic_compute_entrypoint() {
    let readme = fs::read_to_string(repo_path("README.md")).unwrap();
    let api_index = fs::read_to_string(repo_path("docs/api_index.md")).unwrap();

    for crate_name in DYNAMIC_SPLIT_PUBLIC_CRATES {
        assert!(
            readme.contains(crate_name),
            "README should mention public crate {crate_name}"
        );
        assert!(
            api_index.contains(crate_name),
            "docs/api_index.md should mention public crate {crate_name}"
        );
    }
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

#[test]
fn extension_and_internal_directory_basenames_match_package_names() {
    let cargo = fs::read_to_string(repo_path("Cargo.toml")).unwrap();
    let workspace_members = manifest_array_entries(&cargo, "members");

    for member in workspace_members {
        if !(member.starts_with("extension/") || member.starts_with("internal/")) {
            continue;
        }

        let manifest_path = repo_path(&format!("{member}/Cargo.toml"));
        let manifest = fs::read_to_string(&manifest_path).unwrap();
        let package_name =
            manifest_package_name(&manifest).expect("member manifest should define package.name");
        let basename = manifest_path
            .parent()
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .expect("member path should have a UTF-8 directory basename");

        assert_eq!(
            basename, package_name,
            "directory basename should match package.name for {member}"
        );
    }
}

#[test]
fn current_docs_do_not_use_pre_rename_extension_crate_names() {
    let files = [
        "README.md",
        "docs/design/capi.md",
        "docs/design/einsum.md",
        "docs/design/gpu-backend-design.md",
        "docs/design/reference/einsum-algorithm-comparison.md",
        "docs/design/testing.md",
        "tenferro-algebra/src/lib.rs",
        "docs/design/integrations/burn.md",
        "extension/tenferro-ext-mdarray/src/lib.rs",
        "extension/tenferro-ext-ndarray/src/lib.rs",
        "extension/tenferro-ext-tropical-capi/src/lib.rs",
    ];
    let forbidden = [
        "tenferro-tropical",
        "tenferro_tropical",
        "tenferro-burn",
        "tenferro_burn",
        "tenferro-mdarray",
        "tenferro_mdarray",
        "tenferro-ndarray",
        "tenferro_ndarray",
        "extension/tenferro-tropical",
        "extension/tenferro-tropical-capi",
        "extension/tenferro-burn",
        "extension/tenferro-mdarray",
        "extension/tenferro-ndarray",
    ];

    for relative in files {
        let contents = fs::read_to_string(repo_path(relative)).unwrap();
        for needle in forbidden {
            assert!(
                !contents.contains(needle),
                "{relative} should not reference the pre-rename extension name {needle}"
            );
        }
    }
}

#[test]
fn tenferro_unit_tests_move_into_internal_owners() {
    for relative in ["tenferro/src/core/value/tests", "tenferro/src/tape/tests"] {
        assert!(
            !repo_path(relative).exists(),
            "tenferro should not keep unit-test trees for migrated AD core modules: {relative}"
        );
    }

    for relative in [
        "internal/tenferro-internal-ad-core/src/tests/core_value.rs",
        "internal/tenferro-internal-ad-core/src/tests/core_value_organization.rs",
        "internal/tenferro-internal-ad-core/src/tests/core_value_reverse_api.rs",
        "internal/tenferro-internal-ad-core/src/tests/tape_frontend.rs",
        "internal/tenferro-internal-ad-core/src/tests/tape_organization.rs",
    ] {
        assert!(
            repo_path(relative).exists(),
            "internal AD core should own migrated unit tests: {relative}"
        );
    }
}

#[test]
fn tenferro_has_no_src_unit_test_tree_after_split() {
    let cargo = fs::read_to_string(repo_path("tenferro/Cargo.toml")).unwrap();
    assert!(
        cargo.contains("[lib]") && cargo.contains("test = false"),
        "tenferro should disable lib tests once src/**/tests have moved out"
    );

    let mut leftover_paths = Vec::new();
    for root in ["tenferro/src/ops", "tenferro/src/structured"] {
        collect_test_files(&repo_path(root), &mut leftover_paths);
    }
    let leftovers: Vec<String> = leftover_paths
        .into_iter()
        .map(|path| {
            path.strip_prefix(repo_path(""))
                .unwrap()
                .display()
                .to_string()
        })
        .collect();
    assert!(
        leftovers.is_empty(),
        "tenferro should not keep src unit test files after the split: {leftovers:?}"
    );
}

#[test]
fn crates_with_lib_test_disabled_do_not_keep_src_unit_tests() {
    let cargo = fs::read_to_string(repo_path("Cargo.toml")).unwrap();
    let workspace_members = manifest_array_entries(&cargo, "members");

    for member in workspace_members {
        let manifest_path = repo_path(&format!("{member}/Cargo.toml"));
        let manifest = fs::read_to_string(&manifest_path).unwrap();
        if !manifest_lib_test_disabled(&manifest) {
            continue;
        }

        let src_root = repo_path(&format!("{member}/src"));
        let mut leftover_test_paths = Vec::new();
        collect_test_files(&src_root, &mut leftover_test_paths);
        let leftovers: Vec<String> = leftover_test_paths
            .into_iter()
            .map(|path| {
                path.strip_prefix(repo_path(""))
                    .unwrap()
                    .display()
                    .to_string()
            })
            .collect();
        assert!(
            leftovers.is_empty(),
            "{member} sets [lib] test = false, so it must not keep src/**/tests: {leftovers:?}"
        );

        let mut rust_files = Vec::new();
        collect_rust_files(&src_root, &mut rust_files);
        let inline_test_modules: Vec<String> = rust_files
            .into_iter()
            .filter_map(|path| {
                let contents = fs::read_to_string(&path).unwrap();
                if contents.contains("mod tests;") || contents.contains("mod tests {") {
                    Some(
                        path.strip_prefix(repo_path(""))
                            .unwrap()
                            .display()
                            .to_string(),
                    )
                } else {
                    None
                }
            })
            .collect();
        assert!(
            inline_test_modules.is_empty(),
            "{member} sets [lib] test = false, so it must not keep inline unit-test modules: {inline_test_modules:?}"
        );
    }
}

#[test]
fn ci_runs_doctest_inside_workspace_job_and_keeps_docs_site_for_pr_checks() {
    let ci = fs::read_to_string(repo_path(".github/workflows/ci.yml")).unwrap();
    let docs_workflow = fs::read_to_string(repo_path(".github/workflows/docs.yml")).unwrap();
    let repo_settings = fs::read_to_string(repo_path("ai/repo-settings.local.json")).unwrap();

    assert!(
        ci.contains("name: cargo test (workspace)"),
        "CI must keep the workspace test job"
    );
    assert!(
        ci.contains("cargo nextest run --workspace --release --no-fail-fast"),
        "workspace test job must keep nextest coverage for unit and integration tests"
    );
    assert!(
        ci.contains("cargo test --doc --workspace --release"),
        "workspace test job must execute workspace doctests"
    );
    assert!(
        !ci.contains("name: doctest"),
        "CI should not keep a separate doctest job once doctests return to the workspace test job"
    );
    assert!(
        ci.contains("name: docs-site"),
        "CI must restore the docs-site job in PR CI"
    );
    assert!(
        ci.contains("bash scripts/build_docs_site.sh"),
        "docs-site job must build the full docs site"
    );
    assert!(
        !ci.contains("name: docs-rust"),
        "PR CI should not keep the temporary docs-rust replacement once docs-site is restored"
    );

    assert!(
        docs_workflow.contains("Build docs-site artifact"),
        "docs deploy workflow must keep the docs-site build"
    );
    assert!(
        docs_workflow.contains("push:\n    branches: [main]"),
        "docs deploy workflow must stay push-to-main based"
    );

    assert!(
        !repo_settings.contains("\"doctest\""),
        "required PR checks must not keep a separate doctest context once doctests return to the workspace test job"
    );
    assert!(
        repo_settings.contains("\"docs-site\""),
        "required PR checks must include docs-site"
    );
    assert!(
        !repo_settings.contains("\"docs-rust\""),
        "required PR checks must not keep docs-rust once docs-site is restored"
    );
}
