use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use tenferro_core_ops::{all_primitive_descriptors, OpCategory};

use crate::ElementwiseFusionOp;

#[test]
fn tensor_view_public_surface_uses_canonical_names() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_dir
        .parent()
        .expect("tenferro-tensor should live directly under the workspace root");
    let current_files = collect_current_text_files(repo_root);
    let types_rs = read_repo_file(repo_root, "tenferro-tensor/src/types.rs");

    assert!(
        types_rs.contains("pub enum TensorView<'a>"),
        "dtype-erased read-only views should use TensorView"
    );
    assert!(
        types_rs.contains("pub enum TensorRead<'a>"),
        "dtype-erased tensor inputs should use TensorRead"
    );
    assert!(
        types_rs.contains("pub fn transpose_view"),
        "metadata-only axis permutations should use transpose_view"
    );

    for obsolete in [
        "TypedStridedTensorView",
        "pub enum StridedTensorView",
        "pub enum StridedTensorViewMut",
        "pub use strided_view::{StridedSliceSpec, StridedTensorView",
        "TypedTensorView::new",
        "permute_view",
        "try_permute_axes",
        "tenferro_tensor_core::TypedTensor",
        "tenferro_tensor_core::TypedTensorView",
        "Tensor::permute()",
        "TensorPrims::permute",
    ] {
        let offenders = files_containing(&current_files, obsolete);
        assert!(
            offenders.is_empty(),
            "current source/docs should not expose obsolete `{obsolete}` outside docs/plans: {offenders:?}"
        );
    }
}

fn read_repo_file(repo_root: &Path, relative_path: &str) -> String {
    fs::read_to_string(repo_root.join(relative_path))
        .unwrap_or_else(|err| panic!("failed to read {relative_path}: {err}"))
}

fn collect_current_text_files(repo_root: &Path) -> Vec<(PathBuf, String)> {
    let mut files = Vec::new();
    collect_current_text_files_inner(repo_root, repo_root, &mut files);
    files
}

fn collect_current_text_files_inner(
    repo_root: &Path,
    dir: &Path,
    files: &mut Vec<(PathBuf, String)>,
) {
    for entry in fs::read_dir(dir).unwrap_or_else(|err| panic!("failed to read {dir:?}: {err}")) {
        let entry = entry.unwrap_or_else(|err| panic!("failed to read directory entry: {err}"));
        let path = entry.path();
        let relative_path = path
            .strip_prefix(repo_root)
            .expect("walked path should stay under repo root");

        if should_skip_path(relative_path) {
            continue;
        }

        if path.is_dir() {
            collect_current_text_files_inner(repo_root, &path, files);
        } else if is_current_text_file(relative_path) {
            let contents = fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("failed to read {relative_path:?}: {err}"));
            files.push((relative_path.to_path_buf(), contents));
        }
    }
}

fn should_skip_path(relative_path: &Path) -> bool {
    relative_path.starts_with("docs/plans")
        || relative_path.starts_with("target")
        || relative_path.starts_with(".git")
        || relative_path == Path::new("tenferro-tensor/src/tests/op_vocabulary_contract_tests.rs")
}

fn is_current_text_file(relative_path: &Path) -> bool {
    matches!(
        relative_path
            .extension()
            .and_then(|extension| extension.to_str()),
        Some("rs" | "md")
    )
}

fn files_containing(files: &[(PathBuf, String)], needle: &str) -> Vec<PathBuf> {
    files
        .iter()
        .filter(|(_, contents)| contents.contains(needle))
        .map(|(path, _)| path.clone())
        .collect()
}

#[test]
fn elementwise_fusion_ops_round_trip_through_catalog_kinds() {
    let fusion_kinds: HashSet<_> = ElementwiseFusionOp::iter()
        .map(|op| op.primitive_kind())
        .collect();

    for kind in &fusion_kinds {
        let descriptor = all_primitive_descriptors()
            .iter()
            .find(|descriptor| descriptor.kind == *kind)
            .unwrap_or_else(|| panic!("missing descriptor for fusion op {kind:?}"));
        assert!(
            matches!(
                descriptor.category,
                OpCategory::Elementwise | OpCategory::Analytic
            ),
            "fusion op {kind:?} should be cataloged as elementwise or analytic"
        );
        assert!(
            !descriptor.host_only,
            "fusion op {kind:?} should not be host-only"
        );
        assert_eq!(
            ElementwiseFusionOp::from_primitive_kind(*kind).map(|op| op.primitive_kind()),
            Some(*kind),
            "fusion op {kind:?} should round-trip through PrimitiveOpKind"
        );
    }

    assert_eq!(
        fusion_kinds.len(),
        ElementwiseFusionOp::iter().count(),
        "ElementwiseFusionOp::iter should list each variant exactly once"
    );
}
