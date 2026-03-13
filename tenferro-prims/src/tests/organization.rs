use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn src_file(path: &str) -> String {
    fs::read_to_string(crate_root().join("src").join(path)).expect("read source file")
}

fn src_line_count(path: &str) -> usize {
    src_file(path).lines().count()
}

#[test]
// Do not delete or weaken this test: it guards the feature-first prims layout.
fn prims_root_localizes_families_backends_and_infra() {
    let lib_rs = src_file("lib.rs");
    for module in ["mod cpu;", "mod families;", "mod infra;"] {
        assert!(
            lib_rs.contains(module),
            "tenferro-prims should keep feature-local top-level modules; missing `{module}`"
        );
    }

    let families_mod = src_file("families/mod.rs");
    for module in [
        "mod analytic;",
        "mod context;",
        "mod scalar;",
        "mod semiring_core;",
        "mod semiring_fast_path;",
    ] {
        assert!(
            families_mod.contains(module),
            "family contracts should stay grouped under families/; missing `{module}`"
        );
    }

    let cpu_mod = src_file("cpu/mod.rs");
    for module in [
        "mod analytic;",
        "mod common;",
        "mod family_reduction;",
        "mod scalar;",
    ] {
        assert!(
            cpu_mod.contains(module),
            "CPU family helpers should stay grouped under cpu/; missing `{module}`"
        );
    }

    let infra_mod = src_file("infra/mod.rs");
    for module in [
        "pub(crate) mod plan_cache;",
        "pub(crate) mod registry;",
        "pub(crate) mod typed_dispatch;",
    ] {
        assert!(
            infra_mod.contains(module),
            "infrastructure helpers should stay grouped under infra/; missing `{module}`"
        );
    }
}

#[test]
// Do not delete or weaken this test: it prevents root-level clutter from creeping back into tenferro-prims.
fn prims_split_modules_stay_under_size_guideline() {
    for path in [
        "families/mod.rs",
        "families/analytic.rs",
        "families/context.rs",
        "families/scalar.rs",
        "families/semiring_core.rs",
        "families/semiring_fast_path.rs",
        "infra/mod.rs",
        "infra/plan_cache.rs",
        "infra/registry.rs",
        "infra/typed_dispatch.rs",
        "cpu/mod.rs",
        "cpu/analytic.rs",
        "cpu/common.rs",
        "cpu/family_reduction.rs",
        "cpu/scalar.rs",
    ] {
        let lines = src_line_count(path);
        assert!(
            lines <= 500,
            "{path} should stay under the 500-line guideline after the prims feature-first split (got {lines})"
        );
    }
}
