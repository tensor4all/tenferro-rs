use std::{fs, path::Path};

#[test]
fn cubecl_scatter_does_not_use_single_thread_launch_fallback() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("cubecl");
    let mod_source =
        fs::read_to_string(root.join("mod.rs")).expect("CubeCL module source should be readable");
    let scatter_start = mod_source
        .find("    fn scatter(")
        .expect("CubeCL backend should define scatter");
    let scatter_end = mod_source[scatter_start..]
        .find("    fn slice(")
        .map(|offset| scatter_start + offset)
        .expect("CubeCL backend should define slice after scatter");
    let scatter_source = &mod_source[scatter_start..scatter_end];

    let dispatch_source = fs::read_to_string(root.join("dispatch.rs"))
        .expect("CubeCL dispatch source should be readable");
    let sources = [
        ("cubecl/mod.rs scatter body", scatter_source),
        ("cubecl/dispatch.rs", dispatch_source.as_str()),
    ];
    let banned = ["single_thread_launch_config", "CubeCount::new_single()"];

    let mut violations = Vec::new();
    for (name, source) in sources {
        for needle in banned {
            if source.contains(needle) {
                violations.push(format!("{name} contains {needle}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "CubeCL scatter launch must not use a single-thread fallback:\n{}",
        violations.join("\n")
    );
}
