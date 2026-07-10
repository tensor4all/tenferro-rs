use std::path::{Path, PathBuf};

const DEFAULT_TBLIS_VER: &str = "eb719e718976572e0ab53975f4e0c799faeb35f2";

fn build_tblis() {
    if !cfg!(feature = "build_from_source") {
        return;
    }

    let tblis_src = std::env::var("TBLIS_SRC")
        .unwrap_or_else(|_| "https://github.com/MatthewsResearchGroup/tblis.git".to_string());
    let tblis_ver = std::env::var("TBLIS_VER").unwrap_or_else(|_| DEFAULT_TBLIS_VER.to_string());

    println!("cargo:rerun-if-env-changed=TBLIS_SRC");
    println!("cargo:rerun-if-env-changed=TBLIS_VER");

    let mut cfg = cmake::Config::new("cmake");
    cfg.define("TBLIS_SRC", &tblis_src)
        .define("TBLIS_VER", tblis_ver)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("CMAKE_DISABLE_FIND_PACKAGE_BLIS", "ON")
        .define("CMAKE_FIND_USE_PACKAGE_REGISTRY", "FALSE")
        .define("CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY", "FALSE");

    if Path::new(&tblis_src).is_dir() {
        cfg.define("TBLIS_SRC_IS_LOCAL_DIR", "ON");
    } else {
        cfg.define("TBLIS_FORCE_BUNDLED_BLIS", "ON");
    }

    let dst = cfg.build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-search=native={}/lib/tblis", dst.display());
}

fn generate_link_search_paths(paths: &str) -> Vec<String> {
    let split_char = if cfg!(windows) { ";" } else { ":" };
    paths.split(split_char).map(str::to_string).collect()
}

fn root_candidates(env_candidates: &[&str]) -> Vec<PathBuf> {
    let root_candidates = ["/usr", "/usr/local", "/usr/local/share", "/opt"];

    env_candidates
        .iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .flat_map(|path| generate_link_search_paths(&path))
        .filter(|path| !path.is_empty())
        .chain(root_candidates.into_iter().map(str::to_string))
        .map(PathBuf::from)
        .collect()
}

fn lib_candidates() -> impl Iterator<Item = PathBuf> {
    [
        "",
        "lib",
        "lib/stubs",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ]
    .into_iter()
    .map(PathBuf::from)
}

fn path_candidates(env_candidates: &[&str]) -> impl Iterator<Item = PathBuf> {
    root_candidates(env_candidates)
        .into_iter()
        .flat_map(|root| lib_candidates().map(move |lib| root.join(lib)))
        .filter(|path| path.exists())
        .filter_map(|path| std::fs::canonicalize(path).ok())
}

fn link_tblis() {
    if !cfg!(feature = "build_from_source") {
        return;
    }

    let env_candidates = [
        "TBLIS_DIR",
        "REST_EXT_DIR",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "PATH",
    ];

    println!("cargo:rerun-if-env-changed=TBLIS_DIR");
    for path in path_candidates(&env_candidates) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    if cfg!(feature = "static") {
        println!("cargo:rustc-link-lib=static=tblis");
        println!("cargo:rustc-link-lib=static=tci");
        println!("cargo:rustc-link-lib=static=blis_tblis");
        println!("cargo:rustc-link-lib=static=blis_core");
    } else {
        println!("cargo:rustc-link-lib=tblis");
    }
}

fn main() {
    build_tblis();
    link_tblis();
}
