# t4a-tblis-src

`t4a-tblis-src` is the Tensor4All-maintained source-build and native-link
provider for [TBLIS](https://github.com/MatthewsResearchGroup/tblis). It is a
small, independently versioned package stored in the tenferro-rs repository,
but intentionally excluded from the tenferro Cargo workspace.

The package does not vendor TBLIS. With `build_from_source`, its build script
fetches the pinned TBLIS revision recorded in `build.rs`, including TBLIS's
submodules, and builds it through CMake. The optional `static` feature emits
static link directives for TBLIS and its bundled native libraries.

Use a neutral dependency alias so the Rust crate name remains `tblis_src`:

```toml
[dependencies]
tblis-src = { package = "t4a-tblis-src", version = "0.1.0", features = [
    "build_from_source",
    "static",
] }
```

Keep the source provider linked alongside the FFI crate:

```rust
extern crate tblis_src as _;
```

`TBLIS_SRC` may override the source URL or point to a local checkout with its
submodules initialized. `TBLIS_VER` may override the requested revision.
Native builds require Git, CMake, Make, and suitable C/C++ compilers.

The Rust build glue is licensed under Apache-2.0. It is derived from
RESTGroup/tblis-rs and carries forward that license and attribution; see
`NOTICE.md`. The downloaded native projects have their own BSD-3-Clause
licenses; see `THIRD_PARTY_LICENSES.md` before redistributing linked binaries.
