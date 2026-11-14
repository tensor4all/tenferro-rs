# Pins for the CubeCL 0.11 migration: what the pin-content policy allows (superseded)

> **Superseded: tenferro no longer depends on CubeCL or CubeK.** The CUDA backend
> moved to the standalone `t4a-cuda-runtime`
> ([tensor4all/t4a-gpu-backend](https://github.com/tensor4all/t4a-gpu-backend)) and
> the `t4a-cuda-kernels` NVRTC catalogue on top of `cudarc` (#2208), and the
> CubeCL-era artifacts were retired in #2312. The workspace has no `cubecl` or
> `cubek` dependency, so the "Why the ported 0.11 revisions cannot be pinned
> yet", "Resolution" and "Downstream order" sections below are history of the
> CubeCL line only.
>
> The pin policy itself is still current: the `t4a-cuda-runtime` and
> `t4a-cuda-kernels` entries in `[workspace.dependencies]` follow it (package
> identity, exact registry version, git URL and 40-digit rev, checked by
> `scripts/check-git-pin-content.py`).

## The policy

`scripts/release-publish.py` (`parse_workspace_git_dependencies`) requires every
workspace git dependency to declare **package identity, an exact registry
version, git, and a 40-digit rev**:

```toml
t4a-cuda-runtime = { git = "https://github.com/tensor4all/t4a-gpu-backend",
                     rev = "<40 hex>", version = "=0.1.1" }
```

`scripts/check-git-pin-content.py` then compares the pinned revision against the
**crate archive of the version Cargo would resolve** and fails on any content
difference (that is what reported `t4a-cubecl-{cuda,runtime,wgpu} 0.10.2:
registry release content differs from the pinned revision` on `main` before the
0.11 releases landed).

## Why the ported 0.11 revisions cannot be pinned yet

The ported content is `0.11.0-pre.4` upstream plus the ported commits, and
`effective_version` (`check-git-pin-content.py:157-173`) **skips every
pre-release candidate**, so a requirement like `=0.11.0-pre.4` resolves to no
registry release at all. That lands in the "no crates.io release" branch, which
is meant for crates such as `cpueinsum` that are allowlisted in
`scripts/git-pin-content-exceptions.toml`. Adding an exception for cubecl/cubek
is exactly what the migration's rules forbid ("resolved, not excepted").

So a git pin can only be written once a **stable release contains the ports**:

- a fork release publishing the ported content under the fork's crate names
  (`t4a-cubecl` / `t4a-cubek` at a stable version), or
- the ports merged upstream and pinned at a stable upstream release.

## Resolution

The first option landed: `tensor4all/cubecl` `release/t4a-cubecl-0.11.0` is
published as `t4a-cubecl*` 0.11.0 and `tensor4all/cubek`
`release/t4a-cubek-0.3.0` as `t4a-cubek*` 0.3.0, so the workspace depends on
those versions instead of a revision. That also removes the `[patch.crates-io]`
the interim git pin needed: the CubeK release already names
`t4a-cubecl = "=0.11.0"` from the registry, so both sides resolve one CubeCL
copy (`cargo tree -d` reports no duplicate of any `t4a-cubecl*` crate).

The second option is the remaining fork-exit step: replace the `t4a-` names with
merged upstream revisions once the ports land upstream (#2116).

## Downstream order (unchanged)

tenferro-rs first, then cubek's own cubecl pin, then Tensor4all.jl and
tenferro-benchmark — the benchmark report only against a merged revision. Each
of those is a git pin in another repository and follows the same release rule.
