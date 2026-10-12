# Pins for the 0.11 migration: what the pin-content policy allows

Status: the migration branch stages cubecl/cubek from local working trees; a
*git* pin at the ported revisions is not permitted by the repository's policy
until a stable release contains them.

## The policy

`scripts/release-publish.py` (`parse_workspace_git_dependencies`) requires every
workspace git dependency to declare **package identity, an exact registry
version, git, and a 40-digit rev**:

```toml
cubecl = { package = "t4a-cubecl", git = "https://github.com/tensor4all/cubecl",
           rev = "<40 hex>", version = "=0.10.2" }
```

`scripts/check-git-pin-content.py` then compares the pinned revision against the
**crate archive of the version Cargo would resolve** and fails on any content
difference (that is what reports `t4a-cubecl-{cuda,runtime,wgpu} 0.10.2: registry
release content differs from the pinned revision` on `main` today).

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

## What the branch does instead

`Cargo.toml` on `migration/cubecl-0.11-upstream` resolves cubecl and cubek from
three sibling working trees, checked out at the pushed revisions:

| Path | Repository and revision |
| --- | --- |
| `../cubecl-ports-combined` | `tensor4all/cubecl` `feat/tenferro-upstream-0.11-fixes` `9f097e00e` |
| `../cubek-upstream-tenferro` | `tensor4all/cubek` `feat/tenferro-upstream-kernels` `24435486` |
| `../cubek-upstream-fft` | `tensor4all/cubek` `feat/tenferro-upstream-interleaved-fft` `a25de257` |

`python3 scripts/check-git-pin-content.py` on that tree: **no content drift**
(4 warnings, all unrelated to cubecl/cubek: two allowlisted `cpueinsum` entries
and a non-exact `computegraph` pin). The `[patch."https://github.com/tracel-ai/cubecl"]`
entries exist because the CubeK revisions still name upstream cubecl; they
disappear once the CubeK ports carry the fork's cubecl pin, or once cubecl's
ports land upstream.

## Downstream order (unchanged)

tenferro-rs first, then cubek's own cubecl pin, then Tensor4all.jl and
tenferro-benchmark — the benchmark report only against a merged revision. Each
of those is a git pin in another repository and follows the same release rule.
