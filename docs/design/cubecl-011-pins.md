# Pins for the 0.11 migration: what the pin-content policy allows

Status: resolved. The fork published the ported content as the 0.11.0 / 0.3.0
releases, so the workspace takes them from the **registry** directly
(`Cargo.toml`: `cubecl = { package = "t4a-cubecl", version = "=0.11.0", ... }`),
no workspace git pin remains, and `scripts/check-git-pin-content.py` reports
nothing. The policy below is what a future pin must follow; the constraint that
blocked pinning during the migration is history.

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
