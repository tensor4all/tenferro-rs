# #1854: git-pin content verification before publication

## Decisions

- Verify pinned-dependency **content**, not just version existence. `cargo
  publish` strips the `git` source and keeps only `version`, so a registry
  package can exist at the required version with different contents or
  different dependency wiring. The v0.6.0 release failed ten crates into the
  order because crates.io `strided-kernel 0.4.0` was the pre-refactor crate
  while the pin held the post-refactor facade; `strided-perm 0.4.0` drifted
  silently in `src/hptt/plan.rs`.
- Compare only files Cargo always packages (`src/**`, `build.rs`) plus normal
  and build dependency wiring with `[workspace.dependencies]` inheritance
  resolved. Re-implementing `include`/`exclude` selection and normalizing
  cargo-generated members would duplicate `cargo package` semantics; the narrow
  comparison still catches every mismatch class seen so far, including the
  `cubek-fft` case where the registry package is a different project.
- Resolve against the version Cargo would actually pick (highest matching
  release of a caret range) and warn when that differs from the version the
  pinned revision declares. The requirement string itself is not made exact,
  because publishing a newer compatible release from the fork is the recovery
  path that unblocked v0.6.0.
- Rely on an explicit exception file with an issue and a reason rather than a
  silent allowlist. The check still prints an excepted mismatch as a warning on
  every run and warns when an entry becomes stale.
- Change the release boundary in the same PR: publication is executed by the
  agent under step-by-step maintainer approval instead of the maintainer
  copy-pasting a guarded script. The enforced boundary is the fail-closed
  preflight, unchanged.

## Verification conclusions and constraints

- On current `main` the check reports one content mismatch: `cubek-fft` 0.2.0,
  where the crates.io name belongs to upstream and the fork builds the crate
  with `publish = false`. It is excepted and tracked in #1855; the remaining
  warnings are caret-range notices for `strided-*` and `computegraph`.
- `strided-perm 0.4.1` and `strided-kernel 0.4.1`, published to unblock
  v0.6.0, match their pinned revision, so the check would have passed on the
  fixed state and failed on the pre-fix state.
- The check was exercised against the real registry for all 15 pins and is
  wired into both the `ci-config` CI lane and the publication preflight, so a
  drift introduced by a pin update is caught on that PR rather than at release.
- Not verified here: whether a mismatch introduced by a *newer* fork release
  than the pin (rather than a divergent registry package) should force a pin
  update. The check reports it; the disposition stays a release-time decision.
