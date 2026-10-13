# Interim fork release line while the upstream PRs are open

## Interim model while the upstream PRs are open

The PRs are opened against the stable base (`v0.11.0` / `v0.3.0`) and stay open
through review; tenferro keeps consuming a **fork release** in the meantime.

- The fork release means "the upstream release it mirrors **plus the ports that
  are still unmerged**", published as `t4a-cubecl-*` / `t4a-cubek-*` with the
  version mirroring the upstream release it derives from (the 0.10.x precedent).
- Every fork release is rebased onto the then-latest upstream release and carries
  **only** the ports upstream has not taken yet, so the delta shrinks as PRs
  merge. A port that upstream adopts is dropped from the next fork release.
- The tenferro pin names the fork version and the exact release revision, so
  `check-git-pin-content.py` verifies content, and the pin's comment records which
  upstream release it mirrors (the fork's `0.11.0` is *not* upstream's `0.11.0`
  content, only the same number in the fork's own crate namespace).
- While the PRs wait, rebase them onto new upstream releases so they do not rot.
- **Exit**: once the last port is merged upstream and a release contains it,
  tenferro pins the upstream revision, this fork release line stops, and no new
  `t4a-` versions are cut (the published ones stay for old pins).
