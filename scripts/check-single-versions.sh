#!/usr/bin/env bash
# Every crate whose types cross a crate boundary in this graph must resolve to
# one version: two semver-compatible copies are two incompatible types that
# cannot meet (E0308). The manifests use compatible ranges rather than exact
# pins so a dependency's patch fix can be locked (R-git-workflow-58); this check
# is what stops a range from silently splitting an identity.
#
# `cargo tree -d` lists each duplicated crate as an unindented `name vX.Y.Z`
# line with its dependents indented under it, so only those top-level lines are
# matched.
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

# The CPU provider stack, the one faer identity, and the tensor/engine crates
# that pass their own values into it.
identities='strided[-a-z]*|tcontract[-a-z]*|cpueinsum[-a-z]*|tlinalg[-a-z]*|t4a-faer[-a-z]*|elem-simd-math|tenferro[-a-z]*|hataori[-a-z]*|t4a-cubecl[-a-z]*|t4a-cubek[-a-z]*|cubecl[-a-z]*|cubek[-a-z]*'

# A resolution error must fail this check too, so the tree is built first.
tree=$(cargo tree --locked -d -e normal)
duplicates=$(grep -E "^(${identities}) v" <<<"$tree" || true)
if [[ -n "$duplicates" ]]; then
    printf '%s\n' "$duplicates" >&2
    echo "check-single-versions: duplicate dependency identities in the normal graph" >&2
    exit 1
fi
echo "check-single-versions: one version per dependency identity"
