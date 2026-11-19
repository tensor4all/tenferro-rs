#!/usr/bin/env bash
# Public API surface of every published tenferro crate (#2552, S0; #2663).
#
#   scripts/api-snapshot.sh    compile the surface into a temporary directory
#                               and check it against the api_ops! registries
#
# Nothing is committed (#2663): the surface is generated, and the verdict is
# scripts/check-api-registry-conformance.py's (#2602) — an undeclared eager
# entry point, a duplicate alias or an engine argument outside the session
# boundary fails whatever the surface contains. The generated files stay in
# the temporary directory for inspection when the check fails.
#
# Needs `cargo-public-api` and the pinned nightly below (rustdoc JSON); the
# rendering of rustdoc JSON changes between nightlies, so CI installs the same
# dated toolchain (`.github/workflows/ci-pr-workspace-tests.yml`, `api-surface`).
# Override with API_SNAPSHOT_TOOLCHAIN only to try a newer nightly, then re-pin
# both. The feature combinations are fixed below and documented in
# api/README.md; change both together. The surface is review evidence, not a
# proof of behaviour.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 0 ]; then
    echo "usage: $0" >&2
    exit 2
fi

toolchain="${API_SNAPSHOT_TOOLCHAIN:-nightly-2026-07-18}"

# Every root-workspace package that is published (`publish = false` packages,
# including the standalone ext/sparse and ext/tropical manifests, are excluded).
# tenferro-internal-extension-macros is a proc-macro crate, for which rustdoc
# emits no JSON; its exported macros are reviewed through the crates using them.
# Format: <crate> <file-suffix> <features ("-" for the default set)>.
# The second row of a crate is its docs.rs feature set, which is the combination
# that widens the documented public surface; a crate with one row has no default
# feature set, so its default build and its docs.rs build document the same
# surface (tenferro-cpu's `ledger`/`route-report` only gate `#[doc(hidden)]` items).
COMBOS=(
    "tenferro-tensor-core default -"
    "tenferro-tensor default -"
    "tenferro-core-ops default -"
    "tenferro-internal-cpu-kernels default -"
    "tenferro-cpu-basic default -"
    "tenferro-cpu-fused default -"
    "tenferro-cpu default -"
    "tenferro-runtime default -"
    "tenferro-internal-ops default -"
    "tenferro-internal-ops all autodiff,cuda"
    "tenferro-gpu default -"
    "tenferro-gpu cuda cuda"
    "tenferro-gpu metal metal"
    "tenferro-xla default -"
    "tenferro-xla pjrt pjrt"
    "tenferro-einsum default -"
    "tenferro-einsum autodiff-cuda autodiff,cuda"
    "tenferro-ad default -"
    "tenferro-ad cuda cuda"
    "tenferro-linalg default -"
    "tenferro-linalg autodiff-cuda autodiff,cuda"
    "tenferro-fft default -"
    "tenferro-fft autodiff-cuda autodiff,cuda"
)

out_dir=$(mktemp -d)
trap 'rm -rf "$out_dir"' EXIT

for combo in "${COMBOS[@]}"; do
    read -r crate suffix features <<<"$combo"
    args=(-p "$crate" -ss)
    if [ "$features" != - ]; then
        args+=(--features "$features")
    fi
    if [ "$suffix" = default ]; then
        file="$out_dir/$crate.txt"
    else
        file="$out_dir/$crate+$suffix.txt"
    fi
    echo "api-snapshot: $crate ($suffix)" >&2
    RUSTUP_TOOLCHAIN="$toolchain" cargo public-api "${args[@]}" >"$file"
done

if python3 scripts/check-api-registry-conformance.py --surface-dir "$out_dir"; then
    echo "api-snapshot: ok" >&2
else
    trap - EXIT
    echo "api-snapshot: the compiled surface does not conform to the api_ops! registries; surface kept at $out_dir" >&2
    exit 1
fi
