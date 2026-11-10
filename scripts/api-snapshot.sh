#!/usr/bin/env bash
# Public API snapshot of every published tenferro crate (#2552, S0).
#
#   scripts/api-snapshot.sh --write   regenerate api/*.txt
#   scripts/api-snapshot.sh --check   regenerate into a temporary directory and
#                                     diff against api/; exit 1 on any difference
#
# Needs `cargo-public-api` and a nightly toolchain (rustdoc JSON). The feature
# combinations are fixed below and documented in api/README.md; change both
# together. A snapshot is review evidence, not a proof of behaviour.
set -euo pipefail
cd "$(dirname "$0")/.."

mode="${1:-}"
case "$mode" in
--write | --check) ;;
*)
    echo "usage: $0 --write | --check" >&2
    exit 2
    ;;
esac

# Every root-workspace package that is published (`publish = false` packages,
# including the standalone ext/sparse and ext/tropical manifests, are excluded).
# tenferro-internal-extension-macros is a proc-macro crate, for which rustdoc
# emits no JSON; its exported macros are reviewed through the crates using them.
# Format: <crate> <file-suffix> <features ("-" for the default set)>.
# The second row of a crate is its docs.rs feature set, which is the combination
# that widens the public surface; it is omitted where that set equals the
# default set (tenferro-cpu and tenferro-runtime: `native` is the default).
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
    "tenferro-gpu native-cuda native,cuda"
    "tenferro-xla default -"
    "tenferro-xla pjrt pjrt"
    "tenferro-einsum default -"
    "tenferro-einsum autodiff-native-cuda autodiff,native,cuda"
    "tenferro-ad default -"
    "tenferro-ad native-cuda native,cuda"
    "tenferro-linalg default -"
    "tenferro-linalg autodiff-native-cuda autodiff,native,cuda"
    "tenferro-fft default -"
    "tenferro-fft autodiff-native-cuda autodiff,native,cuda"
)

if [ "$mode" = --write ]; then
    out_dir=api
else
    out_dir=$(mktemp -d)
    trap 'rm -rf "$out_dir"' EXIT
fi
mkdir -p "$out_dir"

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
    cargo public-api "${args[@]}" >"$file"
done

if [ "$mode" = --check ]; then
    status=0
    for file in "$out_dir"/*.txt; do
        name=$(basename "$file")
        if [ ! -f "api/$name" ]; then
            echo "api-snapshot: api/$name is missing (run $0 --write)" >&2
            status=1
            continue
        fi
        if ! diff -u "api/$name" "$file"; then
            status=1
        fi
    done
    for file in api/*.txt; do
        name=$(basename "$file")
        if [ ! -f "$out_dir/$name" ]; then
            echo "api-snapshot: api/$name is stale (no such crate/feature combination)" >&2
            status=1
        fi
    done
    if [ "$status" -ne 0 ]; then
        echo "api-snapshot: the public API differs from api/; review the diff and run $0 --write" >&2
        exit 1
    fi
    echo "api-snapshot: ok" >&2
fi
