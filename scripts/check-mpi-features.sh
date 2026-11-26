#!/usr/bin/env bash
# Check the backend-feature combinations of `tenferro-mpi`.
#
#   scripts/check-mpi-features.sh tenferro-mpi
#   scripts/check-mpi-features.sh tenferro-mpi --runtime-only
#
# `--runtime-only` skips the `system` graph and the both-feature check, which need
# system MPI headers and libclang. Use it from the runtime lane, which installs none.
#
# `tenferro-mpi` is a workspace member with `default = []`, so the root default
# lanes compile it without an MPI binding. This check pins the rest of the contract:
#
#   default / `engine`  no `rsmpi-rt`, no `rsmpi-sys`, no `mpi-native-sys`, no `mpi`;
#   `system`            exactly one rsmpi-rt 0.9.0 and one rsmpi-sys 0.2.4, no
#                       mpi-native-sys; needs system MPI headers and libclang;
#   `runtime`           exactly one rsmpi-rt 0.9.0 and one mpi-native-sys 0.1.0, no
#                       rsmpi-sys; resolves MPI at run time from the host's library;
#   `system`+`runtime`  must not build; rsmpi-rt rejects its two backends;
#   every graph         no `mpi-rt-sys`: the MPItrampoline backend is deleted
#                       (docs/design/mpi-native-abi.md, "What depends on the absence");
#   `tenferro-mpi-capi` exactly one mpi-native-sys, no mpi-rt-sys, no rsmpi-sys, and no
#                       `panic = "abort"` profile anywhere (D10: every panic stub would
#                       abort the host process).
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

crate=${1:-tenferro-mpi}
mode=${2:-all}
jobs=${JOBS:-4}
# rsmpi-sys builds bindgen output; clang needs the GCC private include directory for
# stddef.h on hosts where it is not on clang's default search path.
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

[[ "$crate" == "tenferro-mpi" ]] || {
    echo "check-mpi-features: unknown crate $crate (only tenferro-mpi remains)" >&2
    exit 2
}
[[ "$mode" == "all" || "$mode" == "--runtime-only" ]] || {
    echo "check-mpi-features: unknown mode $mode (use --runtime-only or omit it)" >&2
    exit 2
}
[[ -f "crates/tenferro-mpi/Cargo.toml" ]] || {
    echo "check-mpi-features: no manifest at crates/tenferro-mpi/Cargo.toml" >&2
    exit 2
}

tree() {
    cargo tree -p tenferro-mpi "$@" -e normal --prefix none
}

expect_absent() {
    local label=$1
    shift
    if tree "$@" | grep -qE '^(rsmpi-rt|rsmpi-sys|mpi-rt-sys|mpi-native-sys|mpi) v'; then
        echo "check-mpi-features: $label graph pulled an MPI package" >&2
        tree "$@" >&2
        exit 1
    fi
}

# The default build is the protocol shell only; it must not pull a binding.
cargo check -j "$jobs" -p tenferro-mpi
expect_absent default

# `engine` alone has no MPI at all, for a caller that supplies its own backend.
cargo check -j "$jobs" -p tenferro-mpi --features engine
expect_absent engine --features engine

if [[ "$mode" == "all" ]]; then
# The system backend: exactly one rsmpi-rt and one rsmpi-sys.
cargo check -j "$jobs" -p tenferro-mpi --features system
system_tree=$(tree --features system)
found=$(grep -cE '^rsmpi-rt v0\.9\.0' <<<"$system_tree" || true)
[[ "$found" == 1 ]] || {
    echo "check-mpi-features: expected one rsmpi-rt 0.9.0 in the system graph, found $found" >&2
    printf '%s\n' "$system_tree" >&2
    exit 1
}
found=$(grep -cE '^rsmpi-sys v0\.2\.4' <<<"$system_tree" || true)
[[ "$found" == 1 ]] || {
    echo "check-mpi-features: expected one rsmpi-sys 0.2.4 in the system graph, found $found" >&2
    printf '%s\n' "$system_tree" >&2
    exit 1
}
if grep -qE '^mpi-native-sys v' <<<"$system_tree"; then
    echo "check-mpi-features: the system backend also resolved mpi-native-sys" >&2
    exit 1
fi
if grep -qE '^mpi-rt-sys v' <<<"$system_tree"; then
    echo "check-mpi-features: the system graph resolved the deleted mpi-rt-sys" >&2
    exit 1
fi
fi

# The runtime backend: exactly one rsmpi-rt and one mpi-native-sys, no bindgen sys crate.
cargo check -j "$jobs" -p tenferro-mpi --features runtime
runtime_tree=$(tree --features runtime)
found=$(grep -cE '^rsmpi-rt v0\.9\.0' <<<"$runtime_tree" || true)
[[ "$found" == 1 ]] || {
    echo "check-mpi-features: expected one rsmpi-rt 0.9.0 in the runtime graph, found $found" >&2
    printf '%s\n' "$runtime_tree" >&2
    exit 1
}
found=$(grep -cE '^mpi-native-sys v0\.1\.0' <<<"$runtime_tree" || true)
[[ "$found" == 1 ]] || {
    echo "check-mpi-features: expected one mpi-native-sys 0.1.0 in the runtime graph, found $found" >&2
    printf '%s\n' "$runtime_tree" >&2
    exit 1
}
if grep -qE '^mpi-rt-sys v' <<<"$runtime_tree"; then
    echo "check-mpi-features: the runtime graph resolved the deleted mpi-rt-sys" >&2
    exit 1
fi
if grep -qE '^(rsmpi-sys|mpi-sys) v' <<<"$runtime_tree"; then
    echo "check-mpi-features: the runtime backend resolved the system sys crate" >&2
    exit 1
fi

# The C entry crate: the native backend only, and never an aborting panic profile.
cargo check -j "$jobs" -p tenferro-mpi-capi
capi_tree=$(cargo tree -p tenferro-mpi-capi -e normal --prefix none)
found=$(grep -cE '^mpi-native-sys v' <<<"$capi_tree" || true)
[[ "$found" == 1 ]] || {
    echo "check-mpi-features: expected one mpi-native-sys in the tenferro-mpi-capi graph, found $found" >&2
    exit 1
}
if grep -qE '^(mpi-rt-sys|rsmpi-sys|mpi-sys) v' <<<"$capi_tree"; then
    echo "check-mpi-features: the tenferro-mpi-capi graph resolved a deleted or system sys crate" >&2
    exit 1
fi
if grep -nE '^[[:space:]]*panic[[:space:]]*=[[:space:]]*"abort"' Cargo.toml crates/*/Cargo.toml; then
    echo "check-mpi-features: a profile sets panic = \"abort\"; the C entry needs unwinding (D10)" >&2
    exit 1
fi

if [[ "$mode" == "all" ]]; then
# Enabling both bindings is a configuration error. rsmpi-rt itself rejects its two
# backend features, so the failure is reported inside the dependency before this
# crate compiles; the property under test is that the combination does not build.
if cargo check -j "$jobs" -p tenferro-mpi --features system,runtime >/dev/null 2>&1; then
    echo "check-mpi-features: system+runtime built instead of failing" >&2
    exit 1
fi
fi

echo "check-mpi-features: $crate feature combinations are consistent"
