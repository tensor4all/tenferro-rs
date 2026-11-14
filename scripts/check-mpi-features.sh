#!/usr/bin/env bash
# Check the backend-feature combinations of one MPI adapter crate.
#
#   scripts/check-mpi-features.sh ext/mpi
#   scripts/check-mpi-features.sh ext/mpi-rt
#
# Both crates are standalone workspaces, so a feature mistake there is invisible to
# the root `cargo clippy`/`cargo test` lanes. Two properties are checked, and both
# are about what a dependency graph may contain:
#
#   ext/mpi     `--no-default-features` must fail with the crate's own message: an
#               adapter with no MPI binding compiles nothing useful, and a silent
#               empty build is how a missing backend reaches a release.
#               `--no-default-features --features engine` must succeed, because
#               that is exactly how tenferro-mpi-rt reuses the protocol.
#   ext/mpi-rt  must resolve without the crates.io `mpi`/`mpi-sys` identity: `mpi-sys`
#               declares `links = "mpi"` wherever it comes from, so two of them in one
#               graph cannot link.
set -Eeuo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

crate=${1:-ext/mpi}
jobs=${JOBS:-4}
# mpi-sys builds bindgen output; clang needs the GCC private include directory for
# stddef.h on hosts where it is not on clang's default search path.
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

manifest="$crate/Cargo.toml"
[[ -f "$manifest" ]] || {
    echo "check-mpi-features: no manifest at $manifest" >&2
    exit 2
}

case "$crate" in
ext/mpi)
    # A build with no backend must stop with the compile_error!, not produce an
    # adapter whose entry points are all feature-gated away.
    if output=$(cargo check -j "$jobs" --manifest-path "$manifest" --no-default-features 2>&1); then
        echo "check-mpi-features: --no-default-features built without a backend" >&2
        exit 1
    fi
    grep -q "enable the \`upstream\` MPI backend feature" <<<"$output" || {
        printf '%s\n' "$output" >&2
        echo "check-mpi-features: the no-backend build failed for a different reason" >&2
        exit 1
    }
    cargo check -j "$jobs" --manifest-path "$manifest" --no-default-features --features engine

    # The adapter must never pull an MPI binding into a consumer's graph unless that
    # consumer asked for one, so `engine` alone has no `mpi` at all.
    if cargo tree --manifest-path "$manifest" --no-default-features --features engine -e normal |
        grep -qE '^mpi v|─ mpi v'; then
        echo "check-mpi-features: the engine feature pulled an MPI binding" >&2
        exit 1
    fi

    # And the default graph must contain exactly one `mpi` and no `tenferro-tensor` MPI.
    found=$(cargo tree --manifest-path "$manifest" -e normal --prefix none | grep -c '^mpi v' || true)
    [[ "$found" == 1 ]] || {
        echo "check-mpi-features: expected one mpi identity, found $found" >&2
        exit 1
    }
    ;;
ext/mpi-rt)
    cargo check -j "$jobs" --manifest-path "$manifest" --all-targets
    # rsmpi-rt's `mpi` must be the only binding: a second `mpi-sys` would declare
    # `links = "mpi"` again, and two link declarations cannot coexist.
    if cargo tree --manifest-path "$manifest" -e normal |
        grep -q 'mpi-sys v'; then
        echo "check-mpi-features: the runtime backend resolved a second mpi-sys" >&2
        exit 1
    fi
    ;;
*)
    echo "check-mpi-features: unknown crate $crate" >&2
    exit 2
    ;;
esac

echo "check-mpi-features: $crate feature combinations are consistent"
