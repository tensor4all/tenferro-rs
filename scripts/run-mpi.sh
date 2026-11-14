#!/usr/bin/env bash
# Run the two-rank transport scenarios against one backend.
#
#   scripts/run-mpi.sh [upstream|rt]
#
# `upstream` is `tenferro-mpi --features system` (rsmpi-rt's bindgen backend, needs
# system MPI headers). `rt` is `tenferro-mpi --features runtime` (rsmpi-rt's
# MPIABI backend, needs `MPI_RT_LIB` and MPIwrapper at run time).
#
# The scenarios need a real MPI launcher; a plain `cargo test` runs the
# single-process checks only.
set -euo pipefail

root=$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
backend=${1:-upstream}
jobs=${JOBS:-4}
ranks=${RANKS:-2}
launcher=${MPIEXEC:-mpiexec}

# mpi-sys builds bindgen output; without this clang cannot find stddef.h on Debian and
# Ubuntu hosts where the GCC private include directory is not on clang's default path.
export BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-"-I$(gcc -print-file-name=include)"}

cd "$root"
case "$backend" in
upstream)
    cargo_args=(-p tenferro-mpi --features system --test mpi_p2p)
    ;;
rt)
    cargo_args=(-p tenferro-mpi --features runtime --test mpi_p2p_runtime)
    if [[ -z "${MPI_RT_LIB:-}" ]]; then
        echo "MPI_RT_LIB must point at a loaded MPI ABI wrapper library for the rt backend" >&2
        echo "build MPIwrapper against the system MPI and point MPI_RT_LIB at it" >&2
        echo "(see crates/tenferro-mpi/README.md)" >&2
        exit 2
    fi
    ;;
*)
    echo "usage: $0 [upstream|rt]" >&2
    exit 2
    ;;
esac

# Ask cargo for the scenario binary instead of guessing its path: a build-cache
# wrapper (kache here) relocates test executables outside target/debug/deps.
binary=$(cargo test -j "$jobs" "${cargo_args[@]}" --no-run --message-format=json 2>/dev/null |
    python3 -c '
import json, sys
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    if message.get("profile", {}).get("test") and message.get("executable"):
        print(message["executable"])
' | tail -n 1)
if [[ -z "$binary" ]]; then
    echo "cargo did not report a scenario executable" >&2
    exit 1
fi
echo "== $backend: $binary on $ranks ranks =="
"$launcher" -n "$ranks" "$binary"
