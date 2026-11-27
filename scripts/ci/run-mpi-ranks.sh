#!/usr/bin/env bash
# Run the two-rank scenarios and require every rank to report success.
#
#   scripts/ci/run-mpi-ranks.sh [upstream|rt|capi]
#
# Wraps scripts/run-mpi.sh without changing it. The scenario binaries print
# "needs two ranks, skipping" and exit 0 when they see a world of one, so a launcher
# that starts two singletons instead of one two-rank world (an mpiexec from a
# different MPI family than the library the binary linked, for example) would pass
# without a single message crossing between processes. This makes that a failure.
set -euo pipefail

root=$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
# The scenarios assert a world of exactly two ranks.
ranks=${RANKS:-2}

log=$(mktemp)
trap 'rm -f "$log"' EXIT

"$root/scripts/run-mpi.sh" "$@" 2>&1 | tee "$log"

status=0
if grep -q 'needs two ranks, skipping' "$log"; then
    echo "error: a process saw a world of one rank; the launcher did not form a ${ranks}-rank world" >&2
    status=1
fi
for ((rank = 0; rank < ranks; rank++)); do
    if ! grep -qx "rank ${rank}: all scenarios passed" "$log"; then
        echo "error: rank ${rank} did not report 'all scenarios passed'" >&2
        status=1
    fi
done
exit "$status"
