#!/usr/bin/env bash
# Fail if a built binary links an Apple GPU framework at build time.
#
# The Metal backend opens Metal, MetalPerformanceShaders and Accelerate with `dlopen` on
# first use (design §7, tensor4all/tenferro-rs#2402), so a binary that uses it links none of
# them. An entry here means a dependency crept back to build-time linking, which would make
# the binary fail to load on a host without those frameworks instead of reporting a typed
# error from the probe.
#
# Usage: scripts/check-macos-no-framework-link.sh [binary]
# Without an argument, the newest `metal_engine` binary under the Cargo target directory is
# checked. That directory is `target` by default and `bitbake`-style overrides through
# `CARGO_TARGET_DIR` are honoured; the search covers every profile, because hosted CI builds
# the `ci` profile (`target/ci`) while a local run builds `dev` (`target/debug`).
set -euo pipefail

binary=${1:-}
if [ -z "$binary" ]; then
    target=${CARGO_TARGET_DIR:-target}
    if [ ! -d "$target" ]; then
        echo "check-macos-no-framework-link: $target does not exist; build the test first" >&2
        exit 1
    fi
    binary=$(find "$target" \
        \( -name doc -o -name package -o -name incremental \) -prune -o \
        -type f -name 'metal_engine-*' ! -name '*.d' -print 2>/dev/null \
        | xargs ls -t 2>/dev/null | head -1 || true)
fi
if [ -z "$binary" ] || [ ! -f "$binary" ]; then
    echo "check-macos-no-framework-link: no metal_engine test binary; build it first" >&2
    exit 1
fi

linked=$(otool -L "$binary" | grep -E 'Metal|MetalPerformanceShaders|Accelerate' || true)
if [ -n "$linked" ]; then
    echo "check-macos-no-framework-link: $binary links an Apple GPU framework" >&2
    echo "$linked" >&2
    exit 1
fi
echo "check-macos-no-framework-link: $binary links no Apple GPU framework"
