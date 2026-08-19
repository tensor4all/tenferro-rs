#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
manifest="$root/tests/fixtures/extension-authoring/Cargo.toml"

if grep -q 'tenferro-internal-' "$manifest"; then
  echo "extension authoring fixture must not depend on tenferro-internal-* crates" >&2
  exit 1
fi

cargo run --quiet --manifest-path "$manifest"
