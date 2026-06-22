#!/usr/bin/env bash
set -euo pipefail

crate_tree="$(cargo tree -p tenferro-tensor-core -e normal --no-default-features)"

for forbidden in \
  tenferro-tensor \
  tenferro-gpu \
  tenferro-runtime \
  tenferro-linalg \
  cblas-sys \
  lapack \
  cubecl \
  t4a-cubecl
do
  if grep -Eq "(^|[[:space:]])${forbidden//-/-}[[:space:]v]" <<<"${crate_tree}"; then
    echo "tenferro-tensor-core must not depend on ${forbidden}" >&2
    echo "${crate_tree}" >&2
    exit 1
  fi
done

echo "tensor-core-deps-ok"
