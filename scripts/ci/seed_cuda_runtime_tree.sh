#!/usr/bin/env bash
# Copy CUDA JIT headers and shared runtime libraries without compiler/static
# archives or duplicate lib64/include alias trees. No package installation.
set -euo pipefail
SOURCE_DIR="${1:?usage: seed_cuda_runtime_tree.sh <installed-cuda-dir> <dest-dir>}"
DEST_DIR="${2:?usage: seed_cuda_runtime_tree.sh <installed-cuda-dir> <dest-dir>}"
source_include="$(realpath "${SOURCE_DIR}/include")"
source_lib="$(realpath "${SOURCE_DIR}/lib64")"
test -d "${source_include}"
test -d "${source_lib}"
# Validate the source before replacing a partial destination.
compgen -G "${source_lib}/*.so*" >/dev/null
rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}/targets/x86_64-linux/include" "${DEST_DIR}/targets/x86_64-linux/lib"
cp -aL "${source_include}/." "${DEST_DIR}/targets/x86_64-linux/include/"
# Preserve soname symlinks; each real shared library is copied once.
find "${source_lib}" -maxdepth 1 -name '*.so*' -exec cp -a -t "${DEST_DIR}/targets/x86_64-linux/lib/" -- {} +
ln -s targets/x86_64-linux/include "${DEST_DIR}/include"
ln -s targets/x86_64-linux/lib "${DEST_DIR}/lib64"
# Catch an incomplete shared-library family before marking the tree ready.
find -L "${DEST_DIR}/lib64/" -maxdepth 1 -type l -print > "${DEST_DIR}/broken-links"
if [ -s "${DEST_DIR}/broken-links" ]; then
  cat "${DEST_DIR}/broken-links" >&2
  exit 1
fi
rm "${DEST_DIR}/broken-links"
touch "${DEST_DIR}/.seed-complete"
