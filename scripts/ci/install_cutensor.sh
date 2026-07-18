#!/usr/bin/env bash
# Download and extract the cuTENSOR redistributable for CUDA 12 into the
# requested destination directory (lib/ subdirectory holds the libraries).
#
# Usage: install_cutensor.sh <cutensor-version> <dest-dir>
set -euo pipefail

CUTENSOR_VERSION="${1:?usage: install_cutensor.sh <cutensor-version> <dest-dir>}"
DEST_DIR="${2:?usage: install_cutensor.sh <cutensor-version> <dest-dir>}"

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

cutensor_archive="libcutensor-linux-x86_64-${CUTENSOR_VERSION}_cuda12-archive.tar.xz"
cutensor_url="https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/${cutensor_archive}"
cutensor_min_bytes=400000000

echo "Installing cuTENSOR ${CUTENSOR_VERSION} redistributable for CUDA 12..."
tmpdir="$(mktemp -d)"
archive_path="${tmpdir}/${cutensor_archive}"
curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors \
  -o "${archive_path}" "${cutensor_url}"
archive_bytes="$(wc -c < "${archive_path}" | tr -d ' ')"
if [ "${archive_bytes}" -lt "${cutensor_min_bytes}" ]; then
  echo "cuTENSOR download too small (${archive_bytes} bytes); expected at least ${cutensor_min_bytes}."
  rm -rf "${tmpdir}"
  exit 1
fi
as_root rm -rf "${DEST_DIR}"
as_root mkdir -p "${DEST_DIR}"
as_root tar -xJf "${archive_path}" -C "${DEST_DIR}" \
  --strip-components=1 \
  "libcutensor-linux-x86_64-${CUTENSOR_VERSION}_cuda12-archive/lib"
rm -rf "${tmpdir}"
if [ "$(id -u)" -ne 0 ]; then
  sudo chown -R "$(id -u):$(id -g)" "${DEST_DIR}"
fi

if [ ! -e "${DEST_DIR}/lib/libcutensor.so.2" ] && \
   [ ! -e "${DEST_DIR}/lib/libcutensor.so.2.6.0" ]; then
  echo "cuTENSOR shared library not found after redistributable install."
  ls -la "${DEST_DIR}/lib" 2>/dev/null || echo "Missing directory: ${DEST_DIR}/lib"
  exit 1
fi
echo "cuTENSOR ${CUTENSOR_VERSION} ready at ${DEST_DIR}/lib."
