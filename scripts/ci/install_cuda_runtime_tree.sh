#!/usr/bin/env bash
# Install the minimal CUDA runtime packages (no compiler, no driver) and seed
# a relocatable runtime tree at the requested destination.
#
# Usage: install_cuda_runtime_tree.sh <cuda-runtime-version> <dest-dir>
#
# The destination tree ends with a `.seed-complete` marker so consumers can
# distinguish a fully seeded tree from a partial copy. Runs as root directly
# (RunPod pods) or through sudo (GitHub-hosted runners).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

CUDA_RUNTIME_VERSION="${1:?usage: install_cuda_runtime_tree.sh <cuda-runtime-version> <dest-dir>}"
DEST_DIR="${2:?usage: install_cuda_runtime_tree.sh <cuda-runtime-version> <dest-dir>}"

as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

echo "Installing minimal CUDA ${CUDA_RUNTIME_VERSION} runtime packages without driver packages..."
cuda_apt_suffix="${CUDA_RUNTIME_VERSION//./-}"
if ! ls /etc/apt/sources.list.d/cuda*.list >/dev/null 2>&1; then
  tmpdir="$(mktemp -d)"
  curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors \
    -o "${tmpdir}/cuda-keyring.deb" \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  as_root dpkg -i "${tmpdir}/cuda-keyring.deb"
  rm -rf "${tmpdir}"
fi
as_root apt-get update
as_root apt-get install -y --no-install-recommends \
  "cuda-cudart-${cuda_apt_suffix}" \
  "cuda-cudart-dev-${cuda_apt_suffix}" \
  "cuda-nvrtc-${cuda_apt_suffix}" \
  "cuda-nvrtc-dev-${cuda_apt_suffix}" \
  "libcublas-${cuda_apt_suffix}" \
  "libcusolver-${cuda_apt_suffix}" \
  "libcusparse-${cuda_apt_suffix}" \
  "libnvjitlink-${cuda_apt_suffix}"

installed_path=""
for candidate in "/usr/local/cuda-${CUDA_RUNTIME_VERSION}" /usr/local/cuda; do
  if [ -d "${candidate}/lib64" ] || [ -d "${candidate}/targets/x86_64-linux/lib" ]; then
    installed_path="${candidate}"
    break
  fi
done
if [ -z "${installed_path}" ]; then
  echo "CUDA toolkit path not found after minimal CUDA ${CUDA_RUNTIME_VERSION} runtime package install."
  exit 1
fi

echo "Seeding CUDA runtime tree at ${DEST_DIR} from ${installed_path}..."
as_root rm -rf "${DEST_DIR}"
as_root mkdir -p "${DEST_DIR}"
as_root cp -aL "${installed_path}/." "${DEST_DIR}/"
as_root touch "${DEST_DIR}/.seed-complete"
if [ "$(id -u)" -ne 0 ]; then
  sudo chown -R "$(id -u):$(id -g)" "${DEST_DIR}"
fi
echo "CUDA ${CUDA_RUNTIME_VERSION} runtime tree ready at ${DEST_DIR}."
