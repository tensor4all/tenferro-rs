#!/usr/bin/env bash
# Install (or discover) a CUDA toolkit with nvcc on a GitHub-hosted Ubuntu
# runner for building the CUDA/PJRT PTX test archives.
#
# Usage: install_cuda_toolkit_hosted.sh <cuda-runtime-version>
#
# Accepts any toolkit whose version is >= the requested version. Exports
# CUDA_PATH and LD_LIBRARY_PATH through GITHUB_ENV and prepends the toolkit
# bin directory to GITHUB_PATH so later workflow steps can use nvcc.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

CUDA_RUNTIME_VERSION="${1:?usage: install_cuda_toolkit_hosted.sh <cuda-runtime-version>}"

cuda_toolkit_version() {
  local root="$1"
  if [ -f "${root}/version.json" ]; then
    python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cuda']['version'].rsplit('.', 1)[0])" "${root}/version.json" 2>/dev/null || true
  elif [ -f "${root}/version.txt" ]; then
    sed -n 's/.*CUDA Version \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' "${root}/version.txt" | head -1
  else
    basename "${root}" | sed -n 's/^cuda-\([0-9][0-9]*\.[0-9][0-9]*\)$/\1/p'
  fi
}

cuda_version_ge() {
  local version="$1"
  local min_version="$2"
  if [ -z "${version}" ]; then
    return 1
  fi
  local ver_major="${version%%.*}"
  local ver_minor="${version#*.}"
  local min_major="${min_version%%.*}"
  local min_minor="${min_version#*.}"
  if [ "${ver_major}" -gt "${min_major}" ]; then
    return 0
  fi
  if [ "${ver_major}" -lt "${min_major}" ]; then
    return 1
  fi
  [ "${ver_minor}" -ge "${min_minor}" ]
}

accept_cuda_toolkit() {
  local candidate="$1"
  if [ ! -d "${candidate}" ]; then
    return 1
  fi
  if [ ! -d "${candidate}/lib64" ] && [ ! -d "${candidate}/targets/x86_64-linux/lib" ]; then
    echo "Skipping CUDA toolkit at ${candidate} (library directory not found)."
    return 1
  fi
  local version
  version="$(cuda_toolkit_version "${candidate}")"
  if cuda_version_ge "${version}" "${CUDA_RUNTIME_VERSION}"; then
    cuda_path="${candidate}"
    echo "Using CUDA toolkit at ${candidate} (version ${version})."
    return 0
  fi
  if [ -n "${version}" ]; then
    echo "Skipping CUDA toolkit at ${candidate} (found ${version}, need >= ${CUDA_RUNTIME_VERSION})."
  else
    echo "Skipping CUDA toolkit at ${candidate} (version metadata not found, need >= ${CUDA_RUNTIME_VERSION})."
  fi
  return 1
}

cuda_path=""
if [ -n "${CUDA_PATH:-}" ]; then
  accept_cuda_toolkit "${CUDA_PATH}" || cuda_path=""
fi
if [ -z "${cuda_path}" ] && command -v nvcc >/dev/null 2>&1; then
  accept_cuda_toolkit "$(dirname "$(dirname "$(command -v nvcc)")")" || true
fi
if [ -z "${cuda_path}" ]; then
  for candidate in \
    "/usr/local/cuda-${CUDA_RUNTIME_VERSION}" \
    /usr/local/cuda; do
    if accept_cuda_toolkit "${candidate}"; then
      break
    fi
  done
fi

if [ -z "${cuda_path}" ]; then
  echo "Installing CUDA ${CUDA_RUNTIME_VERSION} toolkit packages for archive PTX..."
  cuda_apt_suffix="${CUDA_RUNTIME_VERSION//./-}"
  if ! ls /etc/apt/sources.list.d/cuda*.list >/dev/null 2>&1; then
    tmpdir="$(mktemp -d)"
    curl -fsSL --retry 5 --retry-delay 5 --retry-all-errors \
      -o "${tmpdir}/cuda-keyring.deb" \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i "${tmpdir}/cuda-keyring.deb"
    rm -rf "${tmpdir}"
  fi
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    "cuda-cudart-${cuda_apt_suffix}" \
    "cuda-cudart-dev-${cuda_apt_suffix}" \
    "cuda-nvcc-${cuda_apt_suffix}" \
    "cuda-nvrtc-${cuda_apt_suffix}" \
    "cuda-nvrtc-dev-${cuda_apt_suffix}" \
    "libcublas-${cuda_apt_suffix}" \
    "libcublas-dev-${cuda_apt_suffix}" \
    "libcusparse-${cuda_apt_suffix}" \
    "libnvjitlink-${cuda_apt_suffix}"
  cuda_path=""
  for candidate in "/usr/local/cuda-${CUDA_RUNTIME_VERSION}" /usr/local/cuda; do
    if accept_cuda_toolkit "${candidate}"; then
      break
    fi
  done
fi

if [ -z "${cuda_path}" ]; then
  echo "CUDA toolkit path not found after CUDA ${CUDA_RUNTIME_VERSION} install on archive runner."
  exit 1
fi

nvcc_bin="${cuda_path}/bin/nvcc"
if [ ! -x "${nvcc_bin}" ]; then
  echo "nvcc not found at ${nvcc_bin}" >&2
  ls -la "${cuda_path}/bin" || true
  exit 1
fi

cuda_library_path="${cuda_path}/lib64"
if [ -d "${cuda_path}/targets/x86_64-linux/lib" ]; then
  cuda_library_path="${cuda_library_path}:${cuda_path}/targets/x86_64-linux/lib"
fi

echo "CUDA_PATH=${cuda_path}" >> "${GITHUB_ENV}"
echo "${cuda_path}/bin" >> "${GITHUB_PATH}"
echo "LD_LIBRARY_PATH=${cuda_library_path}:${LD_LIBRARY_PATH:-}" >> "${GITHUB_ENV}"
"${nvcc_bin}" --version
