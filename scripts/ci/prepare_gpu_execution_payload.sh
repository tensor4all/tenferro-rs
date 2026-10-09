#!/usr/bin/env bash
# Prepare execution dependencies on the hosted runner before GPU allocation.
set -euo pipefail
payload="$PWD/runtime-payload"
mkdir -p "$payload/bin" "$payload/wheels" "$payload/opt/tenferro-ci" "$payload/usr/local"
cp "$(rustup which cargo)" "$payload/bin/cargo"
cp "$(command -v cargo-nextest)" "$payload/bin/cargo-nextest"
cutensor_root="$TENFERRO_CI_CACHE_ROOT/cutensor-$CUTENSOR_VERSION"
if [ ! -s "$cutensor_root/lib/libcutensor.so.2" ]; then
  bash scripts/ci/install_cutensor.sh "$CUTENSOR_VERSION" "$cutensor_root"
fi
test -s "$cutensor_root/lib/libcutensor.so.2"
# Only libcutensor's shared ABI is loaded by this single-GPU test suite.
# Static archives and the independent multi-GPU/MPI providers never execute.
mkdir -p "$payload/opt/tenferro-ci/cutensor-$CUTENSOR_VERSION/lib"
cp -a "$cutensor_root"/lib/libcutensor.so* "$payload/opt/tenferro-ci/cutensor-$CUTENSOR_VERSION/lib/"
python3 -m pip download --only-binary=:all: --no-deps --python-version 312 --platform manylinux_2_27_x86_64 --platform manylinux2014_x86_64 --implementation cp --abi cp312 \
  --dest "$payload/wheels" "jax-cuda12-pjrt==$JAX_CUDA12_PJRT_VERSION" \
  "nvidia-cudnn-cu12==$NVIDIA_CUDNN_CU12_VERSION" "nvidia-cuda-nvcc-cu12==$NVIDIA_CUDA_NVCC_CU12_VERSION"
cuda_tree_ready() {
  local root="$1" lib
  test -f "$root/.seed-complete" || return 1
  for lib in nvrtc cublas cusolver cusparse; do
    local candidates=("$root"/lib64/lib"${lib}".so*)
    test -s "${candidates[0]}" || return 1
  done
  python3 scripts/ci/check_cuda_headers.py --cuda-root "$root"
}
for runtime in 12.6 12.8; do
  sdk_root="$TENFERRO_CI_CACHE_ROOT/cuda-runtime-$runtime"
  if ! cuda_tree_ready "$sdk_root"; then
    bash scripts/ci/install_cuda_runtime_tree.sh "$runtime" "$sdk_root"
    cuda_tree_ready "$sdk_root"
  fi
  transfer="$PWD/runtime-sdk-$runtime"
  mkdir -p "$transfer"
  tar --zstd -cf "$transfer/sdk.tar.zst" -C "$sdk_root" .
  (cd "$transfer"; sha256sum sdk.tar.zst > sdk.sha256; split -n 5 -d -a 2 sdk.tar.zst sdk.part; rm sdk.tar.zst)
  for part in 00 01 02 03 04; do
    mkdir "$transfer/$part"
    mv "$transfer/sdk.part$part" "$transfer/$part/"
  done
  mv "$transfer/sdk.sha256" "$transfer/00/"
done
transfer="$PWD/runtime-common"
mkdir -p "$transfer"
tar --zstd -cf "$transfer/runtime.tar.zst" -C "$payload" .
(cd "$transfer"; sha256sum runtime.tar.zst > runtime.sha256; split -n 5 -d -a 2 runtime.tar.zst runtime.part; rm runtime.tar.zst)
for part in 00 01 02 03 04; do
  mkdir "$transfer/$part"
  mv "$transfer/runtime.part$part" "$transfer/$part/"
done
mv "$transfer/runtime.sha256" "$transfer/00/"
