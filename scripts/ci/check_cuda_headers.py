"""Compile a CubeCL-style CUDA source with the bundled NVRTC, without a GPU."""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

HEADERS = ('cuda_runtime.h', 'cuda_fp16.h', 'cuda_bf16.h', 'cuComplex.h',
           'mma.h', 'cooperative_groups.h', 'cuda/barrier')
SOURCE = '\n'.join(f'#include <{name}>' for name in HEADERS) + '''
extern "C" __global__ void header_proof(int *output) { output[0] = 42; }
'''


def check(cuda_root: Path) -> None:
    include = cuda_root / 'include'
    for header in HEADERS:
        if not any((directory / header).is_file() for directory in (include, include / 'cccl')):
            raise ValueError(f'Missing bundled CUDA JIT header: {header}')
    builtins = sorted((cuda_root / 'lib64').glob('libnvrtc-builtins.so.*'))
    if not builtins:
        raise ValueError('Missing bundled NVRTC builtins')
    ctypes.CDLL(str(builtins[0]), mode=ctypes.RTLD_GLOBAL)
    nvrtc = ctypes.CDLL(str(cuda_root / 'lib64/libnvrtc.so.12'))
    program = ctypes.c_void_p()
    status = nvrtc.nvrtcCreateProgram(ctypes.byref(program), SOURCE.encode(),
                                     b'bundled_headers.cu', 0, None, None)
    if status:
        raise ValueError(f'NVRTC program creation failed: {status}')
    try:
        options = (ctypes.c_char_p * 4)(
            b'--gpu-architecture=compute_89', b'--std=c++17',
            f'--include-path={include}'.encode(), f'--include-path={include / "cccl"}'.encode())
        status = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if status:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            raise ValueError(f'Bundled CUDA JIT header compilation failed: {log.value.decode()}')
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
    print('Bundled CUDA/CubeCL JIT headers compiled successfully without a GPU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda-root', type=Path, required=True)
    check(parser.parse_args().cuda_root)
