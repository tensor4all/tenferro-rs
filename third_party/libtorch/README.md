# LibTorch Benchmark Cache

This directory is the default repo-local cache for the optional LibTorch C++
CPU benchmark baseline.

The benchmark script stores downloaded LibTorch ZIP files and extracted
libraries here by default. These files are intentionally ignored by git because
they are large binary artifacts. When running from a linked git worktree, the
script resolves the main worktree with `git worktree list` and uses this
directory there, so each temporary worktree does not download its own copy.

Override the location with `TENFERRO_BENCH_DEPS_DIR` or point directly at an
existing LibTorch install with `LIBTORCH_DIR`.
