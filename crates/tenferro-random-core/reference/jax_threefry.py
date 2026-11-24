"""JAX values for the Threefry2x32-20 known-answer tests (tenferro-rs#2714).

Prints the block function, `split`, `fold_in` and the key of a 64-bit seed as
computed by JAX, for `tests/threefry.rs`. Recorded with JAX 0.12.1 (CPU),
where `jax_threefry_partitionable` is on by default:

    python3 reference/jax_threefry.py

Its output with JAX 0.12.1 (CPU), copied into the tests:

    jax 0.12.1
    block key=0x13198a2e, 0x03707344 counter=0x243f6a88, 0x85a308d3 -> 0xc4923a9c, 0x483df7a0
    block key=0x00000000, 0x00000007 counter=0x00000000, 0x00000005 -> 0xd5917e25, 0x7415e54e
    block key=0xdeadbeef, 0x01234567 counter=0xffffffff, 0x00000001 -> 0x0c8f26c9, 0xe6d6c435
    seed 0x0 -> 0x00000000, 0x00000000
    seed 0x2a -> 0x00000000, 0x0000002a
    seed 0x123456789abcdef -> 0x01234567, 0x89abcdef
    split(seed 42, 4) -> 0x6d3e048f, 0x1022172d, 0x03d7b32d, 0xadd083f4, 0x92fb20ea, 0x0f38d913, 0xbad56946, 0x354ba891
    fold_in(seed 42, 0x0) -> 0x6d3e048f, 0x1022172d
    fold_in(seed 42, 0x1) -> 0x03d7b32d, 0xadd083f4
    fold_in(seed 42, 0x7) -> 0x97d0552f, 0x51bf719f
    fold_in(seed 42, 0xffffffff) -> 0x8ef6c48d, 0xe7eb97ca

With `jax_threefry_partitionable` off (the pre-0.5.0 default), `split` gives
different values: that mode encrypts the two halves of a flattened counter
array, so key `i` depends on the split size (`split(seed 42, 2)` is then
`[0x92fb20ea, 0xdb4c98db], [0x0f38d913, 0x0ff68959]`). `fold_in` is the same in
both modes.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
from jax.extend.random import threefry_2x32

assert jax.config.jax_threefry_partitionable, "the layout assumes partitionable Threefry"
print(f"jax {jax.__version__}")


def words(array):
    return ", ".join(f"0x{int(w):08x}" for w in jnp.ravel(array))


def raw(key):
    return random.key_data(key) if jnp.issubdtype(key.dtype, jax.dtypes.prng_key) else key


# The block function on (counter, key) pairs; threefry_2x32 encrypts the
# counter halves [x0..., x1...] of its flattened count.
for key, counter in (
    ((0x13198A2E, 0x03707344), (0x243F6A88, 0x85A308D3)),
    ((0, 7), (0, 5)),
    ((0xDEADBEEF, 0x01234567), (0xFFFFFFFF, 0x00000001)),
):
    out = threefry_2x32(jnp.array(key, jnp.uint32), jnp.array(counter, jnp.uint32))
    print(f"block key={words(jnp.array(key))} counter={words(jnp.array(counter))} -> {words(out)}")

# The key of a 64-bit seed is [seed >> 32, seed & 0xFFFFFFFF].
for seed in (0, 42, 0x0123_4567_89AB_CDEF):
    print(f"seed {seed:#x} -> {words(raw(random.PRNGKey(jnp.uint64(seed))))}")

key = random.PRNGKey(jnp.uint64(42))
print(f"split(seed 42, 4) -> {words(raw(random.split(key, 4)))}")
for data in (0, 1, 7, 0xFFFFFFFF):
    print(f"fold_in(seed 42, {data:#x}) -> {words(raw(random.fold_in(key, jnp.uint32(data))))}")
