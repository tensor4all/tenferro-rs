"""JAX values for the Threefry2x32-20 known-answer tests (tenferro-rs#2714).

Prints the block function, `split`, `fold_in` and the key of a 64-bit seed as
computed by JAX, for `tests/threefry.rs`. Recorded with JAX 0.12.1 (CPU),
where `jax_threefry_partitionable` is on by default:

    python3 reference/jax_threefry.py
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
