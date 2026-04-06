# Oracle Replay CI Design

**Date:** 2026-04-06
**Goal:** Replay the full `tensor-ad-oracles` database (171 ops, 1000+ cases) as always-on integration tests in `tenferro`, validated against PyTorch reference derivatives.

## Architecture

### Test Location

```
tenferro/tests/oracle_replay/
├── main.rs          # Test entrypoint + summary output
├── decode.rs        # JSONL parsing, row-major → col-major tensor conversion
├── db.rs            # cases/ directory enumeration, case registry
├── dispatch.rs      # op name → tenferro API mapping
└── compare.rs       # Tolerance-aware tensor comparison
```

Tests live in the `tenferro` facade crate because it has access to all public APIs: TracedTensor, linalg free functions, einsum, and elementwise ops.

### Data Flow

1. `db.rs` scans `third_party/tensor-ad-oracles/cases/*/` and enumerates JSONL files
2. `decode.rs` parses each record's `inputs`, `probes`, `comparison` fields
3. Tensors are converted from row-major flat JSON arrays to col-major `Tensor`
4. `dispatch.rs` maps the `op` field to tenferro API calls:
   - Scalar ops (`sin`, `exp`, `add`, ...): `TracedTensor` methods
   - Linalg ops (`svd`, `qr`, `eigh`, ...): free functions
   - Unimplemented ops: skip + count
5. For each probe:
   - **Forward**: compare tenferro execution result against oracle expected output
   - **JVP**: compare `TracedTensor::jvp()` result against `pytorch_ref.jvp`
   - **VJP**: compare `TracedTensor::grad()`/`vjp()` against `pytorch_ref.vjp`
   - **HVP**: compare `grad().jvp()` (FoR) against `pytorch_ref.hvp`
6. `compare.rs` uses per-case tolerances from `comparison.first_order` and `comparison.second_order`

### Skip Policy

| Condition | Handling |
|-----------|----------|
| Unimplemented op (det, inv, lu, ...) | Skip, list in summary |
| Unsupported dtype (f32 linalg, etc.) | Skip, show count in summary |
| `expected_behavior: "error"` | Skip treated as expected (gauge-ill-defined, etc.) |

### Observable Mapping

| Oracle observable | tenferro implementation |
|---|---|
| `identity` | Compare op outputs directly |
| `svd_s` | `svd()` → compare sigma only |
| `svd_u_abs` / `svd_vh_abs` | `svd()` → abs(U), abs(Vt) to remove gauge ambiguity |
| `svd_uvh_product` | `svd()` → reconstruct U @ diag(S) @ Vt |
| `eigh_values_vectors_abs` | `eigh()` → eigenvalues + abs(eigenvectors) |

### Summary Output

```
Oracle replay: 847 passed, 0 failed, 144 skipped (dtype), 53 skipped (unimplemented op), 12 expected error
Unimplemented ops: det, inv, pinv, lu, cond, norm, ...
```

### Dependencies

Add to `tenferro/Cargo.toml` `[dev-dependencies]`:
```toml
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
```

### CI Integration

Oracle replay runs within the existing `test-workspace` CI job via `cargo nextest run --workspace --release`. No separate job needed — this avoids duplicate builds.

### Oracle JSONL Schema (key fields)

```
{
  "case_id": "qr_f64_identity_001",
  "op": "qr",
  "dtype": "float64",
  "family": "identity",
  "expected_behavior": "success",
  "comparison": {
    "first_order": { "kind": "allclose", "rtol": 0.001, "atol": 1e-6 },
    "second_order": { "kind": "allclose", "rtol": 0.0001, "atol": 1e-6 }
  },
  "inputs": { "a": { "dtype": "float64", "shape": [5, 5], "order": "row_major", "data": [...] } },
  "observable": { "kind": "identity" },
  "probes": [{
    "probe_id": "p0",
    "direction": { "a": {...} },
    "cotangent": { "output_0": {...} },
    "pytorch_ref": { "jvp": {...}, "vjp": {...}, "hvp": {...} },
    "fd_ref": { "jvp": {...}, "hvp": {...} }
  }]
}
```

Tensors: row-major flat arrays. Real dtypes use flat floats; complex dtypes use `[re, im]` pairs.
