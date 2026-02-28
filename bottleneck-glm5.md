# tenferro-einsum vs strided-rs 性能差分析

## ベンチマーク結果 (1T, opt_flops)

特に差が大きいケース:

| Instance | 比率 (tenferro/strided) | 特徴 |
|----------|------------------------|------|
| lm_batch_likelihood_brackets_4_4d | 2.29x | 84テンソル、多数の小さなGEMM、batch次元あり |
| lm_batch_likelihood_sentence_3_12d | 1.90x | 38テンソル、batch次元あり |
| lm_batch_likelihood_sentence_4_4d | 2.29x | 84テンソル、batch次元あり |
| gm_queen5_5_3.wcsp | 1.96x | 160テンソル、多数の小さなcontraction |
| tensornetwork_permutation_* | 1.59x | 315-415テンソル、大規模contraction |
| bin_outer_product_4096 | 1.54x | outer product path (broadcast) |

共通パターン: **多数の小さなcontractionを連続実行するワークロードで性能差が拡大**

---

## 原因1: バッチ次元の反復処理

### strided-rs (bgemm_faer.rs:209-233)

```rust
// Fast path: when batch dims are contiguous for all operands, use pointer
// increments instead of MultiIndex carry-based iteration.
let fused_a = try_fuse_group(batch_dims, a_batch_strides);
let fused_b = try_fuse_group(batch_dims, b_batch_strides);
let fused_c = try_fuse_group(batch_dims, c_batch_strides);

if let (Some((total, a_step)), Some((_, b_step)), Some((_, c_step))) =
    (fused_a, fused_b, fused_c)
{
    let mut a_off = 0isize;
    let mut b_off = 0isize;
    let mut c_off = 0isize;
    for _ in 0..total {
        do_batch(a_off, b_off, c_off);
        a_off += a_step;
        b_off += b_step;
        c_off += c_step;
    }
} else {
    // Fallback: MultiIndex carry-based iteration
    let mut batch_iter = MultiIndex::new(batch_dims);
    while batch_iter.next().is_some() {
        let a_batch_off = batch_iter.offset(a_batch_strides);
        ...
    }
}
```

### tenferro-prims (cpu.rs:1033-1055)

```rust
// 常にcarry-based loop
let mut carried_all = true;
for ax in 0..nb {
    let dim = batch_dims[ax];
    let next = idx[ax] + 1;
    if next < dim {
        idx[ax] = next;
        a_off += a_batch[ax];
        b_off += b_batch[ax];
        c_off += c_batch[ax];
        carried_all = false;
        break;
    } else {
        idx[ax] = 0;
        let back = (dim as isize - 1).max(0);
        a_off -= back * a_batch[ax];  // reset and carry
        b_off -= back * b_batch[ax];
        c_off -= back * c_batch[ax];
    }
}
if carried_all {
    break;
}
```

### 影響

- `lm_batch_likelihood_*` は多数の小さなbatch GEMMを実行
- strided-rs: batch dimsがfuse可能な場合 **ポインタ加算のみ** (O(1) per batch)
- tenferro: **carry-basedでインデックス計算** (O(nb) per batch, nb = batch次元数)
- 小さなGEMMが多数回呼ばれる場合、このオーバーヘッドが累積

---

## 原因2: 次元フュージョン最適化の欠如

### strided-rs (util.rs)

```rust
/// Try to fuse a group of dimensions into a single stride.
/// Returns (total_size, stride) if the dimensions are contiguous.
pub fn try_fuse_group(dims: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    if dims.is_empty() {
        return Some((1, 0));
    }
    // Check if strides form a contiguous pattern
    // e.g., [1, d0, d0*d1, ...] for column-major
    ...
}
```

### tenferro-prims

この最適化が存在しない。各次元を個別に処理するため:
- batch iterationが遅い
- より多くのインデックス計算

---

## 原因3: 中間テンソル割り当てオーバーヘッド

### strided-rs (contiguous.rs)

```rust
pub fn prepare_input_view<T>(
    view: &StridedView<T>,
    n_group1: usize,
    n_group2: usize,
    ...
) -> Result<ContiguousOperand<T>> {
    // 1. Check if already contiguous -> zero-copy view
    // 2. If not, allocate from thread-local pool
    // 3. Copy data if needed
}
```

### tenferro-einsum (lib.rs)

```rust
fn fallback_pairwise_contraction<A, B>(...) {
    // 毎回 alloc_tensor_pooled() を呼び出し
    let mut temp = alloc_tensor_pooled::<A::Scalar>(&c_gemm_shape, memory_space);
    ...
    // さらに permute_or_copy() でコピーが発生する可能性
}
```

### 影響

- `gm_queen5_5_3.wcsp` は160テンソルを159ステップで縮約
- 各ステップで中間テンソル割り当てが発生
- tenferroはより多くのアロケーション/コピーを行っている可能性

---

## 原因4: outer product path の非効率性

### ベンチマーク結果

```
bin_outer_product_4096: tenferro 3.272ms vs strided-rs 2.124ms (1.54x)
```

### tenferro-einsum (lib.rs:1056-1150)

```rust
fn try_outer_elementwise_contraction<A, B>(...) {
    // 1. reshape to broadcast-compatible shapes
    let a_reshaped = a.reshape(&a_ext_shape)?;
    let b_reshaped = b.reshape(&b_ext_shape)?;
    
    // 2. broadcast (可能な限りゼロコピーだがstride計算が必要)
    let a_bcast = a_reshaped.broadcast(&canonical_shape)?;
    let b_bcast = b_reshaped.broadcast(&canonical_shape)?;
    
    // 3. ElementwiseMul primitive
    let desc = PrimDescriptor::ElementwiseMul;
    ...
}
```

### strided-rs

- GEMMのk=1パスとして直接処理
- faerの最適化されたmatmulを使用
- broadcast overheadなし

---

## 推奨される改善策

### 優先度1: バッチ反復のポインタ加算化

```rust
// tenferro-prims/src/cpu.rs に追加
fn try_fuse_batch_dims(
    batch_dims: &[usize],
    a_batch_strides: &[isize],
    b_batch_strides: &[isize],
    c_batch_strides: &[isize],
) -> Option<(usize, isize, isize, isize)> {
    // strided-rsのtry_fuse_groupと同様のロジック
}
```

期待効果: `lm_batch_likelihood_*` で 20-30% 改善

### 優先度2: 次元フュージョン

`strided_perm::try_fuse_group` を使用して、batch dimsだけでなくlo/sum/ro dimsもフュージョン。

期待効果: 全体的に 10-20% 改善

### 優先度3: 中間バッファの削減

- `fallback_pairwise_contraction` でのtemp tensor割り当てを削減
- より積極的なゼロコピーview活用
- 既存の `alloc_tensor_pooled` の使用を再検討

期待効果: `gm_queen5_5_3.wcsp` で 10-15% 改善

### 優先度4: outer product pathの最適化

- `try_outer_elementwise_contraction` をGEMM k=1パスに変更
- または、より効率的なbroadcast実装

期待効果: `bin_outer_product_4096` で 30% 改善

---

## 参考ファイル

- `strided-rs/strided-einsum2/src/bgemm_faer.rs` - バッチ反復最適化
- `strided-rs/strided-einsum2/src/util.rs` - try_fuse_group
- `strided-rs/strided-einsum2/src/contiguous.rs` - メモリ準備
- `tenferro-rs/tenferro-einsum/src/lib.rs` - fallback_pairwise_contraction
- `tenferro-rs/tenferro-prims/src/cpu.rs` - execute_batched_gemm_f64
