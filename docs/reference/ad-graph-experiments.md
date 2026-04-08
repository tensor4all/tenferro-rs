# Graph Transform Experiments for `v2 AD Architecture`

> **Note:** This document is an early experiment record. The `merge` operation
> used here was later separated into `resolve` (logical view, cheap) and
> `materialize_merge` (physical flattening, once before compile). Higher-order
> AD requires only `resolve`, not physical merge. The `Scale(c, t)` and
> `AddLin(u, v)` primitives correspond to `Mul(c, t) mode=Linear {
> active_mask=[fixed, active] }` and `Add(u, v) mode=Linear {
> active_mask=[active, active] }` in the final design. The `Dup` primitive
> used in these experiments has been removed from the final design;
> fan-out accumulation is now handled internally by `tidu::transpose`
> (JAX-style bucketing by `GlobalValKey`). See `../architecture/ad-pipeline.md` for
> the current design.

## Scope

This document records manual graph-construction experiments for an earlier
version of the design:

- `differentiate` produces a **linear graph fragment** with external references
- `merge` resolves external references and must be inserted before the next `differentiate`
- `transpose` reverses **tangent I/O**
- reverse mode requires **cotangent accumulation** when multiple reverse contributions land on the same tangent node

The goal is not to revise wording, but to test whether the procedure works when node names and transformation steps are written explicitly.

---

## Conventions

Node prefixes:

- `p*`: primal graph
- `l*`: first-order linear fragment
- `t*`: transpose fragment
- `q*`: second-order linear fragment
- `r*`, `s*`: later fragments

We use these scalar primitive rules:

### Primal rules

- `Add(a, b)`  
  `d = AddLin(da, db)`

- `Mul(a, b)`  
  `d = AddLin(Scale(b, da), Scale(a, db))`

- `Exp(a)`  
  `d = Scale(out, da)`

### Linear primitive rules

- `Scale(c, t)`  
  `d = AddLin(Scale(t, dc), Scale(c, dt))`

- `AddLin(u, v)`  
  `d = AddLin(du, dv)`

- `Dup(u)`  
  `d = Dup(du)`

### Transpose rules

- `Scale(c, ·)  ↔  Scale(c, ·)` (real)
- `Scale(c, ·)  ↔  Scale(Conj(c), ·)` (complex)
- `AddLin  ↔  Dup`
- `Dup     ↔  AddLin`

### Important implementation rule

After local transpose rules are applied, reverse cotangent contributions that land on the **same original tangent node** must be accumulated with `AddLin`.

---

## Experiment 1 — Baseline: `f(x) = x + x`

### Primal graph `P`

```text
p0 = Input(x)
p1 = Add(p0, p0)                  # y
```

### First differentiation

`differentiate(P, output=p1, wrt=x with tangent l0=dx)`

```text
l0 = Input(dx)
l1 = AddLin(l0, l0)               # dy
```

### Merge

```text
p0 = Input(x)
p1 = Add(p0, p0)

l0 = Input(dx)
l1 = AddLin(l0, l0)
```

### Transpose

Seed `t0 = ct_y`.

```text
t0 = Input(ct_y)
t1, t2 = Dup(t0)                  # transpose of AddLin
t3 = AddLin(t1, t2)               # accumulation to the same tangent input l0
```

Result:

```text
ct_dx = t3 = ct_y + ct_y = 2 * ct_y
```

### What this checks

`AddLin ↔ Dup` alone is not enough. Reverse mode needs a separate **accumulation phase** keyed by original tangent-node identity.

---

## Experiment 2 — Baseline: `f(x, y) = x * y`

### Primal graph `P`

```text
p0 = Input(x)
p1 = Input(y)
p2 = Mul(p0, p1)                  # z
```

### First differentiation

`differentiate(P, output=p2, wrt=[x -> l0, y -> l1])`

```text
l0 = Input(dx)
l1 = Input(dy)

l2 = Scale(p1, l0)                # y * dx
l3 = Scale(p0, l1)                # x * dy
l4 = AddLin(l2, l3)               # dz
```

### Merge

```text
p0 = Input(x)
p1 = Input(y)
p2 = Mul(p0, p1)

l0 = Input(dx)
l1 = Input(dy)
l2 = Scale(p1, l0)
l3 = Scale(p0, l1)
l4 = AddLin(l2, l3)
```

### Transpose

Seed `t0 = ct_z`.

```text
t0 = Input(ct_z)

t1, t2 = Dup(t0)                  # transpose of AddLin
t3 = Scale(p1, t1)                # ct_dx = y * ct_z
t4 = Scale(p0, t2)                # ct_dy = x * ct_z
```

No accumulation is needed because contributions land on different tangent inputs (`dx`, `dy`).

---

## Experiment 3 — Baseline (complex): `f(z) = c * z`

### Primal graph `P`

```text
p0 = Input(c)                     # complex coefficient
p1 = Input(z)
p2 = Mul(p0, p1)                  # w
```

### First differentiation

`differentiate(P, output=p2, wrt=z with tangent l0=dz)`

```text
l0 = Input(dz)
l1 = Scale(p0, l0)                # dw = c * dz
```

### Merge

```text
p0 = Input(c)
p1 = Input(z)
p2 = Mul(p0, p1)

l0 = Input(dz)
l1 = Scale(p0, l0)
```

### Transpose

Seed `t0 = ct_w`.

```text
t0 = Input(ct_w)
t1 = Scale(Conj(p0), t0)          # ct_dz = Conj(c) * ct_w
```

This confirms: `Conj` appears in `transpose`, not in `differentiate`.

---

## Experiment 4 — `f(x) = x * x`

This is the smallest example that tests:

- duplicated primal input
- duplicated first-order linear input
- transpose accumulation
- second-order rules for `Scale`
- all four second-order mode families

### 4.1 Primal graph `P`

```text
p0 = Input(x)
p1 = Mul(p0, p0)                  # y = x^2
```

---

### 4.2 First differentiation

`differentiate(P, output=p1, wrt=x with tangent l0=dx1)`

`Mul(p0, p0)` linearizes to:

```text
Scale(p0, l0) + Scale(p0, l0)
```

Because both terms are identical, `op_cache` can unify them.

```text
l0 = Input(dx1)
l1 = Scale(p0, l0)
l2 = AddLin(l1, l1)               # dy = 2*x*dx1
```

### 4.3 Merge `M1 = merge(P, L1)`

```text
p0 = Input(x)
p1 = Mul(p0, p0)

l0 = Input(dx1)
l1 = Scale(p0, l0)
l2 = AddLin(l1, l1)
```

---

## 4.4 FoF (forward over forward)

Differentiate `M1` again with respect to `x`.

`differentiate(M1, output=l2, wrt=x with tangent q0=dx2)`

### Reachable nodes and derivatives

- `l1 = Scale(p0, l0)`  
  `dl1 = Scale(l0, q0)` because `dp0 = q0`, `dl0 = 0`

- `l2 = AddLin(l1, l1)`  
  `dl2 = AddLin(dl1, dl1)`

So the new fragment is:

```text
q0 = Input(dx2)
q1 = Scale(l0, q0)
q2 = AddLin(q1, q1)               # d2y = 2 * dx1 * dx2
```

### Merge `M2 = merge(M1, Q1)`

```text
p0 = Input(x)
p1 = Mul(p0, p0)

l0 = Input(dx1)
l1 = Scale(p0, l0)
l2 = AddLin(l1, l1)

q0 = Input(dx2)
q1 = Scale(l0, q0)
q2 = AddLin(q1, q1)
```

**FoF result**:

```text
q2 = 2 * dx1 * dx2
```

If `dx1 = dx2 = 1`, then `d²f/dx² = 2`.

---

## 4.5 First reverse graph (gradient seed)

Transpose the first-order linear graph inside `M1`.

Seed `t0 = ct1`.

```text
t0 = Input(ct1)

t1, t2 = Dup(t0)                  # reverse of l2 = AddLin(l1, l1)
t3 = AddLin(t1, t2)               # accumulation onto l1
t4 = Scale(p0, t3)                # reverse of l1 = Scale(p0, l0)
```

`ct_x = t4 = 2 * x * ct1`

Merged gradient graph:

```text
p0 = Input(x)
p1 = Mul(p0, p0)

t0 = Input(ct1)
t1, t2 = Dup(t0)
t3 = AddLin(t1, t2)
t4 = Scale(p0, t3)                # gradient output
```

---

## 4.6 FoR (forward over reverse)

Differentiate the gradient graph with respect to `x`.

`differentiate(G_grad, output=t4, wrt=x with tangent q0=dx2)`

Only `t4 = Scale(p0, t3)` depends on `x`.

```text
q0 = Input(dx2)
q1 = Scale(t3, q0)                # d(2*x*ct1) = 2*ct1*dx2
```

**FoR result**:

```text
q1 = 2 * ct1 * dx2
```

With `ct1 = 1`, `dx2 = 1`, this is `2`.

---

## 4.7 RoF (reverse over forward)

Transpose the FoF linear graph `Q1`.

Recall:

```text
q0 = Input(dx2)
q1 = Scale(l0, q0)
q2 = AddLin(q1, q1)
```

Seed `r0 = ct2`.

```text
r0 = Input(ct2)

r1, r2 = Dup(r0)                  # reverse of q2
r3 = AddLin(r1, r2)               # accumulation onto q1
r4 = Scale(l0, r3)                # reverse of q1
```

**RoF result**:

```text
r4 = 2 * dx1 * ct2
```

With `dx1 = 1`, `ct2 = 1`, this is `2`.

---

## 4.8 RoR (reverse over reverse)

Transpose the FoR linear graph.

Recall FoR graph:

```text
q0 = Input(dx2)
q1 = Scale(t3, q0)
```

Seed `s0 = ct2`.

```text
s0 = Input(ct2)
s1 = Scale(t3, s0)
```

Since `t3 = ct1 + ct1 = 2*ct1`:

**RoR result**:

```text
s1 = 2 * ct1 * ct2
```

With `ct1 = ct2 = 1`, this is `2`.

---

## 4.9 Summary for `x^2`

| Mode | Output |
|---|---|
| FoF | `2 * dx1 * dx2` |
| FoR | `2 * ct1 * dx2` |
| RoF | `2 * dx1 * ct2` |
| RoR | `2 * ct1 * ct2` |

This example validates all four second-order mode families.

---

## Experiment 5 — `f(x) = exp(a * x)`

This is the main chain-rule example. It tests:

- external references
- repeated `merge`
- higher-order `Scale` behavior
- all second-order modes
- third order

### 5.1 Primal graph `P`

```text
p0 = Input(x)
p1 = Input(a)
p2 = Mul(p0, p1)
p3 = Exp(p2)                      # y = exp(a*x)
```

---

### 5.2 First differentiation

`differentiate(P, output=p3, wrt=x with tangent l0=dx1)`

```text
l0 = Input(dx1)
l1 = Scale(p1, l0)                # dp2 = a * dx1
l2 = Scale(p3, l1)                # dy  = exp(a*x) * a * dx1
```

### 5.3 Merge `M1 = merge(P, L1)`

```text
p0 = Input(x)
p1 = Input(a)
p2 = Mul(p0, p1)
p3 = Exp(p2)

l0 = Input(dx1)
l1 = Scale(p1, l0)
l2 = Scale(p3, l1)
```

---

## 5.4 FoF

Differentiate `M1` again with respect to `x`.

`differentiate(M1, output=l2, wrt=x with tangent q0=dx2)`

### Reachable nodes

- `p2 = Mul(p0, p1)`  
  `q1 = Scale(p1, q0)`

- `p3 = Exp(p2)`  
  `q2 = Scale(p3, q1)`

- `l2 = Scale(p3, l1)`  
  since `dl1 = 0` with respect to `x` in this pass, only the coefficient moves:
  `q3 = Scale(l1, q2)`

Fragment:

```text
q0 = Input(dx2)
q1 = Scale(p1, q0)
q2 = Scale(p3, q1)
q3 = Scale(l1, q2)
```

Result:

```text
q3 = a^2 * exp(a*x) * dx1 * dx2
```

---

## 5.5 First reverse graph (gradient seed)

Transpose the first-order linear graph.

Seed `t0 = ct1`.

```text
t0 = Input(ct1)
t1 = Scale(p3, t0)                # reverse of l2
t2 = Scale(p1, t1)                # reverse of l1
```

So:

```text
t2 = a * exp(a*x) * ct1
```

Merged gradient graph:

```text
p0 = Input(x)
p1 = Input(a)
p2 = Mul(p0, p1)
p3 = Exp(p2)

t0 = Input(ct1)
t1 = Scale(p3, t0)
t2 = Scale(p1, t1)                # gradient output
```

---

## 5.6 FoR

Differentiate the gradient graph with respect to `x`.

`differentiate(G_grad, output=t2, wrt=x with tangent q0=dx2)`

### Reachable nodes

```text
q0 = Input(dx2)
q1 = Scale(p1, q0)                # dp2
q2 = Scale(p3, q1)                # dp3
q3 = Scale(t0, q2)                # dt1
q4 = Scale(p1, q3)                # dt2
```

Result:

```text
q4 = a^2 * exp(a*x) * ct1 * dx2
```

With `ct1 = 1`, this is the Hessian-vector product.

---

## 5.7 RoF

Transpose the FoF graph.

Recall:

```text
q0 = Input(dx2)
q1 = Scale(p1, q0)
q2 = Scale(p3, q1)
q3 = Scale(l1, q2)
```

Seed `r0 = ct2`.

```text
r0 = Input(ct2)
r1 = Scale(l1, r0)                # reverse of q3
r2 = Scale(p3, r1)                # reverse of q2
r3 = Scale(p1, r2)                # reverse of q1
```

Result:

```text
r3 = a^2 * exp(a*x) * dx1 * ct2
```

---

## 5.8 RoR

Transpose the FoR graph.

Recall FoR graph:

```text
q0 = Input(dx2)
q1 = Scale(p1, q0)
q2 = Scale(p3, q1)
q3 = Scale(t0, q2)
q4 = Scale(p1, q3)
```

Seed `s0 = ct2`.

```text
s0 = Input(ct2)
s1 = Scale(p1, s0)                # reverse of q4
s2 = Scale(t0, s1)                # reverse of q3
s3 = Scale(p3, s2)                # reverse of q2
s4 = Scale(p1, s3)                # reverse of q1
```

Result:

```text
s4 = a^2 * exp(a*x) * ct1 * ct2
```

---

## 5.9 Summary for `exp(a*x)`

| Mode | Output |
|---|---|
| FoF | `a^2 * exp(a*x) * dx1 * dx2` |
| FoR | `a^2 * exp(a*x) * ct1 * dx2` |
| RoF | `a^2 * exp(a*x) * dx1 * ct2` |
| RoR | `a^2 * exp(a*x) * ct1 * ct2` |

When the seeds are all `1`, every mode yields:

```text
a^2 * exp(a*x)
```

---

## 5.10 Third order for `exp(a*x)`

Differentiate once more after FoF.

We start from the FoF fragment:

```text
q0 = Input(dx2)
q1 = Scale(p1, q0)
q2 = Scale(p3, q1)
q3 = Scale(l1, q2)
```

Differentiate with respect to `x` using `r0 = dx3`.

### Reachable nodes

- `p2 = Mul(p0, p1)`  
  `r1 = Scale(p1, r0)`

- `p3 = Exp(p2)`  
  `r2 = Scale(p3, r1)`

- `q2 = Scale(p3, q1)`  
  `q1` is fixed in this pass, so:
  `r3 = Scale(q1, r2)`

- `q3 = Scale(l1, q2)`  
  `l1` is fixed in this pass, so:
  `r4 = Scale(l1, r3)`

Thus:

```text
r0 = Input(dx3)
r1 = Scale(p1, r0)
r2 = Scale(p3, r1)
r3 = Scale(q1, r2)
r4 = Scale(l1, r3)
```

Result:

```text
r4 = a^3 * exp(a*x) * dx1 * dx2 * dx3
```

This confirms that repeated
`differentiate -> merge -> differentiate -> merge -> ...`
continues to work at third order for this example.

---

## Experiment 6 — Mixed partial: `f(x, y) = x * y`

This checks symmetry of mixed second derivatives.

### 6.1 First do `x`, then `y`

Primal graph:

```text
p0 = Input(x)
p1 = Input(y)
p2 = Mul(p0, p1)
```

First differentiation with respect to `x`:

```text
l0 = Input(dx)
l1 = Scale(p1, l0)
```

Merge:

```text
p0 = Input(x)
p1 = Input(y)
p2 = Mul(p0, p1)

l0 = Input(dx)
l1 = Scale(p1, l0)
```

Second differentiation with respect to `y`:

```text
q0 = Input(dy)
q1 = Scale(l0, q0)
```

Result:

```text
q1 = dx * dy
```

### 6.2 Reverse the order: first `y`, then `x`

First differentiation with respect to `y`:

```text
m0 = Input(dy)
m1 = Scale(p0, m0)
```

Merge and differentiate with respect to `x`:

```text
n0 = Input(dx)
n1 = Scale(m0, n0)
```

Result:

```text
n1 = dx * dy
```

### Conclusion

The mixed partial agrees in both orders for this example.

---

## Consolidated findings

### 1. `merge` is semantically required before the next `differentiate`

This is not just an optimization pass.
Without `merge`, external references remain dangling and higher-order tracing cannot see the upstream primal dependency.

The correct repeated pattern is:

```text
differentiate -> merge -> differentiate -> merge -> ...
```

---

### 2. `transpose` needs a distinct accumulation phase

Local transpose rules are not sufficient.

Example: `x + x` and `x * x` both generate reverse contributions that land on the same tangent node. Those must be combined by `AddLin`.

Practical algorithm:

```text
transpose(fragment):
  1. place cotangent seeds on the linear outputs
  2. traverse linear ops in reverse topological order
  3. emit reverse contributions by local transpose rule
  4. bucket contributions by original tangent-node identity
  5. combine each bucket with AddLin
```

---

### 3. The coefficient of `Scale(coeff, tangent)` is not necessarily “primal”

At higher order, the coefficient can be:

- a primal value
- an earlier tangent value
- a reverse-produced cotangent-side residual

What matters is not “primal vs tangent” globally, but:

> in the current transform, which operand is treated as fixed, and which operand is the active linear variable?

This is the right interpretation of `Scale`.

---

### 4. `op_cache` / CSE matters immediately

Even in `x^2`, the first linearization creates two identical `Scale(p0, dx1)` terms.
Canonicalization or caching keeps the graph smaller and makes later transpose behavior easier to reason about.

---

### 5. The current procedure works on the tested examples

The experiments succeeded for:

- `x + x`
- `x * y`
- `c * z`
- `x^2`
- `exp(a*x)`
- mixed partials of `x*y`
- third order of `exp(a*x)`

and all second-order mode families:

- FoF
- FoR
- RoF
- RoR

---

## Minimal “golden test” set suggested by the experiments

1. `x + x`  
   checks transpose accumulation

2. `x * y`  
   checks binary linearization and distinct reverse sinks

3. `c * z`  
   checks `Conj` placement only in transpose

4. `x^2`  
   checks duplication, CSE, and all second-order modes

5. `exp(a*x)`  
   checks chain rule, nonconstant coefficients, and repeated merge

6. `x*y` mixed partial  
   checks order symmetry for a simple multivariate case

7. `exp(a*x)` third order  
   checks repeated higher-order closure

