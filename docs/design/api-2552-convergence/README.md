# The #2552 convergence record

Four documents and the ruling that came out of them, kept together because the decision is only
checkable against what the two models actually wrote. Nothing here is a specification; the rule the
decision produced lives in `../api-consolidation-policy.md` as R8 (and the amended R3).

| file | what it is |
|---|---|
| `fable-proposal.md` | Claude Fable 5.2's independent design, written from the issue thread and the three entry signatures, before seeing any other proposal. |
| `astra-proposal.md` | GPT Astra's independent design, same prompt, same material, written in parallel. |
| `fable-crossread.md` | Fable's round-2 document: after reading Astra's proposal. It concedes the carrier and holds five semantic points. |
| `astra-crossread.md` | Astra's round-2 document: after reading Fable's proposal. It concedes the carrier the other way and holds its own. |
| `ruling.md` | The decision, the compile evidence that decided the one open question, and the five overclaims an adversarial review of the ruling found. |

## What the reader should take from the pair, not from either alone

**The two crossed.** Each conceded the carrier to the other, so both round-2 documents open with a
convergence they do not actually share: `fable-crossread.md` §4 proposes the plain argument,
`astra-crossread.md` §4 proposes the `ExecutionRequest` wrapper. "Both models converged" was never
true here, and the ruling had to decide it. A later reader who reads only one of the two will think
the question was settled by agreement.

**The compile question decided it, not the argument.** `ruling.md` carries the four-file check, and
the check is narrower than a first reading suggests: the wrapper **as Fable wrote it**
(`fable-proposal.md:16`, `R: ?Sized = dyn Engine + 'a`) compiles. What does not compile is the naive
form without that default — `dyn Engine` is invariant in `R` and its default object lifetime is
`'static` — which is a pitfall a reader would meet if they wrote the type themselves, not a defect of
the submitted proposal. The decision therefore rests on the second half of the check: with the
correct default in place, a helper that holds `&mut dyn Engine` and calls twice needs `reborrow()` on
all but the last call, while the plain argument compiles that helper untouched. The wrapper's other
claimed advantage — that sugar cannot drop the policy — is a lint setting (`deny(unused_variables)`),
not a type. And the wrapper's claimed advantage — that sugar cannot drop the policy — is a lint
setting (`deny(unused_variables)`), not a type.

**The scope sentence was wrong in all three documents.** Both proposals and the first ruling said
"every public entry that executes numerical work". The repository holds **149** functions taking an
execution resource (`tenferro-linalg` 73, `tenferro-einsum` 29, `tenferro-ad` 22, `tenferro-cpu` 9,
`tenferro-runtime` 8, `tenferro-fft` 8), most of them internal wiring. R8 is scoped to the operation
entries; the count and the scope correction are the reason it says so.

## Reading order

1. `ruling.md` — the decision and the evidence.
2. `fable-crossread.md` §1 and `astra-crossread.md` §1 — the eleven differences, which is the most
   useful part of the round: each row names what breaks in the other design.
3. The two proposals, only if you need the un-merged versions.
