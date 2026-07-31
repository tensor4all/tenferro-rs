# Issue #1546: Faer dot-general policy audit

Date: 2026-07-31

## Scope

Audit the reported `Par::rayon(0)` bypass in ordinary CPU Faer
`dot_general`, including the cross-repository `strided-einsum2` dependency and
the `CpuContext` thread-policy boundary.

## Findings

The issue body describes an older route that is not present at the current
`main` revision. The active `tenferro-cpu` implementation uses its local
validated `FaerGemm` provider. `execute_faer_request_typed` obtains
`context.faer_parallelism()`, and `CpuExecutionContext::faer_parallelism`
returns `faer::Par::Seq` for one-thread or non-inner execution and explicit
`faer::Par::rayon(n)` for bounded inner Rayon execution. The enclosing session
also enters `CpuExecutionContext::with_native_parallelism`, so the Faer policy
does not inherit an unrelated ambient Rayon degree.

Focused tests cover the policy mapping, native execution scope, and the source
contract forbidding ambient/ad-hoc policy selection.

## Remediation

The current tree still carried a stale `strided-einsum2` workspace dependency,
optional CPU features, feature-contract assertions, active design wording, and
rules text. Those declarations were reintroduced by the later strided pin
update even though the tenferro-cpu Faer/BLAS contraction implementation no
longer uses the crate. This change removes that unused build graph and updates
the active contract text to describe the current tenferro-owned Faer provider.

No strided-rs code change is required for #1546, and no `Par::rayon(0)` call is
introduced in tenferro-cpu.

## Verification

- Focused Faer policy mapping test.
- Focused native-policy source-contract test.
- Focused active documentation/rules contract test.
- Full package and workspace checks are run on the committed PR head.
