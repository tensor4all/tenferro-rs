# Retire the graph AD emitter, retain its shared vocabulary

Issue: [#2339](https://github.com/tensor4all/tenferro-rs/issues/2339).

## Decisions

The internal-ops package is imported as `tenferro_ops`. Its AD directory was
not wholly dead: semantic AD and linalg use its rule vocabulary and residual
metadata, and runtime uses its shape/global-metadata context.

The split therefore retires only the unreachable core derivative emitter and
its registry/dispatch tests. Shared contracts keep their existing public paths;
definitions mixed with emitter code move to a private `primitive_rules` owner.
Context/support helpers stay in place. The residual lookup preserves all 52
pre-retirement declarations without keeping formula function pointers.

## Verification conclusions and constraints

Context, support and residual contract tests remain, while existing semantic
AD/linalg numerical tests are unchanged. Local default and AD-enabled tests,
the no-default internal-ops configuration, and the workspace build pass.
This does not claim one-for-one semantic replacements for the removed debug
residual-access detector or registry-payload mismatch checks.

The removed graph entry points are a breaking API removal; the workspace and
internal dependency requirements move from published 0.8.0 to 0.9.0. Publishing,
tagging and merging are not part of this change.
