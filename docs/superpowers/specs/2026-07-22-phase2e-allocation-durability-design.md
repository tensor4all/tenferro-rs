# Phase 2E Allocation Durability Design

## Scope

Harden the Phase 2E allocation campaign against forged terminal evidence,
initialization crashes, concurrent attempts, and ambiguous JSON scalar types.
The existing sealed executable launch, build provenance, and atomic
finalization contracts remain unchanged.

## Durable ownership and serialization

`orchestrator.lock`, located in the same canonical outer evidence root as
`evidence-ledger.json`, is the stable shared lock authority. The allocation
runner opens it without following symlinks, proves it is the expected canonical
regular file, acquires an exclusive `flock`, and holds that lock from before
recovery or artifact-root reservation until terminal cleanup. It never locks
the replaceable ledger inode. Cleanup errors are suppressed behind an active
control exception, and every descriptor is closed exactly once.

Allocation ledger attempts gain exact artifact ownership fields: canonical
root path, device, inode, and ownership state. Timing attempts carry the same
schema with `None` identities and `NOT_APPLICABLE` state. Allocation starts by
persisting a `RESERVED` attempt, creates the root exclusively, persists its
`BOUND` identity, then writes the canonical `RUNNING` manifest. A different
root cannot run or recover the same attempt.

## Initialization recovery

A crash before reservation leaves no state. A crash with only a `RESERVED`
attempt is closed as validity `INCONCLUSIVE` without touching an unproven
artifact root. A `BOUND` attempt is recoverable only when the canonical path
still names the recorded directory identity. An empty bound root or a bound
root with a canonical `RUNNING` prefix is terminalized as strict
`INCONCLUSIVE`, without launching probes. Atomic-write pre-commit and
committed-then-raise states are classified by rereading exact canonical state.

## Evidence validation

All persisted allocation, build, probe, ledger, stage, and marker JSON uses a
single strict decoder. It rejects duplicate keys, non-finite numbers,
noncanonical bytes, and scalar subclasses such as booleans where exact integers
are required.

The allocation validator defines exact schemas for `RUNNING`, terminal
`COMPLETE`, and terminal `INCONCLUSIVE`. It reconstructs the 168-entry canonical
case/order/position/role sequence, validates contiguous launch indices and
exact record schemas, checks within-case binary consistency, recomputes every
candidate-versus-baseline allocation inequality and the final PASS/FAIL gate,
and binds embedded build, lock, and executable identities to authoritative
validated inputs. An inconclusive terminal must be one exact canonical prefix,
must retain an exact failure location and reason, and cannot select a result.
Stage, marker, and published inode/digest checks occur only after this semantic
validation.

## Verification

Tests cover forged closed PASS evidence, the full adversarial mutation matrix,
all durable initialization crash windows, pre/post-commit atomic-write states,
control-exception identity, and a real two-process race using two different
artifact roots for one attempt. Existing focused allocation/build tests and the
shared protocol plus Phase 1 suites must remain green.
