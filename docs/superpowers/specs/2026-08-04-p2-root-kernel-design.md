# P2 Root Kernel Design

## Scope

This phase implements only the private root-resource owner kernel described by
the latest #1558 amendment. It starts from the accepted Task 2 candidate
`e938ed4b55f300bfcfd41d8371a1da0eeeb0218b` and activates only the P2
`p2-root-claims` ledger row.

The kernel owns one provider allocation through one `Arc<RootResource>` and
one non-`Clone` root-bound claim. Rust borrowing supplies read/write exclusion:
`StorageRef<'a>` is derived from `&'a OwnedStorage`, and `StorageMut<'a>` is
derived from `&'a mut OwnedStorage`. The kernel exposes no provider access,
mapping, queueing, split/group, legacy bridge, compatibility, recovery,
quarantine, cryptographic, or repeated-validation machinery.

## Types and construction

`BackendAllocation` is the single private unsafe boundary. It is `Debug + Send
+ Sync + 'static` and reports one checked `RootResourceExtent`, provider kind,
capabilities, and immutable type-erased diagnostics. Its unsafe implementor
contract covers unique root ownership, stable metadata, valid `Send`/`Sync`,
and exactly-once destruction. The core does not attempt to repair a violated
unsafe contract.

`import_unique_root(Box<dyn BackendAllocation>)` reads and validates the root
extent once, mints one `RootResourceIdentity`, derives the full checked
`RootBoundSpan`, and constructs:

```text
RootResource { identity, extent, allocation }
RootResourcePin(Arc<RootResource>)
OwnedSpanClaim { root: identity, span }
OwnedStorage { pin, claim }
```

The pin and claim are opaque and non-`Clone`; the pin exposes no `Arc` clone or
provider handle. The claim stores the exact identity-bearing span from Task 2.
All production constructors are private to `storage`; only the checked import
path can create an owner. Validation errors use the existing typed operation
envelope and return before any owner is constructed.

## Borrow capabilities

`OwnedStorage::as_ref(&self)` and `OwnedStorage::as_mut(&mut self)` are the only
capability constructors. `StorageRef` and `StorageMut` retain their source
borrow and expose only checked identity/span metadata. No conversion from a
root ID, span, raw pointer, `Arc`, provider handle, or shared reference exists.

The final `Arc<RootResource>` drop owns the provider destructor exactly once.
There is no second claim/hold accounting state machine; asynchronous holds and
prepared access belong to later phases.

## Verification

RED tests cover the private import boundary, successful owner construction,
invalid extent rejection, exact provider drop count, non-clone owner shape, and
shared/exclusive borrow behavior. The P2 ledger row is promoted only after the
unit and compile-contract tests pass. Later RED fixtures remain absent and
deferred.
