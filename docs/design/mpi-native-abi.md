# MPI native handles: one runtime backend, MPItrampoline removed

Design for #2735. **Not implemented.** Nothing in this note describes current behaviour;
[mpi-transport.md](./mpi-transport.md) is the contract of what exists. When this design
ships, its contract parts (feature table, C entry points, families, error text) move
into `mpi-transport.md` and this note stays as the decision record. It is a sibling
document rather than an amendment for that reason: the transport note must keep matching
the code, and a design that has not been built would make it describe behaviour the code
does not have.

## Goal

tenferro runs inside a Julia or Python process and uses **the communicator that MPI.jl or
mpi4py already set up**, through **native MPI handles**, with no MPIwrapper and no
MPItrampoline.

**Maintainer decision (2026-11-26), overriding the issue text:** the MPItrampoline path
is **removed**, not kept as a fallback. The native-handle backend becomes the only
runtime-loading path. A site MPI outside the supported families gets a named error, not
a wrapper.

## Decisions at a glance

| # | Decision |
|---|----------|
| D1 | Two ABI families: **MPICH ABI** (MPICH ≥ 3.1, Intel MPI ≥ 2019, MVAPICH ≥ 2, Cray MPICH ≥ 7, HPE HMPT) and **Open MPI ABI** (Open MPI 4.x/5.x, Fujitsu MPI, IBM Spectrum MPI). Everything else is rejected by name. |
| D2 | When a host has already initialized MPI, tenferro **uses the `libmpi` already loaded in the process** (found by walking the loaded objects) and never `dlopen`s a second one. A fresh load by name happens only for a standalone Rust program. |
| D3 | The family comes from `MPI_Get_library_version` **and** a symbol witness that must agree with it (`ompi_mpi_comm_world` for Open MPI, `MPIR_Err_return_comm` for MPICH). If the two disagree, or the family is unknown, attach fails before any handle is used. |
| D4 | The C API takes the communicator as its **Fortran integer handle** (`MPI_Comm_c2f`, carried as `int64_t`), so the caller never needs to know the family or the handle width. |
| D5 | Inside Rust, a handle is a family-neutral `usize`. The conversion to the family's real `int` or pointer happens in one place, the native sys crate, through separate function-pointer types for each family. |
| D6 | tenferro never calls `MPI_Init*`, `MPI_Finalize`, `MPI_Abort`, `MPI_Comm_dup`, `MPI_Comm_free`, `MPI_Comm_set_errhandler` or any attribute or keyval function on a host communicator. It checks `MPI_Finalized` on every C API entry. |
| D7 | `MPI_Status` is an opaque 64-byte buffer. The message count comes from `MPI_Get_count`, and the source and tag are already known because tenferro never probes with a wildcard. Only the layouts of tested families are encoded, and tenferro's correctness does not read them. |
| D8 | The Cargo feature keeps its name, `runtime`, because it still means "MPI is loaded at run time, no headers". Its backend changes from `mpi-rt-sys` to a new `mpi-native-sys`, and `MPI_RT_LIB` goes away. |
| D9 | The deletion of the MPItrampoline path is one revertable commit per repository, and its absence is asserted mechanically (see [Deletion](#deletion)). |

## 1. Families and detection

### Supported families

| Family | Implementations (version-string prefix, as in MPIPreferences.jl 0.1.12) | Library soname (Linux) |
|--------|---------------------------------------------------------------------------|------------------------|
| MPICH ABI | `MPICH Version:` (≥ 3.1), `Intel(R) MPI Library` (≥ 2019), `MVAPICH` (≥ 2), `CRAY MPICH` (≥ 7), `HPE HMPT` | `libmpi.so.12` (Cray: `libmpi_cray.so.12`) |
| Open MPI ABI | `Open MPI v` (including IBM Spectrum MPI), `FUJITSU MPI Library` | `libmpi.so.40` |

The minimum is MPICH, Open MPI and Intel MPI. MVAPICH, Cray MPICH, HPE HMPT, Fujitsu and
Spectrum MPI belong to those families by the ABI each one declares. They are accepted
but **unverified** until a run is recorded (see [Assumptions](#assumptions-that-would-sink-this-design)).

**Rejected by name:** MPItrampoline (`libmpitrampoline.so`) and MPIwrapper
(`MPIwrapper` prefix), the MPI 5.0 standard ABI (`libmpi_abi.so`, see Q1), Microsoft MPI,
HPE MPT (not HMPT), NEC MPI, and any string that matches no prefix.

### Finding the library (D2)

**Attach path** (host owns MPI; the C API in §2):

1. Walk the loaded objects with `dl_iterate_phdr`, including the main executable. For
   each object, `dlopen(path, RTLD_NOLOAD | RTLD_LAZY)` and look up
   `PMPI_Get_library_version`. The PMPI name is used on purpose: profiling interposers
   (LD_PRELOAD tools) define `MPI_*` but not `PMPI_*`, so it identifies the real
   library. `RTLD_NOLOAD` never loads anything new; it only takes a reference to an
   object that is already mapped.
2. Exactly one object must define it. **None** → `NoMpiInProcess`. **More than one** →
   `AmbiguousMpi`, listing the paths. This is also what happens under MPItrampoline,
   where the trampoline, the wrapper and the real library are all loaded.
3. The handle is never `dlclose`d. Unloading `libmpi` while the host uses it is fatal,
   and the reference costs nothing.

Walking the loaded objects is what makes this work when the host loaded `libmpi` with
`RTLD_LOCAL`: Python's `ctypes` and extension-module default, and MPI.jl through
`Libdl`. In that case `dlsym(RTLD_DEFAULT, …)` cannot see it.

**Standalone path** (a Rust program or the scenario tests, where the program itself
calls `mpi::initialize()`): if no loaded object defines the symbol, load
`MPI_NATIVE_LIB` when it is set. Otherwise try `libmpi.so.12`, then `libmpi.so.40`,
then `libmpi_cray.so.12`; on macOS, `libmpi.12.dylib` and `libmpi.40.dylib` (see Q6).
If nothing loads, the error lists the names it tried. `MPI_RT_LIB` is not read.

### Identifying the family (D3)

1. Call `MPI_Get_library_version` into an 8192-byte buffer. That is MPICH's
   `MPI_MAX_LIBRARY_VERSION_STRING`; Open MPI's is 256. The standard allows this call
   before `MPI_Init` and after `MPI_Finalize`.
2. Classify the string by the prefixes above. The table is the one MPIPreferences.jl
   uses, so tenferro and MPI.jl agree on every library both have seen.
3. Require the witness for that family in the same object:
   - Open MPI ABI: the data symbol `ompi_mpi_comm_world` is present.
   - MPICH ABI: `MPIR_Err_return_comm` is present and `ompi_mpi_comm_world` is absent.
     Open MPI exports three `MPIR_` symbols, all part of the debugger interface
     (`MPIR_debug_gate`, `MPIR_dll_name`, `MPIR_debug_typedefs_sizeof`), so the witness
     names one specific MPICH-internal function, not the prefix.
4. If `MPI_Abi_get_version` is exported, call it. A result of major ≥ 1 means the
   standard ABI is in use (its handles are not this family's), so attach fails with
   `StandardAbiNotSupported`. The symbol's mere **presence** says nothing: MPICH 5.0.2's
   native `libmpi.so.12` and Open MPI `main` both export it.
5. The string and the witness disagree → `FamilyMismatch`. That is the guard against a
   silent wrong-handle pass when a wrapper or trampoline forwards the underlying
   library's version string.

Detection runs once per process (`OnceLock`) and its result is reported by
`tfe_mpi_library_info` (§2), so a host package can call it when it loads, before any
communicator exists.

### Error text

Every loader error names the library path, the version string and the outcome. The
unsupported-family error, verbatim:

```text
tenferro-mpi: unsupported MPI library
  library: /opt/nec/ve/mpi/3.5.0/lib64/ve/libmpi.so.1
  MPI_Get_library_version: "NEC MPI 3.5.0 ..."
  supported: MPICH ABI (MPICH, Intel MPI, MVAPICH, Cray MPICH, HPE HMPT)
             and Open MPI ABI (Open MPI, Fujitsu MPI, IBM Spectrum MPI)
tenferro passes native MPI handles and has no wrapper fallback: MPItrampoline support
was removed in tensor4all/tenferro-rs#2735. Run with an MPI from a supported family, or
open an issue at https://github.com/tensor4all/tenferro-rs/issues with this library's
version string so its family table can be added.
```

The other variants share that layout and change only the first and last lines:

| Error | First line / remedy |
|-------|---------------------|
| `MpiTrampolineNotSupported` | "the host uses MPItrampoline/MPIwrapper" / "select a native MPI in the host: `MPIPreferences.use_system_binary()` or the default MPICH_jll for MPI.jl, a native-MPI mpi4py build for Python" |
| `StandardAbiNotSupported` | "the MPI library uses the MPI 5.0 standard ABI" / "use the implementation's native-ABI library (`libmpi.so.12`, `libmpi.so.40`)" |
| `FamilyMismatch` | "version string says <A> but symbols say <B>" / "this is a tenferro bug or an unknown wrapper; report it" |
| `NoMpiInProcess` | "no MPI library is loaded in this process" / "initialize MPI in the host before attaching; a statically linked host MPI is not supported (Q2)" |
| `AmbiguousMpi` | "more than one MPI library is loaded: <paths>" / "load exactly one MPI" |

## 2. Handles

### What a handle is (verified sources in [Verification](#verification))

| Item | MPICH ABI (MPICH 5.0.2, Intel MPI 2021.19 headers) | Open MPI ABI (Open MPI 4.1.6, 5.0.8 `mpi.h.in`) |
|------|------------------------------------------------------|----------------------------------------------------|
| `MPI_Comm`, `MPI_Datatype`, `MPI_Message`, `MPI_Request` | `int`, 4 bytes | pointer to an opaque struct, 8 bytes on 64-bit |
| `MPI_COMM_WORLD` / `_SELF` / `_NULL` | `0x44000000` / `0x44000001` / `0x04000000` | `&ompi_mpi_comm_world` / `&ompi_mpi_comm_self` / `&ompi_mpi_comm_null` (exported data symbols) |
| `MPI_BYTE`, `MPI_UINT8_T` | `0x4c00010d`, `0x4c00013b` | `&ompi_mpi_byte`, `&ompi_mpi_uint8_t` |
| `MPI_MESSAGE_NULL` / `_NO_PROC` | `0x2c000000` / `0x6c000000` | `&ompi_message_null` / `&ompi_message_no_proc` |
| `MPI_STATUS_IGNORE` | `(MPI_Status *)1` | `(MPI_Status *)0` |
| `MPI_Status` | 20 bytes: `count_lo, count_hi_and_cancelled, MPI_SOURCE@8, MPI_TAG@12, MPI_ERROR@16` | 24 bytes: `MPI_SOURCE@0, MPI_TAG@4, MPI_ERROR@8, _cancelled@12, size_t _ucount@16` |
| `MPI_ANY_SOURCE` / `MPI_PROC_NULL` | `-2` / `-1` | `-1` / `-2` |
| `MPI_Fint` | `int` | `ompi_fortran_integer_t` (4 bytes unless Fortran was built with 8-byte default `INTEGER`) |
| `MPI_Comm_f2c` | identity **macro**; Intel MPI does not export the function | exported function |

### C API representation (D4)

The C entry hands over the **Fortran integer handle**:

```c
typedef struct tfe_mpi_comm tfe_mpi_comm;          /* opaque, owned by tenferro */

/* Process-local; no message is sent. Fails with a named error on an unsupported
   family, an uninitialized or finalized MPI, or a rank/size mismatch. */
tfe_status_t tfe_mpi_comm_attach_f(int64_t fortran_comm,
                                   int32_t expected_rank, int32_t expected_size,
                                   tfe_mpi_comm **out, tfe_status_t *status);
tfe_status_t tfe_mpi_comm_detach(tfe_mpi_comm *comm, tfe_status_t *status);
/* Family, library path and version string; callable before MPI_Init. */
tfe_status_t tfe_mpi_library_info(char *buf, size_t len, size_t *needed,
                                  tfe_status_t *status);
/* send / recv over a compact column-major host buffer + dtype code + shape;
   same wire protocol and errors as mpi-transport.md. */
```

Status codes follow `capi-error-handling.md`: an MPI-state error is
`TFE_INVALID_ARGUMENT`, and the text goes through the thread-local last error. The
signature puts `tfe_status_t *status` last, as `capi.md` requires. These entry points
live in a new `tenferro-mpi-capi` `cdylib` crate, because `tenferro-capi` does not exist
yet (`capi.md` is still a plan). The crate folds into `tenferro-capi` when that crate
lands.

**Why the Fortran handle:** in both families it is an integer of known width, produced
by a standard, public call on the host side: `comm.py2f()` in mpi4py, and MPI.jl's
`MPI_Comm_c2f` binding. The C side converts it back with the family's rule: identity
for the MPICH ABI, an `MPI_Comm_f2c` call for Open MPI. A C caller therefore never
writes a family-dependent width, and the Rust side never receives a raw pointer it has
to trust. `int64_t` carries an 8-byte `MPI_Fint`; the value is range-checked against the
detected family's `MPI_Fint` width (4 bytes in every library listed above) and rejected
when it does not fit.

**Why `expected_rank`/`expected_size`:** the host knows both. Attach compares them with
`MPI_Comm_rank`/`MPI_Comm_size` on the converted handle. On the MPICH ABI, a wrong
integer can be another live communicator; this check turns that into an attach error
instead of misdirected messages. It is process-local and costs two calls.

### Inside Rust (D5)

`mpi-native-sys` exposes the same Rust-facing types `mpi-rt-sys` exposed today: handles
as `usize`, a neutral `MPI_Status` with public `MPI_SOURCE`/`MPI_TAG`/`MPI_ERROR`
fields. rsmpi-rt's `ffi` module therefore needs one new `cfg` arm and no other source
change. The family is chosen once, at load time:

- **MPICH ABI**: the function pointers are typed with `c_int` handles. A neutral handle
  holds the 32 bits zero-extended (`h as u32 as usize`). Converting back requires the
  upper bits to be zero, and a non-zero upper half is an internal error, never a
  truncation. User communicators such as `0x84000002` are negative as `int`, which is
  why the extension is defined on `u32`.
- **Open MPI ABI**: the function pointers are typed with `*mut c_void` handles, and the
  neutral handle is the address.
- An out-parameter handle (the `MPI_Message *` of `MPI_Mprobe`) is received into a
  stack slot of the family's width, then widened.

## 3. Who owns MPI (D6)

The host owns the MPI lifecycle and the communicator; tenferro borrows both.

**May:** `MPI_Get_library_version`, `MPI_Abi_get_version`, `MPI_Initialized`,
`MPI_Finalized`, `MPI_Query_thread`, `MPI_Is_thread_main`, `MPI_Comm_f2c`,
`MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Comm_test_inter`, `MPI_Send`, `MPI_Mprobe`,
`MPI_Mrecv`, `MPI_Get_count`, `MPI_Error_string`, all on the borrowed communicator with
tags the caller passes.

**May not:** `MPI_Init*`, `MPI_Finalize`, `MPI_Abort`, `MPI_Comm_dup` or any
constructor, `MPI_Comm_free`, `MPI_Comm_disconnect`, `MPI_Comm_set_errhandler`,
attribute or keyval calls, `MPI_Comm_set_name`, wildcard receives. The
`no_mpi_lifecycle` source scan extends to the new names.

Consequences:

- **No `MPI_Comm_free` on drop.** rsmpi-rt's `CommunicatorHandle::User` calls
  `MPI_Comm_free` when dropped, and `SimpleCommunicator::from_raw` produces that variant
  for a host sub-communicator. `tfe_mpi_comm` therefore holds the communicator in
  `ManuallyDrop`. A `from_raw_borrowed` constructor in rsmpi-rt is the cleaner shape
  (upstream consequence U3) but is not required.
- **Attach preconditions:** `MPI_Initialized` is true, `MPI_Finalized` is false, the
  handle is not `MPI_COMM_NULL`, it is an intra-communicator (inter-communicators are out
  of scope for the v1 wire protocol), and rank and size match the host's.
- **Thread level:** attach records `MPI_Query_thread`. Below
  `MPI_THREAD_SERIALIZED`, each call checks `MPI_Is_thread_main` and returns an error
  from any other thread. At `SERIALIZED` the caller serializes; this is documented, not
  checked.
- **Host finalized first:** every C API entry calls `MPI_Finalized`, one of the few
  calls allowed after finalize, and returns "the host finalized MPI; this communicator is
  no longer usable" without making any other MPI call. `tfe_mpi_comm_detach` after
  finalize frees only Rust memory. A host that finalizes **concurrently** with a
  tenferro call on another thread is outside the MPI standard; that is documented as the
  host's bug.
- **Error handler:** the communicator's error handler belongs to the host. mpi4py
  installs `MPI_ERRORS_RETURN` on the predefined communicators by default, and rsmpi-rt
  assumes `MPI_ERRORS_ARE_FATAL` and ignores return codes. So `mpi-native-sys` checks
  every return code (`MPI_SUCCESS` is 0 in every family), and tenferro's native backend
  surfaces a non-zero code as `Error::Mpi { code, message }` built with
  `MPI_Error_string` (1024-byte buffer; MPICH and Intel MPI define `MPI_MAX_ERROR_STRING` as 512, Open MPI as `OPAL_MAX_ERROR_STRING`, whose value is checked at implementation). This needs the
  `MpiBackend` methods to return `Result`: a small breaking change to an
  `engine`-feature trait, recorded as consequence T1.
- **Tags:** tenferro shares the host's communicator without duplicating it, so the host
  must not use the tag of a call that is in flight. This is the v1 contract from
  `mpi-transport.md`, which now crosses a language boundary.

## 4. Symbol resolution and ABI risk

Every function is resolved with `dlsym` on the handle of the identified object (D2) and
cached in a `OnceLock`, as `mpi-rt-sys` does today. Open MPI constants are the
**addresses** of the exported data symbols, read once. MPICH constants are integer
literals in a per-family table.

What differs across families, and why the design does not depend on it:

| Differs | How it is handled |
|---------|-------------------|
| Handle width and kind | Per-family function-pointer types (D5); one neutral `usize` above them. |
| Predefined constants | Per-family table; Open MPI by `dlsym` address, MPICH by literal. |
| `MPI_Status` layout and size | Passed to C as an opaque 64-byte, 8-aligned buffer, bigger than every layout (20, 24, MPIABI 40). The count comes from `MPI_Get_count` on that buffer and never from its fields. tenferro already knows the source and tag (explicit `process_at_rank` and tag); the per-family offsets only fill rsmpi-rt's neutral `MPI_SOURCE`/`MPI_TAG` for API completeness. |
| `MPI_STATUS_IGNORE` | Per-family sentinel substituted for the neutral one. |
| `MPI_ANY_SOURCE`, `MPI_PROC_NULL` | Not used by tenferro, whose API has no wildcard. Values are in the table for rsmpi-rt's `any_process()`. |
| `MPI_MAX_*_STRING` | Buffers are sized to the largest known value (8192 for the version string, MPICH's value; 1024 for the error string, twice MPICH's 512). |
| `MPI_Comm_f2c` | Identity for the MPICH ABI (it is a macro, and Intel MPI does not export it), `dlsym` for Open MPI. |
| Fortran `INTEGER` width | `int64_t` carrier plus a range check. |

What the design **does** rely on being identical: `MPI_SUCCESS == 0`, `int` counts,
ranks and tags, the C calling convention of every listed function, and the
pre-init/post-finalize callability of `MPI_Get_library_version`, `MPI_Initialized` and
`MPI_Finalized`. The MPI standard fixes all of these.

### rsmpi-rt's compile-time surface

rsmpi-rt's `ffi` module references about 134 `MPI_*` functions and 187 `ffi::` names in
total (count in [Verification](#verification)), so `mpi-native-sys` must define all of
those names. It is generated from the same spec as `mpi-rt-sys` (`gen_rust.py`):

- the **carried set** (§3 "may" list, plus what rsmpi-rt's communicator construction
  calls: `MPI_Comm_test_inter`, `MPI_Comm_get_parent`, `MPI_Comm_compare`) gets real
  per-family thunks;
- every other function gets a body that panics with
  `"rsmpi-rt native backend: MPI_<name> is outside the supported surface (see tenferro-rs docs/design/mpi-native-abi.md)"`.

tenferro-mpi reaches only the carried set. Q3 records whether to cfg-gate the rest at
compile time instead.

## 5. Deletion

The MPItrampoline path is removed in full. It has no consumer outside this repository:
on crates.io, `mpi-rt-sys` has exactly one reverse dependency (rsmpi-rt), rsmpi-rt has
none, and the path never had a hosted CI lane.

### What is deleted or changed (tenferro-rs)

| Item | Change |
|------|--------|
| `crates/tenferro-mpi/Cargo.toml` | `runtime = ["dep:mpi", "mpi/mpi-native-sys-backend"]`; rsmpi-rt `0.9`. `docs.rs` keeps `features = ["runtime"]`, which is still header-free. |
| `crates/tenferro-mpi/src/{lib,backend}.rs` | Doc text naming MPIABI, `mpi-rt-sys` or MPIwrapper is rewritten. `Rt` keeps its name and gains `Result` returns (T1). |
| `crates/tenferro-mpi/tests/mpi_p2p_runtime.rs` | The `MPI_RT_LIB` skip is deleted. The test skips only on the loader's `NotFound`; every other loader error fails the test, so a family error cannot hide as a skip. |
| `crates/tenferro-mpi/README.md` | The MPIwrapper build recipe and `MPI_RT_LIB` are deleted. The co-installation paragraph now says `mpi-native-sys` declares no `links` key, which becomes a requirement on that crate. |
| `scripts/run-mpi.sh` | The `rt` branch keeps its name (it still selects `--features runtime`). Its `MPI_RT_LIB` check and message are deleted. |
| `scripts/check-mpi-features.sh` | The `runtime` graph expects one `mpi-native-sys`. **`mpi-rt-sys` must be absent from every graph.** The `--runtime-only` comment no longer mentions an MPIwrapper lane. |
| `scripts/ci/run_profile.py`, `.github/workflows/ci.yml` | A new hosted `mpi-runtime` profile and job: Open MPI and MPICH from apt, no wrapper, `scripts/run-mpi.sh rt`. This is the lane the issue asks for, now affordable because nothing has to be built. |
| `docs/design/mpi-transport.md` | The feature-table row, the "one `mpi` identity" and co-installation paragraphs, and the Verification bullet on MPIwrapper are rewritten from this note. |
| `Cargo.lock` | `mpi-rt-sys` disappears. |
| `docs/worklogs/*` | **Unchanged.** They are dated history. |

### Upstream consequences (tensor4all/rsmpi-rt; nothing is filed by this PR)

- **U1.** Add a `mpi-native-sys` crate and a `mpi-native-sys-backend` feature.
- **U2.** Remove `mpi-rt-sys-backend` and the `mpi-rt-sys` workspace member. Removing a
  feature is breaking, so this ships as **rsmpi-rt 0.9.0**. `mpi-rt-sys 0.1.0` stays
  published, unyanked, with a deprecation note: yanking breaks old lockfiles and gains
  nothing.
- **U3** (optional). Add `SimpleCommunicator::from_raw_borrowed`, which never frees.

### What depends on the absence (asserted, not hoped)

1. `check-mpi-features.sh` fails if `mpi-rt-sys` appears in any graph.
2. A source scan (an extension of `no_mpi_lifecycle`) fails if `MPI_RT_LIB`,
   `MPIABI_` or `libmpiwrapper` appears under `crates/` or `scripts/`. Worklogs are
   excluded.
3. At run time, the loader rejects a library that exports `MPIABI_Get_library_version`
   or reports the `MPIwrapper` prefix (`MpiTrampolineNotSupported`). A stale
   `MPI_NATIVE_LIB=…/libmpiwrapper.so` therefore fails by name instead of passing
   MPIABI handles as native ones.

### Migration

| Before | After |
|--------|-------|
| `cargo build -p tenferro-mpi --features runtime` | unchanged |
| build MPIwrapper with CMake + Fortran, `export MPI_RT_LIB=…/libmpiwrapper.so` | nothing; the site `libmpi` is found in the process or by soname, or `export MPI_NATIVE_LIB=/path/to/libmpi.so.N` |
| `MPI_RT_LIB=… scripts/run-mpi.sh rt` | `scripts/run-mpi.sh rt` |
| MPI.jl with `MPItrampoline_jll` + tenferro | MPI.jl with the default `MPICH_jll` or `MPIPreferences.use_system_binary()` |

A leftover `MPI_RT_LIB` in the environment is ignored.

### What a user on an unsupported family loses

Before: a site MPI that MPIwrapper could be built against (NEC MPI, HPE MPT, anything
with an `mpi.h`) worked, **provided the program used MPItrampoline end to end**. For a
host language that meant MPI.jl with `MPItrampoline_jll`, since mpi4py and MPI.jl with a
system binary hold native handles that the wrapper path never accepted. After: attach or
standalone initialization fails with the error in §1 at start-up. The loss is real for
two groups: Rust programs on exotic MPIs, and MPI.jl users who chose MPItrampoline.

## 6. Testing without a cluster

**Unit tests (no MPI, in the default lanes):**

- classification of recorded version strings: MPICH 5.0.2, `Intel(R) MPI Library
  2021.19`, `Open MPI v4.1.6`, `MPIwrapper Version:`, an NEC sample, garbage;
- witness agreement and disagreement;
- `MPI_Fint` range checks and the MPICH zero-extension round trip, including
  `0x84000002`;
- the layout table as `const` assertions (sizes 20 and 24, offsets 8/12 and 0/4, buffer
  of 64 ≥ each).

**Loader integration (one process, singleton MPI, the test harness plays host):** for
each library installed here (Open MPI 4.1.6 from apt, MPICH 5.0.2 from the Julia
`MPICH_jll` artifact, Intel MPI 2021.19 from oneAPI):

- load it, check its family;
- `MPI_Init`, then attach `MPI_COMM_WORLD` via its Fortran handle and check rank 0 /
  size 1;
- attach with a wrong `expected_size` and check it is refused;
- `MPI_Finalize`, then check that a transport call returns the finalized error and
  does not crash.

**Two ranks:** `MPIEXEC=<family's launcher> scripts/run-mpi.sh rt` runs the existing
eight `mpi_p2p_runtime` scenarios (seven dtypes, rank 0 and empty extents, typed and
erased crossings, rejections with recovery, aborts, tags, invalid `bool`, foreign
frames) unchanged, once per family. The hosted lane runs Open MPI and MPICH; Intel MPI
is local evidence in the work log.

**Julia demo** (`mpiexec -n 2 julia --project demo/mpi.jl`, run once with
`MPICH_jll` and once with the system Open MPI). It asserts:

1. `tfe_mpi_library_info` reports the same family MPIPreferences chose;
2. attach succeeds with `MPI.Comm_rank`/`MPI.Comm_size` as the expected values;
3. rank 0 sends tensor `A` through tenferro on tag 1 and the same bytes through
   `MPI.Send` on tag 2; rank 1 receives both and checks them bit for bit;
4. host traffic and tenferro traffic interleave on one communicator without crossing
   tags;
5. after `MPI.Finalize()`, a tenferro call returns the finalized error;
6. negative case: with `MPItrampoline_jll` selected, attach fails with
   `MpiTrampolineNotSupported`.

**mpi4py demo** (`mpiexec -n 2 python demo/mpi.py`): the same items 1–5, using
`comm.py2f()` and NumPy for the reference. It also asserts that an MPI error under
mpi4py's default `MPI_ERRORS_RETURN` comes back as tenferro's `Mpi` error and is not
swallowed. The trigger is a tag above `MPI_TAG_UB`, refused by MPI rather than by
tenferro's local check.

## Rejected alternatives

- **Keep MPItrampoline as a fallback:** superseded by the maintainer decision of
  2026-11-26; two runtime paths double the test matrix to serve users who do not exist
  yet.
- **Pass the raw C handle (`uintptr_t`) through the C API:** the caller would have to
  know the width and sign convention, and mpi4py's `_handleof` is private API.
- **A tenferro-local `MpiBackend` with its own tiny `dlsym` table:** it bypasses
  rsmpi-rt, so Rust callers' `Process` values cannot reach it, and the process would
  hold two MPI bindings.
- **`dlsym(RTLD_DEFAULT)` only:** it misses a host `libmpi` that was loaded
  `RTLD_LOCAL`, the normal case for Python and Julia.
- **`dlopen` a fresh `libmpi` by name in the attach path:** it can load a second MPI
  next to the host's, which crashes.
- **Family from the version string alone:** a trampoline that forwards the underlying
  string would pass the wrong handles silently.
- **Family from `MPI_Abi_get_version` presence:** native MPICH 5 and Open MPI `main`
  both export it.
- **A finalize-time attribute callback on `MPI_COMM_SELF` to learn about finalize:**
  it needs a keyval (process-global state) and a per-family callback ABI; a
  `MPI_Finalized` check per call is enough.
- **`MPI_Comm_dup` the host communicator for tag isolation:** dup is collective, so
  attach would have to run on all ranks at once, and v1's no-constructor rule exists to
  keep it local.
- **An attach-time handshake message between ranks:** this also makes attach
  collective; the rank/size check catches the wrong-handle case locally.
- **Rename the feature to `native`:** the name `runtime` is still accurate, and a
  rename breaks every manifest for no behaviour difference.

## Open questions (each with the decision it needs)

- **Q1. The MPI 5.0 standard ABI family** (`libmpi_abi.so`: MPICH ≥ 5, mpi4py and
  MPI.jl already support it). *Decision needed:* add it in v1 or reject it until a host
  asks. The cost is one more table, with constants from `mpi_abi.h`, which has not been
  read here. Adding it lowers the regret in §7.
- **Q2. A statically linked host MPI** (no shared object to find). *Decision needed:*
  build `tfe_mpi_comm_attach_vtable`, where the host passes the ~15 function pointers
  it uses, now or on the first report. Until then: `NoMpiInProcess`. The removed
  wrapper path never handled this case either.
- **Q3. Unsupported rsmpi-rt functions:** runtime panic stubs (this design) or
  compile-time `cfg` gating of rsmpi-rt's collectives, requests and attributes.
  *Decision needed:* will any consumer other than tenferro-mpi select the native backend
  before collectives are carried? If yes, gate.
- **Q4. Profilers:** resolving through the identified object's handle bypasses
  `LD_PRELOAD` interposers for tenferro's own traffic. *Decision needed:* accept that,
  or resolve `MPI_*` through `RTLD_DEFAULT` first when the library is in global scope.
- **Q5. Where `MpiBackend` becomes fallible (T1):** a breaking change to the
  `engine`-only trait. *Decision needed:* accept the break, since there are no known
  external implementers, or add a parallel fallible trait.
- **Q6. macOS:** there is no `dl_iterate_phdr`; the walk needs
  `_dyld_image_count`/`_dyld_get_image_name`. *Decision needed:* is the issue's "laptop"
  run a Mac? If so, macOS is v1 scope; otherwise Linux only.

## 7. What would make us regret the deletion, and the recovery cost

Here are the conditions under which removing MPItrampoline turns out wrong.

| Regret condition | Would the old path have helped? | Recovery and cost |
|------------------|--------------------------------|-------------------|
| A site MPI with its own ABI (NEC MPI on SX-Aurora, HPE MPT, a future vendor MPI) is where the paper's cluster run or a user must run. | Only if that user ran MPItrampoline end to end. A mpi4py or MPI.jl-system host still fails. | Add a family table: about 1–2 days per family when the library exports its constants or uses integer handles, plus a site run to verify. If it is the MPI 5 standard ABI, Q1 covers it. |
| The host's MPI is statically linked and exports no symbols. | No: MPItrampoline also needs the host on MPItrampoline. | Q2's vtable entry: about 2–3 days, family-free, because the host supplies the functions. |
| Detection is silently wrong: an MPICH-derived library changes handle encoding, or a wrapper forwards both the version string and an `MPIR_` symbol. | Yes: MPIABI handles are family-free by construction. | Before shipping, the rank/size check at attach catches most cases. Afterwards, add a witness and a table row. |
| Many existing users depend on MPItrampoline. | Yes. | **Revert:** D9 keeps the deletion as one commit per repository, `mpi-rt-sys 0.1.0` stays on crates.io, and MPIABI exports `MPIABI_Comm_f2c`, so the Fortran-handle C API (D4) works over it unchanged. Re-adding it as a second feature is about 1 week including the rsmpi-rt release. |

The bet is that the families in D1 cover the sites the paper and its readers use, and
that a host language with native handles (mpi4py, MPI.jl with a system or JLL MPI) is
the common case. The wrapper path could not serve that case anyway.

## Assumptions that would sink this design

- **A1.** The version-string prefixes identify the family. MPIPreferences.jl relies on
  the same table; checked against the local Intel MPI, MPICH and Open MPI binaries.
- **A2.** Every MPICH-ABI implementation uses 32-bit `int` handles with the constants in
  §2. Verified for MPICH 5.0.2 and Intel MPI 2021.19 headers; MVAPICH, Cray MPICH and
  HPE HMPT rely on the MPICH ABI initiative and are **unverified**.
- **A3.** Open MPI 4.x and 5.x handles are addresses of exported `ompi_mpi_*` data
  symbols. Verified in the 4.1.6 binary and the 4.1.6 and 5.0.8 headers. An Open MPI
  that adopts the standard ABI as its default `libmpi` breaks this; the
  `MPI_Abi_get_version` check (§1 step 4) is the guard.
- **A4.** `MPIR_Err_return_comm` is exported by every MPICH-derived library and by
  nothing else. Verified present in MPICH 5.0.2 and Intel MPI 2021.19, and absent in Open
  MPI 4.1.6; **unverified** for Cray MPICH, MVAPICH and `libmpitrampoline`.
- **A5.** On a standard-ABI build `MPI_Abi_get_version` reports major ≥ 1, and on a
  native-ABI build it reports something else. **Unverified**: run it against
  `MPICH_jll`'s `libmpi.so.12` and `MPIABI_jll`'s `libmpi_abi.so` before relying on
  it.
- **A6.** The host's MPI is one shared object, visible to `dl_iterate_phdr`, that
  exports `PMPI_Get_library_version`; exactly one is loaded.
- **A7.** `MPI_Fint` fits in `int64_t`, and the host can produce it publicly
  (`comm.py2f()`, MPI.jl's `MPI_Comm_c2f`); the exact MPI.jl spelling is checked when
  the demo is written.
- **A8.** No one outside this repository uses the MPItrampoline path. On crates.io,
  `mpi-rt-sys` has rsmpi-rt as its only reverse dependency and rsmpi-rt has none.
  Unpublished users cannot be seen.

## Verification

Commands run on 2026-11-26 (Linux x86_64) to check the claims above:

```bash
# MPItrampoline path as it exists: handles are usize, loader reads MPI_RT_LIB,
# functions are MPIABI_*-prefixed (mpi-rt-sys 0.1.0 from the cargo registry)
sed -n 1,40p ~/.cargo/registry/src/*/mpi-rt-sys-0.1.0/src/types.rs     # MPI_Comm = usize; MPI_Status 40 bytes
cat ~/.cargo/registry/src/*/mpi-rt-sys-0.1.0/src/loader.rs              # env MPI_RT_LIB, panics if unset
grep -n MPIABI_Send ~/.cargo/registry/src/*/mpi-rt-sys-0.1.0/src/functions.rs
# rsmpi-rt 0.8.2: ffi arms, CommunicatorHandle::User frees on drop, Status fields read directly
sed -n 140,152p ~/.cargo/registry/src/*/rsmpi-rt-0.8.2/src/lib.rs
grep -n "MPI_Comm_free\|MPI_SOURCE\|MPI_TAG" -r ~/.cargo/registry/src/*/rsmpi-rt-0.8.2/src
grep -rhoE "ffi::MPI_[A-Z][a-z_]+[a-z]" ~/.cargo/registry/src/*/rsmpi-rt-0.8.2/src | sort -u | wc -l   # 134
grep -rhoE "ffi::(MPI|RSMPI)_[A-Za-z_]+" ~/.cargo/registry/src/*/rsmpi-rt-0.8.2/src | sort -u | wc -l # 187
# MPICH ABI handles and constants
grep -nE "typedef int MPI_(Comm|Datatype|Message|Request|Fint);|define MPI_(COMM_(WORLD|SELF|NULL)|BYTE|UINT8_T|MESSAGE_NULL|MESSAGE_NO_PROC|ANY_SOURCE|PROC_NULL|STATUS_IGNORE|MAX_ERROR_STRING|MAX_LIBRARY_VERSION_STRING) |MPI_Comm_f2c" \
  /opt/intel/oneapi/mpi/2021.19/include/mpi.h \
  ~/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/include/mpi.h   # MPICH 5.0.2 (MPICH_jll)
awk '/typedef struct MPI_Status/,/} MPI_Status;/' <same headers>
# Open MPI: exported constants and witness, from the installed 4.1.6 binary
nm -D --defined-only /usr/lib/x86_64-linux-gnu/libmpi.so.40 | grep -E "ompi_mpi_(comm_world|comm_self|comm_null|byte|uint8_t)|ompi_message_(null|no_proc)|MPIR_"
# Open MPI headers (not installed locally): mpi.h.in at v4.1.6, v5.0.8 and main
gh api "repos/open-mpi/ompi/contents/ompi/include/mpi.h.in?ref=v4.1.6" -q .content | base64 -d \
  | grep -nE "MPI_Comm;|MPI_Message;|MPI_(ANY_SOURCE|PROC_NULL|STATUS_IGNORE|MAX_LIBRARY_VERSION_STRING) |MPI_Fint"
#   (same for v5.0.8; main also declares MPI_Abi_get_version)
# Exports and witnesses across the three libraries
for L in <MPICH_jll libmpi.so.12> /opt/intel/oneapi/mpi/2021.19/lib/libmpi.so.12 /usr/lib/x86_64-linux-gnu/libmpi.so.40; do
  nm -D --defined-only $L | awk '{print $3}' | grep -E "^(MPIR_Err_return_comm|MPI_Abi_get_version|MPI_Comm_f2c|MPI_Mprobe|MPI_Mrecv|MPI_Get_count|MPI_Is_thread_main|ompi_mpi_comm_world)$"
done
strings <each libmpi> | grep -E "MPICH Version|Intel\(R\) MPI Library|Open MPI v"
# Detection table used by MPI.jl
sed -n 270,372p ~/.julia/packages/MPIPreferences/wIB4a/src/MPIPreferences.jl   # MPIPreferences 0.1.12
grep -rn libmpi ~/.julia/packages/MPIABI_jll/*/src/wrappers/x86_64-linux-gnu*.jl          # libmpi_abi.so.1
grep -rn libmpi ~/.julia/packages/MPItrampoline_jll/*/src/wrappers/x86_64-linux-gnu*.jl   # libmpitrampoline.so.5
# Who depends on the path being removed
curl -s https://crates.io/api/v1/crates/mpi-rt-sys/reverse_dependencies   # rsmpi-rt only
curl -s https://crates.io/api/v1/crates/rsmpi-rt/reverse_dependencies     # none
grep -rnlE "mpi-rt-sys|MPI_RT_LIB|MPIwrapper|mpi_p2p_runtime" crates scripts docs .github   # deletion inventory, §5
```

Results that the design depends on:

- MPICH 5.0.2 and Intel MPI 2021.19 have `typedef int` for every handle and `MPI_Fint`.
  Their constants and `MPI_Status` (20 bytes, source at 8) match §2. Both define
  `MPI_Comm_f2c` as an identity macro; Intel MPI does not export the function, MPICH
  5.0.2 does.
- Open MPI 4.1.6 exports `ompi_mpi_comm_world`, `ompi_mpi_byte`, `ompi_mpi_uint8_t`,
  `ompi_message_null` and `ompi_message_no_proc`. Its only `MPIR_` symbols are
  `MPIR_debug_gate`, `MPIR_dll_name` and `MPIR_debug_typedefs_sizeof`, and it lacks
  `MPIR_Err_return_comm`. Its headers (4.1.6 and 5.0.8) give pointer handles,
  `MPI_ANY_SOURCE -1`, `MPI_PROC_NULL -2`, `MPI_STATUS_IGNORE ((MPI_Status *) 0)`, a
  24-byte status with source at 0, and `MPI_MAX_LIBRARY_VERSION_STRING 256`.
- MPICH 5.0.2's native `libmpi.so.12` and Intel MPI 2021.19 both export
  `MPI_Abi_get_version`, and Open MPI `main` declares it. That is why presence is not
  used as a family signal.
- `mpi-rt-sys 0.1.0` handles are `usize`, its status is 40 bytes, and its loader panics
  without `MPI_RT_LIB`. rsmpi-rt 0.8.2 frees a `User` communicator on drop and reads
  `MPI_SOURCE`/`MPI_TAG` as struct fields.

Not checked here (and therefore listed under Assumptions): Cray MPICH, MVAPICH, HPE
HMPT, Fujitsu and Spectrum MPI binaries; `libmpitrampoline`'s exports (the artifact is
not installed); the MPI 5 `mpi_abi.h` constants; `MPI_Abi_get_version`'s return value on
each build; the MPI.jl spelling of `MPI_Comm_c2f`; mpi4py's default `MPI_ERRORS_RETURN` on the predefined communicators; Open MPI's `OPAL_MAX_ERROR_STRING`. Nothing here was run under
`mpiexec`, because this PR is design only.
