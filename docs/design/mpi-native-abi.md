# MPI native handles: one runtime backend, MPItrampoline removed

Design for #2735. **Implemented** (tenferro-rs#2735 and tensor4all/rsmpi-rt#8); the contract
parts (feature table, C entry points, families, error text) now live in
[mpi-transport.md](./mpi-transport.md), and this note stays as the decision record. Where
the implementation settled something this note left open, the notes below say so. It is a sibling
document rather than an amendment for that reason: the transport note must keep matching
the code, and a design that has not been built would make it describe behaviour the code
does not have.

Revision 2 answers the independent audit on #2736 (verdict: sound-with-changes). The
[audit response](#audit-response-revision-2) at the end maps each finding to what
changed.

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
| D1 | Three ABI families: the **MPI 5.0 standard ABI** (`libmpi_abi`: MPICH `--enable-mpi-abi`, Open MPI's `mpi_abi_wrapper`), the **MPICH ABI** (MPICH, Intel MPI, MVAPICH, Cray MPICH, HPE HMPT) and the **Open MPI ABI** (Open MPI, Fujitsu MPI, IBM Spectrum MPI). The classification table is **tenferro's own**, derived from MPIPreferences.jl at a pinned commit, and its differences from that table are listed. Everything else is rejected by name. |
| D2 | When a host has already initialized MPI, tenferro **uses the MPI library already loaded in the process** (found by walking the loaded objects) and never `dlopen`s a second one. A fresh load by name happens only for a standalone Rust program. |
| D3 | Detection order: **(1)** MPIwrapper present → reject; **(2)** standard ABI, decided by the *return value* of `MPI_Abi_get_version`; **(3)** only then the version string and a per-family symbol witness for the two native ABIs. The standard-ABI check runs before the native witness, so a standard-ABI library is never misreported as an unknown family. |
| D4 | The C API takes the communicator as its **raw C handle** (`uint64_t`), as the host's own MPI library represents it. tenferro reads that handle through **the same library object the host uses** (D2), so the encoding cannot disagree. tenferro does **not** claim to validate the handle's identity; no local check can. |
| D5 | Inside Rust, a handle is a family-neutral `usize`. The conversion to the family's real `int` or pointer happens in one place, the native sys crate, through separate function-pointer types for each family. |
| D6 | tenferro never calls `MPI_Init*`, `MPI_Finalize`, `MPI_Abort`, `MPI_Comm_dup`, `MPI_Comm_free`, `MPI_Comm_set_errhandler` or any attribute *write* or keyval function. It checks `MPI_Finalized` on every C API entry, reports the communicator's error handler at attach, and validates locally everything that would otherwise reach that handler. |
| D7 | `MPI_Status` is an opaque 64-byte buffer. The message count comes from `MPI_Get_count`; source and tag are already known because tenferro never probes with a wildcard. |
| D8 | The Cargo feature keeps its name, `runtime`, because it still means "MPI is loaded at run time, no headers". Its backend changes from `mpi-rt-sys` to a new `mpi-native-sys`, and `MPI_RT_LIB` goes away. |
| D9 | The deletion of the MPItrampoline path is one revertable commit per repository, and its absence is asserted mechanically (see [Deletion](#5-deletion)). |
| D10 | **No panic crosses the C boundary.** Every `extern "C"` entry runs inside `catch_unwind`; the cdylib is built with `panic = "unwind"`. A panic poisons the communicator object. |
| D11 | **Thread safety is defined per MPI thread level**, and the host bindings enforce what tenferro cannot see. Under MPI.jl's default `:serialized`, the Julia binding refuses to attach unless the process runs a single Julia thread or MPI provides `MPI_THREAD_MULTIPLE`. |
| D12 | Detection caches **only success**. A failure (for example "no MPI loaded yet") is returned and retried on the next call, so calling into tenferro before MPI is loaded cannot fail permanently. |

## 1. Families and detection

### The classification table (D1)

The table is tenferro's own. Its source is MPIPreferences.jl's
`identify_implementation_version_abi` at JuliaParallel/MPI.jl `ad793d65`
(`lib/MPIPreferences/src/MPIPreferences.jl`, still versioned 0.1.12). Revision 1 cited
"the table MPIPreferences.jl uses" from the released 0.1.12 installed locally. That
release has no `MPIABI` branch, and the claim of a prefix match was wrong for Cray. The
rows below follow `ad793d65`; the **Differs** column says where tenferro deviates, and
why.

| Match on the version string (in this order) | Implementation | Version rule | tenferro family | Differs from MPIPreferences |
|---|---|---|---|---|
| starts with `MPIwrapper` | MPIwrapper | — | **rejected** (`MpiTrampolineNotSupported`) | MPIPreferences maps it to `MPItrampoline`, which tenferro no longer supports |
| starts with `MPICH` **and** contains `--enable-mpi-abi` | MPICH standard-ABI build | — | standard ABI | same |
| starts with `mpi_abi_wrapper` | Open MPI's standard-ABI wrapper | `MPI_ABI_VERSION n.m` | standard ABI | same |
| starts with `MPICH` | MPICH | ≥ 3.1 | MPICH ABI | same |
| starts with `Open MPI` | Open MPI, IBM Spectrum MPI | — | Open MPI ABI | same |
| starts with `Microsoft MPI` | MS-MPI | — | **rejected** | MPIPreferences supports it; tenferro is Linux/macOS only |
| starts with `Intel` | Intel MPI | **> 2014** | MPICH ABI | same; revision 1's "≥ 2019" was wrong |
| starts with `MVAPICH` | MVAPICH | ≥ 2 | MPICH ABI | same |
| **contains** `CRAY MPICH` | Cray MPICH | ≥ 7 | MPICH ABI | same. The real string is `MPI VERSION    : CRAY MPICH version 7.7.10 (ANL base 3.2)`, so this row is a substring match, not a prefix |
| starts with `FUJITSU MPI` | Fujitsu MPI | — | Open MPI ABI | same |
| starts with `HPE MPT` | HPE MPT | — | **rejected** | MPIPreferences has its own `HPE MPT` ABI, and MPI.jl a separate `src/api/mpt.jl` with `MPI_Comm = Cuint`. It is not MPICH-ABI compatible, and tenferro has no table for it |
| starts with `HPE HMPT` | HPE HMPT | — | MPICH ABI | same. HMPT is a different product from MPT and declares the MPICH ABI |
| anything else | — | — | **rejected** (`UnsupportedMpi`) | MPIPreferences says `unknown` |

The order matters in two places: the standard-ABI MPICH row must come before plain
`MPICH`, and `MPIwrapper` comes first. The version-string table is only consulted at
step 3 of the detection below. A standard-ABI library is recognized from
`MPI_Abi_get_version` before its string is read, so the two standard-ABI rows are a
cross-check, not the gate.

### Finding the library (D2)

**Attach path** (the host owns MPI; the C API in §2):

1. Walk the loaded objects with `dl_iterate_phdr`, including the main executable. For
   each object, take a reference with `dlopen(path, RTLD_NOLOAD | RTLD_LAZY)`.
   `RTLD_NOLOAD` never maps anything new.
2. **MPIwrapper:** if any object exports `MPIABI_Get_library_version`, fail with
   `MpiTrampolineNotSupported`. This runs first because under MPItrampoline the
   trampoline, the wrapper and the real library are all loaded.
3. **Standard ABI:** for every object that exports `MPI_Abi_get_version`, call it.
   - Exactly one object reports `abi_major ≥ 1`: that object is the host's MPI, in the
     standard-ABI family. Other MPI objects may be loaded too; `mpi_abi_wrapper` sits on
     top of a native `libmpi.so.40`. They are ignored, because a process with a
     standard-ABI library loaded is assumed to call MPI through it (assumption A9).
   - More than one reports `abi_major ≥ 1`: fail with `AmbiguousMpi`.
   - Calling the function is the signal; the symbol's **presence** is not. MPICH 5.0.2's
     native `libmpi.so.12`, Intel MPI 2021.19 and Open MPI `main` all export it. The
     audit reports that native builds return `-1, -1` (MPICH generates it, Open MPI
     documents it). I have not run the call myself; see A5.
4. **Native ABIs:** otherwise, exactly one object must export
   `PMPI_Get_library_version`. Profiling interposers (LD_PRELOAD tools) define `MPI_*`
   but not `PMPI_*`, so this names the real library. **None** → `NoMpiInProcess`.
   **More than one** → `AmbiguousMpi`, listing the paths.
5. The reference is never `dlclose`d. That also means a host `dlclose` cannot unmap the
   library under a live tenferro object: the reference count keeps it mapped.

Walking the loaded objects covers a host that loaded `libmpi` with `RTLD_LOCAL`, which
is the default for Python's `ctypes` and for an extension module's dependencies.
`dlsym(RTLD_DEFAULT, …)` cannot see such a library. MPI.jl opens `libmpi` with
`RTLD_GLOBAL`, so the walk is needed only for Python, but it is correct for both.

**Standalone path** (a Rust program or the scenario tests, where the program itself
calls `mpi::initialize()`): if steps 2–4 find nothing, load `MPI_NATIVE_LIB` when it is
set. Otherwise try `libmpi.so.12`, then `libmpi.so.40`, then `libmpi_cray.so.12`, then
`libmpi_abi.so.1`; on macOS, `libmpi.12.dylib` and `libmpi.40.dylib` (see Q6). Then run
steps 2–4 on the loaded object. If nothing loads, the error lists the names it tried.
`MPI_RT_LIB` is not read.

### Identifying the family (D3)

| Step | Check | Outcome |
|------|-------|---------|
| 1 | an object exports `MPIABI_Get_library_version` | `MpiTrampolineNotSupported` |
| 2 | `MPI_Abi_get_version` returns `abi_major ≥ 1` | **standard ABI**. The version string is then classified and must land on a standard-ABI row of the table; any other row → `FamilyMismatch`. An `abi_major` above 1 → `UnsupportedAbiVersion` (only ABI v1 has tables) |
| 3 | `MPI_Get_library_version` (8192-byte buffer), classified by the table | MPICH ABI, Open MPI ABI, or a named rejection |
| 4 | witness in the same object: Open MPI ABI requires the data symbol `ompi_mpi_comm_world`; MPICH ABI requires `MPIR_Err_return_comm` and the absence of `ompi_mpi_comm_world` | mismatch → `FamilyMismatch` |

Open MPI exports three `MPIR_` symbols, all part of the debugger interface
(`MPIR_debug_gate`, `MPIR_dll_name`, `MPIR_debug_typedefs_sizeof`), so the MPICH
witness names one specific MPICH-internal function, not the prefix. The standard-ABI
family has no witness symbol: the `MPI_Abi_get_version` return value is its witness, and
step 2's string cross-check is the second signal.

`MPI_Get_library_version` and `MPI_Initialized` are callable before `MPI_Init` and after
`MPI_Finalize`. `MPI_Abi_get_version` is assumed to be callable at any time as well
(A10).

### Caching (D12)

Detection runs under a `Mutex`. A **successful** result goes into a `OnceLock` and is
final for the process; the library reference behind it is never released. A failure is
returned to the caller and **not** stored. So `NoMpiInProcess` from a host package that
called `tfe_mpi_library_info` at import time, before `MPI.Init`, does not stick: the
next call detects again. This is also what makes revision 1's `OnceLock` safe to keep for
the success case.

### Error text

Every loader error names the library path, the version string and the outcome. The
unsupported-family error, verbatim:

```text
tenferro-mpi: unsupported MPI library
  library: /opt/nec/ve/mpi/3.5.0/lib64/ve/libmpi.so.1
  MPI_Get_library_version: "NEC MPI 3.5.0 ..."
  supported: MPI 5.0 standard ABI (libmpi_abi: MPICH --enable-mpi-abi, mpi_abi_wrapper),
             MPICH ABI (MPICH, Intel MPI, MVAPICH, Cray MPICH, HPE HMPT)
             and Open MPI ABI (Open MPI, Fujitsu MPI, IBM Spectrum MPI)
tenferro passes native MPI handles and has no wrapper fallback: MPItrampoline support
was removed in tensor4all/tenferro-rs#2735. Run with an MPI from a supported family
(a standard-ABI build of your MPI also works), or open an issue at
https://github.com/tensor4all/tenferro-rs/issues with this library's version string
so its family table can be added.
```

The other variants share that layout and change only the first and last lines:

| Error | First line / remedy |
|-------|---------------------|
| `MpiTrampolineNotSupported` | "the host uses MPItrampoline/MPIwrapper" / "select a native MPI in the host: `MPIPreferences.use_system_binary()`, the default `MPICH_jll` or `MPIABI_jll` for MPI.jl; a native or standard-ABI mpi4py build for Python" |
| `UnsupportedAbiVersion` | "the MPI library implements standard ABI version <n>" / "tenferro knows ABI version 1; report it" |
| `FamilyMismatch` | "version string says <A> but symbols say <B>" / "this is a tenferro bug or an unknown wrapper; report it" |
| `NoMpiInProcess` | "no MPI library is loaded in this process" / "initialize MPI in the host before attaching; a statically linked host MPI is not supported (Q2)" |
| `AmbiguousMpi` | "more than one MPI library is loaded: <paths>" / "load exactly one MPI" |

## 2. Handles

### What a handle is (verified sources in [Verification](#verification))

| Item | MPICH ABI (MPICH 5.0.2, Intel MPI 2021.19 headers) | Open MPI ABI (Open MPI 4.1.6, 5.0.8 `mpi.h.in`) | Standard ABI v1 (`mpi-forum/mpi-abi-stubs` `d224b2bf` `mpi.h`) |
|------|------|------|------|
| `MPI_Comm`, `MPI_Datatype`, `MPI_Message`, `MPI_Errhandler` | `int`, 4 bytes | pointer to an opaque struct, 8 bytes on 64-bit | pointer-width token (`struct MPI_ABI_Comm *`) |
| `MPI_COMM_WORLD` / `_SELF` / `_NULL` | `0x44000000` / `0x44000001` / `0x04000000` | `&ompi_mpi_comm_world` / `&ompi_mpi_comm_self` / `&ompi_mpi_comm_null` (exported data symbols) | `0x101` / `0x102` / `0x100` |
| `MPI_BYTE`, `MPI_UINT8_T` | `0x4c00010d`, `0x4c00013b` | `&ompi_mpi_byte`, `&ompi_mpi_uint8_t` | `0x247`, `0x241` |
| `MPI_MESSAGE_NULL` / `_NO_PROC` | `0x2c000000` / `0x6c000000` | `&ompi_message_null` / `&ompi_message_no_proc` | `0x128` / `0x129` |
| `MPI_ERRORS_ARE_FATAL` / `_RETURN` | `0x54000000` / `0x54000001` | `&ompi_mpi_errors_are_fatal` / `&ompi_mpi_errors_return` | `0x141` / `0x143` |
| `MPI_STATUS_IGNORE` | `(MPI_Status *)1` | `(MPI_Status *)0` | `(MPI_Status *)0` |
| `MPI_Status` | 20 bytes: `count_lo, count_hi_and_cancelled, MPI_SOURCE@8, MPI_TAG@12, MPI_ERROR@16` | 24 bytes: `MPI_SOURCE@0, MPI_TAG@4, MPI_ERROR@8, _cancelled@12, size_t _ucount@16` | 32 bytes: `MPI_SOURCE@0, MPI_TAG@4, MPI_ERROR@8, MPI_internal[5]` |
| `MPI_ANY_SOURCE` / `MPI_PROC_NULL` | `-2` / `-1` | `-1` / `-2` | `-1` / `-3` |
| `MPI_MAX_ERROR_STRING` / `_LIBRARY_VERSION_STRING` | 512 / 8192 | `OPAL_MAX_ERROR_STRING` / 256 | 512 / 8192 |

### C API representation (D4)

The C entry takes the **raw C handle**, widened to 64 bits:

```c
typedef struct tfe_mpi_comm tfe_mpi_comm;          /* opaque, owned by tenferro */

/* Process-local; no message is sent. Detects the library (D3), then checks the
   handle as described below. */
tfe_status_t tfe_mpi_comm_attach(uint64_t comm_handle, tfe_mpi_comm **out,
                                 tfe_status_t *status);
tfe_status_t tfe_mpi_comm_detach(tfe_mpi_comm *comm, tfe_status_t *status);
/* Family, library path, version string; for an attached comm also the provided
   thread level and whether its error handler is MPI_ERRORS_ARE_FATAL. Callable
   before MPI_Init (D12). */
tfe_status_t tfe_mpi_library_info(char *buf, size_t len, size_t *needed,
                                  tfe_status_t *status);
tfe_status_t tfe_mpi_comm_info(const tfe_mpi_comm *comm, char *buf, size_t len,
                               size_t *needed, tfe_status_t *status);
/* send / recv over a compact column-major host buffer + dtype code + shape;
   same wire protocol and errors as mpi-transport.md. */
```

**Encoding of `comm_handle`.** The handle is the host library's `MPI_Comm`, converted
the way C converts it to an unsigned 64-bit integer:

- For **pointer handles** (Open MPI ABI, standard ABI), the value is the address.
- For **`int` handles** (MPICH ABI), the low 32 bits are the handle. The upper 32 bits
  must be all zero (zero-extended) or all one (sign-extended); anything else is
  `TFE_INVALID_ARGUMENT`. Both forms are accepted because the hosts differ: mpi4py's
  `comm.handle` is `<Py_uintptr_t> self.ob_mpi`, which sign-extends a negative `int`
  such as the user communicator `0x84000002`. The Julia binding passes
  `UInt64(reinterpret(UInt32, comm.val))`.

Host side: mpi4py ≥ 4 has `comm.handle`, which is public API. MPI.jl's `comm.val` holds
the library's `MPI_Comm` (`Cint`, `Ptr` or `Cuint` depending on the ABI), and the
binding widens it by type.

**Why not the Fortran handle (revision 1).** The audit showed three problems:

- Open MPI's `MPI_Comm_f2c` returns `NULL` for an out-of-range index without invoking an
  error handler (`ompi/mpi/c/comm_f2c.c`). The next call on that `NULL` then goes
  through whatever the host installed.
- The standard ABI has no `MPI_Comm_c2f`, only `MPI_Comm_toint`/`fromint`, and mpi4py
  master's `py2f()` calls `MPI_Comm_toint`.
- MPI.jl's generated API has no `MPI_Comm_c2f` binding.

A raw handle has none of these problems, and the reason it is safe is D2: the library
that reads it is the library that produced it.

**What attach checks, and what it cannot.** Before the handle is passed to any MPI
call:

- it is not `0`;
- it is not the family's `MPI_COMM_NULL`;
- an `int` handle satisfies the upper-bits rule.

Then attach makes these calls on it, in this order:

- `MPI_Comm_test_inter`; an inter-communicator is refused;
- `MPI_Comm_rank` and `MPI_Comm_size`, recorded;
- `MPI_Comm_get_errhandler`, compared with the family's `MPI_ERRORS_ARE_FATAL`, then
  `MPI_Errhandler_free` on the returned reference, as the standard requires;
- `MPI_Comm_get_attr(MPI_TAG_UB)`, recorded.

**Revision 1's rank/size check is withdrawn.** It is not an identity check: two
same-size communicators from `MPI_Comm_split` pass it. No local check can tell a
wrong-but-live communicator from the right one. That is the host's responsibility, as it
is for every MPI library API that takes an `MPI_Comm`.

What tenferro does guarantee is narrower and holds: it interprets the handle exactly as
the host's library does, because it calls that library. A **dead or garbage** handle
reaches the first call above and is handled by the host's error handler:

- under `MPI_ERRORS_RETURN` (the default in **both** MPI.jl, `errors_return=true`, and
  mpi4py, `rc.errors = "exception"`), it becomes an attach error;
- under `MPI_ERRORS_ARE_FATAL`, the job aborts, exactly as if the host had made the
  call;
- on the pointer ABIs a garbage address can fault before any handler runs.

The document says this instead of claiming a safety net.

### Inside Rust (D5)

`mpi-native-sys` exposes the same Rust-facing types `mpi-rt-sys` exposed today: handles
as `usize`, a neutral `MPI_Status` with public `MPI_SOURCE`/`MPI_TAG`/`MPI_ERROR`
fields. rsmpi-rt's `ffi` module therefore needs one new `cfg` arm and no other source
change. The family is chosen once, at load time:

- **MPICH ABI**: the function pointers are typed with `c_int` handles. A neutral handle
  holds the 32 bits zero-extended (`h as u32 as usize`). Converting back requires the
  upper bits to be zero, and a non-zero upper half is an internal error, never a
  truncation.
- **Open MPI ABI and standard ABI**: the function pointers are typed with
  `*mut c_void` handles, and the neutral handle is the address. The two families differ
  only in their constant tables and status layout.
- An out-parameter handle (the `MPI_Message *` of `MPI_Mprobe`, the
  `MPI_Errhandler *` of `MPI_Comm_get_errhandler`) is received into a stack slot of the
  family's width, then widened.

## 3. Who owns MPI (D6, D10, D11)

The host owns the MPI lifecycle and the communicator; tenferro borrows both.

**May:** `MPI_Get_library_version`, `MPI_Abi_get_version`, `MPI_Initialized`,
`MPI_Finalized`, `MPI_Query_thread`, `MPI_Is_thread_main`, `MPI_Comm_rank`,
`MPI_Comm_size`, `MPI_Comm_test_inter`, `MPI_Comm_get_errhandler` with
`MPI_Errhandler_free` on the returned reference, `MPI_Comm_get_attr` for the predefined
`MPI_TAG_UB` (a read), `MPI_Send`, `MPI_Mprobe`, `MPI_Mrecv`, `MPI_Get_count`,
`MPI_Error_string`. All of these run on the borrowed communicator with tags the caller
passes.

**May not:** `MPI_Init*`, `MPI_Finalize`, `MPI_Abort`, `MPI_Comm_dup` or any
constructor, `MPI_Comm_free`, `MPI_Comm_disconnect`, `MPI_Comm_set_errhandler`,
attribute writes, keyvals, `MPI_Comm_set_name`, wildcard receives. The
`no_mpi_lifecycle` source scan extends to the new names.

### Lifetime

- **No `MPI_Comm_free` on drop.** rsmpi-rt's `CommunicatorHandle::User` calls
  `MPI_Comm_free` when dropped, and `SimpleCommunicator::from_raw` produces that variant
  for a host sub-communicator. `tfe_mpi_comm` therefore holds the communicator in
  `ManuallyDrop`. A `from_raw_borrowed` constructor in rsmpi-rt is the cleaner shape
  (upstream consequence U3) but is not required.
- **Attach preconditions:** `MPI_Initialized` is true and `MPI_Finalized` is false,
  then the handle checks in §2.
- **Host finalized first:** every C API entry calls `MPI_Finalized` (one of the few
  calls allowed after finalize) and returns "the host finalized MPI; this communicator is
  no longer usable" without any other MPI call.
  - `tfe_mpi_comm_detach` after finalize frees only Rust memory. That covers a Julia
    finalizer that runs after MPI.jl's default `finalize_atexit` hook.
  - `Drop` makes no MPI call at all, and the library stays mapped (§1 step 5). A late
    drop therefore never calls into unmapped code.
  - A host that finalizes **concurrently** with a tenferro call on another thread is
    outside the MPI standard; that is documented as the host's bug.
- **`MPI_Error_string` is called only at the failure site**, while MPI is known to be
  initialized: `MPI_Finalized` was checked at entry, and concurrent finalize is out of
  contract. Its text is captured into `Error::Mpi { code, message }` at once. Nothing
  formats an MPI error later, so no `MPI_Error_string` call can happen after finalize,
  whatever the standard version says about calling it then.

### Error handler

The communicator's error handler belongs to the host, and tenferro never changes it.
Attach records whether it is `MPI_ERRORS_ARE_FATAL` and reports it in
`tfe_mpi_comm_info`.

Every MPI error tenferro can predict is turned into a typed tenferro error **before**
the MPI call:

- a negative tag, or one above `MPI_TAG_UB`;
- a destination outside `0..size`;
- a self-send;
- an element count above `i32::MAX` (already v1).

So under `MPI_ERRORS_ARE_FATAL` only a genuinely unexpected MPI failure aborts the job,
such as a lost network peer or a dead handle. That is the behaviour the host chose for
its own calls, and tenferro does not soften it.

Under `MPI_ERRORS_RETURN`, `mpi-native-sys` checks every return code (`MPI_SUCCESS` is
0 in every family), and the native backend returns `Error::Mpi`. This needs the
`MpiBackend` methods to return `Result` (consequence T1).

### Threads (D11)

tenferro cannot see the host's MPI calls, so it cannot serialize against them. What it
guarantees, and what the host must guarantee, depends on the level the host's MPI
provides. Attach records that level with `MPI_Query_thread`.

| Provided level | tenferro checks | Host must guarantee | Default in |
|---|---|---|---|
| `MPI_THREAD_MULTIPLE` | nothing extra | distinct tags in flight (v1 contract) | mpi4py (`rc.thread_level = "multiple"`) |
| `MPI_THREAD_SERIALIZED` | nothing it can check | no other thread in MPI while a tenferro call runs | MPI.jl (`Init(; threadlevel=:serialized)`) |
| `MPI_THREAD_FUNNELED`, `MPI_THREAD_SINGLE` | `MPI_Is_thread_main` on every call; an error from any other thread | MPI calls only from the main thread | — |

At `SERIALIZED` a C library cannot help: a blocking `MPI_Send` inside tenferro and an
`MPI.Send` in another Julia thread would overlap. The rule is therefore enforced in the
**host binding**, which can see the host's threads:

- **Julia binding** (and demo): attach is refused unless `Threads.maxthreadid() == 1` or
  the provided level is `MPI_THREAD_MULTIPLE`. With one Julia thread, a `ccall` into
  tenferro blocks the only thread, and no other Julia task can run until it returns.
  The default `:serialized` is then race-free by construction, not by documentation. The
  demo runs `julia --threads=1,0` and asserts `Threads.maxthreadid() == 1`. The `,0`
  matters: it disables the interactive thread pool, which recent Julia versions enable
  by default.
- **Python binding** (and demo): requires `MPI_THREAD_MULTIPLE`, which is mpi4py's
  default, and refuses below it. The GIL is no substitute. mpi4py releases the GIL in
  blocking calls, and `ctypes` releases it around foreign calls, so two Python threads
  can be inside MPI at once.

### Panics (D10)

`mpi-native-sys` defines all of rsmpi-rt's `ffi` names. Those outside the carried set
are stubs that panic, about 120 functions (§4). rsmpi-rt also asserts in places. A panic
that unwinds out of an `extern "C"` function aborts the process (Rust ≥ 1.81), and the
process belongs to Julia or Python. So:

- Every `extern "C"` entry of `tenferro-mpi-capi` runs its body inside
  `std::panic::catch_unwind` and maps a panic to `TFE_INTERNAL_ERROR`, with the message
  in the thread-local last error. This is `capi.md` principle 2 (and
  `capi-error-handling.md` "Panics"), applied here.
- The cdylib must be built with `panic = "unwind"`; `panic = "abort"` would make every
  stub a host abort. A CI check asserts the profile.
- Stubs panic with a typed payload, `UnsupportedMpiCall { name }`, which the boundary
  maps to the message `"MPI_<name> is outside tenferro's native MPI surface"`.
- After a caught panic the `tfe_mpi_comm` is **poisoned**: every later call returns
  `TFE_INTERNAL_ERROR` without touching MPI. The peer rank may be left blocked in a
  matching call, because a panic mid-protocol is not recoverable on the wire. That is
  documented, not hidden.
- The scenarios count stub hits through a test-only counter in `mpi-native-sys` and
  assert zero, so a tenferro code path that reaches a stub fails CI instead of a user's
  session.
- Loader failures never panic. The C entry runs detection (D12) and returns an error
  **before** rsmpi-rt is touched. rsmpi-rt's `ffi` functions are only reachable after a
  successful load.

### Tags and a dedicated communicator

tenferro shares the host's communicator without duplicating it. A host message on a tag
that a tenferro call is using is therefore not detected; the call blocks forever, it
does not fail. The v1 contract already says this, and here it crosses a language
boundary. The recommended pattern, which both demos use, is for the host to give
tenferro a **communicator of its own**: `MPI.Comm_dup(comm)` or `comm.Dup()`, created
and freed by the host. Tag collisions then become impossible, and tenferro still never
creates a communicator.

## 4. Symbol resolution and ABI risk

Every function is resolved with `dlsym` on the reference to the identified object (D2)
and cached in a `OnceLock`, as `mpi-rt-sys` does today. Constants come from three
sources:

- Open MPI: the **addresses** of the exported data symbols, read once;
- MPICH ABI: integer literals in a per-family table;
- standard ABI: integer literals from `mpi.h` of the ABI stubs.

What differs across families, and why the design does not depend on it:

| Differs | How it is handled |
|---------|-------------------|
| Handle width and kind | Per-family function-pointer types (D5); one neutral `usize` above them. |
| Predefined constants | Per-family table (§2). |
| `MPI_Status` layout and size | Passed to C as an opaque 64-byte, 8-aligned buffer, bigger than every layout (20, 24, 32, and MPIABI's 40). The count comes from `MPI_Get_count` on that buffer and never from its fields. tenferro already knows the source and tag (explicit `process_at_rank` and tag); the per-family offsets only fill rsmpi-rt's neutral `MPI_SOURCE`/`MPI_TAG` for API completeness. |
| `MPI_STATUS_IGNORE` | Per-family sentinel substituted for the neutral one. |
| `MPI_ANY_SOURCE`, `MPI_PROC_NULL` | Not used by tenferro, whose API has no wildcard. Values are in the table for rsmpi-rt's `any_process()`. |
| `MPI_MAX_*_STRING` | Buffers sized to the largest value (8192 for the version string; 1024 for the error string, twice the 512 of MPICH and the standard ABI). |
| Fortran handles | Not used (D4). |

What the design **does** rely on being identical: `MPI_SUCCESS == 0`, `int` counts,
ranks and tags, the C calling convention of every listed function, and the
pre-init/post-finalize callability of `MPI_Get_library_version`, `MPI_Initialized` and
`MPI_Finalized`. The MPI standard fixes all of these. `MPI_Abi_get_version`'s
callability is A10.

### rsmpi-rt's compile-time surface

rsmpi-rt's `ffi` module references about 134 `MPI_*` functions and 187 `ffi::` names in
total (count in [Verification](#verification)), so `mpi-native-sys` must define all of
those names. It is generated from the same spec as `mpi-rt-sys` (`gen_rust.py`):

- the **carried set** (§3 "may" list, plus what rsmpi-rt's communicator construction
  calls: `MPI_Comm_get_parent`, `MPI_Comm_compare`) gets real per-family thunks;
- every other function, roughly 120, is a stub that panics with
  `UnsupportedMpiCall { name }`, which D10 catches at the C boundary.

tenferro-mpi reaches only the carried set, and the scenarios assert zero stub hits.
Q3 records whether to cfg-gate the rest at compile time instead.

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
2. A source scan (an extension of `no_mpi_lifecycle`) fails if `MPI_RT_LIB` or
   `libmpiwrapper` appears under `crates/` or `scripts/`, or if `MPIABI_` appears
   anywhere except the one loader function that rejects it. Worklogs are excluded.
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
| MPI.jl with `MPItrampoline_jll` + tenferro | MPI.jl with the default `MPICH_jll`, `MPIABI_jll`, or `MPIPreferences.use_system_binary()` |

A leftover `MPI_RT_LIB` in the environment is ignored.

### What a user on an unsupported family loses

Before: a site MPI that MPIwrapper could be built against (NEC MPI, HPE MPT, anything
with an `mpi.h`) worked, **provided the program used MPItrampoline end to end**. For a
host language that meant MPI.jl with `MPItrampoline_jll`, since mpi4py and MPI.jl with a
system binary hold native handles that the wrapper path never accepted. After: attach or
standalone initialization fails with the error in §1 at start-up. The loss is real for
two groups: Rust programs on exotic MPIs, and MPI.jl users who chose MPItrampoline. The
standard-ABI family narrows the first group to MPIs that ship neither a supported native
ABI nor a standard-ABI build.

## 6. Testing without a cluster

**Unit tests (no MPI, in the default lanes):**

- classification of recorded version strings, one per table row, including:
  - the Cray string `MPI VERSION    : CRAY MPICH version 7.7.10 (ANL base 3.2)`, which
    must classify through the substring rule;
  - an MPICH string containing `--enable-mpi-abi`;
  - the `mpi_abi_wrapper 1.2.0 … wrapping: Open MPI v5.0.10` sample;
  - `Intel(R) MPI Library 2018 Update 3`, accepted as > 2014;
  - `HPE MPT 2.23`, rejected;
  - `MPIwrapper Version:`, an NEC sample and garbage;
- the detection order. With a mocked object list, a standard-ABI library whose string
  matches no native row must classify as standard ABI, not `UnsupportedMpi`; this is
  the audit's finding 2 as a test. Also MPIwrapper detected before ambiguity, and
  `abi_major = -1` falling through to the native steps;
- witness agreement and disagreement;
- the `comm_handle` encoding: zero- and sign-extended `0x84000002` both accepted, mixed
  upper bits refused, `0` and each family's `MPI_COMM_NULL` refused before any MPI call;
- the layout table as `const` assertions (status sizes 20, 24 and 32; source offsets
  8, 0 and 0; buffer of 64 ≥ each);
- D12: a failed detection followed by a successful one returns the success;
- D10: a forced stub hit through the C entry returns `TFE_INTERNAL_ERROR`, does not
  abort, and poisons the object.

**Loader integration (one process, singleton MPI, the test harness plays host):** for
each library available here:

- Open MPI 4.1.6 from apt;
- MPICH 5.0.2 from the Julia `MPICH_jll` artifact;
- Intel MPI 2021.19 from oneAPI;
- `libmpi_abi.so.1` from `MPIABI_jll`, once downloaded.

For each:

- load it and check its family;
- call `tfe_mpi_library_info` before `MPI_Init` (it fails without poisoning, D12),
  then `MPI_Init`;
- attach `MPI_COMM_WORLD` by its raw handle and check rank 0 / size 1 and the reported
  error handler;
- `MPI_Finalize`, then check that a transport call returns the finalized error and does
  not crash.

**Two ranks:** `MPIEXEC=<family's launcher> scripts/run-mpi.sh rt` runs the existing
eight `mpi_p2p_runtime` scenarios (seven dtypes, rank 0 and empty extents, typed and
erased crossings, rejections with recovery, aborts, tags, invalid `bool`, foreign
frames) unchanged, once per family, plus the zero-stub-hit assertion. The hosted lane
runs Open MPI and MPICH. Intel MPI and the standard ABI are local evidence in the work
log.

**Julia demo** (`mpiexec -n 2 julia --threads=1,0 --project demo/mpi.jl`, run with
`MPICH_jll`, with the system Open MPI, and with `MPIABI_jll`). MPI.jl's default
`Init()` is used, so the level is `:serialized` and `errors_return=true`. The demo
asserts:

1. `Threads.maxthreadid() == 1`, the condition under which the default level is
   race-free (D11). A second run with `--threads=2` must be refused by the binding;
2. `tfe_mpi_library_info` reports the family MPIPreferences chose;
3. attach on `tcomm = MPI.Comm_dup(MPI.COMM_WORLD)` succeeds, and
   `tfe_mpi_comm_info` reports rank and size equal to MPI.jl's and an error handler that
   is not fatal;
4. rank 0 sends tensor `A` through tenferro on `tcomm` and the same bytes through
   `MPI.Send` on `MPI.COMM_WORLD`; rank 1 receives both and checks them bit for bit.
   Host and tenferro traffic run at the same time on separate communicators;
5. after `MPI.Finalize()`, a tenferro call returns the finalized error, and a finalizer
   that detaches afterwards does not crash;
6. negative case: with `MPItrampoline_jll` selected, attach fails with
   `MpiTrampolineNotSupported`.

**mpi4py demo** (`mpiexec -n 2 python demo/mpi.py`, default `rc`: `thread_level =
"multiple"`, `errors = "exception"`):

- items 2–5 as above, using `comm.Dup()` and `comm.handle`, with NumPy for the
  reference;
- the binding refuses to attach when MPI was initialized with
  `mpi4py.rc.thread_level = "serialized"`;
- an MPI error under `MPI_ERRORS_RETURN` comes back as tenferro's `Mpi` error and is
  not swallowed. The trigger is attaching a communicator the host has already freed;
  under MPICH, the freed `int` handle fails `MPI_Comm_test_inter` with an error code.
  This item runs on the MPICH leg only: on the pointer ABIs a freed handle is a dangling
  pointer, and calling MPI with it is undefined behaviour.

## Rejected alternatives

- **Keep MPItrampoline as a fallback:** superseded by the maintainer decision of
  2026-11-26; two runtime paths double the test matrix to serve users who do not exist
  yet.
- **Pass the Fortran handle (revision 1):** Open MPI's `f2c` returns `NULL` silently on
  a stale index, the standard ABI has no `c2f`, and MPI.jl has no `c2f` binding.
- **Claim a handle-identity check (revision 1's rank/size):** no local check can tell
  two live, same-size communicators apart; the design now says so instead.
- **A tenferro-local `MpiBackend` with its own tiny `dlsym` table:** it bypasses
  rsmpi-rt, so Rust callers' `Process` values cannot reach it, and the process would
  hold two MPI bindings.
- **`dlsym(RTLD_DEFAULT)` only:** it misses a host `libmpi` that was loaded
  `RTLD_LOCAL`, the normal case for Python.
- **`dlopen` a fresh `libmpi` by name in the attach path:** it can load a second MPI
  next to the host's, which crashes.
- **Family from the version string alone:** a trampoline that forwards the underlying
  string would pass the wrong handles silently.
- **Family from `MPI_Abi_get_version` presence:** native MPICH 5, Intel MPI 2021.19
  and Open MPI `main` all export it.
- **Reject the standard ABI in v1 (revision 1):** it is the cheapest family (fixed
  literal constants, no witness symbols to trust), and MPI.jl and mpi4py already
  support it.
- **A finalize-time attribute callback on `MPI_COMM_SELF` to learn about finalize:**
  it needs a keyval (process-global state) and a per-family callback ABI; a
  `MPI_Finalized` check per call is enough.
- **`MPI_Comm_dup` inside tenferro for tag isolation:** dup is collective, so attach
  would have to run on all ranks at once; the host dups instead (§3).
- **Set `MPI_ERRORS_RETURN` on the borrowed communicator:** it changes host state that
  outlives the call; the host chose its handler.
- **A tenferro-side lock for `MPI_THREAD_SERIALIZED`:** the host's own MPI calls do not
  take it, so it serializes nothing.
- **Rename the feature to `native`:** the name `runtime` is still accurate, and a
  rename breaks every manifest for no behaviour difference.

## Open questions (each with the decision it needs)

- **Q1. Closed in revision 2:** the standard ABI is a v1 family (D1). What remains open
  is ABI version 2: refuse it until it exists (`UnsupportedAbiVersion`) or accept it
  speculatively. *Decision taken:* refuse; there is nothing to test against.
- **Q2. A statically linked host MPI** (no shared object to find). *Decision needed:*
  build `tfe_mpi_comm_attach_vtable`, where the host passes the ~20 function pointers
  in the carried set, now or on the first report. Until then: `NoMpiInProcess`. The
  removed wrapper path never handled this case either.
- **Q3. Unsupported rsmpi-rt functions:** runtime panic stubs caught at the C boundary
  (this design) or compile-time `cfg` gating of rsmpi-rt's collectives, requests and
  attributes. *Decision needed:* will any consumer other than tenferro-mpi select the
  native backend before collectives are carried? If yes, gate.
- **Q4. Profilers:** resolving through the identified object's reference bypasses
  `LD_PRELOAD` interposers for tenferro's own traffic. *Decision needed:* accept that,
  or resolve `MPI_*` through `RTLD_DEFAULT` first when the library is in global scope.
- **Q5. Where `MpiBackend` becomes fallible (T1):** a breaking change to the
  `engine`-only trait. *Decision needed:* accept the break, since there are no known
  external implementers, or add a parallel fallible trait.
- **Q6. macOS:** there is no `dl_iterate_phdr`; the walk needs
  `_dyld_image_count`/`_dyld_get_image_name`. *Decision needed:* is the issue's "laptop"
  run a Mac? If so, macOS is v1 scope; otherwise Linux only.
- **Q7. Multi-threaded Julia at `:serialized`:** the binding refuses it (D11). *Decision
  needed:* is that acceptable for the paper's demo, or must the Julia binding offer a
  lock that user code wraps around its own MPI.jl calls? Such a lock is opt-in and
  cannot cover MPI.jl's internal calls.

## 7. What would make us regret the deletion, and the recovery cost

Here are the conditions under which removing MPItrampoline turns out wrong.

| Regret condition | Would the old path have helped? | Recovery and cost |
|------------------|--------------------------------|-------------------|
| A site MPI with its own ABI (NEC MPI on SX-Aurora, HPE MPT, a future vendor MPI) is where the paper's cluster run or a user must run. | Only if that user ran MPItrampoline end to end. A mpi4py or MPI.jl-system host still fails. | First check whether the vendor ships a standard-ABI build; if so, it already works. Otherwise add a family table: about 1–2 days per family when the library exports its constants or uses integer handles, plus a site run to verify. MPI.jl's `mpt.jl` already documents HPE MPT's constants. |
| **The standard ABI becomes the common case before our native tables age out.** This is the direction MPICH (`--enable-mpi-abi`), Open MPI (`mpi_abi_wrapper`), MPI.jl (`MPIABI_jll`) and mpi4py (`_mpiabi`) are already moving. | No: MPItrampoline's MPIABI is a different ABI from the MPI 5.0 one. | No regret, by design: the standard ABI is a v1 family (D1). Its cost was one constant table from `mpi.h`, a 32-byte status that fits the opaque buffer, pointer handles that reuse the Open MPI path's function-pointer types, and no `c2f`, which D4 no longer needs: about 1–2 days. Revision 1 left this row out; that was an omission, because this is the path that makes the per-family tables matter less over time. |
| The host's MPI is statically linked and exports no symbols. | No: MPItrampoline also needs the host on MPItrampoline. | Q2's vtable entry: about 2–3 days, family-free, because the host supplies the functions. |
| Detection is silently wrong: an MPICH-derived library changes handle encoding, or a wrapper forwards both the version string and an `MPIR_` symbol. | Yes: MPIABI handles are family-free by construction. | There is no local identity check (§2), so this surfaces as MPI errors or crashes on the first call. Recovery: add a witness and a table row, a day. The standard ABI shrinks this risk for every implementation that ships it. |
| Many existing users depend on MPItrampoline. | Yes. | **Revert:** D9 keeps the deletion as one commit per repository, and `mpi-rt-sys 0.1.0` stays on crates.io. The raw-handle C API (D4) works over MPIABI handles too, since they are `uintptr_t`. Re-adding it as a second feature is about 1 week including the rsmpi-rt release. |

The bet is that the families in D1 cover the sites the paper and its readers use, and
that a host language with native handles (mpi4py, MPI.jl with a system or JLL MPI) is
the common case. The wrapper path could not serve that case anyway.

## Assumptions that would sink this design

- **A1.** The classification table in §1 identifies the family. It is tenferro's own,
  derived from MPIPreferences.jl `ad793d65`. Checked against the local Intel MPI, MPICH
  and Open MPI binaries; the Cray, Fujitsu, Spectrum, MVAPICH and HMPT strings are
  MPIPreferences' samples, not observed here.
- **A2.** Every MPICH-ABI implementation uses 32-bit `int` handles with the constants in
  §2. Verified for MPICH 5.0.2 and Intel MPI 2021.19 headers; MVAPICH, Cray MPICH and
  HPE HMPT rely on the MPICH ABI initiative and are **unverified**.
- **A3.** Open MPI 4.x and 5.x handles are addresses of exported `ompi_mpi_*` data
  symbols. Verified in the 4.1.6 binary and the 4.1.6 and 5.0.8 headers.
- **A4.** `MPIR_Err_return_comm` is exported by every MPICH-derived library and by
  nothing else. Verified present in MPICH 5.0.2 and Intel MPI 2021.19, and absent in Open
  MPI 4.1.6; **unverified** for Cray MPICH, MVAPICH and `libmpitrampoline`.
- **A5.** `MPI_Abi_get_version` returns `abi_major ≥ 1` exactly on standard-ABI
  builds. The audit reports MPICH generates `-1, -1` for native builds and Open MPI
  documents `-1`. **Not run here**; the loader integration test runs it against
  `MPICH_jll` and `MPIABI_jll` before anything relies on it.
- **A6.** The host's MPI is one shared object, visible to `dl_iterate_phdr`, that
  exports `PMPI_Get_library_version` (native) or `MPI_Abi_get_version` (standard ABI).
- **A7.** The host can produce the raw handle publicly: mpi4py ≥ 4's `comm.handle`
  (verified in mpi4py master `a542c4d2`) and MPI.jl's `comm.val`.
- **A8.** No one outside this repository uses the MPItrampoline path. On crates.io,
  `mpi-rt-sys` has rsmpi-rt as its only reverse dependency and rsmpi-rt has none.
  Unpublished users cannot be seen.
- **A9.** A process that has a standard-ABI library loaded calls MPI through it, even
  when a native library is also loaded beneath it, as with `mpi_abi_wrapper`. If a host
  loads both but calls the native one, tenferro would read native handles as ABI tokens.
  The pointer ABIs fail on the first call; there is no silent success path, because the
  two families' constants do not overlap.
- **A10.** `MPI_Abi_get_version` may be called before `MPI_Init` and after
  `MPI_Finalize`, like `MPI_Get_version`. If not, step 3 of the detection must run only
  between init and finalize, which the attach path already guarantees, and only
  `tfe_mpi_library_info` before init would lose the standard-ABI answer.

## Verification

Commands run on 2026-11-26 (Linux x86_64) to check the claims above.

Revision 1:

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
grep -nE "typedef int MPI_(Comm|Datatype|Message|Request|Fint);|define MPI_(COMM_(WORLD|SELF|NULL)|BYTE|UINT8_T|MESSAGE_NULL|MESSAGE_NO_PROC|ANY_SOURCE|PROC_NULL|STATUS_IGNORE|MAX_ERROR_STRING|MAX_LIBRARY_VERSION_STRING|ERRORS_ARE_FATAL|ERRORS_RETURN) |MPI_Comm_f2c" \
  /opt/intel/oneapi/mpi/2021.19/include/mpi.h \
  ~/.julia/artifacts/1a3503e280039bd45ad4bfc9f5343a76c0574d67/include/mpi.h   # MPICH 5.0.2 (MPICH_jll)
awk '/typedef struct MPI_Status/,/} MPI_Status;/' <same headers>
# Open MPI: exported constants and witness, from the installed 4.1.6 binary
nm -D --defined-only /usr/lib/x86_64-linux-gnu/libmpi.so.40 | grep -E "ompi_mpi_(comm_world|comm_self|comm_null|byte|uint8_t)|ompi_message_(null|no_proc)|MPIR_"
# Open MPI headers (not installed locally): mpi.h.in at v4.1.6, v5.0.8 and main
gh api "repos/open-mpi/ompi/contents/ompi/include/mpi.h.in?ref=v4.1.6" -q .content | base64 -d \
  | grep -nE "MPI_Comm;|MPI_Message;|MPI_(ANY_SOURCE|PROC_NULL|STATUS_IGNORE|MAX_LIBRARY_VERSION_STRING|ERRORS_ARE_FATAL|ERRORS_RETURN) |MPI_Fint"
# Exports and witnesses across the three libraries
for L in <MPICH_jll libmpi.so.12> /opt/intel/oneapi/mpi/2021.19/lib/libmpi.so.12 /usr/lib/x86_64-linux-gnu/libmpi.so.40; do
  nm -D --defined-only $L | awk '{print $3}' | grep -E "^(MPIR_Err_return_comm|MPI_Abi_get_version|MPI_Comm_f2c|MPI_Mprobe|MPI_Mrecv|MPI_Get_count|MPI_Is_thread_main|ompi_mpi_comm_world)$"
done
strings <each libmpi> | grep -E "MPICH Version|Intel\(R\) MPI Library|Open MPI v"
grep -rn libmpi ~/.julia/packages/MPIABI_jll/*/src/wrappers/x86_64-linux-gnu*.jl          # libmpi_abi.so.1
grep -rn libmpi ~/.julia/packages/MPItrampoline_jll/*/src/wrappers/x86_64-linux-gnu*.jl   # libmpitrampoline.so.5
curl -s https://crates.io/api/v1/crates/mpi-rt-sys/reverse_dependencies   # rsmpi-rt only
curl -s https://crates.io/api/v1/crates/rsmpi-rt/reverse_dependencies     # none
grep -rnlE "mpi-rt-sys|MPI_RT_LIB|MPIwrapper|mpi_p2p_runtime" crates scripts docs .github   # deletion inventory, §5
```

Revision 2, added for the audit findings:

```bash
# MPIPreferences detection table at MPI.jl master ad793d65 (still versioned 0.1.12):
# MPIABI rows, Cray by occursin, Intel > v"2014", HPE MPT separate
gh api "repos/JuliaParallel/MPI.jl/contents/lib/MPIPreferences/src/MPIPreferences.jl?ref=master" -q .content | base64 -d \
  | sed -n 255,375p
# The locally installed release lacks the MPIABI branch (revision 1's source):
sed -n 270,372p ~/.julia/packages/MPIPreferences/wIB4a/src/MPIPreferences.jl
# MPI.jl: Init defaults, per-ABI constant files, no MPI_Comm_c2f, MPT handle type
gh api "repos/JuliaParallel/MPI.jl/contents/src/environment.jl" -q .content | base64 -d | sed -n 89,114p
gh api repos/JuliaParallel/MPI.jl/contents/src/api -q '.[].name'           # mpiabi.jl mpich.jl mpt.jl openmpi.jl ...
gh api repos/JuliaParallel/MPI.jl/contents/src/api/generated_api.jl -q .content | base64 -d | grep -c "Comm_c2f"   # 0
gh api repos/JuliaParallel/MPI.jl/contents/src/api/mpt.jl -q .content | base64 -d | grep "const MPI_Comm "      # Cuint
# MPI 5.0 standard ABI constants and status (mpi-forum/mpi-abi-stubs d224b2bf)
gh api "repos/mpi-forum/mpi-abi-stubs/contents/mpi.h?ref=main" -q .content | base64 -d \
  | grep -nE "MPI_ABI_Comm\*|define MPI_(COMM_(WORLD|SELF|NULL)|BYTE|UINT8_T|STATUS_IGNORE|ERRORS_ARE_FATAL|ERRORS_RETURN|MESSAGE_NULL|MESSAGE_NO_PROC|MAX_ERROR_STRING|MAX_LIBRARY_VERSION_STRING) |MPI_ANY_SOURCE|MPI_PROC_NULL|Comm_(toint|fromint|c2f)|MPI_internal"
# Open MPI f2c returns NULL on an out-of-range index, without an error handler
gh api "repos/open-mpi/ompi/contents/ompi/mpi/c/comm_f2c.c?ref=v5.0.8" -q .content | base64 -d | sed -n 40,59p
# mpi4py master a542c4d2: comm.handle is <Py_uintptr_t> ob_mpi; py2f uses MPI_Comm_toint; rc defaults
gh api repos/mpi4py/mpi4py/contents/src/mpi4py/MPI.src/Comm.pyx -q .content | base64 -d | sed -n 55,71p
gh api repos/mpi4py/mpi4py/contents/src/mpi4py/MPI.src/objmodel.pxi -q .content | base64 -d | sed -n 370,372p
gh api repos/mpi4py/mpi4py/contents/src/mpi4py/__init__.py -q .content | base64 -d | grep -nE "thread_level =|errors ="
```

Results that the design depends on:

- **MPICH ABI.** MPICH 5.0.2 and Intel MPI 2021.19 have `typedef int` for every handle.
  Their constants, error handlers (`0x54000000/1`), string maxima (512/8192) and
  `MPI_Status` (20 bytes, source at 8) match §2.
- **Open MPI.** 4.1.6 exports the `ompi_mpi_*`/`ompi_message_*` symbols. Its only `MPIR_`
  symbols are `MPIR_debug_gate`, `MPIR_dll_name` and `MPIR_debug_typedefs_sizeof`, and it
  lacks `MPIR_Err_return_comm`. Its headers give pointer handles, `MPI_ANY_SOURCE -1`,
  `MPI_PROC_NULL -2`, `MPI_STATUS_IGNORE 0` and a 24-byte status with source at 0.
  `MPI_Comm_f2c` returns `NULL` for an out-of-range index.
- **Standard ABI v1 (stubs `d224b2bf`).** Handles are `struct MPI_ABI_Comm *` tokens and
  `MPI_COMM_WORLD` is `0x101`. The status is 32 bytes with source at 0, there is no
  `MPI_Comm_c2f` (only `toint`/`fromint`), and `MPI_PROC_NULL` is `-3`.
- **`MPI_Abi_get_version`.** It is exported by MPICH 5.0.2's native `libmpi.so.12` and by
  Intel MPI 2021.19, and declared by Open MPI `main`; presence is not a signal.
- **MPIPreferences (`ad793d65`).** It has two `MPIABI` rows (`MPICH … --enable-mpi-abi`,
  `mpi_abi_wrapper`), matches Cray by `occursin`, accepts Intel `> v"2014"`, and gives HPE
  MPT its own ABI. The installed 0.1.12 release lacks the `MPIABI` rows.
- **MPI.jl.** `Init(; threadlevel=:serialized, finalize_atexit=true,
  errors_return=true)`. It has no `MPI_Comm_c2f` binding, and its `mpt.jl` has
  `MPI_Comm = Cuint`.
- **mpi4py master.** `comm.handle` is `<Py_uintptr_t> self.ob_mpi`, which sign-extends a
  negative `int` handle. `py2f()` calls `MPI_Comm_toint`. The `rc` defaults are
  `thread_level = "multiple"` and `errors = "exception"`.
- **mpi-rt-sys and rsmpi-rt.** `mpi-rt-sys 0.1.0` handles are `usize`, its status is 40
  bytes, and its loader panics without `MPI_RT_LIB`. rsmpi-rt 0.8.2 frees a `User`
  communicator on drop and reads `MPI_SOURCE`/`MPI_TAG` as struct fields.

Not checked here, and therefore listed under Assumptions:

- binaries for Cray MPICH, MVAPICH, HPE HMPT, Fujitsu and Spectrum MPI;
- `libmpitrampoline`'s exports (the artifact is not installed);
- `MPI_Abi_get_version`'s return value on each build (A5) and its callability (A10);
- Open MPI's `OPAL_MAX_ERROR_STRING`.

Nothing here was run under `mpiexec`, because this PR is design only.

## Audit response (revision 2)

| Audit finding | Change |
|---|---|
| 1. The family table's provenance is false (Cray, Intel, `MPIABI`, HPE MPT) | §1 table rewritten as tenferro's own, pinned to MPIPreferences `ad793d65`, with a **Differs** column. Cray matches by substring; Intel is aligned to `> 2014`; both `MPIABI` rows are added as the standard-ABI family. HPE MPT is rejected with the reason (own ABI, `Cuint` handles in `mpt.jl`), and HMPT is kept as MPICH ABI, matching MPIPreferences. |
| 2. The ordering makes the ABI guard unreachable | D3 and §1 reordered: MPIwrapper → `MPI_Abi_get_version` value → string → witness. A standard-ABI library is classified before any native witness is consulted; a unit test pins it. |
| 3. D4's safety argument does not hold | Fortran handle replaced by the raw C handle; rank/size check withdrawn and the non-claim stated; behaviour under `ERRORS_RETURN`, `ERRORS_ARE_FATAL` and garbage pointers spelled out (§2). MPI.jl's `:serialized` default answered in D11/§3 "Threads": the Julia binding requires one Julia thread or `MULTIPLE`, and the demo asserts it. |
| Panic across FFI | D10 and §3 "Panics": `catch_unwind` on every entry, `panic = "unwind"` asserted, typed stub payload, poisoning, zero-stub-hit assertion. |
| `OnceLock` trippable before MPI is loaded | D12: only success is cached. |
| `MPI_Error_string` illegal after finalize | Called only at the failure site, after an `MPI_Finalized` check, and captured at once (§3 "Lifetime"). |
| Open MPI destroys the f2c table at finalize | Moot: no Fortran handles (D4). |
| Host `dlclose` → late `Drop` into unmapped code | The `RTLD_NOLOAD` reference keeps the library mapped, and `Drop` makes no MPI call (§1 step 5, §3). |
| `ERRORS_ARE_FATAL` aborting the host | Reported at attach; predictable errors are validated locally first; tenferro does not change the handler (§3 "Error handler"). |
| A stray tag is a permanent hang | Stated as such; mitigated by the host passing a dup it owns, as both demos do (§3 "Tags"). |
| The standard ABI is absent from §7 | It is a v1 family now, and has its own §7 row with the real cost. |

## Implementation notes (#2735)

What the implementation checked, and what it settled where this note was silent.

### A5 and A10: verified

Run on 2026-11-26 with `MPI_Abi_get_version` called through `ctypes` (the call needs no MPI header):

| Library | `MPI_Initialized` | Before `MPI_Init` | After `MPI_Init` | After `MPI_Finalize` |
|---|---|---|---|---|
| `libmpi_abi.so.1` (MPIABI_jll 1.0.1, MPICH 5.0.2 `--enable-mpi-abi`) | 0 | rc 0, **(1, 0)** | rc 0, (1, 0) | rc 0, (1, 0) |
| MPICH_jll 5.0.2 `libmpi.so.12` | 0 | rc 0, **(-1, -1)** | rc 0, (-1, -1) | rc 0, (-1, -1) |
| Intel MPI 2021.19 `libmpi.so.12` | 0 | rc 0, **(-1, -1)** | rc 0, (-1, -1) | rc 0, (-1, -1) |
| Open MPI 4.1.6 `libmpi.so.40` | | symbol not exported | | |

So A5 holds for every build that was available (native builds return `-1`, the standard-ABI
build returns `1`), and A10 holds: the call works before init and after finalize. Open MPI 5.x and
`mpi_abi_wrapper` were not available to run. The loader does not rely on A10 for correctness: it
treats a non-zero return code as "not a standard ABI".

`libmpi_abi.so.1` also exports `MPIR_Err_return_comm` and `PMPI_Get_library_version`. Without the
detection order of D3 it would be classified as a native MPICH-ABI library, so the unit test that
pins the order uses exactly that shape.

### Settled where the design was silent

- **Standard-ABI thread levels are not 0..3.** `MPI_THREAD_FUNNELED/SERIALIZED/MULTIPLE` are
  `1024/2048/4096` there. All thread constants go through the per-family table, never literals.
- **Open MPI's `MPI_TAG_UB` is `INT_MAX`.** A tag above it does not exist, so the local tag check
  uses the attribute's value and the scenario only tests it where it is smaller.
- **`dlsym` on a handle also searches the handle's dependencies.** A library that merely links
  `libmpi_abi` would look like a second MPI, so a symbol counts only if its address lies in one of
  the object's own `PT_LOAD` segments.
- **The carried set** is the §3 "may" list plus `MPI_Comm_get_parent`, `MPI_Comm_compare`, and for a
  standalone Rust host `MPI_Comm_dup`, `MPI_Comm_free`, `MPI_Barrier` and `MPI_Recv`. tenferro's own
  sources call none of the last four (the scan enforces it). Stubs: 779 generated names.
- **T1 / Q5.** `MpiBackend` methods return `Result` (the break is accepted). rsmpi-rt's high-level
  `send_with_tag` and `matched_probe_with_tag` ignore return codes, so `Rt` calls `ffi::MPI_*`
  directly.
- **Receive over the C boundary is two calls**, `tfe_mpi_recv` (into a buffer the object holds) and
  `tfe_mpi_recv_take`, so a too-small buffer loses no message.
- **Status codes** beyond `capi-error-handling.md`: `-10` library, `-11` not initialized, `-12`
  finalized, `-13` MPI call failed, `-14` protocol, `-15` thread, `-16` no library yet.
- **Q2, Q3, Q4:** as decided in this note (no vtable entry, runtime panic stubs, resolve through the
  identified object). **Q6:** Linux only for now; `mpi-native-sys` uses `dl_iterate_phdr`, and macOS
  would need the `_dyld_*` walk. **Q7:** the Julia binding is a later chain node.
- **rsmpi-rt dependency.** `rsmpi-rt 0.9.0` is not on crates.io yet; the workspace takes it from
  tensor4all/rsmpi-rt#8 by git rev under `[patch.crates-io]`, like the other pre-release
  dependencies. The PR is not mergeable until that is released and the patch line removed.
