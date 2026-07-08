# rocJITsu DBI SuperCollider

This directory tracks the rocJITsu DBI MVP for SuperCollider-style LDS race
instrumentation on AMD RDNA4 / `gfx1201`.

The current implementation is a trap-first, HSA-tools-loaded proof path. It can
modify native `gfx1201` code objects at load time, patch compact native-DS IREE
kernels through inline padding, local NOP caves, or appended `.text` caves, and
patch selected likely group/LDS flat helper-function accesses in hip-moi. It is
not yet full SuperCollider race detection. An opt-in prototype report-buffer
mode can replace `s_trap` with a one-word marker write to a caller-supplied
device-visible address.

## Start Here

- [DESIGN.md](DESIGN.md): technical design and comparison with the
  SuperCollider paper, including flat/generic access handling.
- [USAGE.md](USAGE.md): commands, environment variables, and test runbook.
- [JAKUB-DEMO.md](JAKUB-DEMO.md): compact evidence packet from the one-hour
  Jakub demo pass.

## Current Demo Claim

With the default combined check/trap scope:

```sh
export HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
export LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RJ_DBI_SUPERCOLLIDER=1
export RJ_DBI_LOG=1
export RJ_DBI_SC_DELAY_MODE=sleep
export RJ_DBI_SC_DELAY=1
export RJ_DBI_SC_MAX_PATCHES=4
export RJ_DBI_SC_REQUIRE_PATCH=1
```

the focused IREE WMMA ROCm/HIP e2e test has logged a native-DS patch:

```text
kind=local-cave-lds-load-check-trap anchor=0x3cc trampoline=0x810 original_size=8 scratch_vgpr=104
```

and passed. Broader IREE coverage also passes under the same patch-required
configuration:

- 13/13 for the full RDNA4 ROCm/HIP matmul e2e set exposed by this build,
- 10/10 for a focused linalg/matmul/StableHLO slice covering narrow matmul,
  f16/f8/i8 TileAndFuse variants, DT f8, and StableHLO stream-dot variants.

The same focused WMMA control passes with `RJ_DBI_SC_DELAY_MODE=sleep` and with
`RJ_DBI_SC_DELAY_MODE=sleep_var`.

The rocJITsu GPU smoke tests also exercise the non-trapping marker-buffer path:
a clean padded LDS store leaves the report word at zero, while the racy padded
LDS store writes the configured marker and lets the dispatch complete.

## What Is Instrumented Today

The current check/trap proof paths cover:

- padded native LDS `ds_load_b{32,64,128}`,
- padded native LDS `ds_load_u16_d16` and `ds_load_u16_d16_hi`,
- padded native LDS `ds_store_b{32,64,128}`,
- compact native LDS sites through a local NOP cave when one is reachable, or an
  appended `.text` cave when that is the safe available placement,
- likely group/LDS `flat_load_b{32,64,128}`,
- likely group/LDS `flat_store_b{32,64,128}`.

Native DS sites can use enough trailing `s_nop 0` padding for an in-place
sequence, or reachable local NOP caves for compact sites. Flat/VFLAT sites use
the same total `RJ_DBI_SC_MAX_PATCHES` budget for non-overlapping padded sites
and reachable local NOP caves, and can now compose after native DS patches when
patch ranges remain mappable in the original code object. Ordinary hip-moi
matmul helper code has shown likely group/LDS flat sites rather than native
`ds_*`, which is why the flat path matters.

See [DESIGN.md](DESIGN.md) for the exact instruction policy and the current
address-space provenance heuristic.

## Main Gaps

- `RJ_DBI_SC_DELAY_MODE=sleep` emits `s_sleep N`, and
  `RJ_DBI_SC_DELAY_MODE=sleep_var` emits `s_sleep_var` from a scalar source
  operand. The remaining delay gap is randomized sampling policy, not the basic
  sleep instruction mechanism.
- `RJ_DBI_SC_MAX_PATCHES=N` can patch multiple native-DS and flat/VFLAT
  check/trap sites in one code object, bounded by non-overlapping in-place
  ranges and reachable local NOP caves.
- Default reporting is still `s_trap`, but `RJ_DBI_SC_REPORT_BUFFER=0x...`
  enables a simple marker-buffer prototype. On mismatch, the injected sequence
  writes one 32-bit marker word and continues. The rocJITsu HIP smoke tests now
  allocate such a report word and verify both clean and racy outcomes.
- Current flat provenance is conservative and heuristic. `MaybeGroup` is useful
  for MVP bring-up, but it is not the same as a formal proof that an arbitrary
  flat access targets LDS.
- Native DS d16 support currently covers `ds_load_u16_d16` and
  `ds_load_u16_d16_hi`; other 8/16-bit LDS forms remain deferred.

## Test Discipline

hip-moi is the inner loop. IREE is broader compatibility coverage and should be
run roughly once per work session. Keep GPU test parallelism at or below 8.
