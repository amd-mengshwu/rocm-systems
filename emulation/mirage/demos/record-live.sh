#!/usr/bin/env bash
# Record a *real* `mirage debug` session against a live rocjitsu kernel.
#
# Unlike the `--demo` casts (which use a deterministic in-process fixture
# to showcase the UI), this drives the actual daemon + a real hipcc-built
# vector_add kernel and captures the genuine wavefront/register output.
#
# Requires: asciinema, a rocjitsu CMake build with librocjitsu.so and the
# hip_vector_add_test binary, and hipcc on PATH at build time.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mirage_dir="$(cd "$here/.." && pwd)"
rj_build="$(cd "$mirage_dir/../rocjitsu/build" && pwd)"

lib="$rj_build/librocjitsu.so"
workload="$rj_build/tests/hip_vector_add_test"
cast="$here/00-live-real-kernel.cast"

if [[ ! -f "$lib" ]]; then
  echo "missing $lib — build rocjitsu first (ninja librocjitsu.so)" >&2
  exit 1
fi
if [[ ! -f "$workload" ]]; then
  echo "missing $workload — build the rocjitsu tests first" >&2
  exit 1
fi
if ! command -v asciinema >/dev/null; then
  echo "asciinema not installed" >&2
  exit 1
fi

# Stage a ROCM_HOME layout so kmd_preload() discovers the fresh library.
rocm_home="$(mktemp -d)"
mkdir -p "$rocm_home/lib"
ln -sf "$lib" "$rocm_home/lib/librocjitsu.so"
trap 'rm -rf "$rocm_home"' EXIT

echo "building the live demo example…" >&2
( cd "$mirage_dir" && cargo build -q -p mirage_rocjitsu --example live_debug_demo )

# Record the example. Engine CP logs go to stderr; keep the cast to the
# clean debugger transcript on stdout.
asciinema rec --overwrite \
  --title "mirage debug — live vector_add kernel (real, no mock)" \
  --command "cd '$mirage_dir' && ROCM_HOME='$rocm_home' \
    RJ_HIP_VECTOR_ADD_BIN='$workload' RJ_PRELOAD_LIB='$lib' \
    cargo run -q -p mirage_rocjitsu --example live_debug_demo 2>/dev/null" \
  "$cast"

echo "wrote $cast" >&2
