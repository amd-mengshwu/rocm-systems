# `mirage debug` demos

A series of [asciinema](https://asciinema.org) casts showing off the
`mirage debug` GPU wavefront debugger.

There are two flavours:

* **`00-live-real-kernel.cast`** — the real deal. A real rocjitsu daemon
  runs a real `hipcc`-built `vector_add` kernel through the full ROCr
  stack; the debugger suspends the live `SimulationEngine`, single-steps
  to the dispatch, and prints the **actual** wavefronts and registers of
  the running CDNA compute units. Nothing is mocked.
* **`01`–`06`** — UI showcases driven by the deterministic in-process
  fixture backend (`mirage debug --demo`). These exercise the exact same
  REPL/TUI renderer but with a reproducible mock GPU, so they need no
  hardware or daemon and reproduce byte-for-byte on every machine.

## Playing the casts

```sh
asciinema play demos/00-live-real-kernel.cast
```

| Cast | Shows |
| ---- | ----- |
| `00-live-real-kernel.cast`   | **Real** kernel: live wavefronts + registers read straight from the engine |
| `01-attach-and-inspect.cast` | Attaching, listing wavefronts, selecting one, dumping its registers |
| `02-stepping.cast`           | Single-stepping (`stepi`) and watching the PC / registers advance |
| `03-breakpoints.cast`        | Setting a PC breakpoint and continuing to it |
| `04-memory.cast`             | Examining (`x/`) and writing (`mw`) device memory |
| `05-registers.cast`          | Reading and writing scalar registers |
| `06-tui.cast`                | The full-screen `--tui` dashboard |

## Recording the live (real-kernel) cast

This needs a rocjitsu CMake build that produced `librocjitsu.so` and the
`hip_vector_add_test` binary (build with `hipcc` available):

```sh
# build the emulator + HIP test once
( cd ../../rocjitsu/build && ninja librocjitsu.so hip_vector_add_test )
# record the live session
demos/record-live.sh        # writes demos/00-live-real-kernel.cast
```

The same live path is covered by an integration test
(`debug_observes_live_kernel_waves` in `rocjitsu/tests/daemon.rs`), which
catches the real wavefronts and reads their PC/SGPR/VGPR state. Run it
with:

```sh
ROCM_HOME=<dir-with-lib> \
RJ_HIP_VECTOR_ADD_BIN=<build>/tests/hip_vector_add_test \
RJ_PRELOAD_LIB=<build>/librocjitsu.so \
  cargo test -p mirage_rocjitsu --test daemon debug_observes_live_kernel_waves
```

## Regenerating the REPL (fixture) casts

The casts are derived from the actual debugger output, so they stay in
sync with the tool. After changing the debugger, rebuild and regenerate:

```sh
cargo build -p mirage
python3 demos/generate_casts.py
```

## Recording the TUI

The full-screen TUI (`mirage debug --tui`) is interactive, so its cast is
recorded live rather than generated. Build with the `tui` feature and run
the helper, which drives the UI through a scripted `tmux` session:

```sh
cargo build -p mirage --features tui
demos/record-tui.sh
```

This writes `demos/06-tui.cast`. Requires `asciinema` and `tmux`.

## Trying it yourself

```sh
# scripted, non-interactive
mirage debug --demo --batch -x "info threads" -x "stepi 3" -x "info registers"

# interactive gdb-style REPL
mirage debug --demo

# full-screen TUI (needs a `--features tui` build)
mirage debug --tui --demo
```

When a real session is running under `--daemon` with a debug-capable
emulator, attach to it instead of the demo backend:

```sh
mirage debug <session-id>
# or point at the socket directly:
mirage debug --socket <runtime>/session/<id>/rocjitsu/debug.sock
```
