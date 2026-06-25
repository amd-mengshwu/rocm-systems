//! End-to-end tests for the `mirage debug` debugger front-end.
//!
//! These drive the real `mirage` binary as a subprocess against the
//! deterministic in-process demo backend (`--demo`), so they need no GPU,
//! no daemon, and no session. Because the demo backend is fully
//! reproducible, the rendered output is asserted exactly where it matters
//! (PC advancement, register values, memory contents) — the same
//! guarantees the asciinema demos rely on.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;
use predicates::str;
use tempfile::TempDir;

struct Env {
    _dir: TempDir,
    config: PathBuf,
    runtime: PathBuf,
    mirage_bin: PathBuf,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config");
        let runtime = dir.path().join("runtime");
        let mirage_bin = PathBuf::from(env!("CARGO_BIN_EXE_mirage"));
        Self {
            _dir: dir,
            config,
            runtime,
            mirage_bin,
        }
    }

    fn mirage(&self) -> Command {
        let mut c = Command::new(&self.mirage_bin);
        c.env("XDG_CONFIG_HOME", &self.config)
            .env("XDG_RUNTIME_DIR", &self.runtime)
            .env("XDG_STATE_HOME", self._dir.path().join("state"))
            .env("MIRAGE_BIN", &self.mirage_bin)
            .env_remove("MIRAGE_LOG");
        c
    }
}

/// A bare `--demo --batch` with a single command enumerates the four demo
/// wavefronts at the entry PC.
#[test]
fn demo_lists_wavefronts() {
    let env = Env::new();
    env.mirage()
        .args(["debug", "--demo", "--batch", "-x", "info threads"])
        .assert()
        .success()
        .stdout(str::contains("0.0.0.0"))
        .stdout(str::contains("0.1.1.0"))
        .stdout(str::contains("0x1000"));
}

/// Stepping advances the program counter deterministically: three
/// `stepi`s from the 0x1000 entry land at 0x100c, and M0 (the executed
/// instruction count in the demo model) reads 3.
#[test]
fn demo_step_advances_pc_and_registers() {
    let env = Env::new();
    env.mirage()
        .args([
            "debug",
            "--demo",
            "--batch",
            "-x",
            "stepi 3",
            "-x",
            "info registers",
        ])
        .assert()
        .success()
        .stdout(str::contains("stopped: step"))
        .stdout(str::contains("0x000000000000100c"))
        .stdout(str::contains("m0     0x00000003"));
}

/// A breakpoint set ahead of the entry PC stops a continued run at that
/// address.
#[test]
fn demo_breakpoint_stops_the_run() {
    let env = Env::new();
    env.mirage()
        .args([
            "debug",
            "--demo",
            "--batch",
            "-x",
            "break 0x1010",
            "-x",
            "continue",
            "-x",
            "info registers",
        ])
        .assert()
        .success()
        .stdout(str::contains("stopped: breakpoint"))
        .stdout(str::contains("0x0000000000001010"));
}

/// The demo data region is a 0..255 byte ramp; examining it renders the
/// expected ascending bytes.
#[test]
fn demo_examine_memory_ramp() {
    let env = Env::new();
    env.mirage()
        .args([
            "debug",
            "--demo",
            "--batch",
            "-x",
            "x/16xb 0x100000",
        ])
        .assert()
        .success()
        .stdout(str::contains("00 01 02 03 04 05 06 07"));
}

/// Writing a scalar register is observable on a subsequent read.
#[test]
fn demo_register_write_roundtrips() {
    let env = Env::new();
    env.mirage()
        .args([
            "debug",
            "--demo",
            "--batch",
            "-x",
            "set $s4 = 0xcafe",
            "-x",
            "print $s4",
        ])
        .assert()
        .success()
        .stdout(str::contains("0x0000cafe"));
}

/// `--tui` on a build without the `tui` feature fails cleanly with a
/// helpful message rather than panicking. (When the test binary *is*
/// built with `--features tui` this path isn't reachable in batch mode,
/// so the assertion is gated to the default feature set.)
#[test]
#[cfg(not(feature = "tui"))]
fn demo_tui_without_feature_errors_cleanly() {
    let env = Env::new();
    env.mirage()
        .args(["debug", "--demo", "--tui"])
        .assert()
        .failure()
        .stderr(str::contains("without the debugger TUI"));
}
