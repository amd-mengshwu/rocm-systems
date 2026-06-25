//! Live, **un-mocked** `mirage debug` demo.
//!
//! Stands up the real in-process rocjitsu daemon (clocked, so wavefronts
//! persist across simulation ticks), launches a real `hipcc`-built
//! `vector_add` kernel through the full ROCr stack, single-steps the live
//! `SimulationEngine` until the kernel's wavefronts are in flight, and
//! then drives the **actual** debugger REPL — the same renderer the
//! `mirage debug` CLI uses — against the live engine. Every wavefront,
//! program counter and register printed below is read straight out of the
//! emulated CDNA compute units; nothing here is a fixture.
//!
//! Run it (the rocjitsu CMake build exports the two binaries):
//! ```sh
//! ROCM_HOME=/path/with/lib \
//! RJ_HIP_VECTOR_ADD_BIN=/path/to/hip_vector_add_test \
//! RJ_PRELOAD_LIB=/path/to/librocjitsu.so \
//!   cargo run -p mirage_rocjitsu --example live_debug_demo
//! ```

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDaemon, EmulatorDef, ExecMode};
use mirage_debug::{Client, Repl};
use mirage_rocjitsu::daemon::Daemon;
use mirage_rocjitsu::{kmd_config, kmd_preload};

const C_TITLE: &str = "\x1b[1;36m";
const C_DIM: &str = "\x1b[2m";
const C_GREEN: &str = "\x1b[32m";
const C_RESET: &str = "\x1b[0m";

fn main() {
    let workload = match std::env::var_os("RJ_HIP_VECTOR_ADD_BIN") {
        Some(v) => v,
        None => {
            eprintln!("set RJ_HIP_VECTOR_ADD_BIN to the hip_vector_add_test binary");
            std::process::exit(2);
        }
    };
    let preload = match std::env::var_os("RJ_PRELOAD_LIB") {
        Some(v) => v,
        None => {
            eprintln!("set RJ_PRELOAD_LIB to librocjitsu.so");
            std::process::exit(2);
        }
    };

    let mut out = std::io::stdout();
    let _ = writeln!(
        out,
        "{C_TITLE}# mirage debug — live vector_add kernel on the rocjitsu engine{C_RESET}"
    );
    let _ = writeln!(
        out,
        "{C_DIM}# real daemon + real HIP kernel + real wavefronts (no mock){C_RESET}\n"
    );

    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let Some(lib) = kmd_preload() else {
        eprintln!("librocjitsu.so not found (set ROCM_HOME=<dir-with-lib>)");
        std::process::exit(2);
    };

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report.iter().map(|(n, _)| n.clone()).next().unwrap();
    let def = EmulatorDef {
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Clocked,
        options: Default::default(),
        topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref(agent_name),
        }),
    };
    let config = kmd_config(&def, None).unwrap();
    let runtime_dir = tmp.path().join("rt");

    let _ = writeln!(out, "{C_DIM}# starting the rocjitsu daemon (clocked)…{C_RESET}");
    let daemon = Daemon::start(&lib, &config, &runtime_dir).expect("daemon should start");
    let debug_sock = daemon
        .debug_socket_path()
        .expect("library must expose the debug API")
        .to_path_buf();

    let _ = writeln!(
        out,
        "{C_DIM}# launching the HIP vector_add workload through ROCr…{C_RESET}"
    );
    let mut child = Command::new(&workload)
        .env("ROCJITSU_RUNTIME_DIR", &runtime_dir)
        .env("LD_PRELOAD", &preload)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn the HIP workload");

    let mut client = Client::connect(&debug_sock).expect("attach to the debug socket");

    // Stop the world, then single-step the engine until the kernel's
    // wavefronts are live. This is the real suspend/step control plane.
    let _ = writeln!(out, "{C_GREEN}(mirage-dbg) {C_RESET}suspend");
    client.suspend().expect("suspend");
    let _ = writeln!(out, "target stopped");
    let _ = writeln!(
        out,
        "{C_GREEN}(mirage-dbg) {C_RESET}{C_DIM}# advancing the engine to the kernel dispatch…{C_RESET}"
    );

    let start = Instant::now();
    let mut live = false;
    while start.elapsed() < Duration::from_secs(90) {
        client.step(500).expect("step");
        let _ = client.wait_stop(5_000);
        if !client.wave_list().unwrap_or_default().is_empty() {
            live = true;
            break;
        }
    }
    if !live {
        eprintln!("did not reach live wavefronts in time");
        let _ = child.kill();
        Box::new(daemon).stop();
        std::process::exit(1);
    }

    // Hand the live client to the real REPL renderer and show off the
    // actual debugger output against the running kernel.
    let mut repl = Repl::new(client);

    // First, a snapshot of the live wavefronts and the registers of one.
    let _ = repl.run_script(
        ["info threads", "wave 1", "info registers", "info registers sgpr"],
        &mut out,
    );

    // Now single-step the engine through several cycles so you can watch
    // the program counter and registers of the live wave actually move.
    let _ = writeln!(
        out,
        "\n{C_DIM}# stepping the live engine — watch the PC and registers change{C_RESET}",
    );
    const STEP_TICKS: u64 = 50_000_000;
    let mut cycle = Vec::new();
    for _ in 0..8 {
        cycle.push(format!("stepi {STEP_TICKS}"));
        cycle.push("info registers".to_string());
    }
    let _ = repl.run_script(&cycle, &mut out);

    // Finally, re-list the wavefronts: as the dispatch progresses, the set
    // of live waves evolves too.
    let _ = repl.run_script(["info threads", "detach"], &mut out);

    let _ = writeln!(
        out,
        "\n{C_DIM}# detached; tearing down the daemon and workload{C_RESET}"
    );
    drop(repl);
    let _ = child.kill();
    let _ = child.wait();
    Box::new(daemon).stop();
}
