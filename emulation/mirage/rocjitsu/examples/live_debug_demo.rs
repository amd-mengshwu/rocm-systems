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

    // Step the engine through several cycles. This kernel is memory-bound
    // at entry (the wave is stalled on a long-latency load), so the PC
    // holds — but the real simulation clock visibly advances each cycle.
    let _ = writeln!(
        out,
        "\n{C_DIM}# stepping the live engine — watch the simulation clock advance{C_RESET}",
    );
    const STEP_TICKS: u64 = 20_000_000;
    let mut cycle = Vec::new();
    for _ in 0..5 {
        cycle.push(format!("stepi {STEP_TICKS}"));
        cycle.push("status".to_string());
    }
    let _ = repl.run_script(&cycle, &mut out);

    // The debugger can also *mutate* live state. Write a scalar register
    // straight into the running wave and read it back changed.
    let _ = writeln!(
        out,
        "\n{C_DIM}# the debugger can change live state too — write s4 and read it back{C_RESET}",
    );
    let _ = repl.run_script(
        [
            "print $s4",
            "set $s4 = 0xdeadbeef",
            "print $s4",
            "set $s4 = 0x1234",
            "print $s4",
        ],
        &mut out,
    );

    // Re-list the wavefronts and detach.
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
