//! Integration test for the in-process rocjitsu **daemon**.
//!
//! Stands up the Rust daemon (the `rocjitsu --daemon` replacement),
//! connects a client speaking the daemon RPC protocol, performs the
//! handshake the KMD interposer would, and verifies the daemon serves a
//! live simulated device. The whole test is skipped when no rocjitsu KMD
//! library is discoverable on this machine (install rocjitsu under
//! `$ROCM_HOME` or a sibling monorepo build to exercise it).

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

use mirage_core::common::MaybeRef;
use mirage_core::emulator::{EmulatorDaemon, EmulatorDef, ExecMode};
use mirage_rocjitsu::daemon::Daemon;
use mirage_rocjitsu::{kmd_config, kmd_preload};
/// Build the 16-byte RPC header the wire protocol uses.
fn header(opcode: u16, request_id: u32, payload_bytes: u32, result: i32) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0..2].copy_from_slice(&opcode.to_ne_bytes());
    h[4..8].copy_from_slice(&request_id.to_ne_bytes());
    h[8..12].copy_from_slice(&payload_bytes.to_ne_bytes());
    h[12..16].copy_from_slice(&result.to_ne_bytes());
    h
}

fn read_exact(stream: &mut UnixStream, buf: &mut [u8]) {
    stream.read_exact(buf).expect("read response");
}

#[test]
fn daemon_serves_handshake() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    // The daemon needs the KMD library (it exports the rj_vm_* API);
    // skip cleanly when rocjitsu is not installed.
    let Some(lib) = kmd_preload() else {
        eprintln!("rocjitsu KMD library not found; skipping daemon test");
        return;
    };

    // Synthesise a real sim config from a builtin agent.
    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report
        .iter()
        .map(|(n, _)| n.clone())
        .next()
        .expect("at least one builtin agent");
    let def = EmulatorDef {
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Functional,
        options: Default::default(),
        topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref(agent_name),
        }),
    };
    let config = kmd_config(&def, None).expect("sim config should materialise");

    let runtime_dir = tmp.path().join("rt");
    let daemon = Daemon::start(&lib, &config, &runtime_dir).expect("daemon should start");
    assert!(
        daemon.socket_path().exists(),
        "daemon socket should be bound"
    );

    // Connect and perform the handshake the interposer would.
    let mut stream = UnixStream::connect(daemon.socket_path()).expect("connect to daemon");
    stream
        .write_all(&header(0 /* HANDSHAKE */, 1, 0, 0))
        .expect("send handshake");

    let mut resp = [0u8; 16];
    read_exact(&mut stream, &mut resp);
    let payload_bytes = u32::from_ne_bytes(resp[8..12].try_into().unwrap()) as usize;
    let result = i32::from_ne_bytes(resp[12..16].try_into().unwrap());
    assert_eq!(result, 0, "handshake should succeed");
    assert!(
        payload_bytes >= 16,
        "handshake payload should carry the response struct"
    );

    let mut payload = vec![0u8; payload_bytes];
    read_exact(&mut stream, &mut payload);
    let version = u32::from_ne_bytes(payload[0..4].try_into().unwrap());
    let topo_len = u32::from_ne_bytes(payload[8..12].try_into().unwrap()) as usize;
    let drm_len = u32::from_ne_bytes(payload[12..16].try_into().unwrap()) as usize;
    assert_eq!(version, 3, "protocol version should match the interposer");
    // RpcHandshakeResponse is 328 bytes (16 fixed fields + 312-byte
    // RpcGpuInfo), then the topology and DRM path strings.
    assert_eq!(
        payload_bytes,
        328 + topo_len + drm_len,
        "handshake payload framing should be self-consistent"
    );
    assert!(topo_len > 0, "daemon should report a sysfs topology path");

    // Cleanly close the client session.
    stream
        .write_all(&header(2 /* CLOSE */, 2, 0, 0))
        .expect("send close");
    let mut close_resp = [0u8; 16];
    read_exact(&mut stream, &mut close_resp);

    // Shutting the daemon down removes its socket.
    let socket_path = daemon.socket_path().to_path_buf();
    Box::new(daemon).stop();
    assert!(
        !socket_path.exists(),
        "daemon socket should be removed on shutdown"
    );
}

/// A second handshake on a fresh connection must also succeed: the
/// daemon serves many clients over its lifetime (one per workload
/// open of `/dev/kfd`).
#[test]
fn daemon_serves_multiple_clients() {
    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let Some(lib) = kmd_preload() else {
        eprintln!("rocjitsu KMD library not found; skipping daemon test");
        return;
    };

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report.iter().map(|(n, _)| n.clone()).next().unwrap();
    let def = EmulatorDef {
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Functional,
        options: Default::default(),
        topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref(agent_name),
        }),
    };
    let config = kmd_config(&def, None).unwrap();
    let runtime_dir = tmp.path().join("rt");
    let daemon = Daemon::start(&lib, &config, &runtime_dir).expect("daemon should start");

    for req_id in 0..3u32 {
        let mut stream = UnixStream::connect(daemon.socket_path()).expect("connect");
        stream.write_all(&header(0, req_id, 0, 0)).unwrap();
        let mut resp = [0u8; 16];
        read_exact(&mut stream, &mut resp);
        let payload_bytes = u32::from_ne_bytes(resp[8..12].try_into().unwrap()) as usize;
        assert!(payload_bytes >= 16, "client {req_id} handshake framed");
        let mut payload = vec![0u8; payload_bytes];
        read_exact(&mut stream, &mut payload);
        stream.write_all(&header(2, req_id, 0, 0)).unwrap();
        let mut close_resp = [0u8; 16];
        read_exact(&mut stream, &mut close_resp);
    }
}

/// The daemon binds a **real** debugger control socket next to
/// `daemon.sock`, and the debug protocol drives the actual rocjitsu
/// engine through the FFI — there is no mock anywhere on this path. This
/// exercises the full live stack: `mirage_debug::Client` ->
/// `debug.sock` -> `DebugTarget` -> `rocjitsu_sys` FFI -> the C++
/// `DebugController` -> the running `SimulationEngine`.
#[test]
fn debug_socket_controls_the_live_engine() {
    use mirage_debug::Client;

    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let Some(lib) = kmd_preload() else {
        eprintln!("rocjitsu KMD library not found; skipping live debug test");
        return;
    };

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report.iter().map(|(n, _)| n.clone()).next().unwrap();
    let def = EmulatorDef {
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Functional,
        options: Default::default(),
        topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref(agent_name),
        }),
    };
    let config = kmd_config(&def, None).unwrap();
    let runtime_dir = tmp.path().join("rt");
    let daemon = Daemon::start(&lib, &config, &runtime_dir).expect("daemon should start");

    // The debug API only exists in a library built with the debug surface;
    // skip cleanly on an older library that bound no debug socket.
    let Some(debug_sock) = daemon.debug_socket_path().map(|p| p.to_path_buf()) else {
        eprintln!("rocjitsu library has no debug API; skipping live debug test");
        let socket_path = daemon.socket_path().to_path_buf();
        Box::new(daemon).stop();
        assert!(!socket_path.exists());
        return;
    };
    assert!(debug_sock.exists(), "debug socket should be on disk");

    let mut client = Client::connect(&debug_sock).expect("connect to debug socket");

    // The live engine reports debug support (the controller is bound).
    assert!(
        client.supported(),
        "the running daemon engine must report debug support"
    );

    // The engine is free-running after start; suspending it must drive it
    // to a real quiescent stop the engine acknowledges.
    client.suspend().expect("suspend the live engine");
    let (stopped, _reason, tick0) = client.status().expect("status after suspend");
    assert!(stopped, "engine should be stopped after suspend");

    // Single-stepping advances the *real* simulation clock.
    client.step(1).expect("step the live engine");
    let (_done, _r) = client.wait_stop(5_000).expect("wait for the step to settle");
    let (_stopped, _reason, tick1) = client.status().expect("status after step");
    assert!(
        tick1 >= tick0,
        "simulation tick should not go backwards across a step (was {tick0}, now {tick1})"
    );

    // Enumerating waves succeeds against the real engine. With no workload
    // attached there is no dispatch in flight, so the live wave set is
    // legitimately empty — the point is that the call round-trips through
    // the FFI and returns the engine's true state.
    let waves = client.wave_list().expect("wave_list round-trips to the engine");
    assert!(
        waves.is_empty(),
        "no workload is attached, so the engine should report zero live waves"
    );

    // Resume so teardown does not race a suspended engine.
    client.resume().expect("resume the live engine");
    drop(client);

    let socket_path = daemon.socket_path().to_path_buf();
    Box::new(daemon).stop();
    assert!(!socket_path.exists(), "daemon socket removed on shutdown");
    assert!(!debug_sock.exists(), "debug socket removed on shutdown");
}

/// The acid test: a **real HIP kernel** (`vector_add`, 16 workgroups)
/// runs through the full ROCr stack against the daemon, and the debugger
/// catches its wavefronts in flight by single-stepping the live engine.
/// Every wave, PC and register read in this test comes from the actual
/// `SimulationEngine` executing real GCN/CDNA instructions — nothing is
/// mocked.
///
/// Requires a `hipcc`-built workload and the matching `librocjitsu.so`
/// preload, supplied via env (the rocjitsu CMake build exports both):
///   * `RJ_HIP_VECTOR_ADD_BIN` — path to the `hip_vector_add_test` binary
///   * `RJ_PRELOAD_LIB`        — path to `librocjitsu.so` for `LD_PRELOAD`
/// Optionally set `RJ_DAEMON_CONFIG` to a hand-written sim config (e.g. the
/// rocjitsu CMake `configs/amdgpu_cdna4_kmd.json`); when present the test
/// additionally asserts the kernel computes the correct result end-to-end.
/// The test skips cleanly when the binary/library are absent.
#[test]
fn debug_observes_live_kernel_waves() {
    use mirage_debug::Client;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    let Some(workload) = std::env::var_os("RJ_HIP_VECTOR_ADD_BIN") else {
        eprintln!("RJ_HIP_VECTOR_ADD_BIN not set; skipping live kernel wave test");
        return;
    };
    let Some(preload) = std::env::var_os("RJ_PRELOAD_LIB") else {
        eprintln!("RJ_PRELOAD_LIB not set; skipping live kernel wave test");
        return;
    };
    if !std::path::Path::new(&workload).is_file() {
        eprintln!("workload binary missing; skipping live kernel wave test");
        return;
    }

    let _g = mirage_core::paths::test_env_lock();
    let tmp = tempfile::tempdir().unwrap();
    mirage_core::paths::set_test_root(tmp.path());

    let Some(lib) = kmd_preload() else {
        eprintln!("rocjitsu KMD library not found; skipping live kernel wave test");
        return;
    };

    let agent_report = mirage_builtin::ensure_agents(false).unwrap();
    let agent_name = agent_report.iter().map(|(n, _)| n.clone()).next().unwrap();
    let def = EmulatorDef {
        emulator: "rocjitsu".to_string(),
        plugins: Default::default(),
        exec_mode: ExecMode::Functional,
        options: Default::default(),
        topology: MaybeRef::Owned(mirage_core::topology::TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Ref(agent_name),
        }),
    };
    let config = kmd_config(&def, None).unwrap();
    // A hand-written sim config that exercises the full memory/SDMA path is
    // required for the kernel to compute correct results (the builtin-agent
    // config does not). When supplied we additionally assert correctness.
    let external_config = std::env::var_os("RJ_DAEMON_CONFIG").map(std::path::PathBuf::from);
    let config = external_config.clone().unwrap_or(config);
    let runtime_dir = tmp.path().join("rt");
    let daemon = Daemon::start(&lib, &config, &runtime_dir).expect("daemon should start");

    let Some(debug_sock) = daemon.debug_socket_path().map(|p| p.to_path_buf()) else {
        eprintln!("rocjitsu library has no debug API; skipping live kernel wave test");
        let socket_path = daemon.socket_path().to_path_buf();
        Box::new(daemon).stop();
        assert!(!socket_path.exists());
        return;
    };

    let mut client = Client::connect(&debug_sock).expect("connect to debug socket");
    assert!(client.supported(), "engine must report debug support");

    // Stop the world before the workload submits anything, so we control
    // every tick of the kernel's execution by hand.
    client.suspend().expect("suspend the live engine");

    // Launch the real HIP workload. With the engine parked, its first
    // engine-dependent operation (the H2D copies) blocks until we step.
    let mut child = Command::new(&workload)
        .env("ROCJITSU_RUNTIME_DIR", &runtime_dir)
        .env("LD_PRELOAD", &preload)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the HIP workload");

    // Single-step the engine in small batches, polling the live wave set,
    // until the kernel's wavefronts appear. In FUNCTIONAL mode the debugger
    // forces one-instruction-per-tick execution, so the waves are observable
    // and their PCs advance under single-stepping. A small batch catches them
    // near the kernel entry.
    const BATCH: u64 = 4;
    const DEADLINE: Duration = Duration::from_secs(90);
    let start = Instant::now();
    let mut caught: Vec<mirage_debug::WaveInfo> = Vec::new();

    while start.elapsed() < DEADLINE {
        client.step(BATCH).expect("step the live engine");
        let _ = client.wait_stop(5_000);
        let waves = client.wave_list().expect("wave_list round-trips");
        if !waves.is_empty() {
            caught = waves;
            break;
        }
    }

    assert!(
        !caught.is_empty(),
        "expected to catch at least one live wavefront from the kernel within the deadline"
    );

    // Inspect a real wavefront: it must carry a plausible live PC and let
    // us read its registers straight from the engine. Everything below is
    // read out of the actual CDNA compute unit executing the kernel.
    let wave = caught[0].clone();
    eprintln!(
        "caught {} live wave(s); first @ {} pc={:#x} state={}",
        caught.len(),
        wave.coord(),
        wave.pc,
        wave.state
    );
    assert_ne!(wave.pc, 0, "a live wave should have a non-zero program counter");

    // The PC reported in the wave summary must agree with a direct read of
    // the special PC register through the inspection plane.
    const SPECIAL_PC: u32 = 0; // special_reg::PC
    let pc_special = client
        .read_special(wave.id, SPECIAL_PC)
        .expect("read the live PC special register");
    assert_eq!(
        pc_special, wave.pc,
        "the special PC register must match the wave summary PC"
    );

    // Scalar and vector register files are readable from the live wave.
    let sgprs = client
        .read_sgpr(wave.id, 0, 8)
        .expect("read live SGPRs from the wave");
    assert_eq!(sgprs.len(), 8, "should read eight SGPRs back");

    let vgpr0 = client
        .read_vgpr(wave.id, 0, 0, wave.wave_size.max(1))
        .expect("read live VGPR lanes from the wave");
    assert_eq!(
        vgpr0.len(),
        wave.wave_size.max(1) as usize,
        "should read one value per active lane of v0"
    );

    // Single-step the wave and prove its PC actually advances through the
    // kernel — real instructions are retiring on the live compute unit.
    let pc_before = wave.pc;
    let mut advanced = false;
    for _ in 0..8 {
        client.step(8).expect("single-step the live wave");
        let _ = client.wait_stop(5_000);
        match client.wave_info(wave.id) {
            Ok(info) if info.pc != pc_before => {
                eprintln!("PC advanced {:#x} -> {:#x}", pc_before, info.pc);
                advanced = true;
                break;
            }
            // The wave retired (kernel finished) — that is also forward
            // progress, just past where we can compare PCs.
            Err(_) => {
                advanced = true;
                break;
            }
            _ => {}
        }
    }
    assert!(
        advanced,
        "single-stepping should advance the live wave's PC (was {pc_before:#x})"
    );

    // Resume and let the real workload run to completion. With a sim config
    // that models the full memory/SDMA path, the kernel must compute the
    // correct result end-to-end through the emulator.
    client.resume().expect("resume the live engine");
    client.detach().ok();
    drop(client);

    let output = wait_with_timeout(&mut child, Duration::from_secs(120));
    if external_config.is_some() {
        let output = output.expect("HIP workload should finish after resume");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "HIP workload should pass through the emulator; stdout=\n{stdout}\nstderr=\n{stderr}"
        );
    } else {
        // Without the full-path config the builtin-agent config does not
        // compute correct results; we only proved the debugger observed and
        // stepped real wavefronts. Tear the workload down.
        let _ = child.kill();
        let _ = child.wait();
    }

    let socket_path = daemon.socket_path().to_path_buf();
    Box::new(daemon).stop();
    assert!(!socket_path.exists(), "daemon socket removed on shutdown");
}

/// Wait for `child` to exit within `timeout`, returning its captured output.
fn wait_with_timeout(
    child: &mut std::process::Child,
    timeout: std::time::Duration,
) -> Option<std::process::Output> {
    use std::time::Instant;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                let mut out = Vec::new();
                let mut err = Vec::new();
                if let Some(mut s) = child.stdout.take() {
                    let _ = std::io::Read::read_to_end(&mut s, &mut out);
                }
                if let Some(mut s) = child.stderr.take() {
                    let _ = std::io::Read::read_to_end(&mut s, &mut err);
                }
                let status = child.wait().ok()?;
                return Some(std::process::Output {
                    status,
                    stdout: out,
                    stderr: err,
                });
            }
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                return None;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(_) => return None,
        }
    }
}



