//! In-process rocjitsu **daemon** — a Rust reimplementation of the
//! `rocjitsu --daemon` server, driven directly through the rocjitsu C
//! API ([`rocjitsu_sys`]) instead of the `rocjitsu` command-line tool.
//!
//! # What it does
//!
//! The mirage per-node host starts one of these on the node it serves.
//! It:
//!
//! 1. Loads `librocjitsu.so` (which exports the full `rj_vm_*` API)
//!    and creates a VM in [`RjVmMode::Daemon`] mode from the synthesised
//!    `SimulationConfig`. In daemon mode every GPU allocation is backed
//!    by a `memfd` so it can be shared with the workload process.
//! 2. Spawns the simulation engine on its own thread (`rj_vm_run`).
//! 3. Binds a Unix domain socket at `<runtime_dir>/daemon.sock` — the
//!    exact path the rocjitsu KMD interposer probes first when a
//!    workload opens `/dev/kfd` — and serves the daemon RPC protocol.
//!
//! Because the interposer connects to the daemon socket *before* falling
//! back to in-process (local) emulation, simply standing this server up
//! at the workload's `ROCJITSU_RUNTIME_DIR` switches the workload to
//! daemon-served emulation with no change to the injected environment.
//!
//! # Protocol
//!
//! The wire format mirrors `rocjitsu/lib/rocjitsu/src/rocjitsu/kmd/linux/rpc.h`:
//! a fixed 16-byte header followed by an opcode-specific payload, with
//! GPU `memfd`s passed as `SCM_RIGHTS` ancillary data. This server is
//! byte-compatible with the upstream C daemon and interposer.

use std::ffi::CString;
use std::os::raw::c_void;
use std::os::unix::io::{FromRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use mirage_core::emulator::EmulatorDaemon;
use mirage_debug::DebugBackend;
use mirage_debug::protocol::WaveInfo as DbgWaveInfo;
use rocjitsu_sys::{
    Lib, RjVm, RjVmCmd, RjVmGpuInfo, RjVmMap, RjVmMode, RjVmUnmap, ROCJITSU_STATUS_SUCCESS,
    RjStatus,
};

/// RPC opcodes (must match `enum RpcOpcode` in `rpc.h`).
const RPC_HANDSHAKE: u16 = 0;
#[allow(dead_code)]
const RPC_OPEN: u16 = 1;
const RPC_CLOSE: u16 = 2;
const RPC_MMAP: u16 = 3;
const RPC_MUNMAP: u16 = 4;
const RPC_IOCTL: u16 = 5;

/// RPC protocol version (`kRpcProtocolVersion` in `rpc.h`).
const RPC_PROTOCOL_VERSION: u32 = 3;

/// Size of the fixed RPC header in bytes.
const RPC_HEADER_LEN: usize = 16;

/// Size of the fixed `RpcHandshakeResponse` payload: four `u32` fields
/// (version, gpu_id, topology_path_len, drm_path_len) followed by the
/// 312-byte `RpcGpuInfo`. Must equal `sizeof(RpcHandshakeResponse)`
/// (== 328, asserted in `rpc.h`).
const RPC_HANDSHAKE_RESPONSE_LEN: usize = 16 + std::mem::size_of::<RjVmGpuInfo>();

/// Upper bound on an ioctl payload, mirroring the C daemon's guard
/// against a malicious or corrupt client.
const MAX_IOCTL_PAYLOAD: u32 = 16 * 1024 * 1024;

/// The VM pointer plus the loaded library, shared (read-only) across the
/// engine thread and every client thread.
///
/// rocjitsu serialises access to the VM internally (the daemon shares
/// one VM across all client connections via the `*_as` API), so sharing
/// the raw pointer across threads is sound.
struct Shared {
    lib: Lib,
    vm: *mut RjVm,
}

// Safety: see the type doc — the VM is internally synchronised and the
// resolved `Lib` entry points are plain function pointers.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

/// Live debugger backend: forwards [`mirage_debug`] operations to the
/// rocjitsu C debug API (`rj_vm_debug_*`) for the daemon's shared VM.
///
/// Because the debug API parks the engine thread before any inspecting
/// call returns (see `rj_vm_debug.h`), reads and writes observe a
/// consistent, stop-the-world snapshot. The same `Arc<Shared>` is shared
/// with the engine and KMD client threads; rocjitsu serialises access
/// internally.
struct DebugTarget {
    shared: Arc<Shared>,
}

impl DebugTarget {
    fn new(shared: Arc<Shared>) -> Self {
        Self { shared }
    }

    fn lib(&self) -> &Lib {
        &self.shared.lib
    }

    fn vm(&self) -> *mut RjVm {
        self.shared.vm
    }
}

/// Convert an FFI wave snapshot into the protocol's transport struct.
fn to_wave_info(w: &rocjitsu_sys::RjDbgWaveInfo) -> DbgWaveInfo {
    DbgWaveInfo {
        id: w.id,
        xcc: w.xcc,
        se: w.se,
        cu: w.cu,
        slot: w.slot,
        state: w.state,
        pc: w.pc,
        exec: w.exec,
        vcc: w.vcc,
        status: w.status,
        mode: w.mode,
        m0: w.m0,
        wave_size: w.wave_size,
        num_sgprs: w.num_sgprs,
        num_vgprs: w.num_vgprs,
        wg_id: w.wg_id,
        dispatch_id: w.dispatch_id,
        process_id: w.process_id,
    }
}

/// Map an `RjStatus` into `Result<(), String>` for the debug backend.
fn dbg_ok(status: RjStatus, what: &str) -> Result<(), String> {
    if status == ROCJITSU_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} failed (status {status})"))
    }
}

impl DebugBackend for DebugTarget {
    fn supported(&self) -> bool {
        unsafe { self.lib().dbg_supported(self.vm()) }
    }

    fn status(&self) -> Result<(bool, u32, u64), String> {
        unsafe { self.lib().dbg_status(self.vm()) }.ok_or_else(|| "debug status failed".into())
    }

    fn suspend(&self) -> Result<(), String> {
        dbg_ok(unsafe { self.lib().dbg_suspend(self.vm()) }, "suspend")
    }

    fn resume(&self) -> Result<(), String> {
        dbg_ok(unsafe { self.lib().dbg_resume(self.vm()) }, "resume")
    }

    fn step(&self, ticks: u64) -> Result<(), String> {
        dbg_ok(unsafe { self.lib().dbg_step(self.vm(), ticks) }, "step")
    }

    fn wait_stop(&self, timeout_ms: u64) -> Result<(bool, u32), String> {
        unsafe { self.lib().dbg_wait_stop(self.vm(), timeout_ms) }
            .ok_or_else(|| "wait_stop failed".into())
    }

    fn wave_list(&self) -> Result<Vec<DbgWaveInfo>, String> {
        let waves =
            unsafe { self.lib().dbg_wave_list(self.vm()) }.ok_or_else(|| "wave_list failed".to_string())?;
        Ok(waves.iter().map(to_wave_info).collect())
    }

    fn wave_info(&self, wave: u64) -> Result<DbgWaveInfo, String> {
        let w = unsafe { self.lib().dbg_wave_info(self.vm(), wave) }
            .ok_or_else(|| format!("no such wave: {wave:#x}"))?;
        Ok(to_wave_info(&w))
    }

    fn read_sgpr(&self, wave: u64, first: u32, count: u32) -> Result<Vec<u32>, String> {
        unsafe { self.lib().dbg_read_sgpr(self.vm(), wave, first, count) }
            .ok_or_else(|| "read_sgpr failed".into())
    }

    fn write_sgpr(&self, wave: u64, index: u32, value: u32) -> Result<(), String> {
        dbg_ok(
            unsafe { self.lib().dbg_write_sgpr(self.vm(), wave, index, value) },
            "write_sgpr",
        )
    }

    fn read_vgpr(
        &self,
        wave: u64,
        reg: u32,
        first_lane: u32,
        lane_count: u32,
    ) -> Result<Vec<u32>, String> {
        unsafe {
            self.lib()
                .dbg_read_vgpr(self.vm(), wave, reg, first_lane, lane_count)
        }
        .ok_or_else(|| "read_vgpr failed".into())
    }

    fn write_vgpr(&self, wave: u64, reg: u32, lane: u32, value: u32) -> Result<(), String> {
        dbg_ok(
            unsafe { self.lib().dbg_write_vgpr(self.vm(), wave, reg, lane, value) },
            "write_vgpr",
        )
    }

    fn read_special(&self, wave: u64, which: u32) -> Result<u64, String> {
        unsafe { self.lib().dbg_read_special(self.vm(), wave, which) }
            .ok_or_else(|| "read_special failed".into())
    }

    fn write_special(&self, wave: u64, which: u32, value: u64) -> Result<(), String> {
        dbg_ok(
            unsafe { self.lib().dbg_write_special(self.vm(), wave, which, value) },
            "write_special",
        )
    }

    fn read_memory(&self, vmid: u32, addr: u64, size: u64) -> Result<Vec<u8>, String> {
        unsafe { self.lib().dbg_read_memory(self.vm(), vmid, addr, size) }
            .ok_or_else(|| "read_memory failed".into())
    }

    fn write_memory(&self, vmid: u32, addr: u64, data: &[u8]) -> Result<(), String> {
        dbg_ok(
            unsafe { self.lib().dbg_write_memory(self.vm(), vmid, addr, data) },
            "write_memory",
        )
    }

    fn break_set(&self, pc: u64) -> Result<u32, String> {
        unsafe { self.lib().dbg_break_set(self.vm(), pc) }.ok_or_else(|| "break_set failed".into())
    }

    fn break_clear(&self, id: u32) -> Result<(), String> {
        dbg_ok(unsafe { self.lib().dbg_break_clear(self.vm(), id) }, "break_clear")
    }

    fn break_list(&self) -> Result<Vec<u64>, String> {
        unsafe { self.lib().dbg_break_list(self.vm()) }.ok_or_else(|| "break_list failed".into())
    }
}

/// A running rocjitsu daemon. Dropping it (or calling
/// [`EmulatorDaemon::stop`]) tears the server down cleanly: it stops
/// accepting, unblocks and joins all client threads, stops the engine,
/// destroys the VM, and removes the socket.
pub struct Daemon {
    shared: Arc<Shared>,
    listen_fd: RawFd,
    socket_path: PathBuf,
    stop: Arc<AtomicBool>,
    /// fds of currently-connected clients, so shutdown can unblock them.
    clients: Arc<Mutex<Vec<RawFd>>>,
    accept_thread: Option<JoinHandle<()>>,
    engine_thread: Option<JoinHandle<()>>,
    /// Debugger control socket (`debug.sock`). Best-effort: `None` when the
    /// loaded library lacks the debug API or the socket could not be bound.
    debug_socket_path: PathBuf,
    debug_listen_fd: Option<RawFd>,
    debug_clients: Arc<Mutex<Vec<RawFd>>>,
    debug_thread: Option<JoinHandle<()>>,
}

impl Daemon {
    /// Path of the Unix socket this daemon listens on.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Path of the debugger control socket, if one was bound. `None` when
    /// the loaded library lacks the debug API or the socket failed to bind.
    pub fn debug_socket_path(&self) -> Option<&Path> {
        self.debug_listen_fd.map(|_| self.debug_socket_path.as_path())
    }

    /// Load `lib_path`, create a daemon-mode VM from `config_path`, and
    /// start serving on `<runtime_dir>/daemon.sock`.
    ///
    /// On success the engine is running and the socket is accepting
    /// connections. Returns a human-readable error otherwise.
    pub fn start(
        lib_path: &Path,
        config_path: &Path,
        runtime_dir: &Path,
    ) -> std::result::Result<Self, String> {
        // Load the rocjitsu library and create the VM in daemon mode.
        let lib = unsafe { Lib::open(lib_path) }
            .map_err(|e| format!("rocjitsu daemon: cannot load {}: {e}", lib_path.display()))?;
        let cfg = CString::new(config_path.as_os_str().as_encoded_bytes())
            .map_err(|e| format!("rocjitsu daemon: invalid config path: {e}"))?;
        let (status, vm) = unsafe { lib.vm_create(&cfg, RjVmMode::Daemon) };
        if status != rocjitsu_sys::ROCJITSU_STATUS_SUCCESS || vm.is_null() {
            return Err(format!(
                "rocjitsu daemon: rj_vm_create({}) failed with status {status}",
                config_path.display()
            ));
        }
        let shared = Arc::new(Shared { lib, vm });

        // Bind the listening socket *before* spawning the engine so a
        // bind failure leaves nothing to tear down but the VM.
        let socket_path = runtime_dir.join("daemon.sock");
        let listen_fd = match bind_listen(&socket_path) {
            Ok(fd) => fd,
            Err(e) => {
                // Roll back the VM we just created.
                unsafe {
                    let reason = CString::new("bind failed").unwrap();
                    shared.lib.vm_request_exit(shared.vm, &reason);
                    shared.lib.vm_destroy(shared.vm);
                }
                return Err(e);
            }
        };

        // Run the simulation engine. `rj_vm_run` blocks until
        // `rj_vm_request_exit` is called from `stop`/`drop`.
        let engine_shared = shared.clone();
        let engine_thread = std::thread::Builder::new()
            .name("rocjitsu-engine".to_string())
            .spawn(move || {
                unsafe { engine_shared.lib.vm_run(engine_shared.vm) };
            })
            .map_err(|e| format!("rocjitsu daemon: cannot spawn engine thread: {e}"))?;

        let stop = Arc::new(AtomicBool::new(false));
        let clients = Arc::new(Mutex::new(Vec::new()));
        let accept_shared = shared.clone();
        let accept_stop = stop.clone();
        let accept_clients = clients.clone();
        let accept_thread = std::thread::Builder::new()
            .name("rocjitsu-accept".to_string())
            .spawn(move || {
                accept_loop(listen_fd, accept_shared, accept_stop, accept_clients);
            })
            .map_err(|e| format!("rocjitsu daemon: cannot spawn accept thread: {e}"))?;

        tracing::info!(
            socket = %socket_path.display(),
            config = %config_path.display(),
            "rocjitsu daemon started"
        );

        // Best-effort debugger control socket alongside daemon.sock. A
        // failure here (e.g. an older library without the debug API, or a
        // bind error) leaves the daemon fully functional, just not
        // debuggable.
        let debug_socket_path = runtime_dir.join("debug.sock");
        let debug_clients: Arc<Mutex<Vec<RawFd>>> = Arc::new(Mutex::new(Vec::new()));
        let (debug_listen_fd, debug_thread) = if shared.lib.has_debug() {
            match bind_listen(&debug_socket_path) {
                Ok(fd) => {
                    let backend = Arc::new(DebugTarget::new(shared.clone()));
                    let dbg_stop = stop.clone();
                    let dbg_clients = debug_clients.clone();
                    match std::thread::Builder::new()
                        .name("rocjitsu-debug".to_string())
                        .spawn(move || debug_accept_loop(fd, backend, dbg_stop, dbg_clients))
                    {
                        Ok(t) => {
                            tracing::info!(
                                socket = %debug_socket_path.display(),
                                "rocjitsu debug server started"
                            );
                            (Some(fd), Some(t))
                        }
                        Err(e) => {
                            tracing::warn!("rocjitsu debug server: cannot spawn thread: {e}");
                            unsafe { libc::close(fd) };
                            (None, None)
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("rocjitsu debug server: {e}");
                    (None, None)
                }
            }
        } else {
            tracing::info!("rocjitsu library has no debug API; debugger disabled");
            (None, None)
        };

        Ok(Self {
            shared,
            listen_fd,
            socket_path,
            stop,
            clients,
            accept_thread: Some(accept_thread),
            engine_thread: Some(engine_thread),
            debug_socket_path,
            debug_listen_fd,
            debug_clients,
            debug_thread,
        })
    }

    /// Tear the daemon down. Idempotent; called from both
    /// [`EmulatorDaemon::stop`] and [`Drop`].
    fn teardown(&mut self) {
        if self.stop.swap(true, Ordering::SeqCst) {
            // Already torn down.
            return;
        }
        // Stop accepting new connections and unblock the accept thread.
        unsafe { libc::shutdown(self.listen_fd, libc::SHUT_RDWR) };
        // Unblock any in-flight client recv()s so their threads exit.
        if let Ok(fds) = self.clients.lock() {
            for &fd in fds.iter() {
                unsafe { libc::shutdown(fd, libc::SHUT_RDWR) };
            }
        }
        // The accept thread joins all client threads before returning.
        if let Some(t) = self.accept_thread.take() {
            let _ = t.join();
        }
        unsafe { libc::close(self.listen_fd) };
        let _ = std::fs::remove_file(&self.socket_path);

        // Tear down the debug server the same way: unblock its accept and
        // any connected debug clients, then join.
        if let Some(fd) = self.debug_listen_fd {
            unsafe { libc::shutdown(fd, libc::SHUT_RDWR) };
        }
        if let Ok(fds) = self.debug_clients.lock() {
            for &fd in fds.iter() {
                unsafe { libc::shutdown(fd, libc::SHUT_RDWR) };
            }
        }
        if let Some(t) = self.debug_thread.take() {
            let _ = t.join();
        }
        if let Some(fd) = self.debug_listen_fd {
            unsafe { libc::close(fd) };
        }
        let _ = std::fs::remove_file(&self.debug_socket_path);

        // Stop the engine and reclaim the VM.
        unsafe {
            let reason = CString::new("daemon shutdown").unwrap();
            self.shared.lib.vm_request_exit(self.shared.vm, &reason);
        }
        if let Some(t) = self.engine_thread.take() {
            let _ = t.join();
        }
        unsafe { self.shared.lib.vm_destroy(self.shared.vm) };
        tracing::info!(socket = %self.socket_path.display(), "rocjitsu daemon stopped");
    }
}

impl EmulatorDaemon for Daemon {
    fn stop(mut self: Box<Self>) {
        self.teardown();
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        self.teardown();
    }
}

/// Create, bind, and listen on the daemon Unix socket at `path`.
fn bind_listen(path: &Path) -> std::result::Result<RawFd, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("rocjitsu daemon: cannot create {}: {e}", parent.display()))?;
    }
    let path_bytes = path.as_os_str().as_encoded_bytes();
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    if path_bytes.len() >= std::mem::size_of_val(&addr.sun_path) {
        return Err(format!(
            "rocjitsu daemon: socket path too long ({} bytes): {}",
            path_bytes.len(),
            path.display()
        ));
    }
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    for (dst, &src) in addr.sun_path.iter_mut().zip(path_bytes.iter()) {
        *dst = src as libc::c_char;
    }

    // A stale socket from a previous run would make bind() fail with
    // EADDRINUSE, so remove it first.
    let _ = std::fs::remove_file(path);

    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(format!(
            "rocjitsu daemon: socket() failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let addr_len = std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t;
    let rc = unsafe {
        libc::bind(
            fd,
            &addr as *const libc::sockaddr_un as *const libc::sockaddr,
            addr_len,
        )
    };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(format!(
            "rocjitsu daemon: bind({}) failed: {err}",
            path.display()
        ));
    }
    if unsafe { libc::listen(fd, 16) } != 0 {
        let err = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(format!("rocjitsu daemon: listen() failed: {err}"));
    }
    Ok(fd)
}

/// Accept debugger connections on `listen_fd` until `stop` is set,
/// serving each over the [`mirage_debug`] protocol against `backend`.
///
/// Mirrors [`accept_loop`]: connected client fds are tracked in `clients`
/// so teardown can `shutdown()` them to unblock a parked `read`, letting
/// each connection thread (and then this loop) exit cleanly.
fn debug_accept_loop(
    listen_fd: RawFd,
    backend: Arc<DebugTarget>,
    stop: Arc<AtomicBool>,
    clients: Arc<Mutex<Vec<RawFd>>>,
) {
    let mut handles: Vec<JoinHandle<()>> = Vec::new();
    loop {
        let client = unsafe { libc::accept(listen_fd, std::ptr::null_mut(), std::ptr::null_mut()) };
        if client < 0 {
            break;
        }
        if stop.load(Ordering::SeqCst) {
            unsafe { libc::close(client) };
            break;
        }
        if let Ok(mut fds) = clients.lock() {
            fds.push(client);
        }
        let backend = backend.clone();
        let client_list = clients.clone();
        match std::thread::Builder::new()
            .name("rocjitsu-debug-client".to_string())
            .spawn(move || {
                // The stream owns `client` and closes it on drop.
                let stream = unsafe { UnixStream::from_raw_fd(client) };
                let _ = mirage_debug::server::serve_connection(stream, backend.as_ref());
                if let Ok(mut fds) = client_list.lock() {
                    fds.retain(|&f| f != client);
                }
            }) {
            Ok(h) => handles.push(h),
            Err(_) => unsafe {
                libc::close(client);
            },
        }
    }
    for h in handles {
        let _ = h.join();
    }
}

/// Accept connections until `stop` is set (signalled by shutting the
/// listening socket down), spawning one thread per client and joining
/// them all before returning.
fn accept_loop(
    listen_fd: RawFd,
    shared: Arc<Shared>,
    stop: Arc<AtomicBool>,
    clients: Arc<Mutex<Vec<RawFd>>>,
) {
    let mut handles: Vec<JoinHandle<()>> = Vec::new();
    loop {
        let client = unsafe { libc::accept(listen_fd, std::ptr::null_mut(), std::ptr::null_mut()) };
        if client < 0 {
            break;
        }
        if stop.load(Ordering::SeqCst) {
            unsafe { libc::close(client) };
            break;
        }
        if let Ok(mut fds) = clients.lock() {
            fds.push(client);
        }
        let client_shared = shared.clone();
        let client_list = clients.clone();
        match std::thread::Builder::new()
            .name("rocjitsu-client".to_string())
            .spawn(move || {
                handle_client(client, &client_shared);
                // Drop our fd from the live set once we're done so
                // shutdown does not race a closed/reused descriptor.
                if let Ok(mut fds) = client_list.lock() {
                    fds.retain(|&f| f != client);
                }
                unsafe { libc::close(client) };
            }) {
            Ok(h) => handles.push(h),
            Err(_) => unsafe {
                libc::close(client);
            },
        }
    }
    for h in handles {
        let _ = h.join();
    }
}

/// Serve a single client connection until it closes or errors. Mirrors
/// the C daemon's `handle_client`.
fn handle_client(fd: RawFd, shared: &Shared) {
    let lib = &shared.lib;
    let vm = shared.vm;
    let mut process_id: u32 = 0;

    loop {
        let mut header = [0u8; RPC_HEADER_LEN];
        if !recv_exact(fd, &mut header) {
            break;
        }
        let (opcode, request_id, payload_bytes) = parse_header(&header);

        let keep_going = match opcode {
            RPC_HANDSHAKE => {
                let (status, pid) = unsafe { lib.vm_device_open(vm) };
                if status != rocjitsu_sys::ROCJITSU_STATUS_SUCCESS {
                    let resp = build_header(0, request_id, 0, -1);
                    send_exact(fd, &resp);
                    false
                } else {
                    process_id = pid;
                    let (_s, gpu_id) = unsafe { lib.vm_gpu_id(vm) };
                    let topo = unsafe { lib.vm_topology_path(vm) }
                        .map(|c| c.to_bytes().to_vec())
                        .unwrap_or_default();
                    let drm = unsafe { lib.vm_drm_path(vm) }
                        .map(|c| c.to_bytes().to_vec())
                        .unwrap_or_default();
                    // Device metadata for the client's libdrm/DRM
                    // emulation. A zeroed payload (present == 0) is a
                    // valid fallback for libraries without the symbol.
                    let gpu_info = unsafe { lib.vm_gpu_info(vm) }.unwrap_or_default();
                    let payload = RPC_HANDSHAKE_RESPONSE_LEN + topo.len() + drm.len();
                    let mut msg = Vec::with_capacity(RPC_HEADER_LEN + payload);
                    msg.extend_from_slice(&build_header(0, request_id, payload as u32, 0));
                    // RpcHandshakeResponse: version, gpu_id, topo_len,
                    // drm_len, gpu_info, then the topo/drm path strings.
                    msg.extend_from_slice(&RPC_PROTOCOL_VERSION.to_ne_bytes());
                    msg.extend_from_slice(&gpu_id.to_ne_bytes());
                    msg.extend_from_slice(&(topo.len() as u32).to_ne_bytes());
                    msg.extend_from_slice(&(drm.len() as u32).to_ne_bytes());
                    msg.extend_from_slice(gpu_info.as_bytes());
                    msg.extend_from_slice(&topo);
                    msg.extend_from_slice(&drm);
                    send_exact(fd, &msg)
                }
            }

            RPC_CLOSE => {
                unsafe { lib.vm_device_close(vm, process_id) };
                process_id = 0;
                let resp = build_header(0, request_id, 0, 0);
                send_exact(fd, &resp);
                false
            }

            RPC_MMAP => {
                let mut req = [0u8; 32];
                if !recv_exact(fd, &mut req) {
                    false
                } else {
                    let mut map = RjVmMap {
                        addr: u64::from_ne_bytes(req[0..8].try_into().unwrap()),
                        length: u64::from_ne_bytes(req[8..16].try_into().unwrap()),
                        prot: i32::from_ne_bytes(req[16..20].try_into().unwrap()) as u32,
                        flags: i32::from_ne_bytes(req[20..24].try_into().unwrap()) as u32,
                        offset: i64::from_ne_bytes(req[24..32].try_into().unwrap()),
                        mapped_addr: 0,
                    };
                    let offset = map.offset;
                    unsafe { lib.vm_device_map_as(vm, process_id, &mut map) };
                    let result = if map.mapped_addr == u64::MAX {
                        -last_errno()
                    } else {
                        0
                    };
                    // Header + RpcMmapResponse{mapped_addr}.
                    let mut msg = Vec::with_capacity(RPC_HEADER_LEN + 8);
                    msg.extend_from_slice(&build_header(0, request_id, 8, result));
                    msg.extend_from_slice(&map.mapped_addr.to_ne_bytes());
                    match unsafe { lib.vm_get_shared_mem_as(vm, process_id, offset) } {
                        Some(memfd) => send_msg(fd, &msg, &[memfd]),
                        None => send_exact(fd, &msg),
                    }
                }
            }

            RPC_MUNMAP => {
                let mut req = [0u8; 16];
                if !recv_exact(fd, &mut req) {
                    false
                } else {
                    let mut unmap = RjVmUnmap {
                        addr: u64::from_ne_bytes(req[0..8].try_into().unwrap()),
                        length: u64::from_ne_bytes(req[8..16].try_into().unwrap()),
                    };
                    unsafe { lib.vm_device_unmap_as(vm, process_id, &mut unmap) };
                    let resp = build_header(0, request_id, 0, 0);
                    send_exact(fd, &resp)
                }
            }

            RPC_IOCTL => {
                if payload_bytes > MAX_IOCTL_PAYLOAD || (payload_bytes as usize) < 8 {
                    false
                } else {
                    let mut payload = vec![0u8; payload_bytes as usize];
                    if !recv_exact(fd, &mut payload) {
                        false
                    } else {
                        let ioctl_cmd = u32::from_ne_bytes(payload[0..4].try_into().unwrap());
                        let args_bytes = u32::from_ne_bytes(payload[4..8].try_into().unwrap());
                        let mut cmd = RjVmCmd {
                            cmd: ioctl_cmd,
                            buf: payload[8..].as_mut_ptr() as *mut c_void,
                            buf_size: args_bytes as usize,
                            result: 0,
                            shared_handle: -1,
                        };
                        unsafe { lib.vm_execute_as(vm, process_id, &mut cmd) };
                        // `buf_size` is updated in place; clamp the slice
                        // we read back to what the payload actually holds.
                        let out_len = cmd.buf_size.min(payload.len().saturating_sub(8));
                        let resp =
                            build_header(RPC_IOCTL, request_id, cmd.buf_size as u32, cmd.result);
                        if cmd.shared_handle >= 0 {
                            let mut msg = Vec::with_capacity(RPC_HEADER_LEN + out_len);
                            msg.extend_from_slice(&resp);
                            msg.extend_from_slice(&payload[8..8 + out_len]);
                            send_msg(fd, &msg, &[cmd.shared_handle])
                        } else if send_exact(fd, &resp) {
                            out_len == 0 || send_exact(fd, &payload[8..8 + out_len])
                        } else {
                            false
                        }
                    }
                }
            }

            _ => false,
        };

        if !keep_going {
            break;
        }
    }

    if process_id != 0 {
        unsafe { lib.vm_device_close(vm, process_id) };
    }
}

/// Build a 16-byte RPC header.
fn build_header(opcode: u16, request_id: u32, payload_bytes: u32, result: i32) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0..2].copy_from_slice(&opcode.to_ne_bytes());
    // bytes 2..4 reserved (zero)
    h[4..8].copy_from_slice(&request_id.to_ne_bytes());
    h[8..12].copy_from_slice(&payload_bytes.to_ne_bytes());
    h[12..16].copy_from_slice(&result.to_ne_bytes());
    h
}

/// Parse the `(opcode, request_id, payload_bytes)` fields from a header.
fn parse_header(h: &[u8; 16]) -> (u16, u32, u32) {
    let opcode = u16::from_ne_bytes(h[0..2].try_into().unwrap());
    let request_id = u32::from_ne_bytes(h[4..8].try_into().unwrap());
    let payload_bytes = u32::from_ne_bytes(h[8..12].try_into().unwrap());
    (opcode, request_id, payload_bytes)
}

/// `errno` from the most recent failing libc call.
fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// Read exactly `buf.len()` bytes, handling partial reads. Returns false
/// on EOF/error.
fn recv_exact(fd: RawFd, buf: &mut [u8]) -> bool {
    let mut read = 0;
    while read < buf.len() {
        let n = unsafe {
            libc::recv(
                fd,
                buf[read..].as_mut_ptr() as *mut c_void,
                buf.len() - read,
                0,
            )
        };
        if n < 0 && last_errno() == libc::EINTR {
            continue;
        }
        if n <= 0 {
            return false;
        }
        read += n as usize;
    }
    true
}

/// Write exactly `buf.len()` bytes, handling partial writes. Returns
/// false on error.
fn send_exact(fd: RawFd, buf: &[u8]) -> bool {
    let mut sent = 0;
    while sent < buf.len() {
        let n = unsafe {
            libc::send(
                fd,
                buf[sent..].as_ptr() as *const c_void,
                buf.len() - sent,
                libc::MSG_NOSIGNAL,
            )
        };
        if n < 0 && last_errno() == libc::EINTR {
            continue;
        }
        if n <= 0 {
            return false;
        }
        sent += n as usize;
    }
    true
}

/// Send a message together with `fds` passed as `SCM_RIGHTS` ancillary
/// data (a single `sendmsg`). Used to hand GPU `memfd`s to the workload.
fn send_msg(fd: RawFd, data: &[u8], fds: &[RawFd]) -> bool {
    if fds.is_empty() {
        return send_exact(fd, data);
    }
    let mut iov = libc::iovec {
        iov_base: data.as_ptr() as *mut c_void,
        iov_len: data.len(),
    };
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    let fd_bytes = std::mem::size_of_val(fds);
    let cmsg_space = unsafe { libc::CMSG_SPACE(fd_bytes as u32) } as usize;
    let mut cmsg_buf = vec![0u8; cmsg_space];

    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut c_void;
    msg.msg_controllen = cmsg_space as _;

    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null() {
            return false;
        }
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(fd_bytes as u32) as _;
        std::ptr::copy_nonoverlapping(fds.as_ptr() as *const u8, libc::CMSG_DATA(cmsg), fd_bytes);
        let n = libc::sendmsg(fd, &msg, libc::MSG_NOSIGNAL);
        n >= 0
    }
}
