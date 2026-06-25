//! `rocjitsu_sys` — runtime FFI bindings to the rocjitsu VM C API.
//!
//! mirage drives the rocjitsu functional emulator directly through its
//! public C API (`rj_vm_*`, declared in `rocjitsu/vm/rj_vm.h`) instead
//! of shelling out to the `rocjitsu` command-line tool. This crate is
//! the thin, unsafe binding layer between Rust and that C API.
//!
//! # Why runtime loading?
//!
//! The rocjitsu shared library is *discovered at runtime* — it ships in
//! a ROCm wheel / system install and is never present when mirage is
//! built, so we cannot link it at build time. Instead we `dlopen` it
//! (via [`libloading`]) and resolve the handful of `rj_vm_*` symbols we
//! need. The single self-contained `librocjitsu.so` exports the full
//! VM API in addition to the LD_PRELOAD interposer and the HSA tools
//! hooks, so loading that one library is enough to interpose a workload,
//! host the daemon, *and* translate code objects.
//!
//! # Safety
//!
//! Every function here is `unsafe`: callers must uphold the C API's
//! contract (valid pointers, correct lifetimes, single-threaded VM
//! creation, etc.). Higher layers (`mirage_rocjitsu`) wrap these in
//! safe, RAII-managed abstractions.

use std::ffi::{CStr, OsStr};
use std::os::raw::{c_char, c_int, c_void};

/// Status codes returned by the rocjitsu C API (`rj_status_t`).
pub type RjStatus = c_int;

/// Operation completed successfully.
pub const ROCJITSU_STATUS_SUCCESS: RjStatus = 0;
/// Unspecified error.
pub const ROCJITSU_STATUS_ERROR: RjStatus = 1;
/// One or more arguments are invalid.
pub const ROCJITSU_STATUS_INVALID_ARGUMENT: RjStatus = 2;
/// Insufficient resources to complete the operation.
pub const ROCJITSU_STATUS_OUT_OF_RESOURCES: RjStatus = 3;
/// The supplied code object is malformed or unsupported.
pub const ROCJITSU_STATUS_INVALID_CODE_OBJECT: RjStatus = 4;
/// A required file could not be opened or read.
pub const ROCJITSU_STATUS_INVALID_FILE: RjStatus = 5;

/// Platform-specific handle type (`rj_handle_t`); an fd on Linux.
pub type RjHandle = c_int;

/// Opaque VM handle (`rj_vm_t`). Only ever held behind a pointer.
#[repr(C)]
pub struct RjVm {
    _private: [u8; 0],
}

/// VM creation mode (`rj_vm_mode_t`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RjVmMode {
    /// Standalone simulation driven by the caller.
    Default = 0,
    /// Single-process serving for an LD_PRELOAD interposer (in-process).
    Local = 1,
    /// Multi-process serving (daemon): GPU allocations are memfd-backed
    /// for cross-process sharing via `SCM_RIGHTS`.
    Daemon = 2,
}

/// Device command descriptor (`rj_vm_cmd_t`).
#[repr(C)]
#[derive(Debug)]
pub struct RjVmCmd {
    /// Platform-specific command number (an `AMDKFD_IOC_*` ioctl).
    pub cmd: u32,
    /// Command arguments buffer (with inlined arrays).
    pub buf: *mut c_void,
    /// Total size of the arguments buffer.
    pub buf_size: usize,
    /// `[out]` Return code (0 on success, negative errno on failure).
    pub result: i32,
    /// `[out]` Backing handle for shareable allocations, or -1.
    pub shared_handle: RjHandle,
}

/// Device memory mapping descriptor (`rj_vm_map_t`).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct RjVmMap {
    /// Requested mapping address.
    pub addr: u64,
    /// Length in bytes to map.
    pub length: u64,
    /// Platform-specific offset encoding.
    pub offset: i64,
    /// Memory protection flags.
    pub prot: u32,
    /// Mapping flags.
    pub flags: u32,
    /// `[out]` Address the mapping was placed at.
    pub mapped_addr: u64,
}

/// Device memory unmapping descriptor (`rj_vm_unmap_t`).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct RjVmUnmap {
    /// Address of the mapping to unmap.
    pub addr: u64,
    /// Length in bytes to unmap.
    pub length: u64,
}

/// Simulated GPU metadata (`rj_vm_gpu_info_t`, 312 bytes).
///
/// Sent verbatim inside the daemon handshake response so a workload's
/// interposer can emulate libdrm/DRM device queries client-side. The
/// layout must match `rocjitsu/vm/rj_vm.h` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RjVmGpuInfo {
    /// Non-zero when this payload was populated by the VM.
    pub present: u32,
    pub gpu_id: u32,
    pub gfx_target_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub family_id: u32,
    pub unique_id: u64,
    pub location_id: u32,
    pub domain: u32,
    pub hive_id: u64,
    pub drm_render_minor: u32,
    pub revision_id: u32,
    pub pci_revision_id: u32,
    pub simd_count: u32,
    pub max_waves_per_simd: u32,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub num_cu_per_sh: u32,
    pub simd_per_cu: u32,
    pub wave_front_size: u32,
    pub num_xcc: u32,
    pub max_slots_scratch_cu: u32,
    pub local_mem_size: u64,
    pub vram_type: u32,
    pub lds_size_kb: u32,
    pub mem_width: u32,
    pub mem_clk_max: u32,
    pub l1_size_kb: u32,
    pub l1_line_size: u32,
    pub l1_assoc: u32,
    pub l2_size_kb: u32,
    pub l2_line_size: u32,
    pub l2_assoc: u32,
    pub num_sdma_engines: u32,
    pub num_sdma_xgmi_engines: u32,
    pub num_cp_queues: u32,
    pub max_engine_clk_fcompute: u32,
    pub capability: u32,
    pub capability2: u32,
    pub debug_prop: u64,
    pub fw_version: u32,
    pub sdma_fw_version: u32,
    pub marketing_name: [c_char; 128],
}

impl Default for RjVmGpuInfo {
    fn default() -> Self {
        // All-zero is a valid "absent" payload (`present == 0`).
        unsafe { std::mem::zeroed() }
    }
}

impl RjVmGpuInfo {
    /// The raw little-/native-endian bytes of this struct, for wire
    /// serialisation. Sound because the struct is `#[repr(C)]` POD that
    /// matches `rj_vm_gpu_info_t` exactly.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self as *const Self as *const u8, std::mem::size_of::<Self>())
        }
    }
}

/// Opaque, ABI-stable wavefront identifier (`rj_dbg_wave_id_t`).
///
/// Encodes the physical slot location `(xcc<<48)|(se<<32)|(cu<<16)|slot`.
/// Treat it as opaque; use [`RjDbgWaveInfo`] to decode the coordinates.
pub type RjDbgWaveId = u64;

/// Sentinel wave id that never names a valid wavefront (`RJ_DBG_WAVE_NONE`).
pub const RJ_DBG_WAVE_NONE: RjDbgWaveId = 0xFFFF_FFFF_FFFF_FFFF;

/// Wavefront execution state (`rj_dbg_wave_state_t`).
pub const RJ_DBG_WAVE_HALTED: u32 = 0;
/// Wavefront eligible for scheduling.
pub const RJ_DBG_WAVE_RUNNING: u32 = 1;
/// Wavefront stalled on `s_waitcnt`.
pub const RJ_DBG_WAVE_WAITCNT: u32 = 2;
/// Wavefront stalled at a workgroup barrier.
pub const RJ_DBG_WAVE_BARRIER: u32 = 3;
/// Wavefront has seen `s_endpgm`; memory ops draining.
pub const RJ_DBG_WAVE_ENDING: u32 = 4;

/// Stop reason (`rj_dbg_stop_reason_t`): engine running.
pub const RJ_DBG_STOP_NONE: u32 = 0;
/// Stopped by an explicit suspend request.
pub const RJ_DBG_STOP_USER: u32 = 1;
/// A wavefront reached a PC breakpoint.
pub const RJ_DBG_STOP_BREAKPOINT: u32 = 2;
/// A requested single-step completed.
pub const RJ_DBG_STOP_STEP: u32 = 3;
/// Stopped before the first tick (attach-on-entry).
pub const RJ_DBG_STOP_ENTRY: u32 = 4;
/// Simulation finished; no more work.
pub const RJ_DBG_STOP_EXITED: u32 = 5;

/// Special-register selector (`rj_dbg_special_reg_t`).
pub const RJ_DBG_SPECIAL_PC: u32 = 0;
/// EXEC mask special register.
pub const RJ_DBG_SPECIAL_EXEC: u32 = 1;
/// VCC special register.
pub const RJ_DBG_SPECIAL_VCC: u32 = 2;
/// STATUS special register.
pub const RJ_DBG_SPECIAL_STATUS: u32 = 3;
/// MODE special register.
pub const RJ_DBG_SPECIAL_MODE: u32 = 4;
/// M0 special register.
pub const RJ_DBG_SPECIAL_M0: u32 = 5;

/// Snapshot of a single wavefront's architected control state
/// (`rj_dbg_wave_info_t`, 96 bytes). Layout must match the C struct in
/// `rocjitsu/vm/rj_vm_debug.h` byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RjDbgWaveInfo {
    /// Opaque wave id (see [`RjDbgWaveId`]).
    pub id: RjDbgWaveId,
    /// XCC / XCD index.
    pub xcc: u32,
    /// Shader-engine index within the XCD.
    pub se: u32,
    /// Compute-unit index within the shader engine.
    pub cu: u32,
    /// Wavefront slot index within the compute unit.
    pub slot: u32,
    /// Execution state ([`RJ_DBG_WAVE_RUNNING`] etc.).
    pub state: u32,
    /// Program counter (byte address).
    pub pc: u64,
    /// EXEC mask.
    pub exec: u64,
    /// VCC value.
    pub vcc: u64,
    /// STATUS register.
    pub status: u32,
    /// MODE register.
    pub mode: u32,
    /// M0 register.
    pub m0: u32,
    /// Lanes per wavefront (32 or 64).
    pub wave_size: u32,
    /// Allocated scalar registers.
    pub num_sgprs: u32,
    /// Allocated vector registers.
    pub num_vgprs: u32,
    /// Owning workgroup id.
    pub wg_id: u32,
    /// Owning dispatch id.
    pub dispatch_id: u32,
    /// Owning KFD process id (PASID analog).
    pub process_id: u32,
}

// Raw C function-pointer signatures for the symbols we resolve.
type FnVmCreate = unsafe extern "C" fn(*const c_char, RjVmMode, *mut *mut RjVm) -> RjStatus;
type FnVmCreateFromString =
    unsafe extern "C" fn(*const c_char, RjVmMode, *mut *mut RjVm) -> RjStatus;
type FnVmRun = unsafe extern "C" fn(*mut RjVm, *mut u64) -> RjStatus;
type FnVmRequestExit = unsafe extern "C" fn(*mut RjVm, *const c_char);
type FnVmDestroy = unsafe extern "C" fn(*mut RjVm);
type FnVmDeviceOpen = unsafe extern "C" fn(*mut RjVm, *mut u32) -> RjStatus;
type FnVmDeviceClose = unsafe extern "C" fn(*mut RjVm, u32) -> RjStatus;
type FnVmExecuteAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmCmd) -> RjStatus;
type FnVmDeviceMapAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmMap) -> RjStatus;
type FnVmDeviceUnmapAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmUnmap) -> RjStatus;
type FnVmGpuId = unsafe extern "C" fn(*mut RjVm, *mut u32) -> RjStatus;
type FnVmGpuInfo = unsafe extern "C" fn(*mut RjVm, *mut RjVmGpuInfo) -> RjStatus;
type FnVmTopologyPath = unsafe extern "C" fn(*mut RjVm, *mut *const c_char) -> RjStatus;
type FnVmDrmPath = unsafe extern "C" fn(*mut RjVm, *mut *const c_char) -> RjStatus;
type FnVmGetSharedMemAs = unsafe extern "C" fn(*mut RjVm, u32, i64, *mut RjHandle) -> RjStatus;

// Debug control surface (`rj_vm_debug.h`). Optional: only present in
// rocjitsu libraries built with the debug API. Resolved as a bundle so the
// debugger is either fully available or entirely absent.
type FnDbgSupported = unsafe extern "C" fn(*mut RjVm, *mut c_int) -> RjStatus;
type FnDbgSuspend = unsafe extern "C" fn(*mut RjVm) -> RjStatus;
type FnDbgResume = unsafe extern "C" fn(*mut RjVm) -> RjStatus;
type FnDbgStep = unsafe extern "C" fn(*mut RjVm, u64) -> RjStatus;
type FnDbgStatus = unsafe extern "C" fn(*mut RjVm, *mut c_int, *mut u32, *mut u64) -> RjStatus;
type FnDbgWaitStop = unsafe extern "C" fn(*mut RjVm, u64, *mut c_int, *mut u32) -> RjStatus;
type FnDbgWaveCount = unsafe extern "C" fn(*mut RjVm, *mut u32) -> RjStatus;
type FnDbgWaveList =
    unsafe extern "C" fn(*mut RjVm, *mut RjDbgWaveInfo, u32, *mut u32) -> RjStatus;
type FnDbgWaveInfo = unsafe extern "C" fn(*mut RjVm, RjDbgWaveId, *mut RjDbgWaveInfo) -> RjStatus;
type FnDbgReadSgpr =
    unsafe extern "C" fn(*mut RjVm, RjDbgWaveId, u32, u32, *mut u32) -> RjStatus;
type FnDbgWriteSgpr = unsafe extern "C" fn(*mut RjVm, RjDbgWaveId, u32, u32) -> RjStatus;
type FnDbgReadVgpr =
    unsafe extern "C" fn(*mut RjVm, RjDbgWaveId, u32, u32, u32, *mut u32) -> RjStatus;
type FnDbgWriteVgpr = unsafe extern "C" fn(*mut RjVm, RjDbgWaveId, u32, u32, u32) -> RjStatus;
type FnDbgReadSpecial = unsafe extern "C" fn(*mut RjVm, RjDbgWaveId, u32, *mut u64) -> RjStatus;
type FnDbgWriteSpecial = unsafe extern "C" fn(*mut RjVm, RjDbgWaveId, u32, u64) -> RjStatus;
type FnDbgReadMemory =
    unsafe extern "C" fn(*mut RjVm, u32, u64, *mut c_void, u64) -> RjStatus;
type FnDbgWriteMemory =
    unsafe extern "C" fn(*mut RjVm, u32, u64, *const c_void, u64) -> RjStatus;
type FnDbgBreakSet = unsafe extern "C" fn(*mut RjVm, u64, *mut u32) -> RjStatus;
type FnDbgBreakClear = unsafe extern "C" fn(*mut RjVm, u32) -> RjStatus;
type FnDbgBreakList = unsafe extern "C" fn(*mut RjVm, *mut u64, u32, *mut u32) -> RjStatus;

/// The resolved `rj_vm_debug_*` entry points, present as a unit when the
/// loaded library exports the debug API.
#[derive(Clone, Copy)]
struct DebugFns {
    supported: FnDbgSupported,
    suspend: FnDbgSuspend,
    resume: FnDbgResume,
    step: FnDbgStep,
    status: FnDbgStatus,
    wait_stop: FnDbgWaitStop,
    wave_count: FnDbgWaveCount,
    wave_list: FnDbgWaveList,
    wave_info: FnDbgWaveInfo,
    read_sgpr: FnDbgReadSgpr,
    write_sgpr: FnDbgWriteSgpr,
    read_vgpr: FnDbgReadVgpr,
    write_vgpr: FnDbgWriteVgpr,
    read_special: FnDbgReadSpecial,
    write_special: FnDbgWriteSpecial,
    read_memory: FnDbgReadMemory,
    write_memory: FnDbgWriteMemory,
    break_set: FnDbgBreakSet,
    break_clear: FnDbgBreakClear,
    break_list: FnDbgBreakList,
}


/// A loaded rocjitsu shared library with its `rj_vm_*` entry points
/// resolved.
///
/// The [`libloading::Library`] is kept alive for the lifetime of this
/// struct so the resolved function pointers remain valid. All methods
/// are `unsafe`: they call directly into C and require the caller to
/// uphold the rocjitsu API contract.
pub struct Lib {
    // Resolved function pointers. The owning library must outlive them,
    // so it is kept in `_lib` and dropped last.
    vm_create: FnVmCreate,
    vm_create_from_string: FnVmCreateFromString,
    vm_run: FnVmRun,
    vm_request_exit: FnVmRequestExit,
    vm_destroy: FnVmDestroy,
    vm_device_open: FnVmDeviceOpen,
    vm_device_close: FnVmDeviceClose,
    vm_execute_as: FnVmExecuteAs,
    vm_device_map_as: FnVmDeviceMapAs,
    vm_device_unmap_as: FnVmDeviceUnmapAs,
    vm_gpu_id: FnVmGpuId,
    // Optional: only present in protocol-v3+ rocjitsu libraries. When
    // absent, daemon clients fall back to a zeroed (absent) gpu_info.
    vm_gpu_info: Option<FnVmGpuInfo>,
    vm_topology_path: FnVmTopologyPath,
    vm_drm_path: FnVmDrmPath,
    vm_get_shared_mem_as: FnVmGetSharedMemAs,
    // Optional: present only when the library exports the debug API
    // (`rj_vm_debug.h`). Resolved all-or-nothing.
    debug: Option<DebugFns>,
    _lib: libloading::Library,
}

// The resolved entry points are plain C function pointers and the VM
// they operate on is internally synchronised by rocjitsu (the daemon
// shares one VM across an engine thread and many client threads via the
// `*_as` API), so the handle is safe to move and share across threads.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

impl Lib {
    /// Load the rocjitsu shared library at `path` and resolve the
    /// `rj_vm_*` entry points.
    ///
    /// # Safety
    /// Loading an arbitrary shared library runs its initialisers; the
    /// caller must ensure `path` is a trusted rocjitsu library.
    pub unsafe fn open(path: impl AsRef<OsStr>) -> Result<Self, libloading::Error> {
        unsafe {
            let lib = libloading::Library::new(path.as_ref())?;
            // `*symbol` copies out the raw fn pointer; the symbol's
            // borrow of `lib` ends here but `lib` is moved into the
            // returned struct, keeping the code mapped.
            let vm_create = *lib.get::<FnVmCreate>(b"rj_vm_create\0")?;
            let vm_create_from_string =
                *lib.get::<FnVmCreateFromString>(b"rj_vm_create_from_string\0")?;
            let vm_run = *lib.get::<FnVmRun>(b"rj_vm_run\0")?;
            let vm_request_exit = *lib.get::<FnVmRequestExit>(b"rj_vm_request_exit\0")?;
            let vm_destroy = *lib.get::<FnVmDestroy>(b"rj_vm_destroy\0")?;
            let vm_device_open = *lib.get::<FnVmDeviceOpen>(b"rj_vm_device_open\0")?;
            let vm_device_close = *lib.get::<FnVmDeviceClose>(b"rj_vm_device_close\0")?;
            let vm_execute_as = *lib.get::<FnVmExecuteAs>(b"rj_vm_execute_as\0")?;
            let vm_device_map_as = *lib.get::<FnVmDeviceMapAs>(b"rj_vm_device_map_as\0")?;
            let vm_device_unmap_as = *lib.get::<FnVmDeviceUnmapAs>(b"rj_vm_device_unmap_as\0")?;
            let vm_gpu_id = *lib.get::<FnVmGpuId>(b"rj_vm_gpu_id\0")?;
            // Optional symbol: tolerate older libraries that predate it.
            let vm_gpu_info = lib.get::<FnVmGpuInfo>(b"rj_vm_gpu_info\0").map(|s| *s).ok();
            let vm_topology_path = *lib.get::<FnVmTopologyPath>(b"rj_vm_topology_path\0")?;
            let vm_drm_path = *lib.get::<FnVmDrmPath>(b"rj_vm_drm_path\0")?;
            let vm_get_shared_mem_as =
                *lib.get::<FnVmGetSharedMemAs>(b"rj_vm_get_shared_mem_as\0")?;
            // Resolve the optional debug surface all-or-nothing: a closure
            // that returns None the moment any symbol is missing, so older
            // libraries simply report the debugger as unavailable.
            let debug = (|| {
                Some(DebugFns {
                    supported: *lib.get::<FnDbgSupported>(b"rj_vm_debug_supported\0").ok()?,
                    suspend: *lib.get::<FnDbgSuspend>(b"rj_vm_debug_suspend\0").ok()?,
                    resume: *lib.get::<FnDbgResume>(b"rj_vm_debug_resume\0").ok()?,
                    step: *lib.get::<FnDbgStep>(b"rj_vm_debug_step\0").ok()?,
                    status: *lib.get::<FnDbgStatus>(b"rj_vm_debug_status\0").ok()?,
                    wait_stop: *lib.get::<FnDbgWaitStop>(b"rj_vm_debug_wait_stop\0").ok()?,
                    wave_count: *lib.get::<FnDbgWaveCount>(b"rj_vm_debug_wave_count\0").ok()?,
                    wave_list: *lib.get::<FnDbgWaveList>(b"rj_vm_debug_wave_list\0").ok()?,
                    wave_info: *lib.get::<FnDbgWaveInfo>(b"rj_vm_debug_wave_info\0").ok()?,
                    read_sgpr: *lib.get::<FnDbgReadSgpr>(b"rj_vm_debug_read_sgpr\0").ok()?,
                    write_sgpr: *lib.get::<FnDbgWriteSgpr>(b"rj_vm_debug_write_sgpr\0").ok()?,
                    read_vgpr: *lib.get::<FnDbgReadVgpr>(b"rj_vm_debug_read_vgpr\0").ok()?,
                    write_vgpr: *lib.get::<FnDbgWriteVgpr>(b"rj_vm_debug_write_vgpr\0").ok()?,
                    read_special: *lib
                        .get::<FnDbgReadSpecial>(b"rj_vm_debug_read_special\0")
                        .ok()?,
                    write_special: *lib
                        .get::<FnDbgWriteSpecial>(b"rj_vm_debug_write_special\0")
                        .ok()?,
                    read_memory: *lib
                        .get::<FnDbgReadMemory>(b"rj_vm_debug_read_memory\0")
                        .ok()?,
                    write_memory: *lib
                        .get::<FnDbgWriteMemory>(b"rj_vm_debug_write_memory\0")
                        .ok()?,
                    break_set: *lib.get::<FnDbgBreakSet>(b"rj_vm_debug_break_set\0").ok()?,
                    break_clear: *lib
                        .get::<FnDbgBreakClear>(b"rj_vm_debug_break_clear\0")
                        .ok()?,
                    break_list: *lib.get::<FnDbgBreakList>(b"rj_vm_debug_break_list\0").ok()?,
                })
            })();
            Ok(Self {
                vm_create,
                vm_create_from_string,
                vm_run,
                vm_request_exit,
                vm_destroy,
                vm_device_open,
                vm_device_close,
                vm_execute_as,
                vm_device_map_as,
                vm_device_unmap_as,
                vm_gpu_id,
                vm_gpu_info,
                vm_topology_path,
                vm_drm_path,
                vm_get_shared_mem_as,
                debug,
                _lib: lib,
            })
        }
    }

    /// Create a VM from a JSON config file. Returns the status and the
    /// (possibly null) VM handle.
    ///
    /// # Safety
    /// `json_path` must be a valid C string path; the returned VM must
    /// eventually be released with [`Lib::vm_destroy`].
    pub unsafe fn vm_create(&self, json_path: &CStr, mode: RjVmMode) -> (RjStatus, *mut RjVm) {
        let mut vm: *mut RjVm = std::ptr::null_mut();
        let status = unsafe { (self.vm_create)(json_path.as_ptr(), mode, &mut vm) };
        (status, vm)
    }

    /// Create a VM from a JSON config string.
    ///
    /// # Safety
    /// See [`Lib::vm_create`].
    pub unsafe fn vm_create_from_string(
        &self,
        json: &CStr,
        mode: RjVmMode,
    ) -> (RjStatus, *mut RjVm) {
        let mut vm: *mut RjVm = std::ptr::null_mut();
        let status = unsafe { (self.vm_create_from_string)(json.as_ptr(), mode, &mut vm) };
        (status, vm)
    }

    /// Run the simulation engine until [`Lib::vm_request_exit`] is
    /// called (or the configured tick limit is reached). Blocks.
    ///
    /// # Safety
    /// `vm` must be a live handle from [`Lib::vm_create`].
    pub unsafe fn vm_run(&self, vm: *mut RjVm) -> RjStatus {
        unsafe { (self.vm_run)(vm, std::ptr::null_mut()) }
    }

    /// Ask the engine to stop at the next opportunity. Thread-safe.
    ///
    /// # Safety
    /// `vm` must be a live handle; `reason` (if any) a valid C string.
    pub unsafe fn vm_request_exit(&self, vm: *mut RjVm, reason: &CStr) {
        unsafe { (self.vm_request_exit)(vm, reason.as_ptr()) }
    }

    /// Destroy a VM handle.
    ///
    /// # Safety
    /// `vm` must not be used after this call.
    pub unsafe fn vm_destroy(&self, vm: *mut RjVm) {
        unsafe { (self.vm_destroy)(vm) }
    }

    /// Open the simulated device, creating a new KFD process. Returns
    /// the status and the new process id.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_device_open(&self, vm: *mut RjVm) -> (RjStatus, u32) {
        let mut pid: u32 = 0;
        let status = unsafe { (self.vm_device_open)(vm, &mut pid) };
        (status, pid)
    }

    /// Close a KFD process by id.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_device_close(&self, vm: *mut RjVm, process_id: u32) -> RjStatus {
        unsafe { (self.vm_device_close)(vm, process_id) }
    }

    /// Execute a device command on behalf of `process_id` (daemon mode).
    ///
    /// # Safety
    /// `vm` must be live and `cmd` a valid, writable descriptor whose
    /// `buf`/`buf_size` describe an accessible buffer.
    pub unsafe fn vm_execute_as(
        &self,
        vm: *mut RjVm,
        process_id: u32,
        cmd: *mut RjVmCmd,
    ) -> RjStatus {
        unsafe { (self.vm_execute_as)(vm, process_id, cmd) }
    }

    /// Map device memory on behalf of `process_id` (daemon mode).
    ///
    /// # Safety
    /// `vm` must be live and `map` a valid, writable descriptor.
    pub unsafe fn vm_device_map_as(
        &self,
        vm: *mut RjVm,
        process_id: u32,
        map: *mut RjVmMap,
    ) -> RjStatus {
        unsafe { (self.vm_device_map_as)(vm, process_id, map) }
    }

    /// Unmap device memory on behalf of `process_id` (daemon mode).
    ///
    /// # Safety
    /// `vm` must be live and `unmap` a valid descriptor.
    pub unsafe fn vm_device_unmap_as(
        &self,
        vm: *mut RjVm,
        process_id: u32,
        unmap: *mut RjVmUnmap,
    ) -> RjStatus {
        unsafe { (self.vm_device_unmap_as)(vm, process_id, unmap) }
    }

    /// Get the KFD `gpu_id` for the simulated device.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_gpu_id(&self, vm: *mut RjVm) -> (RjStatus, u32) {
        let mut gpu_id: u32 = 0;
        let status = unsafe { (self.vm_gpu_id)(vm, &mut gpu_id) };
        (status, gpu_id)
    }

    /// Fetch the simulated device metadata (`rj_vm_gpu_info`). Returns
    /// `None` when the loaded library predates the symbol or the call
    /// fails; callers should then send a zeroed (absent) payload.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_gpu_info(&self, vm: *mut RjVm) -> Option<RjVmGpuInfo> {
        let f = self.vm_gpu_info?;
        let mut info = RjVmGpuInfo::default();
        let status = unsafe { f(vm, &mut info) };
        if status != ROCJITSU_STATUS_SUCCESS {
            return None;
        }
        Some(info)
    }

    /// Get the sysfs topology directory path (owned by the VM).
    ///
    /// # Safety
    /// `vm` must be a live handle. The returned string borrows VM-owned
    /// memory valid until the VM is destroyed.
    pub unsafe fn vm_topology_path(&self, vm: *mut RjVm) -> Option<&CStr> {
        let mut ptr: *const c_char = std::ptr::null();
        let status = unsafe { (self.vm_topology_path)(vm, &mut ptr) };
        if status != ROCJITSU_STATUS_SUCCESS || ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) })
    }

    /// Get the DRM sysfs directory path (owned by the VM).
    ///
    /// # Safety
    /// See [`Lib::vm_topology_path`].
    pub unsafe fn vm_drm_path(&self, vm: *mut RjVm) -> Option<&CStr> {
        let mut ptr: *const c_char = std::ptr::null();
        let status = unsafe { (self.vm_drm_path)(vm, &mut ptr) };
        if status != ROCJITSU_STATUS_SUCCESS || ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) })
    }

    /// Get the backing memory handle (memfd) for `process_id` at the
    /// given KFD mmap `offset`, or `None` when there is no backing fd.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_get_shared_mem_as(
        &self,
        vm: *mut RjVm,
        process_id: u32,
        offset: i64,
    ) -> Option<RjHandle> {
        let mut handle: RjHandle = -1;
        let status = unsafe { (self.vm_get_shared_mem_as)(vm, process_id, offset, &mut handle) };
        if status != ROCJITSU_STATUS_SUCCESS || handle < 0 {
            return None;
        }
        Some(handle)
    }

    // ---- Debug control surface (`rj_vm_debug.h`) -----------------------
    //
    // Every method here forwards to the optional debug entry points. When
    // the loaded library predates the debug API they return
    // `ROCJITSU_STATUS_OUT_OF_RESOURCES`, matching the C contract for a VM
    // that cannot be debugged.

    /// Whether the loaded library exports the debug API at all.
    pub fn has_debug(&self) -> bool {
        self.debug.is_some()
    }

    /// Report whether debug control is available for `vm`.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_supported(&self, vm: *mut RjVm) -> bool {
        let Some(d) = self.debug.as_ref() else {
            return false;
        };
        let mut supported: c_int = 0;
        let status = unsafe { (d.supported)(vm, &mut supported) };
        status == ROCJITSU_STATUS_SUCCESS && supported != 0
    }

    /// Suspend the engine and block until it is quiescent.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_suspend(&self, vm: *mut RjVm) -> RjStatus {
        match self.debug.as_ref() {
            Some(d) => unsafe { (d.suspend)(vm) },
            None => ROCJITSU_STATUS_OUT_OF_RESOURCES,
        }
    }

    /// Resume free-running execution.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_resume(&self, vm: *mut RjVm) -> RjStatus {
        match self.debug.as_ref() {
            Some(d) => unsafe { (d.resume)(vm) },
            None => ROCJITSU_STATUS_OUT_OF_RESOURCES,
        }
    }

    /// Single-step the engine by `ticks` ticks, then suspend.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_step(&self, vm: *mut RjVm, ticks: u64) -> RjStatus {
        match self.debug.as_ref() {
            Some(d) => unsafe { (d.step)(vm, ticks) },
            None => ROCJITSU_STATUS_OUT_OF_RESOURCES,
        }
    }

    /// Query run/stop status: `(stopped, stop_reason, tick)`.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_status(&self, vm: *mut RjVm) -> Option<(bool, u32, u64)> {
        let d = self.debug.as_ref()?;
        let (mut stopped, mut reason, mut tick): (c_int, u32, u64) = (0, 0, 0);
        let status = unsafe { (d.status)(vm, &mut stopped, &mut reason, &mut tick) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some((stopped != 0, reason, tick))
    }

    /// Block up to `timeout_ms` for the engine to stop: `(stopped, reason)`.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_wait_stop(&self, vm: *mut RjVm, timeout_ms: u64) -> Option<(bool, u32)> {
        let d = self.debug.as_ref()?;
        let (mut stopped, mut reason): (c_int, u32) = (0, 0);
        let status = unsafe { (d.wait_stop)(vm, timeout_ms, &mut stopped, &mut reason) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some((stopped != 0, reason))
    }

    /// Count live wavefronts (state != HALTED).
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_wave_count(&self, vm: *mut RjVm) -> Option<u32> {
        let d = self.debug.as_ref()?;
        let mut count: u32 = 0;
        let status = unsafe { (d.wave_count)(vm, &mut count) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some(count)
    }

    /// Enumerate live wavefronts. The engine must be suspended.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_wave_list(&self, vm: *mut RjVm) -> Option<Vec<RjDbgWaveInfo>> {
        let d = self.debug.as_ref()?;
        // First query the total, then fetch in one shot.
        let mut total: u32 = 0;
        let status = unsafe { (d.wave_list)(vm, std::ptr::null_mut(), 0, &mut total) };
        if status != ROCJITSU_STATUS_SUCCESS {
            return None;
        }
        let mut out = vec![RjDbgWaveInfo::default(); total as usize];
        let mut got: u32 = 0;
        let status = unsafe { (d.wave_list)(vm, out.as_mut_ptr(), total, &mut got) };
        if status != ROCJITSU_STATUS_SUCCESS {
            return None;
        }
        out.truncate(got.min(total) as usize);
        Some(out)
    }

    /// Fetch one wavefront's control state by id.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_wave_info(&self, vm: *mut RjVm, wave: RjDbgWaveId) -> Option<RjDbgWaveInfo> {
        let d = self.debug.as_ref()?;
        let mut info = RjDbgWaveInfo::default();
        let status = unsafe { (d.wave_info)(vm, wave, &mut info) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some(info)
    }

    /// Read `count` scalar registers starting at `first` from `wave`.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_read_sgpr(
        &self,
        vm: *mut RjVm,
        wave: RjDbgWaveId,
        first: u32,
        count: u32,
    ) -> Option<Vec<u32>> {
        let d = self.debug.as_ref()?;
        let mut out = vec![0u32; count as usize];
        let status = unsafe { (d.read_sgpr)(vm, wave, first, count, out.as_mut_ptr()) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some(out)
    }

    /// Write a single scalar register.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_write_sgpr(
        &self,
        vm: *mut RjVm,
        wave: RjDbgWaveId,
        index: u32,
        value: u32,
    ) -> RjStatus {
        match self.debug.as_ref() {
            Some(d) => unsafe { (d.write_sgpr)(vm, wave, index, value) },
            None => ROCJITSU_STATUS_OUT_OF_RESOURCES,
        }
    }

    /// Read one vector register across `lane_count` lanes from `first_lane`.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_read_vgpr(
        &self,
        vm: *mut RjVm,
        wave: RjDbgWaveId,
        reg: u32,
        first_lane: u32,
        lane_count: u32,
    ) -> Option<Vec<u32>> {
        let d = self.debug.as_ref()?;
        let mut out = vec![0u32; lane_count as usize];
        let status =
            unsafe { (d.read_vgpr)(vm, wave, reg, first_lane, lane_count, out.as_mut_ptr()) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some(out)
    }

    /// Write one lane of a vector register.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_write_vgpr(
        &self,
        vm: *mut RjVm,
        wave: RjDbgWaveId,
        reg: u32,
        lane: u32,
        value: u32,
    ) -> RjStatus {
        match self.debug.as_ref() {
            Some(d) => unsafe { (d.write_vgpr)(vm, wave, reg, lane, value) },
            None => ROCJITSU_STATUS_OUT_OF_RESOURCES,
        }
    }

    /// Read a special register (see `RJ_DBG_SPECIAL_*`).
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_read_special(
        &self,
        vm: *mut RjVm,
        wave: RjDbgWaveId,
        which: u32,
    ) -> Option<u64> {
        let d = self.debug.as_ref()?;
        let mut value: u64 = 0;
        let status = unsafe { (d.read_special)(vm, wave, which, &mut value) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some(value)
    }

    /// Write a special register (see `RJ_DBG_SPECIAL_*`).
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_write_special(
        &self,
        vm: *mut RjVm,
        wave: RjDbgWaveId,
        which: u32,
        value: u64,
    ) -> RjStatus {
        match self.debug.as_ref() {
            Some(d) => unsafe { (d.write_special)(vm, wave, which, value) },
            None => ROCJITSU_STATUS_OUT_OF_RESOURCES,
        }
    }

    /// Read `size` bytes of device memory at `addr` in address space `vmid`.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_read_memory(
        &self,
        vm: *mut RjVm,
        vmid: u32,
        addr: u64,
        size: u64,
    ) -> Option<Vec<u8>> {
        let d = self.debug.as_ref()?;
        let mut out = vec![0u8; size as usize];
        let status =
            unsafe { (d.read_memory)(vm, vmid, addr, out.as_mut_ptr() as *mut c_void, size) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some(out)
    }

    /// Write `data` to device memory at `addr` in address space `vmid`.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_write_memory(
        &self,
        vm: *mut RjVm,
        vmid: u32,
        addr: u64,
        data: &[u8],
    ) -> RjStatus {
        match self.debug.as_ref() {
            Some(d) => unsafe {
                (d.write_memory)(
                    vm,
                    vmid,
                    addr,
                    data.as_ptr() as *const c_void,
                    data.len() as u64,
                )
            },
            None => ROCJITSU_STATUS_OUT_OF_RESOURCES,
        }
    }

    /// Set a PC breakpoint; returns its id.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_break_set(&self, vm: *mut RjVm, pc: u64) -> Option<u32> {
        let d = self.debug.as_ref()?;
        let mut id: u32 = 0;
        let status = unsafe { (d.break_set)(vm, pc, &mut id) };
        (status == ROCJITSU_STATUS_SUCCESS).then_some(id)
    }

    /// Clear a breakpoint by id.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_break_clear(&self, vm: *mut RjVm, id: u32) -> RjStatus {
        match self.debug.as_ref() {
            Some(d) => unsafe { (d.break_clear)(vm, id) },
            None => ROCJITSU_STATUS_OUT_OF_RESOURCES,
        }
    }

    /// List active breakpoint addresses.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn dbg_break_list(&self, vm: *mut RjVm) -> Option<Vec<u64>> {
        let d = self.debug.as_ref()?;
        let mut total: u32 = 0;
        let status = unsafe { (d.break_list)(vm, std::ptr::null_mut(), 0, &mut total) };
        if status != ROCJITSU_STATUS_SUCCESS {
            return None;
        }
        let mut out = vec![0u64; total as usize];
        let mut got: u32 = 0;
        let status = unsafe { (d.break_list)(vm, out.as_mut_ptr(), total, &mut got) };
        if status != ROCJITSU_STATUS_SUCCESS {
            return None;
        }
        out.truncate(got.min(total) as usize);
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C `rj_vm_*` structs must match the rocjitsu headers exactly,
    /// or every FFI call corrupts memory. Pin the sizes the daemon RPC
    /// code relies on.
    #[test]
    fn struct_sizes_match_c_abi() {
        assert_eq!(std::mem::size_of::<RjVmMap>(), 40);
        assert_eq!(std::mem::size_of::<RjVmUnmap>(), 16);
        // rj_vm_cmd_t: u32 + (pad) + ptr + usize + i32 + i32 on 64-bit.
        assert_eq!(std::mem::size_of::<RjVmCmd>(), 32);
        assert_eq!(RjVmMode::Daemon as i32, 2);
        // rj_vm_gpu_info_t — must match the 312-byte RpcGpuInfo the
        // daemon handshake embeds (static_assert in rpc.h).
        assert_eq!(std::mem::size_of::<RjVmGpuInfo>(), 312);
        // rj_dbg_wave_info_t — must match rj_vm_debug.h byte-for-byte
        // (u64 id, then the coordinate/state words, with 4 bytes of pad
        // before the first u64 field `pc`).
        assert_eq!(std::mem::size_of::<RjDbgWaveInfo>(), 96);
    }
}
