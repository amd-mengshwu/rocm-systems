// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm_debug.h
/// @brief Public C API for inspecting and controlling a running rocjitsu VM.
///
/// @details This is a debugger control surface layered on top of the core
/// @ref rj_vm.h API. It exposes the primitives a source/ISA debugger needs:
/// suspend / resume / single-step of the simulation engine, enumeration of
/// in-flight wavefronts, read/write access to a wavefront's architected
/// registers (SGPR / VGPR / special registers), read/write access to global
/// device memory, and program-counter breakpoints.
///
/// ## Relationship to rocgdb / amd-dbgapi
///
/// The data model here is deliberately shaped to match the AMD GPU debug
/// stack so a thin shim can later expose rocjitsu through `amd-dbgapi`
/// (which `rocgdb` consumes) without re-plumbing the engine:
///
///   * @ref rj_dbg_wave_info_t mirrors an `amd_dbgapi_wave_id_t` plus the
///     fields `amd-dbgapi` reports for a wave (PC, EXEC, STATUS, the
///     dispatch / workgroup it belongs to, and its register-file sizes).
///   * @ref rj_dbg_stop_reason_t mirrors `amd_dbgapi_wave_stop_reasons_t`
///     (single-step done, breakpoint, etc.).
///   * Register access is expressed in terms of register *classes*
///     (scalar / vector / special), the same partitioning `amd-dbgapi`
///     uses for `amd_dbgapi_register_class_*`.
///   * Suspend / resume / single-step map 1:1 onto the
///     `amd_dbgapi_process_set_progress` / `amd_dbgapi_wave_stop` /
///     `amd_dbgapi_wave_resume` control flow.
///
/// A future `librocjitsu-dbgapi.so` only needs to translate the opaque
/// dbgapi handles into the ids used here and forward the calls; no engine
/// changes are required. Keeping the surface in pure C (stable ABI) is part
/// of that contract.

#ifndef ROCJITSU_VM_RJ_VM_DEBUG_H_
#define ROCJITSU_VM_RJ_VM_DEBUG_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"
#include "rocjitsu/vm/rj_vm.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup vm
/// @{

/// @brief Opaque, ABI-stable wavefront identifier.
///
/// @details Encodes the physical location of a wavefront slot
/// (XCC / shader-engine / compute-unit / slot). Treat it as opaque; use
/// @ref rj_vm_debug_wave_info to decode the individual coordinates. The
/// value @ref RJ_DBG_WAVE_NONE never names a wave.
typedef uint64_t rj_dbg_wave_id_t;

/// @brief Sentinel wave id that never names a valid wavefront.
#define RJ_DBG_WAVE_NONE ((rj_dbg_wave_id_t)0xFFFFFFFFFFFFFFFFULL)

/// @brief Wavefront execution state (mirrors @c rocjitsu::amdgpu::WfState).
typedef enum rj_dbg_wave_state_e {
  RJ_DBG_WAVE_HALTED = 0,  ///< Slot is idle / available for dispatch.
  RJ_DBG_WAVE_RUNNING = 1, ///< Eligible for scheduling.
  RJ_DBG_WAVE_WAITCNT = 2, ///< Stalled on an s_waitcnt.
  RJ_DBG_WAVE_BARRIER = 3, ///< Stalled at a workgroup barrier.
  RJ_DBG_WAVE_ENDING = 4,  ///< s_endpgm seen; memory ops draining.
} rj_dbg_wave_state_t;

/// @brief Why the engine last stopped (mirrors amd_dbgapi stop reasons).
typedef enum rj_dbg_stop_reason_e {
  RJ_DBG_STOP_NONE = 0,       ///< Engine is running (not stopped).
  RJ_DBG_STOP_USER = 1,       ///< Stopped by an explicit suspend request.
  RJ_DBG_STOP_BREAKPOINT = 2, ///< A wavefront reached a PC breakpoint.
  RJ_DBG_STOP_STEP = 3,       ///< A requested single-step completed.
  RJ_DBG_STOP_ENTRY = 4,      ///< Stopped before the first tick (attach-on-entry).
  RJ_DBG_STOP_EXITED = 5,     ///< Simulation finished; no more work.
} rj_dbg_stop_reason_t;

/// @brief Register class selector (mirrors amd-dbgapi register classes).
typedef enum rj_dbg_reg_class_e {
  RJ_DBG_REG_SCALAR = 0,  ///< SGPRs (32-bit, wavefront-uniform).
  RJ_DBG_REG_VECTOR = 1,  ///< VGPRs (32-bit per lane).
  RJ_DBG_REG_SPECIAL = 2, ///< PC / EXEC / VCC / M0 / STATUS / MODE (see below).
} rj_dbg_reg_class_t;

/// @brief Indices into the @ref RJ_DBG_REG_SPECIAL register class.
typedef enum rj_dbg_special_reg_e {
  RJ_DBG_SPECIAL_PC = 0,     ///< 64-bit program counter.
  RJ_DBG_SPECIAL_EXEC = 1,   ///< 64-bit execute mask.
  RJ_DBG_SPECIAL_VCC = 2,    ///< 64-bit vector condition code.
  RJ_DBG_SPECIAL_STATUS = 3, ///< 32-bit STATUS register.
  RJ_DBG_SPECIAL_MODE = 4,   ///< 32-bit MODE register.
  RJ_DBG_SPECIAL_M0 = 5,     ///< 32-bit M0 register.
  RJ_DBG_SPECIAL_COUNT = 6,
} rj_dbg_special_reg_t;

/// @brief Snapshot of a single wavefront's architected control state.
typedef struct rj_dbg_wave_info_t {
  rj_dbg_wave_id_t id; ///< Opaque wave id (see @ref rj_dbg_wave_id_t).
  uint32_t xcc;        ///< XCC / XCD index.
  uint32_t se;         ///< Shader-engine index within the XCD.
  uint32_t cu;         ///< Compute-unit index within the shader engine.
  uint32_t slot;       ///< Wavefront slot index within the compute unit.

  uint32_t state;       ///< @ref rj_dbg_wave_state_t.
  uint64_t pc;          ///< Program counter (byte address).
  uint64_t exec;        ///< EXEC mask.
  uint64_t vcc;         ///< VCC value.
  uint32_t status;      ///< STATUS register.
  uint32_t mode;        ///< MODE register.
  uint32_t m0;          ///< M0 register.
  uint32_t wave_size;   ///< Lanes per wavefront (32 or 64).
  uint32_t num_sgprs;   ///< Allocated scalar registers.
  uint32_t num_vgprs;   ///< Allocated vector registers.
  uint32_t wg_id;       ///< Owning workgroup id.
  uint32_t dispatch_id; ///< Owning dispatch id.
  uint32_t process_id;  ///< Owning KFD process id (PASID analog).
} rj_dbg_wave_info_t;

/// @brief Report whether debug control is available for this VM.
///
/// @details Debug control requires a single-threaded, serving-mode VM
/// (created with @c RJ_VM_MODE_LOCAL or @c RJ_VM_MODE_DAEMON). For any other
/// configuration the other entry points return
/// @c ROCJITSU_STATUS_OUT_OF_RESOURCES.
/// @param[in] vm VM handle.
/// @param[out] supported Set to non-zero when debug control is available.
RJ_API_EXPORT rj_status_t rj_vm_debug_supported(rj_vm_t *vm, int *supported);

/// @brief Suspend the simulation engine and block until it is quiescent.
///
/// @details On return the engine is parked between ticks and it is safe to
/// read or write wavefront state. Idempotent. The recorded stop reason is
/// @ref RJ_DBG_STOP_USER unless a breakpoint / step / exit already stopped
/// the engine.
/// @param[in] vm VM handle.
RJ_API_EXPORT rj_status_t rj_vm_debug_suspend(rj_vm_t *vm);

/// @brief Resume free-running execution.
/// @param[in] vm VM handle.
RJ_API_EXPORT rj_status_t rj_vm_debug_resume(rj_vm_t *vm);

/// @brief Single-step the engine by @p ticks simulation ticks, then suspend.
///
/// @details Blocks until the steps complete (or the simulation exits). A
/// breakpoint encountered mid-step stops early and is reported via the stop
/// reason. @p ticks of 0 is treated as 1.
/// @param[in] vm VM handle.
/// @param[in] ticks Number of simulation ticks to advance.
RJ_API_EXPORT rj_status_t rj_vm_debug_step(rj_vm_t *vm, uint64_t ticks);

/// @brief Query the current run/stop status.
/// @param[in] vm VM handle.
/// @param[out] stopped Non-zero when the engine is suspended (may be NULL).
/// @param[out] stop_reason @ref rj_dbg_stop_reason_t (may be NULL).
/// @param[out] tick Last executed simulation tick (may be NULL).
RJ_API_EXPORT rj_status_t rj_vm_debug_status(rj_vm_t *vm, int *stopped, uint32_t *stop_reason,
                                             uint64_t *tick);

/// @brief Block until the engine stops, or @p timeout_ms elapses.
///
/// @details Used to wait for a breakpoint / step / exit after a resume.
/// A @p timeout_ms of 0 polls without blocking.
/// @param[in] vm VM handle.
/// @param[in] timeout_ms Maximum time to wait, in milliseconds.
/// @param[out] stopped Non-zero when the engine is suspended (may be NULL).
/// @param[out] stop_reason @ref rj_dbg_stop_reason_t (may be NULL).
RJ_API_EXPORT rj_status_t rj_vm_debug_wait_stop(rj_vm_t *vm, uint64_t timeout_ms, int *stopped,
                                                uint32_t *stop_reason);

/// @brief Count the wavefronts currently dispatched (state != HALTED).
/// @param[in] vm VM handle.
/// @param[out] count Number of live wavefronts.
RJ_API_EXPORT rj_status_t rj_vm_debug_wave_count(rj_vm_t *vm, uint32_t *count);

/// @brief Enumerate live wavefronts.
///
/// @details The engine must be suspended. Writes up to @p capacity entries
/// into @p out and sets @p count to the total number of live wavefronts
/// (which may exceed @p capacity — call again with a larger buffer).
/// @param[in] vm VM handle.
/// @param[out] out Destination array (may be NULL when @p capacity is 0).
/// @param[in] capacity Number of entries @p out can hold.
/// @param[out] count Total number of live wavefronts.
RJ_API_EXPORT rj_status_t rj_vm_debug_wave_list(rj_vm_t *vm, rj_dbg_wave_info_t *out,
                                                uint32_t capacity, uint32_t *count);

/// @brief Fetch one wavefront's control state by id.
/// @param[in] vm VM handle.
/// @param[in] wave Wave id.
/// @param[out] out Destination snapshot.
RJ_API_EXPORT rj_status_t rj_vm_debug_wave_info(rj_vm_t *vm, rj_dbg_wave_id_t wave,
                                                rj_dbg_wave_info_t *out);

/// @brief Read scalar registers @p first .. @p first+@p count from a wave.
///
/// @details The engine must be suspended. Reads are clamped to the wave's
/// allocated SGPR count.
/// @param[in] vm VM handle.
/// @param[in] wave Wave id.
/// @param[in] first First (architected) SGPR index.
/// @param[in] count Number of consecutive SGPRs to read.
/// @param[out] out Destination buffer of @p count uint32_t values.
RJ_API_EXPORT rj_status_t rj_vm_debug_read_sgpr(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t first,
                                                uint32_t count, uint32_t *out);

/// @brief Write a single scalar register.
/// @param[in] vm VM handle.
/// @param[in] wave Wave id.
/// @param[in] index Architected SGPR index.
/// @param[in] value New value.
RJ_API_EXPORT rj_status_t rj_vm_debug_write_sgpr(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t index,
                                                 uint32_t value);

/// @brief Read one vector register across a range of lanes.
///
/// @details The engine must be suspended. Reads are clamped to the wave's
/// allocated VGPR count and lane width.
/// @param[in] vm VM handle.
/// @param[in] wave Wave id.
/// @param[in] reg Architected VGPR index.
/// @param[in] first_lane First lane to read.
/// @param[in] lane_count Number of consecutive lanes to read.
/// @param[out] out Destination buffer of @p lane_count uint32_t values.
RJ_API_EXPORT rj_status_t rj_vm_debug_read_vgpr(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t reg,
                                                uint32_t first_lane, uint32_t lane_count,
                                                uint32_t *out);

/// @brief Write one lane of a vector register.
/// @param[in] vm VM handle.
/// @param[in] wave Wave id.
/// @param[in] reg Architected VGPR index.
/// @param[in] lane Lane index.
/// @param[in] value New value.
RJ_API_EXPORT rj_status_t rj_vm_debug_write_vgpr(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t reg,
                                                 uint32_t lane, uint32_t value);

/// @brief Read a special register (see @ref rj_dbg_special_reg_t).
/// @param[in] vm VM handle.
/// @param[in] wave Wave id.
/// @param[in] which Which special register.
/// @param[out] value 64-bit value (32-bit registers are zero-extended).
RJ_API_EXPORT rj_status_t rj_vm_debug_read_special(rj_vm_t *vm, rj_dbg_wave_id_t wave,
                                                   uint32_t which, uint64_t *value);

/// @brief Write a special register (see @ref rj_dbg_special_reg_t).
/// @param[in] vm VM handle.
/// @param[in] wave Wave id.
/// @param[in] which Which special register.
/// @param[in] value New value (32-bit registers take the low 32 bits).
RJ_API_EXPORT rj_status_t rj_vm_debug_write_special(rj_vm_t *vm, rj_dbg_wave_id_t wave,
                                                    uint32_t which, uint64_t value);

/// @brief Read global device memory.
///
/// @details The engine must be suspended. @p vmid selects the per-process
/// page table (0 = physical / passthrough).
/// @param[in] vm VM handle.
/// @param[in] vmid Address space / process VMID (0 for physical).
/// @param[in] addr Device virtual address.
/// @param[out] out Destination buffer.
/// @param[in] size Number of bytes to read.
RJ_API_EXPORT rj_status_t rj_vm_debug_read_memory(rj_vm_t *vm, uint32_t vmid, uint64_t addr,
                                                  void *out, uint64_t size);

/// @brief Write global device memory.
/// @param[in] vm VM handle.
/// @param[in] vmid Address space / process VMID (0 for physical).
/// @param[in] addr Device virtual address.
/// @param[in] in Source buffer.
/// @param[in] size Number of bytes to write.
RJ_API_EXPORT rj_status_t rj_vm_debug_write_memory(rj_vm_t *vm, uint32_t vmid, uint64_t addr,
                                                   const void *in, uint64_t size);

/// @brief Set a program-counter breakpoint.
///
/// @details When any wavefront's PC reaches @p pc the engine suspends with
/// stop reason @ref RJ_DBG_STOP_BREAKPOINT. Setting the same address twice
/// returns the existing id.
/// @param[in] vm VM handle.
/// @param[in] pc Byte address to break on.
/// @param[out] id Breakpoint id (may be NULL).
RJ_API_EXPORT rj_status_t rj_vm_debug_break_set(rj_vm_t *vm, uint64_t pc, uint32_t *id);

/// @brief Clear a breakpoint by id.
/// @param[in] vm VM handle.
/// @param[in] id Breakpoint id from @ref rj_vm_debug_break_set.
RJ_API_EXPORT rj_status_t rj_vm_debug_break_clear(rj_vm_t *vm, uint32_t id);

/// @brief List active breakpoint addresses.
/// @param[in] vm VM handle.
/// @param[out] pcs Destination array of addresses (may be NULL when @p capacity is 0).
/// @param[in] capacity Number of entries @p pcs can hold.
/// @param[out] count Total number of breakpoints.
RJ_API_EXPORT rj_status_t rj_vm_debug_break_list(rj_vm_t *vm, uint64_t *pcs, uint32_t capacity,
                                                 uint32_t *count);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_VM_RJ_VM_DEBUG_H_
