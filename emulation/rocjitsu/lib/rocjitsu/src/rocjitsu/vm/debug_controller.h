// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file debug_controller.h
/// @brief Engine-driving debug controller backing the @ref rj_vm_debug.h API.

#ifndef ROCJITSU_VM_DEBUG_CONTROLLER_H_
#define ROCJITSU_VM_DEBUG_CONTROLLER_H_

#include "rocjitsu/vm/rj_vm_debug.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace simdojo {
class SimulationEngine;
}

namespace rocjitsu {

class SoC;
namespace amdgpu {
class Wavefront;
}

/// @brief Serialises debugger control of a serving-mode rocjitsu engine.
///
/// @details In serving mode (@c RJ_VM_MODE_LOCAL / @c RJ_VM_MODE_DAEMON) the
/// engine is driven a tick at a time by @ref run_engine_loop instead of the
/// free-running @c SimulationEngine::run(). The controller owns a single
/// mutex/condvar pair that establishes the core invariant the debug API
/// relies on:
///
///   > While the lock is held and @ref parked_ is set, the engine thread is
///   > blocked in the loop's wait and is therefore *not* mutating any
///   > wavefront, register file, or memory page.
///
/// All inspection entry points acquire the lock and wait for that quiescent
/// state before touching VM state, so reads and writes are race-free without
/// any per-component locking. This is the same "stop the world, then peek"
/// model `amd-dbgapi` exposes to `rocgdb`.
class DebugController {
public:
  /// @brief Bind the controller to the engine and primary SoC after build().
  void bind(simdojo::SimulationEngine *engine, SoC *soc) {
    engine_ = engine;
    soc_ = soc;
  }

  /// @brief True once @ref bind has wired a usable engine + SoC.
  bool bound() const { return engine_ != nullptr && soc_ != nullptr; }

  /// @brief Drive the engine until @ref request_exit is called.
  ///
  /// @details Replaces @c SimulationEngine::run() on the engine thread. Starts
  /// free-running; a debugger must call @ref suspend to stop it.
  void run_engine_loop();

  /// @brief Ask the engine loop to return (thread-safe; wakes a parked engine).
  void request_exit();

  // --- Control plane (callable from any client thread) ---------------------

  void suspend();
  void resume();
  void step(uint64_t ticks);
  void status(bool *stopped, rj_dbg_stop_reason_t *reason, uint64_t *tick);
  bool wait_stop(uint64_t timeout_ms, bool *stopped, rj_dbg_stop_reason_t *reason);

  uint32_t break_set(uint64_t pc);
  bool break_clear(uint32_t id);
  std::vector<uint64_t> break_list();

  // --- Inspection plane (require the engine to be suspended) ---------------

  uint32_t wave_count();
  bool wave_list(std::vector<rj_dbg_wave_info_t> &out);
  bool wave_info(rj_dbg_wave_id_t id, rj_dbg_wave_info_t *out);
  bool read_sgpr(rj_dbg_wave_id_t id, uint32_t first, uint32_t count, uint32_t *out);
  bool write_sgpr(rj_dbg_wave_id_t id, uint32_t index, uint32_t value);
  bool read_vgpr(rj_dbg_wave_id_t id, uint32_t reg, uint32_t first_lane, uint32_t lane_count,
                 uint32_t *out);
  bool write_vgpr(rj_dbg_wave_id_t id, uint32_t reg, uint32_t lane, uint32_t value);
  bool read_special(rj_dbg_wave_id_t id, uint32_t which, uint64_t *value);
  bool write_special(rj_dbg_wave_id_t id, uint32_t which, uint64_t value);
  bool read_memory(uint32_t vmid, uint64_t addr, void *out, uint64_t size);
  bool write_memory(uint32_t vmid, uint64_t addr, const void *in, uint64_t size);

  /// @brief Encode physical wave coordinates into an opaque wave id.
  static rj_dbg_wave_id_t encode_id(uint32_t xcc, uint32_t se, uint32_t cu, uint32_t slot) {
    return (static_cast<uint64_t>(xcc) << 48) | (static_cast<uint64_t>(se) << 32) |
           (static_cast<uint64_t>(cu) << 16) | static_cast<uint64_t>(slot);
  }

private:
  /// @brief Whether the engine is currently stopped (locked predicate).
  bool stopped_locked() const { return (pause_requested_ || exited_) && parked_; }

  /// @brief Block until the engine is quiescent; returns false if it is
  /// free-running (caller must @ref suspend first) or shutting down.
  bool wait_until_parked(std::unique_lock<std::mutex> &lk);

  /// @brief Locate a wave by id (no locking; caller holds the lock + parked).
  amdgpu::Wavefront *find_wave(rj_dbg_wave_id_t id);

  /// @brief Populate a snapshot from a wave (no locking).
  void fill_info(amdgpu::Wavefront *wf, uint32_t xcc, uint32_t se, uint32_t cu, uint32_t slot,
                 rj_dbg_wave_info_t *out);

  /// @brief True if any live wave's PC matches a breakpoint (no locking).
  bool any_wave_at_breakpoint();

  /// @brief Push the FUNCTIONAL-mode instruction quantum to every CU based on
  /// the current debug state (no locking; engine must be parked).
  ///
  /// @details While a breakpoint is armed or a step is in flight, each CU
  /// retires a single instruction per engine tick so wavefronts stay visible
  /// and PC breakpoints can stop a wave mid-kernel. Otherwise the CUs run at
  /// full speed (the default quantum) so an un-instrumented @c continue is
  /// not slowed down.
  void apply_exec_granularity_locked();

  simdojo::SimulationEngine *engine_ = nullptr;
  SoC *soc_ = nullptr;

  std::mutex mu_;
  std::condition_variable cv_;
  bool pause_requested_ = false;
  bool parked_ = false;
  bool exit_requested_ = false;
  bool exited_ = false;
  uint64_t step_budget_ = 0;
  uint64_t last_tick_ = 0;
  rj_dbg_stop_reason_t stop_reason_ = RJ_DBG_STOP_NONE;
  std::vector<uint64_t> breakpoints_; ///< Index = id; 0 means a cleared slot.
};

} // namespace rocjitsu

#endif // ROCJITSU_VM_DEBUG_CONTROLLER_H_
