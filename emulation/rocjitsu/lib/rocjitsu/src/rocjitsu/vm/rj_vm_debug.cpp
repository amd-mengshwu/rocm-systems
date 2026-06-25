// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/rj_vm_debug.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/shader_engine.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/debug_controller.h"
#include "rocjitsu/vm/rj_vm_impl.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"

#include <chrono>
#include <cstring>
#include <thread>

using namespace rocjitsu;
using rocjitsu::amdgpu::Wavefront;

// ===========================================================================
// DebugController: engine loop + control plane
// ===========================================================================

void DebugController::run_engine_loop() {
  uint64_t prev_tick = engine_->global_time();
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(mu_);
      parked_ = true;
      cv_.notify_all();
      cv_.wait(lk, [&] {
        return exit_requested_ || (!exited_ && (!pause_requested_ || step_budget_ > 0));
      });
      parked_ = false;
      if (exit_requested_)
        return;
    }

    // Advance one simulation tick with no lock held: this is the only place
    // the engine mutates VM state, and it never runs while parked_ is set.
    bool more = engine_->step();
    uint64_t tick = engine_->global_time();
    bool idle = (tick == prev_tick);
    prev_tick = tick;

    {
      std::unique_lock<std::mutex> lk(mu_);
      last_tick_ = tick;
      if (step_budget_ > 0 && --step_budget_ == 0) {
        pause_requested_ = true;
        if (stop_reason_ != RJ_DBG_STOP_BREAKPOINT)
          stop_reason_ = RJ_DBG_STOP_STEP;
      }
      if (!more) {
        exited_ = true;
        pause_requested_ = true;
        step_budget_ = 0;
        stop_reason_ = RJ_DBG_STOP_EXITED;
      } else if (!breakpoints_.empty() && any_wave_at_breakpoint()) {
        pause_requested_ = true;
        step_budget_ = 0;
        stop_reason_ = RJ_DBG_STOP_BREAKPOINT;
      }
      if (pause_requested_)
        cv_.notify_all();
    }

    // When serving and idle (engine alive only to await async doorbells),
    // avoid pinning a core. Latency-insensitive: the daemon is being debugged.
    if (more && idle && step_budget_ == 0)
      std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

void DebugController::request_exit() {
  std::lock_guard<std::mutex> lk(mu_);
  exit_requested_ = true;
  cv_.notify_all();
}

void DebugController::suspend() {
  std::unique_lock<std::mutex> lk(mu_);
  if (!exited_) {
    pause_requested_ = true;
    if (stop_reason_ == RJ_DBG_STOP_NONE)
      stop_reason_ = RJ_DBG_STOP_USER;
    cv_.notify_all();
  }
  cv_.wait(lk, [&] { return exit_requested_ || parked_; });
}

void DebugController::resume() {
  std::lock_guard<std::mutex> lk(mu_);
  if (exited_)
    return;
  pause_requested_ = false;
  step_budget_ = 0;
  stop_reason_ = RJ_DBG_STOP_NONE;
  cv_.notify_all();
}

void DebugController::step(uint64_t ticks) {
  std::unique_lock<std::mutex> lk(mu_);
  if (exited_)
    return;
  if (ticks == 0)
    ticks = 1;
  step_budget_ = ticks;
  pause_requested_ = true;
  stop_reason_ = RJ_DBG_STOP_STEP;
  cv_.notify_all();
  cv_.wait(lk, [&] { return exit_requested_ || exited_ || (parked_ && step_budget_ == 0); });
}

void DebugController::status(bool *stopped, rj_dbg_stop_reason_t *reason, uint64_t *tick) {
  std::lock_guard<std::mutex> lk(mu_);
  if (stopped)
    *stopped = stopped_locked();
  if (reason)
    *reason = stop_reason_;
  if (tick)
    *tick = last_tick_;
}

bool DebugController::wait_stop(uint64_t timeout_ms, bool *stopped, rj_dbg_stop_reason_t *reason) {
  std::unique_lock<std::mutex> lk(mu_);
  bool got = cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                          [&] { return exit_requested_ || stopped_locked(); });
  if (stopped)
    *stopped = stopped_locked();
  if (reason)
    *reason = stop_reason_;
  return got;
}

uint32_t DebugController::break_set(uint64_t pc) {
  std::lock_guard<std::mutex> lk(mu_);
  for (size_t i = 0; i < breakpoints_.size(); ++i)
    if (breakpoints_[i] == pc)
      return static_cast<uint32_t>(i + 1);
  for (size_t i = 0; i < breakpoints_.size(); ++i)
    if (breakpoints_[i] == 0) {
      breakpoints_[i] = pc;
      return static_cast<uint32_t>(i + 1);
    }
  breakpoints_.push_back(pc);
  return static_cast<uint32_t>(breakpoints_.size());
}

bool DebugController::break_clear(uint32_t id) {
  std::lock_guard<std::mutex> lk(mu_);
  if (id == 0 || id > breakpoints_.size() || breakpoints_[id - 1] == 0)
    return false;
  breakpoints_[id - 1] = 0;
  return true;
}

std::vector<uint64_t> DebugController::break_list() {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<uint64_t> out;
  for (uint64_t pc : breakpoints_)
    if (pc != 0)
      out.push_back(pc);
  return out;
}

bool DebugController::wait_until_parked(std::unique_lock<std::mutex> &lk) {
  if (!pause_requested_ && !exited_)
    return false; // engine is free-running; caller must suspend() first.
  cv_.wait(lk, [&] { return exit_requested_ || parked_; });
  return parked_ && !exit_requested_;
}

// ===========================================================================
// DebugController: wave walking helpers (lock held, engine parked)
// ===========================================================================

Wavefront *DebugController::find_wave(rj_dbg_wave_id_t id) {
  uint32_t xcc = static_cast<uint32_t>((id >> 48) & 0xFFFF);
  uint32_t se = static_cast<uint32_t>((id >> 32) & 0xFFFF);
  uint32_t cu = static_cast<uint32_t>((id >> 16) & 0xFFFF);
  uint32_t slot = static_cast<uint32_t>(id & 0xFFFF);
  if (xcc >= soc_->num_xcds())
    return nullptr;
  auto *xcd = soc_->xcd(xcc);
  if (se >= xcd->num_shader_engines())
    return nullptr;
  auto *se_ptr = xcd->shader_engine(se);
  if (cu >= se_ptr->num_compute_units())
    return nullptr;
  auto *cu_ptr = se_ptr->compute_unit(cu);
  if (slot >= cu_ptr->num_wf_slots())
    return nullptr;
  return cu_ptr->wf(slot);
}

void DebugController::fill_info(Wavefront *wf, uint32_t xcc, uint32_t se, uint32_t cu, uint32_t slot,
                                rj_dbg_wave_info_t *out) {
  std::memset(out, 0, sizeof(*out));
  out->id = encode_id(xcc, se, cu, slot);
  out->xcc = xcc;
  out->se = se;
  out->cu = cu;
  out->slot = slot;
  out->state = static_cast<uint32_t>(wf->state());
  out->pc = wf->pc;
  out->exec = wf->exec();
  out->vcc = wf->vcc();
  out->status = wf->status_raw();
  out->mode = wf->mode_raw();
  out->m0 = wf->m0();
  out->wave_size = wf->wf_size();
  out->num_sgprs = wf->num_sgprs();
  out->num_vgprs = wf->num_vgprs();
  out->wg_id = wf->wg_id();
  out->dispatch_id = wf->dispatch_id();
  out->process_id = wf->process_id();
}

bool DebugController::any_wave_at_breakpoint() {
  for (uint32_t x = 0; x < soc_->num_xcds(); ++x) {
    auto *xcd = soc_->xcd(x);
    for (uint32_t s = 0; s < xcd->num_shader_engines(); ++s) {
      auto *se = xcd->shader_engine(s);
      for (uint32_t c = 0; c < se->num_compute_units(); ++c) {
        auto *cu = se->compute_unit(c);
        for (uint32_t w = 0; w < cu->num_wf_slots(); ++w) {
          auto *wf = cu->wf(w);
          if (wf->state() == amdgpu::WfState::HALTED)
            continue;
          for (uint64_t bp : breakpoints_)
            if (bp != 0 && wf->pc == bp)
              return true;
        }
      }
    }
  }
  return false;
}

// ===========================================================================
// DebugController: inspection plane
// ===========================================================================

uint32_t DebugController::wave_count() {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return 0;
  uint32_t n = 0;
  for (uint32_t x = 0; x < soc_->num_xcds(); ++x) {
    auto *xcd = soc_->xcd(x);
    for (uint32_t s = 0; s < xcd->num_shader_engines(); ++s) {
      auto *se = xcd->shader_engine(s);
      for (uint32_t c = 0; c < se->num_compute_units(); ++c) {
        auto *cu = se->compute_unit(c);
        for (uint32_t w = 0; w < cu->num_wf_slots(); ++w)
          if (cu->wf(w)->state() != amdgpu::WfState::HALTED)
            ++n;
      }
    }
  }
  return n;
}

bool DebugController::wave_list(std::vector<rj_dbg_wave_info_t> &out) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  out.clear();
  for (uint32_t x = 0; x < soc_->num_xcds(); ++x) {
    auto *xcd = soc_->xcd(x);
    for (uint32_t s = 0; s < xcd->num_shader_engines(); ++s) {
      auto *se = xcd->shader_engine(s);
      for (uint32_t c = 0; c < se->num_compute_units(); ++c) {
        auto *cu = se->compute_unit(c);
        for (uint32_t w = 0; w < cu->num_wf_slots(); ++w) {
          auto *wf = cu->wf(w);
          if (wf->state() == amdgpu::WfState::HALTED)
            continue;
          rj_dbg_wave_info_t info;
          fill_info(wf, x, s, c, w, &info);
          out.push_back(info);
        }
      }
    }
  }
  return true;
}

bool DebugController::wave_info(rj_dbg_wave_id_t id, rj_dbg_wave_info_t *out) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *wf = find_wave(id);
  if (!wf)
    return false;
  fill_info(wf, static_cast<uint32_t>((id >> 48) & 0xFFFF),
            static_cast<uint32_t>((id >> 32) & 0xFFFF), static_cast<uint32_t>((id >> 16) & 0xFFFF),
            static_cast<uint32_t>(id & 0xFFFF), out);
  return true;
}

bool DebugController::read_sgpr(rj_dbg_wave_id_t id, uint32_t first, uint32_t count, uint32_t *out) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *wf = find_wave(id);
  if (!wf)
    return false;
  uint32_t avail = wf->num_sgprs();
  uint32_t base = wf->sgpr_alloc().base;
  for (uint32_t i = 0; i < count; ++i)
    out[i] = (first + i < avail) ? wf->cu().read_sgpr(base + first + i) : 0;
  return true;
}

bool DebugController::write_sgpr(rj_dbg_wave_id_t id, uint32_t index, uint32_t value) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *wf = find_wave(id);
  if (!wf || index >= wf->num_sgprs())
    return false;
  wf->cu().write_sgpr(wf->sgpr_alloc().base + index, value);
  return true;
}

bool DebugController::read_vgpr(rj_dbg_wave_id_t id, uint32_t reg, uint32_t first_lane,
                                uint32_t lane_count, uint32_t *out) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *wf = find_wave(id);
  if (!wf || reg >= wf->num_vgprs())
    return false;
  uint32_t phys = wf->vgpr_alloc().base + reg;
  uint32_t lanes = wf->wf_size();
  for (uint32_t i = 0; i < lane_count; ++i) {
    uint32_t lane = first_lane + i;
    out[i] = (lane < lanes) ? wf->cu().read_vgpr(phys, lane) : 0;
  }
  return true;
}

bool DebugController::write_vgpr(rj_dbg_wave_id_t id, uint32_t reg, uint32_t lane, uint32_t value) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *wf = find_wave(id);
  if (!wf || reg >= wf->num_vgprs() || lane >= wf->wf_size())
    return false;
  wf->cu().write_vgpr(wf->vgpr_alloc().base + reg, lane, value);
  return true;
}

bool DebugController::read_special(rj_dbg_wave_id_t id, uint32_t which, uint64_t *value) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *wf = find_wave(id);
  if (!wf)
    return false;
  switch (which) {
  case RJ_DBG_SPECIAL_PC: *value = wf->pc; return true;
  case RJ_DBG_SPECIAL_EXEC: *value = wf->exec(); return true;
  case RJ_DBG_SPECIAL_VCC: *value = wf->vcc(); return true;
  case RJ_DBG_SPECIAL_STATUS: *value = wf->status_raw(); return true;
  case RJ_DBG_SPECIAL_MODE: *value = wf->mode_raw(); return true;
  case RJ_DBG_SPECIAL_M0: *value = wf->m0(); return true;
  default: return false;
  }
}

bool DebugController::write_special(rj_dbg_wave_id_t id, uint32_t which, uint64_t value) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *wf = find_wave(id);
  if (!wf)
    return false;
  switch (which) {
  case RJ_DBG_SPECIAL_PC: wf->pc = value; return true;
  case RJ_DBG_SPECIAL_EXEC: wf->set_exec(value); return true;
  case RJ_DBG_SPECIAL_VCC: wf->set_vcc(value); return true;
  case RJ_DBG_SPECIAL_STATUS: wf->set_status_raw(static_cast<uint32_t>(value)); return true;
  case RJ_DBG_SPECIAL_MODE: wf->set_mode_raw(static_cast<uint32_t>(value)); return true;
  case RJ_DBG_SPECIAL_M0: wf->set_m0(static_cast<uint32_t>(value)); return true;
  default: return false;
  }
}

bool DebugController::read_memory(uint32_t vmid, uint64_t addr, void *out, uint64_t size) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *mem = soc_->memory();
  if (!mem)
    return false;
  auto *dst = static_cast<uint8_t *>(out);
  for (uint64_t i = 0; i < size; ++i)
    dst[i] = mem->read8(addr + i, vmid);
  return true;
}

bool DebugController::write_memory(uint32_t vmid, uint64_t addr, const void *in, uint64_t size) {
  std::unique_lock<std::mutex> lk(mu_);
  if (!wait_until_parked(lk))
    return false;
  auto *mem = soc_->memory();
  if (!mem)
    return false;
  const auto *src = static_cast<const uint8_t *>(in);
  for (uint64_t i = 0; i < size; ++i)
    mem->write8(addr + i, src[i], vmid);
  return true;
}

// ===========================================================================
// C API
// ===========================================================================

namespace {
DebugController *dbg(rj_vm_t *vm) {
  return (vm && vm->debug && vm->debug->bound()) ? vm->debug.get() : nullptr;
}
} // namespace

rj_status_t rj_vm_debug_supported(rj_vm_t *vm, int *supported) {
  if (!supported)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  *supported = dbg(vm) ? 1 : 0;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_suspend(rj_vm_t *vm) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  d->suspend();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_resume(rj_vm_t *vm) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  d->resume();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_step(rj_vm_t *vm, uint64_t ticks) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  d->step(ticks);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_status(rj_vm_t *vm, int *stopped, uint32_t *stop_reason, uint64_t *tick) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  bool s = false;
  rj_dbg_stop_reason_t r = RJ_DBG_STOP_NONE;
  uint64_t t = 0;
  d->status(&s, &r, &t);
  if (stopped)
    *stopped = s ? 1 : 0;
  if (stop_reason)
    *stop_reason = static_cast<uint32_t>(r);
  if (tick)
    *tick = t;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_wait_stop(rj_vm_t *vm, uint64_t timeout_ms, int *stopped,
                                  uint32_t *stop_reason) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  bool s = false;
  rj_dbg_stop_reason_t r = RJ_DBG_STOP_NONE;
  d->wait_stop(timeout_ms, &s, &r);
  if (stopped)
    *stopped = s ? 1 : 0;
  if (stop_reason)
    *stop_reason = static_cast<uint32_t>(r);
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_wave_count(rj_vm_t *vm, uint32_t *count) {
  auto *d = dbg(vm);
  if (!d || !count)
    return d ? ROCJITSU_STATUS_INVALID_ARGUMENT : ROCJITSU_STATUS_OUT_OF_RESOURCES;
  *count = d->wave_count();
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_wave_list(rj_vm_t *vm, rj_dbg_wave_info_t *out, uint32_t capacity,
                                  uint32_t *count) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  if (!count)
    return ROCJITSU_STATUS_INVALID_ARGUMENT;
  std::vector<rj_dbg_wave_info_t> waves;
  if (!d->wave_list(waves))
    return ROCJITSU_STATUS_ERROR;
  *count = static_cast<uint32_t>(waves.size());
  for (uint32_t i = 0; i < waves.size() && i < capacity; ++i)
    out[i] = waves[i];
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_wave_info(rj_vm_t *vm, rj_dbg_wave_id_t wave, rj_dbg_wave_info_t *out) {
  auto *d = dbg(vm);
  if (!d || !out)
    return d ? ROCJITSU_STATUS_INVALID_ARGUMENT : ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->wave_info(wave, out) ? ROCJITSU_STATUS_SUCCESS : ROCJITSU_STATUS_INVALID_ARGUMENT;
}

rj_status_t rj_vm_debug_read_sgpr(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t first, uint32_t count,
                                  uint32_t *out) {
  auto *d = dbg(vm);
  if (!d || !out)
    return d ? ROCJITSU_STATUS_INVALID_ARGUMENT : ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->read_sgpr(wave, first, count, out) ? ROCJITSU_STATUS_SUCCESS
                                               : ROCJITSU_STATUS_INVALID_ARGUMENT;
}

rj_status_t rj_vm_debug_write_sgpr(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t index,
                                   uint32_t value) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->write_sgpr(wave, index, value) ? ROCJITSU_STATUS_SUCCESS
                                           : ROCJITSU_STATUS_INVALID_ARGUMENT;
}

rj_status_t rj_vm_debug_read_vgpr(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t reg,
                                  uint32_t first_lane, uint32_t lane_count, uint32_t *out) {
  auto *d = dbg(vm);
  if (!d || !out)
    return d ? ROCJITSU_STATUS_INVALID_ARGUMENT : ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->read_vgpr(wave, reg, first_lane, lane_count, out) ? ROCJITSU_STATUS_SUCCESS
                                                              : ROCJITSU_STATUS_INVALID_ARGUMENT;
}

rj_status_t rj_vm_debug_write_vgpr(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t reg, uint32_t lane,
                                   uint32_t value) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->write_vgpr(wave, reg, lane, value) ? ROCJITSU_STATUS_SUCCESS
                                               : ROCJITSU_STATUS_INVALID_ARGUMENT;
}

rj_status_t rj_vm_debug_read_special(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t which,
                                     uint64_t *value) {
  auto *d = dbg(vm);
  if (!d || !value)
    return d ? ROCJITSU_STATUS_INVALID_ARGUMENT : ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->read_special(wave, which, value) ? ROCJITSU_STATUS_SUCCESS
                                             : ROCJITSU_STATUS_INVALID_ARGUMENT;
}

rj_status_t rj_vm_debug_write_special(rj_vm_t *vm, rj_dbg_wave_id_t wave, uint32_t which,
                                      uint64_t value) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->write_special(wave, which, value) ? ROCJITSU_STATUS_SUCCESS
                                              : ROCJITSU_STATUS_INVALID_ARGUMENT;
}

rj_status_t rj_vm_debug_read_memory(rj_vm_t *vm, uint32_t vmid, uint64_t addr, void *out,
                                    uint64_t size) {
  auto *d = dbg(vm);
  if (!d || (!out && size))
    return d ? ROCJITSU_STATUS_INVALID_ARGUMENT : ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->read_memory(vmid, addr, out, size) ? ROCJITSU_STATUS_SUCCESS : ROCJITSU_STATUS_ERROR;
}

rj_status_t rj_vm_debug_write_memory(rj_vm_t *vm, uint32_t vmid, uint64_t addr, const void *in,
                                     uint64_t size) {
  auto *d = dbg(vm);
  if (!d || (!in && size))
    return d ? ROCJITSU_STATUS_INVALID_ARGUMENT : ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->write_memory(vmid, addr, in, size) ? ROCJITSU_STATUS_SUCCESS : ROCJITSU_STATUS_ERROR;
}

rj_status_t rj_vm_debug_break_set(rj_vm_t *vm, uint64_t pc, uint32_t *id) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  uint32_t bid = d->break_set(pc);
  if (id)
    *id = bid;
  return ROCJITSU_STATUS_SUCCESS;
}

rj_status_t rj_vm_debug_break_clear(rj_vm_t *vm, uint32_t id) {
  auto *d = dbg(vm);
  if (!d)
    return ROCJITSU_STATUS_OUT_OF_RESOURCES;
  return d->break_clear(id) ? ROCJITSU_STATUS_SUCCESS : ROCJITSU_STATUS_INVALID_ARGUMENT;
}

rj_status_t rj_vm_debug_break_list(rj_vm_t *vm, uint64_t *pcs, uint32_t capacity, uint32_t *count) {
  auto *d = dbg(vm);
  if (!d || !count)
    return d ? ROCJITSU_STATUS_INVALID_ARGUMENT : ROCJITSU_STATUS_OUT_OF_RESOURCES;
  auto list = d->break_list();
  *count = static_cast<uint32_t>(list.size());
  for (uint32_t i = 0; i < list.size() && i < capacity; ++i)
    pcs[i] = list[i];
  return ROCJITSU_STATUS_SUCCESS;
}
