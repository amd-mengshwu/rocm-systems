// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm_impl.h
/// @brief Private definition of rj_vm_t. Internal to the library.

#ifndef ROCJITSU_VM_RJ_VM_IMPL_H_
#define ROCJITSU_VM_RJ_VM_IMPL_H_

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/refcount.h"
#include "rocjitsu/vm/debug_controller.h"
#include "rocjitsu/vm/rj_vm.h"
#include "rocjitsu/vm/virtual_machine.h"

#include "simdojo/sim/simulation.h"

#include <memory>

struct rj_vm_t : rocjitsu::RefCounted {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  simdojo::SimulationEngine::Config engine_config{};
  rocjitsu::config::LoadedConfig loaded;
  rocjitsu::SoC *soc = nullptr;
  rocjitsu::VirtualMachine *vm = nullptr;
  /// VM creation mode; selects free-running vs. debug-driven execution.
  rj_vm_mode_t mode = RJ_VM_MODE_DEFAULT;
  /// Debug control surface. Present (and bound) only for single-threaded
  /// serving-mode VMs; see @ref rj_vm_debug.h. Null otherwise.
  std::unique_ptr<rocjitsu::DebugController> debug;
};

#endif // ROCJITSU_VM_RJ_VM_IMPL_H_
