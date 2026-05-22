// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hip/graph.hpp"

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>  // pulls in <hip/amd_detail/hip_api_trace.hpp>

#include <hip/hip_runtime_api.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace rocprofiler
{
namespace hip
{
namespace graph
{
namespace
{
// Process-global map from hipGraphExec_t handle to a stable monotonic id.
// Reads (one per launch) take the shared lock; writes (one per
// hipGraphInstantiate*/hipGraphExecDestroy) take the exclusive lock.
// Map sizes are small (graphs currently in flight; typically <100).
std::shared_mutex                              g_map_mutex;
std::unordered_map<::hipGraphExec_t, uint64_t> g_exec_to_id;
std::atomic<uint64_t>                          g_next_graph_exec_id{1};  // 0 reserved = "not from a graph"

uint64_t
assign_graph_exec_id(::hipGraphExec_t exec)
{
    if(exec == nullptr) return 0;
    auto id = g_next_graph_exec_id.fetch_add(1, std::memory_order_relaxed);
    std::unique_lock lock{g_map_mutex};
    g_exec_to_id[exec] = id;
    return id;
}

void
forget_graph_exec(::hipGraphExec_t exec)
{
    if(exec == nullptr) return;
    std::unique_lock lock{g_map_mutex};
    g_exec_to_id.erase(exec);
}

// Each instantiation of wrap_instantiate has its own static next_func slot
// because the 3 hipGraphInstantiate* APIs have different signatures (different
// trailing Args...). The lambdas have no captures, so they convert to plain
// function pointers via the unary +.
template <typename RetT, typename... Args>
auto
wrap_instantiate(RetT (*next)(::hipGraphExec_t*, Args...))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t* out, Args... args) -> RetT {
        auto ret = next_func(out, std::forward<Args>(args)...);
        if(ret == hipSuccess && out != nullptr && *out != nullptr)
        {
            assign_graph_exec_id(*out);
        }
        return ret;
    };
}

template <typename RetT>
auto
wrap_destroy(RetT (*next)(::hipGraphExec_t))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t exec) -> RetT {
        // Remove the map entry BEFORE invoking the destroy: HIP destroys the
        // handle even on most error paths, so dropping the mapping first is
        // safer than risking a dangling key.
        forget_graph_exec(exec);
        return next_func(exec);
    };
}

// Per-thread stack of active hipGraphLaunch calls. std::deque (not std::vector)
// because launch_state contains std::atomic<uint64_t> which is non-movable;
// std::deque doesn't move existing elements on growth, so references handed
// out by current_launch_state() remain valid as nested launches push.
thread_local std::deque<launch_state> g_launch_stack;

// Resolve the launch stream's HIP device ordinal to a rocprofiler_agent_id_t.
//
// Implementation notes:
//   * We must use the *saved* (un-wrapped) HIP dispatch table to avoid re-
//     entering rocprofiler's own tracing wrappers from inside hipGraphLaunch
//     -- doing so would emit a spurious HIP_RUNTIME_API record for the helper
//     query and could deadlock or misattribute callbacks.
//   * For a non-null stream we call `hipStreamGetDevice`; for the default
//     stream (nullptr / hipStreamLegacy / hipStreamPerThread) we fall back to
//     the thread's current device via `hipGetDevice`.
//   * GPU agent ordinal = `logical_node_type_id` (this is the type-relative
//     GPU index post-cgroups/ROCR_VISIBLE_DEVICES filtering, which matches
//     HIP's device numbering -- see source/docs/how-to/advanced-rocprofv3-options.rst).
//   * Returns {0} on any failure; callers must tolerate that (the GRAPH_LAUNCH
//     record will then carry agent_id.handle == 0, but this is strictly better
//     than crashing in generateCSV's get_agent_index).
rocprofiler_agent_id_t
resolve_launch_stream_agent(::hipStream_t stream)
{
    auto& saved_table = ::rocprofiler::hip::get_table();
    auto* runtime     = saved_table.runtime;
    if(runtime == nullptr) return rocprofiler_agent_id_t{.handle = 0};

    int device_id = -1;
    // Default-stream sentinels (nullptr, hipStreamLegacy=1, hipStreamPerThread=2)
    // do not have a per-stream device binding in HIP; use the current device.
    auto is_default_stream = (stream == nullptr || stream == hipStreamLegacy ||
                              stream == hipStreamPerThread);

    if(!is_default_stream && runtime->hipStreamGetDevice_fn != nullptr)
    {
        if(runtime->hipStreamGetDevice_fn(stream, &device_id) != hipSuccess) device_id = -1;
    }

    if(device_id < 0 && runtime->hipGetDevice_fn != nullptr)
    {
        if(runtime->hipGetDevice_fn(&device_id) != hipSuccess) device_id = -1;
    }

    if(device_id < 0) return rocprofiler_agent_id_t{.handle = 0};

    for(const auto* a : ::rocprofiler::agent::get_agents())
    {
        if(a != nullptr && a->type == ROCPROFILER_AGENT_TYPE_GPU &&
           a->logical_node_type_id == device_id)
        {
            return a->id;
        }
    }
    return rocprofiler_agent_id_t{.handle = 0};
}

// Forward decl: real body lands in Task 11. Defined as empty stub here so this
// task builds standalone. Task 11 will replace the stub body, keeping it in
// the same anonymous namespace so unqualified name lookup from wrap_launch
// resolves to the same entity.
void
emit_graph_launch_record(const launch_state& s, rocprofiler_timestamp_t end_ts);
void
emit_graph_launch_record(const launch_state&, rocprofiler_timestamp_t)
{}

// hipGraphLaunch and hipGraphLaunch_spt share the SAME signature. A naive
// wrap_launch<RetT> template would collapse them into a single instantiation
// and the two static next_func slots would alias. The LaunchApiTag template
// parameter forces distinct instantiations (and thus distinct next_func
// storage) for each API.
enum class LaunchApiTag
{
    hipGraphLaunch,
    hipGraphLaunch_spt
};

template <LaunchApiTag Tag, typename RetT>
auto
wrap_launch(RetT (*next)(::hipGraphExec_t, ::hipStream_t))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t exec, ::hipStream_t stream) -> RetT {
        g_launch_stack.emplace_back();
        auto& s         = g_launch_stack.back();
        s.graph_exec_id = lookup_graph_exec_id(exec);
        if(s.graph_exec_id == 0)
        {
            // Attach-mid-process fallback: rocprofiler may have attached after
            // hipGraphInstantiate ran, so the map has no entry. Assign now so
            // subsequent dispatches in this launch still get a non-zero ID.
            s.graph_exec_id = assign_graph_exec_id(exec);
        }
        s.thread_id      = common::get_tid();
        // NOTE: correlation_id is left 0 for now. The spec's correlation_id-join
        // contract requires reading the HIP API tracing TLS that is built up by
        // the outer HIP wrapper, but no such TLS accessor exists yet in hip.cpp.
        // A follow-up will add the accessor; the GRAPH_LAUNCH record is still
        // emitted, it just won't join cleanly to HIP API records until then.
        s.correlation_id = 0;
        s.start_ts       = rocprofiler_timestamp_t{common::timestamp_ns()};

        // Resolve agent_id from the *launch stream* (spec §4.2). The GRAPH_LAUNCH
        // record summarizes the entire launch, so its agent_id must reflect the
        // user-visible launch stream, not whatever internal HSA queue an arbitrary
        // graph segment happens to dispatch on (in multi-device graphs they can
        // differ). Stamping at launch enter also guarantees the value is correct
        // regardless of whether the kernel-dispatch WriteInterceptor path runs
        // (e.g., subscriber requests GRAPH_LAUNCH only, not KERNEL_DISPATCH).
        //
        // queue_id is left as the zero-initialized value: a graph launch is not
        // bound to a single HW queue (multiple internal streams may be used for
        // parallel branches), so there is no single defensible queue_id at the
        // launch granularity. Consumers must not assume queue_id is meaningful
        // for GRAPH_LAUNCH records.
        s.agent_id = resolve_launch_stream_agent(stream);

        auto ret = next_func(exec, stream);

        // Per spec §4.4: emit summary record only on hipSuccess. Always emit
        // on success (including dispatch_count == 0) per §4.2.
        auto end_ts = rocprofiler_timestamp_t{common::timestamp_ns()};
        if(ret == hipSuccess)
        {
            emit_graph_launch_record(s, end_ts);
        }
        // Pop unconditionally — TLS state must always be cleaned up, even on
        // error paths.
        g_launch_stack.pop_back();
        return ret;
    };
}
}  // namespace

void
init()
{
    // The map is default-constructed at static-init; nothing to do here yet.
    // Future tasks may add lifecycle wiring (Task 8 hooks instantiate/destroy
    // wrappers via update_table; this init() exists as a stable entry point).
}

launch_state*
current_launch_state()
{
    return g_launch_stack.empty() ? nullptr : &g_launch_stack.back();
}

uint64_t
lookup_graph_exec_id(::hipGraphExec_t exec)
{
    if(exec == nullptr) return 0;
    std::shared_lock lock{g_map_mutex};
    auto             it = g_exec_to_id.find(exec);
    return it == g_exec_to_id.end() ? 0 : it->second;
}

// Explicit specialization for the HIP runtime dispatch table. Wraps the four
// graph-lifecycle entry points so that:
//   - successful hipGraphInstantiate* assigns a fresh monotonic graph_exec_id
//   - hipGraphExecDestroy removes the map entry
//
// Each install site is guarded with a null check so older HIP runtimes that
// lack one of these fn slots don't NPE.
template <>
void
update_table(::HipDispatchTable* table)
{
    if(table == nullptr) return;
    if(table->hipGraphInstantiate_fn)
        table->hipGraphInstantiate_fn = wrap_instantiate(table->hipGraphInstantiate_fn);
    if(table->hipGraphInstantiateWithFlags_fn)
        table->hipGraphInstantiateWithFlags_fn =
            wrap_instantiate(table->hipGraphInstantiateWithFlags_fn);
    if(table->hipGraphInstantiateWithParams_fn)
        table->hipGraphInstantiateWithParams_fn =
            wrap_instantiate(table->hipGraphInstantiateWithParams_fn);
    if(table->hipGraphExecDestroy_fn)
        table->hipGraphExecDestroy_fn = wrap_destroy(table->hipGraphExecDestroy_fn);
    if(table->hipGraphLaunch_fn)
        table->hipGraphLaunch_fn =
            wrap_launch<LaunchApiTag::hipGraphLaunch>(table->hipGraphLaunch_fn);
    if(table->hipGraphLaunch_spt_fn)
        table->hipGraphLaunch_spt_fn =
            wrap_launch<LaunchApiTag::hipGraphLaunch_spt>(table->hipGraphLaunch_spt_fn);
}

}  // namespace graph
}  // namespace hip
}  // namespace rocprofiler
