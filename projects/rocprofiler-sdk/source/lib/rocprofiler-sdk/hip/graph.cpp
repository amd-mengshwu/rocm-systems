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
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>  // pulls in <hip/amd_detail/hip_api_trace.hpp>

#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

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
// for reference stability across nested launches: wrap_launch's lambda holds a
// reference to g_launch_stack.back(), and if a graph host-callback node calls
// back into wrap_launch on this thread it will push another launch_state.
// std::deque does not move existing elements on growth, so the parent's
// reference (and any pointer returned by current_launch_state()) remains
// valid; std::vector would invalidate them on reallocation.
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

void
emit_graph_launch_record(const launch_state& s, rocprofiler_timestamp_t end_ts)
{
    // GRAPH_LAUNCH is a buffer-only domain (no callback tracing kind defined
    // for it), so use the single-DomainIdx populate_contexts overload that
    // fills only buffered_contexts + external_correlation_ids.
    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_BUFFER_TRACING_GRAPH_LAUNCH,
                               /*operation*/ 0u,
                               tracing_data_v.buffered_contexts,
                               tracing_data_v.external_correlation_ids);

    if(tracing_data_v.buffered_contexts.empty()) return;

    // rocprofiler_async_correlation_id_t has 2 fields (internal, external);
    // execute_buffer_record_emplace overwrites correlation_id from its
    // internal_corr_id + external_corr_ids args, so default-construct it here.
    auto record = rocprofiler_buffer_tracing_graph_launch_record_t{
        sizeof(rocprofiler_buffer_tracing_graph_launch_record_t),
        ROCPROFILER_BUFFER_TRACING_GRAPH_LAUNCH,
        /*operation*/ 0u,
        rocprofiler_async_correlation_id_t{},
        s.thread_id,
        s.start_ts,
        end_ts,
        s.agent_id,
        s.queue_id,
        s.graph_exec_id,
        s.dispatch_count};  // launch_state's counter -> record's kernel_dispatch_count

    tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                           s.thread_id,
                                           s.correlation_id,
                                           tracing_data_v.external_correlation_ids,
                                           /*ancestor_corr_id*/ uint64_t{0},
                                           ROCPROFILER_BUFFER_TRACING_GRAPH_LAUNCH,
                                           /*operation*/ 0u,
                                           record);
}

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
        // Spec §7 item 9: GRAPH_LAUNCH.correlation_id must equal the internal
        // correlation_id of the originating HIP_RUNTIME_API record so consumers
        // can join graph_launch_trace.csv to hip_api_trace.csv on Correlation_Id.
        //
        // The outer HIP API tracing wrapper (hip::hip_api_impl::functor in
        // hip.cpp) constructs the correlation_id via
        // correlation_tracing_service::construct(), which pushes it onto the
        // per-thread stack maintained in context/correlation_id.cpp BEFORE
        // calling the wrapped function (= this lambda). The matching pop runs
        // AFTER the wrapped function returns. So while we are executing here
        // inside the HIP API call, get_latest_correlation_id() returns exactly
        // that internal_corr_id. (Returns nullptr only if no HIP API wrapper is
        // active on this thread — e.g., a direct call into hipGraphLaunch from
        // a non-traced context — in which case 0 is the right sentinel.)
        if(auto* cid = ::rocprofiler::context::get_latest_correlation_id())
            s.correlation_id = cid->internal;
        else
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

// Map rocprofiler_hip_graph_operation_t to respective name
template <size_t OpIdx>
struct hip_graph_operation_name;

#define HIP_GRAPH_OPERATION_NAME(ENUM)                                                             \
    template <>                                                                                    \
    struct hip_graph_operation_name<ROCPROFILER_HIP_GRAPH_OPERATION_##ENUM>                        \
    {                                                                                              \
        static constexpr auto name          = "HIP_GRAPH_OPERATION_" #ENUM;                        \
        static constexpr auto operation_idx = ROCPROFILER_HIP_GRAPH_OPERATION_##ENUM;              \
    };

HIP_GRAPH_OPERATION_NAME(NONE)
HIP_GRAPH_OPERATION_NAME(HIP_GRAPH_EXEC_CREATE)
HIP_GRAPH_OPERATION_NAME(HIP_GRAPH_EXEC_DESTROY)
HIP_GRAPH_OPERATION_NAME(HIP_GRAPH_LAUNCH)
#undef HIP_GRAPH_OPERATION_NAME

template <size_t OpIdx, size_t... OpIdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id) return hip_graph_operation_name<OpIdx>::name;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return name_by_id(id, std::index_sequence<OpIdxTail...>{});
    else
        return nullptr;
}

template <size_t OpIdx, size_t... OpIdxTail>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto _idx = hip_graph_operation_name<OpIdx>::operation_idx;
    if(_idx < ROCPROFILER_HIP_GRAPH_OPERATION_LAST) _id_list.emplace_back(_idx);

    if constexpr(sizeof...(OpIdxTail) > 0) get_ids(_id_list, std::index_sequence<OpIdxTail...>{});
}
}  // namespace

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

const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_HIP_GRAPH_OPERATION_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    constexpr auto last_id = ROCPROFILER_HIP_GRAPH_OPERATION_LAST;
    auto           _data   = std::vector<uint32_t>{};
    _data.reserve(last_id);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_HIP_GRAPH_OPERATION_LAST>{});
    return _data;
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
