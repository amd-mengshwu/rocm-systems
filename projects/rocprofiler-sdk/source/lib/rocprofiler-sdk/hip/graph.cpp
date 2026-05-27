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
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
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

// Build the callback payload struct used by all HIP_GRAPH callback fires.
rocprofiler_callback_tracing_hip_graph_data_t
make_hip_graph_payload(uint64_t graph_exec_id, ::hipGraphExec_t exec)
{
    return common::init_public_api_struct(
        rocprofiler_callback_tracing_hip_graph_data_t{},
        graph_exec_id,
        rocprofiler_address_t{.ptr = static_cast<const void*>(exec)});
}

// Fire a HIP_GRAPH callback in phase NONE (used for EXEC_CREATE and EXEC_DESTROY).
//
// Mirrors hip/stream.cpp's create_write_functor / create_destroy_functor pattern:
// populate subscribed callback contexts for the (domain, op) pair, build a payload,
// and dispatch to each via execute_phase_none_callbacks. HIP_GRAPH has no buffered
// counterpart (the GRAPH_LAUNCH buffer record is a separate, summary-level domain
// emitted from emit_graph_launch_record), so we use the single-DomainIdx populate
// overload templated on rocprofiler_callback_tracing_kind_t.
void
fire_hip_graph_none_callback(rocprofiler_hip_graph_operation_t op,
                             uint64_t                          graph_exec_id,
                             ::hipGraphExec_t                  exec)
{
    auto callback_contexts = tracing::callback_context_data_vec_t{};
    auto external_corr_ids = tracing::external_correlation_id_map_t{};

    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                               static_cast<rocprofiler_tracing_operation_t>(op),
                               callback_contexts,
                               external_corr_ids);

    if(callback_contexts.empty()) return;

    auto thr_id = common::get_tid();

    tracing::update_external_correlation_ids(
        external_corr_ids, thr_id, ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API);

    auto*      corr_id          = context::get_latest_correlation_id();
    auto       internal_corr_id = (corr_id) ? corr_id->internal : uint64_t{0};
    auto       ancestor_corr_id = (corr_id) ? corr_id->ancestor : uint64_t{0};
    auto       tracer_data      = make_hip_graph_payload(graph_exec_id, exec);

    tracing::execute_phase_none_callbacks(callback_contexts,
                                          thr_id,
                                          internal_corr_id,
                                          external_corr_ids,
                                          ancestor_corr_id,
                                          ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                                          static_cast<rocprofiler_tracing_operation_t>(op),
                                          tracer_data);
}

// One static next_func slot per template instantiation; the 3 hipGraphInstantiate*
// APIs have distinct signatures so they instantiate separately. The captureless
// lambda decays to a plain function pointer via unary +.
template <typename RetT, typename... Args>
auto
wrap_instantiate(RetT (*next)(::hipGraphExec_t*, Args...))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t* out, Args... args) -> RetT {
        auto ret = next_func(out, std::forward<Args>(args)...);
        if(ret == hipSuccess && out != nullptr && *out != nullptr)
        {
            auto exec_id = assign_graph_exec_id(*out);
            fire_hip_graph_none_callback(
                ROCPROFILER_HIP_GRAPH_OPERATION_HIP_GRAPH_EXEC_CREATE, exec_id, *out);
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
        // Fire the callback before forgetting the map entry so lookup still
        // returns the real id, and before destroy so the handle is still valid.
        if(exec != nullptr)
        {
            auto exec_id = lookup_graph_exec_id(exec);
            fire_hip_graph_none_callback(
                ROCPROFILER_HIP_GRAPH_OPERATION_HIP_GRAPH_EXEC_DESTROY, exec_id, exec);
        }
        forget_graph_exec(exec);
        return next_func(exec);
    };
}

// std::deque (not std::vector) so references to existing entries remain valid
// when nested host-callback launches push onto the same thread's stack.
thread_local std::deque<launch_state> g_launch_stack;

// Returns the GPU agent for the launch stream's device, or {0} on failure.
// Uses the saved (un-wrapped) HIP dispatch table to avoid re-entering rocprofiler's
// own tracing wrappers from inside hipGraphLaunch.
rocprofiler_agent_id_t
resolve_launch_stream_agent(::hipStream_t stream)
{
    auto& saved_table = ::rocprofiler::hip::get_table();
    auto* runtime     = saved_table.runtime;
    if(runtime == nullptr) return rocprofiler_agent_id_t{.handle = 0};

    int  device_id         = -1;
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
    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_BUFFER_TRACING_GRAPH_LAUNCH,
                               /*operation*/ 0u,
                               tracing_data_v.buffered_contexts,
                               tracing_data_v.external_correlation_ids);

    if(tracing_data_v.buffered_contexts.empty()) return;

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
        s.dispatch_count};

    tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                           s.thread_id,
                                           s.correlation_id,
                                           tracing_data_v.external_correlation_ids,
                                           /*ancestor_corr_id*/ uint64_t{0},
                                           ROCPROFILER_BUFFER_TRACING_GRAPH_LAUNCH,
                                           /*operation*/ 0u,
                                           record);
}

// hipGraphLaunch and hipGraphLaunch_spt share a signature; the LaunchApiTag
// template parameter gives each its own static next_func slot.
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
        // The same callback_contexts vector flows through both ENTER and EXIT
        // phases per the tracing::execute_phase_exit_callbacks contract.
        auto callback_contexts = tracing::callback_context_data_vec_t{};
        auto external_corr_ids = tracing::external_correlation_id_map_t{};
        tracing::populate_contexts(
            ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
            static_cast<rocprofiler_tracing_operation_t>(
                ROCPROFILER_HIP_GRAPH_OPERATION_HIP_GRAPH_LAUNCH),
            callback_contexts,
            external_corr_ids);

        g_launch_stack.emplace_back();
        auto& s         = g_launch_stack.back();
        s.graph_exec_id = lookup_graph_exec_id(exec);
        if(s.graph_exec_id == 0)
        {
            // Attach-mid-process fallback: rocprofiler may have attached after
            // hipGraphInstantiate ran, so the map has no entry.
            s.graph_exec_id = assign_graph_exec_id(exec);
        }
        s.thread_id = common::get_tid();
        if(auto* cid = ::rocprofiler::context::get_latest_correlation_id())
            s.correlation_id = cid->internal;
        else
            s.correlation_id = 0;
        s.start_ts = rocprofiler_timestamp_t{common::timestamp_ns()};
        // queue_id intentionally left zero: a graph launch may dispatch across
        // multiple internal HW queues for parallel branches.
        s.agent_id = resolve_launch_stream_agent(stream);

        if(!callback_contexts.empty())
        {
            auto  tracer_data            = make_hip_graph_payload(s.graph_exec_id, exec);
            auto* enter_cid              = ::rocprofiler::context::get_latest_correlation_id();
            auto  enter_internal_corr_id = (enter_cid) ? enter_cid->internal : uint64_t{0};
            auto  enter_ancestor_corr_id = (enter_cid) ? enter_cid->ancestor : uint64_t{0};

            tracing::update_external_correlation_ids(
                external_corr_ids,
                s.thread_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API);

            tracing::execute_phase_enter_callbacks(
                callback_contexts,
                s.thread_id,
                enter_internal_corr_id,
                external_corr_ids,
                enter_ancestor_corr_id,
                ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                static_cast<rocprofiler_tracing_operation_t>(
                    ROCPROFILER_HIP_GRAPH_OPERATION_HIP_GRAPH_LAUNCH),
                tracer_data);
        }

        auto ret = next_func(exec, stream);

        // EXIT must fire BEFORE g_launch_stack.pop_back() so subscribers
        // calling current_launch_state() still see the active launch.
        if(!callback_contexts.empty())
        {
            auto tracer_data = make_hip_graph_payload(s.graph_exec_id, exec);
            tracing::execute_phase_exit_callbacks(
                callback_contexts,
                external_corr_ids,
                ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                static_cast<rocprofiler_tracing_operation_t>(
                    ROCPROFILER_HIP_GRAPH_OPERATION_HIP_GRAPH_LAUNCH),
                tracer_data);
        }

        auto end_ts = rocprofiler_timestamp_t{common::timestamp_ns()};
        if(ret == hipSuccess)
        {
            emit_graph_launch_record(s, end_ts);
        }
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
