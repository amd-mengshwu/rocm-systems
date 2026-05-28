#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import csv
import os
import sys
import pytest
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_csv(filename):
    out = []
    with open(filename, "r") as inp:
        for row in csv.DictReader(inp):
            out.append(row)
    return out


def _is_empty(v):
    return v == "" or v is None


def _graph_rows(kernel_input_data):
    """KERNEL_DISPATCH rows that carry a populated, nonzero Graph_Exec_Id."""
    out = []
    for r in kernel_input_data:
        if r.get("Kind") != "KERNEL_DISPATCH":
            continue
        v = r.get("Graph_Exec_Id", "")
        if _is_empty(v) or int(v) == 0:
            continue
        out.append(r)
    return out


def _memcpy_graph_rows(memory_copy_input_data):
    """memory_copy rows that carry a populated, nonzero Graph_Exec_Id."""
    out = []
    for r in memory_copy_input_data:
        v = r.get("Graph_Exec_Id", "")
        if _is_empty(v) or int(v) == 0:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# Kernel CSV column presence / empty-on-zero rendering
# ---------------------------------------------------------------------------


def test_columns_present(kernel_input_data):
    """The new Graph_Exec_Id and Graph_Node_Id columns must be present."""
    row = kernel_input_data[0]
    assert "Graph_Exec_Id" in row, list(row.keys())
    assert "Graph_Node_Id" in row, list(row.keys())


def test_non_graph_rows_render_empty(kernel_input_data):
    """Dispatches not from a graph launch render Graph_Exec_Id and
    Graph_Node_Id as empty strings (not literal '0')."""
    non_graph_rows = [
        r
        for r in kernel_input_data
        if r.get("Kind") == "KERNEL_DISPATCH" and _is_empty(r.get("Graph_Exec_Id"))
    ]
    if not non_graph_rows:
        pytest.skip("no non-graph KERNEL_DISPATCH rows in this trace")
    for r in non_graph_rows:
        assert r["Graph_Exec_Id"] == "", r
        assert r["Graph_Node_Id"] == "", r


# ---------------------------------------------------------------------------
# Kernel CSV: structural counts
# ---------------------------------------------------------------------------


def test_total_graph_dispatch_count(
    kernel_input_data,
    expected_iterations,
    expected_execs,
    expected_kernel_nodes_per_launch,
):
    """Total graph kernel dispatches = (iterations * execs + 1) *
    kernel_nodes_per_launch. The +1 is the extra exec_b launch after the
    failed launch."""
    rows = _graph_rows(kernel_input_data)
    expected_launches = expected_iterations * expected_execs + 1
    expected = expected_launches * expected_kernel_nodes_per_launch
    assert len(rows) == expected, (len(rows), expected)


def test_two_distinct_exec_ids(kernel_input_data, expected_execs):
    """The two hipGraphInstantiate calls must produce two distinct nonzero
    Graph_Exec_Ids."""
    exec_ids = {int(r["Graph_Exec_Id"]) for r in _graph_rows(kernel_input_data)}
    assert 0 not in exec_ids
    assert len(exec_ids) == expected_execs, exec_ids


# ---------------------------------------------------------------------------
# Per-launch node_id structure and stability
# ---------------------------------------------------------------------------


def test_graph_node_id_range_per_launch(
    kernel_input_data,
    expected_kernel_nodes_per_launch,
    expected_iterations,
    expected_execs,
):
    """Each launch contributes exactly kernel_nodes_per_launch unique
    Graph_Node_Id values via its kernel-dispatch rows. The full
    0..nodes_per_launch-1 range is verified across kernel + memcpy rows in
    test_graph_node_id_range_per_launch_full_topology."""
    rows = _graph_rows(kernel_input_data)
    by_corr = defaultdict(list)
    for r in rows:
        by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    expected_launches = expected_iterations * expected_execs + 1
    assert len(by_corr) == expected_launches, (len(by_corr), expected_launches)
    for corr, nodes in by_corr.items():
        assert len(nodes) == expected_kernel_nodes_per_launch, (corr, nodes)
        assert len(set(nodes)) == len(nodes), (corr, sorted(nodes))


def test_graph_node_id_stable_per_exec(kernel_input_data, expected_nodes_per_launch):
    """For each (Graph_Exec_Id, Graph_Node_Id) pair, all kernel-dispatch
    occurrences across the trace must report the same Kernel_Name. Per-position
    stability."""
    by_exec_node = defaultdict(list)
    for r in _graph_rows(kernel_input_data):
        key = (int(r["Graph_Exec_Id"]), int(r["Graph_Node_Id"]))
        by_exec_node[key].append(r["Kernel_Name"])
    for key, names in by_exec_node.items():
        unique = set(names)
        assert len(unique) == 1, (key, unique)


def test_distinct_kernel_nodes_remain_distinct(
    kernel_input_data, expected_distinct_kernels
):
    """Two distinct nodes that launch the same kernel must still have
    distinct Graph_Node_Id values within a single launch."""
    rows = _graph_rows(kernel_input_data)
    first_corr = rows[0]["Correlation_Id"]
    first_launch = [r for r in rows if r["Correlation_Id"] == first_corr]
    kernel_counts = Counter(r["Kernel_Name"] for r in first_launch)
    assert len(kernel_counts) == expected_distinct_kernels, kernel_counts
    a_ids = {
        int(r["Graph_Node_Id"]) for r in first_launch if "kernel_a" in r["Kernel_Name"]
    }
    assert len(a_ids) == 3, a_ids


# ---------------------------------------------------------------------------
# GRAPH_LAUNCH summary record
# ---------------------------------------------------------------------------


def test_graph_launch_record_count(
    graph_launch_input_data, expected_iterations, expected_execs
):
    """One GRAPH_LAUNCH record per successful hipGraphLaunch. Failed
    launches must NOT emit a record."""
    expected_launches = expected_iterations * expected_execs + 1
    assert len(graph_launch_input_data) == expected_launches, (
        len(graph_launch_input_data),
        expected_launches,
    )


def test_graph_launch_dispatch_counts(
    graph_launch_input_data, expected_kernel_nodes_per_launch
):
    """Each GRAPH_LAUNCH record reports Kernel_Dispatch_Count =
    kernel_nodes_per_launch. The HSA queue interceptor counts every AQL
    kernel dispatch packet observed during the launch."""
    for r in graph_launch_input_data:
        dc = int(r["Kernel_Dispatch_Count"])
        assert dc == expected_kernel_nodes_per_launch, (r, dc)


def test_graph_launch_exec_ids_match_kernel_csv(
    graph_launch_input_data, kernel_input_data, expected_execs
):
    """Graph_Exec_Ids in GRAPH_LAUNCH records must match the set on kernel
    CSV's graph rows."""
    launch_ids = {int(r["Graph_Exec_Id"]) for r in graph_launch_input_data}
    kernel_ids = {int(r["Graph_Exec_Id"]) for r in _graph_rows(kernel_input_data)}
    assert launch_ids == kernel_ids, (launch_ids, kernel_ids)
    assert len(launch_ids) == expected_execs


def test_graph_launch_correlation_joins_to_hip_api(
    graph_launch_input_data, hip_api_input_data
):
    """Every GRAPH_LAUNCH Correlation_Id must join to a HIP_RUNTIME_API row
    for hipGraphLaunch or hipGraphLaunch_spt, so consumers can join the two
    CSVs on Correlation_Id."""
    api_by_corr = {}
    for r in hip_api_input_data:
        name = r.get("Function") or r.get("Name") or r.get("Operation")
        api_by_corr[r["Correlation_Id"]] = name
    accepted = {"hipGraphLaunch", "hipGraphLaunch_spt"}
    for gr in graph_launch_input_data:
        corr = gr["Correlation_Id"]
        assert corr not in ("0", ""), gr
        assert corr in api_by_corr, (corr, list(api_by_corr.items())[:3])
        assert api_by_corr[corr] in accepted, (corr, api_by_corr[corr])


# ---------------------------------------------------------------------------
# Memory copy CSV
# ---------------------------------------------------------------------------


def test_memcpy_columns_present(memory_copy_input_data):
    row = memory_copy_input_data[0]
    assert "Graph_Exec_Id" in row, list(row.keys())
    assert "Graph_Node_Id" in row, list(row.keys())


def test_memcpy_graph_attribution_count(
    memory_copy_input_data,
    expected_iterations,
    expected_execs,
    expected_memcpy_nodes_per_launch,
):
    """Graph-attributed memcpy count = (iterations * execs + 1) *
    memcpy_nodes_per_launch.

    On AMD HIP this is typically 0 because in-graph memcpys are dispatched
    as blit kernels and surface as KERNEL_DISPATCH records rather than
    MEMORY_COPY records."""
    rows = _memcpy_graph_rows(memory_copy_input_data)
    expected_launches = expected_iterations * expected_execs + 1
    expected = expected_launches * expected_memcpy_nodes_per_launch
    assert len(rows) == expected, (len(rows), expected)


def test_memcpy_exec_ids_match_kernel_csv(
    memory_copy_input_data, kernel_input_data, expected_execs
):
    """When graph-attributed memcpy rows exist, their Graph_Exec_Id set must
    be a subset of the kernel-CSV set."""
    memcpy_rows = _memcpy_graph_rows(memory_copy_input_data)
    if not memcpy_rows:
        pytest.skip("no graph-attributed memcpys (AMD HIP blit-kernel path)")
    memcpy_ids = {int(r["Graph_Exec_Id"]) for r in memcpy_rows}
    kernel_ids = {int(r["Graph_Exec_Id"]) for r in _graph_rows(kernel_input_data)}
    assert memcpy_ids.issubset(kernel_ids), (memcpy_ids, kernel_ids)


def test_non_graph_memcpy_rows_render_empty(memory_copy_input_data):
    """Memcpys not from a graph launch render the new columns as empty."""
    non_graph_rows = [
        r for r in memory_copy_input_data if _is_empty(r.get("Graph_Exec_Id"))
    ]
    if not non_graph_rows:
        pytest.skip("no non-graph memcpy rows in this trace")
    for r in non_graph_rows:
        assert r["Graph_Exec_Id"] == "", r
        assert r["Graph_Node_Id"] == "", r


# ---------------------------------------------------------------------------
# Full-topology view (kernel + memcpy union per launch)
# ---------------------------------------------------------------------------


def test_graph_node_id_range_per_launch_full_topology(
    kernel_input_data,
    request,
    expected_nodes_per_launch,
    expected_iterations,
    expected_execs,
):
    """Union of kernel + memcpy graph_node_ids per launch must equal
    {0..nodes_per_launch-1}. The node counter is shared across all graph
    nodes — every external_corr_id request fired during the launch
    increments it, regardless of whether the resulting record surfaces as
    a KERNEL_DISPATCH or MEMORY_COPY."""
    by_corr = defaultdict(list)
    for r in _graph_rows(kernel_input_data):
        by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    memcpy_path = request.config.getoption("--memory-copy-input")
    if memcpy_path and os.path.exists(memcpy_path):
        for r in _memcpy_graph_rows(_read_csv(memcpy_path)):
            by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    expected_launches = expected_iterations * expected_execs + 1
    assert len(by_corr) == expected_launches, (len(by_corr), expected_launches)
    for corr, nodes in by_corr.items():
        assert sorted(nodes) == list(range(expected_nodes_per_launch)), (
            corr,
            sorted(nodes),
        )


# ---------------------------------------------------------------------------
# Additional checks added in this revision
# ---------------------------------------------------------------------------


def test_per_exec_kernel_launch_counts(
    kernel_input_data, expected_iterations, expected_kernel_nodes_per_launch
):
    """exec_a issues `iterations` valid launches (then destroyed; the
    post-destroy launch fails). exec_b issues `iterations + 1` valid
    launches. The kernel CSV must reflect this split per Graph_Exec_Id."""
    rows = _graph_rows(kernel_input_data)
    launches_per_exec = defaultdict(set)
    for r in rows:
        launches_per_exec[int(r["Graph_Exec_Id"])].add(r["Correlation_Id"])
    counts = sorted(len(corrs) for corrs in launches_per_exec.values())
    assert counts == [expected_iterations, expected_iterations + 1], counts


def test_per_launch_node_sequence_stable(kernel_input_data):
    """The full ordered (Graph_Node_Id, Kernel_Name) sequence per launch must
    be identical across every launch of the same Graph_Exec_Id. Stronger
    than per-position name stability: also catches reordering bugs."""
    by_exec_launch = defaultdict(lambda: defaultdict(list))
    for r in _graph_rows(kernel_input_data):
        exec_id = int(r["Graph_Exec_Id"])
        by_exec_launch[exec_id][r["Correlation_Id"]].append(
            (int(r["Graph_Node_Id"]), r["Kernel_Name"])
        )
    for exec_id, launches in by_exec_launch.items():
        signatures = set()
        for nodes in launches.values():
            nodes.sort()
            signatures.add(tuple(nodes))
        assert len(signatures) == 1, (exec_id, signatures)


def test_graph_launch_per_exec_record_counts(
    graph_launch_input_data, expected_iterations, expected_execs
):
    """One exec has `iterations` GRAPH_LAUNCH records; the other has
    `iterations + 1`. Confirms failed-launch suppression at the per-exec level."""
    by_exec = Counter(int(r["Graph_Exec_Id"]) for r in graph_launch_input_data)
    assert len(by_exec) == expected_execs, by_exec
    counts = sorted(by_exec.values())
    assert counts == [expected_iterations, expected_iterations + 1], counts


def test_graph_launch_timestamps_sane(graph_launch_input_data):
    """End_Timestamp >= Start_Timestamp; Start_Timestamp > 0."""
    for r in graph_launch_input_data:
        start = int(r["Start_Timestamp"])
        end = int(r["End_Timestamp"])
        assert start > 0, r
        assert end >= start, r


def test_graph_launch_agent_id_nonzero(graph_launch_input_data):
    """Agent_Id must be a real handle (resolved from the launch stream's
    device). A zero handle indicates a regression in the agent resolution
    path that would crash downstream consumers."""
    for r in graph_launch_input_data:
        agent = r["Agent_Id"]
        assert agent and agent != "0", r


def test_graph_launch_correlation_unique(graph_launch_input_data):
    """Each GRAPH_LAUNCH record carries a distinct nonzero Correlation_Id."""
    corrs = [r["Correlation_Id"] for r in graph_launch_input_data]
    assert all(c not in ("0", "") for c in corrs)
    assert len(set(corrs)) == len(corrs), Counter(corrs).most_common(3)


def test_graph_kernels_carry_stream_id(kernel_input_data):
    """Kernel dispatches from a graph launch also carry a Stream_Id (the
    launch stream). HIP_STREAM and HIP_GRAPH attribution must coexist."""
    rows = _graph_rows(kernel_input_data)
    if "Stream_Id" not in rows[0]:
        pytest.skip("Stream_Id column not present in this trace")
    stream_ids = {r["Stream_Id"] for r in rows}
    stream_ids.discard("")
    stream_ids.discard("0")
    assert stream_ids, "no nonzero Stream_Id values on graph kernel dispatches"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
