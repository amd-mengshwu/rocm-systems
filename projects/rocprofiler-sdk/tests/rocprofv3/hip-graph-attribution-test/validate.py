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


def _graph_kernel_rows(kernel_input_data):
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
# Kernel CSV: column presence and empty-on-zero rendering
# ---------------------------------------------------------------------------


def test_kernel_columns_present(kernel_input_data):
    row = kernel_input_data[0]
    assert "Graph_Exec_Id" in row, list(row.keys())
    assert "Graph_Node_Id" in row, list(row.keys())


def test_kernel_non_graph_rows_render_empty(kernel_input_data):
    """Non-graph dispatches render Graph_Exec_Id and Graph_Node_Id as empty
    strings, not literal '0'."""
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
# Kernel CSV: structural counts per launch
# ---------------------------------------------------------------------------


def test_graph_dispatch_total_count(
    kernel_input_data,
    expected_iterations,
    expected_execs,
    expected_kernel_nodes_per_launch,
):
    """Total kernel dispatches from graph launches must equal
    (iterations * execs + 1) * kernel_nodes_per_launch.

    The +1 is the extra exec_b launch issued after the failed exec_a launch."""
    rows = _graph_kernel_rows(kernel_input_data)
    expected_launches = expected_iterations * expected_execs + 1
    expected = expected_launches * expected_kernel_nodes_per_launch
    assert len(rows) == expected, (len(rows), expected)


def test_graph_exec_ids_exact(kernel_input_data, expected_execs):
    """Exactly `expected_execs` distinct nonzero Graph_Exec_Ids appear."""
    exec_ids = {int(r["Graph_Exec_Id"]) for r in _graph_kernel_rows(kernel_input_data)}
    assert 0 not in exec_ids
    assert len(exec_ids) == expected_execs, exec_ids


def test_per_exec_launch_counts(
    kernel_input_data, expected_iterations, expected_kernel_nodes_per_launch
):
    """The destroyed exec_a issues `iterations` valid launches before destruction.
    exec_b issues `iterations + 1` valid launches (the loop iterations plus one
    extra after exec_a's failed-launch test). Their Graph_Exec_Ids are assigned
    in instantiate order, but we don't know which is exec_a vs exec_b — we know
    one must have `iterations` launches and the other `iterations + 1`."""
    rows = _graph_kernel_rows(kernel_input_data)
    launches_per_exec = defaultdict(set)
    for r in rows:
        launches_per_exec[int(r["Graph_Exec_Id"])].add(r["Correlation_Id"])
    launch_counts = sorted(len(corrs) for corrs in launches_per_exec.values())
    assert launch_counts == [expected_iterations, expected_iterations + 1], launch_counts


# ---------------------------------------------------------------------------
# Kernel CSV: per-launch node_id structure
# ---------------------------------------------------------------------------


def test_node_id_count_per_launch(
    kernel_input_data,
    expected_kernel_nodes_per_launch,
    expected_iterations,
    expected_execs,
):
    """Each launch contributes exactly kernel_nodes_per_launch unique
    Graph_Node_Id values from its kernel-dispatch rows."""
    rows = _graph_kernel_rows(kernel_input_data)
    by_corr = defaultdict(list)
    for r in rows:
        by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    expected_launches = expected_iterations * expected_execs + 1
    assert len(by_corr) == expected_launches, (len(by_corr), expected_launches)
    for corr, nodes in by_corr.items():
        assert len(nodes) == expected_kernel_nodes_per_launch, (corr, nodes)
        assert len(set(nodes)) == len(nodes), (corr, sorted(nodes))


def test_node_id_to_kernel_name_stable_per_exec(kernel_input_data):
    """For each (Graph_Exec_Id, Graph_Node_Id) pair, all kernel-dispatch
    occurrences across the trace must report the same Kernel_Name. This is
    the strongest stability check: position N within exec X always maps to
    the same kernel."""
    by_exec_node = defaultdict(list)
    for r in _graph_kernel_rows(kernel_input_data):
        key = (int(r["Graph_Exec_Id"]), int(r["Graph_Node_Id"]))
        by_exec_node[key].append(r["Kernel_Name"])
    for key, names in by_exec_node.items():
        unique = set(names)
        assert len(unique) == 1, (key, unique)


def test_per_launch_node_sequence_stable(kernel_input_data):
    """The full ordered sequence of (Graph_Node_Id, Kernel_Name) within a
    launch must be identical across every launch of the same Graph_Exec_Id.
    Stronger than per-position stability: also catches reordering bugs."""
    rows = _graph_kernel_rows(kernel_input_data)

    # Group by (exec_id, correlation_id) -> sorted list of (node_id, kernel_name)
    by_exec_launch = defaultdict(lambda: defaultdict(list))
    for r in rows:
        exec_id = int(r["Graph_Exec_Id"])
        corr = r["Correlation_Id"]
        by_exec_launch[exec_id][corr].append(
            (int(r["Graph_Node_Id"]), r["Kernel_Name"])
        )

    for exec_id, launches in by_exec_launch.items():
        signatures = set()
        for corr, nodes in launches.items():
            nodes.sort()
            signatures.add(tuple(nodes))
        assert len(signatures) == 1, (
            f"exec_id {exec_id}: {len(launches)} launches produced "
            f"{len(signatures)} different (node_id, kernel_name) sequences: "
            f"{signatures}"
        )


def test_distinct_kernel_a_nodes(kernel_input_data, expected_distinct_kernels):
    """The 3 distinct graph nodes that all call kernel_a must occupy 3
    distinct Graph_Node_Id values within a launch."""
    rows = _graph_kernel_rows(kernel_input_data)
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
    """One record per successful hipGraphLaunch; failed launches must not emit."""
    expected_launches = expected_iterations * expected_execs + 1
    assert len(graph_launch_input_data) == expected_launches, (
        len(graph_launch_input_data),
        expected_launches,
    )


def test_graph_launch_per_exec_counts(
    graph_launch_input_data, expected_iterations, expected_execs
):
    """One exec gets `iterations` GRAPH_LAUNCH records; the other gets
    `iterations + 1`. Confirms failed-launch suppression at the per-exec level."""
    by_exec = Counter(int(r["Graph_Exec_Id"]) for r in graph_launch_input_data)
    assert len(by_exec) == expected_execs, by_exec
    counts = sorted(by_exec.values())
    assert counts == [expected_iterations, expected_iterations + 1], counts


def test_graph_launch_dispatch_count_matches(
    graph_launch_input_data, expected_kernel_nodes_per_launch
):
    """Each GRAPH_LAUNCH record reports Kernel_Dispatch_Count =
    expected_kernel_nodes_per_launch (the HSA queue interceptor counts every
    kernel AQL packet observed during the launch)."""
    for r in graph_launch_input_data:
        dc = int(r["Kernel_Dispatch_Count"])
        assert dc == expected_kernel_nodes_per_launch, (r, dc)


def test_graph_launch_timestamps_sane(graph_launch_input_data):
    """End_Timestamp must be >= Start_Timestamp."""
    for r in graph_launch_input_data:
        start = int(r["Start_Timestamp"])
        end = int(r["End_Timestamp"])
        assert end >= start, r
        assert start > 0, r


def test_graph_launch_agent_id_nonzero(graph_launch_input_data):
    """Agent_Id must be a real handle (resolved from the launch stream's
    device). A zero handle indicates resolve_launch_stream_agent failed and
    would crash downstream consumers."""
    for r in graph_launch_input_data:
        agent = r["Agent_Id"]
        assert agent and agent != "0", r


def test_graph_launch_exec_ids_match_kernel_csv(
    graph_launch_input_data, kernel_input_data, expected_execs
):
    """Set of Graph_Exec_Ids in GRAPH_LAUNCH records must equal the set
    observed on the kernel CSV's graph rows."""
    launch_exec_ids = {int(r["Graph_Exec_Id"]) for r in graph_launch_input_data}
    kernel_exec_ids = {
        int(r["Graph_Exec_Id"]) for r in _graph_kernel_rows(kernel_input_data)
    }
    assert launch_exec_ids == kernel_exec_ids, (launch_exec_ids, kernel_exec_ids)
    assert len(launch_exec_ids) == expected_execs


def test_graph_launch_correlation_unique(graph_launch_input_data):
    """Each GRAPH_LAUNCH record carries a distinct nonzero Correlation_Id."""
    corrs = [r["Correlation_Id"] for r in graph_launch_input_data]
    assert all(c not in ("0", "") for c in corrs)
    assert len(set(corrs)) == len(corrs), Counter(corrs).most_common(3)


def test_graph_launch_correlation_joins_to_hip_api(
    graph_launch_input_data, hip_api_input_data
):
    """Every GRAPH_LAUNCH Correlation_Id must join to a HIP_RUNTIME_API row
    named hipGraphLaunch or hipGraphLaunch_spt, so consumers can join the
    two CSVs on Correlation_Id."""
    api_by_corr = {}
    for r in hip_api_input_data:
        name = r.get("Function") or r.get("Name") or r.get("Operation")
        api_by_corr[r["Correlation_Id"]] = name
    accepted = {"hipGraphLaunch", "hipGraphLaunch_spt"}
    for gr in graph_launch_input_data:
        corr = gr["Correlation_Id"]
        assert corr in api_by_corr, (corr, list(api_by_corr.items())[:3])
        assert api_by_corr[corr] in accepted, (corr, api_by_corr[corr])


# ---------------------------------------------------------------------------
# Cross-stream consistency: HIP_STREAM and HIP_GRAPH attribution coexist
# ---------------------------------------------------------------------------


def test_graph_kernels_carry_stream_id(kernel_input_data):
    """Kernel dispatches from a graph launch should also carry a Stream_Id
    (the launch stream). This verifies HIP_STREAM and HIP_GRAPH attribution
    do not conflict when both are active."""
    rows = _graph_kernel_rows(kernel_input_data)
    if "Stream_Id" not in rows[0]:
        pytest.skip("Stream_Id column not present in this trace")
    stream_ids = {r["Stream_Id"] for r in rows}
    stream_ids.discard("")
    stream_ids.discard("0")
    assert stream_ids, "no nonzero Stream_Id values on graph kernel dispatches"


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
    """Graph-attributed memcpy count = (iterations * execs + 1) * memcpy_nodes.

    On AMD HIP this is typically 0 because in-graph memcpys are dispatched as
    blit kernels and surface as KERNEL_DISPATCH records. When the
    --expected-memcpy-nodes-per-launch parameter is 0, this still asserts the
    absence of graph-attributed memcpys."""
    rows = _memcpy_graph_rows(memory_copy_input_data)
    expected_launches = expected_iterations * expected_execs + 1
    expected = expected_launches * expected_memcpy_nodes_per_launch
    assert len(rows) == expected, (len(rows), expected)


def test_memcpy_non_graph_rows_render_empty(memory_copy_input_data):
    """Out-of-graph memcpys render the new columns as empty strings."""
    non_graph_rows = [
        r for r in memory_copy_input_data if _is_empty(r.get("Graph_Exec_Id"))
    ]
    if not non_graph_rows:
        pytest.skip("no non-graph memcpy rows in this trace")
    for r in non_graph_rows:
        assert r["Graph_Exec_Id"] == "", r
        assert r["Graph_Node_Id"] == "", r


def test_memcpy_exec_ids_subset_of_kernel_csv(
    memory_copy_input_data, kernel_input_data
):
    """When graph-attributed memcpy rows exist, their Graph_Exec_Id set must
    be a subset of the kernel CSV's set (both must reference real execs)."""
    memcpy_rows = _memcpy_graph_rows(memory_copy_input_data)
    if not memcpy_rows:
        pytest.skip("no graph-attributed memcpys (AMD HIP blit-kernel path)")
    memcpy_ids = {int(r["Graph_Exec_Id"]) for r in memcpy_rows}
    kernel_ids = {
        int(r["Graph_Exec_Id"]) for r in _graph_kernel_rows(kernel_input_data)
    }
    assert memcpy_ids.issubset(kernel_ids), (memcpy_ids, kernel_ids)


# ---------------------------------------------------------------------------
# Full-topology view (kernel + memcpy union)
# ---------------------------------------------------------------------------


def test_node_id_range_full_topology(
    kernel_input_data,
    request,
    expected_nodes_per_launch,
    expected_iterations,
    expected_execs,
):
    """Union of kernel + memcpy graph_node_ids per launch must equal
    {0..nodes_per_launch-1}. The node counter is shared across all graph
    nodes — every external_corr_id request fired during the launch increments
    it, regardless of whether the resulting record surfaces as a
    KERNEL_DISPATCH or MEMORY_COPY."""
    by_corr = defaultdict(list)
    for r in _graph_kernel_rows(kernel_input_data):
        by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    memcpy_path = request.config.getoption("--memory-copy-input")
    if memcpy_path and os.path.exists(memcpy_path):
        for r in _memcpy_graph_rows(_read_csv(memcpy_path)):
            by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    expected_launches = expected_iterations * expected_execs + 1
    assert len(by_corr) == expected_launches, (len(by_corr), expected_launches)
    for corr, nodes in by_corr.items():
        assert sorted(nodes) == list(range(expected_nodes_per_launch)), (corr, sorted(nodes))


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
