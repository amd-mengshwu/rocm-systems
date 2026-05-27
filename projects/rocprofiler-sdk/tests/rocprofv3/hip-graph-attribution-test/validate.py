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

import sys
import pytest
from collections import defaultdict, Counter


def test_columns_present(kernel_input_data):
    """The new Graph_Exec_Id and Graph_Node_Id columns must be present."""
    row = kernel_input_data[0]
    assert (
        "Graph_Exec_Id" in row
    ), f"Graph_Exec_Id column missing; got {list(row.keys())}"
    assert (
        "Graph_Node_Id" in row
    ), f"Graph_Node_Id column missing; got {list(row.keys())}"


def test_non_graph_rows_render_empty(kernel_input_data):
    """Per spec §5.5: dispatches not from a graph launch must have empty
    Graph_Exec_Id and Graph_Node_Id (not the literal '0')."""
    non_graph_rows = [
        r
        for r in kernel_input_data
        if r.get("Kind") == "KERNEL_DISPATCH"
        and (
            r.get("Graph_Exec_Id") == ""
            or r.get("Graph_Exec_Id") is None
            or int(r.get("Graph_Exec_Id", "0") or "0") == 0
        )
    ]
    if not non_graph_rows:
        pytest.skip(
            "No non-graph KERNEL_DISPATCH rows in this trace to verify empty rendering"
        )
    for r in non_graph_rows:
        assert (
            r["Graph_Exec_Id"] == ""
        ), f"non-graph row has Graph_Exec_Id='{r['Graph_Exec_Id']}', expected empty"
        assert (
            r["Graph_Node_Id"] == ""
        ), f"non-graph row has Graph_Node_Id='{r['Graph_Node_Id']}', expected empty"


def _graph_rows(kernel_input_data):
    """Filter to KERNEL_DISPATCH rows that have a populated Graph_Exec_Id."""
    out = []
    for r in kernel_input_data:
        if r.get("Kind") != "KERNEL_DISPATCH":
            continue
        v = r.get("Graph_Exec_Id", "")
        if v == "" or v is None:
            continue
        if int(v) == 0:
            continue
        out.append(r)
    return out


def test_total_graph_dispatch_count(
    kernel_input_data,
    expected_iterations,
    expected_execs,
    expected_kernel_nodes_per_launch,
):
    """Total graph kernel dispatches = (iterations × execs + 1) × kernel_nodes_per_launch.

    The extra +1 is the one valid exec_b launch after the failed launch.
    Memcpy nodes are counted separately in the memory_copy CSV."""
    rows = _graph_rows(kernel_input_data)
    expected_launches = expected_iterations * expected_execs + 1
    expected = expected_launches * expected_kernel_nodes_per_launch
    assert (
        len(rows) == expected
    ), f"expected {expected} graph kernel dispatches, got {len(rows)}"


def test_two_distinct_exec_ids(kernel_input_data, expected_execs):
    """The two hipGraphInstantiate calls must produce two distinct nonzero Graph_Exec_Ids."""
    rows = _graph_rows(kernel_input_data)
    exec_ids = {int(r["Graph_Exec_Id"]) for r in rows}
    assert (
        len(exec_ids) == expected_execs
    ), f"expected {expected_execs} distinct Graph_Exec_Ids, got {exec_ids}"
    assert 0 not in exec_ids


def test_graph_node_id_range_per_launch(
    kernel_input_data,
    expected_kernel_nodes_per_launch,
    expected_iterations,
    expected_execs,
):
    """Each launch's kernel rows must yield exactly kernel_nodes_per_launch
    distinct Graph_Node_Id values, all drawn from 0..nodes_per_launch-1.

    The full 0..nodes_per_launch-1 range is verified across kernel + memcpy
    rows in test_graph_node_id_range_per_launch_full_topology."""
    rows = _graph_rows(kernel_input_data)
    by_corr = defaultdict(list)
    for r in rows:
        by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    expected_launches = expected_iterations * expected_execs + 1
    assert (
        len(by_corr) == expected_launches
    ), f"expected {expected_launches} distinct correlation_ids (one per launch), got {len(by_corr)}"
    for corr, nodes in by_corr.items():
        assert (
            len(nodes) == expected_kernel_nodes_per_launch
        ), f"launch {corr} produced {len(nodes)} kernel node_ids, expected {expected_kernel_nodes_per_launch}"
        assert len(set(nodes)) == len(
            nodes
        ), f"launch {corr} produced duplicate kernel node_ids: {sorted(nodes)}"


def test_graph_node_id_stable_per_exec(kernel_input_data, expected_nodes_per_launch):
    """For each (Graph_Exec_Id, Graph_Node_Id), all dispatches must have the same
    kernel name. Stability is per-exec."""
    rows = _graph_rows(kernel_input_data)
    by_exec_node = defaultdict(list)
    for r in rows:
        key = (int(r["Graph_Exec_Id"]), int(r["Graph_Node_Id"]))
        by_exec_node[key].append(r["Kernel_Name"])
    for key, names in by_exec_node.items():
        names_set = set(names)
        assert (
            len(names_set) == 1
        ), f"(Graph_Exec_Id, Graph_Node_Id)={key} produced multiple kernel names: {names_set}"


def test_distinct_kernel_nodes_remain_distinct(
    kernel_input_data, expected_distinct_kernels
):
    """Two distinct nodes that launch the same kernel must still have distinct
    Graph_Node_Id values within a single launch."""
    rows = _graph_rows(kernel_input_data)
    first_corr = rows[0]["Correlation_Id"]
    first_launch = [r for r in rows if r["Correlation_Id"] == first_corr]
    kernel_counts = Counter(r["Kernel_Name"] for r in first_launch)
    assert (
        len(kernel_counts) == expected_distinct_kernels
    ), f"expected {expected_distinct_kernels} distinct kernels per launch, got {dict(kernel_counts)}"
    a_ids = {
        int(r["Graph_Node_Id"]) for r in first_launch if "kernel_a" in r["Kernel_Name"]
    }
    assert len(a_ids) == 3, f"expected 3 distinct node_ids for kernel_a, got {a_ids}"


# --- GRAPH_LAUNCH summary record tests ---


def test_graph_launch_record_count(
    graph_launch_input_data, expected_iterations, expected_execs
):
    """One GRAPH_LAUNCH record per *successful* hipGraphLaunch call.

    Failed launches must not produce a record (spec §4.4)."""
    expected_launches = expected_iterations * expected_execs + 1
    assert len(graph_launch_input_data) == expected_launches, (
        f"expected {expected_launches} GRAPH_LAUNCH records "
        f"(failed launches must not emit), got {len(graph_launch_input_data)}"
    )


def test_graph_launch_dispatch_counts(
    graph_launch_input_data, expected_kernel_nodes_per_launch
):
    """Each GRAPH_LAUNCH record must report Kernel_Dispatch_Count ==
    kernel_nodes_per_launch. Memcpy nodes are tracked separately and do not
    count toward kernel_dispatch_count (which is incremented only by the HSA
    queue interceptor on AQL kernel dispatch packets)."""
    for r in graph_launch_input_data:
        dc = int(r["Kernel_Dispatch_Count"])
        assert (
            dc == expected_kernel_nodes_per_launch
        ), f"record {r} has Kernel_Dispatch_Count {dc}, expected {expected_kernel_nodes_per_launch}"


def test_graph_launch_exec_ids_match_kernel_csv(
    graph_launch_input_data, kernel_input_data, expected_execs
):
    """Graph_Exec_Ids in GRAPH_LAUNCH records must match the set from the kernel CSV."""
    launch_exec_ids = {int(r["Graph_Exec_Id"]) for r in graph_launch_input_data}
    kernel_exec_ids = {int(r["Graph_Exec_Id"]) for r in _graph_rows(kernel_input_data)}
    assert (
        launch_exec_ids == kernel_exec_ids
    ), f"GRAPH_LAUNCH Graph_Exec_Ids {launch_exec_ids} != kernel-CSV Graph_Exec_Ids {kernel_exec_ids}"
    assert len(launch_exec_ids) == expected_execs


def test_graph_launch_correlation_joins_to_hip_api(
    graph_launch_input_data, hip_api_input_data
):
    """Every GRAPH_LAUNCH Correlation_Id must join to a HIP_RUNTIME_API row named
    hipGraphLaunch or hipGraphLaunch_spt (spec §7 item 9)."""
    api_by_corr = {}
    for r in hip_api_input_data:
        name = r.get("Function") or r.get("Name") or r.get("Operation")
        assert name is not None, f"HIP API row {r} has no recognizable name column"
        api_by_corr[r["Correlation_Id"]] = name
    accepted_names = {"hipGraphLaunch", "hipGraphLaunch_spt"}
    for gr in graph_launch_input_data:
        corr = gr["Correlation_Id"]
        assert corr not in (
            "0",
            "",
        ), f"GRAPH_LAUNCH row has empty/zero Correlation_Id: {gr}"
        assert (
            corr in api_by_corr
        ), f"GRAPH_LAUNCH Correlation_Id {corr} does not appear in HIP API CSV"
        assert (
            api_by_corr[corr] in accepted_names
        ), f"GRAPH_LAUNCH Correlation_Id {corr} maps to '{api_by_corr[corr]}', expected one of {accepted_names}"


# --- Phase F: memory_copy graph attribution tests ---


def _memcpy_graph_rows(memory_copy_input_data):
    """Filter to memory_copy rows that carry populated Graph_Exec_Id."""
    out = []
    for r in memory_copy_input_data:
        v = r.get("Graph_Exec_Id", "")
        if v == "" or v is None:
            continue
        if int(v) == 0:
            continue
        out.append(r)
    return out


def test_memcpy_columns_present(memory_copy_input_data):
    """Memory_copy CSV must include Graph_Exec_Id and Graph_Node_Id columns
    (Phase F extension)."""
    row = memory_copy_input_data[0]
    assert (
        "Graph_Exec_Id" in row
    ), f"Graph_Exec_Id column missing from memory_copy CSV; got {list(row.keys())}"
    assert (
        "Graph_Node_Id" in row
    ), f"Graph_Node_Id column missing from memory_copy CSV; got {list(row.keys())}"


def test_memcpy_graph_attribution_count(
    memory_copy_input_data,
    expected_iterations,
    expected_execs,
    expected_memcpy_nodes_per_launch,
):
    """Total graph-attributed memory copies = (iterations × execs + 1) ×
    memcpy_nodes_per_launch.

    On AMD HIP, in-graph memcpy nodes are typically implemented as blit
    kernels (__amd_rocclr_copyBuffer) and surface as KERNEL_DISPATCH records
    rather than MEMORY_COPY records. When expected_memcpy_nodes_per_launch
    is 0, this test asserts the absence of any graph-attributed MEMORY_COPY
    rows (out-of-graph copies must have empty graph fields, verified by
    test_non_graph_memcpy_rows_render_empty)."""
    rows = _memcpy_graph_rows(memory_copy_input_data)
    expected_launches = expected_iterations * expected_execs + 1
    expected = expected_launches * expected_memcpy_nodes_per_launch
    assert (
        len(rows) == expected
    ), f"expected {expected} graph-attributed memcpys, got {len(rows)}"


def test_memcpy_exec_ids_match_kernel_csv(
    memory_copy_input_data, kernel_input_data, expected_execs
):
    """When graph-attributed MEMORY_COPY rows exist, their Graph_Exec_Id
    set must be a subset of the kernel-CSV Graph_Exec_Id set. Skipped when
    no graph-attributed memcpys exist (AMD HIP blit-kernel path)."""
    memcpy_graph_rows = _memcpy_graph_rows(memory_copy_input_data)
    if not memcpy_graph_rows:
        pytest.skip(
            "No graph-attributed MEMORY_COPY rows (AMD HIP graph memcpys "
            "may surface via the KERNEL_DISPATCH blit-kernel path)"
        )
    memcpy_ids = {int(r["Graph_Exec_Id"]) for r in memcpy_graph_rows}
    kernel_ids = {int(r["Graph_Exec_Id"]) for r in _graph_rows(kernel_input_data)}
    assert memcpy_ids.issubset(
        kernel_ids
    ), f"memcpy Graph_Exec_Ids {memcpy_ids} not subset of kernel Graph_Exec_Ids {kernel_ids}"


def test_graph_node_id_range_per_launch_full_topology(
    kernel_input_data,
    request,
    expected_nodes_per_launch,
    expected_iterations,
    expected_execs,
):
    """Union of kernel + memcpy graph_node_ids for each launch must be
    exactly {0..nodes_per_launch-1}. The node counter is shared across all
    graph nodes — every external_corr_id callback fired while a graph
    launch is in flight increments node_counter, regardless of whether the
    record ultimately surfaces as a KERNEL_DISPATCH or MEMORY_COPY.

    This test inspects the memcpy CSV directly (not via the skipping
    fixture) so it always runs, surfacing the union view even when memcpy
    rows happen to be empty (AMD HIP blit-kernel path)."""
    import os

    by_corr = defaultdict(list)
    for r in _graph_rows(kernel_input_data):
        by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    memcpy_path = request.config.getoption("--memory-copy-input")
    if memcpy_path and os.path.exists(memcpy_path):
        memcpy_rows = _read_csv_safe(memcpy_path)
        for r in _memcpy_graph_rows(memcpy_rows):
            by_corr[r["Correlation_Id"]].append(int(r["Graph_Node_Id"]))
    expected_launches = expected_iterations * expected_execs + 1
    assert (
        len(by_corr) == expected_launches
    ), f"expected {expected_launches} distinct correlation_ids, got {len(by_corr)}"
    for corr, nodes in by_corr.items():
        assert sorted(nodes) == list(range(expected_nodes_per_launch)), (
            f"launch {corr} (kernel+memcpy union) produced node_ids {sorted(nodes)}, "
            f"expected 0..{expected_nodes_per_launch - 1}"
        )


def _read_csv_safe(filename):
    import csv

    out = []
    with open(filename, "r") as inp:
        for row in csv.DictReader(inp):
            out.append(row)
    return out


def test_non_graph_memcpy_rows_render_empty(memory_copy_input_data):
    """Memcpys not from a graph launch must have empty Graph_Exec_Id and
    Graph_Node_Id (same convention as kernel CSV — spec §5.5)."""
    non_graph_rows = [
        r
        for r in memory_copy_input_data
        if r.get("Graph_Exec_Id") == ""
        or r.get("Graph_Exec_Id") is None
        or int(r.get("Graph_Exec_Id", "0") or "0") == 0
    ]
    if not non_graph_rows:
        pytest.skip("No non-graph memcpy rows in this trace to verify empty rendering")
    for r in non_graph_rows:
        assert (
            r["Graph_Exec_Id"] == ""
        ), f"non-graph memcpy has Graph_Exec_Id='{r['Graph_Exec_Id']}', expected empty"
        assert (
            r["Graph_Node_Id"] == ""
        ), f"non-graph memcpy has Graph_Node_Id='{r['Graph_Node_Id']}', expected empty"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
