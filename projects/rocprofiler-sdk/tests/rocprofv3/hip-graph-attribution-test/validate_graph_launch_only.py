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


def test_graph_launch_record_count(
    graph_launch_input_data, expected_iterations, expected_execs
):
    """One GRAPH_LAUNCH record per *successful* hipGraphLaunch call."""
    expected_launches = expected_iterations * expected_execs + 1
    assert (
        len(graph_launch_input_data) == expected_launches
    ), f"expected {expected_launches} GRAPH_LAUNCH records, got {len(graph_launch_input_data)}"


def test_graph_launch_kernel_dispatch_count(
    graph_launch_input_data, expected_nodes_per_launch
):
    """SPEC §4.2 subscription independence: when only GRAPH_LAUNCH is subscribed,
    each summary record must still report the correct Kernel_Dispatch_Count.

    This is the critical test that proves the WriteInterceptor counter increments
    independently of kernel-dispatch subscription state."""
    for r in graph_launch_input_data:
        dc = int(r["Kernel_Dispatch_Count"])
        assert dc == expected_nodes_per_launch, (
            f"GRAPH_LAUNCH-only mode: record {r} has Kernel_Dispatch_Count {dc}, "
            f"expected {expected_nodes_per_launch}. WriteInterceptor likely was "
            f"not activated for GRAPH_LAUNCH-only subscription (see Task 10's "
            f"context_filter and early-return extensions)."
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
