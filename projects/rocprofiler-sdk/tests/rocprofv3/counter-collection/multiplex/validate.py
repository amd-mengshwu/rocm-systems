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

import pandas as pd
import os
import sys
import pytest


def test_agent_info(agent_info_input_data):
    logical_node_id = max([int(itr["Logical_Node_Id"]) for itr in agent_info_input_data])

    assert logical_node_id + 1 == len(agent_info_input_data)

    for row in agent_info_input_data:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
            assert int(row["Max_Waves_Per_Simd"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0
            assert int(row["Max_Waves_Per_Simd"]) > 0


def expected_group_index(dispatch_id, pmc_group_interval, num_groups):
    # The interval increments per dispatch on the device (i.e. Dispatch_Id) and
    # once it is reached the next pmc_group is selected, wrapping around after
    # the last group. Deriving the expected group from the dispatch index and
    # the interval generalizes the layout to any number of groups (the previous
    # hard-coded `(dispatch_id - 1) % 2` is just this formula for two groups and
    # an interval of one).
    return ((dispatch_id - 1) // pmc_group_interval) % num_groups


def test_counter_collection_multiplex(counter_input_data, multiplex_layout):
    pmc_groups, pmc_group_interval = multiplex_layout
    num_groups = len(pmc_groups)

    group_counters = [set(group) for group in pmc_groups]
    all_counters = set().union(*group_counters)

    di_list = []
    dispatch_counters = {}

    for row in counter_input_data:
        assert int(row["Queue_Id"]) > 0
        assert int(row["Process_Id"]) > 0
        assert len(row["Kernel_Name"]) > 0

        assert len(row["Counter_Value"]) > 0
        assert row["Counter_Name"] in all_counters
        assert float(row["Counter_Value"]) > 0

        dispatch_id = int(row["Dispatch_Id"])
        di_list.append(dispatch_id)
        dispatch_counters.setdefault(dispatch_id, set()).add(row["Counter_Name"])

    assert len(dispatch_counters) > 0, "no counter collection data was produced"

    # track which counters each defined group actually collected across the run
    observed_group_counters = [set() for _ in pmc_groups]

    for dispatch_id, seen_counters in dispatch_counters.items():
        group_id = expected_group_index(dispatch_id, pmc_group_interval, num_groups)
        expected_counters = group_counters[group_id]

        # Every dispatch must map to exactly one group AND collect that whole
        # group in a single pass. The multiplexing contract is that when a
        # pmc_group is scheduled for a dispatch, all of its counters are
        # collected together, so the set of counters seen for the dispatch must
        # equal (not merely be a subset of) the scheduled group. Equality both
        # rejects counters leaking in from another group and catches a group
        # that is only partially collected. This is safe because rows for a
        # dispatch are aggregated into `seen_counters` before comparing (a group
        # may emit one row per counter), and because zero-valued counter rows
        # are never dropped (see the `Counter_Value > 0` assertion above). For
        # the 2-counter group [SQ_WAVES, GRBM_COUNT] this means both counters
        # must appear for each dispatch scheduled to that group.
        assert seen_counters, f"dispatch {dispatch_id} collected no counters"
        assert seen_counters == expected_counters, (
            f"dispatch {dispatch_id} maps to group {group_id} "
            f"({sorted(expected_counters)}) but collected {sorted(seen_counters)}"
        )

        observed_group_counters[group_id] |= seen_counters

    # every defined group must appear across the run and every counter packed
    # into a group must be collected by that group at least once.
    for group_id, expected_counters in enumerate(group_counters):
        assert observed_group_counters[group_id] == expected_counters, (
            f"group {group_id} ({sorted(expected_counters)}) was not fully "
            f"collected, saw {sorted(observed_group_counters[group_id])}"
        )

    # make sure the dispatch ids are unique and ordered
    di_list = list(dict.fromkeys(di_list))
    di_expect = [idx + 1 for idx in range(len(di_list))]
    assert di_expect == di_list


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
