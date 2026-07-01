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

import json
import pytest
import csv
import yaml

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption(
        "--agent-input",
        action="store",
        help="Path to agent info CSV file.",
    )
    parser.addoption(
        "--counter-input",
        action="store",
        help="Path to counter collection CSV file.",
    )
    parser.addoption(
        "--multiplex-input",
        action="store",
        help="Path to the JSON/YAML input file describing the multiplex layout "
        "(pmc_groups and pmc_group_interval) used for the run.",
    )


@pytest.fixture
def agent_info_input_data(request):
    filename = request.config.getoption("--agent-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def counter_input_data(request):
    filename = request.config.getoption("--counter-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def multiplex_layout(request):
    # Read the multiplex layout straight from the input file used for the run so
    # the validator always stays in sync with the layout under test (number of
    # groups, their counters and the pmc_group_interval), regardless of whether
    # the run was driven by JSON or YAML input.
    filename = request.config.getoption("--multiplex-input")
    with open(filename, "r") as inp:
        if filename.endswith((".yml", ".yaml")):
            config = yaml.safe_load(inp)
        else:
            config = json.load(inp)

    job = config["jobs"][0]
    pmc_groups = [list(group) for group in job["pmc_groups"]]
    pmc_group_interval = int(job.get("pmc_group_interval", 1))

    return pmc_groups, pmc_group_interval
