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
import json
import pytest

from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.dotdict import dotdict


def pytest_addoption(parser):
    parser.addoption("--kernel-input", action="store", default=None)
    parser.addoption("--graph-launch-input", action="store", default=None)
    parser.addoption("--hip-api-input", action="store", default=None)
    parser.addoption("--json-input", action="store", default=None)
    parser.addoption("--expected-iterations", action="store", type=int, default=None)
    parser.addoption("--expected-execs", action="store", type=int, default=None)
    parser.addoption(
        "--expected-nodes-per-launch", action="store", type=int, default=None
    )
    parser.addoption(
        "--expected-distinct-kernels", action="store", type=int, default=None
    )


def _read_csv(filename):
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)
    return data


@pytest.fixture
def kernel_input_data(request):
    filename = request.config.getoption("--kernel-input")
    if filename is None:
        pytest.fail("--kernel-input argument is required")
    data = _read_csv(filename)
    assert len(data) > 0, f"CSV file '{filename}' contained no data rows"
    return data


@pytest.fixture
def graph_launch_input_data(request):
    filename = request.config.getoption("--graph-launch-input")
    if filename is None:
        pytest.fail("--graph-launch-input argument is required")
    data = _read_csv(filename)
    assert len(data) > 0, f"CSV file '{filename}' contained no data rows"
    return data


@pytest.fixture
def hip_api_input_data(request):
    filename = request.config.getoption("--hip-api-input")
    if filename is None:
        pytest.fail("--hip-api-input argument is required")
    return _read_csv(filename)


@pytest.fixture
def json_input_data(request):
    filename = request.config.getoption("--json-input")
    if filename is None:
        pytest.fail("--json-input argument is required")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def expected_iterations(request):
    return request.config.getoption("--expected-iterations")


@pytest.fixture
def expected_execs(request):
    return request.config.getoption("--expected-execs")


@pytest.fixture
def expected_nodes_per_launch(request):
    return request.config.getoption("--expected-nodes-per-launch")


@pytest.fixture
def expected_distinct_kernels(request):
    return request.config.getoption("--expected-distinct-kernels")
