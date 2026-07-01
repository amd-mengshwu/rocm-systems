# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-multiplex
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# All multiplex tests below are marked UNSTABLE. The multiplexed counter-collection
# execute path intermittently hard-hangs the GPU on back-to-back runs (observed on gfx942,
# and reproducible even with the original 2-group tests), so this is a tool/driver issue
# independent of the test logic or the app being profiled. Because
# ROCPROFILER_DISABLE_UNSTABLE_CTESTS defaults ON, these tests only run when configured
# with -DROCPROFILER_DISABLE_UNSTABLE_CTESTS=OFF.
# TODO(AIROCVAL-62): drop UNSTABLE once the multiplex counter-collection GPU hang is
# resolved.

# JSON input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o
        out_json -f csv -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex
    UNSTABLE)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml input_3group.json input_3group.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input.json
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex
    UNSTABLE)

# YAML input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiple-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.yml
        -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o out_yaml -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiple-yaml
    UNSTABLE)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiple-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml input_3group.json input_3group.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input.yml
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiple-yaml
    UNSTABLE)

# 3-group multiplex layout tests. simple-transpose only issues two device dispatches,
# which is not enough to exercise a three-group layout, so reproducible-dispatch-count is
# used to issue a deterministic number of dispatches (9 kernels on a single synchronized
# stream) so that all three groups are guaranteed to be scheduled across the run. The
# interval of 2 also exercises the generalized `dispatch-index / interval` group
# derivation (not just a per-dispatch rotation).

# JSON 3-group input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex-3group
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_3group.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-json-cc -o out_3group_json -f csv --
        $<TARGET_FILE:reproducible-dispatch-count> 9 1 100 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex-3group
    UNSTABLE)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-3group
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml input_3group.json input_3group.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-json-cc/out_3group_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-json-cc/out_3group_json_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input_3group.json
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex-3group
    UNSTABLE)

# YAML 3-group input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex-3group-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_3group.yml -d
        ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-yaml-cc -o out_3group_yaml -f csv --
        $<TARGET_FILE:reproducible-dispatch-count> 9 1 100 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex-3group-yaml
    UNSTABLE)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-3group-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml input_3group.json input_3group.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-yaml-cc/out_3group_yaml_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-yaml-cc/out_3group_yaml_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input_3group.yml
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex-3group-yaml
    UNSTABLE)
