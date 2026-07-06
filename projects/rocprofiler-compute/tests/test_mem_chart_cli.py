# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""CLI integration tests for the Memory Chart panel."""

import common
import pytest

config = {"cleanup": True}

indirs = [
    "tests/workloads/vcopy/MI100",
    "tests/workloads/vcopy/MI200",
    "tests/workloads/vcopy/MI300A_A1",
    "tests/workloads/vcopy/MI300X_A1",
    "tests/workloads/vcopy/MI350",
    "tests/workloads/vcopy/RDNA35_HALO",
]


class TestMemChartCLI:
    """CLI integration tests for Memory Chart analysis."""

    def test_memory_chart_runs_for_supported_workloads(
        self,
        binary_handler_analyze_rocprof_compute,
    ):
        """Memory Chart analyze runs cleanly across supported workloads."""
        for workload_path in indirs:
            workload_dir = common.setup_workload_dir(workload_path)
            try:
                code = binary_handler_analyze_rocprof_compute([
                    "analyze",
                    "--path",
                    workload_dir,
                    "--block",
                    "3",
                ])
                assert code == 0, f"Memory Chart analyze failed for {workload_path}"
            finally:
                common.clean_output_dir(config["cleanup"], workload_dir)

    @pytest.mark.parametrize(
        ("workload_path", "workload_id", "expected_labels"),
        [
            pytest.param(
                "tests/workloads/vcopy/MI300X_A1",
                "MI300X_A1",
                [
                    "LDS",
                    "Vector L1 Cache",
                    "Scalar L1D Cache",
                    "Instr L1 Cache",
                    "L2 Cache",
                    "Fabric",
                    "HBM",
                ],
                id="MI300X_A1",
            ),
            pytest.param(
                "tests/workloads/vcopy/RDNA35_HALO",
                "RDNA35_HALO",
                [
                    "LDS",
                    "GL0 (TCP Cache)",
                    "GL1 Cache",
                    "GL2 Cache",
                    "SQC",
                    "GCEA",
                    "DRAM",
                ],
                id="RDNA35_HALO",
            ),
        ],
    )
    def test_memory_chart_output_includes_memory_hierarchy(
        self,
        binary_handler_analyze_rocprof_compute,
        capsys,
        workload_path,
        workload_id,
        expected_labels,
    ):
        """Memory Chart output contains expected memory-hierarchy components."""
        workload_dir = common.setup_workload_dir(
            workload_path,
            param_id=workload_id,
        )
        try:
            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "3",
            ])
            assert code == 0

            captured = capsys.readouterr()
            output = common.strip_ansi(captured.out)

            for expected_label in expected_labels:
                assert expected_label in output, (
                    f"Memory Chart output for {workload_path} should include "
                    f"{expected_label}"
                )
        finally:
            common.clean_output_dir(config["cleanup"], workload_dir)

    @pytest.mark.parametrize(
        "normal_unit",
        ["per_wave", "per_cycle", "per_second", "per_kernel"],
    )
    def test_memory_chart_accepts_normal_unit(
        self,
        binary_handler_analyze_rocprof_compute,
        normal_unit,
    ):
        """Memory Chart runs cleanly under different normalization units."""
        workload_dir = common.setup_workload_dir(
            "tests/workloads/vcopy/MI350",
            param_id=normal_unit,
        )
        try:
            code = binary_handler_analyze_rocprof_compute([
                "analyze",
                "--path",
                workload_dir,
                "--block",
                "3",
                "--normal-unit",
                normal_unit,
            ])
            assert code == 0, (
                f"Memory Chart analyze failed with --normal-unit {normal_unit}"
            )
        finally:
            common.clean_output_dir(config["cleanup"], workload_dir)
