#!/usr/bin/env python3
# Copyright (C) Advanced Micro Devices. All rights reserved.
"""Verify the Device Metrics Exporter Prometheus endpoint.

Replaces the inline curl/grep loop in ``dme-amdsmi-ci.yml`` Phase 5 with
a real Prometheus exposition-format check that asserts:

* Endpoint returns HTTP 200 within ``--max-retries`` attempts.
* Response is valid Prometheus text (``# HELP`` + ``# TYPE`` headers).
* Every metric in ``--required-metric`` is exposed and has at least one
  numeric sample emitted.

The default required-metric list covers the AMDSMI-sourced GPU metrics
that gate this integration test -- the previous bash check passed even
when DME exported no GPU metrics at all.
"""

import argparse
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from ._common import configure_logging, gh_error, gh_warning

logger = logging.getLogger("dme.metrics")

_HELP_RE = re.compile(r"^# HELP ", re.MULTILINE)
_TYPE_RE = re.compile(r"^# TYPE ", re.MULTILINE)
# Prometheus sample lines: ``metric{labels...} <number>`` (with optional timestamp).
# Captures metric name in group 1.
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[0-9eE+\-.]+|NaN|\+Inf|-Inf)"
    r"(?:\s+\d+)?\s*$",
    re.MULTILINE,
)

# Default GPU metrics that should always be exposed by the AMDSMI-backed
# Device Metrics Exporter once GPU Agent is up. Override via CLI flag.
_DEFAULT_REQUIRED_METRICS = (
    "gpu_edge_temperature",
    "gpu_power_usage",
    "gpu_gfx_activity",
)


def _fetch(url: str, timeout: float) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, ""


def _assert_prometheus_format(body: str) -> None:
    if not _HELP_RE.search(body) or not _TYPE_RE.search(body):
        raise AssertionError("Response is missing # HELP / # TYPE headers")


def _exposed_metric_names(body: str) -> set[str]:
    return {m.group("name") for m in _SAMPLE_RE.finditer(body)}


def verify(
    *,
    url: str,
    required_metrics: tuple[str, ...],
    max_retries: int,
    retry_delay: float,
    request_timeout: float,
    output_path: Path | None = None,
) -> None:
    body = ""
    last_status = 0
    for attempt in range(1, max_retries + 1):
        logger.info("attempt %d/%d: GET %s", attempt, max_retries, url)
        status, body = _fetch(url, timeout=request_timeout)
        last_status = status
        if status == 200 and body:
            break
        logger.info("HTTP %s -- retrying in %.1fs", status, retry_delay)
        time.sleep(retry_delay)
    else:
        gh_error(f"Metrics endpoint unreachable after {max_retries} attempts (last status {last_status})")
        raise SystemExit(1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(body)
        logger.info("saved metrics output to %s", output_path)

    _assert_prometheus_format(body)
    logger.info("Prometheus exposition format OK")

    exposed = _exposed_metric_names(body)
    missing = [m for m in required_metrics if m not in exposed]
    if missing:
        gh_error(
            "Required GPU metrics missing from /metrics: " + ", ".join(missing)
        )
        sample = ", ".join(sorted(exposed)[:20])
        gh_warning(f"Exposed metrics (sample): {sample}")
        raise SystemExit(1)

    logger.info(
        "All %d required metrics present (total exposed: %d)",
        len(required_metrics),
        len(exposed),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--required-metric",
        action="append",
        default=None,
        help="Metric name that must appear (repeatable). Defaults to GPU metric set.",
    )
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    required = tuple(args.required_metric) if args.required_metric else _DEFAULT_REQUIRED_METRICS
    verify(
        url=args.url,
        required_metrics=required,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_timeout=args.request_timeout,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
