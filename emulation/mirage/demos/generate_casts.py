#!/usr/bin/env python3
"""Generate deterministic asciinema casts for `mirage debug`.

The casts are *reproducible*: every frame is derived from the actual
output of `mirage debug --demo`, which runs against the in-process mock
GPU and is fully deterministic. We run each demo's command list once in
`--batch` mode (the REPL echoes `(mirage-dbg) <cmd>` before each command's
output, so the captured transcript already reads like an interactive
session), then re-time it into an asciinema v2 cast with a typing
animation for the commands.

Usage:
    python3 demos/generate_casts.py [--bin PATH] [--out DIR]

Re-run after changing the debugger to refresh the checked-in casts. Play
one with:  asciinema play demos/01-attach-and-inspect.cast
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Each demo: (filename, title, [commands]). Commands are fed to the REPL
# in order; state (PC, registers, breakpoints) persists across them.
DEMOS: list[tuple[str, str, list[str]]] = [
    (
        "01-attach-and-inspect",
        "Attach and inspect wavefronts",
        [
            "info threads",
            "wave 1",
            "info wave",
            "info registers",
        ],
    ),
    (
        "02-stepping",
        "Single-stepping execution",
        [
            "status",
            "stepi",
            "stepi",
            "stepi 5",
            "info registers",
        ],
    ),
    (
        "03-breakpoints",
        "Breakpoints and continue",
        [
            "break 0x1010",
            "info breakpoints",
            "continue",
            "info registers",
        ],
    ),
    (
        "04-memory",
        "Examining and writing memory",
        [
            "x/16xb 0x100000",
            "x/8xw 0x100000",
            "mw 0x100000 de ad be ef",
            "x/8xb 0x100000",
        ],
    ),
    (
        "05-registers",
        "Reading and writing registers",
        [
            "print $s0",
            "set $s4 = 0xcafe",
            "print $s4",
            "info registers sgpr",
        ],
    ),
]

PROMPT = "(mirage-dbg) "

# Timing (seconds).
CHAR_DELAY = 0.035      # per typed character
SUBMIT_PAUSE = 0.30     # pause after pressing Enter, before output
OUTPUT_LINE_DELAY = 0.04
COMMAND_GAP = 0.6       # pause between commands
BANNER_PAUSE = 0.8


def capture_transcript(bin_path: Path, commands: list[str], env: dict) -> str:
    """Run the demo once in batch mode and return the REPL transcript."""
    args = [str(bin_path), "debug", "--demo", "--batch"]
    for c in commands:
        args += ["-x", c]
    result = subprocess.run(
        args, env=env, capture_output=True, text=True, check=True
    )
    return result.stdout


def split_transcript(transcript: str) -> list[tuple[str, list[str]]]:
    """Split a transcript into (command, output_lines) pairs.

    The REPL prints `(mirage-dbg) <cmd>` for each command followed by its
    output lines, so we partition on the prompt marker.
    """
    pairs: list[tuple[str, list[str]]] = []
    current_cmd: str | None = None
    current_out: list[str] = []
    for line in transcript.splitlines():
        if line.startswith(PROMPT):
            if current_cmd is not None:
                pairs.append((current_cmd, current_out))
            current_cmd = line[len(PROMPT):]
            current_out = []
        else:
            current_out.append(line)
    if current_cmd is not None:
        pairs.append((current_cmd, current_out))
    return pairs


def write_cast(path: Path, title: str, pairs: list[tuple[str, list[str]]],
               width: int, height: int) -> None:
    """Emit an asciinema v2 cast with a typing animation."""
    events: list[tuple[float, str, str]] = []
    t = 0.0

    def emit(s: str) -> None:
        events.append((round(t, 3), "o", s))

    # Header banner so the viewer knows what they're watching.
    emit(f"\x1b[1;36m# mirage debug — {title}\x1b[0m\r\n")
    t += BANNER_PAUSE
    emit("\x1b[2m# deterministic in-process demo (--demo)\x1b[0m\r\n\r\n")
    t += BANNER_PAUSE

    for cmd, out in pairs:
        # Prompt, then animate typing the command.
        emit(f"\x1b[32m{PROMPT}\x1b[0m")
        for ch in cmd:
            t += CHAR_DELAY
            emit(ch)
        t += SUBMIT_PAUSE
        emit("\r\n")
        # Command output.
        for line in out:
            t += OUTPUT_LINE_DELAY
            emit(line + "\r\n")
        t += COMMAND_GAP

    # Closing prompt + a quiet beat.
    emit(f"\x1b[32m{PROMPT}\x1b[0m")
    t += 0.4
    emit("quit\r\n")

    header = {
        "version": 2,
        "width": width,
        "height": height,
        "title": f"mirage debug — {title}",
        "env": {"SHELL": "/bin/sh", "TERM": "xterm-256color"},
    }
    with path.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for ev in events:
            f.write(json.dumps(list(ev)) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    default_bin = here.parent / "target" / "debug" / "mirage"
    parser.add_argument("--bin", type=Path, default=default_bin,
                        help="path to the mirage binary")
    parser.add_argument("--out", type=Path, default=here,
                        help="output directory for .cast files")
    parser.add_argument("--width", type=int, default=92)
    parser.add_argument("--height", type=int, default=28)
    args = parser.parse_args()

    if not args.bin.exists():
        print(f"error: mirage binary not found at {args.bin}\n"
              f"build it first: cargo build -p mirage", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    # Isolate the CLI from the user's real mirage state.
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ)
        env["XDG_CONFIG_HOME"] = str(Path(tmp) / "config")
        env["XDG_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        env["XDG_STATE_HOME"] = str(Path(tmp) / "state")
        env.pop("MIRAGE_LOG", None)

        for filename, title, commands in DEMOS:
            transcript = capture_transcript(args.bin, commands, env)
            pairs = split_transcript(transcript)
            out_path = args.out / f"{filename}.cast"
            write_cast(out_path, title, pairs, args.width, args.height)
            print(f"wrote {out_path}  ({len(pairs)} commands)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
