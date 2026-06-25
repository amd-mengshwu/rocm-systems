#!/usr/bin/env bash
# Record a live asciinema cast of the `mirage debug` full-screen TUI.
#
# The TUI reads key events straight from the controlling terminal, so it
# can't be fed scripted stdin like the REPL. Instead we drive it inside a
# detached tmux session and record that pane with asciinema, sending a
# scripted, deterministic sequence of keystrokes.
#
# Requires: asciinema, tmux, and a `--features tui` build of mirage.
#
#   cargo build -p mirage --features tui
#   demos/record-tui.sh
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bin="${MIRAGE_BIN:-$here/../target/debug/mirage}"
out="$here/06-tui.cast"
session="mirage-tui-demo"

if [[ ! -x "$bin" ]]; then
  echo "error: mirage binary not found at $bin" >&2
  echo "build it: cargo build -p mirage --features tui" >&2
  exit 1
fi
for tool in asciinema tmux; do
  command -v "$tool" >/dev/null || { echo "error: $tool not installed" >&2; exit 1; }
done

# Isolate from the user's real mirage state.
workdir="$(mktemp -d)"
trap 'tmux kill-session -t "$session" 2>/dev/null || true; rm -rf "$workdir"' EXIT
export XDG_CONFIG_HOME="$workdir/config"
export XDG_RUNTIME_DIR="$workdir/runtime"
export XDG_STATE_HOME="$workdir/state"

# Sequence of (keys, pause-seconds) driving the TUI. Keys map to the
# TUI's bindings: j/k select, s step, n step10, c continue, b breakpoint,
# ] pages memory, q quits.
drive() {
  sleep 1.5
  for step in "j:0.8" "j:0.8" "s:1.0" "s:1.0" "n:1.2" "k:0.8" "b:1.0" \
              "c:1.5" "]:0.8" "]:0.8" "k:0.8" "s:1.0" "q:0.5"; do
    key="${step%%:*}"; pause="${step##*:}"
    tmux send-keys -t "$session" "$key"
    sleep "$pause"
  done
}

# Start the TUI in a detached tmux session, then record the live pane.
tmux new-session -d -s "$session" -x 92 -y 28 \
  "'$bin' debug --demo --tui"
sleep 0.5

asciinema rec --overwrite \
  --title "mirage debug — full-screen TUI" \
  --command "tmux attach -t '$session'" \
  "$out" &
rec_pid=$!

drive
wait "$rec_pid" 2>/dev/null || true

echo "wrote $out"
