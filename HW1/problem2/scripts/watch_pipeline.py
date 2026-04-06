#!/usr/bin/env python3
"""Live progress monitor for the prob2 parallel training pipeline.

Open a second tmux pane and run:
    cd /home/jimmy/quantum_computing
    python3 HW1/problem2/scripts/watch_pipeline.py
"""
from __future__ import annotations
import re, sys, time
from pathlib import Path

LOG_DIR = Path("HW1/problem2/hf_space/runtime/logs")
REFRESH = 1.0

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_TQDM = re.compile(r"\d+%\|")
_BEST = re.compile(r"best=([0-9.]+)")
_ACC  = re.compile(r"avg_acc=([0-9.]+)")
_STEP = re.compile(r"step=(\d+)")


def strip(s: str) -> str:
    return _ANSI.sub("", s)


def last_progress(log: Path) -> tuple[str, str]:
    """Return (status_line, state) where state is 'wait'|'run'|'done'|'err'."""
    if not log.exists() or log.stat().st_size == 0:
        return "waiting…", "wait"
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return "reading…", "wait"

    lines = [l for l in text.splitlines() if strip(l).strip()]

    for line in reversed(lines):
        if "Traceback" in line or "Error:" in line or "ERROR" in line:
            return f"ERROR — {strip(line)[:60]}", "err"

    for line in reversed(lines):
        if "Saved artifact" in line:
            # grab best accuracy from tqdm postfix if present
            extra = ""
            for l in reversed(lines):
                m = _BEST.search(l)
                if m:
                    extra = f"  best={m.group(1)}"
                    break
            return f"done{extra}", "done"

    # last tqdm line
    for line in reversed(lines):
        clean = strip(line).strip()
        if _TQDM.search(clean):
            # condense: keep epoch bar + postfix only
            clean = re.sub(r"\s+", " ", clean)
            return clean[:74], "run"

    return strip(lines[-1]).strip()[:74] if lines else "starting…", "run"


# ── colour helpers ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
WHITE  = "\033[97m"

STATE_STYLE = {
    "wait": (DIM,   "·"),
    "run":  (CYAN,  "▸"),
    "done": (GREEN, "✓"),
    "err":  (RED,   "✗"),
}

BAR_WIDTH = 20

def make_bar(clean_status: str) -> str:
    """Try to draw a progress bar from a tqdm line like '40%|██░░ 20/50'."""
    m = re.search(r"(\d+)%\|", clean_status)
    if not m:
        return ""
    pct = int(m.group(1))
    filled = round(BAR_WIDTH * pct / 100)
    bar = "█" * filled + "░" * (BAR_WIDTH - filled)
    return f"[{bar}] {pct:3d}%"


def draw(prev_n: int) -> int:
    if not LOG_DIR.exists():
        rows = [f"{DIM}Waiting for log directory…{RESET}"]
    else:
        logs = sorted(f for f in LOG_DIR.glob("*.log"))
        if not logs:
            rows = [f"{DIM}No log files yet…{RESET}"]
        else:
            rows = [f"{BOLD}Prob2 pipeline{RESET}  {DIM}{time.strftime('%H:%M:%S')}{RESET}"]

            # Group: run_name → [(method, log)]
            groups: dict[str, list[tuple[str, Path]]] = {}
            for log in logs:
                stem = log.stem
                parts = stem.rsplit("-", 1)
                run, method = (parts[0], parts[1]) if len(parts) == 2 else (stem, "?")
                groups.setdefault(run, []).append((method, log))

            for run, entries in groups.items():
                rows.append(f"  {YELLOW}{run}{RESET}")
                for method, log in sorted(entries):
                    status, state = last_progress(log)
                    color, icon = STATE_STYLE.get(state, (DIM, "·"))
                    bar = make_bar(status) if state == "run" else ""

                    # For running jobs show bar + postfix; for done show done msg
                    if bar:
                        # extract postfix (everything after the numbers/bar part)
                        postfix_m = re.search(r"(\d+/\d+\s+.+)", status)
                        postfix = postfix_m.group(1) if postfix_m else ""
                        # keep only the key=value pairs
                        kv = re.findall(r"(?:step|avg_acc|best)=[^\s,]+", postfix)
                        display = f"{bar}  {' '.join(kv)}"
                    else:
                        display = status

                    rows.append(
                        f"    {color}{icon} {method:<14}{RESET} {display}"
                    )

    # Redraw: cursor up + clear to end, then print new rows
    if prev_n:
        sys.stdout.write(f"\033[{prev_n}A\033[J")
    for row in rows:
        print(row)
    sys.stdout.flush()
    return len(rows)


if __name__ == "__main__":
    print(f"Watching {LOG_DIR}   Ctrl-C to quit\n")
    n = 0
    try:
        while True:
            n = draw(n)
            time.sleep(REFRESH)
    except KeyboardInterrupt:
        print("\nBye.")
