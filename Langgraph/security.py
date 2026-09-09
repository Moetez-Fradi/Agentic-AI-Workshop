"""
security.py - the human-in-the-loop approval gate.

Nothing that touches the filesystem or runs a shell command executes
without a person typing 'y' first. This is the one file a student should
read to understand the whole safety model of the agent. The actual pause
happens via LangGraph's interrupt()/Command(resume=...) mechanism in
agent.py's tools node - this module is just the terminal I/O for it.
"""

import sys

try:
    import termios
except ImportError:
    termios = None

from ui import console


def _drain_stale_input() -> None:
    """Discard anything already buffered on stdin before asking a real
    security question, using the POSIX tty-level flush (not a manual
    select()+readline() loop, which has its own double-buffering hazards
    with Python's io layer). Only applies to a real interactive terminal -
    piped stdin (tests, automation) is left untouched."""
    if termios is None or not sys.stdin.isatty():
        return
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def ask_permission(tool_name: str, args: dict) -> bool:
    """Pause and ask the human before running a potentially risky tool."""
    _drain_stale_input()
    console.print(
        f"\n[bold yellow][SECURITY] Agent wants to execute: "
        f"{tool_name}({args}). Allow? [y/N]:[/bold yellow] ",
        end="",
    )
    answer = input()
    return answer.strip().lower() == "y"
