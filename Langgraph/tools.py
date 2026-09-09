"""
tools.py - the 3 plain Python functions the agent is allowed to call,
wrapped as LangChain tools (@tool) so they can be bound directly to a
chat model and used as LangGraph tool-call nodes.
"""

import subprocess

from langchain_core.tools import tool


@tool
def read_file(filepath: str) -> str:
    """Read and return the contents of a text file."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file, creating or overwriting it."""
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} chars to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def run_cmd(command: str) -> str:
    """Execute a shell command and return its combined stdout + stderr."""
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        return r.stdout + r.stderr
    except Exception as e:
        return f"Error running command: {e}"


TOOLS = [read_file, write_file, run_cmd]

# Name -> tool, so a graph node can invoke by the name a tool_call gives us.
TOOL_FUNCTIONS = {t.name: t for t in TOOLS}

# Tools that touch the filesystem / run code require human approval before running.
DANGEROUS_TOOLS = {"write_file", "run_cmd"}
