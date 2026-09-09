# The Minimal Coding Agent (LangGraph)

A minimal LangGraph agent demo. Unlike a data-analyst agent that writes
throwaway Python to answer questions, this one is a small **terminal coding
agent**: it can read files, write files, and run shell commands to get a
task done - with a human approval gate in front of anything destructive.
Uses Gemini Flash by default, with a local Ollama model as an automatic
fallback.

## Files

- `agent.py` - the LangGraph graph (agent node + tools node) and the CLI loop. Run this.
- `tools.py` - the 3 tools the agent can call: `read_file`, `write_file`, `run_cmd`.
- `llm.py` - model selection: Gemini Flash by default, local Ollama fallback.
- `security.py` - the human-in-the-loop approval prompt.
- `config.py` - environment/config loading, system prompt, sample prompt.
- `ui.py` - terminal output formatting (rich).

## Setup

1. Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey).

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY`.

## Run it

```bash
python agent.py
```

This starts an interactive loop. Try the suggested sample prompt printed at
startup, or type your own request, e.g.:

```
You: Write a Python script called calculator.py with add, subtract,
multiply, and divide functions, then run it to confirm it works.
```

Watch the terminal - you'll see the agent's reasoning, each tool call it
wants to make, and a `[SECURITY]` prompt before it's allowed to write a
file or run a command. Type `exit` or `quit` to leave.

## Model selection: Gemini by default, Ollama as fallback

By default `llm.py` uses Gemini Flash (`GEMINI_MODEL`, via the Google AI
Studio OpenAI-compatible endpoint) for every call. If a call to Gemini
actually fails - missing/invalid key, quota, network error - and a local
Ollama server is serving `OLLAMA_MODEL`, that specific call is retried
against Ollama instead. Each new call tries Gemini again, so a transient
failure doesn't permanently downgrade the rest of the session.

- No `GEMINI_API_KEY` set -> runs on local Ollama only (`OLLAMA_MODEL` /
  `OLLAMA_BASE_URL` in `.env`).
- `GEMINI_API_KEY` set but Ollama isn't reachable -> runs on Gemini only,
  with no fallback net.
- Both available -> Gemini first, Ollama as the safety net.

The startup banner prints which backend actually answered last (or will be
tried first), e.g. `model: qwen2.5-coder:7b (Ollama - no GEMINI_API_KEY set)`.

To use a different local fallback model, edit `.env`:

```
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

Any model works, but pick one decent at following tool-call formatting -
small models sometimes emit a tool call as plain-text JSON instead of a
real tool call. `llm.py`'s `_recover_inline_tool_calls` scans for that
pattern and repairs it before it reaches the graph (Gemini's endpoint
doesn't need this - it's only wired into the Ollama chain).

## Observability with LangSmith

By default the agent just prints to your terminal. To see full traces
(every LLM call, tool call, and timing) in a UI, no code changes needed:

1. Get a free account at [LangSmith](https://smith.langchain.com).
2. Copy `.env.example` to `.env`, set `LANGSMITH_TRACING=true`, and fill in
   `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`).
3. Run `python agent.py` again - the startup banner will show
   `Tracing: enabled (LangSmith)`, and a full trace will appear in your
   LangSmith project.

This works because LangChain auto-instruments every model/tool call once
`LANGSMITH_TRACING` and `LANGSMITH_API_KEY` are present in the environment.

## Security: human-in-the-loop approval

Nothing that touches the filesystem or runs a shell command executes
without a person typing `y` first. `write_file` and `run_cmd` are marked
`DANGEROUS_TOOLS` in `tools.py`; `read_file` is not, since it can't change
anything. `security.py` is the one file to read to understand the whole
safety model.

Mechanically, this runs on LangGraph's `interrupt()` /
`Command(resume=...)`: the `tools` node calls `interrupt()` for every
dangerous tool call in a turn *before* running any of them, which pauses
the graph and surfaces the pending call(s) back to `run_agent()` in
`agent.py`. Once a person answers via `ask_permission()`, the graph resumes
from exactly where it paused - approved calls run, denied ones return
`"Tool execution denied by human."` to the model so it can react and try
something else.

## How it works

`agent.py` builds a two-node `StateGraph`:

- **`agent`** - calls the model (bound to the 3 tools) with the running
  message history.
- **`tools`** - executes every tool call the model just made, collecting
  human approvals first, then running each approved call and feeding its
  result back as a `ToolMessage`.

The graph loops `agent -> tools -> agent` until the model responds with no
further tool calls, or `MAX_TURNS` is hit (`GraphRecursionError`). An
`InMemorySaver` checkpointer keeps the conversation (and any pending
interrupt) tied to a `thread_id`, so the CLI's `while` loop can keep
talking to the same session turn after turn.

Note: `run_cmd` executes shell commands directly on your machine - fine
for a workshop with the approval gate in front of it, not a sandbox for
untrusted input.
