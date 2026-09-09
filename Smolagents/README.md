# The Dynamic CSV/Data Analyst (smolagents)

A minimal smolagents `CodeAgent` demo. Instead of calling fixed tools, the
agent **writes real Python** (pandas/matplotlib) and executes it itself to
answer questions about a CSV file. Uses Gemini Flash by default, with a
local Ollama model as an automatic fallback.

## Files

- `agent.py` — the whole demo. Run this.
- `ollama_model.py` — model selection: Gemini Flash by default, local Ollama fallback.
- `observability.py` — optional Langfuse tracing (off by default).
- `generate_data.py` — regenerates `data/marketing_sales.csv` (already generated for you).
- `data/marketing_sales.csv` — sample dataset: 90 days x 3 regions x 6 products.

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

This asks the default question: *"find the top 3 underperforming products
in EMEA, compute their 14-day rolling average revenue, and plot a chart."*
Watch the terminal — you'll see the agent's reasoning, the Python code it
writes, and the code's actual output, step by step.

Or ask your own question:

```bash
python agent.py "What was total revenue per region in June 2026?"
```

## Model selection: Gemini by default, Ollama as fallback

By default `agent.py` uses Gemini Flash (`GEMINI_MODEL`, via the Google AI
Studio OpenAI-compatible endpoint) for every call. If a call to Gemini
actually fails - missing/invalid key, quota, network error - and a local
Ollama server is serving `OLLAMA_MODEL`, that specific call is retried
against Ollama instead. Each new call tries Gemini again, so a transient
failure doesn't permanently downgrade the rest of the run.

- No `GEMINI_API_KEY` set → runs on local Ollama only (`OLLAMA_MODEL` /
  `OLLAMA_BASE_URL` in `.env`).
- `GEMINI_API_KEY` set but Ollama isn't reachable → runs on Gemini only,
  with no fallback net.
- Both available → Gemini first, Ollama as the safety net.

To use a different local fallback model, edit `.env`:

```
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Any model works, but pick one decent at writing code — small
non-code-tuned models tend to write broken pandas.

## Observability with Langfuse

By default the agent just prints to your terminal. To see full traces
(every LLM call, the exact generated code, stdout/stderr, tokens, timing)
in a UI:

1. Get a free project at [Langfuse Cloud](https://cloud.langfuse.com) (or self-host).
2. Copy `.env.example` to `.env` and fill in `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`.
3. Run `python agent.py` again — you'll see "Tracing enabled" and a full
   trace will appear in your Langfuse project's **Traces** tab.

This works via [OpenInference](https://github.com/Arize-ai/openinference)
instrumenting smolagents and exporting standard OTel spans to Langfuse's
OTel endpoint — no code changes needed in `agent.py` itself.

## How it works

`CodeAgent` doesn't call tools via JSON — it generates a Python script each
step, runs it in a restricted local executor, and feeds the printed output
back to itself until it has an answer. `additional_authorized_imports` in
`agent.py` is the allowlist of libraries it's permitted to `import`
(`pandas`, `numpy`, `matplotlib.pyplot`). This is what eliminates math
hallucination: the agent isn't "guessing" a rolling average, it's running
`df.rolling(14).mean()` for real.

Note: the default `executor_type="local"` runs code in-process — fine for
a workshop, not a sandbox. For untrusted input, swap in `executor_type="docker"`
or `"e2b"` (see smolagents docs).
