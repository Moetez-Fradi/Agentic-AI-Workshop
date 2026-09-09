import json
import os
import urllib.request

from smolagents import OpenAIServerModel

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def get_ollama_model(
    model_id: str | None = None,
    base_url: str | None = None,
) -> OpenAIServerModel:
    model_id = model_id or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
    base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    return OpenAIServerModel(
        model_id=model_id,
        api_base=base_url,
        api_key="ollama",
    )


def get_gemini_model(model_id: str | None = None) -> OpenAIServerModel:
    model_id = model_id or os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY isn't set - required to use Gemini.")

    return OpenAIServerModel(
        model_id=model_id,
        api_base=GEMINI_BASE_URL,
        api_key=api_key,
    )


def _ollama_available(model_id: str, base_url: str, timeout: float = 2.0) -> bool:
    """True only if Ollama is reachable AND model_id is actually pulled -
    a running server with the wrong/missing model doesn't count as a usable
    fallback."""
    root = base_url.removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as resp:
            models = json.load(resp)["models"]
        names = {m["model"] for m in models} | {m["model"].split(":")[0] for m in models}
        return model_id in names
    except Exception:
        return False


class FallbackModel:
    """smolagents Model wrapper: try `primary` (Gemini) for every call; if a
    call actually raises (bad/missing key, quota, network...), retry that
    same call against `secondary` (local Ollama) instead. Each new call
    tries the primary again, so a transient Gemini failure doesn't
    permanently downgrade the rest of the run.

    Duck-types smolagents' Model interface: implements generate() and
    generate_stream() explicitly (both are called by CodeAgent) and proxies
    everything else (model_id, parse_tool_calls, to_dict, ...) to whichever
    backend most recently answered a call.
    """

    def __init__(self, primary: OpenAIServerModel, secondary: OpenAIServerModel, secondary_label: str):
        self._primary = primary
        self._secondary = secondary
        self._secondary_label = secondary_label
        self._active = primary
        self.model_id = primary.model_id

    def __getattr__(self, name):
        return getattr(self._active, name)

    def _use(self, model: OpenAIServerModel) -> None:
        self._active = model
        self.model_id = model.model_id

    def generate(self, *args, **kwargs):
        try:
            result = self._primary.generate(*args, **kwargs)
            self._use(self._primary)
            return result
        except Exception as e:
            print(f"[model] Gemini call failed ({e}) - falling back to {self._secondary_label}.")
            self._use(self._secondary)
            return self._secondary.generate(*args, **kwargs)

    def generate_stream(self, *args, **kwargs):
        yielded_any = False
        try:
            for chunk in self._primary.generate_stream(*args, **kwargs):
                yielded_any = True
                yield chunk
            self._use(self._primary)
            return
        except Exception as e:
            if yielded_any:
                raise  # partial output already delivered - can't safely restart mid-stream
            print(f"[model] Gemini call failed ({e}) - falling back to {self._secondary_label}.")
            self._use(self._secondary)
        yield from self._secondary.generate_stream(*args, **kwargs)


def get_model(
    model_id: str | None = None,
    base_url: str | None = None,
) -> OpenAIServerModel | FallbackModel:
    """Gemini Flash is the default backend. If a call to it actually fails
    (missing/invalid key, quota, network...) it falls back to the local
    Ollama server for that call."""
    ollama_model_id = model_id or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
    ollama_base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("[model] GEMINI_API_KEY not set - using local Ollama only.")
        return get_ollama_model(ollama_model_id, ollama_base_url)

    gemini = get_gemini_model()

    if not _ollama_available(ollama_model_id, ollama_base_url):
        print(
            f"[model] Using Gemini ({gemini.model_id}) - no local Ollama fallback available "
            f"('{ollama_model_id}' not being served at {ollama_base_url})."
        )
        return gemini

    ollama = get_ollama_model(ollama_model_id, ollama_base_url)
    print(f"[model] Using Gemini ({gemini.model_id}) as primary, Ollama ({ollama_model_id}) as fallback.")
    return FallbackModel(primary=gemini, secondary=ollama, secondary_label=f"Ollama ({ollama_model_id})")
