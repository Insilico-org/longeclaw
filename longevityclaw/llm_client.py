"""
L-LLM (Longevity LLM) client for aging biology queries.

Talks to the longevity model over an OpenAI-compatible chat API. Three backends:

- HuggingFace Inference Endpoint (default): LLM_BACKEND=hf, set HF_TOKEN
- Self-hosted vLLM (bf16 official weights): LLM_BACKEND=local, LOCAL_ENGINE=vllm
- Self-hosted llama.cpp (GGUF quant):       LLM_BACKEND=local, LOCAL_ENGINE=llamacpp

The two self-hosted engines speak the same protocol but differ in one place that
matters here: how reasoning ("thinking") is switched off. vLLM honours the nested
``chat_template_kwargs.enable_thinking`` flag; llama.cpp ignores it and instead
respects the Qwen ``/no_think`` soft switch appended to the user turn. The local
query path picks the right lever from LOCAL_ENGINE.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


# HuggingFace configuration
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://lllmurl.us-east-2.aws.endpoints.huggingface.cloud")
HF_MODEL = os.environ.get("HF_MODEL", "longevity-llm")

# Self-hosted backend (vLLM or llama.cpp). Endpoint/model are re-read at call
# time for runtime switching; these module constants are the fallbacks.
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT") or os.environ.get("LOCAL_ENDPOINT")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "longevity-llm")

# Backend selection: "hf" (default) or a local engine (see _LOCAL_BACKENDS).
LLM_BACKEND = os.environ.get("LLM_BACKEND", "hf").lower()

# Self-hosted engine for the local backend: "vllm" (default) or "llamacpp".
LOCAL_ENGINE = os.environ.get("LOCAL_ENGINE", "vllm").lower()

# LLM_BACKEND values that mean "self-hosted". "vllm"/"llamacpp" double as engine
# shorthands (they pin the engine); "local" defers the engine to LOCAL_ENGINE.
_LOCAL_BACKENDS = {"local", "vllm", "llamacpp", "llama.cpp", "llama_cpp", "gguf"}
_ENGINE_ALIASES = {"llama.cpp": "llamacpp", "llama_cpp": "llamacpp", "gguf": "llamacpp"}

# Per-engine default sampling extras. A repetition penalty > 1.0 is not optional:
# the model loops at 1.0 (167-response sweep). llama.cpp spells it repeat_penalty.
_ENGINE_DEFAULT_EXTRA = {
    "vllm": {"repetition_penalty": 1.1},
    "llamacpp": {"repeat_penalty": 1.1},
}

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7

# Session-cumulative L-LLM usage, surfaced by the CLI's /usage command. Counts
# every call routed through query_llm (the direct tool plus internal callers).
_LLM_STATS: dict[str, Any] = {"calls": 0, "input": 0, "output": 0, "by_model": {}}


def get_llm_stats() -> dict:
    """Return a copy of session-cumulative L-LLM usage."""
    return {
        "calls": _LLM_STATS["calls"],
        "input": _LLM_STATS["input"],
        "output": _LLM_STATS["output"],
        "by_model": dict(_LLM_STATS["by_model"]),
    }


def reset_llm_stats() -> None:
    _LLM_STATS.update(calls=0, input=0, output=0, by_model={})


def _record_llm_call(resp: "LLMResponse") -> None:
    _LLM_STATS["calls"] += 1
    usage = getattr(resp, "usage", None) or {}
    _LLM_STATS["input"] += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    _LLM_STATS["output"] += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    model = getattr(resp, "model", None) or "L-LLM"
    _LLM_STATS["by_model"][model] = _LLM_STATS["by_model"].get(model, 0) + 1


@dataclass(frozen=True)
class LLMResponse:
    """Response from L-LLM query."""

    content: str
    model: str
    usage: dict[str, int]
    raw_response: dict[str, Any]


def get_hf_token() -> str:
    """Get HuggingFace token from environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    return token


def get_hf_endpoint() -> str:
    """Get HuggingFace endpoint from environment (re-read for runtime switching)."""
    endpoint = os.environ.get("HF_ENDPOINT", HF_ENDPOINT)
    if not endpoint:
        raise ValueError("HF_ENDPOINT environment variable not set")
    return endpoint


def get_vllm_api_key() -> str:
    """Bearer token for the local server.

    Optional — self-hosted vLLM/llama.cpp usually ignore it, and the OpenAI-style
    clients still want a non-empty string, so default to a placeholder rather than
    raising. Set VLLM_API_KEY only when the server was started with --api-key.
    """
    return os.environ.get("VLLM_API_KEY") or "none"


def get_current_backend() -> str:
    """Current LLM backend (re-read from env for runtime switching)."""
    return os.environ.get("LLM_BACKEND", "hf").lower()


def is_local_backend(backend: str | None = None) -> bool:
    """True when *backend* (or the configured one) is a self-hosted engine."""
    return (backend or get_current_backend()).lower() in _LOCAL_BACKENDS


def _normalize_engine(name: str) -> str:
    """Canonical engine name: 'vllm' or 'llamacpp'."""
    n = (name or "vllm").lower()
    n = _ENGINE_ALIASES.get(n, n)
    return n if n in _ENGINE_DEFAULT_EXTRA else "vllm"


def get_local_engine(backend: str | None = None) -> str:
    """Resolve the self-hosted engine.

    A backend that names an engine ("vllm"/"llamacpp") pins it; the generic
    "local" defers to the LOCAL_ENGINE env var.
    """
    b = (backend or get_current_backend()).lower()
    if b in _ENGINE_DEFAULT_EXTRA or b in _ENGINE_ALIASES:
        return _normalize_engine(b)
    return _normalize_engine(os.environ.get("LOCAL_ENGINE", LOCAL_ENGINE))


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge so nested objects (e.g. chat_template_kwargs) combine
    rather than overwrite."""
    return base | {
        k: _deep_merge(base[k], v)
        if isinstance(v, dict) and isinstance(base.get(k), dict) else v
        for k, v in override.items()
    }


def _local_extra_body(engine: str) -> dict:
    """Sampling extras for a local request: per-engine defaults overlaid with the
    optional LOCAL_EXTRA_BODY JSON override (e.g. '{"top_p": 0.8, "top_k": 20}')."""
    extra = dict(_ENGINE_DEFAULT_EXTRA[engine])
    if raw := os.environ.get("LOCAL_EXTRA_BODY"):
        try:
            extra = _deep_merge(extra, json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass
    return extra


def _strip_thinking(content: str) -> str:
    """Remove a chain-of-thought preamble the model may prepend.

    The official vLLM container pre-opens ``<think>`` and runs with no reasoning
    parser, so the trace arrives inside ``message.content`` terminated by a bare
    ``</think>`` (interface-contract layout C). Split on it; otherwise return the
    content untouched. (The GGUF on llama.cpp emits an untagged "Thinking
    Process:" ramble with no delimiter — suppressed via /no_think, not stripped.)
    """
    if content and "</think>" in content:
        return content.split("</think>")[-1].strip()
    return content.strip() if content else content


def query_llm(
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    enable_thinking: bool = False,
    timeout: float = 120.0,
    backend: str | None = None,
) -> LLMResponse:
    """
    Query L-LLM with a prompt.

    Args:
        prompt: User prompt to send to the model.
        system_prompt: Optional system prompt for context.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0.0-1.0).
        enable_thinking: Enable extended thinking mode (increases latency).
        timeout: Request timeout in seconds.
        backend: Override backend ("hf", "local", "vllm", or "llamacpp"). If None,
            uses the LLM_BACKEND env var.

    Returns:
        LLMResponse with content and metadata.

    Raises:
        httpx.HTTPStatusError: On API errors.
        ValueError: If required token/key not set.
    """
    # Determine backend
    use_backend = (backend or get_current_backend()).lower()

    if is_local_backend(use_backend):
        resp = _query_local(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            engine=get_local_engine(use_backend),
            timeout=timeout,
        )
    else:
        resp = _query_hf(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
            timeout=timeout,
        )
    _record_llm_call(resp)
    return resp


def _query_hf(
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    enable_thinking: bool = False,
    timeout: float = 120.0,
) -> LLMResponse:
    """Query HuggingFace endpoint."""
    token = get_hf_token()
    endpoint = get_hf_endpoint()

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
        response = client.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    choice = data["choices"][0]
    return LLMResponse(
        content=choice["message"]["content"],
        model=data.get("model", HF_MODEL),
        usage=data.get("usage", {}),
        raw_response=data,
    )


def _query_local(
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    enable_thinking: bool = False,
    engine: str = "vllm",
    timeout: float = 120.0,
) -> LLMResponse:
    """Query a self-hosted OpenAI-compatible server (vLLM or llama.cpp).

    Reasoning control is engine-specific. vLLM honours
    ``chat_template_kwargs.enable_thinking`` (thinking is on by default in this
    model's template). llama.cpp ignores that flag, so when thinking is off we
    append the Qwen ``/no_think`` soft switch to the user turn instead. In both
    cases a repetition penalty is sent (see _ENGINE_DEFAULT_EXTRA) and any
    ``<think>`` preamble is stripped from the reply.
    """
    engine = _normalize_engine(engine)
    endpoint = os.environ.get("VLLM_ENDPOINT") or os.environ.get("LOCAL_ENDPOINT") or VLLM_ENDPOINT
    if not endpoint:
        raise ValueError("VLLM_ENDPOINT (or LOCAL_ENDPOINT) not set for the local L-LLM backend")
    model = os.environ.get("VLLM_MODEL", VLLM_MODEL)
    api_key = get_vllm_api_key()

    user_content = prompt
    extra_body = _local_extra_body(engine)
    if engine == "vllm":
        extra_body = _deep_merge(extra_body, {"chat_template_kwargs": {"enable_thinking": enable_thinking}})
    elif not enable_thinking:
        user_content = f"{prompt} /no_think"

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **extra_body,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
        response = client.post(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    choice = data["choices"][0]
    return LLMResponse(
        content=_strip_thinking(choice["message"]["content"]),
        model=data.get("model", model),
        usage=data.get("usage", {}),
        raw_response=data,
    )


def predict_lifespan_effect(
    compound: str,
    species: str,
    pubmed_abstract: str | None = None,
) -> dict[str, Any]:
    """
    Predict lifespan effect of a compound using L-LLM.

    Uses prompt format aligned with SynergyAge lifespan regression training task.

    Args:
        compound: Name of the compound/drug.
        species: Target species.
        pubmed_abstract: Optional PubMed abstract for context.

    Returns:
        Dictionary with predicted lifespan_change_percent, confidence,
        and raw_response.
    """
    import re

    # Build context similar to training format
    context_parts = [f"Species: {species}"]

    if pubmed_abstract:
        context_parts.append(f"\nRelevant literature:\n{pubmed_abstract[:2000]}")

    context = "\n".join(context_parts)

    # Explicit prompt for percentage prediction (model trained on gene tasks, adapt for compounds)
    prompt = f"""What is the percent lifespan change of {compound} treatment compared to wild type?

{context}

IMPORTANT: You MUST start your response with a signed percentage (e.g., +15%, -8%, +22.5%).
Then briefly explain the mechanisms.

Example format:
+12%
Brief explanation of why...

Now predict the lifespan effect of {compound}:"""

    # System prompt from training data
    system_prompt = "You are a biomedical AI specialized in aging biology, trained on genomic, proteomic, and clinical data."

    response = query_llm(
        prompt,
        system_prompt=system_prompt,
        temperature=0.3,  # Lower temperature for consistent predictions
    )

    # Parse response - model trained to return just a percentage like "+16.9%"
    result: dict[str, Any] = {
        "compound": compound,
        "species": species,
        "raw_response": response.content,
        "lifespan_change_percent": None,
        "avg_lifespan_change": None,  # Alias for compatibility
        "max_lifespan_change": None,
        "confidence": "medium",  # Default confidence
        "mechanisms": None,
        "reasoning": response.content,  # Full response as reasoning
    }

    content = response.content.strip()

    # Try to find percentage pattern like "+16.9%", "-5.2%", "15%"
    percentage_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", content)
    if percentage_match:
        try:
            value = float(percentage_match.group(1))
            result["lifespan_change_percent"] = value
            result["avg_lifespan_change"] = value  # Alias
        except ValueError:
            pass

    # Fallback: look for phrases like "increase lifespan by X", "extend by X%"
    if result["lifespan_change_percent"] is None:
        # Try patterns like "10-20%", "15 percent", "extend lifespan by 10"
        fallback_patterns = [
            r"(?:increase|extend|improve|prolong).*?(?:by|of)\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:percent|to\s*\d+\s*percent)",
            r"lifespan.*?(\d+(?:\.\d+)?)\s*%",
        ]
        for pattern in fallback_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    if value > 0 and value < 100:  # Sanity check
                        result["lifespan_change_percent"] = value
                        result["avg_lifespan_change"] = value
                        break
                except ValueError:
                    pass

    # If response contains additional explanation, try to extract it
    lines = content.split("\n")
    if len(lines) > 1:
        # First line is likely the percentage, rest is explanation
        explanation_lines = [ln for ln in lines[1:] if ln.strip()]
        if explanation_lines:
            result["mechanisms"] = " ".join(explanation_lines)

    # Set confidence based on response clarity
    if result["lifespan_change_percent"] is not None:
        result["confidence"] = "high" if len(content) < 50 else "medium"
    else:
        result["confidence"] = "low"

    return result


def analyze_aging_mechanism(query: str) -> LLMResponse:
    """
    General aging biology query to L-LLM.

    Args:
        query: Question about aging mechanisms, pathways, or interventions.

    Returns:
        LLMResponse with the analysis.
    """
    system_prompt = """You are an expert in aging biology and longevity research.
Provide detailed, scientifically accurate answers about aging mechanisms,
longevity interventions, and related biological pathways."""

    return query_llm(query, system_prompt=system_prompt)


def score_pathway_with_llm(
    pathway_name: str,
    gene_list: list[str],
    context: str | None = None,
) -> dict[str, Any]:
    """
    Score a pathway's relevance to aging using L-LLM.

    Args:
        pathway_name: Name of the biological pathway.
        gene_list: Genes in the pathway.
        context: Additional context about the analysis.

    Returns:
        Dictionary with aging_relevance_score (0-10), mechanisms, and reasoning.
    """
    genes_str = ", ".join(gene_list[:50])  # Limit genes for context
    context_str = f"\nContext: {context}" if context else ""

    prompt = f"""Analyze the pathway "{pathway_name}" for its relevance to aging biology.

Genes in pathway: {genes_str}
{context_str}

Score this pathway's relevance to aging and longevity on a scale of 0-10.
Explain the key aging-related mechanisms involved.

Respond in this format:
AGING_SCORE: <0-10>
KEY_GENES: <most relevant genes for aging>
MECHANISMS: <aging mechanisms involved>
REASONING: <detailed explanation>"""

    system_prompt = """You are an expert in aging biology and pathway analysis.
Evaluate pathways for their relevance to aging, longevity, and age-related diseases."""

    response = query_llm(prompt, system_prompt=system_prompt, temperature=0.3)

    result: dict[str, Any] = {
        "pathway": pathway_name,
        "raw_response": response.content,
        "aging_score": None,
        "key_genes": None,
        "mechanisms": None,
        "reasoning": None,
    }

    for line in response.content.split("\n"):
        line = line.strip()
        if line.startswith("AGING_SCORE:"):
            try:
                val = line.split(":")[1].strip()
                result["aging_score"] = float(val)
            except (ValueError, IndexError):
                pass
        elif line.startswith("KEY_GENES:"):
            result["key_genes"] = line.split(":", 1)[1].strip()
        elif line.startswith("MECHANISMS:"):
            result["mechanisms"] = line.split(":", 1)[1].strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result
