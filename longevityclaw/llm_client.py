"""
L-LLM (Longevity LLM) client for aging biology queries.

Provides integration with the L-LLM model deployed on HuggingFace Inference Endpoints
or local vLLM server.

Backends:
- HuggingFace (default): Set HF_TOKEN env var
- Local vLLM: Set LLM_BACKEND=local, VLLM_ENDPOINT, VLLM_API_KEY
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


# HuggingFace configuration
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://lllmurl.us-east-2.aws.endpoints.huggingface.cloud")
HF_MODEL = os.environ.get("HF_MODEL", "longevity-llm")

# Local vLLM configuration (from environment)
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT")  # Required for local vLLM
VLLM_MODEL = os.environ.get("VLLM_MODEL", "longevity-llm-local")

# Backend selection: "hf" (default) or "local"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "hf").lower()

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
    """Get vLLM API key from environment."""
    key = os.environ.get("VLLM_API_KEY")
    if not key:
        raise ValueError("VLLM_API_KEY environment variable not set")
    return key


def get_current_backend() -> str:
    """Get current LLM backend (re-read from env for runtime switching)."""
    return os.environ.get("LLM_BACKEND", "hf").lower()


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
        backend: Override backend ("hf" or "local"). If None, uses LLM_BACKEND env.

    Returns:
        LLMResponse with content and metadata.

    Raises:
        httpx.HTTPStatusError: On API errors.
        ValueError: If required token/key not set.
    """
    # Determine backend
    use_backend = backend if backend else get_current_backend()

    if use_backend == "local":
        resp = _query_vllm(
            prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
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


def _query_vllm(
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: float = 120.0,
) -> LLMResponse:
    """Query local vLLM server."""
    api_key = get_vllm_api_key()
    endpoint = os.environ.get("VLLM_ENDPOINT", VLLM_ENDPOINT)
    model = os.environ.get("VLLM_MODEL", VLLM_MODEL)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
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
