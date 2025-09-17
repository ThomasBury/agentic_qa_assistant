"""LLM utility helpers for model-aware parameter mapping.

This adapter ensures reasoning models (e.g., gpt-5-*, o*) receive the correct
parameter names for token limits and safe temperature handling when calling
OpenAI APIs.

- Chat Completions:
  * reasoning models: use `max_completion_tokens`; omit `temperature`
  * non-reasoning: use `max_tokens`; include `temperature` if provided

- Responses API:
  * reasoning models: use `max_output_tokens`; omit `temperature`
  * non-reasoning: use `max_tokens`; include `temperature` if provided

References:
- Chat Completions (OpenAI): https://platform.openai.com/docs/api-reference/chat/create
- Responses API (OpenAI):   https://platform.openai.com/docs/api-reference/responses/create
"""
from __future__ import annotations

from typing import Optional, Dict


def is_reasoning_model(model: str) -> bool:
    """Check if a model name indicates it is a "reasoning" model.

    Reasoning models currently include names starting with "gpt-5-" or "o".

    Parameters
    ----------
    model : str
        The name of the model to check.

    Returns
    -------
    bool
        True if the model is identified as a reasoning model, False otherwise.

    """
    m = (model or "").lower()
    return m.startswith("gpt-5-") or m.startswith("o")


def token_params_for_model(model: str, max_tokens: Optional[int]) -> Dict[str, int]:
    """Map a max token value to the correct Chat Completions API parameter.

    For reasoning models (gpt-5-*, o*), Chat Completions expects
    `max_completion_tokens`. For older/non-reasoning models it expects
    `max_tokens`.

    If `max_tokens` is None, returns an empty dict.

    Parameters
    ----------
    model : str
        The name of the target model.
    max_tokens : Optional[int]
        The desired maximum number of tokens for the completion.

    Returns
    -------
    Dict[str, int]
        A dictionary with the appropriate parameter name and value.

    """
    if max_tokens is None:
        return {}
    if is_reasoning_model(model):
        return {"max_completion_tokens": int(max_tokens)}
    return {"max_tokens": int(max_tokens)}


def temperature_params_for_model(model: str, temperature: Optional[float]) -> Dict[str, float]:
    """Return the temperature parameter only if the model supports it.

    Many reasoning models (gpt-5-*, o*) either ignore or reject non-default temperatures.
    To avoid 400s, omit the `temperature` param for such models and rely on the default.

    Parameters
    ----------
    model : str
        The name of the target model.
    temperature : Optional[float]
        The desired temperature for the completion.

    Returns
    -------
    Dict[str, float]
        A dictionary with the temperature parameter, or an empty dictionary if
        not supported.

    """
    if temperature is None:
        return {}
    if is_reasoning_model(model):
        # Omit temperature to avoid unsupported_value errors; default is used by API
        return {}
    return {"temperature": float(temperature)}


def chat_params_for_model(model: str, max_tokens: Optional[int], temperature: Optional[float] = None) -> Dict[str, float]:
    """Get a dictionary of model-aware parameters for the Chat Completions API.

    This convenience helper merges the correct token and temperature parameters
    based on whether the model is a reasoning model.

    Parameters
    ----------
    model : str
        The name of the target model.
    max_tokens : Optional[int]
        The desired maximum number of tokens for the completion.
    temperature : Optional[float], optional
        The desired temperature for the completion.

    Returns
    -------
    Dict[str, float]
        A dictionary of parameters suitable for the Chat Completions API.

    """
    params: Dict[str, float] = {}
    params.update(token_params_for_model(model, max_tokens))
    params.update(temperature_params_for_model(model, temperature))
    return params


def responses_token_params_for_model(model: str, max_tokens: Optional[int]) -> Dict[str, int]:
    """Map a max token value to the correct Responses API parameter.

    Reasoning models: `max_output_tokens`. Non-reasoning: `max_tokens`.

    Parameters
    ----------
    model : str
        The name of the target model.
    max_tokens : Optional[int]
        The desired maximum number of tokens for the response.

    Returns
    -------
    Dict[str, int]
        A dictionary with the appropriate parameter name and value.

    """
    if max_tokens is None:
        return {}
    if is_reasoning_model(model):
        return {"max_output_tokens": int(max_tokens)}
    return {"max_tokens": int(max_tokens)}


def responses_params_for_model(model: str, max_tokens: Optional[int], temperature: Optional[float] = None) -> Dict[str, float]:
    """Get a dictionary of model-aware parameters for the Responses API.

    Reasoning models use `max_output_tokens` and omit `temperature`.

    Parameters
    ----------
    model : str
        The name of the target model.
    max_tokens : Optional[int]
        The desired maximum number of tokens for the response.
    temperature : Optional[float], optional
        The desired temperature for the response.

    Returns
    -------
    Dict[str, float]
        A dictionary of parameters suitable for the Responses API.

    """
    params: Dict[str, float] = {}
    params.update(responses_token_params_for_model(model, max_tokens))
    if not is_reasoning_model(model) and temperature is not None:
        params["temperature"] = float(temperature)
    return params
