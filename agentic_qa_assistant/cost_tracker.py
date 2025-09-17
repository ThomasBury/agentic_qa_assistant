from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, Any

import tiktoken


def _normalize_model_key(model: str) -> str:
    """Normalize a model name into a key-friendly format.

    Parameters
    ----------
    model : str
        The model name string.

    Returns
    -------
    str
        A normalized string suitable for use as a dictionary key.

    """
    return ''.join(c if c.isalnum() else '_' for c in model).upper()


@dataclass
class ModelCost:
    """Tracks token usage for a single model.

    Attributes
    ----------
    input_tokens : int
        Number of tokens in the prompt.
    input_cached_tokens : int
        Number of cached tokens in the prompt (if supported by API).
    output_tokens : int
        Number of tokens in the completion.
    reasoning_tokens : int
        Number of tokens used for reasoning/tool use (if supported by API).
    total_tokens : int
        Total tokens for a chat completion call.
    embedding_tokens : int
        Total tokens for an embedding call.

    """
    # Chat usage
    input_tokens: int = 0
    input_cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    # Embeddings usage
    embedding_tokens: int = 0


@dataclass
class CostTracker:
    """Tracks token usage and associated costs across multiple models.

    Attributes
    ----------
    started_at : float
        The timestamp when the tracker was initialized.
    models : Dict[str, ModelCost]
        A dictionary mapping model names to their respective ModelCost objects.

    """
    started_at: float = field(default_factory=time.time)
    models: Dict[str, ModelCost] = field(default_factory=dict)

    def _ensure_model(self, model: str) -> ModelCost:
        """Ensure a ModelCost entry exists for a given model.

        If an entry for the model does not exist, it is created.

        Parameters
        ----------
        model : str
            The name of the model.

        Returns
        -------
        ModelCost
            The cost tracking object for the specified model.

        """
        if model not in self.models:
            self.models[model] = ModelCost()
        return self.models[model]

    def add_chat_usage_detailed(self, model: str, *, prompt_tokens: int = 0, completion_tokens: int = 0,
                                 total_tokens: int = 0, cached_prompt_tokens: int = 0, reasoning_tokens: int = 0) -> None:
        """Add detailed chat completion usage to the tracker.

        Parameters
        ----------
        model : str
            The name of the model used for the chat completion.
        prompt_tokens : int, optional
            Number of tokens in the prompt.
        completion_tokens : int, optional
            Number of tokens in the completion.
        total_tokens : int, optional
            Total tokens for the API call. If not provided, it's the sum of
            prompt and completion tokens.
        cached_prompt_tokens : int, optional
            Number of cached tokens in the prompt.
        reasoning_tokens : int, optional
            Number of tokens used for reasoning or tool use.

        """
        mc = self._ensure_model(model)
        mc.input_tokens += int(prompt_tokens or 0)
        mc.output_tokens += int(completion_tokens or 0)
        mc.total_tokens += int(total_tokens or (prompt_tokens + completion_tokens))
        mc.input_cached_tokens += int(cached_prompt_tokens or 0)
        mc.reasoning_tokens += int(reasoning_tokens or 0)

    def add_embeddings_usage(self, model: str, total_tokens: int) -> None:
        """Add embedding usage to the tracker.

        Parameters
        ----------
        model : str
            The name of the model used for the embedding.
        total_tokens : int
            Total tokens processed by the embedding model.

        """
        mc = self._ensure_model(model)
        mc.embedding_tokens += int(total_tokens or 0)

    def estimate_tokens(self, texts: Iterable[str], encoding_name: str = "cl100k_base") -> int:
        """Estimate the number of tokens for a list of texts.

        Uses a tiktoken encoder. Falls back to a crude word count if encoding fails.

        Parameters
        ----------
        texts : Iterable[str]
            An iterable of text strings to estimate tokens for.
        encoding_name : str, optional
            The name of the tiktoken encoding to use, by default "cl100k_base".

        Returns
        -------
        int
            The estimated total number of tokens.

        """
        enc = tiktoken.get_encoding(encoding_name)
        total = 0
        for t in texts:
            try:
                total += len(enc.encode(t or ""))
            except Exception:  # noqa: S110
                total += len((t or "").split())  # crude fallback
        return total

    def summary(self) -> Dict[str, Any]:
        """Generate a dictionary summary of all token usage.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the token usage summary.

        """
        out = {
            'started_at': self.started_at,
            'models': {},
        }
        for model, mc in self.models.items():
            out['models'][model] = {
                'input_tokens': mc.input_tokens,
                'input_cached_tokens': mc.input_cached_tokens,
                'output_tokens': mc.output_tokens,
                'reasoning_tokens': mc.reasoning_tokens,
                'total_tokens': mc.total_tokens,
                'embedding_tokens': mc.embedding_tokens,
            }
        return out

    def summary_text(self) -> str:
        """Generate a human-readable text summary of token usage.

        Returns
        -------
        str
            A formatted string summarizing token usage.

        """
        s = self.summary()
        lines = ["Token Usage Summary:"]
        for model, m in s['models'].items():
            lines.append(
                f"- {model}: input={m['input_tokens']} (cached={m['input_cached_tokens']}), output={m['output_tokens']}, reasoning={m['reasoning_tokens']}, embed={m['embedding_tokens']}, total={m['total_tokens']}"
            )
        return "\n".join(lines)

    def summary_json(self) -> str:
        """Generate a JSON string summary of token usage.

        Returns
        -------
        str
            A JSON-formatted string summarizing token usage.

        """
        return json.dumps(self.summary(), indent=2)