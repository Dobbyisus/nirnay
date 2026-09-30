"""Token prices and cost calculation, in rupees."""

from __future__ import annotations

# ₹ per 1M tokens: (input, cached input, output).
# Source: docs.sarvam.ai pricing page, checked 30 Sep 2026. Prices change; keep this current.
PRICES_INR: dict[str, tuple[float, float, float]] = {
    "sarvam-105b": (29.28, 10.98, 73.20),
    "sarvam-105b-conversations": (29.28, 10.98, 73.20),
    "gemma4": (36.6, 13.73, 91.5),
    "glm5.3": (126.0, 23.4, 396.0),
    "deepseekv4-flash": (19.8, 0.63, 59.4),
}


def cost_inr(
    model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int
) -> float | None:
    """Cost of one or more calls in ₹, or ``None`` if the model's price is unknown.

    ``cached_input_tokens`` is the part of ``input_tokens`` served from Sarvam's prompt cache,
    billed at the cheaper cached rate.
    """
    prices = PRICES_INR.get(model)
    if prices is None:
        return None
    p_in, p_cached, p_out = prices
    uncached = input_tokens - cached_input_tokens
    return (uncached * p_in + cached_input_tokens * p_cached + output_tokens * p_out) / 1_000_000
