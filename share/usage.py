# USD per 1M tokens, from the provider's pricing page.
INPUT_PRICE_PER_M = 0.1
CACHED_INPUT_PRICE_PER_M = 0.01
OUTPUT_PRICE_PER_M = 0.5


def cost_usd(usage):
    # cached_tokens is a subset of input_tokens, so bill only the uncached rest at the full input price.
    # reasoning_tokens is likewise a subset of output_tokens and needs no separate line.
    cached = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    uncached = usage.input_tokens - cached
    return (
        uncached * INPUT_PRICE_PER_M
        + cached * CACHED_INPUT_PRICE_PER_M
        + usage.output_tokens * OUTPUT_PRICE_PER_M
    ) / 1_000_000


def format_usage(usage):
    cached = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
    reasoning = usage.output_tokens_details.reasoning_tokens if usage.output_tokens_details else 0
    return (
        f"[input_tokens={usage.input_tokens} (cached={cached}), "
        f"output_tokens={usage.output_tokens} (reasoning={reasoning}), "
        f"cost=${cost_usd(usage):.6f}]"
    )
