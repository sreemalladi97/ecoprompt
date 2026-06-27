"""
EcoPrompt - Routing Matrix
Classifies prompt complexity and routes to the appropriate model.
Simple tasks → cheap model (gpt-4o-mini)
Complex tasks → powerful model (gpt-4o)
"""

import logging

logger = logging.getLogger("ecoprompt.router")

# Model tiers (swap these out for any LiteLLM-supported models)
CHEAP_MODEL = "gpt-4o-mini"    # ~20x cheaper than gpt-4o
POWERFUL_MODEL = "gpt-4o"      # for complex tasks

# Keywords that signal a complex task
COMPLEX_KEYWORDS = [
    # Code
    "code", "debug", "function", "class", "algorithm", "implement",
    "refactor", "optimize", "bug", "error", "script", "program",
    # Reasoning
    "analyze", "compare", "evaluate", "critique", "explain why",
    "pros and cons", "trade-off", "difference between",
    # Multi-step
    "step by step", "plan", "strategy", "design", "architect",
    "how should i", "what is the best way",
    # Long-form
    "write a report", "write an essay", "detailed", "comprehensive",
]

# Token threshold — long prompts are usually complex
LONG_PROMPT_THRESHOLD = 200  # words


def classify(messages: list) -> str:
    """
    Classify the complexity of a request.
    Returns: "simple" or "complex"
    """
    # Combine all message content for analysis
    full_text = " ".join(
        m.get("content", "") for m in messages
    ).lower()

    word_count = len(full_text.split())

    # Rule 1: Long prompts are complex
    if word_count > LONG_PROMPT_THRESHOLD:
        logger.debug(f"Complex: long prompt ({word_count} words)")
        return "complex"

    # Rule 2: Complex keywords present
    for keyword in COMPLEX_KEYWORDS:
        if keyword in full_text:
            logger.debug(f"Complex: keyword '{keyword}' found")
            return "complex"

    logger.debug("Simple: no complexity signals found")
    return "simple"


def route(messages: list, requested_model: str) -> str:
    """
    Decide which model to use based on prompt complexity.

    Args:
        messages: The conversation messages
        requested_model: The model the client originally requested

    Returns:
        The model name to actually use
    """
    complexity = classify(messages)

    if complexity == "simple":
        chosen = CHEAP_MODEL
        reason = "simple task → cheap model"
    else:
        chosen = POWERFUL_MODEL
        reason = "complex task → powerful model"

    logger.info(f"Router: {requested_model} → {chosen} ({reason})")
    return chosen
