"""
EcoPrompt - Token Compressor
Uses LLMLingua-2 to compress prompts before sending to the AI model.
"""

import logging
from typing import Optional

logger = logging.getLogger("ecoprompt.compressor")

# Lazy-load the compressor so startup is fast
_compressor = None


def get_compressor():
    global _compressor
    if _compressor is None:
        logger.info("Loading LLMLingua-2 model (first time only, ~30s)...")
        from llmlingua import PromptCompressor
        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",  # works on MacBook without GPU
        )
        logger.info("LLMLingua-2 model loaded.")
    return _compressor


def compress_messages(messages: list, target_rate: float = 0.5) -> tuple[list, dict]:
    """
    Compress the content of messages using LLMLingua-2.

    Args:
        messages: List of OpenAI-style message dicts
        target_rate: How aggressively to compress (0.5 = keep 50% of tokens)

    Returns:
        (compressed_messages, stats) where stats has token counts
    """
    compressor = get_compressor()

    original_tokens = 0
    compressed_tokens = 0
    compressed_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Only compress user messages — system and assistant messages stay intact
        if role == "user" and len(content.split()) > 20:
            try:
                result = compressor.compress_prompt(
                    content,
                    rate=target_rate,
                    force_tokens=["\n", "?", "!"],
                )
                compressed_content = result["compressed_prompt"]
                original_tokens += result.get("origin_tokens", len(content.split()))
                compressed_tokens += result.get("compressed_tokens", len(compressed_content.split()))

                compressed_messages.append({"role": role, "content": compressed_content})
                logger.debug(
                    f"Compressed message: {result.get('origin_tokens')} -> "
                    f"{result.get('compressed_tokens')} tokens "
                    f"(ratio: {result.get('ratio', 'N/A')})"
                )
            except Exception as e:
                # If compression fails, pass through original
                logger.warning(f"Compression failed, passing through original: {e}")
                compressed_messages.append(msg)
                original_tokens += len(content.split())
                compressed_tokens += len(content.split())
        else:
            # Short messages or system/assistant — pass through unchanged
            compressed_messages.append(msg)
            token_count = len(content.split())
            original_tokens += token_count
            compressed_tokens += token_count

    stats = {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": original_tokens - compressed_tokens,
        "compression_ratio": round(original_tokens / max(compressed_tokens, 1), 2),
    }

    return compressed_messages, stats
