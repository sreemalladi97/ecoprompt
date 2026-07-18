"""
Query rewriting layer for EcoPrompt semantic cache.
Resolves context-dependent follow-up questions into standalone cacheable queries.

Pipeline:
  raw message + history → extract topic → resolve pronouns → confidence check
                        → rewritten standalone query OR bypass cache
"""

import re
import hashlib
import logging

logger = logging.getLogger("ecoprompt.cache.rewriter")

# ── Pronoun signals ───────────────────────────────────────────────────────────
# If a message contains these, it's likely context-dependent
PRONOUN_SIGNALS = [
    r'\bit\b', r'\bits\b', r'\bthey\b', r'\btheir\b', r'\bthem\b',
    r'\bthese\b', r'\bthose\b', r'\bthat\b', r'\bthis\b',
    r'\bthe same\b', r'\bthe above\b', r'\bthe previous\b',
]

# ── Reference phrase signals ──────────────────────────────────────────────────
# Short contextual phrases that only make sense in conversation
REFERENCE_PHRASES = [
    'tell me more', 'more about', 'what about', 'go deeper', 'elaborate',
    'give me an example', 'what else', 'continue', 'and what', 'how about',
    'why is that', 'can you explain', 'what do you mean', 'are you sure',
    'why though', 'how so', 'what does that mean', 'say more',
]

# ── Short message threshold ───────────────────────────────────────────────────
# Messages under this word count are likely follow-ups
SHORT_MESSAGE_THRESHOLD = 8


def extract_topic(last_assistant_message: str) -> str:
    """
    Extracts the main topic/subject from the last AI response.
    Uses spaCy if available, falls back to simple noun extraction.
    Returns empty string if no clear topic found.
    """
    if not last_assistant_message:
        return ""

    try:
        import spacy
        # Use small model — already installed via sentence-transformers deps
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded — fall back to simple extraction
            return _simple_topic_extract(last_assistant_message)

        doc = nlp(last_assistant_message[:500])  # only look at first 500 chars

        # Prefer named entities (proper nouns like "Python", "Raspberry Pi")
        entities = [ent.text for ent in doc.ents if ent.label_ in
                    ("PRODUCT", "ORG", "PERSON", "GPE", "LANGUAGE", "WORK_OF_ART", "NORP")]
        if entities:
            return entities[0]

        # Fall back to first noun chunk
        chunks = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 4]
        if chunks:
            return chunks[0]

        return ""

    except ImportError:
        return _simple_topic_extract(last_assistant_message)


def _simple_topic_extract(text: str) -> str:
    """
    Fallback topic extraction without spaCy.
    Looks for capitalized multi-word phrases in the first sentence.
    """
    first_sentence = text.split('.')[0]
    # Find capitalized words/phrases (likely proper nouns)
    matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', first_sentence)
    # Filter out sentence starters (words at position 0)
    meaningful = [m for m in matches if m not in first_sentence[:len(m)+1]]
    return meaningful[0] if meaningful else ""


def has_pronoun_signals(text: str) -> bool:
    """Returns True if the message contains pronouns or reference phrases."""
    lower = text.lower()

    # Check reference phrases
    if any(phrase in lower for phrase in REFERENCE_PHRASES):
        return True

    # Check pronoun patterns
    if any(re.search(pattern, lower) for pattern in PRONOUN_SIGNALS):
        return True

    return False


def is_short_followup(text: str) -> bool:
    """Returns True if the message is suspiciously short (likely a follow-up)."""
    return len(text.strip().split()) < SHORT_MESSAGE_THRESHOLD


def resolve_query(messages: list) -> tuple[str, float]:
    """
    Takes the full conversation history and returns:
      (standalone_query, confidence)

    confidence: 1.0 = original message is self-contained, safe to cache as-is
                0.7 = resolved with topic injection, probably correct
                0.0 = can't resolve confidently, bypass cache

    The standalone_query is what gets used as the cache key.
    """
    if not messages:
        return "", 0.0

    # Get the last user message
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "").strip()
            break

    if not last_user:
        return "", 0.0

    # Get previous messages (everything before the last user message)
    history = messages[:-1] if messages[-1].get("role") == "user" else messages

    # No history — original message is standalone
    if not any(m.get("role") == "assistant" for m in history):
        return last_user, 1.0

    # Get last assistant message for topic extraction
    last_assistant = ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            last_assistant = m.get("content", "").strip()
            break

    # Check if message needs resolution
    needs_resolution = has_pronoun_signals(last_user) or is_short_followup(last_user)

    if not needs_resolution:
        # Message looks self-contained — cache as-is
        return last_user, 1.0

    # Try to resolve the topic
    topic = extract_topic(last_assistant)

    if not topic:
        # Can't extract topic — bypass cache entirely
        logger.info(f"Query rewriter: no topic found, bypassing cache")
        return last_user, 0.0

    # Replace pronouns with the extracted topic
    resolved = last_user
    pronoun_map = {
        r'\bit\b': topic,
        r'\bits\b': f"{topic}'s",
        r'\bthey\b': topic,
        r'\btheir\b': f"{topic}'s",
        r'\bthem\b': topic,
        r'\bthese\b': f"these {topic}",
        r'\bthose\b': f"those {topic}",
    }

    for pattern, replacement in pronoun_map.items():
        resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)

    # If nothing changed after substitution (only reference phrases triggered),
    # append topic as context
    if resolved.lower() == last_user.lower():
        resolved = f"{last_user} (about {topic})"

    logger.info(f"Query rewriter: '{last_user}' → '{resolved}' (topic: {topic})")
    return resolved, 0.7


def memory_fingerprint() -> str:
    """
    Returns a short hash of the current memory state.
    Used to scope cache lookups — same question in different memory contexts
    should not share cached answers.
    Returns empty string if memory is empty or unavailable.
    """
    try:
        from core.memory import memory_get_all
        data = memory_get_all()
        if not data:
            return ""
        # Sort keys for deterministic hash
        fingerprint_str = str(sorted(data.items()))
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:8]
    except Exception:
        return ""
