"""
EcoPrompt - Content-Type Detection
Ported from headroom's ContentRouter idea: LLMLingua is tuned for prose,
so running it on code, JSON, or log/stack-trace text risks corrupting
syntax for a marginal token saving. Detect those cases and let the
compressor skip them rather than compress everything uniformly.

Heuristic, not a parser — deliberately biased toward over-detecting
code/json/logs rather than under-detecting: mistaking prose for code
just costs a missed compression opportunity (safe), but mistaking code
for prose means LLMLingua mangles it (harmful). When unsure, this
returns the non-prose classification.
"""

import json
import re

# Structural markers that are rare in ordinary English regardless of topic
# (parens-with-def, arrows, comparison operators, #include, ```, ...).
_STRONG_CODE_SIGNALS = [
    "```", "def ", "function(", "function (", "#include",
    "=>", "==", "!=", "();", "() {", "{}",
]

# Keywords that double as ordinary English words or commonly appear in
# prose *about* code/SQL (e.g. "why does SELECT ... FROM ... work") — only
# meaningful as a code signal when several show up together, or the line
# is short enough to plausibly be a pasted snippet rather than a sentence.
_WEAK_CODE_SIGNALS = [
    "class ", "import ", "SELECT ", "FROM ", "const ", "let ", "var ",
    "public class", "private ", "return ",
]

_CODE_SIGNALS = _STRONG_CODE_SIGNALS + _WEAK_CODE_SIGNALS

_LOG_LINE_RE = re.compile(
    r"^\s*(\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2}[,.]?\d*)\s*[\]\-]?\s*"
    r"(ERROR|WARN|WARNING|INFO|DEBUG|TRACE|FATAL)\b",
    re.IGNORECASE | re.MULTILINE,
)
_TRACEBACK_RE = re.compile(
    r"Traceback \(most recent call last\)|^\s*at \S+\(.*\)|File \"[^\"]+\", line \d+",
    re.MULTILINE,
)


def detect_content_type(text: str) -> str:
    """
    Returns "json", "logs", "code", or "prose" for the given text.
    """
    stripped = text.strip()
    if not stripped:
        return "prose"

    # JSON: only worth the parse attempt if it's actually bracket-shaped —
    # avoids paying for a failed parse on every plain sentence.
    if stripped[0] in "{[" and stripped[-1] in "}]":
        try:
            json.loads(stripped)
            return "json"
        except (ValueError, TypeError):
            pass

    # Logs: timestamped log lines or a stack trace. Requires 2+ log-level
    # lines so a single stray "ERROR" mentioned in a sentence doesn't misfire.
    if len(_LOG_LINE_RE.findall(stripped)) >= 2 or _TRACEBACK_RE.search(stripped):
        return "logs"

    if "```" in stripped:
        return "code"

    lines = stripped.splitlines()
    if len(lines) > 1:
        # Multi-line: require a real density of code-like lines, not just
        # a mention of a keyword — e.g. four lines of prose that each
        # explain a different OOP concept ("A class defines...", "The
        # return statement...") aren't code just because each line
        # contains one weak signal. A line only counts as code-like via a
        # weak signal if it's also short (real code lines are terse;
        # explanatory sentences aren't) — a strong signal counts on its own.
        def _line_is_codelike(line):
            if any(sig in line for sig in _STRONG_CODE_SIGNALS):
                return True
            weak_hits = sum(1 for sig in _WEAK_CODE_SIGNALS if sig in line)
            return weak_hits >= 1 and len(line.split()) <= 8

        code_lines = sum(1 for line in lines if _line_is_codelike(line))
        if code_lines / len(lines) >= 0.3:
            return "code"
    else:
        # Single line: API callers send most prose as one line with no
        # embedded newlines, so weak signals (English words that double as
        # keywords, like "class" or "return") only count as code when
        # several appear together AND the line is short enough to
        # plausibly be a pasted snippet — a long sentence that happens to
        # mention two programming terms is still just a sentence. A
        # strong signal (rare in ordinary English regardless of length)
        # is enough on its own.
        word_count = len(stripped.split())
        if any(sig in stripped for sig in _STRONG_CODE_SIGNALS):
            return "code"
        weak_hits = sum(1 for sig in _WEAK_CODE_SIGNALS if sig in stripped)
        if weak_hits >= 2 and word_count <= 15:
            return "code"

    return "prose"
