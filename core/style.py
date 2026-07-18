"""
EcoPrompt - Lazy-Senior-Dev Style Injection
Inspired by ponytail (github.com/DietrichGebert/ponytail): a system-prompt
ruleset that nudges a coding-agent-driving caller toward minimal,
reuse-first code instead of over-engineered answers. Opt-in per request —
ecoprompt is a general chat proxy, most callers aren't asking for code at
all, so this stays off unless requested via the x-ecoprompt-style: lazy
header.
"""

LAZY_MODE_RULESET = (
    "You write the minimum code needed, nothing more. Before writing "
    "anything, work down this ladder and stop at the first rung that "
    "solves it:\n"
    "1. Does this need to exist at all? If the task can be done without "
    "new code, say so instead of writing it.\n"
    "2. Can existing code in this codebase already do it? Reuse before "
    "rewriting.\n"
    "3. Can the language/platform standard library do it? Prefer stdlib "
    "over a new dependency.\n"
    "4. Is there already an installed dependency that does it? Use it "
    "before adding a new one.\n"
    "5. Can it be done in one line? Don't build an abstraction for one "
    "call site.\n"
    "6. Otherwise, write the smallest, most direct code that correctly "
    "solves the task.\n\n"
    "Non-negotiable exceptions — never cut these for brevity: input "
    "validation at trust boundaries, security checks, error handling for "
    "real failure modes, accessibility, and anything preventing data "
    "loss. Minimal means no unneeded code, not no safety."
)


def is_lazy_mode(style_header: str) -> bool:
    return (style_header or "").strip().lower() == "lazy"


def inject_lazy_mode(messages: list) -> list:
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + LAZY_MODE_RULESET
    else:
        messages.insert(0, {"role": "system", "content": LAZY_MODE_RULESET})
    return messages
