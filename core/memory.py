"""
EcoPrompt - Memory Layer
Stores reusable operational facts about a user or project.
Injects them into prompts to avoid repeated context.
"""

import json
import logging
import os

logger = logging.getLogger("ecoprompt.memory")

MEMORY_FILE = "./memory_store.json"


def _load() -> dict:
    """Load all memories from disk."""
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load memory: {e}")
        return {}


def _save(data: dict):
    """Save all memories to disk."""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save memory: {e}")


def memory_store(key: str, value: str):
    """
    Save a fact to memory.
    Example: memory_store("language", "Python")
    """
    data = _load()
    data[key.strip().lower()] = value.strip()
    _save(data)
    logger.info(f"Memory stored: {key} = {value}")


def memory_get(key: str) -> str:
    """
    Retrieve a specific fact from memory.
    Returns empty string if not found.
    """
    data = _load()
    return data.get(key.strip().lower(), "")


def memory_get_all() -> dict:
    """
    Retrieve all stored facts.
    """
    return _load()


def memory_delete(key: str):
    """
    Remove a specific fact from memory.
    """
    data = _load()
    if key.strip().lower() in data:
        del data[key.strip().lower()]
        _save(data)
        logger.info(f"Memory deleted: {key}")


def memory_clear():
    """
    Wipe all stored memories.
    """
    _save({})
    logger.info("Memory cleared.")

def memory_inject(messages: list) -> list:
    """
    Prepends stored memory facts into the message list as a system message.
    This gives the AI context without the user having to repeat themselves.

    Example:
        Input:  [{"role": "user", "content": "Write a database connection function"}]
        Output: [
            {"role": "system", "content": "Context: language=Python, framework=Flask, database=PostgreSQL"},
            {"role": "user", "content": "Write a database connection function"}
        ]
    """
    data = _load()

    if not data:
        # Nothing in memory — return messages unchanged
        return messages

    # Build a context string from all stored facts
    context_parts = [f"{k}={v}" for k, v in data.items()]
    context_string = "Context: " + ", ".join(context_parts)

    # Check if there's already a system message
    if messages and messages[0].get("role") == "system":
        # Append memory to existing system message instead of creating a new one
        existing = messages[0]["content"]
        updated_system = existing + "\n" + context_string
        return [{"role": "system", "content": updated_system}] + messages[1:]
    else:
        # Prepend a new system message with the memory context
        memory_message = {"role": "system", "content": context_string}
        return [memory_message] + messages
    
def parse_memory_command(message: str) -> tuple[bool, str, str]:
    """
    Checks if a message is a memory command.
    Format: "remember: key = value" or "remember: key is value"

    Returns:
        (is_memory_command, key, value)

    Examples:
        "remember: language = Python"  → (True, "language", "Python")
        "remember: my database is PostgreSQL" → (True, "database", "PostgreSQL")
        "What is Python?" → (False, "", "")
    """
    message = message.strip().lower()

    if not message.startswith("remember:"):
        return False, "", ""

    # Extract the part after "remember:"
    content = message[len("remember:"):].strip()

    # Try "key = value" format first
    if "=" in content:
        parts = content.split("=", 1)
        key = parts[0].strip()
        value = parts[1].strip()
        return True, key, value

    # Try "key is value" format
    if " is " in content:
        parts = content.split(" is ", 1)
        key = parts[0].strip()
        value = parts[1].strip()
        # Clean up common phrases like "my language is Python" → key = "language"
        key = key.replace("my ", "").replace("the ", "").strip()
        return True, key, value

    return False, "", ""


def handle_memory_command(messages: list) -> tuple[bool, str]:
    """
    Checks the latest user message for a memory command.
    If found, stores the fact and returns a confirmation message.

    Returns:
        (was_memory_command, confirmation_message)
    """
    if not messages:
        return False, ""

    # Get the last user message
    last_user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break

    is_command, key, value = parse_memory_command(last_user_message)

    if is_command and key and value:
        memory_store(key, value)
        confirmation = f"Got it! I'll remember that {key} = {value}."
        logger.info(f"Memory command processed: {key} = {value}")
        return True, confirmation

    return False, ""