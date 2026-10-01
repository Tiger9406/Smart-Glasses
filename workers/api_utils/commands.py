from typing import Any


def command_name(command: dict[str, Any]) -> str:
    return (command.get("cmd") or "").upper()


def intent_event(
    command: dict[str, Any], result: dict[str, Any] | None, now: float
) -> dict[str, Any] | None:
    if not command.get("text") or result is None:
        return None
    return {
        "type": "intent",
        "cmd": result.get("cmd", "CHAT"),
        "args": result.get("args", {}),
        "timestamp": command.get("timestamp", now),
        "voice_embedding": command.get("voice_embedding"),
    }


def memory_event(
    command: dict[str, Any], result: Any, now: float, default_name: str
) -> dict[str, Any] | None:
    if not command.get("conversation_history"):
        return None
    return {
        "type": "memory_result",
        "subject": command.get("subject", default_name),
        "facts": result,
        "timestamp": now,
    }
