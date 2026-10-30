"""Services package for MedTwin AI."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.chat_service import ChatService, get_chat_service
    from app.services.voice_service import VoiceService, get_voice_service


def __getattr__(name: str):
    """Lazy export of service classes to prevent eager import lock contention."""
    if name in ("ChatService", "get_chat_service"):
        from app.services.chat_service import ChatService, get_chat_service
        return ChatService if name == "ChatService" else get_chat_service
    if name in ("VoiceService", "get_voice_service"):
        from app.services.voice_service import VoiceService, get_voice_service
        return VoiceService if name == "VoiceService" else get_voice_service
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ChatService", "get_chat_service", "VoiceService", "get_voice_service"]
