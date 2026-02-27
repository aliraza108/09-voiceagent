"""
models/schemas.py — Request / Response Pydantic schemas
"""
from pydantic import BaseModel, Field
from typing import Optional


# ─── TTS ──────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech", max_length=5000)
    voice_id: Optional[str] = Field(None, description="Override default voice ID")
    stability: float = Field(0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(0.8, ge=0.0, le=1.0)
    style: float = Field(0.4, ge=0.0, le=1.0, description="Speaking style expressiveness")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed")


class TTSResponse(BaseModel):
    audio_url: Optional[str] = None
    message: str = "Audio generated successfully"
    characters_used: int = 0


# ─── STT ──────────────────────────────────────────────
class STTResponse(BaseModel):
    transcript: str
    duration_seconds: Optional[float] = None
    message: str = "Transcription successful"


# ─── Chat ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., description="User text message", max_length=10000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")


class ChatResponse(BaseModel):
    reply: str
    session_id: Optional[str] = None


# ─── Voice to Voice ───────────────────────────────────
class VoiceToVoiceResponse(BaseModel):
    transcript: str
    reply_text: str
    session_id: Optional[str] = None
    message: str = "Voice response generated"


# ─── Session ──────────────────────────────────────────
class SessionHistoryResponse(BaseModel):
    session_id: str
    history: list[dict]
    turn_count: int


class SessionClearResponse(BaseModel):
    session_id: str
    cleared: bool
    message: str


# ─── Voices ───────────────────────────────────────────
class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    category: str
    description: str
    preview_url: str
