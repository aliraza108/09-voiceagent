"""
core/elevenlabs_service.py
Handles:
  - Text-to-Speech  (TTS)  via ElevenLabs eleven_v3 + Liam voice
  - Speech-to-Text  (STT)  via ElevenLabs /v1/speech-to-text
  - Voice listing         via GET /v1/voices
"""
import httpx
import io
from core.config import settings

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"

HEADERS = {
    "xi-api-key": settings.ELEVENLABS_API_KEY,
}


# ─────────────────────────────────────────
# Text → Speech
# ─────────────────────────────────────────
async def text_to_speech(
    text: str,
    voice_id: str = None,
    model_id: str = None,
    stability: float = 0.5,
    similarity_boost: float = 0.8,
    style: float = 0.4,
    speed: float = 1.0,
) -> bytes:
    """
    Convert text to audio bytes (mp3) using ElevenLabs.
    Returns raw audio bytes ready to stream back to the client.
    """
    voice_id = voice_id or settings.ELEVENLABS_VOICE_ID
    model_id = model_id or settings.ELEVENLABS_MODEL_ID

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": True,
        },
        "seed": None,
        "pronunciation_dictionary_locators": [],
    }

    # eleven_v3 supports output_format param
    params = {"output_format": "mp3_44100_128"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}",
            json=payload,
            headers={**HEADERS, "Content-Type": "application/json"},
            params=params,
        )
        response.raise_for_status()
        return response.content


# ─────────────────────────────────────────
# Speech → Text
# ─────────────────────────────────────────
async def speech_to_text(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """
    Transcribe audio bytes using ElevenLabs Speech-to-Text API.
    Supports: mp3, wav, webm, ogg, m4a, flac
    Returns transcribed text string.
    """
    # Determine mime type from filename
    ext = filename.rsplit(".", 1)[-1].lower()
    mime_map = {
        "mp3":  "audio/mpeg",
        "wav":  "audio/wav",
        "webm": "audio/webm",
        "ogg":  "audio/ogg",
        "m4a":  "audio/mp4",
        "flac": "audio/flac",
    }
    mime_type = mime_map.get(ext, "audio/webm")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ELEVENLABS_BASE}/speech-to-text",
            headers=HEADERS,
            files={
                "file": (filename, io.BytesIO(audio_bytes), mime_type),
            },
            data={
                "model_id": "scribe_v1",   # ElevenLabs STT model
                "language_code": "en",
            },
        )
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()


# ─────────────────────────────────────────
# List Available Voices
# ─────────────────────────────────────────
async def list_voices() -> list[dict]:
    """
    Fetch all available voices from ElevenLabs.
    Use this to find/confirm the Liam voice_id.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{ELEVENLABS_BASE}/voices",
            headers=HEADERS,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "voice_id":    v["voice_id"],
                "name":        v["name"],
                "category":    v.get("category", ""),
                "description": v.get("description", ""),
                "preview_url": v.get("preview_url", ""),
            }
            for v in data.get("voices", [])
        ]
