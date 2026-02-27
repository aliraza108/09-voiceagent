"""
core/elevenlabs_service.py
Handles:
  - Text-to-Speech (TTS)  via ElevenLabs eleven_v3 + Liam voice
  - Speech-to-Text (STT)  via Gemini 2.0 Flash (primary — free, no plan needed)
                           with ElevenLabs scribe_v1 as fallback
  - Voice listing         via GET /v1/voices
"""
import httpx
import io
import base64
import google.generativeai as genai
from core.config import settings

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"

# Strip whitespace from keys — hidden spaces cause 401s
_EL_KEY = settings.ELEVENLABS_API_KEY.strip()

HEADERS = {
    "xi-api-key": _EL_KEY,
}

# Configure Gemini for STT
genai.configure(api_key=settings.GEMINI_API_KEY.strip())


# ─────────────────────────────────────────
# Text → Speech (ElevenLabs TTS)
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
    Convert text to audio bytes (mp3) using ElevenLabs Liam voice.
    Returns raw MP3 bytes ready to stream back to the client.
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
    }

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
# Speech → Text  (Gemini primary)
# ─────────────────────────────────────────
async def speech_to_text(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """
    Transcribe audio using Gemini 2.0 Flash (primary — no plan restrictions).
    Falls back to ElevenLabs scribe_v1 if Gemini fails.
    """
    try:
        return await _stt_gemini(audio_bytes, filename)
    except Exception as gemini_err:
        print(f"[STT] Gemini failed ({gemini_err}), trying ElevenLabs fallback...")
        try:
            return await _stt_elevenlabs(audio_bytes, filename)
        except Exception as el_err:
            raise RuntimeError(
                f"Both STT providers failed.\n"
                f"Gemini: {gemini_err}\n"
                f"ElevenLabs: {el_err}"
            )


async def _stt_gemini(audio_bytes: bytes, filename: str) -> str:
    """
    Transcribe audio using Gemini's native audio understanding.
    Works with any Gemini API key — no paid tier needed.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    mime_map = {
        "mp3":  "audio/mp3",
        "wav":  "audio/wav",
        "webm": "audio/webm",
        "ogg":  "audio/ogg",
        "m4a":  "audio/mp4",
        "flac": "audio/flac",
        "aac":  "audio/aac",
    }
    mime_type = mime_map.get(ext, "audio/webm")

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content([
        {
            "inline_data": {
                "mime_type": mime_type,
                "data": audio_b64,
            }
        },
        (
            "Transcribe the speech in this audio exactly as spoken. "
            "Return ONLY the transcribed text — no labels, no commentary, "
            "no formatting, no quotation marks. "
            "If no speech is detected, return an empty string."
        ),
    ])

    transcript = response.text.strip()

    # Strip any Gemini meta-commentary prefixes
    for prefix in ["Transcription:", "Transcript:", "The speaker says:", "Audio:", "Sure,"]:
        if transcript.lower().startswith(prefix.lower()):
            transcript = transcript[len(prefix):].strip()

    return transcript


async def _stt_elevenlabs(audio_bytes: bytes, filename: str) -> str:
    """
    ElevenLabs scribe_v1 STT — requires paid plan (used as fallback only).
    """
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
                "model_id": "scribe_v1",
                "language_code": "en",
            },
        )
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()


# ─────────────────────────────────────────
# List Available Voices (ElevenLabs)
# ─────────────────────────────────────────
async def list_voices() -> list[dict]:
    """
    Fetch all available voices from ElevenLabs.
    Use GET /voice/voices to confirm the Liam voice_id.
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