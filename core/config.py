"""
core/config.py — Centralised settings loaded from .env with hardcoded fallbacks
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from the project root (voiceagent/) regardless of working directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)


class Settings:
    # ── ElevenLabs ──────────────────────────────────────────
    ELEVENLABS_API_KEY: str  = os.getenv(
        "ELEVENLABS_API_KEY",
        "sk_d050e1c613dd1ed6be31e0a4307dc6a18b41fd71a086ccc1"   # ← your key
    )
    ELEVENLABS_VOICE_ID: str = os.getenv(
        "ELEVENLABS_VOICE_ID",
        "TX3LPaxmHKxFdv7VOQHJ"   # Liam — confirm via GET /voice/voices
    )
    ELEVENLABS_MODEL_ID: str = os.getenv(
        "ELEVENLABS_MODEL_ID",
        "eleven_v3"               # Best quality, supports Liam's style
    )

    # ── Google Gemini ────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv(
        "GEMINI_API_KEY",
        "AIzaSyBUovu9TP8pzULOB8asarFB1eg5_JSCdqQ"              # ← your key
    )
    GEMINI_MODEL: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.0-flash"
    )

    # ── Agent Personality ────────────────────────────────────
    AGENT_NAME: str = os.getenv("AGENT_NAME", "Liam")
    AGENT_SYSTEM_PROMPT: str = os.getenv(
        "AGENT_SYSTEM_PROMPT",
        (
            "You are Liam, an energetic and engaging AI voice assistant with the personality "
            "of a social media creator. You speak naturally, keep responses concise (2-3 sentences "
            "max for voice), use casual and upbeat language, and always sound enthusiastic. "
            "Never use markdown or bullet points — speak in plain conversational sentences only."
        )
    )

    # ── Server ───────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()


# ── Sanity check on startup ──────────────────────────────────
def validate_settings():
    missing = []
    if not settings.ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if not settings.GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required config keys: {', '.join(missing)}\n"
            f"Check your .env file at: {_env_path}"
        )
    print(f"✅ Config loaded — Agent: {settings.AGENT_NAME} | "
          f"Voice: {settings.ELEVENLABS_MODEL_ID} | LLM: {settings.GEMINI_MODEL}")