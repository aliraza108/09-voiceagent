"""
main.py — VoiceAgent AI Backend
FastAPI application entry point

Endpoints:
  POST /tts/generate              → Text → Speech (MP3 stream)
  POST /tts/generate/json         → Text → Speech (base64 JSON)
  POST /stt/transcribe            → Audio → Text
  POST /chat/message              → Text → AI Text reply
  POST /chat/session/new          → Create new session
  GET  /chat/session/{id}/history → Get conversation history
  DELETE /chat/session/{id}       → Clear session
  POST /voice/talk                → Audio → AI Audio (MP3 stream)
  POST /voice/talk/json           → Audio → AI Audio (base64 JSON)
  GET  /voice/voices              → List all ElevenLabs voices
  WS   /ws/voice                  → Real-time voice conversation

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings, validate_settings
from routes import tts, stt, chat, voice, websocket_voice

# Validate config on startup
validate_settings()

# ─── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="🎙 VoiceAgent AI",
    description=(
        "Autonomous Conversational Voice Intelligence Platform powered by "
        "ElevenLabs (Liam voice, eleven_v3 model) + Google Gemini.\n\n"
        "**Capabilities:** Text-to-Speech · Speech-to-Text · Voice-to-Voice · "
        "Real-time WebSocket conversations with memory."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────
app.include_router(tts.router)
app.include_router(stt.router)
app.include_router(chat.router)
app.include_router(voice.router)
app.include_router(websocket_voice.router)

# ─── Health & Root ────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "service":    "VoiceAgent AI",
        "status":     "running",
        "version":    "1.0.0",
        "agent":      settings.AGENT_NAME,
        "voice_model": settings.ELEVENLABS_MODEL_ID,
        "llm_model":   settings.GEMINI_MODEL,
        "docs":       "/docs",
        "endpoints": {
            "tts":           "POST /tts/generate",
            "stt":           "POST /stt/transcribe",
            "chat":          "POST /chat/message",
            "voice_to_voice": "POST /voice/talk",
            "websocket":     "WS /ws/voice",
            "list_voices":   "GET /voice/voices",
        },
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "agent": settings.AGENT_NAME}


# ─── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )