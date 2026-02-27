"""
routes/tts.py — Text-to-Speech endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import io
from models.schemas import TTSRequest
from core.elevenlabs_service import text_to_speech

router = APIRouter(prefix="/tts", tags=["Text to Speech"])


@router.post(
    "/generate",
    summary="Convert text to speech audio",
    description="Sends text to ElevenLabs and returns MP3 audio using the Liam voice (eleven_v3 model).",
    responses={200: {"content": {"audio/mpeg": {}}}},
)
async def generate_speech(request: TTSRequest):
    """
    Convert text → speech.
    Returns raw MP3 audio bytes for direct playback.
    """
    try:
        audio_bytes = await text_to_speech(
            text=request.text,
            voice_id=request.voice_id,
            stability=request.stability,
            similarity_boost=request.similarity_boost,
            style=request.style,
            speed=request.speed,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs TTS error: {str(e)}")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline; filename=response.mp3",
            "X-Characters-Used": str(len(request.text)),
        },
    )


@router.post(
    "/generate/json",
    summary="Convert text to speech (returns JSON + base64 audio)",
    response_class=JSONResponse,
)
async def generate_speech_json(request: TTSRequest):
    """
    Returns audio as base64-encoded string inside JSON.
    Useful for browser clients that prefer JSON over binary streaming.
    """
    import base64

    try:
        audio_bytes = await text_to_speech(
            text=request.text,
            voice_id=request.voice_id,
            stability=request.stability,
            similarity_boost=request.similarity_boost,
            style=request.style,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs TTS error: {str(e)}")

    return {
        "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
        "audio_format":  "mp3",
        "characters_used": len(request.text),
        "message": "Audio generated successfully",
    }
