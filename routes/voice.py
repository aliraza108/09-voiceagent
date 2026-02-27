"""
routes/voice.py — Voice-to-Voice full pipeline endpoints
Pipeline: Audio → STT → Gemini LLM → TTS → Audio
"""
import io
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import base64

from core.elevenlabs_service import speech_to_text, text_to_speech, list_voices
from core.gemini_service import chat

router = APIRouter(prefix="/voice", tags=["Voice to Voice"])

ALLOWED_EXTENSIONS = {"mp3", "wav", "webm", "ogg", "m4a", "flac"}


@router.post(
    "/talk",
    summary="🎙 Voice-to-Voice — Full Pipeline",
    description=(
        "Upload audio → transcribed → Gemini replies → Liam speaks back. "
        "Returns MP3 audio of the AI response."
    ),
    responses={200: {"content": {"audio/mpeg": {}}}},
)
async def voice_to_voice(
    audio: UploadFile = File(..., description="Your voice audio file"),
    session_id: Optional[str] = Form(None, description="Session ID for conversation memory"),
):
    """
    Full Voice-to-Voice pipeline.
    1. Receives audio upload
    2. Transcribes with ElevenLabs STT
    3. Sends transcript to Gemini
    4. Converts Gemini reply to Liam's voice
    5. Streams MP3 audio back
    """
    filename = audio.filename or "audio.webm"
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: .{ext}")

    # Step 1: Read audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Step 2: STT — user voice → text
    try:
        transcript = await speech_to_text(audio_bytes, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"STT failed: {str(e)}")

    if not transcript:
        raise HTTPException(status_code=422, detail="No speech detected in audio.")

    # Step 3: LLM — text → AI reply
    try:
        reply_text, _ = await chat(user_message=transcript, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    # Step 4: TTS — AI reply → Liam's voice audio
    try:
        reply_audio = await text_to_speech(reply_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {str(e)}")

    # Step 5: Stream audio back
    return StreamingResponse(
        io.BytesIO(reply_audio),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline; filename=liam_reply.mp3",
            "X-Transcript":       transcript[:200],   # User said...
            "X-Reply-Text":       reply_text[:200],   # AI said...
            "X-Session-Id":       session_id or "",
        },
    )


@router.post(
    "/talk/json",
    summary="🎙 Voice-to-Voice — Returns JSON with transcript + base64 audio",
    response_class=JSONResponse,
)
async def voice_to_voice_json(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """
    Same pipeline as /voice/talk but returns JSON with:
    - transcript (what you said)
    - reply_text (what Liam said)
    - audio_base64 (Liam's voice as base64 MP3)
    - session_id
    
    Useful for web apps / mobile apps.
    """
    filename = audio.filename or "audio.webm"
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: .{ext}")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # STT
    try:
        transcript = await speech_to_text(audio_bytes, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"STT failed: {str(e)}")

    if not transcript:
        raise HTTPException(status_code=422, detail="No speech detected.")

    # LLM
    try:
        reply_text, returned_session = await chat(user_message=transcript, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    # TTS
    try:
        reply_audio = await text_to_speech(reply_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TTS failed: {str(e)}")

    return {
        "transcript":    transcript,
        "reply_text":    reply_text,
        "audio_base64":  base64.b64encode(reply_audio).decode("utf-8"),
        "audio_format":  "mp3",
        "session_id":    returned_session or session_id,
        "message":       "Voice-to-voice complete",
    }


@router.get(
    "/voices",
    summary="List all ElevenLabs voices",
    description="Useful to find and confirm the Liam voice ID.",
)
async def get_voices():
    try:
        voices = await list_voices()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch voices: {str(e)}")
    
    # Highlight Liam if found
    liam_voices = [v for v in voices if "liam" in v["name"].lower()]
    return {
        "total": len(voices),
        "liam_matches": liam_voices,
        "all_voices": voices,
    }
