"""
routes/stt.py — Speech-to-Text endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import STTResponse
from core.elevenlabs_service import speech_to_text

router = APIRouter(prefix="/stt", tags=["Speech to Text"])

ALLOWED_EXTENSIONS = {"mp3", "wav", "webm", "ogg", "m4a", "flac"}
MAX_FILE_SIZE_MB = 25


@router.post(
    "/transcribe",
    response_model=STTResponse,
    summary="Transcribe audio file to text",
    description="Upload an audio file (mp3/wav/webm/ogg/m4a/flac) and receive a text transcription.",
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file to transcribe (max 25MB)"),
):
    """
    Speech → Text.
    Upload any supported audio format and get back the transcript.
    """
    # Validate file extension
    filename = audio.filename or "audio.webm"
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file
    audio_bytes = await audio.read()

    # Check size
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB",
        )

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received.")

    try:
        transcript = await speech_to_text(audio_bytes, filename=filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ElevenLabs STT error: {str(e)}")

    if not transcript:
        raise HTTPException(status_code=422, detail="No speech detected in audio.")

    return STTResponse(
        transcript=transcript,
        message="Transcription successful",
    )
