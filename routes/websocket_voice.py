"""
routes/websocket_voice.py
Real-time voice conversation via WebSocket.

Protocol:
  Client sends → binary audio chunk (webm/wav)
  Server responds → JSON: {"type": "transcript", "text": "..."}
                    JSON: {"type": "reply_text", "text": "..."}
                    binary: MP3 audio of AI reply
                    JSON: {"type": "done"}
"""
import uuid
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from core.elevenlabs_service import speech_to_text, text_to_speech
from core.gemini_service import chat

router = APIRouter(tags=["WebSocket Voice"])


@router.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice conversation with Liam.

    Connection URL: ws://localhost:8000/ws/voice?session_id=<optional>

    Flow per turn:
      1. Client sends binary audio bytes
      2. Server transcribes (STT) → sends JSON {"type":"transcript","text":"..."}
      3. Server gets Gemini reply → sends JSON {"type":"reply_text","text":"..."}
      4. Server sends binary MP3 audio
      5. Server sends JSON {"type":"done"}
    """
    await websocket.accept()
    
    # Get session_id from query params for conversation memory
    session_id = websocket.query_params.get("session_id") or str(uuid.uuid4())
    
    # Notify client of session ID
    await websocket.send_text(json.dumps({
        "type": "connected",
        "session_id": session_id,
        "message": f"Connected to Liam. Session: {session_id}",
    }))

    try:
        while True:
            # Receive audio chunk from client (binary)
            data = await websocket.receive()

            if "bytes" in data:
                audio_bytes = data["bytes"]
                
                if not audio_bytes:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Empty audio received.",
                    }))
                    continue

                # ── Step 1: STT ──────────────────────────────
                try:
                    transcript = await speech_to_text(audio_bytes, filename="audio.webm")
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"STT failed: {str(e)}",
                    }))
                    continue

                if not transcript:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "No speech detected.",
                    }))
                    continue

                # Send transcript to client
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "text": transcript,
                }))

                # ── Step 2: Gemini LLM ───────────────────────
                try:
                    reply_text, _ = await chat(
                        user_message=transcript,
                        session_id=session_id,
                    )
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"LLM failed: {str(e)}",
                    }))
                    continue

                # Send reply text to client
                await websocket.send_text(json.dumps({
                    "type": "reply_text",
                    "text": reply_text,
                }))

                # ── Step 3: TTS ──────────────────────────────
                try:
                    reply_audio = await text_to_speech(reply_text)
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"TTS failed: {str(e)}",
                    }))
                    continue

                # Send audio bytes to client
                await websocket.send_bytes(reply_audio)

                # Signal turn complete
                await websocket.send_text(json.dumps({"type": "done"}))

            elif "text" in data:
                # Handle text commands via WebSocket
                try:
                    cmd = json.loads(data["text"])
                    if cmd.get("action") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif cmd.get("action") == "clear_memory":
                        from core.gemini_service import clear_session
                        clear_session(session_id)
                        await websocket.send_text(json.dumps({
                            "type": "memory_cleared",
                            "session_id": session_id,
                        }))
                    elif cmd.get("action") == "chat_text":
                        # Text-only chat through WebSocket
                        msg = cmd.get("message", "")
                        if msg:
                            reply, _ = await chat(msg, session_id=session_id)
                            await websocket.send_text(json.dumps({
                                "type": "reply_text",
                                "text": reply,
                            }))
                except json.JSONDecodeError:
                    pass  # Ignore malformed text messages

    except WebSocketDisconnect:
        pass  # Client disconnected cleanly
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Unexpected error: {str(e)}",
            }))
        except Exception:
            pass
