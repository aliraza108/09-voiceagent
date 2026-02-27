"""
routes/chat.py — Text chat endpoints (Gemini LLM)
"""
import uuid
from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, SessionHistoryResponse, SessionClearResponse
from core.gemini_service import chat, get_session_history, clear_session, list_sessions

router = APIRouter(prefix="/chat", tags=["Text Chat"])


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a text message to the AI",
    description="Chat with Liam (Gemini-powered). Pass a session_id for memory, or omit for one-off responses.",
)
async def send_message(request: ChatRequest):
    """
    Text → Text chat with Gemini.
    - If session_id is provided → multi-turn conversation with memory
    - If session_id is omitted  → single-turn, stateless response
    """
    # Auto-generate session ID if not provided but user wants memory
    session_id = request.session_id

    try:
        reply, returned_session_id = await chat(
            user_message=request.message,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {str(e)}")

    return ChatResponse(reply=reply, session_id=returned_session_id or None)


@router.post(
    "/session/new",
    summary="Create a new conversation session",
    description="Returns a new unique session ID to use for multi-turn conversations.",
)
async def create_session():
    """Generate a new session ID for tracking conversation memory."""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id, "message": "New session created. Pass this session_id in your chat requests."}


@router.get(
    "/session/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get conversation history for a session",
)
async def get_history(session_id: str):
    history = get_session_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        history=history,
        turn_count=len(history),
    )


@router.delete(
    "/session/{session_id}",
    response_model=SessionClearResponse,
    summary="Clear a conversation session",
)
async def delete_session(session_id: str):
    cleared = clear_session(session_id)
    return SessionClearResponse(
        session_id=session_id,
        cleared=cleared,
        message="Session cleared." if cleared else "Session not found.",
    )


@router.get(
    "/sessions",
    summary="List all active session IDs",
)
async def get_sessions():
    sessions = list_sessions()
    return {"active_sessions": sessions, "count": len(sessions)}
