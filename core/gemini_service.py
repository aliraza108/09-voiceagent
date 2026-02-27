"""
core/gemini_service.py
Handles all LLM interactions via Google Gemini.
  - Single-turn chat response
  - Multi-turn conversation with memory
  - Session-based chat history management
"""
import google.generativeai as genai
from typing import Optional
from core.config import settings

# Configure Gemini client
genai.configure(api_key=settings.GEMINI_API_KEY)

# In-memory session store: session_id → list of messages
# Format: [{"role": "user"|"model", "parts": ["text"]}]
_session_store: dict[str, list[dict]] = {}


def _get_model() -> genai.GenerativeModel:
    """Create a Gemini model instance with the agent's system prompt."""
    return genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        system_instruction=settings.AGENT_SYSTEM_PROMPT,
    )


def _get_or_create_session(session_id: str) -> list[dict]:
    """Return existing session history or create new one."""
    if session_id not in _session_store:
        _session_store[session_id] = []
    return _session_store[session_id]


async def chat(
    user_message: str,
    session_id: Optional[str] = None,
) -> tuple[str, str]:
    """
    Send a message to Gemini and get a response.
    
    Args:
        user_message: The user's input text
        session_id:   If provided, maintains conversation memory across turns
    
    Returns:
        Tuple of (response_text, session_id)
    """
    model = _get_model()

    if session_id:
        # Multi-turn with memory
        history = _get_or_create_session(session_id)
        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(user_message)
        
        # Persist updated history
        _session_store[session_id] = chat_session.history
        return response.text, session_id
    else:
        # Single-turn, no memory
        response = model.generate_content(user_message)
        return response.text, ""


def get_session_history(session_id: str) -> list[dict]:
    """Get conversation history for a session."""
    history = _session_store.get(session_id, [])
    result = []
    for turn in history:
        role = turn.role if hasattr(turn, "role") else turn.get("role", "")
        parts = turn.parts if hasattr(turn, "parts") else turn.get("parts", [])
        text = ""
        for p in parts:
            if hasattr(p, "text"):
                text += p.text
            elif isinstance(p, str):
                text += p
        result.append({"role": role, "text": text})
    return result


def clear_session(session_id: str) -> bool:
    """Clear conversation history for a session."""
    if session_id in _session_store:
        del _session_store[session_id]
        return True
    return False


def list_sessions() -> list[str]:
    """Return all active session IDs."""
    return list(_session_store.keys())
