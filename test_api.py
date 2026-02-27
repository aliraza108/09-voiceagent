"""
test_api.py — Quick manual tests for VoiceAgent AI
Run: python test_api.py

Make sure the server is running first:
  uvicorn main:app --reload --port 8000
"""
import httpx
import asyncio
import json
import base64
import os

BASE_URL = "http://localhost:8000"


async def test_health():
    print("\n🔍 Testing /health ...")
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/health")
        print(f"  Status: {r.status_code}")
        print(f"  Response: {r.json()}")


async def test_tts(text: str = "Hey there! I'm Liam, your AI voice assistant. Let's build something amazing together!"):
    print(f"\n🔊 Testing TTS: '{text[:60]}...'")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{BASE_URL}/tts/generate/json",
            json={"text": text, "style": 0.5, "speed": 1.0}
        )
        if r.status_code == 200:
            data = r.json()
            audio_b64 = data["audio_base64"]
            # Save to file
            with open("test_tts_output.mp3", "wb") as f:
                f.write(base64.b64decode(audio_b64))
            print(f"  ✅ TTS success! Audio saved → test_tts_output.mp3 ({len(audio_b64)} base64 chars)")
        else:
            print(f"  ❌ TTS failed: {r.status_code} — {r.text}")


async def test_chat(message: str = "Tell me something energetic about AI!"):
    print(f"\n💬 Testing Chat: '{message}'")
    async with httpx.AsyncClient(timeout=30) as client:
        # Create session
        r_session = await client.post(f"{BASE_URL}/chat/session/new")
        session_id = r_session.json()["session_id"]
        print(f"  Session: {session_id}")
        
        # Send message
        r = await client.post(
            f"{BASE_URL}/chat/message",
            json={"message": message, "session_id": session_id}
        )
        if r.status_code == 200:
            data = r.json()
            print(f"  ✅ Liam: {data['reply']}")
        else:
            print(f"  ❌ Chat failed: {r.status_code} — {r.text}")


async def test_stt(audio_path: str = None):
    if not audio_path or not os.path.exists(audio_path):
        print("\n🎙 STT test skipped — provide an audio file path as argument")
        return
    
    print(f"\n🎙 Testing STT with: {audio_path}")
    async with httpx.AsyncClient(timeout=30) as client:
        with open(audio_path, "rb") as f:
            r = await client.post(
                f"{BASE_URL}/stt/transcribe",
                files={"audio": (os.path.basename(audio_path), f, "audio/mpeg")}
            )
        if r.status_code == 200:
            print(f"  ✅ Transcript: {r.json()['transcript']}")
        else:
            print(f"  ❌ STT failed: {r.status_code} — {r.text}")


async def test_voices():
    print("\n🎭 Fetching ElevenLabs voices to find Liam...")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{BASE_URL}/voice/voices")
        if r.status_code == 200:
            data = r.json()
            print(f"  Total voices: {data['total']}")
            if data["liam_matches"]:
                for v in data["liam_matches"]:
                    print(f"  🎯 LIAM FOUND: voice_id={v['voice_id']} | name={v['name']}")
            else:
                print("  ⚠️  Liam not found by name. Check /voice/voices for full list.")
        else:
            print(f"  ❌ Failed: {r.status_code} — {r.text}")


async def main():
    print("=" * 60)
    print("  VoiceAgent AI — Test Suite")
    print("=" * 60)
    
    await test_health()
    await test_voices()
    await test_chat()
    await test_tts()
    
    print("\n" + "=" * 60)
    print("  Tests complete! Check test_tts_output.mp3 for audio.")
    print("  WebSocket test: ws://localhost:8000/ws/voice")
    print("  Full docs:      http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
