import os
import io
import sys
import asyncio
import tempfile
from contextlib import contextmanager
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from fastapi import HTTPException
from starlette.testclient import TestClient
from backend import main as bmain
from backend import transcription as stt
from backend.main import (
    app,
    sanitize_meeting_name,
    save_and_read_vault_context,
    parse_claude_json_response,
)
from backend.transcription import (
    transcribe_audio_groq,
    validate_audio_file,
    get_groq_api_key,
)

# All tests in this file are mocked: no test here can reach the real Groq API.
# Placeholder or fake keys are used, and every HTTP call goes through httpx.MockTransport.
FAKE_GROQ_KEY = "gsk_test_mock_key_12345"
PLACEHOLDER_GROQ_KEY = "your_groq_api_key_here"
SPOKEN_TEXT = "Hello, this is a test recording for OmniScribe AI. We are checking whether the application can accurately convert speech into text."

# Retry delay is shortened so retry tests run instantly
stt.RETRY_DELAY_SECONDS = 0.0

client = TestClient(app)

@contextmanager
def groq_key(value):
    original = os.environ.get("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = value
    try:
        yield
    finally:
        if original is not None:
            os.environ["GROQ_API_KEY"] = original
        else:
            os.environ.pop("GROQ_API_KEY", None)

@contextmanager
def isolated_vault():
    """Points the backend at a temporary vault so tests never touch real meeting data."""
    original = bmain.VAULT_DIR
    with tempfile.TemporaryDirectory() as tmp:
        bmain.VAULT_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            bmain.VAULT_DIR = original

@contextmanager
def stubbed_transcription(replacement):
    original = bmain.transcribe_audio_groq
    bmain.transcribe_audio_groq = replacement
    try:
        yield
    finally:
        bmain.transcribe_audio_groq = original

@contextmanager
def mock_mode(enabled: bool):
    original = bmain.MOCK_AI_RESPONSE
    bmain.MOCK_AI_RESPONSE = enabled
    try:
        yield
    finally:
        bmain.MOCK_AI_RESPONSE = original

def make_valid_wav_bytes(duration_samples: int = 160) -> bytes:
    """Generates a minimal valid PCM WAV byte sequence with proper RIFF/WAVE headers."""
    sample_rate = 8000
    data_size = duration_samples * 2
    buffer = bytearray(44 + data_size)
    buffer[0:4] = b"RIFF"
    buffer[4:8] = (36 + data_size).to_bytes(4, "little")
    buffer[8:12] = b"WAVE"
    buffer[12:16] = b"fmt "
    buffer[16:20] = (16).to_bytes(4, "little")
    buffer[20:22] = (1).to_bytes(2, "little")  # PCM
    buffer[22:24] = (1).to_bytes(2, "little")  # Mono
    buffer[24:28] = sample_rate.to_bytes(4, "little")
    buffer[28:32] = (sample_rate * 2).to_bytes(4, "little")
    buffer[32:34] = (2).to_bytes(2, "little")
    buffer[34:36] = (16).to_bytes(2, "little")
    buffer[36:40] = b"data"
    buffer[40:44] = data_size.to_bytes(4, "little")
    return bytes(buffer)

def run_transcription(handler, audio_bytes=None, filename="test.wav"):
    """Runs transcribe_audio_groq against a MockTransport handler. Returns transcript or raises."""
    async def _run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            return await transcribe_audio_groq(filename, audio_bytes or make_valid_wav_bytes(), client=http_client)
    with groq_key(FAKE_GROQ_KEY):
        return asyncio.run(_run())

def expect_http_error(fn, status_code: int, detail_fragment: str = ""):
    try:
        fn()
    except HTTPException as e:
        assert e.status_code == status_code, f"expected {status_code}, got {e.status_code}: {e.detail}"
        assert detail_fragment.lower() in e.detail.lower(), f"'{detail_fragment}' not in '{e.detail}'"
        return e
    raise AssertionError(f"Expected HTTPException {status_code}, but no error was raised")

# -----------------------------------------------------------------------------
# Existing backend behaviour (preserved from Milestone 3)
# -----------------------------------------------------------------------------
def test_sanitize_meeting_name():
    assert sanitize_meeting_name("Q4 Product Sync") == "Q4-Product-Sync"
    assert sanitize_meeting_name("Team/Review..Test") == "TeamReviewTest"
    assert sanitize_meeting_name("") == "untitled-meeting"
    assert sanitize_meeting_name(None) == "untitled-meeting"
    assert sanitize_meeting_name("   ") == "untitled-meeting"
    print("  [PASS] test_sanitize_meeting_name")

def test_vault_path_safety():
    with isolated_vault() as vault:
        safe_name = sanitize_meeting_name("Sprint-Planning-42")
        sample_text = "[Sample Transcript Segment]"
        history = save_and_read_vault_context(safe_name, sample_text)
        assert sample_text in history
        assert (vault / f"{safe_name}.txt").exists()
    print("  [PASS] test_vault_path_safety")

def test_parse_claude_json_response_fallback():
    parsed = parse_claude_json_response("not json at all")
    assert parsed["action_items"] == [] and parsed["qa_items"] == []
    print("  [PASS] test_parse_claude_json_response_fallback")

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "groq_configured" in data
    assert "anthropic_configured" in data
    assert "vault_path" in data
    print("  [PASS] test_health_endpoint")

# -----------------------------------------------------------------------------
# Validation (no network: every case is rejected before the Groq call)
# -----------------------------------------------------------------------------
def test_transcription_empty_file_validation():
    expect_http_error(lambda: validate_audio_file("empty.wav", b""), 400, "empty")
    print("  [PASS] test_transcription_empty_file_validation")

def test_transcription_unsupported_format_validation():
    expect_http_error(lambda: validate_audio_file("document.pdf", b"%PDF-1.4..."), 400, "unsupported")
    print("  [PASS] test_transcription_unsupported_format_validation")

def test_transcription_oversized_file_validation():
    # Simulate a file larger than 25 MB without allocating massive RAM
    class FakeOversizedBytes:
        def __len__(self):
            return 26 * 1024 * 1024

    expect_http_error(lambda: validate_audio_file("huge_meeting.wav", FakeOversizedBytes()), 413, "25 MB")
    print("  [PASS] test_transcription_oversized_file_validation (25 MB ceiling enforced)")

def test_transcription_filename_not_trusted_over_content():
    # A WAV payload renamed to .webm must be rejected because the WebM header is missing
    expect_http_error(lambda: validate_audio_file("fake.webm", make_valid_wav_bytes()), 400, "webm")
    print("  [PASS] test_transcription_filename_not_trusted_over_content")

def test_transcription_accepts_browser_formats():
    assert validate_audio_file("live.webm", b"\x1a\x45\xdf\xa3" + b"\x00" * 20)[1] == "audio/webm"
    assert validate_audio_file("live.m4a", b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 20)[1] == "audio/m4a"
    assert validate_audio_file("live.ogg", b"OggS" + b"\x00" * 20)[1] == "audio/ogg"
    assert validate_audio_file("live.mpeg", b"ID3" + b"\x00" * 20)[1] == "audio/mpeg"
    expect_http_error(lambda: validate_audio_file("bad.ogg", b"RIFF...."), 400, "ogg")
    expect_http_error(lambda: validate_audio_file("bad.mp3", b"\x00\x01\x02\x03"), 400, "mp3")
    print("  [PASS] test_transcription_accepts_browser_formats (webm, m4a, ogg, mp3/mpeg headers)")

def test_transcription_missing_api_key():
    with groq_key(PLACEHOLDER_GROQ_KEY):
        expect_http_error(get_groq_api_key, 503, "GROQ_API_KEY is not configured")
    print("  [PASS] test_transcription_missing_api_key")

# -----------------------------------------------------------------------------
# Transcription with mocked Groq responses (no quota consumed)
# -----------------------------------------------------------------------------
def test_transcription_mock_success_parsing():
    def handler(request):
        assert "api.groq.com" in str(request.url)
        assert request.headers["Authorization"] == f"Bearer {FAKE_GROQ_KEY}"
        return httpx.Response(200, json={"text": SPOKEN_TEXT})

    assert run_transcription(handler) == SPOKEN_TEXT
    print("  [PASS] test_transcription_mock_success_parsing")

def test_transcription_mock_rate_limit_429():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "15"},
                              json={"error": {"message": "Rate limit exceeded: please slow down."}})

    expect_http_error(lambda: run_transcription(handler), 429, "15 seconds")
    assert len(calls) == 1, "429 must never be retried"
    print("  [PASS] test_transcription_mock_rate_limit_429 (Retry-After respected, no retry)")

def test_transcription_mock_daily_quota_429():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"message": "Rate limit reached for model `whisper-large-v3-turbo` on requests per day (RPD): Limit 2000, Used 2000"}})

    expect_http_error(lambda: run_transcription(handler), 429, "daily quota")
    assert len(calls) == 1, "daily quota exhaustion must stop processing, not retry"
    print("  [PASS] test_transcription_mock_daily_quota_429 (daily quota stops processing)")

def test_transcription_mock_timeout():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("Read timed out")

    expect_http_error(lambda: run_transcription(handler), 504, "timed out")
    assert len(calls) == 1, "timeouts must never be retried"
    print("  [PASS] test_transcription_mock_timeout (no retry)")

def test_transcription_mock_server_error_retries_once():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="Service temporarily unavailable")

    expect_http_error(lambda: run_transcription(handler), 502, "Speech-to-text service error")
    assert len(calls) == 2, f"expected exactly one retry (2 calls), got {len(calls)}"
    print("  [PASS] test_transcription_mock_server_error_retries_once (retry capped at one)")

def test_transcription_mock_transient_error_then_success():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"text": SPOKEN_TEXT})

    assert run_transcription(handler) == SPOKEN_TEXT
    assert len(calls) == 2
    print("  [PASS] test_transcription_mock_transient_error_then_success")

def test_transcription_mock_connection_error_retries_once():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("connection refused")

    expect_http_error(lambda: run_transcription(handler), 503, "Network error")
    assert len(calls) == 2
    print("  [PASS] test_transcription_mock_connection_error_retries_once")

def test_transcription_mock_malformed_server_error():
    # Non-JSON 500 body (the original "malformed" case) must become a clean 502, not a crash
    expect_http_error(
        lambda: run_transcription(lambda request: httpx.Response(500, text="Internal Server Error Gateway")),
        502
    )
    print("  [PASS] test_transcription_mock_malformed_response (500 text body)")

def test_transcription_mock_malformed_200_not_json():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text="<html>unexpected proxy page</html>")

    expect_http_error(lambda: run_transcription(handler), 502, "unexpected response")
    assert len(calls) == 1, "a 200 response must not trigger a retry"
    print("  [PASS] test_transcription_mock_malformed_200_not_json")

def test_transcription_mock_malformed_200_missing_text():
    expect_http_error(
        lambda: run_transcription(lambda request: httpx.Response(200, json={"segments": []})),
        502, "unexpected response"
    )
    print("  [PASS] test_transcription_mock_malformed_200_missing_text")

def test_transcription_mock_silence_is_not_invented():
    expect_http_error(
        lambda: run_transcription(lambda request: httpx.Response(200, json={"text": "   "})),
        422, "no audible speech"
    )
    print("  [PASS] test_transcription_mock_silence_is_not_invented")

def test_transcription_mock_unauthorized():
    expect_http_error(
        lambda: run_transcription(lambda request: httpx.Response(401, json={"error": {"message": "Invalid API Key"}})),
        401, "GROQ_API_KEY"
    )
    print("  [PASS] test_transcription_mock_unauthorized")

def test_transcription_mock_groq_rejects_audio():
    expect_http_error(
        lambda: run_transcription(lambda request: httpx.Response(400, json={"error": {"message": "could not decode audio"}})),
        422, "could not read this audio file"
    )
    print("  [PASS] test_transcription_mock_groq_rejects_audio")

# -----------------------------------------------------------------------------
# Endpoint behaviour (/api/process-meeting)
# -----------------------------------------------------------------------------
EXPECTED_RESPONSE_KEYS = {"success", "meeting_name", "transcript_segment", "summary", "action_items", "qa_items"}

def test_endpoint_rejects_empty_upload():
    with groq_key(PLACEHOLDER_GROQ_KEY), mock_mode(True):
        response = client.post(
            "/api/process-meeting",
            files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
            data={"meeting_name": "Empty"}
        )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    print("  [PASS] test_endpoint_rejects_empty_upload")

def test_endpoint_rejects_unsupported_format():
    with groq_key(PLACEHOLDER_GROQ_KEY), mock_mode(True):
        response = client.post(
            "/api/process-meeting",
            files={"file": ("notes.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            data={"meeting_name": "PDF"}
        )
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()
    print("  [PASS] test_endpoint_rejects_unsupported_format")

def test_endpoint_rejects_oversized_before_groq_call():
    # Placeholder key: if the size check were skipped, the request would fail with 503 instead of 413
    oversized = make_valid_wav_bytes() + b"\x00" * (26 * 1024 * 1024)
    with groq_key(PLACEHOLDER_GROQ_KEY), mock_mode(True):
        response = client.post(
            "/api/process-meeting",
            files={"file": ("long-meeting.wav", io.BytesIO(oversized), "audio/wav")},
            data={"meeting_name": "Too Big"}
        )
    assert response.status_code == 413
    assert "25 MB" in response.json()["detail"]
    print("  [PASS] test_endpoint_rejects_oversized_before_groq_call")

def test_endpoint_missing_key_returns_503():
    with groq_key(PLACEHOLDER_GROQ_KEY), mock_mode(True):
        response = client.post(
            "/api/process-meeting",
            files={"file": ("meeting.wav", io.BytesIO(make_valid_wav_bytes()), "audio/wav")},
            data={"meeting_name": "No Key"}
        )
    assert response.status_code == 503
    assert "GROQ_API_KEY" in response.json()["detail"]
    print("  [PASS] test_endpoint_missing_key_returns_503")

def test_full_pipeline_with_mocked_transcription():
    """Preservation check: the existing endpoint contract still works with a stubbed transcription."""
    async def mock_stt(filename, audio_bytes):
        return SPOKEN_TEXT

    with isolated_vault() as vault, mock_mode(True), stubbed_transcription(mock_stt):
        response = client.post(
            "/api/process-meeting",
            files={"file": ("meeting-sample.wav", io.BytesIO(make_valid_wav_bytes()), "audio/wav")},
            data={"meeting_name": "Groq-Whisper-Verification", "is_internal_context": "true", "is_web_fallback": "true"}
        )
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == EXPECTED_RESPONSE_KEYS
        assert data["success"] is True
        assert data["meeting_name"] == "Groq-Whisper-Verification"
        assert data["transcript_segment"] == SPOKEN_TEXT
        assert len(data["action_items"]) > 0
        assert len(data["qa_items"]) > 0
        # Successful transcript is saved to the (temporary) vault
        assert SPOKEN_TEXT in (vault / "Groq-Whisper-Verification.txt").read_text(encoding="utf-8")
    print("  [PASS] test_full_pipeline_with_mocked_transcription (Preservation of existing backend behavior)")

def test_mock_analysis_is_labelled():
    async def mock_stt(filename, audio_bytes):
        return SPOKEN_TEXT

    with isolated_vault(), mock_mode(True), stubbed_transcription(mock_stt):
        data = client.post(
            "/api/process-meeting",
            files={"file": ("meeting.wav", io.BytesIO(make_valid_wav_bytes()), "audio/wav")},
            data={"meeting_name": "Label Check"}
        ).json()
    assert data["summary"].startswith("[MOCK AI ANALYSIS]")
    assert all(item.startswith("[MOCK]") for item in data["action_items"])
    assert all(qa["question"].startswith("[MOCK]") for qa in data["qa_items"])
    print("  [PASS] test_mock_analysis_is_labelled (mock output never looks like real AI output)")

def test_failed_transcription_is_not_saved_to_vault():
    async def failing_stt(filename, audio_bytes):
        raise HTTPException(status_code=429, detail="Groq free-tier daily quota has been used up.")

    with isolated_vault() as vault, mock_mode(True), stubbed_transcription(failing_stt):
        response = client.post(
            "/api/process-meeting",
            files={"file": ("meeting.wav", io.BytesIO(make_valid_wav_bytes()), "audio/wav")},
            data={"meeting_name": "Failed Run", "is_internal_context": "true"}
        )
        assert response.status_code == 429
        assert list(vault.iterdir()) == [], "failed transcriptions must not create vault files"
    print("  [PASS] test_failed_transcription_is_not_saved_to_vault")

if __name__ == "__main__":
    print("Starting OmniScribe AI Milestone 4 Phase B Test Suite (all mocked, no Groq quota used)...")
    tests = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"\nAll {len(tests)} automated verification tests passed successfully!")
