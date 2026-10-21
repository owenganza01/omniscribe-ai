import os
import io
import sys
import asyncio
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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
    extract_search_query,
    MAX_SEARCH_QUERY_CHARS,
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

# -----------------------------------------------------------------------------
# Web search flag parsing ([NEED_EXTERNAL_SEARCH: ...] emitted by Claude)
# -----------------------------------------------------------------------------
def test_search_flag_valid():
    text = '{"summary": "s"} [NEED_EXTERNAL_SEARCH: typical onboarding time B2B]'
    assert extract_search_query(text) == "typical onboarding time B2B"
    print("  [PASS] test_search_flag_valid")

def test_search_flag_unterminated_sends_no_query():
    # Previously returned a truncated query ('typical onboarding tim'); now must return nothing
    assert extract_search_query('{"summary": "s"} [NEED_EXTERNAL_SEARCH: typical onboarding time') == ""
    # A closing bracket belonging to later JSON must not complete the flag
    assert extract_search_query('[NEED_EXTERNAL_SEARCH: typical onboarding time\n{"action_items": ["x"]}') == ""
    assert extract_search_query('[NEED_EXTERNAL_SEARCH: typical onboarding time {"action_items": ["x"]}') == ""
    print("  [PASS] test_search_flag_unterminated_sends_no_query")

def test_search_flag_empty_query():
    assert extract_search_query('[NEED_EXTERNAL_SEARCH:   ] {"summary": "s"}') == ""
    assert extract_search_query('[NEED_EXTERNAL_SEARCH:]') == ""
    print("  [PASS] test_search_flag_empty_query")

def test_search_flag_absent():
    assert extract_search_query('{"summary": "no flag here"}') == ""
    print("  [PASS] test_search_flag_absent")

def test_search_flag_allows_quotes_and_braces():
    assert extract_search_query('[NEED_EXTERNAL_SEARCH: best "exact phrase" for {draft} notes]') == 'best "exact phrase" for {draft} notes'
    print("  [PASS] test_search_flag_allows_quotes_and_braces")

def test_search_flag_rejects_square_bracket_in_query():
    assert extract_search_query('[NEED_EXTERNAL_SEARCH: arr[0] index]') == ""
    print("  [PASS] test_search_flag_rejects_square_bracket_in_query")

def test_search_flag_length_limit():
    at_limit = "a" * MAX_SEARCH_QUERY_CHARS
    over_limit = "a" * (MAX_SEARCH_QUERY_CHARS + 1)
    assert extract_search_query(f"[NEED_EXTERNAL_SEARCH: {at_limit}]") == at_limit
    # Oversized queries are rejected whole, never truncated to the limit
    assert extract_search_query(f"[NEED_EXTERNAL_SEARCH: {over_limit}]") == ""
    print(f"  [PASS] test_search_flag_length_limit ({MAX_SEARCH_QUERY_CHARS} accepted, {MAX_SEARCH_QUERY_CHARS + 1} rejected)")

class _ScriptedMessages:
    """Mock Anthropic messages API: returns canned replies in order, counts calls, never reaches the network."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.prompts = []

    def create(self, model, max_tokens, messages):
        self.calls += 1
        self.prompts.append(messages)
        return SimpleNamespace(content=[SimpleNamespace(text=self.replies.pop(0))])

@contextmanager
def patched_attr(owner, name, value):
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)

def _recording_web_search(result="[Live Web Search]: stub result", error=None):
    """Mock search: records each query. Returns (fake_function, list_of_queries)."""
    queries = []

    async def fake_web_search(query):
        queries.append(query)
        if error is not None:
            raise error
        return result

    return fake_web_search, queries

def _post_with_scripted_claude(claude_replies, fake_web_search, web_on=True, vault_on=True):
    """Runs /api/process-meeting with a mocked Claude client and a mocked search.
    Returns (response, scripted_messages). No network access and no paid calls."""
    async def fake_stt(filename, audio_bytes):
        return SPOKEN_TEXT

    messages = _ScriptedMessages(claude_replies)
    fake_client = SimpleNamespace(messages=messages)

    with isolated_vault(), mock_mode(False), stubbed_transcription(fake_stt), \
            patched_attr(bmain, "query_public_internet_fallback", fake_web_search), \
            patched_attr(bmain, "get_anthropic_client", lambda: fake_client):
        response = client.post(
            "/api/process-meeting",
            files={"file": ("meeting.wav", io.BytesIO(make_valid_wav_bytes()), "audio/wav")},
            data={"meeting_name": "Flag Check", "is_web_fallback": str(web_on).lower(),
                  "is_internal_context": str(vault_on).lower()}
        )
    return response, messages

def test_unterminated_flag_does_not_trigger_web_search():
    fake_search, queries = _recording_web_search()
    response, messages = _post_with_scripted_claude(
        ['{"summary": "s", "action_items": [], "qa_items": []} [NEED_EXTERNAL_SEARCH: typical onboarding time'],
        fake_search,
    )

    assert response.status_code == 200
    assert queries == [], f"web search was called with {queries}"
    assert messages.calls == 1, "no refinement call should be made for a malformed flag"
    assert response.json()["summary"] == "s", "initial analysis must be kept"
    print("  [PASS] test_unterminated_flag_does_not_trigger_web_search")

def test_empty_flag_does_not_trigger_web_search_or_refinement():
    fake_search, queries = _recording_web_search()
    response, messages = _post_with_scripted_claude(
        ['{"summary": "s", "action_items": [], "qa_items": []} [NEED_EXTERNAL_SEARCH:   ]'],
        fake_search,
    )

    assert response.status_code == 200
    assert queries == [], f"web search was called with {queries}"
    assert messages.calls == 1, "no refinement call should be made for an empty flag"
    assert response.json()["summary"] == "s", "initial analysis must be kept"
    print("  [PASS] test_empty_flag_does_not_trigger_web_search_or_refinement")

def test_failed_web_search_preserves_initial_analysis():
    fake_search, queries = _recording_web_search(error=RuntimeError("simulated search outage"))
    response, messages = _post_with_scripted_claude(
        ['{"summary": "s", "action_items": ["a1"], "qa_items": [{"question": "q", "answer": "a", "source": "none"}]} '
         '[NEED_EXTERNAL_SEARCH: typical onboarding time B2B]'],
        fake_search,
    )

    assert response.status_code == 200, f"failed search must not become an error: {response.status_code}"
    assert queries == ["typical onboarding time B2B"], "the valid query should have been attempted"
    assert messages.calls == 1, "refinement must not run after the search failed"
    body = response.json()
    assert body["summary"] == "s"
    assert body["action_items"] == ["a1"]
    assert body["qa_items"][0]["question"] == "q" and body["qa_items"][0]["source"] == "none"
    print("  [PASS] test_failed_web_search_preserves_initial_analysis")

# -----------------------------------------------------------------------------
# Claude JSON response handling: valid JSON that is not an object must not cause a 500
# -----------------------------------------------------------------------------
NON_OBJECT_JSON = {
    "array": '["just", "a", "list"]',
    "string": '"just a string"',
    "number": "42",
    "boolean": "true",
    "null": "null",
}

def test_parse_valid_object_is_returned_unchanged():
    parsed = parse_claude_json_response('{"summary": "s", "action_items": ["a"], "qa_items": []}')
    assert parsed == {"summary": "s", "action_items": ["a"], "qa_items": []}
    print("  [PASS] test_parse_valid_object_is_returned_unchanged")

def test_parse_non_object_json_uses_fallback():
    for name, raw in NON_OBJECT_JSON.items():
        parsed = parse_claude_json_response(raw)
        assert isinstance(parsed, dict), f"{name}: parser returned {type(parsed).__name__}"
        assert parsed["action_items"] == [] and parsed["qa_items"] == [], name
        assert parsed["summary"] == raw, f"{name}: existing fallback should keep the raw text as summary"
    print("  [PASS] test_parse_non_object_json_uses_fallback (array, string, number, boolean, null)")

def test_parse_malformed_json_uses_fallback():
    parsed = parse_claude_json_response('{"summary": "unterminated')
    assert isinstance(parsed, dict) and parsed["action_items"] == [] and parsed["qa_items"] == []
    print("  [PASS] test_parse_malformed_json_uses_fallback")

def test_endpoint_non_object_json_does_not_return_500():
    for name, raw in NON_OBJECT_JSON.items():
        fake_search, queries = _recording_web_search()
        response, _ = _post_with_scripted_claude([raw], fake_search)
        assert response.status_code == 200, f"{name}: HTTP {response.status_code}"
        body = response.json()
        assert body["action_items"] == [] and body["qa_items"] == [], name
        assert queries == [], f"{name}: unexpected web search"
    print("  [PASS] test_endpoint_non_object_json_does_not_return_500 (array, string, number, boolean, null)")

def test_endpoint_valid_object_is_preserved():
    raw = ('{"summary": "Team agreed on Friday.", "action_items": ["Send notes"], '
           '"qa_items": [{"question": "When?", "answer": "Friday.", "source": "VAULT"}, '
           '{"question": "Where?", "answer": "Unknown.", "source": "bogus"}]}')
    response, _ = _post_with_scripted_claude([raw], _recording_web_search()[0])
    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "Team agreed on Friday."
    assert body["action_items"] == ["Send notes"]
    assert [q["source"] for q in body["qa_items"]] == ["vault", "none"], "source is lower-cased; unknown becomes none"
    print("  [PASS] test_endpoint_valid_object_is_preserved (known source kept, unknown source -> none)")

def test_endpoint_wrong_field_types_use_safe_defaults():
    raw = ('{"summary": null, "action_items": "not a list", '
           '"qa_items": [{"question": "q", "answer": "a", "source": 9}, "not an object"]}')
    response, _ = _post_with_scripted_claude([raw], _recording_web_search()[0])
    assert response.status_code == 200, f"HTTP {response.status_code}"
    body = response.json()
    assert body["summary"] == "Meeting segment processed successfully."
    assert body["action_items"] == []
    assert len(body["qa_items"]) == 1 and body["qa_items"][0]["source"] == "none"

    raw = '{"summary": 5, "action_items": ["x"], "qa_items": 7}'
    response, _ = _post_with_scripted_claude([raw], _recording_web_search()[0])
    assert response.status_code == 200, f"HTTP {response.status_code}"
    body = response.json()
    assert body["summary"] == "Meeting segment processed successfully."
    assert body["action_items"] == ["x"] and body["qa_items"] == []
    print("  [PASS] test_endpoint_wrong_field_types_use_safe_defaults")

# -----------------------------------------------------------------------------
# Public web search and refinement: failed or empty search must not be treated as evidence
# -----------------------------------------------------------------------------
_REAL_ASYNC_CLIENT = httpx.AsyncClient
FLAGGED_INITIAL = ('{"summary": "s", "action_items": ["a1"], "qa_items": [{"question": "q", "answer": "a", "source": "none"}]} '
                   '[NEED_EXTERNAL_SEARCH: typical onboarding time B2B]')

@contextmanager
def mocked_http(handler):
    """Routes every httpx.AsyncClient created by the backend through a MockTransport. No network access."""
    def factory(**kwargs):
        kwargs.pop("transport", None)
        return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)
    with patched_attr(httpx, "AsyncClient", factory):
        yield

def test_public_search_returns_none_when_no_results():
    requests_seen = []

    def unreachable(request):
        requests_seen.append(str(request.url))
        raise httpx.ConnectError("no network in tests")

    with mocked_http(unreachable):
        result = asyncio.run(bmain.query_public_internet_fallback("typical onboarding time"))
        empty = asyncio.run(bmain.query_public_internet_fallback("   "))
    assert result is None, f"failure must not be returned as text: {result!r}"
    assert empty is None
    assert len(requests_seen) == 2, "both DuckDuckGo steps should be attempted for a non-empty query"
    print("  [PASS] test_public_search_returns_none_when_no_results (both steps fail, empty query)")

def test_public_search_returns_instant_answer():
    def instant_answer(request):
        return httpx.Response(200, json={"AbstractText": "Typically 2 to 6 weeks.", "AbstractSource": "Wiki"})

    with mocked_http(instant_answer):
        result = asyncio.run(bmain.query_public_internet_fallback("typical onboarding time"))
    assert result == "[Live Web Search (Wiki)]: Typically 2 to 6 weeks."
    print("  [PASS] test_public_search_returns_instant_answer")

def test_no_public_results_skips_refinement():
    fake_search, queries = _recording_web_search(result=None)
    response, messages = _post_with_scripted_claude([FLAGGED_INITIAL], fake_search)

    assert response.status_code == 200
    assert queries == ["typical onboarding time B2B"], "the search should have been attempted"
    assert messages.calls == 1, "no refinement call when the search has no usable result"
    body = response.json()
    assert body["summary"] == "s" and body["action_items"] == ["a1"]
    assert body["qa_items"] == [{"question": "q", "answer": "a", "source": "none"}], "must not be relabelled as web"
    print("  [PASS] test_no_public_results_skips_refinement")

def test_valid_search_results_still_refine():
    refined = ('{"summary": "s", "action_items": ["a1"], "qa_items": '
               '[{"question": "q", "answer": "2 to 6 weeks.", "source": "web"}]}')
    fake_search, queries = _recording_web_search(result="[Live Web Search (Wiki)]: Typically 2 to 6 weeks.")
    response, messages = _post_with_scripted_claude([FLAGGED_INITIAL, refined], fake_search)

    assert response.status_code == 200
    assert messages.calls == 2, "a usable result should trigger exactly one refinement call"
    assert "Typically 2 to 6 weeks." in messages.prompts[1][-1]["content"], "refinement must receive the web result"
    body = response.json()
    assert body["qa_items"][0]["answer"] == "2 to 6 weeks." and body["qa_items"][0]["source"] == "web"
    print("  [PASS] test_valid_search_results_still_refine")

def test_malformed_refinement_preserves_initial_analysis():
    fake_search, _ = _recording_web_search(result="[Live Web Search]: stub result")
    response, messages = _post_with_scripted_claude(
        [FLAGGED_INITIAL, "Sorry, I could not update the JSON."], fake_search)

    assert response.status_code == 200
    assert messages.calls == 2
    body = response.json()
    assert body["summary"] == "s"
    assert body["action_items"] == ["a1"]
    assert body["qa_items"] == [{"question": "q", "answer": "a", "source": "none"}]
    print("  [PASS] test_malformed_refinement_preserves_initial_analysis")

def test_non_object_refinement_preserves_initial_analysis():
    invalid_refinements = {
        "array": '["x"]',
        "string": '"text"',
        "number": "42",
        "boolean": "true",
        "null": "null",
        "object with wrong field types": '{"summary": 5, "action_items": [], "qa_items": []}',
    }
    for name, refined in invalid_refinements.items():
        fake_search, _ = _recording_web_search(result="[Live Web Search]: stub result")
        response, _ = _post_with_scripted_claude([FLAGGED_INITIAL, refined], fake_search)
        assert response.status_code == 200, f"{name}: HTTP {response.status_code}"
        body = response.json()
        assert body["summary"] == "s", f"{name}: initial summary was replaced"
        assert body["action_items"] == ["a1"], f"{name}: initial action items were lost"
        assert body["qa_items"] == [{"question": "q", "answer": "a", "source": "none"}], f"{name}: initial Q&A was lost"
    print("  [PASS] test_non_object_refinement_preserves_initial_analysis (array, string, number, boolean, null, wrong types)")

# -----------------------------------------------------------------------------
# Source attribution: a source label is reported only when that source was actually used
# -----------------------------------------------------------------------------
WEB_LABEL_ANSWER = ('{"summary": "s", "action_items": [], "qa_items": '
                    '[{"question": "q", "answer": "2 to 6 weeks.", "source": "web"}]}')
VAULT_LABEL_ANSWER = ('{"summary": "s", "action_items": [], "qa_items": '
                      '[{"question": "q", "answer": "a", "source": "vault"}]}')

def test_web_disabled_downgrades_web_label():
    fake_search, queries = _recording_web_search()
    response, messages = _post_with_scripted_claude([WEB_LABEL_ANSWER], fake_search, web_on=False)
    assert response.status_code == 200
    assert response.json()["qa_items"][0]["source"] == "none", "web was disabled, so 'web' must not be reported"
    assert queries == [] and messages.calls == 1
    print("  [PASS] test_web_disabled_downgrades_web_label")

def test_vault_disabled_downgrades_vault_label():
    fake_search, _ = _recording_web_search()
    response, _ = _post_with_scripted_claude([VAULT_LABEL_ANSWER], fake_search, web_on=True, vault_on=False)
    assert response.status_code == 200
    assert response.json()["qa_items"][0]["source"] == "none", "vault context was disabled, so 'vault' must not be reported"
    print("  [PASS] test_vault_disabled_downgrades_vault_label")

def test_web_enabled_without_search_downgrades_web_label():
    fake_search, queries = _recording_web_search()
    response, messages = _post_with_scripted_claude([WEB_LABEL_ANSWER], fake_search, web_on=True)
    assert response.status_code == 200
    assert response.json()["qa_items"][0]["source"] == "none", "no search was requested, so 'web' must not be reported"
    assert queries == [] and messages.calls == 1
    print("  [PASS] test_web_enabled_without_search_downgrades_web_label")

def test_successful_vault_attribution_is_kept():
    fake_search, _ = _recording_web_search()
    response, _ = _post_with_scripted_claude([VAULT_LABEL_ANSWER], fake_search, web_on=False, vault_on=True)
    assert response.status_code == 200
    assert response.json()["qa_items"][0]["source"] == "vault", "vault context was enabled and used"
    print("  [PASS] test_successful_vault_attribution_is_kept")

def test_rejected_refinement_downgrades_unbacked_web_label():
    # The initial answer carries a web label, but the search refinement that would support it was rejected
    initial = ('{"summary": "s", "action_items": [], "qa_items": '
               '[{"question": "q", "answer": "2 to 6 weeks.", "source": "web"}]} '
               '[NEED_EXTERNAL_SEARCH: typical onboarding time B2B]')
    fake_search, queries = _recording_web_search(result="[Live Web Search (Wiki)]: Typically 2 to 6 weeks.")
    response, messages = _post_with_scripted_claude([initial, "Sorry, not JSON."], fake_search, web_on=True)
    assert response.status_code == 200
    assert queries == ["typical onboarding time B2B"] and messages.calls == 2
    body = response.json()
    assert body["summary"] == "s", "initial analysis must be kept"
    assert body["qa_items"][0]["source"] == "none", "the rejected refinement cannot back a 'web' label"
    print("  [PASS] test_rejected_refinement_downgrades_unbacked_web_label")

def test_mock_answers_report_no_source():
    async def fake_stt(filename, audio_bytes):
        return SPOKEN_TEXT

    with isolated_vault(), mock_mode(True), stubbed_transcription(fake_stt):
        body = client.post(
            "/api/process-meeting",
            files={"file": ("meeting.wav", io.BytesIO(make_valid_wav_bytes()), "audio/wav")},
            data={"meeting_name": "Mock Sources", "is_internal_context": "true", "is_web_fallback": "true"}
        ).json()
    assert [qa["source"] for qa in body["qa_items"]] == ["none", "none"], "mock answers perform no lookup"
    print("  [PASS] test_mock_answers_report_no_source")

if __name__ == "__main__":
    print("Starting OmniScribe AI Milestone 4 Phase B Test Suite (all mocked, no Groq quota used)...")
    tests = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"\nAll {len(tests)} automated verification tests passed successfully!")
