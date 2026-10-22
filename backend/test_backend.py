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

# -----------------------------------------------------------------------------
# Live segment transcription: /api/transcribe-segment (Phase 3.1). All Groq calls are mocked.
# -----------------------------------------------------------------------------
import logging
from backend.transcription import (
    transcribe_segment_groq,
    SegmentTranscriptionError,
    parse_retry_after,
    MAX_SEGMENT_PROMPT_CHARS,
)

def _verbose_payload(segments, text=None, duration=15.0):
    joined = "".join(seg["text"] for seg in segments) if text is None else text
    return {"text": joined, "language": "en", "duration": duration, "segments": segments}

def _seg(text, start=0.0, end=5.0, no_speech=0.01, logprob=-0.2, compression=1.3):
    return {"id": 0, "seek": 0, "start": start, "end": end, "text": text,
            "no_speech_prob": no_speech, "avg_logprob": logprob, "compression_ratio": compression}

@contextmanager
def segment_groq(handler):
    """Routes the endpoint's Groq calls to a MockTransport handler. Returns the list of requests made."""
    calls = []
    original = bmain.transcribe_segment_groq

    def recording_handler(request):
        request.read()
        calls.append(request)
        return handler(request)

    async def wrapped(filename, audio_bytes, prompt=None):
        async with httpx.AsyncClient(transport=httpx.MockTransport(recording_handler)) as http_client:
            return await original(filename, audio_bytes, prompt=prompt, client=http_client)

    bmain.transcribe_segment_groq = wrapped
    try:
        with groq_key(FAKE_GROQ_KEY):
            yield calls
    finally:
        bmain.transcribe_segment_groq = original

def post_segment(data=None, filename="segment-0001.wav", audio=None, include_file=True):
    files = {"file": (filename, io.BytesIO(make_valid_wav_bytes() if audio is None else audio), "audio/wav")} if include_file else None
    return client.post("/api/transcribe-segment", files=files, data=data or {})

def form_field(request, name):
    body = request.content.decode("latin1")
    marker = f'name="{name}"\r\n\r\n'
    if marker not in body:
        return None
    return body.split(marker, 1)[1].split("\r\n", 1)[0]

def assert_error(response, status_code, code, retryable=None):
    assert response.status_code == status_code, f"expected {status_code}, got {response.status_code}: {response.text}"
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == code, f"expected {code}, got {body['error']['code']}"
    assert set(body["error"].keys()) == {"code", "message", "retry_after_seconds", "retryable"}
    assert body["error"]["message"]
    if retryable is not None:
        assert body["error"]["retryable"] is retryable
    assert FAKE_GROQ_KEY not in response.text, "the Groq key leaked into an error response"
    return body

SEGMENT_RESPONSE_KEYS = {"success", "status", "sequence", "text", "duration_seconds", "language",
                         "segments", "filtered_segment_count", "model"}

def test_segment_success_structured_response():
    payload = _verbose_payload([_seg(" Hello team,", 0.0, 2.0), _seg(" let's start the review.", 2.0, 5.5)])
    with segment_groq(lambda r: httpx.Response(200, json=payload)) as calls:
        response = post_segment({"sequence": "7"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body.keys()) == SEGMENT_RESPONSE_KEYS
    assert body["success"] is True and body["status"] == "transcribed"
    assert body["sequence"] == 7
    assert body["text"] == "Hello team, let's start the review."
    assert body["model"] == "whisper-large-v3-turbo"
    assert body["language"] == "en" and body["duration_seconds"] == 15.0
    assert body["filtered_segment_count"] == 0
    assert [s["start"] for s in body["segments"]] == [0.0, 2.0]
    assert all(s["filtered"] is False and s["low_confidence"] is False for s in body["segments"])
    assert len(calls) == 1
    assert form_field(calls[0], "response_format") == "verbose_json"
    assert form_field(calls[0], "model") == "whisper-large-v3-turbo"
    assert form_field(calls[0], "prompt") is None, "no prompt was given, so none may be sent"
    print("  [PASS] test_segment_success_structured_response")

def test_segment_without_segment_details_uses_text():
    with segment_groq(lambda r: httpx.Response(200, json={"text": " Plain text only. "})):
        body = post_segment().json()
    assert body["status"] == "transcribed" and body["text"] == "Plain text only."
    assert body["segments"] == [] and body["sequence"] is None
    print("  [PASS] test_segment_without_segment_details_uses_text")

def test_segment_prompt_is_passed_for_continuity():
    with segment_groq(lambda r: httpx.Response(200, json=_verbose_payload([_seg(" continued.")]))) as calls:
        response = post_segment({"prompt": "  ...and the budget for Q3 is  "})
    assert response.status_code == 200
    assert form_field(calls[0], "prompt") == "...and the budget for Q3 is"
    with segment_groq(lambda r: httpx.Response(200, json=_verbose_payload([_seg(" x")]))) as calls:
        post_segment({"prompt": "   "})
    assert form_field(calls[0], "prompt") is None, "a blank prompt must not be sent"
    print("  [PASS] test_segment_prompt_is_passed_for_continuity")

def test_segment_rejects_excessive_or_malformed_prompt():
    with segment_groq(lambda r: httpx.Response(200, json={"text": "x"})) as calls:
        too_long = post_segment({"prompt": "a" * (MAX_SEGMENT_PROMPT_CHARS + 1)})
        control = post_segment({"prompt": "hello\x00world"})
        at_limit = post_segment({"prompt": "a" * MAX_SEGMENT_PROMPT_CHARS})
    assert_error(too_long, 400, "INVALID_PROMPT", retryable=False)
    assert_error(control, 400, "INVALID_PROMPT", retryable=False)
    assert at_limit.status_code == 200
    assert len(calls) == 1, "invalid prompts must be rejected before any Groq request"
    print("  [PASS] test_segment_rejects_excessive_or_malformed_prompt (no Groq call)")

def test_segment_silence_is_a_result_not_an_error():
    with segment_groq(lambda r: httpx.Response(200, json={"text": "   ", "segments": []})):
        empty = post_segment()
    assert empty.status_code == 200
    assert empty.json()["status"] == "silent" and empty.json()["text"] == ""

    hallucination = _verbose_payload([_seg(" Thank you.", no_speech=0.95, logprob=-1.4)])
    with segment_groq(lambda r: httpx.Response(200, json=hallucination)):
        body = post_segment().json()
    assert body["status"] == "silent" and body["text"] == ""
    assert body["filtered_segment_count"] == 1
    assert body["segments"][0]["filtered"] is True and body["segments"][0]["text"] == "Thank you."
    print("  [PASS] test_segment_silence_is_a_result_not_an_error (empty text and likely-silence filter)")

def test_segment_silence_filter_is_conservative():
    payload = _verbose_payload([
        _seg(" Quiet but confident words.", no_speech=0.92, logprob=-0.3),  # one signal only: kept
        _seg(" mumbled reply", no_speech=0.10, logprob=-1.6),                # one signal only: kept, flagged
        _seg(" yes yes yes yes yes", compression=3.1),                       # repetitive: kept, flagged
        _seg(" Thanks for watching.", no_speech=0.85, logprob=-1.2),         # both signals: filtered
    ])
    with segment_groq(lambda r: httpx.Response(200, json=payload)):
        body = post_segment().json()
    assert body["text"] == "Quiet but confident words. mumbled reply yes yes yes yes yes"
    assert [s["filtered"] for s in body["segments"]] == [False, False, False, True]
    assert [s["low_confidence"] for s in body["segments"]] == [False, True, True, False]
    assert body["filtered_segment_count"] == 1 and body["status"] == "transcribed"
    print("  [PASS] test_segment_silence_filter_is_conservative (one low value never drops speech)")

def test_segment_rejects_bad_audio_before_groq():
    oversized = make_valid_wav_bytes() + b"\x00" * (26 * 1024 * 1024)
    with segment_groq(lambda r: httpx.Response(200, json={"text": "x"})) as calls:
        empty = post_segment(audio=b"")
        pdf = post_segment(filename="notes.pdf", audio=b"%PDF-1.4")
        fake_webm = post_segment(filename="seg.webm", audio=make_valid_wav_bytes())
        big = post_segment(filename="seg.wav", audio=oversized)
        missing = post_segment(include_file=False)
        bad_seq = post_segment({"sequence": "-1"})
    assert_error(empty, 400, "EMPTY_AUDIO", retryable=False)
    assert_error(pdf, 400, "INVALID_AUDIO", retryable=False)
    assert_error(fake_webm, 400, "INVALID_AUDIO")
    assert_error(big, 413, "FILE_TOO_LARGE", retryable=False)
    assert_error(missing, 400, "INVALID_REQUEST")
    assert_error(bad_seq, 400, "INVALID_REQUEST")
    assert len(calls) == 0, "invalid segments must never reach Groq"
    print("  [PASS] test_segment_rejects_bad_audio_before_groq (empty, format, header, 25 MB, missing file, bad sequence)")

def test_segment_rate_limited_with_retry_after():
    handler = lambda r: httpx.Response(429, headers={"retry-after": "45"},
                                       json={"error": {"message": "Rate limit reached on requests per minute (RPM)"}})
    with segment_groq(handler) as calls:
        response = post_segment({"sequence": "3"})
    body = assert_error(response, 429, "RATE_LIMITED", retryable=True)
    assert body["error"]["retry_after_seconds"] == 45
    assert "45 seconds" in body["error"]["message"]
    assert response.headers.get("retry-after") == "45"
    assert body["sequence"] == 3
    assert len(calls) == 1, "a 429 must never be retried automatically"
    print("  [PASS] test_segment_rate_limited_with_retry_after (45 s passed through, no retry)")

def test_segment_rate_limited_without_retry_after_invents_nothing():
    with segment_groq(lambda r: httpx.Response(429, json={"error": {"message": "Too many requests"}})):
        response = post_segment()
    body = assert_error(response, 429, "RATE_LIMITED", retryable=True)
    assert body["error"]["retry_after_seconds"] is None
    assert "retry-after" not in response.headers
    assert not any(ch.isdigit() for ch in body["error"]["message"]), "no wait time may be invented"
    print("  [PASS] test_segment_rate_limited_without_retry_after_invents_nothing")

def test_segment_retry_after_parsing():
    assert parse_retry_after("45") == 45
    assert parse_retry_after("2.5") == 3
    assert parse_retry_after(" 0 ") == 0
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert parse_retry_after("-5") is None
    assert parse_retry_after("nan") is None
    print("  [PASS] test_segment_retry_after_parsing (numbers rounded up; dates and junk give no value)")

def test_segment_daily_quota_exhausted():
    rpd = lambda r: httpx.Response(429, json={"error": {"message": "Rate limit reached for model `whisper-large-v3-turbo` on requests per day (RPD): Limit 2000, Used 2000"}})
    asd = lambda r: httpx.Response(429, headers={"retry-after": "3600"},
                                   json={"error": {"message": "Rate limit reached on audio seconds per day (ASD)"}})
    ash = lambda r: httpx.Response(429, headers={"retry-after": "120"},
                                   json={"error": {"message": "Rate limit reached on audio seconds per hour (ASH)"}})
    with segment_groq(rpd) as calls:
        body = assert_error(post_segment(), 429, "QUOTA_EXHAUSTED", retryable=False)
    assert body["error"]["retry_after_seconds"] is None and len(calls) == 1
    with segment_groq(asd):
        body = assert_error(post_segment(), 429, "QUOTA_EXHAUSTED", retryable=False)
    assert body["error"]["retry_after_seconds"] == 3600 and "about 60 minutes" in body["error"]["message"]
    with segment_groq(ash):
        body = assert_error(post_segment(), 429, "RATE_LIMITED", retryable=True)
    assert body["error"]["retry_after_seconds"] == 120
    print("  [PASS] test_segment_daily_quota_exhausted (RPD/ASD = quota; hourly limit stays temporary)")

def test_segment_timeout():
    def handler(request):
        raise httpx.ReadTimeout("Read timed out")
    with segment_groq(handler) as calls:
        assert_error(post_segment(), 504, "TRANSCRIPTION_TIMEOUT", retryable=True)
    assert len(calls) == 1, "timeouts must not be retried"
    print("  [PASS] test_segment_timeout (no retry)")

def test_segment_service_failure_and_network_error():
    with segment_groq(lambda r: httpx.Response(503, text="busy")) as calls:
        assert_error(post_segment(), 503, "TRANSCRIPTION_UNAVAILABLE", retryable=True)
    assert len(calls) == 2, "transient 5xx is retried exactly once"

    def refused(request):
        raise httpx.ConnectError("connection refused")
    with segment_groq(refused) as calls:
        assert_error(post_segment(), 503, "TRANSCRIPTION_UNAVAILABLE", retryable=True)
    assert len(calls) == 2

    replies = [httpx.Response(502, text="bad gateway"), httpx.Response(200, json={"text": " Recovered."})]
    with segment_groq(lambda r: replies.pop(0)) as calls:
        response = post_segment()
    assert response.status_code == 200 and response.json()["text"] == "Recovered." and len(calls) == 2

    with segment_groq(lambda r: httpx.Response(200, text="<html>proxy page</html>")) as calls:
        assert_error(post_segment(), 502, "TRANSCRIPTION_UNAVAILABLE", retryable=True)
    assert len(calls) == 1
    print("  [PASS] test_segment_service_failure_and_network_error (one retry, then a clear error)")

def test_segment_groq_rejects_audio_or_key():
    with segment_groq(lambda r: httpx.Response(400, json={"error": {"message": "could not decode audio"}})):
        assert_error(post_segment(), 422, "INVALID_AUDIO", retryable=False)
    with segment_groq(lambda r: httpx.Response(401, json={"error": {"message": "Invalid API Key"}})):
        assert_error(post_segment(), 503, "TRANSCRIPTION_NOT_CONFIGURED", retryable=False)
    with groq_key(PLACEHOLDER_GROQ_KEY):
        assert_error(post_segment(), 503, "TRANSCRIPTION_NOT_CONFIGURED", retryable=False)
    print("  [PASS] test_segment_groq_rejects_audio_or_key")

def test_segment_internal_error_is_safe():
    original = bmain.transcribe_segment_groq

    async def broken(filename, audio_bytes, prompt=None):
        raise RuntimeError(f"secret internal path C:/srv/omniscribe with key {FAKE_GROQ_KEY}")

    bmain.transcribe_segment_groq = broken
    try:
        response = post_segment({"sequence": "11"})
    finally:
        bmain.transcribe_segment_groq = original
    body = assert_error(response, 500, "INTERNAL_ERROR", retryable=True)
    assert "secret" not in response.text and "C:/srv" not in response.text
    assert body["sequence"] == 11
    print("  [PASS] test_segment_internal_error_is_safe (no exception text or key in the response)")

def test_segment_never_touches_vault_analysis_or_web():
    touched = []
    originals = (bmain.save_and_read_vault_context, bmain.query_public_internet_fallback, bmain.get_anthropic_client)

    def forbid(name):
        def _f(*args, **kwargs):
            touched.append(name)
            raise AssertionError(f"{name} must not be called by /api/transcribe-segment")
        return _f

    bmain.save_and_read_vault_context = forbid("vault")
    bmain.query_public_internet_fallback = forbid("web search")
    bmain.get_anthropic_client = forbid("anthropic")
    try:
        # Real-AI mode on purpose: even then, the segment endpoint must not analyze anything
        with isolated_vault() as vault, mock_mode(False), \
                segment_groq(lambda r: httpx.Response(200, json=_verbose_payload([_seg(" Private words.")]))):
            response = post_segment({"prompt": "earlier private words"})
            assert response.status_code == 200
            assert list(vault.iterdir()) == [], "a segment was written to the Vault"
    finally:
        bmain.save_and_read_vault_context, bmain.query_public_internet_fallback, bmain.get_anthropic_client = originals
    assert touched == []
    assert set(response.json().keys()) == SEGMENT_RESPONSE_KEYS, "segment responses never include summaries or answers"
    print("  [PASS] test_segment_never_touches_vault_analysis_or_web")

def test_segment_does_not_log_transcript_or_prompt():
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Collect(level=logging.DEBUG)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        with segment_groq(lambda r: httpx.Response(200, json=_verbose_payload([_seg(" Confidential salary figures.")]))):
            post_segment({"prompt": "board meeting secrets"})
        with segment_groq(lambda r: httpx.Response(429, json={"error": {"message": "slow down"}})):
            post_segment({"prompt": "board meeting secrets"})
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    logged = "\n".join(records)
    assert "Confidential" not in logged and "secrets" not in logged
    print("  [PASS] test_segment_does_not_log_transcript_or_prompt")

def test_full_recording_request_is_unchanged():
    """The full-recording path still sends plain JSON format with no prompt, and returns plain text."""
    seen = []

    def handler(request):
        request.read()
        seen.append(request)
        return httpx.Response(200, json={"text": SPOKEN_TEXT})

    assert run_transcription(handler) == SPOKEN_TEXT
    assert form_field(seen[0], "response_format") == "json"
    assert form_field(seen[0], "model") == "whisper-large-v3-turbo"
    assert form_field(seen[0], "temperature") == "0.0"
    assert form_field(seen[0], "prompt") is None
    print("  [PASS] test_full_recording_request_is_unchanged (process-meeting still sends the same Groq request)")

# -----------------------------------------------------------------------------
# Analyze an existing transcript: /api/analyze-transcript (Milestone 3.2). No Groq or Anthropic calls.
# -----------------------------------------------------------------------------
from backend.main import MAX_TRANSCRIPT_CHARS, MAX_ANALYZE_BODY_BYTES, MAX_MEETING_NAME_CHARS

LIVE_TRANSCRIPT = "Hello team, let's start the review. The launch moves to Friday. Can someone update the client?"

@contextmanager
def no_transcription_allowed():
    """Fails the test if either Groq transcription path is used. Uses a placeholder key as a second guard."""
    calls = []
    originals = (bmain.transcribe_audio_groq, bmain.transcribe_segment_groq)

    async def forbidden(*args, **kwargs):
        calls.append("groq")
        raise AssertionError("analyze-transcript must never call Groq")

    bmain.transcribe_audio_groq = forbidden
    bmain.transcribe_segment_groq = forbidden
    try:
        with groq_key(PLACEHOLDER_GROQ_KEY):
            yield calls
    finally:
        bmain.transcribe_audio_groq, bmain.transcribe_segment_groq = originals

@contextmanager
def patched(name, replacement):
    original = getattr(bmain, name)
    setattr(bmain, name, replacement)
    try:
        yield
    finally:
        setattr(bmain, name, original)

def post_analyze(body=None, raw=None):
    if raw is not None:
        return client.post("/api/analyze-transcript", content=raw, headers={"Content-Type": "application/json"})
    return client.post("/api/analyze-transcript", json=body)

def assert_api_error(response, status_code, code, retryable=None):
    assert response.status_code == status_code, f"expected {status_code}, got {response.status_code}: {response.text}"
    body = response.json()
    assert body["success"] is False and body["error"]["code"] == code, body
    assert set(body["error"].keys()) == {"code", "message", "retry_after_seconds", "retryable"}
    assert body["error"]["retry_after_seconds"] is None
    if retryable is not None:
        assert body["error"]["retryable"] is retryable
    assert "Traceback" not in response.text
    return body

def test_analyze_success():
    with isolated_vault(), mock_mode(True), no_transcription_allowed() as groq_calls:
        response = post_analyze({"transcript": f"  {LIVE_TRANSCRIPT}  ", "meeting_name": "Launch Sync",
                                 "is_internal_context": False, "is_web_fallback": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True and body["meeting_name"] == "Launch Sync"
    assert body["transcript_segment"] == LIVE_TRANSCRIPT, "transcript is returned trimmed and otherwise unchanged"
    assert groq_calls == []
    print("  [PASS] test_analyze_success")

def test_analyze_matches_process_meeting_response():
    async def fake_stt(filename, audio_bytes):
        return LIVE_TRANSCRIPT

    form = {"meeting_name": "Same Shape", "is_internal_context": "false", "is_web_fallback": "true"}
    with isolated_vault(), mock_mode(True), stubbed_transcription(fake_stt):
        from_audio = client.post("/api/process-meeting", data=form,
                                 files={"file": ("m.wav", io.BytesIO(make_valid_wav_bytes()), "audio/wav")}).json()
    with isolated_vault(), mock_mode(True), no_transcription_allowed():
        from_text = post_analyze({"transcript": LIVE_TRANSCRIPT, "meeting_name": "Same Shape",
                                  "is_internal_context": False, "is_web_fallback": True}).json()
    assert set(from_text.keys()) == EXPECTED_RESPONSE_KEYS
    assert from_text == from_audio, "same transcript and settings must give the identical response"
    print("  [PASS] test_analyze_matches_process_meeting_response (identical body)")

def test_analyze_defaults_and_untitled_name():
    with isolated_vault() as vault, mock_mode(True), no_transcription_allowed():
        body = post_analyze({"transcript": LIVE_TRANSCRIPT}).json()
        assert list(vault.iterdir()) == [], "Vault saving must default to OFF"
    assert body["meeting_name"] == "Untitled meeting"
    assert body["qa_items"][1]["answer"].startswith("[MOCK]"), "web fallback defaults to ON (mock wording)"
    print("  [PASS] test_analyze_defaults_and_untitled_name (Vault off, web on, Untitled meeting)")

def test_analyze_rejects_empty_transcript():
    for transcript in ["", "   \n\t  "]:
        assert_api_error(post_analyze({"transcript": transcript}), 400, "EMPTY_TRANSCRIPT", retryable=False)
    assert_api_error(post_analyze({"meeting_name": "No transcript"}), 400, "INVALID_REQUEST")
    print("  [PASS] test_analyze_rejects_empty_transcript")

def test_analyze_rejects_oversized_transcript():
    with isolated_vault() as vault, mock_mode(True):
        at_limit = post_analyze({"transcript": "a" * MAX_TRANSCRIPT_CHARS})
        too_long = post_analyze({"transcript": "a" * (MAX_TRANSCRIPT_CHARS + 1), "is_internal_context": True})
        huge_body = post_analyze(raw=b'{"transcript": "' + b"a" * (MAX_ANALYZE_BODY_BYTES + 10) + b'"}')
        assert list(vault.iterdir()) == [], "a rejected transcript must not reach the Vault"
    assert at_limit.status_code == 200
    body = assert_api_error(too_long, 413, "TRANSCRIPT_TOO_LARGE", retryable=False)
    assert "aaaa" not in body["error"]["message"], "transcript text must not be echoed"
    assert_api_error(huge_body, 413, "TRANSCRIPT_TOO_LARGE")
    print("  [PASS] test_analyze_rejects_oversized_transcript (200,000 characters accepted, more rejected)")

def test_analyze_rejects_invalid_meeting_name():
    cases = [123, "x" * (MAX_MEETING_NAME_CHARS + 1), "Sync\x00name", "Line\nbreak"]
    with isolated_vault() as vault, mock_mode(True):
        for name in cases:
            assert_api_error(post_analyze({"transcript": LIVE_TRANSCRIPT, "meeting_name": name, "is_internal_context": True}),
                             400, "INVALID_MEETING_NAME", retryable=False)
        assert list(vault.iterdir()) == []
        # Names made only of symbols are accepted and filed as untitled, exactly like /api/process-meeting
        ok = post_analyze({"transcript": LIVE_TRANSCRIPT, "meeting_name": "???", "is_internal_context": True})
        assert ok.status_code == 200 and ok.json()["meeting_name"] == "???"
        assert (vault / "untitled-meeting.txt").exists()
    print("  [PASS] test_analyze_rejects_invalid_meeting_name (type, length, control characters)")

def test_analyze_vault_on_saves_exactly_once():
    writes = []
    original = bmain.save_and_read_vault_context

    def counting(name, text):
        writes.append(name)
        return original(name, text)

    with isolated_vault() as vault, mock_mode(True), no_transcription_allowed(), \
            patched("save_and_read_vault_context", counting):
        response = post_analyze({"transcript": LIVE_TRANSCRIPT, "meeting_name": "Vault Sync", "is_internal_context": True})
        assert response.status_code == 200
        saved = (vault / "Vault-Sync.txt").read_text(encoding="utf-8")
    assert writes == ["Vault-Sync"], f"expected one Vault write, got {writes}"
    assert saved == LIVE_TRANSCRIPT + "\n" and saved.count(LIVE_TRANSCRIPT) == 1
    print("  [PASS] test_analyze_vault_on_saves_exactly_once")

def test_analyze_vault_off_saves_nothing():
    def forbid(*args, **kwargs):
        raise AssertionError("Vault must not be written when is_internal_context is false")

    with isolated_vault() as vault, mock_mode(True), patched("save_and_read_vault_context", forbid):
        response = post_analyze({"transcript": LIVE_TRANSCRIPT, "meeting_name": "Private", "is_internal_context": False})
        assert response.status_code == 200
        assert list(vault.iterdir()) == []
    print("  [PASS] test_analyze_vault_off_saves_nothing")

def test_analyze_vault_failure_is_safe():
    def broken(name, text):
        raise OSError(r"[Errno 28] No space left on device: 'C:\Users\HP\omniscribe-ai\backend\vault\Board.txt'")

    with mock_mode(True), patched("save_and_read_vault_context", broken):
        response = post_analyze({"transcript": "Confidential board notes.", "meeting_name": "Board", "is_internal_context": True})
    body = assert_api_error(response, 500, "VAULT_WRITE_FAILED", retryable=True)
    assert "C:\\" not in response.text and "Errno" not in response.text and "Confidential" not in response.text
    assert "Vault" in body["error"]["message"]
    print("  [PASS] test_analyze_vault_failure_is_safe (no path, error number or transcript in the response)")

def test_process_meeting_vault_failure_unchanged():
    async def fake_stt(filename, audio_bytes):
        return SPOKEN_TEXT

    def broken(name, text):
        raise OSError("disk full")

    with mock_mode(True), stubbed_transcription(fake_stt), patched("save_and_read_vault_context", broken):
        response = client.post("/api/process-meeting", data={"meeting_name": "X", "is_internal_context": "true"},
                               files={"file": ("m.wav", io.BytesIO(make_valid_wav_bytes()), "audio/wav")})
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to update internal vault storage: disk full"}, "existing error must be unchanged"
    print("  [PASS] test_process_meeting_vault_failure_unchanged (same status and detail as before the refactor)")

def test_analyze_stays_in_mock_mode_without_web_or_claude():
    touched = []

    def forbid(name):
        def _f(*args, **kwargs):
            touched.append(name)
            raise AssertionError(name)
        return _f

    with isolated_vault(), mock_mode(True), no_transcription_allowed(), \
            patched("get_anthropic_client", forbid("anthropic")), \
            patched("query_public_internet_fallback", forbid("web search")):
        body = post_analyze({"transcript": "What is the typical onboarding time for B2B clients?",
                             "is_internal_context": True, "is_web_fallback": True}).json()
    assert touched == []
    assert body["summary"].startswith("[MOCK AI ANALYSIS]")
    assert all(item.startswith("[MOCK]") for item in body["action_items"])
    assert [qa["source"] for qa in body["qa_items"]] == ["none", "none"]
    print("  [PASS] test_analyze_stays_in_mock_mode_without_web_or_claude")

def test_analyze_malformed_requests():
    cases = {
        "not JSON": post_analyze(raw=b'{"transcript": "unterminated'),
        "invalid UTF-8": post_analyze(raw=b'{"transcript": "\xff\xfe"}'),
        "JSON array": post_analyze([LIVE_TRANSCRIPT]),
        "JSON string": post_analyze(raw=b'"just text"'),
        "empty body": post_analyze(raw=b""),
        "transcript not text": post_analyze({"transcript": 42}),
        "flag as string": post_analyze({"transcript": LIVE_TRANSCRIPT, "is_internal_context": "true"}),
        "flag as number": post_analyze({"transcript": LIVE_TRANSCRIPT, "is_web_fallback": 1}),
        "misspelled flag": post_analyze({"transcript": LIVE_TRANSCRIPT, "is_internal_contxt": True}),
        "null byte": post_analyze({"transcript": "abc\x00def"}),
    }
    for name, response in cases.items():
        try:
            assert_api_error(response, 400, "INVALID_REQUEST", retryable=False)
        except AssertionError as e:
            raise AssertionError(f"{name}: {e}")
    assert "is_internal_contxt" in cases["misspelled flag"].json()["error"]["message"]
    print("  [PASS] test_analyze_malformed_requests (bad JSON, wrong types, unknown fields; nothing is guessed)")

def test_analyze_unexpected_errors_are_safe():
    async def crash(*args, **kwargs):
        raise RuntimeError(f"secret stack detail C:/srv with key {FAKE_GROQ_KEY}")

    with isolated_vault(), mock_mode(True), patched("run_meeting_analysis", crash):
        response = post_analyze({"transcript": "Private remark."})
    assert_api_error(response, 500, "INTERNAL_ERROR", retryable=True)
    assert "secret" not in response.text and FAKE_GROQ_KEY not in response.text and "Private remark" not in response.text

    def missing_key():
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured.")
    with isolated_vault(), mock_mode(False), patched("get_anthropic_client", missing_key):
        response = post_analyze({"transcript": LIVE_TRANSCRIPT})
    assert_api_error(response, 503, "ANALYSIS_NOT_CONFIGURED", retryable=False)
    assert "ANTHROPIC_API_KEY" not in response.text

    async def failing_analysis(*args, **kwargs):
        raise HTTPException(status_code=502, detail="Anthropic API error: internal upstream detail")
    with isolated_vault(), mock_mode(True), patched("run_meeting_analysis", failing_analysis):
        response = post_analyze({"transcript": LIVE_TRANSCRIPT})
    assert_api_error(response, 502, "ANALYSIS_FAILED", retryable=True)
    assert "upstream" not in response.text
    print("  [PASS] test_analyze_unexpected_errors_are_safe (crash, missing AI key, analysis failure)")

def test_analyze_does_not_log_transcript():
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    def broken(name, text):
        raise OSError("disk full")

    handler = Collect(level=logging.DEBUG)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        with isolated_vault(), mock_mode(True):
            post_analyze({"transcript": "Salary bands for the new hires.", "meeting_name": "HR"})
        with mock_mode(True), patched("save_and_read_vault_context", broken):
            post_analyze({"transcript": "Salary bands for the new hires.", "is_internal_context": True})
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    assert "Salary" not in "\n".join(records)
    print("  [PASS] test_analyze_does_not_log_transcript")

if __name__ == "__main__":
    print("Starting OmniScribe AI backend test suite (all mocked, no Groq quota used)...")
    tests = [value for name, value in list(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"\nAll {len(tests)} automated verification tests passed successfully!")
