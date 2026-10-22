import os
import math
import asyncio
from pathlib import Path
from typing import Tuple, Optional
import httpx
from fastapi import HTTPException, status

# -----------------------------------------------------------------------------
# Groq Whisper Large V3 Turbo Configuration (Zero-Payment Free Cloud STT)
# -----------------------------------------------------------------------------
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB Groq free-tier ceiling
GROQ_WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
GROQ_TIMEOUT = httpx.Timeout(120.0, connect=10.0)  # Large uploads need more than the default 5s

# Only transient server errors are retried, and only once. HTTP 429 and timeouts are never retried,
# so a single recording cannot silently spend the free quota twice.
TRANSIENT_STATUS_CODES = {500, 502, 503, 504}
RETRY_DELAY_SECONDS = 2.0

# Groq includes the limit type in 429 messages, e.g. "on requests per day (RPD)"
DAILY_QUOTA_MARKERS = ("per day", "(rpd)", "(asd)", "(tpd)")

# Formats officially supported by Groq Audio API
SUPPORTED_EXTENSIONS = {
    ".webm", ".m4a", ".mp3", ".wav", ".mp4", ".ogg", ".flac", ".mpeg", ".mpga"
}

def get_groq_api_key() -> str:
    """
    Safely retrieves the Groq API key from environment variables.
    Raises an informative HTTP 503 error if the key is missing or is the placeholder.
    """
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is not configured. Please add your free Groq API key to the backend .env file."
        )
    return key

def validate_audio_file(filename: str, audio_bytes: bytes) -> Tuple[str, str]:
    """
    Validates audio file presence, size limits, file extension, and container headers.
    Returns (sanitized_filename, mime_type).
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    # 1. Size Validation (Zero byte check & 25 MB ceiling)
    size = len(audio_bytes)
    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded audio payload is empty (0 bytes)."
        )
    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File size ({size / (1024 * 1024):.1f} MB) exceeds Groq's 25 MB limit for speech-to-text."
        )

    # 2. Extension Validation
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{ext}'. Supported formats: {supported_list}."
        )

    # 3. Magic Bytes / Header Validation (the filename alone is never trusted)
    mime_type = "application/octet-stream"
    if ext == ".wav":
        if not (audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File has .wav extension but contains an invalid WAV container header."
            )
        mime_type = "audio/wav"
    elif ext == ".webm":
        if not audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File has .webm extension but contains an invalid WebM container header."
            )
        mime_type = "audio/webm"
    elif ext in (".mp4", ".m4a"):
        if b"ftyp" not in audio_bytes[:16]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File has {ext} extension but contains an invalid MP4/M4A container header."
            )
        mime_type = "audio/mp4" if ext == ".mp4" else "audio/m4a"
    elif ext == ".ogg":
        if not audio_bytes.startswith(b"OggS"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File has .ogg extension but contains an invalid OGG container header."
            )
        mime_type = "audio/ogg"
    elif ext == ".flac":
        if not audio_bytes.startswith(b"fLaC"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File has .flac extension but contains an invalid FLAC container header."
            )
        mime_type = "audio/flac"
    elif ext in (".mp3", ".mpeg", ".mpga"):
        # MPEG audio begins with an ID3 tag or an MPEG sync word (0xFF 0xEx)
        is_id3 = audio_bytes.startswith(b"ID3")
        is_sync = len(audio_bytes) > 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0
        if not (is_id3 or is_sync):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File has {ext} extension but contains an invalid MPEG audio header."
            )
        mime_type = "audio/mpeg"

    sanitized_filename = Path(filename).name
    return sanitized_filename, mime_type

def _extract_error_message(response: httpx.Response) -> str:
    """Returns Groq's human-readable error message, or an empty string if none is present."""
    try:
        err_json = response.json()
    except ValueError:
        return ""
    if isinstance(err_json, dict) and isinstance(err_json.get("error"), dict):
        message = err_json["error"].get("message")
        if isinstance(message, str):
            return message
    return ""

def _extract_transcript(response: httpx.Response) -> str:
    """
    Reads the transcript from a 200 response. Raises a descriptive error for malformed
    payloads and for audio that contained no speech, instead of returning invented text.
    """
    try:
        payload = response.json()
    except ValueError:
        payload = None

    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Speech-to-text service returned an unexpected response. Please try again."
        )

    transcript_text = text.strip()
    if not transcript_text:
        raise HTTPException(
            status_code=422,
            detail="Audio processed, but no audible speech was detected in the recording."
        )
    return transcript_text

def _rate_limit_error(response: httpx.Response) -> HTTPException:
    """Builds the user-facing error for HTTP 429, distinguishing daily quota from short-term limits."""
    message = _extract_error_message(response).lower()
    if any(marker in message for marker in DAILY_QUOTA_MARKERS):
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The Groq free-tier daily quota for speech-to-text has been used up. Processing has stopped and no paid service was used. Please try again after the quota resets."
        )

    retry_after = response.headers.get("Retry-After", "")
    detail = "Groq free-tier rate limit reached."
    if retry_after:
        detail += f" Please wait {retry_after} seconds before trying again."
    else:
        detail += " Please wait a moment before sending another recording."
    return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)

async def _send_transcription_request(
    client: httpx.AsyncClient,
    headers: dict,
    filename: str,
    audio_bytes: bytes,
    mime_type: str,
    form_fields: Optional[dict] = None
) -> httpx.Response:
    """
    Sends one multipart request to Groq. Keeps no state, so it can be safely repeated.
    Without form_fields it sends exactly what the full-recording path has always sent.
    """
    return await client.post(
        GROQ_WHISPER_URL,
        headers=headers,
        files={"file": (filename, audio_bytes, mime_type)},
        data=form_fields or {"model": GROQ_WHISPER_MODEL, "response_format": "json", "temperature": "0.0"}
    )

async def _post_with_single_retry(
    client: httpx.AsyncClient,
    headers: dict,
    filename: str,
    audio_bytes: bytes,
    mime_type: str,
    form_fields: Optional[dict] = None
) -> httpx.Response:
    """
    Sends the request, retrying at most once for transient 5xx responses or dropped connections.
    Timeouts propagate immediately without a retry.
    """
    try:
        response = await _send_transcription_request(client, headers, filename, audio_bytes, mime_type, form_fields)
    except httpx.TimeoutException:
        raise
    except httpx.RequestError:
        response = None

    if response is not None and response.status_code not in TRANSIENT_STATUS_CODES:
        return response

    await asyncio.sleep(RETRY_DELAY_SECONDS)
    return await _send_transcription_request(client, headers, filename, audio_bytes, mime_type, form_fields)

async def transcribe_audio_groq(
    filename: str,
    audio_bytes: bytes,
    client: Optional[httpx.AsyncClient] = None
) -> str:
    """
    Transcribes audio using Groq's hosted Whisper Large V3 Turbo API.
    Raises descriptive HTTPException on failures without crashing the server.
    Never returns simulated or invented text.
    """
    # 1. Validate file format and enforce 25 MB limit
    sanitized_filename, mime_type = validate_audio_file(filename, audio_bytes)

    # 2. Check Groq API Key (never logged or echoed back to the client)
    api_key = get_groq_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    should_close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=GROQ_TIMEOUT)
        should_close_client = True

    try:
        response = await _post_with_single_retry(client, headers, sanitized_filename, audio_bytes, mime_type)

        # 3. Handle Successful Response
        if response.status_code == 200:
            return _extract_transcript(response)

        # 4. Handle Specific Groq API Error Status Codes
        if response.status_code == 429:
            raise _rate_limit_error(response)

        if response.status_code == 401:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Groq authentication failed. Please verify your GROQ_API_KEY in .env."
            )

        if response.status_code == 413:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail="Audio file exceeds Groq's 25 MB maximum upload limit."
            )

        if response.status_code == 400:
            raise HTTPException(
                status_code=422,
                detail=f"Groq could not read this audio file. Try re-recording or exporting it as MP3, WAV, or M4A. ({_extract_error_message(response) or 'no details'})"
            )

        # General Groq API Error Message Extraction
        err_detail = _extract_error_message(response) or f"Groq API error ({response.status_code})"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Speech-to-text service error: {err_detail}"
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Transcription request timed out. Please try again with a shorter recording."
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Network error connecting to Groq transcription service: {type(e).__name__}"
        )
    finally:
        if should_close_client:
            await client.aclose()


# -----------------------------------------------------------------------------
# Live segment transcription (Phase 3.1)
# -----------------------------------------------------------------------------
# One short, independently decodable audio segment is transcribed per request. Unlike
# transcribe_audio_groq above, failures are raised as SegmentTranscriptionError with a stable
# machine-readable code, so the future live-transcription frontend can tell a rate limit from a
# used-up quota, a timeout or bad audio. Silence is a normal result ("silent"), not an error.
# Nothing here writes to the Vault, logs transcript text, or keeps the audio.

MAX_SEGMENT_PROMPT_CHARS = 500   # Groq accepts up to 224 prompt tokens; 500 characters stays well inside that

# Whisper's own heuristic treats a segment as silence only when BOTH signals agree.
# A single low value is never enough to drop text.
NO_SPEECH_THRESHOLD = 0.8        # no_speech_prob at or above this...
LOW_LOGPROB_THRESHOLD = -1.0     # ...AND avg_logprob at or below this marks likely silence
HIGH_COMPRESSION_RATIO = 2.4     # repetitive output; flagged as low confidence, never dropped


class SegmentTranscriptionError(Exception):
    """A segment could not be transcribed. Carries a safe message and a stable error code."""

    def __init__(self, status_code: int, code: str, message: str,
                 retry_after_seconds: Optional[int] = None, retryable: bool = False):
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retry_after_seconds = retry_after_seconds
        self.retryable = retryable

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "retry_after_seconds": self.retry_after_seconds,
            "retryable": self.retryable,
        }


def parse_retry_after(value: Optional[str]) -> Optional[int]:
    """
    Returns Retry-After as whole seconds (rounded up), or None when absent or not a number.
    An HTTP-date or unreadable value gives None: a wait time is never guessed.
    """
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    if math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return None
    return int(math.ceil(seconds))


def _describe_wait(seconds: int) -> str:
    if seconds < 120:
        return f"{seconds} seconds"
    if seconds < 7200:
        return f"about {round(seconds / 60)} minutes"
    return f"about {round(seconds / 3600)} hours"


def validate_segment_prompt(prompt: Optional[str]) -> Optional[str]:
    """Returns the cleaned continuity prompt, None when there is none, or raises INVALID_PROMPT."""
    if prompt is None:
        return None
    if not isinstance(prompt, str):
        raise SegmentTranscriptionError(400, "INVALID_PROMPT", "The context prompt must be text.")
    cleaned = prompt.strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_SEGMENT_PROMPT_CHARS:
        raise SegmentTranscriptionError(
            400, "INVALID_PROMPT",
            f"The context prompt is too long ({len(cleaned)} characters, maximum {MAX_SEGMENT_PROMPT_CHARS}). "
            "Send only the end of the previous transcript."
        )
    if any((ord(ch) < 32 and ch not in "\n\r\t") or ord(ch) == 127 for ch in cleaned):
        raise SegmentTranscriptionError(400, "INVALID_PROMPT", "The context prompt contains unsupported control characters.")
    return cleaned


def validate_segment_audio(filename: str, audio_bytes: bytes) -> Tuple[str, str]:
    """Checks one segment before any Groq call. Reuses validate_audio_file for format and header checks."""
    if not filename:
        raise SegmentTranscriptionError(400, "INVALID_AUDIO", "The audio segment must have a filename.")
    size = len(audio_bytes)
    if size == 0:
        raise SegmentTranscriptionError(400, "EMPTY_AUDIO", "The audio segment is empty (0 bytes), so there was nothing to transcribe.")
    if size > MAX_FILE_SIZE_BYTES:
        raise SegmentTranscriptionError(
            413, "FILE_TOO_LARGE",
            f"The audio segment is {size / (1024 * 1024):.1f} MB, above the 25 MB speech-to-text limit."
        )
    try:
        return validate_audio_file(filename, audio_bytes)
    except HTTPException as e:
        # Our own validation messages (format or header problems) are safe to show
        raise SegmentTranscriptionError(400, "INVALID_AUDIO", str(e.detail))


def _finite_number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _build_segment_result(payload) -> dict:
    """Turns a Groq verbose_json payload into the endpoint's result, applying the conservative silence filter."""
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        raise SegmentTranscriptionError(
            502, "TRANSCRIPTION_UNAVAILABLE",
            "The speech-to-text service returned an unexpected response. This audio segment was not transcribed.",
            retryable=True
        )

    pieces = []
    kept_text = []
    filtered_count = 0
    raw_segments = payload.get("segments")
    for seg in raw_segments if isinstance(raw_segments, list) else []:
        if not isinstance(seg, dict):
            continue
        seg_text = seg.get("text") if isinstance(seg.get("text"), str) else ""
        no_speech = _finite_number(seg.get("no_speech_prob"))
        logprob = _finite_number(seg.get("avg_logprob"))
        compression = _finite_number(seg.get("compression_ratio"))
        likely_silence = (no_speech is not None and logprob is not None
                          and no_speech >= NO_SPEECH_THRESHOLD and logprob <= LOW_LOGPROB_THRESHOLD)
        low_confidence = not likely_silence and (
            (logprob is not None and logprob <= LOW_LOGPROB_THRESHOLD)
            or (compression is not None and compression >= HIGH_COMPRESSION_RATIO)
        )
        pieces.append({
            "start": _finite_number(seg.get("start")),
            "end": _finite_number(seg.get("end")),
            "text": seg_text.strip(),
            "no_speech_prob": no_speech,
            "avg_logprob": logprob,
            "compression_ratio": compression,
            "filtered": likely_silence,
            "low_confidence": low_confidence,
        })
        if likely_silence:
            filtered_count += 1
        elif seg_text.strip():
            kept_text.append(seg_text.strip())

    # Without usable segment details, the top-level text is used as returned
    final_text = " ".join(kept_text) if pieces else text.strip()
    language = payload.get("language")
    return {
        "status": "transcribed" if final_text else "silent",
        "text": final_text,
        "duration_seconds": _finite_number(payload.get("duration")),
        "language": language if isinstance(language, str) else None,
        "segments": pieces,
        "filtered_segment_count": filtered_count,
        "model": GROQ_WHISPER_MODEL,
    }


def _segment_rate_limit_error(response: httpx.Response) -> SegmentTranscriptionError:
    """HTTP 429: a used-up daily allowance only when Groq's message says so; otherwise a temporary limit."""
    message = _extract_error_message(response).lower()
    retry_after = parse_retry_after(response.headers.get("retry-after"))
    if any(marker in message for marker in DAILY_QUOTA_MARKERS):
        text = "The daily speech-to-text allowance has been reached. Transcription is unavailable until it resets."
        if retry_after is not None:
            text += f" Groq suggests trying again in {_describe_wait(retry_after)}."
        return SegmentTranscriptionError(429, "QUOTA_EXHAUSTED", text, retry_after, retryable=False)
    text = "Transcription is temporarily paused by the speech-to-text rate limit."
    text += f" Retry in {_describe_wait(retry_after)}." if retry_after is not None else " Wait a moment before retrying."
    return SegmentTranscriptionError(429, "RATE_LIMITED", text, retry_after, retryable=True)


async def transcribe_segment_groq(
    filename: str,
    audio_bytes: bytes,
    prompt: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None
) -> dict:
    """
    Transcribes one complete audio segment with Groq Whisper Large V3 Turbo (verbose_json).
    Returns the structured result; raises SegmentTranscriptionError on any failure.
    Makes at most two Groq requests (one retry for transient 5xx or dropped connections),
    never retries 429s or timeouts, and never logs or stores the audio or transcript.
    """
    clean_prompt = validate_segment_prompt(prompt)
    sanitized_filename, mime_type = validate_segment_audio(filename, audio_bytes)

    try:
        api_key = get_groq_api_key()
    except HTTPException:
        raise SegmentTranscriptionError(
            503, "TRANSCRIPTION_NOT_CONFIGURED",
            "Speech-to-text is not set up on the server (the Groq API key is missing)."
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    form_fields = {"model": GROQ_WHISPER_MODEL, "response_format": "verbose_json", "temperature": "0.0"}
    if clean_prompt:
        form_fields["prompt"] = clean_prompt

    should_close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=GROQ_TIMEOUT)
    try:
        try:
            response = await _post_with_single_retry(
                client, headers, sanitized_filename, audio_bytes, mime_type, form_fields
            )
        except httpx.TimeoutException:
            raise SegmentTranscriptionError(
                504, "TRANSCRIPTION_TIMEOUT",
                "Transcription took too long and was stopped. This audio segment was not transcribed.",
                retryable=True
            )
        except httpx.RequestError:
            raise SegmentTranscriptionError(
                503, "TRANSCRIPTION_UNAVAILABLE",
                "Transcription is temporarily unavailable. This audio segment could not be processed.",
                retryable=True
            )

        if response.status_code == 200:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            return _build_segment_result(payload)
        if response.status_code == 429:
            raise _segment_rate_limit_error(response)
        if response.status_code == 401:
            raise SegmentTranscriptionError(
                503, "TRANSCRIPTION_NOT_CONFIGURED",
                "Speech-to-text is not set up correctly on the server (the Groq API key was rejected)."
            )
        if response.status_code == 413:
            raise SegmentTranscriptionError(413, "FILE_TOO_LARGE", "The audio segment is above the 25 MB speech-to-text limit.")
        if response.status_code == 400:
            raise SegmentTranscriptionError(
                422, "INVALID_AUDIO",
                "The speech-to-text service could not read this audio segment. It may be damaged or incomplete."
            )
        if response.status_code in TRANSIENT_STATUS_CODES:
            raise SegmentTranscriptionError(
                503, "TRANSCRIPTION_UNAVAILABLE",
                "Transcription is temporarily unavailable. This audio segment could not be processed.",
                retryable=True
            )
        raise SegmentTranscriptionError(
            502, "TRANSCRIPTION_UNAVAILABLE",
            f"The speech-to-text service returned an unexpected error (HTTP {response.status_code}). This audio segment was not transcribed."
        )
    finally:
        if should_close_client:
            await client.aclose()
