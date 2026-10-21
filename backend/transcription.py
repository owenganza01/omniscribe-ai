import os
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
    mime_type: str
) -> httpx.Response:
    """Sends one multipart request to Groq. Keeps no state, so it can be safely repeated."""
    return await client.post(
        GROQ_WHISPER_URL,
        headers=headers,
        files={"file": (filename, audio_bytes, mime_type)},
        data={"model": GROQ_WHISPER_MODEL, "response_format": "json", "temperature": "0.0"}
    )

async def _post_with_single_retry(
    client: httpx.AsyncClient,
    headers: dict,
    filename: str,
    audio_bytes: bytes,
    mime_type: str
) -> httpx.Response:
    """
    Sends the request, retrying at most once for transient 5xx responses or dropped connections.
    Timeouts propagate immediately without a retry.
    """
    try:
        response = await _send_transcription_request(client, headers, filename, audio_bytes, mime_type)
    except httpx.TimeoutException:
        raise
    except httpx.RequestError:
        response = None

    if response is not None and response.status_code not in TRANSIENT_STATUS_CODES:
        return response

    await asyncio.sleep(RETRY_DELAY_SECONDS)
    return await _send_transcription_request(client, headers, filename, audio_bytes, mime_type)

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
