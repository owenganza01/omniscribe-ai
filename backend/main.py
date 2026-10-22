import os
import re
import json
import logging
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
from anthropic import Anthropic, APIError, AuthenticationError
from backend.transcription import transcribe_audio_groq, transcribe_segment_groq, SegmentTranscriptionError

# Logs error types only. Transcript text, prompts and audio are never logged.
logger = logging.getLogger("omniscribe")

# -----------------------------------------------------------------------------
# 1. Environment & Paths Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Always load .env from the project root if it exists
load_dotenv(PROJECT_ROOT / ".env")

# Model configuration
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest").strip()
MOCK_AI_RESPONSE = os.environ.get("MOCK_AI_RESPONSE", "").lower() in ("true", "1", "yes")

# Secure Vault directory anchored to backend/vault
VAULT_DIR = BASE_DIR / "vault"
VAULT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. FastAPI Application Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="OmniScribe AI API",
    version="0.2.0",
    description="Context-Aware Meeting Intelligence & Real-time Q&A Routing API"
)

# Standardized CORS configuration for local development and mobile devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 3. Data Models
# -----------------------------------------------------------------------------
class QAItem(BaseModel):
    question: str
    answer: str
    source: str = Field(
        ...,
        description="Origin of answer evidence: 'vault' (internal documents/transcript), 'web' (public internet), or 'none' (unanswered)"
    )

class ProcessMeetingResponse(BaseModel):
    success: bool
    meeting_name: str
    transcript_segment: str
    summary: str
    action_items: List[str]
    qa_items: List[QAItem]

class SegmentPiece(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None
    text: str
    no_speech_prob: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    filtered: bool = Field(..., description="True when Whisper's own signals both indicate silence; its text is left out of 'text'")
    low_confidence: bool = Field(..., description="Kept in 'text', but the model was unsure or the output looks repetitive")

class TranscribeSegmentResponse(BaseModel):
    success: bool
    status: str = Field(..., description="'transcribed' (speech found) or 'silent' (no speech; not an error)")
    sequence: Optional[int] = Field(None, description="Echo of the client's segment number, for ordering")
    text: str
    duration_seconds: Optional[float] = None
    language: Optional[str] = None
    segments: List[SegmentPiece]
    filtered_segment_count: int
    model: str

class HealthResponse(BaseModel):
    status: str
    anthropic_configured: bool
    groq_configured: bool
    mock_mode: bool
    model: str
    vault_path: str
    web_fallback_ready: bool

# -----------------------------------------------------------------------------
# 4. Helper Utilities
# -----------------------------------------------------------------------------
def get_anthropic_client() -> Anthropic:
    """
    Safely retrieves the Anthropic client if configured.
    Raises an informative HTTP 503 error instead of crashing if the key is missing.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ANTHROPIC_API_KEY is not configured. Please set a valid Anthropic API key in your .env file."
        )
    return Anthropic(api_key=api_key)

def sanitize_meeting_name(name: Optional[str]) -> str:
    """
    Sanitizes meeting names to prevent directory traversal and invalid file characters.
    """
    if not name:
        return "untitled-meeting"
    # Allow alphanumeric, hyphen, underscore, and space
    sanitized = re.sub(r"[^\w\s-]", "", name).strip()
    sanitized = re.sub(r"[\s]+", "-", sanitized)
    if not sanitized or sanitized.replace("-", "").replace("_", "") == "":
        return "untitled-meeting"
    return sanitized[:80]

def save_and_read_vault_context(sanitized_name: str, transcript_segment: str) -> str:
    """
    Appends a new transcript segment to the meeting's vault file and returns
    the accumulated transcript history for context.
    """
    target_file = (VAULT_DIR / f"{sanitized_name}.txt").resolve()

    # Path traversal safety check: ensure target stays inside VAULT_DIR
    if not target_file.is_relative_to(VAULT_DIR.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid meeting name resulting in an unsafe file path."
        )

    with open(target_file, "a", encoding="utf-8") as f:
        f.write(transcript_segment + "\n")

    with open(target_file, "r", encoding="utf-8") as f:
        return f.read()

async def query_public_internet_fallback(search_query: str) -> Optional[str]:
    """
    Asynchronous fallback to query public web information when internal context
    is insufficient and web fallback is enabled.
    Attempts DuckDuckGo Instant Answer API first, falling back to HTML snippets.
    Returns usable result text, or None when there is no usable result (empty query, no results,
    or a network or parsing failure). Never fabricates search results and never returns a failure message.
    """
    clean_query = search_query.strip()
    if not clean_query:
        return None

    encoded_query = urllib.parse.quote_plus(clean_query)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    async with httpx.AsyncClient(headers=headers, timeout=8.0) as client:
        # Step A: Query DuckDuckGo Instant Answer API
        try:
            api_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
            res = await client.get(api_url)
            if res.status_code == 200:
                data = res.json()
                abstract = data.get("AbstractText", "").strip()
                if abstract:
                    source = data.get("AbstractSource", "Web Knowledge")
                    return f"[Live Web Search ({source})]: {abstract}"

                # Check RelatedTopics for summary lines
                related = data.get("RelatedTopics", [])
                snippets = []
                for topic in related[:3]:
                    if isinstance(topic, dict) and "Text" in topic and topic["Text"]:
                        snippets.append(topic["Text"].strip())
                if snippets:
                    return f"[Live Web Search]: " + " | ".join(snippets)
        except Exception:
            pass

        # Step B: Fallback to DuckDuckGo HTML search results if Instant Answer is empty
        try:
            html_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            res = await client.get(html_url)
            if res.status_code == 200:
                matches = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', res.text, re.DOTALL)
                if matches:
                    clean_snippets = [re.sub(r'<[^>]+>', '', m).strip() for m in matches[:2]]
                    combined = " ".join([s for s in clean_snippets if s])
                    if combined:
                        return f"[Live Web Search]: {combined}"
        except Exception:
            pass

    return None

def parse_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Returns the JSON object in Claude's response (handling code fence wrappers), or None if there is none.
    Valid JSON that is not an object (list, string, number, boolean, null) is rejected.
    """
    cleaned = raw_text.strip()
    # Strip markdown code blocks if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Only a JSON object is a valid analysis. Lists, strings, numbers, booleans and null fall through.
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        # Fallback regex search for JSON object inside text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return None

def is_usable_analysis(parsed: Optional[Dict[str, Any]]) -> bool:
    """True when a parsed object has the fields the response needs: a string summary and list action_items and qa_items."""
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("summary"), str)
        and isinstance(parsed.get("action_items"), list)
        and isinstance(parsed.get("qa_items"), list)
    )

def parse_claude_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Safely parses structured JSON from Claude's response, falling back to a safe default structure.
    """
    parsed = parse_json_object(raw_text)
    if parsed is not None:
        return parsed

    # Safe fallback structure if model output failed to format properly
    return {
        "summary": raw_text[:300] if raw_text else "Meeting segment processed.",
        "action_items": [],
        "qa_items": []
    }

NEED_SEARCH_FLAG = "[NEED_EXTERNAL_SEARCH:"
MAX_SEARCH_QUERY_CHARS = 200

def extract_search_query(text: str) -> str:
    """
    Returns the query inside a [NEED_EXTERNAL_SEARCH: ...] flag, or "" if the flag is absent or malformed.
    A valid flag closes with ']' on the same line and contains no '[' (which would mean a JSON array
    or other bracketed text follows an unterminated flag). Queries over MAX_SEARCH_QUERY_CHARS are
    rejected whole, never truncated. A malformed flag returns "" so no partial query is sent to web search.
    """
    start = text.find(NEED_SEARCH_FLAG)
    if start == -1:
        return ""
    start += len(NEED_SEARCH_FLAG)
    line_end = text.find("\n", start)
    if line_end == -1:
        line_end = len(text)
    end = text.find("]", start, line_end)
    if end == -1:
        return ""
    query = text[start:end].strip()
    if "[" in query or len(query) > MAX_SEARCH_QUERY_CHARS:
        return ""
    return query

# -----------------------------------------------------------------------------
# 5. API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint returning system status and configuration state.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    configured = bool(api_key and api_key != "your_anthropic_api_key_here")
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    groq_configured = bool(groq_key and groq_key != "your_groq_api_key_here")
    return {
        "status": "healthy",
        "anthropic_configured": configured,
        "groq_configured": groq_configured,
        "mock_mode": MOCK_AI_RESPONSE,
        "model": ANTHROPIC_MODEL,
        "vault_path": str(VAULT_DIR),
        "web_fallback_ready": True
    }

@app.post("/api/process-meeting", response_model=ProcessMeetingResponse)
async def process_meeting(
    file: UploadFile = File(...),
    is_internal_context: bool = Form(True),
    is_web_fallback: bool = Form(True),
    meeting_name: Optional[str] = Form(None)
):
    """
    Processes incoming meeting audio or uploaded call recordings:
    1. Validates incoming payload.
    2. Transcribes audio to real speech text using Groq Whisper Large V3 Turbo.
    3. Manages Vault historical context (only upon successful transcription).
    4. Routes questions through AI engine against internal vault context.
    5. Optionally queries public web search if internal knowledge is insufficient.
    6. Returns structured summary, action items, and Q&A items with source attribution.
    """
    # 1. Validate incoming file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read incoming audio stream: {str(e)}"
        )

    display_name = (meeting_name or "").strip() or "Untitled meeting"
    sanitized_name = sanitize_meeting_name(display_name)

    # 2. Real Speech-to-Text Transcription via Groq Whisper Large V3 Turbo
    # Validates file size (max 25 MB), format header, and calls Groq Whisper API
    detected_transcript = await transcribe_audio_groq(file.filename, audio_bytes)

    # 3. Vault Context Management (Only saved if transcription succeeded)
    vault_context_stream = ""
    if is_internal_context:
        try:
            vault_context_stream = save_and_read_vault_context(sanitized_name, detected_transcript)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update internal vault storage: {str(e)}"
            )

    # 4. Check for Mock / Non-Paid Test Mode
    if MOCK_AI_RESPONSE:
        # Placeholder analysis for development testing. The transcript above is real Groq output;
        # everything below is fixed text and is labelled [MOCK] so it is never mistaken for AI output.
        # Mock answers state that no lookup happened, so they must not carry a vault or web source label
        q1_source = "none"
        q1_answer = (
            "[MOCK] Placeholder answer. No internal vault lookup was performed."
            if is_internal_context
            else ("[MOCK] Placeholder answer. No public web lookup was performed." if is_web_fallback
                  else "Internal vault context and web search fallback are both disabled.")
        )
        q2_source = "none"
        q2_answer = (
            "[MOCK] Placeholder answer. No public web lookup was performed."
            if is_web_fallback
            else "Not found in internal vault. Turn on Live Web Fallback to look it up online."
        )

        return ProcessMeetingResponse(
            success=True,
            meeting_name=display_name,
            transcript_segment=detected_transcript,
            summary=f"[MOCK AI ANALYSIS] Placeholder summary for '{display_name}'. Not generated from this recording.",
            action_items=[
                "[MOCK] Placeholder action item (not extracted from the transcript)"
            ],
            qa_items=[
                QAItem(
                    question="[MOCK] Placeholder question about internal policy",
                    answer=q1_answer,
                    source=q1_source
                ),
                QAItem(
                    question="[MOCK] Placeholder question about public information",
                    answer=q2_answer,
                    source=q2_source
                )
            ]
        )

    # When not in mock mode, connect to Claude via Anthropic SDK
    client = get_anthropic_client()

    # 5. Dual-Routing Claude Prompt
    routing_prompt = f"""
You are the intelligence engine of OmniScribe AI, a meeting assistant.
Analyze this meeting transcript segment:
"{detected_transcript}"

Meeting Name: "{display_name}"
Internal Context Enabled: {"Yes" if is_internal_context else "No"}
Live Web Fallback Enabled: {"Yes" if is_web_fallback else "No"}

Internal Vault Context:
--- START VAULT FILE ---
{vault_context_stream}
--- END VAULT FILE ---

Instructions:
1. Summarize the transcript segment succinctly (1-2 sentences).
2. Extract concrete action items if any were mentioned.
3. Identify questions asked during the meeting.
4. For each question:
   - If answered by the transcript itself or the internal vault context, answer it and set source to "vault".
   - If NOT answered by internal context and Live Web Fallback is ENABLED:
     If external facts from the internet are needed, include this flag in your response:
     [NEED_EXTERNAL_SEARCH: <specific search query>]
   - If NOT answered by internal context and Live Web Fallback is DISABLED:
     Answer that internal information is insufficient and internet lookups are disabled. Set source to "none".
5. Format your output strictly as a JSON object with this exact structure:
{{
  "summary": "Meeting summary...",
  "action_items": ["Action item 1", "Action item 2"],
  "qa_items": [
    {{
      "question": "Question text?",
      "answer": "Answer text...",
      "source": "vault" | "web" | "none"
    }}
  ]
}}
Do not include commentary outside the JSON object.
"""

    try:
        message = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": routing_prompt}]
        )
        ai_response_text = message.content[0].text
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Anthropic authentication failed. Please verify your ANTHROPIC_API_KEY in .env."
        )
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Anthropic API error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to communicate with Claude AI: {str(e)}"
        )

    # 6. Execute Live Web Fallback if requested by Claude and enabled by user
    # Set only when a search returned results and the refinement passed validation (the only web evidence)
    web_refined = False
    if is_web_fallback and NEED_SEARCH_FLAG in ai_response_text:
        try:
            # Malformed or unterminated flags yield "" and are skipped, keeping the initial analysis
            search_query = extract_search_query(ai_response_text)

            if search_query:
                web_data = await query_public_internet_fallback(search_query)

                # No usable public result: skip refinement so no failure is presented as web evidence
                if web_data:
                    refinement_prompt = f"""
You requested external web search data for query: '{search_query}'.
Here is the live public web result:
{web_data}

Incorporate this data to answer the questions that required outside knowledge.
Set source to "web" for questions answered using this web result.
Return strictly the updated JSON matching the required schema.
"""
                    refined_message = client.messages.create(
                        model=ANTHROPIC_MODEL,
                        max_tokens=1500,
                        messages=[
                            {"role": "user", "content": routing_prompt},
                            {"role": "assistant", "content": ai_response_text},
                            {"role": "user", "content": refinement_prompt}
                        ]
                    )
                    refined_text = refined_message.content[0].text
                    # A malformed or non-object refinement keeps the initial analysis instead of erasing it
                    if is_usable_analysis(parse_json_object(refined_text)):
                        ai_response_text = refined_text
                        web_refined = True
        except Exception:
            # If refinement slips, keep initial output
            pass

    # 7. Parse and Validate Output Structure
    parsed_analysis = parse_claude_json_response(ai_response_text)

    # Wrong field types fall back to safe defaults instead of raising a 500
    summary = parsed_analysis.get("summary")
    if not isinstance(summary, str):
        summary = "Meeting segment processed successfully."

    raw_action_items = parsed_analysis.get("action_items")
    if not isinstance(raw_action_items, list):
        raw_action_items = []
    action_items = [str(item) for item in raw_action_items if item]

    raw_qa_items = parsed_analysis.get("qa_items")
    if not isinstance(raw_qa_items, list):
        raw_qa_items = []
    qa_items: List[QAItem] = []
    for item in raw_qa_items:
        if isinstance(item, dict) and "question" in item and "answer" in item:
            source = item.get("source", "none")
            source = source.lower() if isinstance(source, str) else "none"
            if source not in ("vault", "web", "none"):
                source = "none"
            # Only report a source that was actually used. The model's label alone is not evidence.
            if source == "web" and not web_refined:
                source = "none"
            if source == "vault" and not is_internal_context:
                source = "none"
            qa_items.append(
                QAItem(
                    question=str(item["question"]),
                    answer=str(item["answer"]),
                    source=source
                )
            )

    return ProcessMeetingResponse(
        success=True,
        meeting_name=display_name,
        transcript_segment=detected_transcript,
        summary=summary,
        action_items=action_items,
        qa_items=qa_items
    )


# -----------------------------------------------------------------------------
# 6. Live Segment Transcription (Phase 3.1)
# -----------------------------------------------------------------------------
# Error responses from this endpoint only use this shape (other endpoints keep FastAPI's {"detail": ...}):
#   {"success": false, "sequence": <int|null>, "error": {"code": "RATE_LIMITED", "message": "...",
#    "retry_after_seconds": <int|null>, "retryable": <bool>}}
# A Retry-After header is added only when Groq supplied a wait time.
MAX_SEGMENT_SEQUENCE = 1_000_000

def segment_error_response(err: SegmentTranscriptionError, sequence: Optional[int] = None) -> JSONResponse:
    headers = {"Retry-After": str(err.retry_after_seconds)} if err.retry_after_seconds is not None else None
    return JSONResponse(
        status_code=err.status_code,
        content={"success": False, "sequence": sequence, "error": err.to_dict()},
        headers=headers,
    )

@app.post("/api/transcribe-segment", response_model=TranscribeSegmentResponse)
async def transcribe_segment(
    file: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    sequence: Optional[str] = Form(None)
):
    """
    Transcribes ONE complete, independently decodable audio segment for live transcription.
    Speech-to-text only: no Vault writes, no AI analysis, no web search, and nothing is stored.
    `prompt` is optional context (the end of the previous transcript) for continuity.
    `sequence` is an optional segment number echoed back so the client can keep segments in order.
    """
    seq: Optional[int] = None
    if sequence not in (None, ""):
        if not re.fullmatch(r"[0-9]{1,7}", sequence) or int(sequence) > MAX_SEGMENT_SEQUENCE:
            return segment_error_response(SegmentTranscriptionError(
                400, "INVALID_REQUEST", f"'sequence' must be a whole number from 0 to {MAX_SEGMENT_SEQUENCE}."))
        seq = int(sequence)

    if file is None or not file.filename:
        return segment_error_response(SegmentTranscriptionError(
            400, "INVALID_REQUEST", "Send one audio segment in the 'file' field."), seq)

    try:
        audio_bytes = await file.read()
        result = await transcribe_segment_groq(file.filename, audio_bytes, prompt=prompt)
    except SegmentTranscriptionError as err:
        logger.warning("Segment transcription failed: %s", err.code)
        return segment_error_response(err, seq)
    except Exception as e:
        # The exception text could contain internal details, so only its type is logged and none of it is returned
        logger.error("Unexpected segment transcription error: %s", type(e).__name__)
        return segment_error_response(SegmentTranscriptionError(
            500, "INTERNAL_ERROR",
            "Something went wrong on the server while processing this audio segment. It was not transcribed.",
            retryable=True), seq)

    return TranscribeSegmentResponse(success=True, sequence=seq, **result)
