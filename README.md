# OmniScribe AI

OmniScribe AI is a mobile-first, context-aware meeting intelligence assistant. It helps teams capture live meeting audio or uploaded call recordings, transcribes them into readable logs, retains company knowledge in an internal Vault, and uses Anthropic Claude to extract summaries, action items, and answers to questions (with automatic public web fallback when internal answers are insufficient).

---

## Repository Structure

```text
omniscribe-ai/
├── backend/
│   ├── main.py              # FastAPI application (endpoints, Claude routing, Vault)
│   ├── transcription.py     # Groq Whisper speech-to-text service (validation, retries, errors)
│   ├── test_backend.py      # Automated mocked tests (no API quota used)
│   ├── requirements.txt     # Python dependencies
│   └── vault/               # Internal company knowledge store (.gitkeep tracked, files ignored)
├── frontend/
│   └── index.html           # Official Claude mobile-first UI prototype
├── .env.example             # Template for required environment variables
├── .gitignore               # Ignores virtual environments, credentials, audio files, and vault data
├── OmniScribe AI.html       # Original UI export backup
└── README.md                # Project documentation and local quickstart
```

---

## Quickstart Guide

### Prerequisites
- Python 3.10 or higher installed.
- A modern web browser (Chrome, Edge, Safari, Firefox).
- A free Groq API key for speech-to-text (no credit card required, get one from [console.groq.com/keys](https://console.groq.com/keys)).
- An Anthropic Claude API key is **optional**. Leave `MOCK_AI_RESPONSE=true` to run without it (see [AI analysis](#ai-analysis-currently-mocked)).

---

### 1. Backend Setup

1. **Open your terminal** and navigate to the project root:
   ```bash
   cd omniscribe-ai
   ```

2. **Create and activate a virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - On macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Set up environment variables**:
   Copy `.env.example` to `.env`:
   - On Windows (PowerShell):
     ```powershell
     Copy-Item .env.example .env
     ```
   - On macOS / Linux:
     ```bash
     cp .env.example .env
     ```
   Open `.env` in a text editor and add your free Groq key. Keep `MOCK_AI_RESPONSE=true`:
   ```env
   GROQ_API_KEY=gsk_...
   MOCK_AI_RESPONSE=true
   ```
   `.env` is git-ignored. Never paste keys into the frontend, the code, or commits.

5. **Start the backend development server**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
   The interactive API documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs).

---

### 2. Frontend Setup

The frontend is a lightweight, zero-build client styled with Tailwind CSS.

- **Option A (Direct)**: Open `frontend/index.html` directly in your browser.
- **Option B (Local Web Server)**: In another terminal window, serve the frontend folder:
  ```bash
  python -m http.server 3000 --directory frontend
  ```
  Then navigate to [http://localhost:3000](http://localhost:3000) on your desktop or mobile device.

---

## Speech-to-Text (Groq Whisper Large V3 Turbo)

Recordings are transcribed by `whisper-large-v3-turbo` through Groq's free API tier. The implementation lives in `backend/transcription.py`, and `POST /api/process-meeting` calls it.

**Supported formats:** WebM, M4A, MP3, WAV, MP4, OGG (also FLAC, MPEG, MPGA). The file header is checked, not only the extension. Chrome/Edge (WebM), Firefox (WebM or OGG), and iOS Safari (MP4/M4A) recordings are all accepted.

**Limits:**
- 25 MB per file. Larger files are rejected with a clear message and are never sent to Groq or silently truncated.
- Free-tier quotas (requests and audio seconds per minute, hour, and day) are enforced by Groq. The exact free-plan numbers for this model are not published in Groq's docs, so check them in your Groq console.

**Failure handling:**
- HTTP 429 (rate limit): reported with the `Retry-After` wait time. Not retried.
- HTTP 429 (daily quota used up): processing stops with a message. No paid service is used.
- Transient 5xx responses or dropped connections: retried **once**. Timeouts are not retried.
- Invalid key, unreadable audio, malformed API response, or silence: each returns a specific error.
- Failed transcriptions are never saved to the Vault.

**Long recordings (not yet supported in this milestone):** 25 MB holds roughly 1.5 to 2 hours of mono speech at 32 kbps MP3. Longer meetings need compression or chunking before upload. That is planned for a later milestone and is not tested yet.

**Tests (all mocked, no quota used):**
```bash
python backend/test_backend.py
```

---

## AI Analysis (Currently Mocked)

Summaries, action items, and Q&A come from a **placeholder** while `MOCK_AI_RESPONSE=true`. The transcript is real Groq output. Everything else is labelled `[MOCK]`. The Anthropic integration is kept in `backend/main.py` for later replacement with a free hosted model. It is not used unless `MOCK_AI_RESPONSE` is set to `false`, and it then requires paid Anthropic credits.

---

## Project Milestones

- [x] **Milestone 1**: Repository Cleanup & Standard Structure (Preserved official UI, organized `backend/` and `frontend/`, created dependency manifests and security ignores).
- [ ] **Milestone 2**: Backend Repair & API Standardization (Fix web fallback query, add dotenv loader, accept web search toggle, structure Claude JSON responses).
- [ ] **Milestone 3**: Frontend & Backend Integration (Connect real Web Audio `MediaRecorder` and file uploads to backend API).
- [ ] **Milestone 4**: Speech-to-Text Transcription Integration. Phase B implemented: Groq integration, format and size validation, error handling, and mocked tests pass. Live Groq test is pending approval.
- [ ] **Milestone 5**: Enhanced Vault Search across past meetings.
- [ ] **Milestone 6**: End-to-End Verification and Deployment.
