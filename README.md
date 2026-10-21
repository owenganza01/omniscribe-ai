# OmniScribe AI

OmniScribe AI is a mobile-first, context-aware meeting intelligence assistant. It helps teams capture live meeting audio or uploaded call recordings and transcribes them into readable logs using real speech-to-text (Groq Whisper). It also keeps company knowledge in an internal Vault.

The intended design uses Anthropic Claude to extract summaries, action items, and answers, with public web fallback when internal answers are insufficient. **That part is currently mocked.** See [AI Analysis](#ai-analysis-currently-mocked).

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

2. **Create and activate a virtual environment**. Run each command on its own line:
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
     If PowerShell reports that scripts are disabled, run this once and then activate again:
     ```powershell
     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
     ```
   - macOS / Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   python -m pip install -r backend/requirements.txt
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

**Live status:** A real spoken recording was transcribed through the app using Groq. The returned transcript matched the spoken sentence and was displayed in the results view. The transcript saved to the Vault is that real Groq output.

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

| Output | Status |
|---|---|
| Transcript (`transcript_segment`) | **Real.** Speech-to-text from Groq Whisper Large V3 Turbo. |
| Summary | **Mocked.** Fixed placeholder text, labelled `[MOCK AI ANALYSIS]`. |
| Action items | **Mocked.** Fixed placeholder items, labelled `[MOCK]`. |
| Q&A answers | **Mocked.** Fixed placeholder answers, labelled `[MOCK]`. |

While `MOCK_AI_RESPONSE=true`, no Anthropic request is made. The placeholder analysis does not read the recording. Its text is not derived from the transcript.

No free hosted AI model has been selected or integrated yet. The Anthropic integration in `backend/main.py` remains in place until a free hosted model is chosen and connected. It runs only when `MOCK_AI_RESPONSE` is set to `false`, and it then requires paid Anthropic credits. Do not turn it on without a paid plan.

---

## Project Milestones

- [x] **Milestone 1**: Repository Cleanup & Standard Structure (Preserved official UI, organized `backend/` and `frontend/`, created dependency manifests and security ignores).
- [x] **Milestone 2**: Backend Repair & API Standardization. Complete, verified by 58 automated tests with mocked Claude, search, and transport clients (`python backend/test_backend.py`).
  - **Public web fallback:** DuckDuckGo instant answer, related topics, and HTML results. When nothing usable is found, the search returns no result rather than a message, and no refinement call is made.
  - **`.env` loading:** `backend/main.py` loads `.env` from the project root at startup, so `GROQ_API_KEY` and `MOCK_AI_RESPONSE` are read from it.
  - **Web-search toggle:** when disabled, no search and no refinement run, even if the model requests one.
  - **Structured Claude JSON:** responses are parsed into summary, action items, and Q&A. Malformed or non-object JSON falls back safely. A refinement replaces the initial analysis only when it is a valid object with the required fields. Otherwise the initial analysis is kept.
  - **Source labels:** an unsupported `web` or `vault` label is downgraded to `none`. `web` is kept only when a search result was used in a valid refinement, and `vault` only when vault context was enabled.
  - **Limitations:** real Claude responses have not been tested, because real calls are paid, so all Claude behavior was checked with stubbed clients. The related-topic and HTML search paths, and `.env` loading, are verified by manual checks rather than in the automated suite. Source labels are checked against the request settings, not against the evidence the model actually used, so a `vault` label can still be wrong if the model did not use the vault content.
- [ ] **Milestone 3**: Frontend & Backend Integration (Connect real Web Audio `MediaRecorder` and file uploads to backend API).
- [x] **Milestone 4**: Speech-to-Text Transcription Integration. Groq Whisper integration, format and size validation, error handling, frontend results display, automated mocked tests (58 backend, 8 browser), and a live transcription of a real recording are complete. Not included: real AI analysis (still mocked) and recordings longer than the 25 MB limit.
- [ ] **Milestone 5**: Enhanced Vault Search across past meetings. Not started.
- [ ] **Milestone 6**: End-to-End Verification and Deployment. Not started. The app runs only on a local computer. Nothing has been deployed. Remaining work: a full end-to-end check across browsers and devices, choosing a free hosting provider, production configuration (API URL, CORS, secrets), deployment, and selecting and integrating a free hosted model to replace the mocked AI analysis.
