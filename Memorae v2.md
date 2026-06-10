# Memorae v2 — Architecture & Implementation Guide

> **Version:** 2.0 (draft) · **Status:** Design spec · **Last updated:** 2026-06-10

---

## 1. Current Codebase Analysis & Rating

**Rating: 65 / 100**

### Why this rating?
The current codebase has a solid foundation. It correctly uses asynchronous Python (FastAPI, asyncpg), implements vector search natively with PostgreSQL (`pgvector`), and has a clean directory structure separating database models, queries, and services. It handles physical media gracefully via local buckets and ties into Telegram webhooks effectively.

**However, the current approach is not the most efficient or reliable way to build an agentic assistant.**

**Flaws in the Current (v1) Approach:**
1. **The "TOON" Format & Patch System:** The custom "TOON" encoding alongside `<kb_patch>` JSON block extraction is an anti-pattern. It forces the LLM to read a bespoke pseudo-YAML format and hand-construct JSON inside markdown blocks. This drives up latency and token consumption and produces frequent formatting errors (hallucinated fields, trailing commas, broken JSON).
2. **Double AI Processing (Intent → Notes Lane):** The pipeline does regex intent parsing, falls back to an LLM for JSON intent parsing, then makes *another* LLM call in `notes_lane.py` to generate the patch and reply. This is redundant and slow.
3. **Date/Time Brittleness:** Using an LLM to correct spelling before feeding `dateparser` is fragile and hard to reason about.
4. **No Native Function Calling:** Modern LLMs are fine-tuned for **tool calling (function calling)**. The current architecture ignores this, opting for manual string-matching and JSON-parsing hacks.

---

## 2. The Target Method (Memorae v2)

The recommended architecture is **native tool calling (function calling) combined with a robust RAG (Retrieval-Augmented Generation) pipeline.**

Instead of custom intent parsing and TOON logic, the system operates as a **ReAct-style agent** (Reasoning + Acting). The LLM receives a system prompt describing its persona plus a strict JSON schema of available tools. It decides when to call a tool, waits for backend execution, and then replies.

> **Honest framing:** Tool calling *reduces* formatting errors and latency; it does not eliminate hallucination. Models still occasionally emit invalid arguments, malformed timestamps, or non-existent IDs. The architecture below treats this as a first-class concern (see §2.2 and §7), not an afterthought.

### 2.1 The New Routing Pipeline
1. **Receive:** Webhook receives the message (text, audio, image).
2. **Pre-process:** Audio is transcribed; images are sent to a vision model for OCR/description.
3. **Agent Loop:**
   - Construct a prompt with the user's message, recent conversation history, the user's local time/timezone, and the list of callable tools.
   - The LLM responds. If it calls a tool (e.g., `search_notes`), the backend intercepts, validates arguments, executes the query, and feeds the result back.
   - The LLM processes the tool result and generates the final human-readable reply.

### 2.2 Agent Loop Guards (required)
A ReAct loop without limits is a runaway-cost bug. The loop **must** enforce:
- **Max iterations per turn** (e.g., 5 tool calls). On exceed, stop and return a graceful fallback message.
- **Per-turn token/cost ceiling**, logged and abortable.
- **Argument validation before execution.** Every tool call is validated against its schema (Pydantic). On failure, return a structured error to the LLM so it can self-correct, capped at N retries before giving up.
- **Tool execution errors are returned as data**, not raised — the LLM sees `{"error": "no note found with id ..."}` and can recover instead of the turn crashing.

### 2.3 Framework Decision
**Recommendation: build a native implementation on the provider SDK's `tools` array (e.g., the `openai` or `anthropic` SDK).** Do **not** adopt LangChain/LlamaIndex.

Rationale: the entire thesis of v2 is removing opaque indirection (TOON, multi-stage parsing). LangChain reintroduces hidden control flow and version churn for a system with only ~8 tools. A native loop is ~150 lines, fully inspectable, and easier to add the guards in §2.2. Reserve LlamaIndex only if the RAG layer later needs advanced index types.

---

## 3. Major Feature: Notes System (RAG Pipeline)

The notes system is the heart of Memorae. The v2 pipeline simplifies and strengthens it with modern RAG techniques.

### Note Data Structure
```sql
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    title TEXT NOT NULL,          -- 10-15 dense, keyword-rich words for retrieval
    content TEXT NOT NULL,        -- The actual note of any length
    metadata JSONB,               -- Structured data (category, entities, dates)
    embedding vector(1536),       -- Dimension MUST match the chosen model (see §5)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX notes_user_idx ON notes (user_id);
CREATE INDEX notes_embedding_idx ON notes USING hnsw (embedding vector_cosine_ops);
```

> **Multi-tenancy is non-negotiable:** every notes query MUST be scoped by `user_id`. Tool implementations derive `user_id` from the authenticated session — **never** from an LLM-supplied argument — so the model cannot read or mutate another user's data. This is the single most important security property of an agentic system.

### Embedding Strategy (clarified)
Use **one combined embedding** per note, computed over `title + "\n" + content`. The keyword-dense `title` exists specifically to inject high-signal retrieval terms into that combined vector; it is not a separate embedding. This keeps storage to one vector column and one similarity comparison at query time.

### The CRUD Flow via Tool Calling

#### A. Creation (Write)
- The user sends information.
- The AI invokes `create_note(content: str)`.
- The backend runs a lightweight internal process:
  1. **Title Generation:** A fast model generates a 10-15 word dense, keyword-rich title optimized for retrieval.
  2. **Metadata Extraction:** Extracts dates, entities, and categories into JSONB.
  3. **Embedding:** Embeds `title + content` together and stores the vector.
- **Idempotency:** `create_note` accepts an optional client-side `request_id`; duplicate IDs within a short window are deduped to prevent retries from creating double notes.
- The AI replies: "Saved as: [Title]".

#### B. Retrieval (Read)
- The user asks a question. The AI invokes `search_notes(query: str, filters: dict)`.
- **Query Rewriting:** A fast, cheap model rewrites the casual query ("What was that recipe?") into a dense retrieval query ("pasta recipe ingredients instructions food").
- **Vector Search + Metadata Filtering:** Search `pgvector`, filtered by `user_id` and any applicable `metadata` (e.g., date ranges for "notes from last week").
- **Reranking:** The top 10-15 vector hits pass through a cross-encoder (reranker) to select the top 3 most relevant.
- The top 3 notes are returned to the AI, which answers naturally. If no note clears a relevance threshold, the tool returns an empty result and the AI says it found nothing rather than inventing an answer.

#### C. Updating (Edit)
- The user asks to modify a note.
- The AI uses `search_notes` to find the note, then invokes `edit_note(note_id, content)` with the full rewritten content.
- Backend overwrites content, updates `updated_at`, and **regenerates the embedding only if `content` or `title` changed** (metadata-only edits skip re-embedding to save cost/latency).
- The AI replies: "Updated the note!"

#### D. Deletion (Forget) — with Confirmation State
- The AI uses `search_notes` to find the relevant note.
- **Confirmation:** The AI does not delete immediately. It replies: *"I found this note: [Title]. Delete it?"* and records a **pending action** (`{action: "delete", note_id, expires_at}`) in the conversation state store.
- On the next turn, if the user confirms, the AI invokes `delete_note(note_id)`. The pending action survives between messages because it is persisted server-side (not just held in the LLM's context window) and expires after a short TTL to avoid stale confirmations.

---

## 4. Other Features: Reminders & Calendar

These follow the same tool-calling paradigm — no regex, no manual datetime parsing.

### Reminders
- The `create_reminder` tool requires an **ISO-8601 timestamp with timezone offset**.
- The system prompt injects the user's current local time and IANA timezone (e.g., `Asia/Kolkata`), so the model can resolve "tomorrow at 5 PM" into an explicit timestamp. The backend **re-validates** the timestamp (rejects past times, malformed strings) and converts to UTC for storage — never trusting the model's arithmetic blindly.
- **Timezone & DST:** Store the trigger as TZ-aware UTC *plus* the original IANA timezone, so recurring reminders fire at the correct wall-clock time across DST transitions.
- **Tools:**
  - `create_reminder(title: str, trigger_datetime: str, recurrence: str | None)`
  - `list_reminders()`
  - `delete_reminder(reminder_id: str)`
- **`recurrence` format:** an **RRULE** string (RFC 5545), e.g. `FREQ=DAILY;INTERVAL=1`, or `null` for one-shot. RRULE is chosen over freeform text so the value is unambiguous and directly consumable by a scheduler library.
- **Background Cron:** The existing APScheduler job stays largely the same — it polls the DB each minute for due reminders and pings the Telegram API. Mark reminders as dispatched atomically to avoid double-sends.

### Google Calendar / Google Meet
- **Tools:**
  - `schedule_meeting(title: str, start_time: str, duration_minutes: int, attendees: list[str])`
  - `get_calendar_events(date: str)`
- `schedule_meeting` uses the user's stored Google OAuth tokens to create the event. Token refresh failures return a structured error so the AI can prompt the user to reconnect their account.
- On success, the backend returns the Google Meet link to the tool call and the AI presents it to the user.

---

## 5. Backend & Frontend Infrastructure

### Backend (Python / FastAPI)
- **Framework:** FastAPI (async).
- **Database:** PostgreSQL + `pgvector` (SQLAlchemy + asyncpg).
- **AI orchestration:** Native provider SDK `tools` array (see §2.3 — no LangChain).
- **Embeddings:** Pick **one** and match the schema dimension accordingly:
  - OpenAI `text-embedding-3-small` → `vector(1536)`
  - Local `nomic-embed-text` → `vector(768)`
  - Changing models later requires a migration + full re-embed; pick deliberately.
- **Reranker:** Cohere Rerank (hosted) or a local BGE-reranker model.

### Observability (required)
- Log every tool call: name, validated arguments, latency, and token cost per turn.
- Track per-turn iteration count and cost against the §2.2 ceilings.
- Surface a structured trace per conversation turn for debugging agent behavior.

### Migration from v1
- Existing v1 TOON/`<kb_patch>` notes must be migrated into the v2 `notes` schema: parse stored content, generate titles + metadata, and embed each note with the chosen model. Run as a one-off backfill job, idempotent and resumable.

### Cost & Latency Targets (fill in with measured numbers)
State explicit budgets so "cheaper/faster" claims are verifiable, e.g. target median turn latency and target token cost per turn versus the v1 baseline. Measure before and after.

### Frontend (Optional Web Dashboard)
While Memorae lives in Telegram, a web dashboard helps manage notes and settings.
- **Framework:** Next.js (App Router).
- **Styling:** TailwindCSS + Shadcn/UI, dark-mode focused.
- **Features:**
  - A clean Kanban or masonry layout of all notes.
  - Full-text and vector search bar.
  - Active reminders list.
  - Google account connection settings page.
  - Direct chat widget mirroring the Telegram experience.

---

## 6. Summary of Architectural Shifts from v1 to v2

| Feature | Memorae v1 (Current) | Memorae v2 (Target) |
| :--- | :--- | :--- |
| **Routing** | Regex + LLM intent parse → dedicated handlers | Native LLM tool calling (ReAct agent loop) |
| **Note Updates** | Custom "TOON" encoded patches (`<kb_patch>`) | Native `edit_note(id, content)` JSON tool call |
| **Search** | Direct vector similarity on whole note | Query rewriting → vector search → cross-encoder reranking |
| **Time Parsing** | LLM spellcheck → `dateparser` | LLM ISO-8601 generation + backend re-validation |
| **State** | Implicit memory from DB message history | Buffer-window memory + server-side pending actions |
| **Safety** | Implicit | Loop guards, arg validation, per-user scoping |

---

## 7. Failure Modes & Mitigations

| Failure | Mitigation |
| :--- | :--- |
| LLM emits invalid tool arguments | Pydantic validation → structured error back to model → retry (capped) |
| LLM references a non-existent `note_id` | Tool returns `{"error": ...}` as data; AI recovers |
| Infinite / runaway tool-call loop | Max-iteration + cost ceiling per turn (§2.2) |
| Duplicate `create_note` on retry | `request_id` idempotency key |
| Reminder fires at wrong wall-clock time | Store UTC + IANA tz; RRULE for recurrence |
| Double-sent reminders | Atomic "dispatched" flag in cron |
| Cross-user data access | `user_id` from session, never from LLM args |
| OAuth token expiry | Structured error → AI prompts reconnect |
| No relevant note found | Empty result + threshold; AI says so instead of hallucinating |

---

By moving to a function-calling architecture with explicit guards, Memorae v2 should be more robust, cheaper on tokens, and faster — while remaining resilient to the formatting and recovery issues that the v1 TOON implementation cannot handle gracefully. Claims of "cheaper" and "faster" should be backed by the measured budgets in §5.
