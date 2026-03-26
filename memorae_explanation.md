# Memorae - Your AI Telegram Assistant

## 🌟 What is Memorae? (Non-Technical Explanation)

Memorae (often called **"Memo"**) is an intelligent, personal assistant that lives directly inside your Telegram app. It acts like a digital second brain, helping you stay organized, remember things effortlessly, and manage your day-to-day life without needing to open a dozen different apps.

### How is it helpful?
* 🧠 **Smart Note-Taking (Your Second Brain):** You can forward messages, send voice notes, images, or PDFs to Memo and tell it to "remember this." It automatically reads the files, extracts the text, and categorizes it. Later, you can find the information simply by asking naturally, like: *"What was that pasta recipe I saved?"*
* ⏰ **Reminders & Alarms:** Tell Memo, *"Remind me to call John tomorrow at 5 PM"* or *"Remind me every morning to drink water."* It figures out the time and sends you a message exactly when needed.
* 📅 **Calendar Integration:** It connects directly to your Google Calendar. You can tell it to *"Add a meeting with Sarah next Monday at 3 PM"* and it will schedule it for you.
* 💬 **Conversational & Friendly:** Memo isn't just a rigid bot—it has a personality! You can chat with it naturally, and it understands when you're just making conversation versus when you're giving it a specific task to perform.
* 📎 **Media Savvy:** Send it photos or PDFs! It can *"read"* images, save them natively, and recall them. If you saved a photo of your ID, you can later say *"Send me my ID"* and Memo will send the exact photo back to you.

---

## ⚙️ Under the Hood: Technical Architecture (For Tech People)

Memorae is built as an asynchronous **FastAPI** backend that communicates with **Telegram via Webhooks**. It uses **PostgreSQL** (via SQLAlchemy/Asyncpg) as its primary datastore, **APScheduler** for background jobs, and routes its AI capabilities through **OpenRouter** (Claude, OpenAI, Mistral, etc.).

### Core Event Flow
1. **Ingestion:** Telegram pushes updates (messages, images, voice notes, PDFs) to the `/webhook` endpoint.
2. **Context Gathering:** The system creates/fetches the user from PostgreSQL and retrieves the recent conversation history. Media files are downloaded locally to a `media_bucket`.
3. **Intent Parsing:** The system evaluates whether the user is giving a command (save note, set reminder, event) or just chatting.
4. **Execution:** The respective service handles the database storage, external API calls, or background task scheduling.
5. **Response:** The AI streams back a contextually aware text or native media dispatch to the user via the Telegram Client.

### Service Breakdown

Here is what happens behind the back for each specific module/service:

#### 1. Telegram Webhook Service (`routers/webhook.py` & `services/telegram.py`)
* The `/webhook` endpoint listens for incoming generic updates.
* It captures text, natively downloads media files to an `os` directory (the `media_bucket`), and records the raw conversational flow to the database.
* To prevent blocking the FastAPI event loop, webhook updates are immediately deferred to `BackgroundTasks`. 
* Uses the `python-telegram-bot` standard library underneath a custom wrapper (`TelegramClient`) to send async responses and serve static files natively.

#### 2. Toon Service (Persona & Command Routing) (`services/toon.py`)
* **Intent Parsing:** Uses quick RegEx heuristics for obvious commands (`remember`, `remind`, `recall`). If ambiguous, it queries an LLM specialized in intent extraction to output one of the definitive commands (e.g., `CHAT`, `REMEMBER`, `RECALL`, `REMIND`, `ADD_CALENDAR`) in strict JSON.
* **Recurrence Logic:** Wraps Regex to trap patterns like "every day" or "weekly" to determine recurrences so the absolute datetime parser understands strictly when the *first* trigger happens.
* **Persona Engine:** Wraps the base AI with defined System Prompts (like `friendly_assistant` or `professional`) bridging standard completion with custom instructions (e.g., instructing the LLM to yield `(LOCAL_PATH: ...)` tags so the Webhook router knows to ship genuine media files instead of just describing them).

#### 3. AI Service (`services/ai.py`)
* **LLM Engine:** Instantiates an OpenAI-compatible async client that interfaces with OpenRouter to utilize models like Claude-3 Opus or GPT-4o-Mini depending on the task (`ocr_model` vs `ai_model`).
* **Structured Data:** Uses high-temperature JSON-mode completions to map unstructured conversational instructions heavily into usable parameters (`datetime_str`, `content_to_save`, `recurrence`).
* **Datetime Parser:** Extremely vital for reminders and calendar entries. Takes a user's natural language time request along with the `user_timezone` and the system's `UTC now` reference to project a precise `ISO-8601` timestamp for the database/cron.

#### 4. Media & OCR Service (`services/media.py`)
* **Vision Models:** Images are formatted into base64 and sent to Claude/GPT Vision endpoints to obtain a high-fidelity semantic description + text snippet extraction.
* **Audio Transcriptions:** Voice notes and audio files (`.ogg`) are sent directly to `whisper-1` endpoints.
* **PDF Plumber:** For documents, `pdfplumber` acts rapidly to pull standard text. It natively isolates the document's first page, converts it to JPEG format, and routes it back into the Vision module to assure accurate diagram/stamp reading! 

#### 5. Knowledge Base (KB) Service (`services/kb.py`)
* Powered by `pgvector` for true semantic/vector similarity search.
* When a user demands a "remember" action, this service extracts a structural `title`, `content`, and descriptive `tags`. It then generates a 768-dimensional semantic embedding using a local Ollama model (`nomic-embed-text`) and stores it natively alongside the data in the PostgreSQL `kb_entries` table.
* During a "recall" request (e.g. "What's my passport number?"), the service embeds the query via Ollama and uses pgvector's cosine distance operator (`<=>`) to instantly fetch the most semantically relevant memories. The matched results are pushed as *System Prompt Context* alongside the user's question, allowing the LLM to deliver precise answers fluidly.

#### 6. Reminders Scheduler Job (`jobs/reminders.py`)
* Powered by `APScheduler` wrapped in an AsyncIOScheduler tied to FastAPI's lifespan.
* Runs on a frequent IntervalTrigger (configurable, usually every 1 minute).
* Scrutinizes the database natively for `remind_at <= NOW()`. 
* Dispatches Telegram pushes for those reminders.
* **Recurrence Handling:** If an executed reminder is tracked as `daily`, `weekly`, etc., the service logically injects a `timedelta` loop calculating the next valid future trigger and updates the database row inplace rather than deleting or marking it resolved.

#### 7. Google Calendar Service (`services/google_cal.py`)
* Bootstrapped typical OAuth2 flows (`google-auth-oauthlib`).
* Injects Google Auth URLs straight to the Telegram chat. Once clicked and authorized, short-lived `access_tokens` and vital `refresh_tokens` are embedded into the User's database row.
* Wraps the official `googleapiclient` wrapper to interface via raw payload drops directly to Google's API to construct valid `insert()` payload parameters (incorporating correct timezone shifts) or returning `list()` events for their day.
