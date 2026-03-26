-- ─────────────────────────────────────────────────────────────────────────────
-- Memorae – Database Schema
-- ─────────────────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS "vector";    -- pgvector for semantic search

-- ── Users ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    telegram_id     TEXT UNIQUE NOT NULL,          -- Telegram User ID
    display_name    TEXT,
    timezone        TEXT NOT NULL DEFAULT 'Asia/Kolkata',
    google_tokens   JSONB,                         -- {access_token, refresh_token, expiry}
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Conversations ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Messages ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    media_url       TEXT,
    media_type      TEXT,                          -- image | audio | document
    telegram_message_id TEXT UNIQUE,               -- dedup on Telegram's own ID
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at DESC);

-- ── Knowledge Base ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS kb_entries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    media_url       TEXT,
    media_type      TEXT,
    tags            TEXT[] DEFAULT '{}',
    embedding       vector(768),                   -- pgvector semantic embedding (nomic-embed-text)
    source          TEXT DEFAULT 'manual',         -- manual | telegram | calendar | media
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_kb_user  ON kb_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_kb_tags  ON kb_entries USING GIN(tags);
-- trgm search disabled on aapanel

-- ── Reminders ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reminders (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT NOT NULL,
    body            TEXT,
    remind_at       TIMESTAMPTZ NOT NULL,
    recurrence      TEXT,                          -- cron expression or NULL
    google_event_id TEXT,                          -- synced Calendar event
    sent            BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(remind_at) WHERE sent = FALSE;

-- ── Patches (AI-suggested edits / drafts) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS patches (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kb_entry_id     UUID REFERENCES kb_entries(id) ON DELETE SET NULL,
    diff            JSONB NOT NULL,                -- {original, proposed, fields_changed}
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','accepted','rejected')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── updated_at auto-update trigger ───────────────────────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$;

DO $$ DECLARE t TEXT; BEGIN
    FOR t IN SELECT unnest(ARRAY['users','kb_entries']) LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_updated_at ON %I;
             CREATE TRIGGER trg_updated_at BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION set_updated_at();', t, t);
    END LOOP;
END $$;