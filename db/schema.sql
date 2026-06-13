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
    context_clues   TEXT[] DEFAULT '{}',           -- key nouns/entities for semantic matching
    metadata        JSONB DEFAULT '{}'::jsonb,     -- {category, entities[], dates[]} (Memorae v2 §3)
    embedding       vector(1024),                  -- pgvector semantic embedding (nvidia/llama-nemotron-embed-vl-1b-v2)
    source          TEXT DEFAULT 'manual',         -- manual | telegram | calendar | media
    status          TEXT DEFAULT 'active',         -- active | done
    source_message  TEXT,                          -- user's exact original message
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Reconcile columns on databases created before the current schema (CREATE
-- TABLE IF NOT EXISTS above is a no-op once the table already exists, so it
-- never adds missing columns — these ALTERs do).
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS tags          TEXT[] DEFAULT '{}';
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS context_clues TEXT[] DEFAULT '{}';
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS metadata      JSONB DEFAULT '{}'::jsonb;
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS embedding     vector(1024);
-- If the embedding column exists at a different dimension (e.g. the old 768-dim
-- nomic vectors), rebuild it at 1024. Guarded so it only fires on a dimension
-- change — it does NOT wipe embeddings on every restart. Old vectors are dropped
-- because they came from a different model and must be regenerated.
DO $$
DECLARE dim int;
BEGIN
    SELECT atttypmod INTO dim
    FROM pg_attribute
    WHERE attrelid = 'kb_entries'::regclass AND attname = 'embedding' AND NOT attisdropped;
    IF dim IS DISTINCT FROM 1024 THEN
        ALTER TABLE kb_entries DROP COLUMN IF EXISTS embedding;
        ALTER TABLE kb_entries ADD COLUMN embedding vector(1024);
    END IF;
END $$;
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS source        TEXT DEFAULT 'manual';
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS status        TEXT DEFAULT 'active';
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS source_message TEXT;
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS media_url     TEXT;
ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS media_type    TEXT;
CREATE INDEX IF NOT EXISTS idx_kb_user  ON kb_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_kb_tags  ON kb_entries USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_kb_meta  ON kb_entries USING GIN(metadata);
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

-- ── Pending actions (server-side conversation state) ─────────────────────────
-- One in-flight action per user: a media-save awaiting confirmation, or a
-- staged note deletion awaiting confirmation. Survives across workers/restarts
-- (unlike an in-memory dict) and expires via expires_at. (Memorae v2 §3D)
CREATE TABLE IF NOT EXISTS pending_actions (
    user_id     UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,                 -- 'media_save' | 'delete_note'
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    expires_at  TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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