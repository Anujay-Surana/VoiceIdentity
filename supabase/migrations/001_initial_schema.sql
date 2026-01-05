-- Enable pgvector extension for voice embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Organizations table (multi-tenant support)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    api_key TEXT UNIQUE NOT NULL DEFAULT encode(gen_random_bytes(32), 'hex'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create index for API key lookups
CREATE INDEX idx_organizations_api_key ON organizations(api_key);

-- Users table (external users from client applications)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    external_user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    -- Each external_user_id must be unique within an organization
    UNIQUE(org_id, external_user_id)
);

CREATE INDEX idx_users_org_id ON users(org_id);
CREATE INDEX idx_users_external ON users(org_id, external_user_id);

-- Speakers table (unique voice identities per user)
CREATE TABLE speakers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT,
    is_identified BOOLEAN NOT NULL DEFAULT false,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_speakers_user_id ON speakers(user_id);
CREATE INDEX idx_speakers_last_seen ON speakers(last_seen DESC);

-- Voice embeddings table (stores ECAPA-TDNN 192-dim embeddings)
CREATE TABLE voice_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    speaker_id UUID NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    embedding vector(192) NOT NULL,
    quality_score FLOAT NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_voice_embeddings_speaker_id ON voice_embeddings(speaker_id);

-- Create HNSW index for fast similarity search
-- Using cosine distance (inner product on normalized vectors)
CREATE INDEX idx_voice_embeddings_vector ON voice_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Conversations table (audio recordings/sessions)
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    audio_file_url TEXT
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_started_at ON conversations(started_at DESC);

-- Segments table (speaker segments within conversations)
CREATE TABLE segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    speaker_id UUID REFERENCES speakers(id) ON DELETE SET NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    transcript TEXT,
    confidence FLOAT NOT NULL DEFAULT 1.0
);

CREATE INDEX idx_segments_conversation_id ON segments(conversation_id);
CREATE INDEX idx_segments_speaker_id ON segments(speaker_id);

-- Function to match voice embeddings using cosine similarity
CREATE OR REPLACE FUNCTION match_voice_embedding(
    query_embedding vector(192),
    match_user_id UUID,
    match_threshold FLOAT DEFAULT 0.75,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    speaker_id UUID,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ve.speaker_id,
        1 - (ve.embedding <=> query_embedding) AS similarity
    FROM voice_embeddings ve
    JOIN speakers s ON s.id = ve.speaker_id
    WHERE s.user_id = match_user_id
      AND 1 - (ve.embedding <=> query_embedding) >= match_threshold
    ORDER BY ve.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function to get centroid embedding for a speaker
CREATE OR REPLACE FUNCTION get_speaker_centroid(
    target_speaker_id UUID
)
RETURNS vector(192)
LANGUAGE plpgsql
AS $$
DECLARE
    centroid vector(192);
BEGIN
    SELECT AVG(embedding) INTO centroid
    FROM voice_embeddings
    WHERE speaker_id = target_speaker_id;
    
    RETURN centroid;
END;
$$;

-- Row Level Security (RLS) policies
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE speakers ENABLE ROW LEVEL SECURITY;
ALTER TABLE voice_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE segments ENABLE ROW LEVEL SECURITY;

-- Service role can do everything (used by our API)
CREATE POLICY "Service role full access" ON organizations
    FOR ALL USING (auth.role() = 'service_role');
    
CREATE POLICY "Service role full access" ON users
    FOR ALL USING (auth.role() = 'service_role');
    
CREATE POLICY "Service role full access" ON speakers
    FOR ALL USING (auth.role() = 'service_role');
    
CREATE POLICY "Service role full access" ON voice_embeddings
    FOR ALL USING (auth.role() = 'service_role');
    
CREATE POLICY "Service role full access" ON conversations
    FOR ALL USING (auth.role() = 'service_role');
    
CREATE POLICY "Service role full access" ON segments
    FOR ALL USING (auth.role() = 'service_role');

-- Create storage bucket for audio files
INSERT INTO storage.buckets (id, name, public)
VALUES ('audio-files', 'audio-files', false)
ON CONFLICT (id) DO NOTHING;
