-- Improved voice matching with speaker-level aggregation
-- This provides more robust matching by aggregating similarity scores per speaker

-- Drop the old function first
DROP FUNCTION IF EXISTS match_voice_embedding(vector(192), UUID, FLOAT, INT);

-- Improved matching function with speaker aggregation
CREATE OR REPLACE FUNCTION match_voice_embedding(
    query_embedding vector(192),
    match_user_id UUID,
    match_threshold FLOAT DEFAULT 0.60,  -- Lowered default threshold
    match_count INT DEFAULT 10  -- Get more candidates for aggregation
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

-- New function: match against speaker centroids for faster, more stable matching
-- This computes similarity against the average embedding for each speaker
CREATE OR REPLACE FUNCTION match_voice_centroid(
    query_embedding vector(192),
    match_user_id UUID,
    match_threshold FLOAT DEFAULT 0.55,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    speaker_id UUID,
    similarity FLOAT,
    embedding_count BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH speaker_centroids AS (
        SELECT 
            ve.speaker_id,
            AVG(ve.embedding) AS centroid,
            COUNT(*) AS embedding_count
        FROM voice_embeddings ve
        JOIN speakers s ON s.id = ve.speaker_id
        WHERE s.user_id = match_user_id
        GROUP BY ve.speaker_id
    )
    SELECT 
        sc.speaker_id,
        1 - (sc.centroid <=> query_embedding) AS similarity,
        sc.embedding_count
    FROM speaker_centroids sc
    WHERE 1 - (sc.centroid <=> query_embedding) >= match_threshold
    ORDER BY sc.centroid <=> query_embedding
    LIMIT match_count;
END;
$$;

-- New function: weighted match that considers quality scores
-- Higher quality embeddings have more influence on the match
CREATE OR REPLACE FUNCTION match_voice_weighted(
    query_embedding vector(192),
    match_user_id UUID,
    match_threshold FLOAT DEFAULT 0.55,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    speaker_id UUID,
    similarity FLOAT,
    weighted_similarity FLOAT,
    max_similarity FLOAT,
    match_count_per_speaker BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH matches AS (
        SELECT 
            ve.speaker_id,
            1 - (ve.embedding <=> query_embedding) AS sim,
            ve.quality_score
        FROM voice_embeddings ve
        JOIN speakers s ON s.id = ve.speaker_id
        WHERE s.user_id = match_user_id
          AND 1 - (ve.embedding <=> query_embedding) >= match_threshold * 0.7
    ),
    speaker_aggregates AS (
        SELECT
            m.speaker_id,
            AVG(m.sim) AS avg_sim,
            MAX(m.sim) AS max_sim,
            SUM(m.sim * m.quality_score) / NULLIF(SUM(m.quality_score), 0) AS weighted_sim,
            COUNT(*) AS match_count
        FROM matches m
        GROUP BY m.speaker_id
    )
    SELECT 
        sa.speaker_id,
        sa.avg_sim AS similarity,
        sa.weighted_sim AS weighted_similarity,
        sa.max_sim AS max_similarity,
        sa.match_count AS match_count_per_speaker
    FROM speaker_aggregates sa
    WHERE sa.max_sim >= match_threshold
    ORDER BY 
        -- Rank by weighted combination: max score + weighted average
        (sa.max_sim * 0.6 + sa.weighted_sim * 0.4) DESC
    LIMIT match_count;
END;
$$;

-- Function to find speakers that should potentially be merged
-- Returns pairs of speakers with high similarity between their embeddings
CREATE OR REPLACE FUNCTION find_merge_candidates(
    target_user_id UUID,
    similarity_threshold FLOAT DEFAULT 0.70
)
RETURNS TABLE (
    speaker1_id UUID,
    speaker2_id UUID,
    similarity FLOAT,
    speaker1_name TEXT,
    speaker2_name TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH speaker_pairs AS (
        SELECT DISTINCT
            s1.id AS s1_id,
            s2.id AS s2_id,
            s1.name AS s1_name,
            s2.name AS s2_name
        FROM speakers s1
        CROSS JOIN speakers s2
        WHERE s1.user_id = target_user_id
          AND s2.user_id = target_user_id
          AND s1.id < s2.id  -- Avoid duplicates
    ),
    centroid_similarities AS (
        SELECT 
            sp.s1_id,
            sp.s2_id,
            sp.s1_name,
            sp.s2_name,
            1 - (c1.centroid <=> c2.centroid) AS centroid_sim
        FROM speaker_pairs sp
        JOIN LATERAL (
            SELECT AVG(ve.embedding) AS centroid
            FROM voice_embeddings ve
            WHERE ve.speaker_id = sp.s1_id
        ) c1 ON true
        JOIN LATERAL (
            SELECT AVG(ve.embedding) AS centroid
            FROM voice_embeddings ve
            WHERE ve.speaker_id = sp.s2_id
        ) c2 ON true
    )
    SELECT 
        cs.s1_id AS speaker1_id,
        cs.s2_id AS speaker2_id,
        cs.centroid_sim AS similarity,
        cs.s1_name AS speaker1_name,
        cs.s2_name AS speaker2_name
    FROM centroid_similarities cs
    WHERE cs.centroid_sim >= similarity_threshold
    ORDER BY cs.centroid_sim DESC;
END;
$$;
