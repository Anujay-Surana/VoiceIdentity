"""Voice matching using pgvector similarity search.

Enhanced features:
- Centroid-based matching for more robust speaker identification
- Adaptive thresholds based on audio quality
- Multi-embedding aggregation
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from supabase import Client

from app.models.schemas import VoiceMatchResult
from app.core.embeddings import EmbeddingExtractor

# Set up logging for debugging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VoiceMatcher:
    """
    Match voice embeddings against known speakers using pgvector.
    
    Uses cosine similarity for matching, with configurable threshold
    to determine if a voice belongs to an existing speaker or is new.
    
    Improved matching strategy:
    - Aggregates similarities per speaker (not just individual embeddings)
    - Uses weighted averaging based on quality scores
    - Supports multi-embedding matching for more robust results
    """
    
    def __init__(
        self,
        supabase: Client,
        embedding_extractor: EmbeddingExtractor,
        threshold: float = 0.45,  # Very low for real-world streaming
        merge_threshold: float = 0.55,  # For auto-merging speakers
        top_k: int = 10,  # Increased to get more candidates for aggregation
    ):
        """
        Initialize the voice matcher.
        
        Args:
            supabase: Supabase client
            embedding_extractor: Embedding extractor for similarity computation
            threshold: Minimum similarity score to consider a match (0-1)
            merge_threshold: Threshold for suggesting speaker merges
            top_k: Number of top candidates to retrieve
        """
        self.supabase = supabase
        self.embedding_extractor = embedding_extractor
        self.threshold = threshold
        self.merge_threshold = merge_threshold
        self.top_k = top_k
    
    async def match(
        self,
        embedding: np.ndarray,
        user_id: str,
        threshold: Optional[float] = None,
    ) -> VoiceMatchResult:
        """
        Match an embedding against known speakers for a user.
        
        Uses aggregated scoring: instead of just taking the best single 
        embedding match, we aggregate scores per speaker to get more
        robust matching.
        
        Args:
            embedding: Voice embedding to match (192-dim)
            user_id: User ID to search within
            threshold: Override similarity threshold
            
        Returns:
            VoiceMatchResult with match status and speaker info
        """
        effective_threshold = threshold or self.threshold
        
        # Use a very low threshold for initial retrieval to see ALL candidates
        retrieval_threshold = 0.30  # Very low to catch everything
        
        # Convert embedding to list for Supabase RPC
        embedding_list = embedding.tolist()
        
        logger.info(f"[MATCH] Searching for matches with retrieval_threshold={retrieval_threshold}, final_threshold={effective_threshold}")
        
        # Call the match_voice_embedding function in Supabase
        # Get more candidates for aggregation
        try:
            result = self.supabase.rpc(
                "match_voice_embedding",
                {
                    "query_embedding": embedding_list,
                    "match_user_id": user_id,
                    "match_threshold": retrieval_threshold,
                    "match_count": 50,  # Get many for debugging
                }
            ).execute()
        except Exception as e:
            logger.error(f"[MATCH] Database query failed: {e}")
            return VoiceMatchResult(is_match=False, speaker_id=None, confidence=0.0)
        
        if not result.data:
            logger.info(f"[MATCH] No embeddings found in database for user {user_id}")
            return VoiceMatchResult(
                is_match=False,
                speaker_id=None,
                confidence=0.0,
            )
        
        # Log all raw matches for debugging
        logger.info(f"[MATCH] Found {len(result.data)} raw matches:")
        for match in result.data[:10]:  # Log top 10
            logger.info(f"  - Speaker {match['speaker_id'][:8]}... similarity={match['similarity']:.4f}")
        
        # Aggregate scores by speaker
        # This helps when a speaker has multiple embeddings with varying scores
        speaker_scores = self._aggregate_speaker_scores(result.data)
        
        # Find best speaker by aggregated score
        best_speaker_id = None
        best_score = 0.0
        
        logger.info(f"[MATCH] Aggregated scores by speaker:")
        for speaker_id, score in sorted(speaker_scores.items(), key=lambda x: -x[1]):
            logger.info(f"  - Speaker {speaker_id[:8]}... aggregated_score={score:.4f}")
            if score > best_score:
                best_score = score
                best_speaker_id = speaker_id
        
        # Check if best aggregated score meets threshold
        if best_score < effective_threshold:
            logger.info(f"[MATCH] Best score {best_score:.4f} < threshold {effective_threshold} - NO MATCH")
            return VoiceMatchResult(
                is_match=False,
                speaker_id=None,
                confidence=best_score,  # Return score for debugging
            )
        
        logger.info(f"[MATCH] MATCHED speaker {best_speaker_id[:8]}... with score {best_score:.4f}")
        return VoiceMatchResult(
            is_match=True,
            speaker_id=best_speaker_id,
            confidence=best_score,
        )
    
    def _aggregate_speaker_scores(self, matches: list[dict]) -> dict[str, float]:
        """
        Aggregate similarity scores by speaker.
        
        Uses a weighted approach:
        - Top score gets weight 1.0
        - Additional scores get diminishing weight (helps with speakers who have more embeddings)
        - Returns the weighted average for each speaker
        
        This prevents bias towards speakers with many embeddings while still
        benefiting from multiple matching embeddings.
        """
        speaker_matches: dict[str, list[float]] = defaultdict(list)
        
        for match in matches:
            speaker_matches[match["speaker_id"]].append(match["similarity"])
        
        aggregated = {}
        for speaker_id, scores in speaker_matches.items():
            # Sort descending
            scores.sort(reverse=True)
            
            # Weighted aggregation: top score matters most
            # Formula: max_score * 0.7 + mean(top_3) * 0.3
            max_score = scores[0]
            top_scores = scores[:3]
            mean_top = sum(top_scores) / len(top_scores)
            
            # Combine: emphasize max but boost if multiple good matches
            aggregated[speaker_id] = max_score * 0.7 + mean_top * 0.3
        
        return aggregated
    
    async def match_with_candidates(
        self,
        embedding: np.ndarray,
        user_id: str,
        threshold: Optional[float] = None,
    ) -> tuple[VoiceMatchResult, list[dict]]:
        """
        Match embedding and return all candidates above threshold.
        
        Useful for debugging or showing alternative matches.
        
        Args:
            embedding: Voice embedding to match
            user_id: User ID to search within
            threshold: Override similarity threshold
            
        Returns:
            Tuple of (best match result, list of all candidates)
        """
        effective_threshold = threshold or self.threshold
        retrieval_threshold = max(0.45, effective_threshold - 0.15)
        embedding_list = embedding.tolist()
        
        result = self.supabase.rpc(
            "match_voice_embedding",
            {
                "query_embedding": embedding_list,
                "match_user_id": user_id,
                "match_threshold": retrieval_threshold,
                "match_count": self.top_k * 3,
            }
        ).execute()
        
        candidates = result.data or []
        
        if not candidates:
            return VoiceMatchResult(
                is_match=False,
                speaker_id=None,
                confidence=0.0,
            ), []
        
        # Aggregate by speaker
        speaker_scores = self._aggregate_speaker_scores(candidates)
        
        # Convert to list of unique speakers with aggregated scores
        unique_candidates = []
        for speaker_id, score in sorted(speaker_scores.items(), key=lambda x: -x[1]):
            unique_candidates.append({
                "speaker_id": speaker_id,
                "similarity": score,
            })
        
        # Find best match above threshold
        best_match = unique_candidates[0] if unique_candidates else None
        
        if not best_match or best_match["similarity"] < effective_threshold:
            return VoiceMatchResult(
                is_match=False,
                speaker_id=None,
                confidence=best_match["similarity"] if best_match else 0.0,
            ), unique_candidates
        
        return VoiceMatchResult(
            is_match=True,
            speaker_id=best_match["speaker_id"],
            confidence=best_match["similarity"],
        ), unique_candidates
    
    async def match_multiple_embeddings(
        self,
        embeddings: list[np.ndarray],
        user_id: str,
        threshold: Optional[float] = None,
    ) -> VoiceMatchResult:
        """
        Match using multiple embeddings for more robust results.
        
        This is useful when you have multiple segments from the same speaker
        (e.g., from diarization) and want a more reliable match.
        
        Strategy:
        - Match each embedding independently
        - Aggregate votes per speaker
        - Use voting + score combination for final decision
        
        Args:
            embeddings: List of voice embeddings to match
            user_id: User ID to search within
            threshold: Override similarity threshold
            
        Returns:
            VoiceMatchResult with aggregated confidence
        """
        if not embeddings:
            return VoiceMatchResult(is_match=False, speaker_id=None, confidence=0.0)
        
        effective_threshold = threshold or self.threshold
        
        # Collect all matches from all embeddings
        all_speaker_scores: dict[str, list[float]] = defaultdict(list)
        
        for emb in embeddings:
            result = self.supabase.rpc(
                "match_voice_embedding",
                {
                    "query_embedding": emb.tolist(),
                    "match_user_id": user_id,
                    "match_threshold": 0.40,  # Low threshold to get candidates
                    "match_count": 20,
                }
            ).execute()
            
            if result.data:
                for match in result.data:
                    all_speaker_scores[match["speaker_id"]].append(match["similarity"])
        
        if not all_speaker_scores:
            return VoiceMatchResult(is_match=False, speaker_id=None, confidence=0.0)
        
        # Score each speaker based on:
        # 1. How many embeddings matched them
        # 2. Average similarity across matches
        best_speaker_id = None
        best_combined_score = 0.0
        
        for speaker_id, scores in all_speaker_scores.items():
            # Vote ratio: what fraction of our embeddings matched this speaker?
            vote_ratio = len(scores) / len(embeddings)
            
            # Average score
            avg_score = sum(scores) / len(scores)
            
            # Max score (best single match)
            max_score = max(scores)
            
            # Combined score: vote ratio + scores
            # Heavy weight on vote ratio (consistency across embeddings)
            combined = vote_ratio * 0.4 + avg_score * 0.3 + max_score * 0.3
            
            if combined > best_combined_score:
                best_combined_score = combined
                best_speaker_id = speaker_id
        
        # Adjust threshold for combined score (which maxes at ~1.0)
        adjusted_threshold = effective_threshold * 0.85
        
        if best_combined_score >= adjusted_threshold:
            return VoiceMatchResult(
                is_match=True,
                speaker_id=best_speaker_id,
                confidence=best_combined_score,
            )
        
        return VoiceMatchResult(
            is_match=False,
            speaker_id=None,
            confidence=best_combined_score,
        )
    
    def are_same_speaker(
        self,
        embeddings1: list[np.ndarray],
        embeddings2: list[np.ndarray],
        threshold: Optional[float] = None,
    ) -> tuple[bool, float]:
        """
        Check if two sets of embeddings likely belong to the same speaker.
        
        Useful for cross-label verification in diarization results.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            threshold: Similarity threshold (defaults to merge_threshold)
            
        Returns:
            Tuple of (is_same_speaker, similarity_score)
        """
        effective_threshold = threshold or self.merge_threshold
        
        if not embeddings1 or not embeddings2:
            return False, 0.0
        
        # Compute all pairwise similarities
        similarities = []
        for e1 in embeddings1:
            for e2 in embeddings2:
                sim = float(np.dot(e1, e2))  # Cosine similarity for normalized vectors
                similarities.append(sim)
        
        # Use multiple metrics for robustness
        max_sim = max(similarities)
        avg_sim = sum(similarities) / len(similarities)
        
        # Take top-k similarities to avoid outliers dragging down average
        top_k = min(5, len(similarities))
        top_sims = sorted(similarities, reverse=True)[:top_k]
        top_avg = sum(top_sims) / len(top_sims)
        
        # Combined score
        combined_score = max_sim * 0.4 + top_avg * 0.4 + avg_sim * 0.2
        
        return combined_score >= effective_threshold, combined_score
    
    async def store_embedding(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        quality_score: float = 1.0,
    ) -> Optional[str]:
        """
        Store a voice embedding for a speaker.
        
        Args:
            speaker_id: Speaker to associate embedding with
            embedding: Voice embedding (192-dim)
            quality_score: Quality score for this embedding (0-1)
            
        Returns:
            ID of the created embedding record, or None on failure
        """
        try:
            result = self.supabase.table("voice_embeddings").insert({
                "speaker_id": speaker_id,
                "embedding": embedding.tolist(),
                "quality_score": quality_score,
            }).execute()
            
            embedding_id = result.data[0]["id"]
            logger.info(f"[STORE] Stored embedding {embedding_id[:8]}... for speaker {speaker_id[:8]}...")
            return embedding_id
        except Exception as e:
            logger.error(f"[STORE] Failed to store embedding for speaker {speaker_id[:8]}...: {e}")
            return None
    
    async def count_user_embeddings(self, user_id: str) -> int:
        """Count total embeddings for a user (for debugging)."""
        try:
            result = self.supabase.table("voice_embeddings").select(
                "id", count="exact"
            ).execute()
            # This counts all embeddings - for user-specific we'd need a join
            return result.count or 0
        except Exception as e:
            logger.error(f"[COUNT] Failed to count embeddings: {e}")
            return 0
    
    async def get_speaker_embeddings(
        self,
        speaker_id: str,
        limit: int = 100,
    ) -> np.ndarray:
        """
        Get all embeddings for a speaker.
        
        Args:
            speaker_id: Speaker to get embeddings for
            limit: Maximum embeddings to retrieve
            
        Returns:
            Array of shape (n_embeddings, 192)
        """
        result = self.supabase.table("voice_embeddings").select(
            "embedding"
        ).eq("speaker_id", speaker_id).limit(limit).execute()
        
        if not result.data:
            return np.array([])
        
        embeddings = [np.array(row["embedding"]) for row in result.data]
        return np.stack(embeddings)
    
    async def get_speaker_centroid(
        self,
        speaker_id: str,
    ) -> Optional[np.ndarray]:
        """
        Get the centroid embedding for a speaker.
        
        Uses the database function for efficiency.
        
        Args:
            speaker_id: Speaker to get centroid for
            
        Returns:
            Centroid embedding or None if speaker has no embeddings
        """
        try:
            result = self.supabase.rpc(
                "get_speaker_centroid",
                {"target_speaker_id": speaker_id}
            ).execute()
            
            if result.data is None:
                return None
            
            # Handle various return types from the RPC
            centroid = np.array(result.data)
            
            # Ensure we have a valid 1D array with expected embedding dimensions
            if centroid.ndim == 0:
                logger.warning(f"[CENTROID] Got 0-dimensional array for speaker {speaker_id[:8]}...")
                return None
            
            if centroid.size == 0:
                return None
            
            return centroid
        except Exception as e:
            logger.error(f"[CENTROID] Error getting centroid for speaker {speaker_id[:8]}...: {e}")
            return None
    
    async def verify_speakers(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Verify if two embeddings are from the same speaker.
        
        This is a simple cosine similarity computation, useful for
        quick verification without database lookup.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        return float(np.dot(embedding1, embedding2))
    
    async def match_with_centroids(
        self,
        embedding: np.ndarray,
        user_id: str,
        threshold: Optional[float] = None,
        quality_score: Optional[float] = None,
        use_adaptive_threshold: bool = True,
    ) -> VoiceMatchResult:
        """
        Match embedding against speaker centroids for more robust matching.
        
        Centroids are the average of all embeddings for a speaker, providing
        a more stable representation than individual embeddings.
        
        Args:
            embedding: Voice embedding to match.
            user_id: User ID to search within.
            threshold: Override similarity threshold.
            quality_score: Audio quality score for adaptive thresholds.
            use_adaptive_threshold: Whether to adjust threshold based on quality.
            
        Returns:
            VoiceMatchResult with match status and speaker info.
        """
        base_threshold = threshold or self.threshold
        
        # Apply adaptive threshold if quality score provided
        if use_adaptive_threshold and quality_score is not None:
            effective_threshold = self._get_adaptive_threshold(quality_score, base_threshold)
            logger.info(f"[CENTROID-MATCH] Adaptive threshold: {effective_threshold:.3f} (quality={quality_score:.2f})")
        else:
            effective_threshold = base_threshold
        
        # Get all speaker IDs for this user
        speakers_result = self.supabase.table("speakers").select(
            "id, name"
        ).eq("user_id", user_id).execute()
        
        if not speakers_result.data:
            logger.info(f"[CENTROID-MATCH] No speakers found for user {user_id[:8]}...")
            return VoiceMatchResult(is_match=False, speaker_id=None, confidence=0.0)
        
        # Compute similarity against each speaker's centroid
        speaker_scores: List[Tuple[str, str, float]] = []
        
        for speaker in speakers_result.data:
            speaker_id = speaker["id"]
            speaker_name = speaker.get("name", "Unknown")
            
            # Get centroid from database
            centroid = await self.get_speaker_centroid(speaker_id)
            
            if centroid is None or centroid.size == 0:
                continue
            
            # Compute similarity
            similarity = float(np.dot(embedding, centroid))
            speaker_scores.append((speaker_id, speaker_name, similarity))
            
            logger.debug(f"[CENTROID-MATCH] Speaker {speaker_name}: sim={similarity:.4f}")
        
        if not speaker_scores:
            logger.info("[CENTROID-MATCH] No speakers with embeddings found")
            return VoiceMatchResult(is_match=False, speaker_id=None, confidence=0.0)
        
        # Sort by similarity
        speaker_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Log top matches
        logger.info(f"[CENTROID-MATCH] Top matches:")
        for sid, name, sim in speaker_scores[:5]:
            logger.info(f"  - {name}: {sim:.4f}")
        
        # Check if best match exceeds threshold
        best_id, best_name, best_sim = speaker_scores[0]
        
        if best_sim >= effective_threshold:
            logger.info(f"[CENTROID-MATCH] MATCHED {best_name} with sim={best_sim:.4f}")
            return VoiceMatchResult(
                is_match=True,
                speaker_id=best_id,
                confidence=best_sim,
            )
        
        logger.info(f"[CENTROID-MATCH] No match (best={best_sim:.4f} < threshold={effective_threshold:.4f})")
        return VoiceMatchResult(
            is_match=False,
            speaker_id=None,
            confidence=best_sim,
        )
    
    async def match_hybrid(
        self,
        embedding: np.ndarray,
        user_id: str,
        threshold: Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> VoiceMatchResult:
        """
        Hybrid matching combining centroid and individual embedding matching.
        
        This provides the best of both worlds:
        - Centroid matching for stability
        - Individual embedding matching for detecting variations
        
        Args:
            embedding: Voice embedding to match.
            user_id: User ID to search within.
            threshold: Override similarity threshold.
            quality_score: Audio quality score for adaptive thresholds.
            
        Returns:
            VoiceMatchResult combining both approaches.
        """
        effective_threshold = threshold or self.threshold
        
        # Get centroid-based match
        centroid_result = await self.match_with_centroids(
            embedding, user_id, threshold, quality_score
        )
        
        # Get individual embedding match
        individual_result = await self.match(embedding, user_id, threshold)
        
        # Combine results
        if centroid_result.is_match and individual_result.is_match:
            # Both agree - use the one with higher confidence
            if centroid_result.speaker_id == individual_result.speaker_id:
                # Same speaker - high confidence
                combined_confidence = max(centroid_result.confidence, individual_result.confidence)
                logger.info(f"[HYBRID] Both methods agree on speaker, conf={combined_confidence:.4f}")
                return VoiceMatchResult(
                    is_match=True,
                    speaker_id=centroid_result.speaker_id,
                    confidence=combined_confidence,
                )
            else:
                # Different speakers - use centroid (more stable)
                logger.warning(f"[HYBRID] Methods disagree! Centroid={centroid_result.speaker_id[:8]}..., "
                             f"Individual={individual_result.speaker_id[:8]}... Using centroid.")
                return centroid_result
        
        elif centroid_result.is_match:
            return centroid_result
        elif individual_result.is_match:
            return individual_result
        else:
            # Neither matched
            return VoiceMatchResult(
                is_match=False,
                speaker_id=None,
                confidence=max(centroid_result.confidence, individual_result.confidence),
            )
    
    def _get_adaptive_threshold(
        self,
        quality_score: float,
        base_threshold: float,
    ) -> float:
        """
        Compute adaptive threshold based on audio quality.
        
        - High quality audio (>0.7): Lower threshold (more permissive)
        - Low quality audio (<0.4): Higher threshold (more strict to avoid false positives)
        - Medium quality: Use base threshold
        
        Args:
            quality_score: Audio quality score (0-1).
            base_threshold: Base matching threshold.
            
        Returns:
            Adjusted threshold.
        """
        if quality_score > 0.7:
            # High quality - be more permissive
            adjustment = -0.05
        elif quality_score > 0.5:
            # Medium-high quality - slight adjustment
            adjustment = -0.02
        elif quality_score > 0.4:
            # Medium quality - no adjustment
            adjustment = 0.0
        elif quality_score > 0.25:
            # Low quality - be more strict
            adjustment = 0.05
        else:
            # Very low quality - very strict
            adjustment = 0.10
        
        adjusted = base_threshold + adjustment
        
        # Clamp to reasonable range
        return float(np.clip(adjusted, 0.30, 0.70))
    
    async def update_speaker_centroid(
        self,
        speaker_id: str,
    ) -> Optional[np.ndarray]:
        """
        Recompute and update the centroid for a speaker.
        
        This should be called after adding new embeddings to keep
        the centroid up to date.
        
        Args:
            speaker_id: Speaker to update centroid for.
            
        Returns:
            New centroid or None if speaker has no embeddings.
        """
        embeddings = await self.get_speaker_embeddings(speaker_id)
        
        if len(embeddings) == 0:
            return None
        
        # Compute centroid (mean of all embeddings)
        centroid = np.mean(embeddings, axis=0)
        
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        
        logger.info(f"[CENTROID] Updated centroid for speaker {speaker_id[:8]}... "
                   f"from {len(embeddings)} embeddings")
        
        return centroid
    
    async def find_similar_speakers(
        self,
        speaker_id: str,
        user_id: str,
        min_similarity: float = 0.6,
    ) -> list[dict]:
        """
        Find speakers similar to a given speaker.
        
        Useful for suggesting potential speaker merges.
        
        Args:
            speaker_id: Speaker to find similar speakers for
            user_id: User ID to search within
            min_similarity: Minimum similarity to include
            
        Returns:
            List of similar speakers with similarity scores
        """
        # Get the target speaker's centroid
        centroid = await self.get_speaker_centroid(speaker_id)
        
        if centroid is None:
            return []
        
        # Find similar speakers (excluding self)
        result = self.supabase.rpc(
            "match_voice_embedding",
            {
                "query_embedding": centroid.tolist(),
                "match_user_id": user_id,
                "match_threshold": min_similarity,
                "match_count": 20,  # Get more candidates
            }
        ).execute()
        
        # Filter out the original speaker and aggregate by speaker
        speaker_scores: dict[str, list[float]] = {}
        
        for match in result.data or []:
            sid = match["speaker_id"]
            if sid != speaker_id:
                if sid not in speaker_scores:
                    speaker_scores[sid] = []
                speaker_scores[sid].append(match["similarity"])
        
        # Calculate average similarity per speaker
        similar_speakers = []
        for sid, scores in speaker_scores.items():
            avg_score = sum(scores) / len(scores)
            
            # Get speaker info
            speaker_info = self.supabase.table("speakers").select(
                "id, name, is_identified"
            ).eq("id", sid).execute()
            
            if speaker_info.data:
                info = speaker_info.data[0]
                similar_speakers.append({
                    "speaker_id": sid,
                    "name": info.get("name"),
                    "is_identified": info.get("is_identified", False),
                    "similarity": avg_score,
                })
        
        # Sort by similarity descending
        similar_speakers.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similar_speakers
