"""
Semantic metrics using embeddings for better evaluation
Requires: sentence-transformers or OpenAI embeddings
"""

import numpy as np
from typing import List, Tuple
import requests
import os


def get_embedding_ollama(text: str, model: str = "nomic-embed-text") -> np.ndarray:
    """Get embedding from Ollama (free, local)"""
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": model,
                "prompt": text
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding", [])
            return np.array(embedding)
        
    except Exception as e:
        print(f"  [WARNING] Ollama embedding failed: {e}")
    
    return None


def get_embedding_openai(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Get embedding from OpenAI (requires API key)"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.embeddings.create(
            model=model,
            input=text
        )
        
        return np.array(response.data[0].embedding)
        
    except Exception as e:
        print(f"  [WARNING] OpenAI embedding failed: {e}")
        return None


def get_embedding(text: str, provider: str = "ollama") -> np.ndarray:
    """
    Get text embedding using specified provider
    
    Args:
        text: Text to embed
        provider: 'ollama' (free, local) or 'openai' (requires API key)
    
    Returns:
        numpy array of embedding, or None if failed
    """
    if provider == "ollama":
        return get_embedding_ollama(text)
    elif provider == "openai":
        return get_embedding_openai(text)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if vec1 is None or vec2 is None:
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def semantic_consistency_score(responses: List[str], provider: str = "ollama") -> float:
    """
    Calculate semantic consistency across multiple responses
    
    Args:
        responses: List of response texts
        provider: Embedding provider ('ollama' or 'openai')
    
    Returns:
        Score from 0.0 to 1.0 (1.0 = perfectly consistent)
    """
    if len(responses) < 2:
        return 1.0  # Single response is trivially consistent
    
    # Get embeddings for all responses
    embeddings = []
    for response in responses:
        emb = get_embedding(response, provider=provider)
        if emb is not None:
            embeddings.append(emb)
    
    if len(embeddings) < 2:
        print("  [WARNING] Could not get embeddings, falling back to heuristic")
        return None  # Fallback to other method
    
    # Calculate pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    # Average similarity
    avg_similarity = np.mean(similarities)
    
    # Convert to 0-1 score
    # Cosine similarity ranges from -1 to 1, but for similar texts it's usually 0.5-1.0
    # We normalize to 0-1 scale where:
    # 0.9-1.0 similarity → 1.0 score (excellent consistency)
    # 0.7-0.9 similarity → 0.7-0.9 score (good consistency)
    # <0.7 similarity → lower score (poor consistency)
    
    if avg_similarity >= 0.9:
        score = 1.0
    elif avg_similarity >= 0.7:
        # Linear mapping from [0.7, 0.9] to [0.7, 1.0]
        score = 0.7 + (avg_similarity - 0.7) * 1.5
    else:
        # Linear mapping from [0, 0.7] to [0, 0.7]
        score = avg_similarity
    
    return min(1.0, max(0.0, score))


def semantic_robustness_score(baseline_response: str, 
                              perturbed_responses: List[str],
                              provider: str = "ollama") -> float:
    """
    Calculate robustness to input perturbations (typos, paraphrases)
    
    Args:
        baseline_response: Response to original query
        perturbed_responses: Responses to perturbed queries
        provider: Embedding provider
    
    Returns:
        Score from 0.0 to 1.0 (1.0 = perfectly robust)
    """
    baseline_emb = get_embedding(baseline_response, provider=provider)
    if baseline_emb is None:
        return None
    
    similarities = []
    for perturbed in perturbed_responses:
        perturbed_emb = get_embedding(perturbed, provider=provider)
        if perturbed_emb is not None:
            sim = cosine_similarity(baseline_emb, perturbed_emb)
            similarities.append(sim)
    
    if not similarities:
        return None
    
    avg_similarity = np.mean(similarities)
    
    # Similar normalization as consistency
    if avg_similarity >= 0.85:
        return 1.0
    elif avg_similarity >= 0.6:
        return 0.6 + (avg_similarity - 0.6) * 1.6
    else:
        return avg_similarity
    
    return min(1.0, max(0.0, avg_similarity))


def extract_key_points(text: str, max_points: int = 5) -> List[str]:
    """
    Extract key points from text using simple heuristics
    (Can be enhanced with LLM later)
    """
    # Simple sentence splitting
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    # Return first N sentences as key points
    return sentences[:max_points]


def semantic_coverage_score(response: str, expected_points: List[str], 
                           provider: str = "ollama") -> float:
    """
    Check if response covers expected key points semantically
    
    Args:
        response: Model's response
        expected_points: List of expected key points/topics
        provider: Embedding provider
    
    Returns:
        Score from 0.0 to 1.0 (1.0 = all points covered)
    """
    response_emb = get_embedding(response, provider=provider)
    if response_emb is None:
        return None
    
    coverage_scores = []
    for point in expected_points:
        point_emb = get_embedding(point, provider=provider)
        if point_emb is not None:
            sim = cosine_similarity(response_emb, point_emb)
            # If similarity > 0.6, consider point covered
            coverage_scores.append(1.0 if sim > 0.6 else sim)
    
    if not coverage_scores:
        return None
    
    return np.mean(coverage_scores)


# Fallback: Simple text-based consistency (no embeddings)
def text_based_consistency(responses: List[str]) -> float:
    """
    Fallback consistency check using simple text metrics
    """
    if len(responses) < 2:
        return 1.0
    
    # Check length consistency
    lengths = [len(r) for r in responses]
    avg_length = np.mean(lengths)
    length_variance = np.std(lengths) / avg_length if avg_length > 0 else 1.0
    
    # Check word overlap
    word_sets = [set(r.lower().split()) for r in responses]
    
    overlaps = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            jaccard = intersection / union if union > 0 else 0
            overlaps.append(jaccard)
    
    avg_overlap = np.mean(overlaps) if overlaps else 0.0
    
    # Combine metrics
    length_score = max(0, 1.0 - length_variance)
    overlap_score = avg_overlap
    
    return (length_score * 0.3 + overlap_score * 0.7)


if __name__ == '__main__':
    # Test semantic similarity
    responses = [
        "GDPR is the General Data Protection Regulation, a European privacy law.",
        "The General Data Protection Regulation (GDPR) is an EU regulation on data protection.",
        "GDPR stands for General Data Protection Regulation and governs data privacy in Europe."
    ]
    
    print("Testing semantic consistency...")
    score = semantic_consistency_score(responses, provider="ollama")
    print(f"Consistency score: {score:.2f}")
    
    # Test fallback
    print("\nTesting text-based fallback...")
    fallback_score = text_based_consistency(responses)
    print(f"Fallback score: {fallback_score:.2f}")
