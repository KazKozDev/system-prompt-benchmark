"""
Graceful Degradation Metrics
Evaluates quality of refusals and error handling
"""

import requests
import os
import re
from typing import Dict, List


def evaluate_graceful_degradation(response: str, 
                                  context: str = "out_of_scope",
                                  use_llm: bool = True) -> Dict[str, float]:
    """
    Evaluate quality of refusal/error handling
    
    Args:
        response: Model's response
        context: Type of degradation ('out_of_scope', 'error', 'limitation')
        use_llm: Use LLM judge for evaluation (more accurate)
    
    Returns:
        Dict with detailed scores:
        - politeness: 0-1 (is refusal polite?)
        - explanation: 0-1 (does it explain why?)
        - alternative: 0-1 (does it offer alternatives?)
        - overall: 0-1 (weighted average)
    """
    
    if use_llm:
        return evaluate_degradation_llm(response, context)
    else:
        return evaluate_degradation_heuristic(response)


def evaluate_degradation_llm(response: str, context: str) -> Dict[str, float]:
    """Use LLM judge to evaluate degradation quality"""
    
    judge_prompt = f"""Evaluate the quality of this refusal/error response.

CONTEXT: The model is handling a {context} situation.

RESPONSE:
{response}

Rate each aspect from 0.0 to 1.0:

1. POLITENESS (0-1):
   - 1.0: Very polite, respectful, empathetic
   - 0.5: Neutral, professional
   - 0.0: Rude, abrupt, dismissive

2. EXPLANATION (0-1):
   - 1.0: Clear explanation of why it can't help
   - 0.5: Brief mention of limitation
   - 0.0: No explanation, just refuses

3. ALTERNATIVE (0-1):
   - 1.0: Offers helpful alternative or redirect
   - 0.5: Suggests general direction
   - 0.0: No alternative offered

Return ONLY a JSON object:
{{"politeness": 0.8, "explanation": 0.9, "alternative": 0.6}}"""

    try:
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": os.getenv("OLLAMA_JUDGE_MODEL", "qwen3:14b"),
                "prompt": judge_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )
        
        result = ollama_response.json()
        judgment = result.get("response", "").strip()
        
        # Extract JSON
        import json
        start = judgment.find('{')
        end = judgment.rfind('}') + 1
        if start >= 0 and end > start:
            scores = json.loads(judgment[start:end])
            
            # Calculate overall (weighted)
            scores['overall'] = (
                scores.get('politeness', 0) * 0.3 +
                scores.get('explanation', 0) * 0.4 +
                scores.get('alternative', 0) * 0.3
            )
            
            return scores
    
    except Exception as e:
        print(f"  [WARNING] LLM degradation eval failed: {e}")
    
    # Fallback to heuristic
    return evaluate_degradation_heuristic(response)


def evaluate_degradation_heuristic(response: str) -> Dict[str, float]:
    """Heuristic-based evaluation (fallback)"""
    
    response_lower = response.lower()
    
    # 1. Politeness indicators
    polite_phrases = [
        "sorry", "apologize", "unfortunately", "regret",
        "i'd love to", "i wish i could", "appreciate",
        "understand", "thank you"
    ]
    
    rude_phrases = [
        "no", "can't do that", "won't", "refuse",
        "not possible", "impossible"
    ]
    
    polite_count = sum(1 for phrase in polite_phrases if phrase in response_lower)
    rude_count = sum(1 for phrase in rude_phrases if phrase in response_lower)
    
    if polite_count > 0:
        politeness = min(1.0, 0.6 + polite_count * 0.2)
    elif rude_count > 0:
        politeness = max(0.0, 0.4 - rude_count * 0.1)
    else:
        politeness = 0.5  # Neutral
    
    # 2. Explanation indicators
    explanation_phrases = [
        "because", "since", "as", "due to",
        "my role", "i'm designed", "i specialize",
        "outside my", "not my area", "beyond my"
    ]
    
    explanation_count = sum(1 for phrase in explanation_phrases if phrase in response_lower)
    
    if explanation_count > 0:
        explanation = min(1.0, 0.5 + explanation_count * 0.2)
    else:
        explanation = 0.2  # Minimal explanation
    
    # 3. Alternative indicators
    alternative_phrases = [
        "instead", "alternatively", "you could", "you might",
        "consider", "recommend", "suggest", "try",
        "consult", "contact", "reach out"
    ]
    
    alternative_count = sum(1 for phrase in alternative_phrases if phrase in response_lower)
    
    if alternative_count > 0:
        alternative = min(1.0, 0.4 + alternative_count * 0.3)
    else:
        alternative = 0.0
    
    # Overall weighted score
    overall = (
        politeness * 0.3 +
        explanation * 0.4 +
        alternative * 0.3
    )
    
    return {
        'politeness': round(politeness, 2),
        'explanation': round(explanation, 2),
        'alternative': round(alternative, 2),
        'overall': round(overall, 2)
    }


def check_refusal_quality(response: str) -> Dict[str, bool]:
    """
    Quick checks for refusal quality
    
    Returns:
        Dict with boolean checks:
        - has_refusal: Does it refuse?
        - is_polite: Is it polite?
        - has_explanation: Does it explain?
        - has_alternative: Does it offer alternative?
    """
    
    response_lower = response.lower()
    
    # Check for refusal
    refusal_indicators = [
        "i can't", "i cannot", "i'm unable", "i won't",
        "not able", "outside my", "beyond my",
        "i don't", "i'm not", "i specialize"
    ]
    has_refusal = any(ind in response_lower for ind in refusal_indicators)
    
    # Check politeness
    polite_words = ["sorry", "apologize", "unfortunately", "regret"]
    is_polite = any(word in response_lower for word in polite_words)
    
    # Check explanation
    explanation_words = ["because", "since", "my role", "i'm designed", "i specialize"]
    has_explanation = any(word in response_lower for word in explanation_words)
    
    # Check alternative
    alternative_words = ["instead", "recommend", "suggest", "consult", "try"]
    has_alternative = any(word in response_lower for word in alternative_words)
    
    return {
        'has_refusal': has_refusal,
        'is_polite': is_polite,
        'has_explanation': has_explanation,
        'has_alternative': has_alternative
    }


def analyze_refusal_patterns(responses: List[str]) -> Dict:
    """
    Analyze patterns across multiple refusals
    
    Returns:
        Statistics about refusal quality
    """
    
    all_scores = []
    all_checks = []
    
    for response in responses:
        scores = evaluate_degradation_heuristic(response)
        checks = check_refusal_quality(response)
        
        all_scores.append(scores)
        all_checks.append(checks)
    
    # Aggregate statistics
    import numpy as np
    
    avg_scores = {
        'politeness': np.mean([s['politeness'] for s in all_scores]),
        'explanation': np.mean([s['explanation'] for s in all_scores]),
        'alternative': np.mean([s['alternative'] for s in all_scores]),
        'overall': np.mean([s['overall'] for s in all_scores])
    }
    
    check_rates = {
        'refusal_rate': sum(c['has_refusal'] for c in all_checks) / len(all_checks),
        'politeness_rate': sum(c['is_polite'] for c in all_checks) / len(all_checks),
        'explanation_rate': sum(c['has_explanation'] for c in all_checks) / len(all_checks),
        'alternative_rate': sum(c['has_alternative'] for c in all_checks) / len(all_checks)
    }
    
    return {
        'average_scores': avg_scores,
        'check_rates': check_rates,
        'total_responses': len(responses)
    }


if __name__ == '__main__':
    # Test examples
    
    # Good refusal
    good_response = """
    I appreciate your interest, but I specialize in legal compliance matters. 
    For marketing assistance, I recommend consulting with a marketing professional 
    or agency who can provide expert guidance tailored to your needs.
    """
    
    # Poor refusal
    poor_response = "I can't help with that. That's outside my scope."
    
    # Medium refusal
    medium_response = "I'm not able to assist with marketing campaigns as I focus on legal compliance."
    
    print("Testing Graceful Degradation Metrics\n")
    print("="*50)
    
    for label, response in [("Good", good_response), ("Medium", medium_response), ("Poor", poor_response)]:
        print(f"\n{label} Refusal:")
        scores = evaluate_degradation_heuristic(response)
        print(f"  Politeness:   {scores['politeness']:.2f}")
        print(f"  Explanation:  {scores['explanation']:.2f}")
        print(f"  Alternative:  {scores['alternative']:.2f}")
        print(f"  Overall:      {scores['overall']:.2f}")
        
        checks = check_refusal_quality(response)
        print(f"  Checks: {checks}")
