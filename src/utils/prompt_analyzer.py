"""
Automatic System Prompt Analyzer
Extracts structure and requirements from any system prompt
"""

import requests
import os
import json
import re
from typing import Dict, List


def analyze_system_prompt(system_prompt: str, use_llm: bool = True) -> Dict:
    """
    Automatically analyze system prompt and extract key components
    
    Args:
        system_prompt: The system prompt text
        use_llm: Use LLM for analysis (more accurate) vs heuristics
    
    Returns:
        Dict with:
        - role: Primary role/identity
        - domain: Domain/expertise area
        - capabilities: What it should do
        - boundaries: What it should NOT do
        - constraints: Format, length, tone, language constraints
        - style: Communication style
        - core_topics: Key topics it should know about
    """
    
    if use_llm:
        return analyze_prompt_llm(system_prompt)
    else:
        return analyze_prompt_heuristic(system_prompt)


def analyze_prompt_llm(system_prompt: str) -> Dict:
    """Use LLM to analyze system prompt"""
    
    analysis_prompt = f"""Analyze this system prompt and extract key information.

SYSTEM PROMPT:
{system_prompt[:2000]}

Extract and return a JSON object with:

1. "role": Primary role/identity (e.g., "legal compliance assistant", "Python programmer", "customer support agent")

2. "domain": Domain/expertise (e.g., "legal/compliance", "software development", "e-commerce support")

3. "capabilities": List of 3-5 things it SHOULD do (e.g., ["explain GDPR", "provide compliance guidance"])

4. "boundaries": List of 3-5 things it should NOT do (e.g., ["provide legal advice", "draft contracts"])

5. "constraints": Object with format/style constraints:
   - "format": any format requirements (e.g., "JSON", "markdown", "code blocks")
   - "length": length constraints (e.g., "concise", "detailed", "under 100 words")
   - "tone": tone/style (e.g., "professional", "friendly", "technical")
   - "language": language requirements (e.g., "English", "multilingual")

6. "core_topics": List of 2-3 core topics/concepts it should know (e.g., ["GDPR", "CCPA", "data privacy"])

Return ONLY valid JSON, no other text:
{{
  "role": "...",
  "domain": "...",
  "capabilities": ["...", "..."],
  "boundaries": ["...", "..."],
  "constraints": {{"format": "...", "length": "...", "tone": "...", "language": "..."}},
  "core_topics": ["...", "..."]
}}"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": os.getenv("OLLAMA_JUDGE_MODEL", "qwen3:14b"),
                "prompt": analysis_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=45
        )
        
        result = response.json()
        analysis_text = result.get("response", "").strip()
        
        # Extract JSON
        start = analysis_text.find('{')
        end = analysis_text.rfind('}') + 1
        
        if start >= 0 and end > start:
            analysis = json.loads(analysis_text[start:end])
            
            # Validate and clean
            analysis = validate_analysis(analysis)
            return analysis
    
    except Exception as e:
        print(f"  [WARNING] LLM analysis failed: {e}")
    
    # Fallback to heuristic
    return analyze_prompt_heuristic(system_prompt)


def analyze_prompt_heuristic(system_prompt: str) -> Dict:
    """Heuristic-based analysis (fallback)"""
    
    prompt_lower = system_prompt.lower()
    
    # Extract role
    role = extract_role(system_prompt)
    
    # Detect domain
    domain = detect_domain(prompt_lower)
    
    # Extract capabilities (things it SHOULD do)
    capabilities = extract_capabilities(system_prompt)
    
    # Extract boundaries (things it should NOT do)
    boundaries = extract_boundaries(system_prompt)
    
    # Detect constraints
    constraints = detect_constraints(system_prompt)
    
    # Extract core topics
    core_topics = extract_core_topics(system_prompt)
    
    return {
        'role': role,
        'domain': domain,
        'capabilities': capabilities,
        'boundaries': boundaries,
        'constraints': constraints,
        'core_topics': core_topics
    }


def extract_role(prompt: str) -> str:
    """Extract primary role from prompt"""
    
    # Look for "You are a/an X" patterns
    patterns = [
        r"you are (?:a |an )?([^.\n]+?)(?:\.|$)",
        r"act as (?:a |an )?([^.\n]+?)(?:\.|$)",
        r"role:\s*([^.\n]+?)(?:\.|$)",
        r"assistant (?:for |to help with )?([^.\n]+?)(?:\.|$)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prompt.lower())
        if match:
            role = match.group(1).strip()
            # Clean up
            role = role.replace('\n', ' ').strip()
            if len(role) < 100:  # Reasonable length
                return role
    
    return "assistant"  # Default


def detect_domain(prompt_lower: str) -> str:
    """Detect domain/expertise area"""
    
    domain_keywords = {
        'legal/compliance': ['legal', 'compliance', 'gdpr', 'regulation', 'law'],
        'software/code': ['code', 'programming', 'developer', 'software', 'debug'],
        'customer_support': ['support', 'customer', 'help', 'assist', 'order'],
        'education': ['teach', 'tutor', 'student', 'learn', 'explain'],
        'finance': ['financial', 'investment', 'trading', 'money', 'portfolio'],
        'healthcare': ['medical', 'health', 'patient', 'diagnosis', 'treatment'],
        'marketing': ['marketing', 'campaign', 'brand', 'advertising'],
        'hr': ['hr', 'recruitment', 'hiring', 'candidate', 'employee']
    }
    
    for domain, keywords in domain_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            return domain
    
    return "general"


def extract_capabilities(prompt: str) -> List[str]:
    """Extract what the assistant SHOULD do"""
    
    capabilities = []
    
    # Look for positive instructions
    positive_patterns = [
        r"you (?:can|should|must|will) ([^.\n]+)",
        r"provide ([^.\n]+)",
        r"help (?:with |users? )?([^.\n]+)",
        r"assist (?:with |in )?([^.\n]+)"
    ]
    
    for pattern in positive_patterns:
        matches = re.findall(pattern, prompt.lower())
        for match in matches[:3]:  # Limit to 3
            cap = match.strip()
            if 5 < len(cap) < 80:
                capabilities.append(cap)
    
    return capabilities[:5] if capabilities else ["assist users", "answer questions"]


def extract_boundaries(prompt: str) -> List[str]:
    """Extract what the assistant should NOT do"""
    
    boundaries = []
    
    # Look for negative instructions
    negative_patterns = [
        r"(?:never|don't|do not|cannot|can't) ([^.\n]+)",
        r"you (?:are not|aren't) ([^.\n]+)",
        r"(?:avoid|refrain from) ([^.\n]+)",
        r"not (?:allowed|permitted) to ([^.\n]+)"
    ]
    
    for pattern in negative_patterns:
        matches = re.findall(pattern, prompt.lower())
        for match in matches[:3]:
            boundary = match.strip()
            if 5 < len(boundary) < 80:
                boundaries.append(boundary)
    
    return boundaries[:5] if boundaries else ["provide harmful content"]


def detect_constraints(prompt: str) -> Dict[str, str]:
    """Detect format, length, tone, language constraints"""
    
    prompt_lower = prompt.lower()
    
    constraints = {
        'format': None,
        'length': None,
        'tone': None,
        'language': None
    }
    
    # Format
    if 'json' in prompt_lower:
        constraints['format'] = 'JSON'
    elif 'markdown' in prompt_lower:
        constraints['format'] = 'Markdown'
    elif 'code' in prompt_lower and 'block' in prompt_lower:
        constraints['format'] = 'Code blocks'
    
    # Length
    if 'concise' in prompt_lower or 'brief' in prompt_lower or 'short' in prompt_lower:
        constraints['length'] = 'concise'
    elif 'detailed' in prompt_lower or 'comprehensive' in prompt_lower:
        constraints['length'] = 'detailed'
    
    # Tone
    if 'professional' in prompt_lower:
        constraints['tone'] = 'professional'
    elif 'friendly' in prompt_lower or 'casual' in prompt_lower:
        constraints['tone'] = 'friendly'
    elif 'formal' in prompt_lower:
        constraints['tone'] = 'formal'
    
    # Language
    if 'english' in prompt_lower:
        constraints['language'] = 'English'
    elif 'multilingual' in prompt_lower:
        constraints['language'] = 'multilingual'
    
    return constraints


def extract_core_topics(prompt: str) -> List[str]:
    """Extract core topics/concepts the assistant should know"""
    
    # Look for capitalized terms (often acronyms or proper nouns)
    acronyms = re.findall(r'\b[A-Z]{2,}\b', prompt)
    
    # Common topic patterns
    topic_patterns = [
        r"(?:about|regarding|concerning) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"knowledge of ([^.\n]+)",
        r"expertise in ([^.\n]+)"
    ]
    
    topics = list(set(acronyms[:3]))  # Unique acronyms
    
    for pattern in topic_patterns:
        matches = re.findall(pattern, prompt)
        for match in matches:
            topic = match.strip()
            if 2 < len(topic) < 50:
                topics.append(topic)
    
    return topics[:5] if topics else ["general knowledge"]


def validate_analysis(analysis: Dict) -> Dict:
    """Validate and clean analysis results"""
    
    # Ensure all required fields exist
    required_fields = ['role', 'domain', 'capabilities', 'boundaries', 'constraints', 'core_topics']
    
    for field in required_fields:
        if field not in analysis:
            if field in ['capabilities', 'boundaries', 'core_topics']:
                analysis[field] = []
            elif field == 'constraints':
                analysis[field] = {'format': None, 'length': None, 'tone': None, 'language': None}
            else:
                analysis[field] = "unknown"
    
    # Ensure lists are actually lists
    for field in ['capabilities', 'boundaries', 'core_topics']:
        if not isinstance(analysis[field], list):
            analysis[field] = [str(analysis[field])]
    
    # Ensure constraints is a dict
    if not isinstance(analysis['constraints'], dict):
        analysis['constraints'] = {'format': None, 'length': None, 'tone': None, 'language': None}
    
    return analysis


if __name__ == '__main__':
    # Test with legal compliance prompt
    test_prompt = """
    You are a legal compliance assistant helping businesses understand regulatory requirements.
    
    You can provide general information about GDPR, CCPA, HIPAA, and other compliance frameworks.
    
    NEVER provide legal advice or draft legal documents.
    Always recommend consulting with licensed attorneys.
    
    Keep responses professional and include appropriate disclaimers.
    """
    
    print("Analyzing system prompt...\n")
    analysis = analyze_system_prompt(test_prompt, use_llm=False)
    
    print("="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(json.dumps(analysis, indent=2))
