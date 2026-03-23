"""Automatic system prompt analysis helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from src.config import ProviderConfig
from src.providers.run_benchmark import create_provider


def analyze_system_prompt(
    system_prompt: str,
    *,
    provider_config: ProviderConfig | None = None,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Automatically analyze system prompt and extract key components
    
    Args:
        system_prompt: The system prompt text.
        provider_config: Selected benchmark provider used for live analysis.
        use_llm: Whether to use a live provider before falling back.
    
    Returns:
        Analysis payload with extracted prompt structure and metadata about
        whether the analysis came from a live provider or heuristics.
    """

    if use_llm and provider_config is not None:
        return analyze_prompt_llm(system_prompt, provider_config)

    fallback_reason = None
    if use_llm and provider_config is None:
        fallback_reason = "No provider configured for live analysis."
    return _build_analysis_result(
        analyze_prompt_heuristic(system_prompt),
        system_prompt=system_prompt,
        method="heuristic",
        fallback_reason=fallback_reason,
    )


def analyze_prompt_llm(
    system_prompt: str,
    provider_config: ProviderConfig,
) -> dict[str, Any]:
    """Use the selected provider to analyze a system prompt."""

    analysis_prompt = _build_analysis_prompt(system_prompt)

    try:
        provider = create_provider(provider_config)
        analysis_text, _tokens, _latency = provider.call(
            (
                "You are a strict prompt analysis engine. Return valid JSON only. "
                "Do not include explanations, markdown, or prose outside the JSON."
            ),
            analysis_prompt,
        )
        analysis = _extract_analysis_json(analysis_text)
        return _build_analysis_result(
            analysis,
            system_prompt=system_prompt,
            method="llm",
            provider_label=provider.get_model_name(),
            provider_config=provider_config,
        )
    except Exception as exc:
        return _build_analysis_result(
            analyze_prompt_heuristic(system_prompt),
            system_prompt=system_prompt,
            method="heuristic",
            fallback_reason=str(exc),
            provider_config=provider_config,
        )


def _build_analysis_prompt(system_prompt: str) -> str:
    return f"""Analyze this system prompt and extract key information.

SYSTEM PROMPT:
{system_prompt}

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


def _extract_analysis_json(analysis_text: str) -> dict[str, Any]:
    start = analysis_text.find("{")
    end = analysis_text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("Provider response did not contain valid JSON.")
    analysis = json.loads(analysis_text[start:end])
    return validate_analysis(analysis)


def analyze_prompt_heuristic(system_prompt: str) -> dict[str, Any]:
    """Heuristic-based analysis fallback."""

    prompt_lower = system_prompt.lower()

    role = extract_role(system_prompt)
    domain = detect_domain(prompt_lower)
    capabilities = extract_capabilities(system_prompt)
    boundaries = extract_boundaries(system_prompt)
    constraints = detect_constraints(system_prompt)
    core_topics = extract_core_topics(system_prompt)

    return {
        "role": role,
        "domain": domain,
        "capabilities": capabilities,
        "boundaries": boundaries,
        "constraints": constraints,
        "core_topics": core_topics,
    }


def _build_analysis_result(
    analysis: dict[str, Any],
    *,
    system_prompt: str,
    method: str,
    provider_label: str | None = None,
    provider_config: ProviderConfig | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    result = validate_analysis(dict(analysis))
    result["analysis_method"] = method
    result["analysis_provider"] = provider_label
    result["analysis_fallback_reason"] = fallback_reason
    result["prompt_characters"] = len(system_prompt)
    result["provider_config"] = (
        {**asdict(provider_config), "api_key": None}
        if provider_config is not None
        else None
    )
    return result


def extract_role(prompt: str) -> str:
    """Extract the primary assistant role from a prompt."""

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
            role = role.replace("\n", " ").strip()
            if len(role) < 100:
                return role

    return "assistant"  # Default


def detect_domain(prompt_lower: str) -> str:
    """Detect the dominant prompt domain using keyword buckets."""

    domain_keywords = {
        "legal/compliance": ["legal", "compliance", "gdpr", "regulation", "law"],
        "software/code": ["code", "programming", "developer", "software", "debug"],
        "customer_support": ["support", "customer", "help", "assist", "order"],
        "education": ["teach", "tutor", "student", "learn", "explain"],
        "finance": ["financial", "investment", "trading", "money", "portfolio"],
        "healthcare": ["medical", "health", "patient", "diagnosis", "treatment"],
        "marketing": ["marketing", "campaign", "brand", "advertising"],
        "hr": ["hr", "recruitment", "hiring", "candidate", "employee"],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            return domain

    return "general"


def extract_capabilities(prompt: str) -> list[str]:
    """Extract positive assistant capabilities from prompt text."""

    capabilities = []

    positive_patterns = [
        r"you (?:can|should|must|will) ([^.\n]+)",
        r"provide ([^.\n]+)",
        r"help (?:with |users? )?([^.\n]+)",
        r"assist (?:with |in )?([^.\n]+)"
    ]
    
    for pattern in positive_patterns:
        matches = re.findall(pattern, prompt.lower())
        for match in matches[:3]:
            cap = match.strip()
            if 5 < len(cap) < 80:
                capabilities.append(cap)

    return capabilities[:5] if capabilities else ["assist users", "answer questions"]


def extract_boundaries(prompt: str) -> list[str]:
    """Extract negative guardrails from prompt text."""

    boundaries = []

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


def detect_constraints(prompt: str) -> dict[str, str | None]:
    """Detect prompt constraints around formatting and tone."""

    prompt_lower = prompt.lower()

    constraints = {
        "format": None,
        "length": None,
        "tone": None,
        "language": None,
    }

    if "json" in prompt_lower:
        constraints["format"] = "JSON"
    elif "markdown" in prompt_lower:
        constraints["format"] = "Markdown"
    elif "code" in prompt_lower and "block" in prompt_lower:
        constraints["format"] = "Code blocks"

    if "concise" in prompt_lower or "brief" in prompt_lower or "short" in prompt_lower:
        constraints["length"] = "concise"
    elif "detailed" in prompt_lower or "comprehensive" in prompt_lower:
        constraints["length"] = "detailed"

    if "professional" in prompt_lower:
        constraints["tone"] = "professional"
    elif "friendly" in prompt_lower or "casual" in prompt_lower:
        constraints["tone"] = "friendly"
    elif "formal" in prompt_lower:
        constraints["tone"] = "formal"

    if "english" in prompt_lower:
        constraints["language"] = "English"
    elif "multilingual" in prompt_lower:
        constraints["language"] = "multilingual"

    return constraints


def extract_core_topics(prompt: str) -> list[str]:
    """Extract the prompt's core topics or acronyms."""

    acronyms = re.findall(r"\b[A-Z]{2,}\b", prompt)

    topic_patterns = [
        r"(?:about|regarding|concerning) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"knowledge of ([^.\n]+)",
        r"expertise in ([^.\n]+)"
    ]

    topics = list(set(acronyms[:3]))

    for pattern in topic_patterns:
        matches = re.findall(pattern, prompt)
        for match in matches:
            topic = match.strip()
            if 2 < len(topic) < 50:
                topics.append(topic)

    return topics[:5] if topics else ["general knowledge"]


def validate_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize analysis results."""

    required_fields = ["role", "domain", "capabilities", "boundaries", "constraints", "core_topics"]

    for field in required_fields:
        if field not in analysis:
            if field in ["capabilities", "boundaries", "core_topics"]:
                analysis[field] = []
            elif field == "constraints":
                analysis[field] = {"format": None, "length": None, "tone": None, "language": None}
            else:
                analysis[field] = "unknown"

    for field in ["capabilities", "boundaries", "core_topics"]:
        if not isinstance(analysis[field], list):
            analysis[field] = [str(analysis[field])]

    if not isinstance(analysis["constraints"], dict):
        analysis["constraints"] = {"format": None, "length": None, "tone": None, "language": None}

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
