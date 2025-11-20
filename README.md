# System Prompt Security Benchmark

Test your LLM system prompts against 287 real-world attack vectors including prompt injection, jailbreaks, and data leaks.

## Features

- **9 Production-Ready Prompts** - Customer support, sales, HR, legal, finance, code review, and more
- **287 Attack Vectors** - Covering all 2024-2025 jailbreak techniques
- **5 LLM Providers** - OpenAI, Anthropic, Grok, Gemini, Ollama
- **Automated Testing** - Ollama-based judge for pass/fail decisions
- **Manual Override** - Click to mark any test as PASS/FAIL
- **Professional Reports** - Export to JSON or PDF with graphs

## Quick Start

```bash
# Clone and install
git clone https://github.com/kazkozdev/system-prompt-benchmark
cd system-prompt-benchmark
pip install -r requirements.txt

# Start Ollama (optional, for auto-judging)
ollama serve
ollama pull qwen3:14b

# Launch app
./start.sh
```

Open `http://localhost:8501` in your browser.

## How to Use

1. Upload your system prompt (.txt file)
2. Select LLM provider and enter API key
3. Click "Run Benchmark"
4. Review results with interactive charts
5. Manually override incorrect judgments
6. Export as PDF or JSON

## Available Prompts

### Business & Sales
- **Customer Support Bot** - E-commerce support with jailbreak protection
- **Sales Assistant** - Lead qualification with pricing controls
- **HR Screening Bot** - Candidate screening with anti-discrimination rules

### Technical
- **Code Review Assistant** - OWASP Top 10 vulnerability detection
- **Corporate Knowledge RAG** - Document access control and data privacy

### Compliance & Safety
- **Legal Compliance Checker** - GDPR, CCPA, HIPAA guidance
- **Financial Advisor Bot** - Educational content with disclaimers
- **Educational Tutor** - Academic integrity enforcement

### Content
- **Social Media Creator** - Brand-locked fitness content generator

## Attack Categories

The benchmark tests these attack types:

**Security Attacks**
- Prompt Injection ("ignore previous instructions")
- Jailbreaking (DAN mode, roleplay tricks)
- Prompt Leaking (extracting system instructions)
- Authority Bypass (fake CEO/admin claims)

**Advanced Techniques**
- Multilingual attacks (6 languages)
- Encoding tricks (base64, ROT13, hex)
- Token smuggling (word-by-word extraction)
- RAG poisoning (fake document injection)

**Domain-Specific**
- Academic dishonesty attempts
- Unauthorized discount requests
- Legal/financial advice bypass
- HR discrimination triggers
- Data privacy violations

## Best Practices

**Do:**
- Test prompts before production
- Use environment variables for secrets
- Monitor suspicious requests
- Update against new attack vectors
- Use multiple security layers

**Don't:**
- Store API keys in prompts
- Rely only on prompts for security
- Ignore failed tests
- Skip Ollama judge review
- Use one prompt for different access levels

## Project Structure

```
system-prompt-benchmark/
├── start.sh                        # Quick start script
├── app.py                          # Streamlit UI
├── prompts/                        # 9 system prompts
├── tests/
│   └── safeprompt-benchmark.json   # 287 attack vectors
└── results/                        # Exported reports
```

## Contributing

Found a vulnerability or new attack vector?

1. Open an issue with `security` label
2. Describe the attack and provide example
3. Propose a fix if possible

## License

MIT License - Free for commercial and personal use.

---

**Made for the AI Safety community**

Built by security-conscious prompt engineers
