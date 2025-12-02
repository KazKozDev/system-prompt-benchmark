<p align="center">
  <img width="350" alt="logo" src="https://github.com/user-attachments/assets/11c6d736-7d23-46d2-a9b9-616396d7e8ed" />
</p>

<h3 align="center">Security testing for production LLM applications</h3>
<p align="center">
  <img src="https://img.shields.io/badge/Attack_Vectors-287-1565C0?style=flat" alt="Vectors">
  <img src="https://img.shields.io/badge/Prompt_Injection-Tested-1976D2?style=flat" alt="Injection">
  <img src="https://img.shields.io/badge/Jailbreak-Detection-2196F3?style=flat" alt="Jailbreak">
  <img src="https://img.shields.io/badge/Data_Leaks-Prevention-42A5F5?style=flat" alt="Leaks">
</p>

Test your LLM system prompts against 287 real-world attack vectors including prompt injection, jailbreaks, and data leaks. Automated security testing for production AI systems.
</p>

## Features

- **9 Production-Ready Prompts** - Customer support, sales, HR, legal, finance, code review, and more
- **287 Attack Vectors** - Covering all 2024-2025 jailbreak techniques
- **5 LLM Providers** - OpenAI, Anthropic, Grok, Gemini, Ollama
- **Prompt Analysis** - AI-powered analysis of your prompt's role, capabilities, and boundaries
- **Automated Testing** - Ollama-based judge for pass/fail decisions
- **Version Comparison** - Track and compare results across multiple test runs
- **Manual Override** - Click to mark any test as PASS/FAIL
- **Professional Reports** - Export to JSON or PDF with graphs

<img width="1286" height="854" alt="Screenshot 2025-11-20 at 16 47 27" src="https://github.com/user-attachments/assets/9de9e8a9-c05d-4cee-963d-990f722f2ffa" />

## Quick Start

```bash
# Clone and install
git clone https://github.com/kazkozdev/system-prompt-benchmark
cd system-prompt-benchmark
pip install -r requirements.txt

# Start Ollama (required for automated scoring)
ollama serve
ollama pull qwen3:14b  # LLM judge for evaluating responses

# Launch app
./start.sh
```

Open `http://localhost:8501` in your browser.

**Note:** Ollama with `qwen3:14b` is used as an LLM judge to automatically score test responses. Without it, you'll need to manually review each result. You can use other Ollama models, but `qwen3:14b` is recommended for accurate scoring.

## How to Use

1. **Upload or select a prompt** - Upload .txt file, paste text, or choose an example
2. **Configure LLM provider** - Select OpenAI, Anthropic, Grok, Gemini, or Ollama
3. **Analyze Prompt** (optional) - Get AI insights on your prompt's structure and security
4. **Run Benchmark** - Test against 287 attack vectors with automated scoring
5. **Compare Versions** - Track improvements across multiple test runs
6. **Export results** - Download as JSON or PDF report

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


## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **GitHub Issues**: [Project Issues](https://github.com/KazKozDev/system-prompt-benchmark/issues)
- **LinkedIn**: [Artem KK](https://www.linkedin.com/in/kazkozdev/)

---

⭐ If you find this project helpful, please give it a star
