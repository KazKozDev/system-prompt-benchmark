# System Prompt Benchmark

A collection of secure system prompts with benchmarking tools to test their safety and effectiveness against prompt injection, jailbreaking, and other attack vectors.

## 📋 Overview

This repository contains production-ready system prompts designed with security best practices and a benchmarking framework to evaluate their robustness.

### Who is this for?

- Prompt Engineers
- AI/ML Developers
- Product Managers working with LLMs
- AI Safety Researchers

## 🎯 Prompts Collection

### 1. Customer Support Bot
**File:** `prompts/customer-support-bot.txt` 

System prompt for an e-commerce customer support chatbot.

**Key Features:**
- Handles product and order inquiries
- Protection against role-playing attacks
- User data isolation

**Use Cases:**
- E-commerce support
- SaaS technical support
- Information bots

---

### 2. Social Media Content Creator
**File:** `prompts/social-media-content-creator.txt` 

Prompt for generating social media content for a fitness brand.

**Key Features:**
- Brand voice control
- Content safety filters
- Medical claim filtering

**Use Cases:**
- Social media posts
- Content marketing
- Creative teams

---

### 3. Corporate Knowledge Base RAG
**File:** `prompts/corporate-knowledge-rag.txt` 

Prompt for a RAG system with access control for corporate knowledge bases.

**Key Features:**
- Document-level access control
- Data leakage protection
- Source citation

**Use Cases:**
- Internal knowledge bases
- Company documentation
- FAQ systems

---

### 4. Educational Tutor
**File:** `prompts/educational-tutor.txt` 

Prompt for an educational assistant for math and science.

**Key Features:**
- Academic integrity focus
- Guidance over direct answers
- Student level adaptation

**Use Cases:**
- EdTech applications
- Tutoring platforms
- Learning systems

## 🛡️ Security Principles

All prompts follow these security principles:

### Protection Against Prompt Injection
- Clear separation of system instructions and user input
- XML tags for structure
- Explicit instructions to ignore embedded commands

### Protection Against Jailbreaking
- Defined behavioral boundaries
- Role-playing attempt handling
- Refusal to execute prohibited actions

### Protection Against Prompt Leaking
- Prohibition on revealing system instructions
- No sensitive data in prompts
- Secure configuration storage

### Data Protection
- User data isolation
- Role-based access control
- Cross-user query prohibition

## 🧪 Benchmarking

### Running the Benchmark
```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"

# Run benchmark
python run_benchmark.py --prompt prompts/customer-support-bot.txt --provider openai
```

### Benchmark Categories

The benchmark tests four categories:

1. **Security Tests** - Prompt injection, jailbreaking, prompt leaking
2. **Functionality Tests** - Normal queries and expected behavior
3. **Boundary Tests** - Out-of-scope request handling
4. **Edge Cases** - Social engineering, multilingual attacks

### Example Results
```
=== BENCHMARK RESULTS ===

Security: 13/13 (100%) ✅
Functionality: 8/8 (100%) ✅
Boundaries: 7/7 (100%) ✅
Edge Cases: 6/6 (100%) ✅

Overall Score: 34/34 (100%)
Model: GPT-5-1
```

## 📊 Test Results

### TechStore Customer Support Bot

**Tested on:** GPT-5-1  
**Date:** 2025-11-19

| Category | Score | Status |
|----------|-------|--------|
| Security | 13/13 (100%) | ✅ |
| Functionality | 8/8 (100%) | ✅ |
| Boundaries | 7/7 (100%) | ✅ |
| Edge Cases | 6/6 (100%) | ✅ |
| **Overall** | **34/34 (100%)** | ✅ |

## 🚀 Quick Start

### Using a Prompt
```python
# With OpenAI
import openai

with open('prompts/customer-support-bot.txt', 'r') as f:
    system_prompt = f.read()

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "How do I return a product?"}
    ]
)
```
```python
# With Anthropic Claude
import anthropic

with open('prompts/customer-support-bot.txt', 'r') as f:
    system_prompt = f.read()

client = anthropic.Anthropic(api_key="your-key")
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=system_prompt,
    messages=[
        {"role": "user", "content": "How do I return a product?"}
    ]
)
```

## 📁 Repository Structure
```
system-prompt-benchmark/
├── README.md
├── LICENSE
├── prompts/                    # System prompts
│   ├── customer-support-bot.txt
│   ├── social-media-content-creator.txt
│   ├── corporate-knowledge-rag.txt
│   └── educational-tutor.txt
├── tests/                      # Benchmark test datasets
│   └── techstore_benchmark.json
├── run_benchmark.py           # Benchmark script
├── requirements.txt           # Python dependencies
└── results/                   # Benchmark results
    └── techstore_results.json
```

## 🔧 Customization

Each prompt can be adapted to your needs:

1. Replace placeholders with your company/product data
2. Add specific rules for your use case
3. Adjust tone and style for your brand
4. Extend boundaries for additional functionality

## 📖 Best Practices

### ✨ Recommendations

1. **Never store secrets in prompts** - use environment variables
2. **Log suspicious requests** - analyze attack patterns
3. **Update regularly** - new attack vectors emerge constantly
4. **Test on production data** - synthetic tests don't cover everything
5. **Use rate limiting** - protection against automated attacks

### 🚨 What NOT to Do

- ❌ Don't rely only on prompts for security
- ❌ Don't store API keys in system prompts
- ❌ Don't use one prompt for different access levels
- ❌ Don't ignore logs - they show real attack attempts

## 🤝 Contributing

We welcome contributions!

### Adding a New Prompt

1. Create a new `.txt` file with the prompt
2. Follow the structure of existing prompts
3. Test against adversarial examples
4. Update README with description
5. Create a Pull Request

### Reporting Security Issues

If you find a vulnerability in the prompts:
- Create an issue with the `security` label
- Describe the attack vector
- Propose a solution if possible

## 📚 Resources

### Further Reading

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [OpenAI Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)

## 📄 License

MIT License - Free to use in commercial and personal projects.

## ⭐ Support

If this repository was helpful, please give it a star!

Made with ❤️ for the AI Safety community
