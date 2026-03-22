# system-prompt-benchmark: Competitive Analysis and Roadmap

## Purpose

This document translates the competitive gap analysis against Augustus, Promptfoo, Spikee, and open prompt-injection benchmark research into an implementation roadmap for `system-prompt-benchmark`.

It is intended to answer four questions:

1. Where the project is strong today
2. Where it is behind competitors
3. What should be built first
4. How to turn the work into actionable epics and issues

## Current Positioning

`system-prompt-benchmark` is strongest as a lightweight visual workbench for hardening system prompts:

- low-friction Streamlit UI
- built-in example prompts
- built-in attack corpus
- local judging via Ollama
- side-by-side run comparison
- exportable reports

Compared with competitors, the project is currently weaker in:

- production automation
- extensibility
- provider breadth
- benchmark rigor
- agent and RAG depth
- custom datasets
- multi-detector scoring

## Competitor Snapshot

### Augustus

Relative advantages over this project:

- broader provider support
- larger detector surface
- stronger security-scanner posture
- attack transformations ("buffs")
- production execution features such as concurrency, rate limiting, and timeouts
- custom endpoint support

### Promptfoo

Relative advantages over this project:

- mature CLI-first workflow
- YAML configuration
- CI/CD integration
- flexible eval definitions
- red teaming workflow
- broader output and automation support

### Spikee

Relative advantages over this project:

- stronger offensive-security orientation
- attack strategy generation
- custom dataset generation
- guardrail and application-level testing
- integration into security tooling workflows

### Research Benchmarks

Relative advantages over this project:

- formal metrics
- reproducible methodology
- stronger taxonomy and benchmark definitions
- clearer separation between attack generation, execution, and evaluation
- broader agent, RAG, and multimodal coverage

## Strategic Direction

The project should not try to copy every competitor equally. The best path is:

1. Keep the fast visual UX as a differentiator.
2. Add a serious automation and scoring core underneath it.
3. Expand from prompt-only hardening toward prompt, RAG, and agent evaluation.

That yields a clear product identity:

> A prompt and injection resilience benchmark platform with a simple UI, a scriptable CLI, and reproducible security-focused evaluation.

## Gap List

### 1. Benchmark Corpus

Needed improvements:

- expand beyond mostly prompt-leaking and jailbreak scenarios
- add adaptive multi-turn attacks
- add indirect injection through HTML, markdown, links, metadata, and retrieved documents
- add agent-specific attacks such as tool hijacking and memory contamination
- add richer RAG poisoning scenarios
- add multimodal scenarios for file and image-bearing systems
- add benign control samples
- add mutation and fuzzing transforms
- add attack metadata: source, severity, modality, target surface, expected impact

### 2. Scoring and Evaluation

Needed improvements:

- support more than one judge backend
- separate detection from grading
- add rule-based and custom evaluators
- add user-defined success criteria
- support disagreement and review-needed states
- preserve judge evidence and rationale
- reduce dependence on a single local model

### 3. Metrics

Needed improvements:

- Attack Success Rate
- precision, recall, F1
- false positive rate on benign controls
- utility retention
- category-level weighted scores
- regression deltas between runs
- cost and reliability metrics

### 4. Execution Layer

Needed improvements:

- full provider abstraction
- custom REST and OpenAI-compatible endpoints
- concurrency controls
- retries, backoff, and timeout handling
- batch execution
- normalized request and response metadata
- reproducibility controls

### 5. Automation

Needed improvements:

- stable CLI
- config-driven runs
- machine-readable outputs for CI
- non-zero exit codes for failed gates
- run filters and subsets
- compare and trend commands
- scheduled benchmark modes

### 6. Datasets and Extensibility

Needed improvements:

- import custom datasets
- export reusable packs
- dataset schema validation
- remote pack updates
- private enterprise packs
- plugin API for providers, judges, exporters, and generators

### 7. Product and Team Workflow

Needed improvements:

- manual review triage workflow
- richer run diffs
- collaboration and annotation
- remediation guidance
- Git-aware prompt versioning
- reproducible reports for audit trails

## Prioritized Roadmap

## Phase 1: Foundation for Real Use

Objective: make the tool reliable enough for repeatable local and CI use.

### Epic 1.1: Production-Grade CLI

Goals:

- promote the current ad hoc CLI into a supported interface
- support all available providers
- support config-file driven execution

Deliverables:

- `spb run -c benchmark.yaml`
- `spb compare run-a.json run-b.json`
- provider/model/test selection flags
- deterministic output artifacts
- CI-friendly exit codes

Success criteria:

- headless benchmark runs work without Streamlit
- one config file can reproduce a run
- failed gates stop CI

### Epic 1.2: Run Reliability

Goals:

- make long runs safe and predictable

Deliverables:

- timeout support
- retries with backoff
- concurrency control
- rate limiting
- structured error classes
- resumable interrupted runs

Success criteria:

- long runs fail gracefully instead of crashing
- provider errors are distinguished from benchmark failures

### Epic 1.3: Documentation Truthfulness

Goals:

- align docs with actual repository behavior

Deliverables:

- README update with actual test corpus and supported flows
- architecture overview
- CLI quickstart
- scoring limitations section

Success criteria:

- docs match the real code paths and supported datasets

## Phase 2: Evaluation Quality

Objective: make scores more trustworthy and more useful than a single pass/fail number.

### Epic 2.1: Multi-Judge Evaluation

Goals:

- support multiple judge strategies

Deliverables:

- local LLM judge
- cloud LLM judge
- regex/rule evaluators
- custom Python evaluator hooks
- ensemble scoring mode

Success criteria:

- users can run without Ollama-only coupling
- users can inspect why a result failed

### Epic 2.2: Formal Metrics Layer

Goals:

- turn benchmark output into defensible metrics

Deliverables:

- ASR
- precision / recall / F1
- false positive rate
- utility retention
- category weighted score
- cost and latency summaries

Success criteria:

- every report clearly separates security, utility, and reliability

### Epic 2.3: Manual Review Workflow

Goals:

- make uncertain results operable

Deliverables:

- `PASS`, `FAIL`, `REVIEW`, `WAIVED`
- reviewer notes
- judge evidence view
- unresolved-result counters

Success criteria:

- ambiguous tests stop being silently flattened into a brittle final score

## Phase 3: Dataset and Coverage Expansion

Objective: close the biggest competitive gaps in attack breadth.

### Epic 3.1: Attack Corpus vNext

Goals:

- broaden scenario coverage

Deliverables:

- adaptive multi-turn attacks
- indirect prompt injection cases
- expanded RAG poisoning
- encoded and transformed attacks
- benign control corpus

Success criteria:

- corpus covers prompt-only, RAG, and agent-adjacent threats

### Epic 3.2: Dataset Builder

Goals:

- let users extend the benchmark

Deliverables:

- CSV/JSON/JSONL import
- schema validation
- metadata editor
- tag/category assignment
- pack export

Success criteria:

- teams can create domain-specific packs without editing code

### Epic 3.3: Attack Mutations

Goals:

- emulate evolving adversaries

Deliverables:

- paraphrase transforms
- translation transforms
- formatting/encoding transforms
- typo/noise transforms
- batch mutation generation

Success criteria:

- each base attack can generate multiple realistic variants

## Phase 4: Platform Breadth

Objective: make the engine usable across real deployment stacks.

### Epic 4.1: Provider Abstraction v2

Goals:

- support more model infrastructure with less code duplication

Deliverables:

- Azure OpenAI
- Bedrock
- Vertex AI
- Groq
- Cohere
- Together
- OpenRouter
- generic OpenAI-compatible endpoint
- generic REST adapter

Success criteria:

- adding a provider is mostly configuration and adapter work

### Epic 4.2: Capability-Aware Execution

Goals:

- respect model and provider differences

Deliverables:

- capability matrix
- system prompt compatibility handling
- multimodal capability flags
- tool-calling capability flags
- JSON-mode handling

Success criteria:

- runs fail early when a requested benchmark mode is incompatible with a provider

## Phase 5: Product Maturity

Objective: strengthen the workflow around benchmark use, not just execution.

### Epic 5.1: Comparison and Trends

Goals:

- make regressions obvious

Deliverables:

- run diff page
- category delta charts
- prompt version comparison
- provider/model comparison tables
- trend history

Success criteria:

- users can identify exactly what got worse and where

### Epic 5.2: Reporting

Goals:

- support different audiences

Deliverables:

- HTML report
- Markdown report
- JSONL
- CSV
- JUnit XML
- SARIF
- executive summary mode
- engineer deep-dive mode

Success criteria:

- one run can produce both management and engineering artifacts

### Epic 5.3: Remediation Support

Goals:

- reduce the gap between finding failures and fixing prompts

Deliverables:

- failure clustering
- remediation hints by category
- prompt section mapping
- hardened prompt suggestions

Success criteria:

- a failed run leads directly to concrete prompt hardening work

## Phase 6: Agents, RAG, and Enterprise Readiness

Objective: move from prompt hardening to system resilience testing.

### Epic 6.1: RAG Benchmarking

Goals:

- support document-grounded systems as a first-class target

Deliverables:

- malicious document fixtures
- retrieval poisoning scenarios
- citation integrity checks
- access-control leakage tests

Success criteria:

- enterprise knowledge bots become a native use case instead of an approximate one

### Epic 6.2: Agent Benchmarking

Goals:

- support tool-using and multi-step systems

Deliverables:

- tool invocation audit
- unauthorized action detection
- memory contamination tests
- cross-turn persistence tests
- function-call trace capture

Success criteria:

- the benchmark can evaluate more than plain chat responses

### Epic 6.3: Shared Deployment

Goals:

- support team usage safely

Deliverables:

- persistent run storage
- role-aware access
- audit logs
- secret redaction in exports
- Prometheus/OpenTelemetry instrumentation
- Docker deployment

Success criteria:

- teams can run this as an internal service

## Recommended Order

Build in this order:

1. production-grade CLI
2. reliability controls
3. truthful docs
4. multi-judge evaluation
5. formal metrics
6. manual review states
7. attack corpus expansion
8. dataset builder
9. provider abstraction v2
10. comparison and trend workflow
11. reporting expansion
12. RAG and agent benchmarking

## Suggested GitHub Epics and Issues

### Epic A: Core Execution

Issues:

- Replace the current benchmark CLI with a supported `spb` command
- Add YAML config loading for benchmark runs
- Add provider/model/test filters to CLI
- Add non-zero exit codes for configurable thresholds
- Add retries, timeouts, and backoff
- Add resumable run checkpoints

### Epic B: Scoring and Metrics

Issues:

- Define evaluator interface for judges
- Add local judge adapter
- Add cloud judge adapter
- Add rule-based evaluator
- Add ensemble evaluator mode
- Implement ASR, precision, recall, F1, FPR
- Split security and utility scoring in reports

### Epic C: Dataset System

Issues:

- Define test schema with metadata fields
- Add dataset validator
- Add CSV/JSON/JSONL import
- Add pack export
- Add benign control datasets
- Add transform pipeline for paraphrase/translation/encoding

### Epic D: Provider Platform

Issues:

- Introduce provider capability matrix
- Add generic OpenAI-compatible provider
- Add generic REST provider
- Add Azure OpenAI provider
- Add Bedrock provider
- Add OpenRouter provider

### Epic E: Product Workflow

Issues:

- Add run comparison report
- Add trend history storage
- Add review states and reviewer notes
- Add HTML and Markdown report export
- Add remediation suggestions by category

### Epic F: RAG and Agents

Issues:

- Add malicious document fixture format
- Add retrieval poisoning scenarios
- Add tool-call trace capture
- Add unauthorized action detector
- Add memory contamination scenarios

## What Not To Do First

Avoid starting with:

- localization
- full plugin marketplace
- complex collaboration features
- highly polished UI redesign
- multimodal expansion before the execution and scoring core is stable

These are useful later, but they should not come before reproducibility, automation, and metric quality.

## Definition of Success

The project becomes meaningfully more competitive when it can satisfy all of the following:

- a benchmark can run headlessly in CI with deterministic artifacts
- scoring is not dependent on one local judge model
- users can import custom attack packs without code changes
- reports distinguish security, utility, and reliability
- runs can be compared across prompt versions, models, and time
- the benchmark can evaluate prompt-only systems today and RAG/agent systems next
