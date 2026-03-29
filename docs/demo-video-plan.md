# Demo Video Plan

This plan follows a proof-first structure for `system-prompt-benchmark` and stays grounded in the current repository:

- README positions the project as automated red-team testing for LLM system prompts across 12 categories.
- The web UI already exposes one clean flow: `Run Benchmark`, `Analyze Prompt`, `Build Pack`, `Compare Versions`.
- The repository already contains a demo still at `assets/demo.png`.
- The CLI and startup commands are real and runnable: `./start.sh` and `python spb.py --help`.

## Strongest Flow To Show

Show the Streamlit web UI, not the CLI.

Why this is the best proof asset:

- it has the clearest visible outcome
- it shows the product as more than a script
- it demonstrates the main promise in one pass: load prompt, run attacks, inspect results
- it lets you briefly surface differentiators without turning the video into a feature dump

Recommended demo scenario:

1. Open the app with a built-in example prompt.
2. Show the benchmark workspace already loaded.
3. Start a quick benchmark or cut directly to finished results if live latency is too slow.
4. Flash the `Analyze Prompt`, `Build Pack`, and `Compare Versions` tabs as proof of depth.
5. Close on the score and failed-example workflow.

## 25-Second Sequence

### 0-3s

Open on the app already working.

Frame target:

- centered title `Audit prompt security`
- sidebar visible
- main workspace visible
- if available, open on a finished results screen rather than an empty state

Voiceover or caption:

`Red-team your system prompt before it ships.`

### 3-6s

Show the left sidebar with:

- built-in example prompt selected
- provider section
- test count section

Move quickly and deliberately. Do not linger on setup.

Caption:

`Pick a prompt, a model, and a test pack.`

### 6-12s

Click `Start Benchmark` or cut from button press to an already-finished run.

What matters on screen:

- the benchmark action is obvious
- the viewer sees this is not a static dashboard

Caption:

`Run adversarial tests across hundreds of attack scenarios.`

### 12-17s

Flash the adjacent tabs in one fast sequence:

- `Analyze Prompt`
- `Build Pack`
- `Compare Versions`

This is enough to prove the product is deeper than a single scorecard.

Caption:

`Analyze prompt structure, build custom packs, compare runs.`

### 17-25s

Close on the results view.

Prioritize these elements:

- overall score hero
- pass rate
- review queue
- category breakdown or charts
- failed example list if readable

Final caption:

`See what broke, where it broke, and what to harden next.`

## Proof Points To Surface

Only surface claims already supported by the repository:

- `12` benchmark categories
- `15+` provider integrations
- three interfaces: CLI, web UI, REST API
- custom dataset pack builder
- run comparison and regression views
- PDF report export
- plugin SDK

Good on-screen text options:

- `12-category system prompt red-team benchmark`
- `CLI, web UI, and API`
- `Custom packs, comparisons, exports`

## Capture Recipe

Use this for the raw recording:

```bash
./start.sh
```

Fallback if you want to verify the CLI surface separately:

```bash
python spb.py --help
```

Recording choices:

- use the existing visual style from `assets/demo.png` as the framing reference
- prefer `Quick (10 tests)` for live capture
- use a built-in example prompt from `prompts/`
- keep the browser window tightly cropped to the app
- hide unrelated tabs, dock noise, and notifications

If live inference is too slow or provider credentials are unavailable:

- record the navigation flow separately
- record a finished-results state separately
- stitch them together with hard cuts
- do not fabricate timings or scores

## What To Cut

Do not include:

- package installation
- long terminal setup
- README scrolling
- raw YAML editing
- admin console
- monitoring stack setup
- plugin internals
- dense detector details

Those are useful for docs, not for the first proof video.

## Recommended Structure For README Placement

Best placement:

1. keep the existing static `Demo` image as poster or fallback
2. add a short MP4 or GIF directly under the `Demo` heading
3. if file size is too large, host the MP4 externally and keep the image as the click target

Preferred asset order:

- lightweight GIF for instant motion in the README
- linked MP4 for readable full-size playback

## Editing Notes

Keep captions short and factual.

Avoid claims like:

- `production-ready for enterprises`
- `used by teams`
- `catches every jailbreak`

Close on visible evidence instead:

- score
- regression view
- concrete failed examples

## One-Line Script

`System Prompt Benchmark red-teams your system prompt across 12 categories, shows what broke, and helps you compare fixes before deployment.`
