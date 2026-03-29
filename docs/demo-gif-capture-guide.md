# Demo GIF Capture Guide

This guide is based on the `video-demo-production` workflow already prepared for this repository.

Use it when you want to record raw material for a short README demo GIF and then hand me the footage plus the minimum text context needed to cut it correctly.

## Goal

Capture one short proof-first flow:

`result -> actions -> result`

The GIF should answer:

1. Is the project real?
2. What does it do?
3. What should the viewer look at first?
4. Why is it worth clicking into?

## Best Flow For This Repo

Use the web UI, not the CLI.

Recommended path:

1. Show a finished benchmark result.
2. Show a few high-value actions.
3. Return to the strongest result frame.

For this project, the strongest actions are:

- `Analyze Prompt`
- `Build Pack`
- `Compare Versions`

Inside the results area, the strongest tabs are:

- `Results`
- `Categories`
- `Review`
- `Export`

Skip the rest unless you are making a longer video.

## What To Record

Record these clips as separate takes if needed. Clean cuts are better than one messy live capture.

### Clip 1: Opening Proof Frame

What to show:

- finished run
- `Overall Score` hero
- pass rate
- review queue

Ideal duration:

- `2-4s`

Why:

- this is the proof frame
- it should be visible in the first second

### Clip 2: Results Detail

What to show:

- `Results` tab
- one readable concrete example from the list

Ideal duration:

- `2-3s`

Why:

- proves the product is not only an aggregate score

### Clip 3: Category Structure

What to show:

- `Categories` tab
- one readable section or table

Ideal duration:

- `1.5-2.5s`

Why:

- shows the benchmark has structured evaluation, not a single opaque metric

### Clip 4: Review Workflow

What to show:

- `Review` tab
- a visible review item or queue state

Ideal duration:

- `1.5-2.5s`

Why:

- shows the product supports action after scoring

### Clip 5: Export Proof

What to show:

- `Export` tab

Ideal duration:

- `1-1.5s`

Why:

- reinforces that results can leave the app

### Clip 6: Product Actions

What to show:

- `Analyze Prompt`
- `Build Pack`
- `Compare Versions`

Ideal duration:

- `1-2s` each

Why:

- these three tabs communicate product depth quickly

### Clip 7: Final Result Frame

What to show:

- back to the strongest finished result screen
- ideally a cleaner or more readable framing than the opening frame

Ideal duration:

- `2-4s`

Why:

- the GIF should close on proof, not on navigation

## Recommended Order

If you want the shortest strong GIF:

1. opening proof frame
2. `Results`
3. `Categories`
4. `Analyze Prompt`
5. `Build Pack`
6. `Compare Versions`
7. final proof frame

If you want slightly more product depth:

1. opening proof frame
2. `Results`
3. `Review`
4. `Export`
5. `Analyze Prompt`
6. `Build Pack`
7. `Compare Versions`
8. final proof frame

## What Not To Record

Do not spend meaningful time on:

- API key entry
- package install
- terminal setup
- long benchmark wait time
- scrolling the README
- admin console
- charts that are too dense to read in GIF size
- logs
- every results tab

If a benchmark run takes too long:

- record the click that starts it
- cut directly to a finished results state

## Recording Rules

- keep the browser tightly cropped to the app
- hide unrelated tabs, desktop clutter, notifications, and dock noise when possible
- move the cursor slowly and deliberately
- avoid tiny text that will disappear in a README GIF
- do not rely on long scrolling
- prefer hard cuts over fancy transitions

## Overlay Text To Use

Use these in order if you want a text-guided GIF:

1. `Red-team your system prompt`
2. `See the score, failures, and review queue`
3. `Pick a prompt, model, and test pack`
4. `Run adversarial scenarios`
5. `Analyze prompts, build packs, compare runs`
6. `Find what broke before deployment`

If the edit gets too busy, use this shorter set:

- `Audit prompt security`
- `Inspect failures`
- `Compare changes`
- `Harden before deployment`

## What Text To Send Me

After you capture the clips, send me one short block in this format:

```text
Goal:
Short README demo GIF for system-prompt-benchmark.

Asset paths:
- /absolute/path/to/clip-01.mp4
- /absolute/path/to/clip-02.mp4
- /absolute/path/to/clip-03.mp4

Use this order:
- opening proof frame
- results detail
- categories
- analyze prompt
- build pack
- compare versions
- final proof frame

Overlay set:
- Red-team your system prompt
- See the score, failures, and review queue
- Analyze prompts, build packs, compare runs
- Find what broke before deployment

Cut notes:
- cut out waiting time
- keep only readable tables
- end on overall score
```

## Minimum Text I Need From You

If you want the shortest possible handoff, send me:

1. the clip paths
2. the order you want
3. whether you want overlays or no overlays
4. which frame should be the final hold

Example:

```text
Use these clips:
- /Users/me/Desktop/opening.mp4
- /Users/me/Desktop/results.mp4
- /Users/me/Desktop/actions.mp4
- /Users/me/Desktop/final.mp4

Order:
opening -> results -> actions -> final

Use overlays:
yes

Final hold:
overall score hero
```

## Handoff Rule

If you are unsure what to record, do not try to show everything.

Capture:

- one strong result
- one concrete example
- two or three meaningful actions
- one final result

That is enough for a strong first demo GIF.
