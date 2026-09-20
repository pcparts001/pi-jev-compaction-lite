# pi-jev-compaction-lite

**Jev decision-based compaction** for [Pi Agent](https://github.com/earendil-works/pi-coding-agent).

Normal compaction asks an LLM to summarize the old conversation, which is **lossy**
(paths, errors, and constraints disappear). This extension **rewrites nothing**.
It asks Jev (a decision model) to judge each tool call/result and removes only the
ones judged stale, or shortens them to their head. Text (user / assistant / thinking)
is kept **verbatim**.

> **⚠️ Experimental software.** This extension is in an experimental stage and is provided "as is", without warranty of any kind. **The author assumes no responsibility whatsoever** for any damage or loss arising from its use.

```
Intercepts pi's session_before_compact
  └─ jev-compaction.ts
       └─ jev-core.mjs
            ├─ collectSegments : messages -> text / pair segments
            ├─ fitState        : fit into Jev's state budget
            ├─ askJev          : noul questions per pair (batched, parallel)
            └─ returns a "pruned verbatim archive" with keep / truncate / drop applied
```

## Install

### Option 1: `pi install` (recommended)

```sh
pi install ssh://git@github.com/pcparts001/pi-jev-compaction-lite
```

This clones the repository into `~/.pi/agent/git/`, runs `npm install` there,
and registers the package in `settings.json`. From then on pi manages it:
it is discovered automatically on startup and re-cloned if the directory is
ever missing.

> The `npm:pi-jev-compaction-lite` form is not available yet — the package is
> not published to npm. Once the repository is public, the shorter forms
> `git:github.com/pcparts001/pi-jev-compaction-lite` and
> `https://github.com/pcparts001/pi-jev-compaction-lite` also work.

**Uninstall:**

```sh
pi remove ssh://git@github.com/pcparts001/pi-jev-compaction-lite
```

This removes the entry from `settings.json` **and** deletes the cloned
directory, so the extension is gone completely.

### Option 2: manual placement

```sh
git clone https://github.com/pcparts001/pi-jev-compaction-lite.git \
  ~/.pi/agent/extensions/jev-compaction
```

Because `package.json` declares the entry point via `pi.extensions`, **no file
renaming is required**.

## Requirements

### API key (required)

```sh
export OPENROUTER_JEV_API_KEY="sk-or-..."
```

This is the same value as `OPENROUTER_API_KEY`. **If it is unset, the extension does
nothing and pi's default summarization is used instead** (safe by design).

### Jev API endpoint

This extension uses OpenRouter's alpha decisions endpoint.

| Item | Value |
|---|---|
| Endpoint | `https://openrouter.ai/api/alpha/decisions` |
| Model | `~typesafe/jev-latest` |

- Works with a regular OpenRouter API key. Usage is billed to your account.
- The Jev model can be overridden via the `JEV_MODEL` environment variable (see below).
  The endpoint itself is fixed to `https://openrouter.ai/api/alpha/decisions` and cannot be changed.

## Privacy & data flow

This extension sends conversation content (user / assistant / thinking text, tool calls
and results) to OpenRouter's `alpha/decisions` endpoint so that Jev can judge it.

- Images are never sent (base64 is replaced with `[image omitted]`).
- The API key is used only in the `Authorization` header and is never logged.
- No telemetry; the only network destination is the endpoint above.

Do not use this extension on sessions whose content you are not permitted to send to OpenRouter.

## Experimental — no warranty

This extension is **experimental software** and is provided "as is", without warranty of
any kind. **The author assumes no responsibility whatsoever** for any damage or loss
arising from its use (including context loss caused by compaction, billing, or
discontinuation of the alpha endpoint). Use at your own risk.

## Usage

The extension activates automatically after installation. It intercepts the moment
pi decides to compact.

### Automatic triggering

No user action is needed. It fires when pi's own condition is met:

```
contextTokens > contextWindow - reserveTokens   # reserveTokens defaults to 16,384
```

For a 1,000,000-token model this means **983,616 tokens**.

### Commands

| Command | Behavior | Gate |
|---|---|---|
| (automatic) | fires on pi's threshold / overflow | pi itself |
| `/jev-compact` | run compaction manually | does nothing below 20% of `contextWindow` |
| `/jev-compact-threshold` | choose the budget from 5 / 8 / 10 (default **5**) | — |
| `/jev-dev-compact` | **for testing**: force a run, ignoring the gate | none |
| `/jev-dev-threshold` | **for testing**: choose the dev budget | — |
| `/jev-stats` | show the last attempt and the current gate state | — |

- The budget setting is **not persisted** (memory only); it resets when pi restarts.
- A `/jev-dev-*` setting can never leak into automatic runs (guarded by `reason === "manual"`).

## What the budget (`/jev-compact-threshold`) means

The budget `pct%` is the **ceiling for this pass's Jev output** (not for the total):

```
budget = contextWindow * pct%     # default 5%
```

From the second compaction onward, **previous summaries are concatenated verbatim**
at the head of the output, and this pass's output is appended after them. The total is
therefore `prev + this pass` and **accumulates**:

```
pass 1: this pass 5%              -> total 5%
pass 2: prev 5% + this pass 5%    -> total 10%
pass 3: prev 10% + this pass 5%   -> total 15%
```

This is deliberate: past compaction results are never lost. If the accumulation would
exceed pi's re-trigger threshold
(`contextWindow - keepRecentTokens - reserveTokens`), the extension defers to pi's
default summary for safety (preventing an infinite loop).

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_JEV_API_KEY` | — | **required**. API key used for Jev |
| `JEV_MODEL` | `~typesafe/jev-latest` | Jev model to use |
| `JEV_KEEP_THRESHOLD` | `0.5` | keep / drop decision threshold |
| `JEV_MIN_REDUCTION` | `0.25` | below this reduction, fall back to pi's default summary |
| `JEV_KEEP_RECENT_TOKENS` | `20000` | used for the re-trigger ceiling (match pi) |
| `JEV_RESERVE_TOKENS` | `16384` | ditto (match pi) |
| `JEV_DEBUG` | — | set to `1` to log to `/tmp/jev-debug.log` |

> When `JEV_DEBUG=1`, `/tmp/jev-debug.log` receives timestamps, decision/token stats, and
> fragments of server error responses. The API key itself is never written to the log.

## How it works

Jev answers two questions (`noul` format) about each tool call:

| Question | Meaning |
|---|---|
| `keep_call` | "Knowing this tool call was made (with its input), does it still matter for continuing?" |
| `keep_result` | "Does this result contain an error, stack trace, or spec detail whose exact wording is needed and cannot be recovered by re-running?" |

The probabilities and the threshold produce three outcomes:

| Decision | Condition | What remains |
|---|---|---|
| **keep** | result >= threshold | the **full** result |
| **truncate** | call >= threshold > result | only the **head** of the result |
| **drop** | both < threshold | the call header plus a note with the result size |

- Text (user / assistant / thinking) is **never judged**; it is kept verbatim.
- Image blocks become `[image omitted]` (base64 is never sent to Jev).
- The **system prompt is not part of compaction** (pi's design).

## Known limitations

- **Jev's scores cluster tightly** (call ~0.54, result ~0.32). Around a 0.5 threshold,
  most tool results end up classified as `truncate` (head only). Decisions are
  therefore **close to category-uniform rather than individually selective**.
- Sessions with few tool logs do not reach a 25% reduction and **fall back to pi's
  default summary** (this is correct behavior).
- Text messages are never removed, so compression is limited on text-heavy sessions.
- **Uses the alpha endpoint** (see above).
- A saturating accumulation is handed to pi's default summary (only that one pass is compressed).

## License

MIT
