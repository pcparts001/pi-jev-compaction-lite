# pi-jev-compaction-lite

**Jev decision-based compaction** for Pi Agent.

Inspired by [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction)
for Claude Code.

> **⚠️ Experimental software.** This extension is in an experimental stage and is provided "as is", without warranty of any kind. **The author assumes no responsibility whatsoever** for any damage or loss arising from its use.

## Install

### Option 1: `pi install` (recommended)

```sh
pi install ssh://git@github.com/pcparts001/pi-jev-compaction-lite
```

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

## About this extension

### The problem with default compaction

Pi's default context compaction asks the model you are using to write an LLM summary
of the conversation so far. That approach has two problems:

**1. Most of the information is thrown away — so you pay for it twice.**
The summary collapses the whole session into a few thousand tokens. File paths,
error messages, stack traces and constraints do not survive, so the agent hits the
same dead ends again and **re-runs tool calls** it has already run.

**2. Summarization itself is not free.**
The summarizer is your own LLM, and it has to read the entire conversation to write
the summary. On a large session that is a full extra pass over the context — paid at
input-token rates — every time compaction fires.

### How Jev compaction addresses this

This extension replaces the LLM summary with a **Jev decision pass**. Jev is not
asked to write anything; it is asked, for each tool call/result pair, a simple
binary question: *"does this still matter?"* Everything not judged stale stays in
the context **word for word**.

- **Fewer re-runs** — paths, errors and constraint text are kept verbatim, so the
  agent does not lose the facts it already gathered.
- **Cheaper** — Jev answers per-item questions with a tiny fixed output; it never
  reads the conversation to write prose.

### What the port adapts for pi

The decision algorithm itself carries over from [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction):
two binary `noul` questions per tool call/result pair (`keep_call` /
`keep_result`), keep / truncate-to-head / drop decisions, verbatim text.
Everything around that core is designed for pi's compaction model:

1. **pi owns the trigger.** The extension cannot decide when to compact. pi
   fires `session_before_compact` on its own schedule — when
   `contextTokens > contextWindow − reserveTokens` (default 16,384), i.e. near
   the ceiling (~984k of 1M), not at 60%. The port obeys that timing and adds
   manual `/jev-compact` (with a 20% floor) plus quarantined `/jev-dev-*`
   commands for testing.
2. **The per-pass output cap is tuned from pi-side measurements rather than
   inherited.** Each pass is capped at 5% of the context window (5 / 8 / 10%
   selectable); overflow is trimmed 60% head / 40% tail, since errors and final
   states sit at the end. That is aggressive on purpose, because the benchmark
   below shows size is not what Jev lacks: forced down to 5,000 or 10,000
   tokens, Jev's fact retention stayed at 72–91% while default compaction
   managed only 34–45%. The cap trims redundancy, not the facts that scored —
   and every token it saves is a token the model re-reads for the rest of the
   session.
3. **pi's context brings its own mix of categories.** A pi session is not
   just user/assistant text with tool uses: assistant messages carry `thinking`
   blocks and structured `toolCall` arguments, tool results arrive as separate
   `toolResult` messages (paired by id; orphans are kept as text for safety),
   user messages may carry image blocks (rendered as `[image omitted]`, base64
   never sent), and already-compacted sessions contribute `compactionSummary` /
   `branchSummary` entries that still count toward pi's gate. The port maps all
   of this into its segment model and tracks a per-category composition
   (tool_result / thinking / toolcall_args / asst_text / user_text, before vs
   after). That instrumentation exposed the port's core tension: on a real
   ~400k-token session, thinking + tool-call arguments alone were ~65% of the
   context — categories absent from the original's model — which drove
   the thinking-tail stages in the state-fitting ladder and the optional
   Jev-judged thinking/args experiments.

What carried over unchanged: the two-questions-per-pair decision format (the
wording of the questions themselves was rewritten for pi — see "How it works"),
the 0.5 `keepThreshold`, head-only truncation of dropped results, the 25%
minimum-reduction fallback (the original's own rule, adopted as
`JEV_MIN_REDUCTION`), and the end-pinning of the first message plus the newest
6 messages. Pi's default summary also appends a list of files
read/written/edited (pi collects it itself and shows it next to the summary);
this port omits that list, because the benchmarked configuration it reproduces
does not use it — and pass 1 is exactly that benchmarked algorithm.

### Measured results

Real coding sessions. Costs below use the measured pricing of the models named
in the tables; session names are anonymized.

#### What a compaction costs

A compaction is an ordinary LLM request: the model reads the conversation and
writes the summary. On a session near the context ceiling that means **reading
the whole context just to produce a summary**.

Measured on a ~984k-token session (a real compaction point):

| Method | Model | Reads | Writes | Cost per compaction |
|---|---|---|---|---|
| Default compaction | model-a — input $0.15/M, output $0.50/M | the whole conversation (~984k tok) | a prose summary (~3.7k tok) | **~$0.12** |
| Jev compaction | model-b — input $0.042/M, output $0.00/M | a <=32k tok state, repeated per batch (~190–260k tok total) | short per-item answers ($0.00/M) | **~$0.008–0.011** |

That is **roughly 12x cheaper per compaction**, for two structural reasons:

- Jev never reads the full conversation. It reads a compressed state (<=32k
  tokens) that fits in a single request, repeated a few times in small batches.
- Its output is a list of per-item probabilities, billed at $0.00/M — not prose
  at full output rates.

(Default cost varies with prompt-cache hits; the figure above is measured.)

#### How much survives compaction

Compaction is only useful if it keeps the *right* things. To measure that, the
summary produced by both methods was forced down to the same size — first
5,000 tokens, then 10,000 tokens — and each result was scored on how much of the
sessions' facts still survived. Because the summary size is identical, the only
remaining difference is *which* information each method chose to keep. Each
column averages a group of anonymized coding sessions: **group-a = 5 sessions,
group-b = 10 sessions**.

| Method | group-a, capped at 5,000 tokens | group-a, capped at 10,000 tokens | group-b, capped at 5,000 tokens | group-b, capped at 10,000 tokens |
|---|---|---|---|---|
| Default compaction | 34% | 45% | 42% | 38% |
| **Jev compaction** | **79%** | **91%** | **72%** | **78%** |

At the same summary size, Jev kept **1.7–2.3× more of the sessions' facts**.

**Sessions at pi's own compaction trigger point** (the token counts are pi's own
estimate of the context it fed into compaction, recorded at the trigger point):

| Session | Tokens compacted | Default summary tokens | Facts kept (default) | Facts kept (Jev) |
|---|---|---|---|---|
| session-a | 983,868 | 3,704 | **2 / 14** | **14 / 14** |
| session-b | 983,756 | 2,798 | **1 / 14** | **6 / 14** |
| session-c | 262,144 | 2,658 | 5 / 14 | **14 / 14** |

Default compaction always produces a 2,700–3,700 token summary no matter how large
the session is, so the bigger the session, the more it loses — down to
**1–5 of 14 facts (7–36%)**.

The Jev figures are **uncapped**: with no size limit this configuration keeps
most tool calls verbatim, and on these three sessions its summaries came to
~175k, ~213k and ~53k tokens respectively. That is exactly what the per-pass
budget (`/jev-compact-threshold`, default 5% of the context window) bounds — and
the equal-size benchmark above shows retention stays at 72–91% even when forced
down to 5,000–10,000 tokens.

> These numbers come from a small set of coding sessions and one model pair.
> Your results will differ depending on what your project is like — how much of a
> session is tool output, how repetitive it is, which models you use, and your
> prompt-cache hit rate. Treat them as an illustration of the mechanism, not as a
> guarantee for your workload.
>
> Note: the fact-retention counts in both tables come from an earlier, looser
> probe scorer (14 probes per session, scored with substring matching that
> slightly over-counted correct answers). A stricter scorer (17–22 sanitized
> probes per session) has since replaced it; these tables have not been
> re-measured with it yet, so treat the exact counts as approximate.

## Requirements

### API key (required)

```sh
export OPENROUTER_JEV_API_KEY="sk-or-..."
```

**If it is unset, the extension does
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
