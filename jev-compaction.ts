/**
 * jev-compaction.ts — pi extension: Jev decision-based compaction.
 *
 * Port of the fast-jev-compaction idea to pi.
 * Intercepts pi's session_before_compact and, instead of an LLM summary,
 * replaces it with a verbatim archive from which only the tool calls/results Jev judged stale are removed.
 *
 * The algorithm is **identical** to `original-like@p<budget>` in benchmark/p5-sessions.mjs:
 *   compactJevChunked(msgs, original-like options)
 *     → capSummary(summary, contextWindow × pct%)
 *     -> falls back to pi's default summary when reduction < 0.25
 * Note: only the input differs. The benchmark cuts the first 70% artificially,
 *   while production uses pi's own messagesToSummarize + turnPrefixMessages.
 *
 * Note: previousSummary is treated as **verbatim-protected** (accumulating scheme / Plan (c)):
 *   pi passes the previous compaction summary as previousSummary.
 *   It is **never fed to Jev**; it is concatenated verbatim at the head of the output.
 *     final = previousSummary (unmodified) + capSummary(this pass's Jev output, budget)
 *   -> Past compaction results accumulate as-is, and Jev only handles the current delta.
 *   -> The budget cap (/jev-compact-threshold pct%, default 5%) applies **only to the current output**.
 *      So 5% is the ceiling for this pass's summary, not for the total.
 *      (The total is prev + this pass; accumulating is by design.)
 *
 *   pass1 (no previousSummary) yields exactly the same result as before, so the
 *   equivalence with the benchmark `original-like@p<budget>` is preserved.
 *
 *   Never mix previousSummary into Jev's **input** (it eats the state budget and
 *   causes HTTP 400 max_tokens_exceeded on 1M-scale sessions). Only ever
 *   concatenate it at the head of the output.
 *
 * fileOps is **not passed** (the benchmark does not use it).
 *
 * On failure or insufficient reduction, returns undefined to fall back to pi's default summary.
 *
 * Usage:
 *   Place in ~/.pi/agent/extensions/jev-compaction/ (global)
 * Environment variables:
 *   OPENROUTER_JEV_API_KEY (required) / JEV_MODEL / JEV_KEEP_THRESHOLD
 *   JEV_MIN_REDUCTION / JEV_DEBUG
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import fs from "node:fs";
import { compactJevChunked } from "./jev-core.mjs";

function dbg(msg: string): void {
	if (!process.env.JEV_DEBUG) return;
	try {
		fs.appendFileSync("/tmp/jev-debug.log", `${new Date().toISOString()} ${msg}\n`);
	} catch {
		// ignore
	}
}

function num(name: string, dflt: number): number {
	const v = process.env[name];
	if (!v) return dflt;
	const n = Number(v);
	return Number.isFinite(n) ? n : dflt;
}

function fmtTokens(n: number): string {
	return n.toLocaleString("en-US");
}

// ---------------------------------------------------------------------------
// Helpers ported **unchanged** from benchmark/p5-sessions.mjs
// (do not modify: changing the formulas breaks comparison with the benchmark measurements)
// ---------------------------------------------------------------------------

const CJK_RE = /[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]/;

/** Estimated tokens for a message list (CJK-aware). Same as p5-sessions.mjs:estTokens. */
function estTokens(messages: any[]): number {
	let cjk = 0;
	let other = 0;
	const scan = (t: string) => {
		for (const ch of t || "") {
			if (CJK_RE.test(ch)) cjk++;
			else other++;
		}
	};
	for (const m of messages) for (const b of m.content || []) {
		if (b.type === "text") scan(b.text);
		else if (b.type === "thinking") scan(b.thinking);
		else if (b.type === "toolCall") scan(b.name + JSON.stringify(b.arguments || {}));
	}
	return Math.ceil(cjk * 1.2 + other / 3.5);
}

/**
 * Hard cap on the summary. Same as p5-sessions.mjs:capSummary.
 * Excess is dropped as head 60% + tail 40% (preserving the tail-priority finding).
 */
function capSummary(text: string, budgetTok: number): { text: string; capped: boolean; before: number; after?: number } {
	if (!budgetTok || budgetTok <= 0 || !text) {
		return { text, capped: false, before: estTokens([{ content: [{ type: "text", text }] }]) };
	}
	const before = estTokens([{ content: [{ type: "text", text }] }]);
	if (before <= budgetTok) return { text, capped: false, before };
	const chars = text.length;
	const targetChars = Math.max(200, Math.floor((chars * budgetTok) / before));
	const head = Math.floor(targetChars * 0.6);
	const tail = targetChars - head;
	const trimmed = text.slice(0, head) + `\n\n[... BUDGET CAP: ${chars - targetChars} chars trimmed ...]\n\n` + text.slice(-tail);
	return { text: trimmed, capped: true, before, after: estTokens([{ content: [{ type: "text", text: trimmed }] }]) };
}

// ---------------------------------------------------------------------------
// Same estimation as pi itself (used to decide whether compaction is possible)
//
// pi's compaction counts session message content as char/4 and only compresses the
// part beyond keepRecentTokens (default 20,000).
// The system prompt and tool definitions are **not included** (unlike the API usage display).
// If this estimate does not match pi's, we misjudge "it should be compactable but is not",
// so estimateTokens / estimateTextAndImageContentChars are copied from pi's
// implementation (dist/core/compaction/compaction.js).
// ---------------------------------------------------------------------------

/** Same estimated characters per image as pi. */
const ESTIMATED_IMAGE_CHARS = 4800;

/** Character count of text/image blocks (same as pi:estimateTextAndImageContentChars). */
function textAndImageChars(content: any): number {
	if (typeof content === "string") return content.length;
	let chars = 0;
	for (const block of content || []) {
		if (block.type === "text" && block.text) chars += block.text.length;
		else if (block.type === "image") chars += ESTIMATED_IMAGE_CHARS;
	}
	return chars;
}

/** Estimated tokens for one message (same as pi:estimateTokens; char/4). */
function piEstimateTokens(message: any): number {
	let chars = 0;
	switch (message?.role) {
		case "user":
			return Math.ceil(textAndImageChars(message.content) / 4);
		case "assistant": {
			for (const block of message.content || []) {
				if (block.type === "text") chars += (block.text || "").length;
				else if (block.type === "thinking") chars += (block.thinking || "").length;
				else if (block.type === "toolCall") chars += (block.name || "").length + JSON.stringify(block.arguments || {}).length;
			}
			return Math.ceil(chars / 4);
		}
		case "custom":
		case "toolResult":
			return Math.ceil(textAndImageChars(message.content) / 4);
		case "bashExecution":
			return Math.ceil(((message.command || "").length + (message.output || "").length) / 4);
		// The previous compaction summary is part of the context (added, same as pi).
		// Missing this causes underestimation on already-compacted sessions.
		case "branchSummary":
		case "compactionSummary":
			return Math.ceil(((message.summary || "").length) / 4);
		default:
			return 0;
	}
}

/**
 * Session entry -> context message (equivalent to pi:sessionEntryToContextMessages).
 * A compaction line becomes a compactionSummary message (its summary joins the context).
 * Missing this underestimates already-compacted sessions.
 */
function entryToContextMessages(entry: any): any[] {
	switch (entry?.type) {
		case "message":
			return [entry.message];
		case "compaction":
			return [{ role: "compactionSummary", summary: entry.summary || "" }];
		case "branch_summary":
			return entry.summary ? [{ role: "branchSummary", summary: entry.summary }] : [];
		case "custom_message":
			return [{ role: "custom", content: entry.content ?? [] }];
		default:
			return [];
	}
}

/**
 * Returns the compaction gate state (values estimated to match pi's behavior).
 *
 * pi compresses only the older part, excluding keepRecentTokens (default 20,000).
 * So **compaction is impossible unless the message volume exceeds keepRecentTokens**.
 */
function compactionGate(ctx: any): {
	messageTokens: number;
	keepRecentTokens: number;
	canCompact: boolean;
	reason: string;
	messageCount: number;
} {
	const keepRecentTokens = Number(process.env.JEV_KEEP_RECENT_TOKENS || 20000);
	let messageTokens = 0;
	let messageCount = 0;
	let lastType = "";
	try {
		// buildContextEntries matches the "actual context" after compaction.
		// Fall back to getBranch() when unavailable.
		const entries =
			typeof ctx.sessionManager?.buildContextEntries === "function"
				? ctx.sessionManager.buildContextEntries()
				: ctx.sessionManager.getBranch();
		for (const e of entries) {
			for (const m of entryToContextMessages(e)) {
				messageTokens += piEstimateTokens(m);
				messageCount++;
			}
			lastType = e.type;
		}
	} catch (err) {
		dbg(`compactionGate failed: ${err instanceof Error ? err.message : err}`);
	}
	if (lastType === "compaction") {
		return { messageTokens, keepRecentTokens, canCompact: false, reason: "immediately after compaction (last entry is a compaction line)", messageCount };
	}
	if (messageTokens < keepRecentTokens) {
		return {
			messageTokens,
			keepRecentTokens,
			canCompact: false,
			reason: `message content is below keepRecentTokens (short by ${fmtTokens(keepRecentTokens - messageTokens)} tok)`,
			messageCount,
		};
	}
	return { messageTokens, keepRecentTokens, canCompact: true, reason: "compactable", messageCount };
}

// ---------------------------------------------------------------------------
// Settings (not persisted; reset to defaults when pi restarts)
// ---------------------------------------------------------------------------

/** Budget % shared by automatic (threshold/overflow) and /jev-compact */
const DEFAULT_PCT = 5;
/** /jev-compact gate: do nothing when the context is below this % of contextWindow */
const COMPACT_GATE_PCT = 20;
const THRESHOLD_CHOICES = [5, 8, 10];
const DEV_THRESHOLD_CHOICES = [2, 5, 8, 10, 20];

interface JevStats {
	model: string;
	stage: number;
	stateTokens: number;
	requests: number;
	batches: number;
	decisions: { keep: number; truncate: number; drop: number; noresult: number };
	chars: { before: number; after: number };
	estTokens: { summarizedBefore: number; summary: number };
	reduction: number;
	jevUsage: { requests: number; input: number; output: number; cost: number };
	elapsedMs: number;
}

export default function (pi: ExtensionAPI) {
	/**
	 * Result of the last compaction attempt. Source for /jev-stats.
	 * Always record "ok", "fallback", and "error".
	 * It used to update only on success, so after a fallback the previous success
	 * was still shown, wrongly suggesting the current compaction had succeeded.
	 */
	type LastAttempt =
		| {
				status: "ok";
				stats: JevStats;
				summaryTokens: number;
				capped: boolean;
				reduction: number;
				budgetPct: number;
				reason: string;
				/** Tokens of this pass's Jev output (after capping) */
				newTokens: number;
				/** Tokens of the inherited previousSummary (0 = first pass) */
				prevTokens: number;
				/** prev + new total (what actually enters the context) */
				totalTokens: number;
				/** Whether protecting prev pushed the total near the threshold and risks re-triggering */
				prevProtected: boolean;
		  }
		| { status: "fallback"; reduction: number; threshold: number; reason: string; detail?: string }
		| { status: "error"; message: string; reason: string };
	let lastAttempt: LastAttempt | null = null;
	let pct = DEFAULT_PCT;
	let devPct = DEFAULT_PCT;
	/**
	 * /jev-dev-compact calls ctx.compact(), ignoring the gate.
	 * pi only reports "manual" as the reason for manual compaction, so a
	 * "use dev settings for the next run only" flag distinguishes them.
	 * It is consumed (cleared) at the top of the handler to avoid mix-ups.
	 */
	let devOneShot = false;

	pi.on("session_before_compact", async (event, ctx) => {
		const apiKey = process.env.OPENROUTER_JEV_API_KEY;
		const { preparation, signal } = event;
		// Previous compaction summary (passed in by pi).
		// It is **never fed to Jev**; it is concatenated verbatim at the head to protect it (accumulating scheme).
		// Mixing it into Jev's input eats the state budget and causes HTTP 400 on 1M-scale sessions.
		const { messagesToSummarize, turnPrefixMessages, tokensBefore, firstKeptEntryId } = preparation;
		const previousSummaryRaw =
			typeof (preparation as { previousSummary?: unknown }).previousSummary === "string"
				? ((preparation as { previousSummary?: string }).previousSummary as string)
				: "";
		const previousSummary = previousSummaryRaw.trim() ? previousSummaryRaw : "";
		const prevTokens = previousSummary
			? estTokens([{ content: [{ type: "text", text: previousSummary }] }])
			: 0;

		// The dev flag is valid only for this one manual run.
		// Checking reason too guarantees it never leaks into automatic (threshold/overflow) runs
		// (/jev-dev-compact only calls ctx.compact(), so leaving the flag set
		//   removes any chance of an automatic compaction slipping in).
		const isDev = devOneShot && event.reason === "manual";
		devOneShot = false;
		const budgetPct = isDev ? devPct : pct;

		if (!apiKey) {
			ctx.ui.notify("jev-compaction: OPENROUTER_JEV_API_KEY not set, falling back to default compaction", "warning");
			return;
		}
		// Production compresses the range pi chose (it does not mimic the benchmark's artificial cut)
		const messages = [...messagesToSummarize, ...turnPrefixMessages];
		if (messages.length === 0) return; // nothing to do → default

		ctx.ui.notify(
			`jev-compaction: scoring ${messages.length} messages (${fmtTokens(tokensBefore)} ctx tokens, budget ${budgetPct}%${isDev ? " [dev]" : ""}) with Jev…`,
			"info",
		);

		try {
			// --- Options identical to the benchmark's original-like@p<budget>.
			// To map 1:1 onto the verified conditions, **never pass arguments the benchmark does not pass**.
			// previousSummary is not passed here either (mixing it into Jev's input eats the state budget
			// and causes HTTP 400 max_tokens_exceeded on 1M-scale sessions). Below, previousSummary
			// is concatenated verbatim at the head of the output to protect it.
			const { summary, stats } = await compactJevChunked(messages, {
				apiKey,
				model: process.env.JEV_MODEL || undefined,
				keepThreshold: num("JEV_KEEP_THRESHOLD", 0.5),
				askThinking: false,
				askArgs: false,
				truncateHeadChars: 300,
				truncateHeadRatio: 1,
				preserveRecentMessages: 6,
				disableBudgetSelection: true,
				maxSummaryTokens: 0,
				thinkingTailChars: 300,
				argsKeepChars: 150,
				summaryBudgetRatio: 0.15,
				// maxStateTokens / maxRequestTokens / safetyRatio / chunkTokens are
				// not passed, same as the benchmark (core defaults = state 32000 / request 65536 / safety 0.85)
				signal,
			});

			// summary is typed string | null. It cannot actually be null since chunks.length >= 1, but guard anyway.
			const summaryText = summary ?? "";
			if (!summaryText) return; // nothing survived -> pi default

			// --- Same post-processing as p5: budget cap -> reduction check ---
			//
			// The budget cap applies **only to this pass's Jev output** (interpretation B).
			// previousSummary is verbatim-protected outside it, so
			// pct% (default 5%) is the ceiling for this pass's summary, not for the total.
			// (The total is prev + this pass; accumulating is by design.)
			// pass1 (no prev) gives exactly the same result as before.
			const ctxWindow = ctx.model?.contextWindow ?? 0;
			const budget = Math.round((ctxWindow * budgetPct) / 100);
			if (budget <= 0) {
				ctx.ui.notify("jev-compaction: cannot read contextWindow, so the budget cap is not applied", "warning");
			}
			const capped = capSummary(summaryText, budget);
			const summaryTokens = capped.capped && capped.after != null ? capped.after : capped.before;
			// The reduction is this pass's Jev output over this pass's input. prev goes in neither numerator nor denominator.
			// (Putting prev in the numerator raises the apparent reduction from pass2 on,
			//   letting it slip past the gate, so it is excluded deliberately.)
			const reduction = 1 - summaryTokens / Math.max(1, estTokens(messages));

			const minReduction = num("JEV_MIN_REDUCTION", 0.25);
			if (reduction < minReduction) {
				lastAttempt = { status: "fallback", reduction, threshold: minReduction, reason: event.reason };
				ctx.ui.notify(
					`jev-compaction: reduction ${(reduction * 100).toFixed(0)}% < ${minReduction * 100}%, falling back to default summary`,
					"warning",
				);
				return;
			}

			// --- Concatenate previousSummary verbatim at the head (accumulating scheme / Plan (c)) ---
			//
			// If the concatenated total approaches pi's re-trigger threshold, compaction runs again
			// right after, and prev gets rewritten by pi's default LLM summary. To avoid that,
			// if the "no re-trigger next time" ceiling is exceeded we give up and defer to pi's default.
			// (As designed, only the final saturating pass breaks.)
			const keepRecentTokens = num("JEV_KEEP_RECENT_TOKENS", 20000);
			// pi's default reserveTokens is 16384 (settings-manager.js).
			// Re-triggering actually happens when contextTokens > contextWindow - reserveTokens.
			const reserveTokens = num("JEV_RESERVE_TOKENS", 16384);
			const ceiling = ctxWindow > 0 ? ctxWindow - keepRecentTokens - reserveTokens : Number.POSITIVE_INFINITY;
			const totalTokens = prevTokens + summaryTokens;
			if (prevTokens > 0 && totalTokens > ceiling) {
				lastAttempt = {
					status: "fallback",
					reduction,
					threshold: minReduction,
					reason: event.reason,
					detail:
						`Cumulative summary ${fmtTokens(totalTokens)} tok (prev ${fmtTokens(prevTokens)} + new ${fmtTokens(summaryTokens)}) ` +
						`exceeds the re-trigger ceiling ${fmtTokens(ceiling)} tok; deferring to pi's default summary`,
				};
				dbg(`jev compaction FALLBACK (ceiling): total=${totalTokens} prev=${prevTokens} ceiling=${ceiling}`);
				ctx.ui.notify(
					`jev-compaction: cumulative ${fmtTokens(totalTokens)} tok exceeds the re-trigger ceiling ${fmtTokens(ceiling)} tok, ` +
						`so pi's default summary will merge the accumulation (prev ${fmtTokens(prevTokens)} + new ${fmtTokens(summaryTokens)})`,
					"warning",
				);
				return;
			}
			const finalText = prevTokens > 0 ? `${previousSummary}\n\n${capped.text}` : capped.text;

			lastAttempt = {
				status: "ok",
				stats,
				summaryTokens,
				capped: capped.capped,
				reduction,
				budgetPct,
				reason: event.reason,
				newTokens: summaryTokens,
				prevTokens,
				totalTokens,
				prevProtected: prevTokens > 0,
			};
			dbg(
				`jev compaction OK: keep=${stats.decisions.keep} trunc=${stats.decisions.truncate} drop=${stats.decisions.drop} ` +
					`reduction=${reduction} capped=${capped.capped} budget=${budget} new=${summaryTokens} prev=${prevTokens} total=${totalTokens} reason=${event.reason}`,
			);
			ctx.ui.notify(
				`jev-compaction: kept ${stats.decisions.keep} / truncated ${stats.decisions.truncate} / dropped ${stats.decisions.drop} tool calls in ${stats.elapsedMs}ms ($${stats.jevUsage.cost.toFixed(6)})` +
					` | new ${fmtTokens(summaryTokens)} tok${capped.capped ? ` (capped to ${budgetPct}% = ${fmtTokens(budget)})` : ""}` +
					(prevTokens > 0
						? ` + prev ${fmtTokens(prevTokens)} tok (verbatim protected) = total ${fmtTokens(totalTokens)} tok`
						: `, no previous summary`),
				"info",
			);

			return {
				compaction: {
					// previousSummary is already concatenated verbatim at the head (accumulating scheme)
					summary: finalText,
					firstKeptEntryId,
					tokensBefore,
					// Map OpenRouter usage onto pi's Usage shape (reflected in the session total)
					usage: {
						input: stats.jevUsage.input,
						output: stats.jevUsage.output,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: stats.jevUsage.input + stats.jevUsage.output,
						cost: {
							input: stats.jevUsage.cost,
							output: 0,
							cacheRead: 0,
							cacheWrite: 0,
							total: stats.jevUsage.cost,
						},
					},
					details: {
						jev: stats,
						budgetPct,
						budget,
						capped: capped.capped,
						prevTokens,
						newTokens: summaryTokens,
						totalTokens,
					},
				},
			};
		} catch (err) {
			const msg = err instanceof Error ? err.message : String(err);
			dbg(`jev compaction FAILED: ${msg}`);
			lastAttempt = { status: "error", message: msg, reason: event.reason };
			if (!signal?.aborted) ctx.ui.notify(`jev-compaction failed: ${msg} — falling back to default summary`, "error");
			return;
		}
	});

	// -----------------------------------------------------------------------
	// Commands
	// -----------------------------------------------------------------------

	/** Trigger manual compaction. The gate (20%) is checked by the caller. */
	function triggerCompact(ctx: any, label: string) {
		ctx.ui.notify(`jev-compaction: ${label} — compacting (budget ${devOneShot ? devPct : pct}%)…`, "info");
		ctx.compact({
			onComplete: (r: any) => dbg(`${label} completed: ${r?.tokensBefore ?? "?"}`),
			onError: (err: Error) => {
				dbg(`${label} ERROR: ${err?.message || err}`);
				ctx.ui.notify(`jev-compaction: ${err?.message || err}`, "error");
			},
		});
	}

	/**
	 * Current context usage (same value as pi). Used for the gate check.
	 *
	 * The old implementation ran estTokens over ctx.sessionManager.getBranch() itself, but
	 * getBranch() returns **only messages after the latest compaction**, so it omitted
	 * the history up to then (the compaction summary) and reported far less than actual use
	 * (observed: pi showed 11% ~ 110k tok while it judged 47k tok).
	 * -> Use ctx.getContextUsage(), which pi itself computes.
	 */
	function currentTokens(ctx: any): number | null {
		try {
			const usage = ctx.getContextUsage?.();
			if (usage && typeof usage.tokens === "number") return usage.tokens;
		} catch (err) {
			dbg(`getContextUsage failed: ${err instanceof Error ? err.message : err}`);
		}
		// Fallback: tokens is null right after startup before any LLM response.
		// In that case estimate all session messages (the compaction summary cannot be included).
		try {
			const messages = ctx.sessionManager
				.getBranch()
				.filter((e: any) => e.type === "message")
				.map((e: any) => e.message);
			return estTokens(messages);
		} catch (err) {
			dbg(`currentTokens failed: ${err instanceof Error ? err.message : err}`);
			return null;
		}
	}

	// /jev-compact - manual compaction (does nothing below 20%)
	pi.registerCommand("jev-compact", {
		description: "Run Jev compaction manually (does nothing when the context is below 20% of contextWindow)",
		handler: async (_args, ctx) => {
			devOneShot = false;
			const ctxWindow = ctx.model?.contextWindow ?? 0;
			const gate = Math.round((ctxWindow * COMPACT_GATE_PCT) / 100);
			const tokens = currentTokens(ctx);
			if (tokens == null) {
				ctx.ui.notify("jev-compaction: cannot read context usage, so compaction was aborted", "warning");
				return;
			}
			if (gate > 0 && tokens < gate) {
				ctx.ui.notify(
					`jev-compaction: context ${fmtTokens(tokens)} tok is below the ${COMPACT_GATE_PCT}% threshold (${fmtTokens(gate)} tok), so it will not compact ` +
						`(use /jev-dev-compact to force it)`,
					"info",
				);
				return;
			}
			triggerCompact(ctx, "/jev-compact");
		},
	});

	// /jev-compact-threshold - the normal budget %
	pi.registerCommand("jev-compact-threshold", {
		description: `Choose the Jev compaction target size (default ${DEFAULT_PCT}%; now ${pct}%)`,
		handler: async (_args, ctx) => {
			const choice = await ctx.ui.select(
				`Jev compaction target size (% of the model's maximum context). Current: ${pct}%`,
				THRESHOLD_CHOICES.map((p) => `${p}%`),
			);
			if (!choice) return;
			pct = Number(choice.replace("%", ""));
			ctx.ui.notify(`jev-compaction: target size set to ${pct}% (applies to automatic compaction and /jev-compact)`, "info");
		},
	});

	// /jev-dev-compact - force compaction ignoring the gate (for testing)
	pi.registerCommand("jev-dev-compact", {
		description: "For testing: force Jev compaction even when the context is below the threshold",
		handler: async (_args, ctx) => {
			devOneShot = true;
			const tokens = currentTokens(ctx);
			ctx.ui.notify(`jev-compaction(dev): forcing a run (currently ${tokens != null ? fmtTokens(tokens) : "?"} tok)`, "warning");
			triggerCompact(ctx, "/jev-dev-compact");
		},
	});

	// /jev-dev-threshold - the dev budget %
	pi.registerCommand("jev-dev-threshold", {
		description: `For testing: choose the target size for /jev-dev-compact (now ${devPct}%)`,
		handler: async (_args, ctx) => {
			const choice = await ctx.ui.select(
				`Target size for /jev-dev-compact (% of the model's maximum context). Current: ${devPct}%`,
				DEV_THRESHOLD_CHOICES.map((p) => `${p}%`),
			);
			if (!choice) return;
			devPct = Number(choice.replace("%", ""));
			ctx.ui.notify(`jev-compaction(dev): target size set to ${devPct}% (applies to /jev-dev-compact only)`, "info");
		},
	});

	// /jev-stats - current gate state plus the last Jev compaction attempt
	pi.registerCommand("jev-stats", {
		description: "Show compaction gate status and the last Jev compaction stats",
		handler: async (_args, ctx) => {
			// ui.notify **overwrites the display on every call**, so send one combined message.
			// Assume newlines may be collapsed (e.g. a one-line TUI status) and build a single " | "-joined line.
			const parts = [];

			// Always show the gate state, with or without lastAttempt
			const g = compactionGate(ctx);
			const ctxWindow = ctx.model?.contextWindow ?? 0;
			const usage = ctx.getContextUsage?.();
			const usageStr =
				usage && typeof usage.tokens === "number" && ctxWindow
					? `model view ${((usage.tokens / ctxWindow) * 100).toFixed(1)}% (API usage ${fmtTokens(usage.tokens)} tok; not included in the compact decision)`
					: "model view unknown (e.g. before any LLM response)";
			parts.push(`${g.canCompact ? "✅ compactable" : "❌ not compactable"} (${g.reason})`);
			parts.push(
				`${g.messageCount} messages, est. ${fmtTokens(g.messageTokens)} tok (pi's char/4, thinking included)`,
			);
			parts.push(`threshold keepRecentTokens=${fmtTokens(g.keepRecentTokens)} tok`);
			parts.push(usageStr);

			if (!lastAttempt) {
				parts.push("no Jev compaction attempted in this session yet");
			} else if (lastAttempt.status === "fallback") {
				parts.push(
					`last compaction fell back (${lastAttempt.detail ?? `reduction ${(lastAttempt.reduction * 100).toFixed(0)}% < ${lastAttempt.threshold * 100}%`}, ` +
						`reason=${lastAttempt.reason})`,
				);
			} else if (lastAttempt.status === "error") {
				parts.push(`last compaction errored (reason=${lastAttempt.reason}): ${lastAttempt.message}`);
			} else {
				const { stats: s, newTokens, prevTokens, totalTokens, capped, reduction, budgetPct, reason } = lastAttempt;
				parts.push(
					`last Jev compaction: ${s.decisions.keep} keep / ${s.decisions.truncate} truncated / ${s.decisions.drop} dropped | ` +
						`new ${fmtTokens(newTokens)} tok (${(reduction * 100).toFixed(0)}% smaller${capped ? `, capped to ${budgetPct}%` : ""})` +
						(prevTokens > 0
							? ` + prev ${fmtTokens(prevTokens)} tok (verbatim) = total ${fmtTokens(totalTokens)} tok`
							: " (no prev)") +
						` | ${s.requests} req, $${s.jevUsage.cost.toFixed(6)}, ${s.elapsedMs}ms | stage ${s.stage}, reason=${reason}`,
				);
			}

			const level =
				lastAttempt?.status === "error" ? "error" : g.canCompact && lastAttempt?.status !== "fallback" ? "info" : "warning";
			ctx.ui.notify("jev-stats: " + parts.join(" | "), level);
		},
	});
}
