/**
 * jev-core.mjs — Jev decision-based compaction core (framework-agnostic).
 *
 * Port of the fast-jev-compaction (tamaratran) idea to pi:
 *  - Never rewrite the conversation. Only tool calls / tool results that Jev
 *    judges unnecessary are dropped or shortened to their head. Text
 *    (user/assistant/thinking) is kept verbatim.
 *  - Shows Jev a state (full text + short notes in place of results) and asks
 *    two noul questions per pair: keep_call / keep_result.
 *  - Uses OpenRouter's /alpha/decisions endpoint.
 */

const DEFAULT_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions";
const DEFAULT_MODEL = "~typesafe/jev-latest";

// ---------------------------------------------------------------------------
// Measured Jev limits (determined by binary search in the benchmark; see the README)
// ---------------------------------------------------------------------------
//
//   input_tokens ceiling : 65,536 (= 2^16)
//                          measured: 65,373 OK / above 65,536 -> HTTP 400 max_tokens_exceeded
//   state-only ceiling   : about 32,958 tok (context window 32K)
//   tokens per question  : about 62 tok (measured; 2 questions per pair -> 124 tok/pair)
//
// Note: the ceiling applies to the **total input_tokens**, not to the number of
//       pairs or questions. A large state leaves less budget for questions, so
//       the remaining budget (input budget minus state) must be divided by the
//       number of pairs before sending.
//
// Old bug: questionsPerBatch (a question count) was used as the step for
//          pairs.slice (a pair count), packing 2x the questions into one request
//          even though 1 pair = 2 questions.

export const JEV_LIMITS = {
	maxInputTokens: 65536, // measured 2^16
	maxStateTokens: 32000, // measured 32,958 (state only)
	tokensPerQuestion: 62, // measured
	tokensPerPair: 124, // = tokensPerQuestion x 2
};

/** Safety ratio against the measured ceilings. Running at the exact limit causes 400s on regressions, so leave headroom. */
const DEFAULT_SAFETY_RATIO = 0.85;

// ---------------------------------------------------------------------------
// token estimation
//
// **Important**: char/4 underestimates CJK (Japanese) by 3-4x (measured: only
// 27-37% of pi's real token count). Since 1 CJK char is about 1.2 tokens, the
// weight depends on the character class. Measured on real sessions near 1M
// tokens, the error against pi's real count stays within +-8%.
// This estimator is shared by reduction / budget / display.
// ---------------------------------------------------------------------------

const CJK_RE = /[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]/;

export function estimateTokens(text) {
	const t = text || "";
	let cjk = 0;
	let other = 0;
	for (const ch of t) {
		if (CJK_RE.test(ch)) cjk++;
		else other++;
	}
	return Math.ceil(cjk * 1.2 + other / 3.5);
}

/**
 * Conservative (larger) estimate. Used to decide whether the state fits in a
 * Jev request. Unified with estimateTokens (CJK-aware).
 */
function estimateTokensConservative(text) {
	return estimateTokens(text);
}

// ---------------------------------------------------------------------------
// AgentMessage[] → segments
// ---------------------------------------------------------------------------

function blocksText(content) {
	const parts = [];
	for (const b of content || []) {
		if (b.type === "text") parts.push(b.text);
		else if (b.type === "image") parts.push("[image omitted]");
	}
	return parts.join("\n");
}

/**
 * Compression policy: keep the tail of thinking / truncate toolCall arguments (Plan C).
 *
 * Measuring a real session (~400k tok) gave this composition:
 *   thinking 31.2% / toolCall args 33.9% / toolResult 25.3% / asstText 9.0% / user 0.5%
 * Under a strict "never rewrite text" rule, thinking+args are 65% of the whole and
 * compaction cannot succeed. So we allow **keeping only the tail of thinking and
 * truncating args** (Plan C).
 *
 * Thinking is an intermediate artifact and its conclusion is reflected in the
 * assistant message, so keeping only the tail (the side closer to the conclusion)
 * should have little effect on downstream quality. <- verified in Phase 4
 *
 * @param {number} thinkingBudgetChars character budget for all thinking (0 = unlimited). Split evenly, tails kept
 * @param {number} argsBudgetChars     character budget for all toolCall args (0 = unlimited). Split evenly
 */
function compressSegments(segs, { thinkingBudgetChars = 0, argsBudgetChars = 0 } = {}) {
	const removed = { thinking: 0, args: 0 };
	if (thinkingBudgetChars <= 0 && argsBudgetChars <= 0) {
		return { segs, removed };
	}
	const out = segs.map((s) => ({ ...s }));

	// thinking: keep the tail (the side closer to the conclusion)
	if (thinkingBudgetChars > 0) {
		const idx = [];
		out.forEach((s, i) => {
			if (s.kind === "text" && s.role === "assistant_thinking" && s.text.length > 240) idx.push(i);
		});
		if (idx.length) {
			const per = Math.max(160, Math.floor(thinkingBudgetChars / idx.length));
			for (const i of idx) {
				const t = out[i].text;
				if (t.length > per) {
					removed.thinking += t.length - per;
					out[i].text = `[… earlier ${t.length - per} chars of thinking omitted …]\n${t.slice(-per)}`;
				}
			}
		}
	}

	// toolCall args: truncate (store the normalized string in argsText)
	if (argsBudgetChars > 0) {
		const idx = [];
		out.forEach((s, i) => {
			if (s.kind === "pair") idx.push(i);
		});
		if (idx.length) {
			const per = Math.max(80, Math.floor(argsBudgetChars / idx.length));
			for (const i of idx) {
				const a = JSON.stringify(out[i].arguments || {});
				if (a.length > per) {
					removed.args += a.length - per;
					out[i].argsText = `${a.slice(0, per)}…(+${a.length - per}ch)`;
				} else {
					out[i].argsText = a;
				}
			}
		}
	}

	// Pairs without argsText get it resolved here (used by later rendering)
	for (const s of out) {
		if (s.kind === "pair" && !s.argsText) s.argsText = argsOneLine(s.arguments, Infinity);
	}
	return { segs: out, removed };
}

/**
 * Convert to an order-preserving list of segments.
 *  - {kind:"text", role, text}          : user / assistant / assistant_thinking (kept verbatim)
 *  - {kind:"pair", idx, id, toolName, args, resultText, resultIsError} : judged by Jev
 * A toolCall in an assistant message is paired with its role:"toolResult" by id.
 *
 * When opts.preserveRecentMessages > 0, pinning compatible with the original
 * (fast-jev-compaction) is enabled: the first message and the last N messages are
 * never judged and are kept verbatim. With 0 (default) no pinning happens at all,
 * which is exactly the previous behavior.
 */
export function collectSegments(messages, opts = {}) {
	const { preserveRecentMessages = 0 } = opts;
	const totalMessages = messages.length;
	// Pinning is enabled only when preserveRecentMessages > 0 (0 is fully backward compatible).
	const isPinnedIndex = (i) =>
		preserveRecentMessages > 0 && (i === 0 || i >= totalMessages - preserveRecentMessages);

	const resultsById = new Map();
	messages.forEach((m, i) => {
		if (m && m.role === "toolResult" && m.toolCallId)
			resultsById.set(m.toolCallId, { message: m, index: i });
	});
	const segs = [];
	let pairIdx = 0;
	messages.forEach((m, mi) => {
		if (!m) return;
		if (m.role === "user") {
			const text = blocksText(m.content);
			if (text) segs.push({ kind: "text", role: "user", text, msgIndex: mi, pinned: segs.length === 0 || isPinnedIndex(mi) });
		} else if (m.role === "assistant") {
			const thinking = [];
			const texts = [];
			const calls = [];
			for (const b of m.content || []) {
				if (b.type === "thinking") thinking.push(b.thinking);
				else if (b.type === "text") texts.push(b.text);
				else if (b.type === "toolCall") calls.push(b);
			}
			if (thinking.length) segs.push({ kind: "text", role: "assistant_thinking", text: thinking.join("\n"), msgIndex: mi, pinned: isPinnedIndex(mi) });
			if (texts.length) segs.push({ kind: "text", role: "assistant", text: texts.join("\n"), msgIndex: mi, pinned: isPinnedIndex(mi) });
			for (const c of calls) {
				const r = resultsById.get(c.id);
				const pinned = isPinnedIndex(mi) || (r ? isPinnedIndex(r.index) : false);
				segs.push({
					kind: "pair",
					idx: pairIdx++,
					id: c.id,
					toolName: c.name,
					args: c.arguments || {},
					resultText: r ? blocksText(r.message.content) : null,
					resultIsError: r ? Boolean(r.message.isError) : false,
					callMsgIndex: mi,
					resultMsgIndex: r ? r.index : -1,
					pinned,
				});
			}
		} else if (m.role === "toolResult") {
			// orphan result (no matching call) -> keep as text for safety
			const text = blocksText(m.content);
			if (text && !resultsById.has(m.toolCallId)) {
				segs.push({ kind: "text", role: "tool_result_orphan", text, msgIndex: mi });
			}
		}
		// Other custom messages are skipped
	});
	return segs;
}

// ---------------------------------------------------------------------------
// state rendering (the full text shown to Jev; results are replaced by short notes)
// ---------------------------------------------------------------------------

function argsOneLine(args, limit) {
	let s = Object.entries(args || {})
		.map(([k, v]) => `${k}=${JSON.stringify(v)}`)
		.join(", ");
	if (limit !== Infinity && s.length > limit) s = s.slice(0, limit) + "…";
	return s;
}

function resultNote(seg) {
	if (seg.resultText === null) return "(no result)";
	const status = seg.resultIsError ? "error" : "ok";
	return `-> ${status} ${seg.resultText.length}ch`;
}

function abridge(text, keep) {
	if (text.length <= keep) return text;
	const head = Math.floor(keep * 0.6);
	const tail = keep - head;
	const tailText = tail > 0 ? text.slice(-tail) : ""; // slice(-0) returns the whole string, so guard it
	return `${text.slice(0, head)}\n[… ${text.length - keep} chars omitted …]\n${tailText}`;
}

function textLine(seg, lim, isFirst) {
	let text = seg.text;
	let tag =
		seg.role === "user" ? "[User]" :
		seg.role === "assistant" ? "[Assistant]" :
		seg.role === "assistant_thinking" ? "[Assistant thinking]" : "[Note]";
	if (seg.role === "assistant_thinking") {
		tag = "[Assistant thinking]";
		// Extension stage: keep only the tail of thinking (closer to the conclusion)
		const tt = lim.thinkingTail ?? Infinity;
		if (tt !== Infinity && text.length > tt) {
			text = `[… earlier ${text.length - tt} chars of thinking omitted …]\n${text.slice(-tt)}`;
		}
	}
	if (!isFirst && lim.collapse !== Infinity && text.length > lim.collapse) {
		return `${tag}: [… ${text.length} chars collapsed …]`;
	}
	if (lim.abridge !== Infinity && text.length > lim.abridge && !isFirst) text = abridge(text, lim.abridge);
	return `${tag}: ${text}`;
}

/** Truncation that keeps head + tail (errors usually appear at the end) */
function truncateHeadTail(text, keep, headRatio = 0.6) {
	if (text.length <= keep) return null; // no truncation needed
	const head = Math.floor(keep * headRatio);
	const tail = keep - head;
	// Note: `slice(-0)` in JS behaves like `slice(0)` and returns the whole string.
	// headRatio=1 makes tail=0, so the empty string has to be explicit.
	const tailText = tail > 0 ? text.slice(-tail) : "";
	return `${text.slice(0, head)}\n[… ${text.length - keep} chars omitted …]\n${tailText}`;
}

/**
 * lim = { argsLimit, abridge, collapse, onelineCalls }
 * Progressively tighten the limits and adopt the first stage that fits the budget.
 */
export function renderState(segs, lim) {
	const parts = [];
	for (const s of segs) {
		if (s.kind === "text") {
			parts.push(textLine(s, lim, s.pinned));
		} else {
			const argsTxt = s.argsText ?? argsOneLine(s.args, lim.argsLimit);
			if (lim.onelineCalls) {
				parts.push(`[${s.idx}] ${s.toolName}(${argsOneLine(s.args, 60)}) ${resultNote(s)}`);
			} else {
				parts.push(`[${s.idx}] TOOL_CALL ${s.toolName}(${argsTxt}) ${resultNote(s)}`);
			}
		}
	}
	return parts.join("\n");
}

const STAGES = [
	{ argsLimit: Infinity, abridge: Infinity, collapse: Infinity, onelineCalls: false, thinkingTail: Infinity },
	{ argsLimit: 1000, abridge: Infinity, collapse: Infinity, onelineCalls: false, thinkingTail: Infinity },
	{ argsLimit: 200, abridge: Infinity, collapse: Infinity, onelineCalls: false, thinkingTail: Infinity },
	{ argsLimit: 60, abridge: Infinity, collapse: Infinity, onelineCalls: false, thinkingTail: Infinity },
	{ argsLimit: 60, abridge: 1200, collapse: Infinity, onelineCalls: false, thinkingTail: Infinity },
	{ argsLimit: 60, abridge: 1200, collapse: 800, onelineCalls: false, thinkingTail: Infinity },
	{ argsLimit: 60, abridge: 1200, collapse: 800, onelineCalls: true, thinkingTail: Infinity },
	// --- Extension stage: truncate thinking by keeping its tail (guards against huge thinking in real sessions) ---
	{ argsLimit: 60, abridge: 1200, collapse: 400, onelineCalls: true, thinkingTail: 2000 },
	{ argsLimit: 60, abridge: 1200, collapse: 200, onelineCalls: true, thinkingTail: 800 },
	{ argsLimit: 60, abridge: 1200, collapse: 100, onelineCalls: true, thinkingTail: 300 },
];

/** Returns a state that fits the budget and the stage used; null if no stage fits. */
export function fitState(segs, maxStateTokens) {
	for (let i = 0; i < STAGES.length; i++) {
		const state = renderState(segs, STAGES[i]);
		if (estimateTokensConservative(state) <= maxStateTokens) return { state, stage: i };
	}
	return null;
}

// ---------------------------------------------------------------------------
// Jev transport
// ---------------------------------------------------------------------------

/** noul questions for thinking / args (Plan D) */
const THINKING_Q = {
	type: "noul",
	instructions:
		"Does this assistant thinking block still matter for continuing the ongoing task (e.g. it contains a conclusion, decision, or finding that has not been expressed elsewhere)?",
	criteria: { true: "Still matters", false: "Superseded or no longer needed" },
};
const ARGS_Q = {
	type: "noul",
	instructions:
		"Do the arguments contain an edit (old text -> new text) or a file path that identifies what was operated on, such that losing them would make it impossible to track what was changed?",
	criteria: { true: "Contains edit diff or path", false: "Generic or re-derivable" },
};

const CALL_Q = {
	type: "noul",
	instructions:
		"Knowing that this tool call was made (with its input), does it still matter for continuing the ongoing task?",
	criteria: { true: "Still matters", false: "No longer needed" },
};
const RES_Q = {
	type: "noul",
	instructions:
		"Does this result contain an error message, stack trace, failure output, or specification detail whose exact wording is needed and cannot be recovered by re-running?",
	criteria: { true: "Contains critical output", false: "Re-derivable or no longer needed" },
};

async function askJevOnce(baseUrl, apiKey, model, state, questions, signal) {
	const body = JSON.stringify({ model, state, questions });
	const resp = await fetch(baseUrl, {
		method: "POST",
		headers: {
			Authorization: `Bearer ${apiKey}`,
			"Content-Type": "application/json",
		},
		body,
		signal,
	});
	if (!resp.ok) {
		const errBody = await resp.text().catch(() => "");
		const err = new Error(`Jev request failed: HTTP ${resp.status} ${errBody.slice(0, 300)}`);
		err.status = resp.status;
		err.isMaxTokens = resp.status === 400 && /max_tokens/i.test(errBody);
		throw err;
	}
	const data = await resp.json();
	// Record body sizes so we can verify "why Jev's input tokens are lower than default"
	data.__requestChars = body.length;
	data.__stateChars = JSON.stringify(state).length;
	data.__questionChars = JSON.stringify(questions).length;
	return data;
}

async function askJev(baseUrl, apiKey, model, state, questions, signal, retries = 2) {
	let lastErr;
	for (let attempt = 0; attempt <= retries; attempt++) {
		try {
			return await askJevOnce(baseUrl, apiKey, model, state, questions, signal);
		} catch (err) {
			if (signal?.aborted) throw err;
			lastErr = err;
			if (attempt < retries) await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
		}
	}
	throw lastErr;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

/**
 * @param {Array} messages  AgentMessage[] (the full set to summarize: messagesToSummarize + turnPrefixMessages)
 * @param {object} options
 *   apiKey, model, keepThreshold, truncateHeadChars, maxStateTokens, maxRequestTokens,
 *   previousSummary, fileOps {read[],written[],edited[]}, signal, baseUrl
 * @returns {Promise<{summary: string, stats: object}>}
 */
/**
 * Split an oversized session and compact it sequentially (multi-pass).
 * Real sessions can exceed 300k tok, where a single pass would blow the state ceiling.
 *
 *   pass1: [first half] -> summary1
 *   pass2: [summary1 + second half] -> final summary
 */
export async function compactJevChunked(messages, options = {}) {
	const { chunkTokens = 120000, preserveRecentMessages = 0 } = options;
	const est = (msgs) => {
		let chars = 0;
		for (const m of msgs) for (const b of m.content || []) {
			chars += (b.text || b.thinking || JSON.stringify(b.arguments || {}) || "").length;
		}
		return Math.ceil(chars / 3); // estimate Japanese generously (conservative side)
	};
	const total = est(messages);
	if (total <= chunkTokens * 1.5) {
		// No split needed
		return { ...(await compactJev(messages, options)), chunked: false };
	}
	// Split messages into chunks of chunkTokens
	const chunks = [];
	let cur = [];
	let curTok = 0;
	for (const m of messages) {
		cur.push(m);
		curTok += est([m]);
		if (curTok >= chunkTokens) {
			chunks.push(cur);
			cur = [];
			curTok = 0;
		}
	}
	if (cur.length) chunks.push(cur);
	console.error(`  [jev-chunked] split into ${chunks.length} chunks (from ${total.toLocaleString()} tok)`);

	let previousSummary = null;
	const allStats = [];
	const allUsage = { input: 0, output: 0, cost: 0, requests: 0 };
	for (let i = 0; i < chunks.length; i++) {
		const chunk = chunks[i];
		// preserveRecentMessages refers to the tail of the *whole* conversation, so it
		// is enabled only on the **final chunk**. Applying it to intermediate chunks
		// would wrongly pin their tail (= the middle of the conversation).
		const isLast = i === chunks.length - 1;
		const r = await compactJev(chunk, {
			...options,
			previousSummary,
			preserveRecentMessages: isLast ? preserveRecentMessages : 0,
		});
		previousSummary = r.summary;
		allStats.push(r.stats);
		allUsage.input += r.stats.jevUsage.input;
		allUsage.output += r.stats.jevUsage.output;
		allUsage.cost += r.stats.jevUsage.cost;
		allUsage.requests += r.stats.jevUsage.requests;
	}
	const last = allStats[allStats.length - 1];
	// Sum of "original size" across all chunks (the content of each chunk itself,
	// excluding previousSummary). This closely matches the estimated tokens of the
	// whole original conversation and is the correct denominator for the reduction rate.
	const summarizedBefore = estTokensBefore_of(allStats);

	// --- Category composition is **summed across all chunks** ---
	// The old implementation returned last (the final chunk) composition as-is, so a
	// split run only saw 1/N of the whole (cases covering just 4-11% were observed).
	const CATEGORY_KEYS = ["tool_result", "thinking", "toolcall_args", "asst_text", "user_text"];
	const mergeComp = (key) => {
		const out = {};
		for (const k of CATEGORY_KEYS) out[k] = { chars: 0, count: 0 };
		for (const s of allStats) {
			const c = s.composition?.[key];
			if (!c) continue;
			for (const k of CATEGORY_KEYS) {
				out[k].chars += c[k]?.chars || 0;
				out[k].count += c[k]?.count || 0;
			}
		}
		return out;
	};
	const mergedBefore = mergeComp("before");
	const mergedAfter = mergeComp("after");

	// --- Hard size limit on the final summary ---
	// In split compaction previousSummary accumulates, so the final summary can
	// exceed the overall budget. Trim it forcibly here.
	const finalBudgetChars = (options.maxSummaryTokens || Math.round(total * 0.15)) * 4;
	let finalSummary = previousSummary;
	if (finalSummary.length > finalBudgetChars) {
		// Keep head 60% + tail 40% (errors appear at the end)
		const head = Math.floor(finalBudgetChars * 0.6);
		const tail = finalBudgetChars - head;
		finalSummary =
			finalSummary.slice(0, head) +
			`\n\n[... FINAL COMPACTION: ${finalSummary.length - finalBudgetChars} chars trimmed to fit budget ...]\n\n` +
			finalSummary.slice(-tail);
	}

	const summarizedSummary = estimateTokens(finalSummary); // same CJK-aware estimator as the single pass

	return {
		summary: finalSummary,
		stats: {
			...last,
			jevUsage: { ...last.jevUsage, ...allUsage, requests: allUsage.requests },
			decisions: last.decisions,
			chunked: true,
			chunks: allStats.map((s2) => ({ decisions: s2.decisions, summaryTokens: s2.estTokens.summary })),
			estTokens: { summarizedBefore, summary: summarizedSummary },
			reduction: 1 - summarizedSummary / Math.max(1, summarizedBefore),
			// Category composition summed across all chunks (the old version used the final chunk only)
			composition: {
				before: mergedBefore,
				after: mergedAfter,
				reductionPct: Object.fromEntries(
					CATEGORY_KEYS.map((k) => {
						const b = mergedBefore[k]?.chars || 0;
						const a = mergedAfter[k]?.chars || 0;
						return [k, b > 0 ? Math.round((1 - a / b) * 1000) / 10 : null];
					}),
				),
				mergedChunks: allStats.length,
			},
			elapsedMs: allStats.reduce((a, s2) => a + (s2.elapsedMs || 0), 0),
		},
		chunked: true,
	};
}
// Helper: sum of summarizedBefore across all chunks
function estTokensBefore_of(allStats) {
	return allStats.reduce((a, s) => a + (s.estTokens?.summarizedBefore || 0), 0);
}

export async function compactJev(messages, options = {}) {
	const {
		apiKey,
		model = DEFAULT_MODEL,
		baseUrl = DEFAULT_DECISIONS_URL,
		keepThreshold = 0.5,
		truncateHeadChars = 300,
		truncateHeadRatio = 0.6, // share allocated to the head (the rest goes to the tail; bash errors appear at the end)
		maxSummaryTokens = 0, // summary ceiling (tok). 0 = unlimited. When exceeded, headChars shrinks automatically
		thinkingBudgetChars = 0, // character budget for all thinking (0 = unlimited)
		argsBudgetChars = 0, // character budget for all toolCall args (0 = unlimited)
		summaryBudgetRatio = 0.15, // what fraction of the original context the summary may occupy (budget-based selection)
		askThinking = false, // let Jev judge thinking (Plan D)
		askArgs = false, // let Jev judge toolCall args (Plan D)
		preserveRecentMessages = 0, // >0 enables original-compatible pinning (first + last N messages)
		disableBudgetSelection = false, // true disables budget-based selection (original-compatible: decide by Jev alone)
		thinkingMinChars = 600, // thinking shorter than this is kept without judging
		argsMinChars = 400, // args shorter than this are kept without judging
		thinkingTailChars = 240, // tail length kept for thinking judged unnecessary
		argsKeepChars = 150, // length kept for args judged unnecessary
		maxStateTokens = JEV_LIMITS.maxStateTokens, // measured ceiling (32,958)
		maxRequestTokens = JEV_LIMITS.maxInputTokens, // measured ceiling 65,536
		safetyRatio = DEFAULT_SAFETY_RATIO,
		tokensPerPair = JEV_LIMITS.tokensPerPair,
		previousSummary = null,
		fileOps = null,
		signal,
	} = options;
	if (!apiKey) throw new Error("Jev compaction: missing API key (OPENROUTER_JEV_API_KEY)");

	const started = Date.now();
	const segsRaw = collectSegments(messages, { preserveRecentMessages });
	const pairsRaw = segsRaw.filter((s) => s.kind === "pair");

	// Compute the original size (before compression) first
	const charsBeforeRaw = segsRaw.reduce((a, s) => a + (s.kind === "text" ? s.text.length : argsOneLine(s.args, Infinity).length + (s.resultText?.length || 0)), 0);
	const estTokensBefore = estimateTokens(
		segsRaw.map((s) => (s.kind === "text" ? s.text : `${s.toolName}(${argsOneLine(s.args, Infinity)}) ${s.resultText || ""}`)).join("\n"),
	);

	// Plan C: compress thinking / args
	const { segs: segsC, removed: policyRemoved } = compressSegments(segsRaw, { thinkingBudgetChars, argsBudgetChars });
	const segs = segsC;
	const allPairs = segs.filter((s) => s.kind === "pair");
	const pinnedPairs = allPairs.filter((s) => s.pinned);
	// Only non-pinned candidates are judged by Jev (pinned pairs are always kept)
	const pairs = allPairs.filter((s) => !s.pinned);

	// --- Original size estimate (for the reduction rate) ---
	const charsBefore = charsBeforeRaw;

	// --- Per-category composition measurement (the signal for the adjustment loop) ---
	// Categories: tool_result / thinking / toolcall_args / asst_text / user_text
	const CATEGORY_KEYS = ["tool_result", "thinking", "toolcall_args", "asst_text", "user_text"];
	const catOf = (s) => {
		if (s.kind === "text") {
			if (s.role === "user") return "user_text";
			if (s.role === "assistant_thinking") return "thinking";
			return "asst_text";
		}
		return null; // pairs are counted separately as args and result
	};
	const beforeComp = {};
	for (const k of CATEGORY_KEYS) beforeComp[k] = { chars: 0, count: 0 };
	for (const s of segsRaw) {
		const cat = catOf(s);
		if (cat) {
			beforeComp[cat].chars += s.kind === "text" ? s.text.length : 0;
			beforeComp[cat].count += 1;
		} else {
			beforeComp.toolcall_args.chars += (s.argsText || argsOneLine(s.args, Infinity)).length;
			beforeComp.toolcall_args.count += 1;
			beforeComp.tool_result.chars += s.resultText?.length || 0;
			beforeComp.tool_result.count += 1; // the old implementation always left this at 0
		}
	}

	// --- Fit the state into the budget ---
	const fitted = fitState(segs, maxStateTokens);
	if (!fitted) throw new Error("Jev compaction: history does not fit into maxStateTokens even at the last stage");
	const { state, stage } = fitted;
	const stateTokens = estimateTokensConservative(state);

	// --- Split questions into batches (all batches are sent concurrently) ---
	//
	// The ceiling applies to the total input_tokens, so the budget left after the
	// state is divided by the number of pairs. The batch count is decided from the
	// pair count before sending (structurally prevents 400 max_tokens_exceeded).
	const inputBudget = Math.floor(maxRequestTokens * safetyRatio);
	// If it does not fit (e.g. the state alone eats the ceiling), fail explicitly
	if (stateTokens > inputBudget) {
		throw new Error(
			`Jev compaction: state (${stateTokens} tok) exceeds the request budget (${inputBudget} tok = ${maxRequestTokens} x ${safetyRatio}). ` +
				`lower maxStateTokens (measured ceiling: state ${JEV_LIMITS.maxStateTokens} tok)`,
		);
	}
	const questionBudget = inputBudget - stateTokens;
	// Question tokens per pair (Plan D adds args judging, turning 2 questions into 3)
	const questionsPerPair = askArgs ? 3 : 2;
	const tokensPerPairActual = questionsPerPair * JEV_LIMITS.tokensPerQuestion;
	// --- Plan D: let Jev judge thinking / args as well ---
	// Thinking is an "intermediate artifact" but occupies 31% of a real session.
	// args occupy 34%. Rather than allocating budget arbitrarily, let Jev itself
	// decide what is needed and what is not.
	const thinkings = askThinking
		? segs
				.map((s, i) => ({ seg: s, segIndex: i }))
				.filter((x) => x.seg.kind === "text" && x.seg.role === "assistant_thinking" && x.seg.text.length > thinkingMinChars)
		: [];

	// Batch count: how many times the total question tokens (pairs + thinking) fit the budget
	const totalQuestionTokens = pairs.length * tokensPerPairActual + thinkings.length * JEV_LIMITS.tokensPerQuestion;
	const batchesNeeded = Math.max(1, Math.ceil(totalQuestionTokens / Math.max(1, questionBudget)));
	const pairsPerBatch = Math.max(1, Math.ceil(pairs.length / batchesNeeded));
	const batches = [];
	for (let i = 0; i < pairs.length; i += pairsPerBatch) {
		batches.push({ pairs: pairs.slice(i, i + pairsPerBatch), thinkings: [] });
	}

	const thinkingSegIds = new Map(thinkings.map((t, i) => [t.seg, i]));
	const totalQuestions = pairs.length * questionsPerPair + thinkings.length;
	// Distribute thinking evenly across batches (1 block = 1 question)
	thinkings.forEach((t, i) => batches[i % Math.max(1, batches.length)].thinkings.push({ ...t, id: i }));

	const largeArgs = askArgs ? pairs.filter((p) => (p.argsText || argsOneLine(p.arguments, Infinity)).length > argsMinChars) : [];
	const decisions = new Map(); // pairIdx -> {call:number, res:number, args?:number}
	const thinkingDecisions = new Map(); // thinkingId -> number
	// Pinned pairs are never judged and are always kept (original-compatible).
	// Registering them in decisions makes applyDecisions treat them as keep and
	// protects them in budget selection too.
	for (const p of pinnedPairs) decisions.set(p.idx, { call: 1, res: 1, pinned: true });
	const usage = { requests: 0, input: 0, output: 0, cost: 0, model, requestChars: 0, stateChars: 0, questionChars: 0 };
	let resolvedModel = model;

	if (batches.length === 0) {
		// Nothing to judge (text only) -> nothing can be removed
	} else {
		const largeArgIds = new Set(largeArgs.map((p) => p.idx));
		/**
		 * Build the questions for one batch.
		 */
		const buildQuestions = (batch) => {
			const questions = {};
			for (const p of batch.pairs) {
				questions[`c${p.idx}_call`] = CALL_Q;
				questions[`c${p.idx}_res`] = RES_Q;
				if (askArgs && largeArgIds.has(p.idx)) questions[`c${p.idx}_args`] = ARGS_Q;
			}
			for (const t of batch.thinkings) {
				questions[`t${t.id}_keep`] = THINKING_Q;
			}
			return questions;
		};

		/**
		 * Self-healing: on a 400 max_tokens_exceeded, split the batch in half and retry.
		 * This converges regardless of estimation error.
		 */
		const runBatchAdaptive = async (batch) => {
			try {
				const questions = buildQuestions(batch);
				const data = await askJev(baseUrl, apiKey, model, state, questions, signal);
				return [data];
			} catch (err) {
				if (!err.isMaxTokens) throw err;
				// A batch too small to split still failed -> give up (treated as an error upstream)
				if (batch.pairs.length <= 1 && batch.thinkings.length <= 1) throw err;
			}
			// Split in half and recurse
			const midP = Math.floor(batch.pairs.length / 2);
			const midT = Math.floor(batch.thinkings.length / 2);
			const halfA = { pairs: batch.pairs.slice(0, midP), thinkings: batch.thinkings.slice(0, midT) };
			const halfB = {
				pairs: batch.pairs.slice(midP),
				thinkings: batch.thinkings.slice(midT),
			};
			const out = [];
			for (const h of [halfA, halfB]) {
				if (!h.pairs.length && !h.thinkings.length) continue;
				out.push(...(await runBatchAdaptive(h)));
			}
			return out;
		};

		const results = (
			await Promise.all(batches.map((batch) => runBatchAdaptive(batch)))
		).flat();
		for (const data of results) {
			resolvedModel = data.model || resolvedModel;
			usage.requests += 1;
			usage.requestChars += data.__requestChars || 0;
			usage.stateChars += data.__stateChars || 0;
			usage.questionChars += data.__questionChars || 0;
			const u = data.usage || {};
			usage.input += u.input_tokens || 0;
			usage.output += u.output_tokens || 0;
			usage.cost += u.cost || 0;
			for (const [qid, ans] of Object.entries(data.answers || {})) {
				const tm = qid.match(/^t(\d+)_keep$/);
				if (tm) {
					thinkingDecisions.set(Number(tm[1]), ans.noul);
					continue;
				}
				const idx = parseInt(qid.replace(/_(call|res|args)$/, "").slice(1), 10);
				if (Number.isNaN(idx)) continue;
				const d = decisions.get(idx) || {};
				if (qid.endsWith("_call")) d.call = ans.noul;
				else if (qid.endsWith("_args")) d.args = ans.noul;
				else d.res = ans.noul;
				decisions.set(idx, d);
			}
		}
	}

	// --- Apply the decisions ---
	const probs = [...decisions.values()];
	const probStat = (key) => {
		const xs = probs.map((d) => d[key]).filter((x) => typeof x === "number");
		if (!xs.length) return null;
		return { min: Math.min(...xs), max: Math.max(...xs), avg: xs.reduce((a, b) => a + b, 0) / xs.length };
	};
	let counts = { keep: 0, truncate: 0, drop: 0, noresult: 0 };
	const dropped = [];
	const outParts = [];

	/**
	 * Apply the decisions and assemble outParts.
	 * @param {number} headChars number of characters kept when truncating
	 */
	const applyDecisions = (headChars) => {
		const counts2 = { keep: 0, truncate: 0, drop: 0, noresult: 0 };
		const dropped2 = [];
		const parts = [];
		const after2 = {};
		for (const k of CATEGORY_KEYS) after2[k] = { chars: 0, count: 0 };
		const addAfter = (cat, len) => {
			after2[cat].chars += len;
			after2[cat].count += 1;
		};
		for (const s of segs) {
			if (s.kind === "text") {
				// Plan D: thinking is either tail-kept or fully kept per Jev's judgment
				if (s.role === "assistant_thinking" && askThinking) {
					const tid = thinkingSegIds.get(s);
					const keep = thinkingDecisions.get(tid);
					if (typeof keep === "number" && keep < keepThreshold && s.text.length > thinkingTailChars) {
						counts2.drop += 1;
						const line = `[Assistant thinking]: [… ${s.text.length - thinkingTailChars} chars of thinking omitted …]\n${s.text.slice(-thinkingTailChars)}`;
						parts.push(line);
						addAfter("thinking", line.length);
						continue;
					}
				}
				const tag =
					s.role === "user" ? "[User]" :
					s.role === "assistant" ? "[Assistant]" :
					s.role === "assistant_thinking" ? "[Assistant thinking]" : "[Note]";
				const pushed = `${tag}: ${s.text}`;
				parts.push(pushed);
				addAfter(catOf(s), pushed.length);
			} else {
				const d = decisions.get(s.idx) || { call: 0, res: 0 };
				const argsText = s.argsText || argsOneLine(s.args, Infinity);
				const callTag = `[${s.idx}] TOOL_CALL ${s.toolName}(${argsText})`;
				addAfter("toolcall_args", callTag.length);
				if (s.resultText === null) {
					counts2.noresult += 1;
					const line = `${callTag}\n[${s.idx}] TOOL_RESULT: (no result recorded)`;
					parts.push(line);
					addAfter("tool_result", line.length - callTag.length);
				} else if (d.res >= keepThreshold) {
					counts2.keep += 1;
					const line = `${callTag}\n[${s.idx}] TOOL_RESULT: ${s.resultText}`;
					parts.push(line);
					addAfter("tool_result", line.length - callTag.length);
				} else if (d.call >= keepThreshold) {
					counts2.truncate += 1;
					const tt = truncateHeadTail(s.resultText, headChars, truncateHeadRatio);
					const line = `${callTag}\n[${s.idx}] TOOL_RESULT: ${tt ?? s.resultText}`;
					parts.push(line);
					addAfter("tool_result", line.length - callTag.length);
				} else {
					counts2.drop += 1;
					const note = `${callTag} ${resultNote(s)}`;
					dropped2.push(note);
					addAfter("tool_result", note.length);
				}
			}
		}
		return { counts: counts2, dropped: dropped2, parts, after: after2 };
	};

	// --- Dynamic budget: when the summary exceeds the ceiling, shrink headChars and rebuild ---
	//
	// In real sessions, "many truncate decisions x fixed headChars" once produced a
	// summary larger than the original context (220,955 tok vs 264,897 tok original).
	// Jev's judgments are complete before truncation, so headChars can be shrunk and
	// the output rebuilt **without any extra API calls** (zero additional cost).
	let applied = applyDecisions(truncateHeadChars);
	let headApplied = truncateHeadChars;
	let summaryShrunk = null;

	// Per-category after-size accounting (the signal for the adjustment loop)
	const afterComp = {};
	for (const k of CATEGORY_KEYS) afterComp[k] = { chars: 0, count: 0 };
	const addAfter = (cat, len) => {
		afterComp[cat].chars += len;
		afterComp[cat].count += 1;
	};
	if (maxSummaryTokens > 0) {
		const size = () => estimateTokens([...applied.parts, ...applied.dropped].join("\n"));
		let guard = 0;
		while (size() > maxSummaryTokens && guard++ < 6) {
			const ratio = size() / maxSummaryTokens;
			headApplied = Math.max(0, Math.floor(headApplied / ratio));
			applied = applyDecisions(headApplied);
			summaryShrunk = { from: truncateHeadChars, to: headApplied };
			if (headApplied === 0) break;
		}
	}
	counts = applied.counts;
	dropped.push(...applied.dropped);
	outParts.push(...applied.parts);
	for (const k of CATEGORY_KEYS) afterComp[k] = applied.after[k];

	// --- Budget-based selection: rank by Jev score and keep the top entries within budget ---
	//
	// Jev alone did not compress enough (the summary was 74% of the original).
	// So Jev's role was shifted from "keep/drop judge" to "importance scorer", and
	// the highest-scoring items are kept within the budget.
	const summaryBudgetTokens = maxSummaryTokens > 0
		? maxSummaryTokens
		: Math.max(2000, Math.round(estTokensBefore * summaryBudgetRatio));

	// If the summary exceeds the budget, re-select by budget
	// (disableBudgetSelection=true keeps the original behavior: decide by Jev alone)
	const currentTokens = estimateTokens(outParts.join("\n"));
	if (!disableBudgetSelection && currentTokens > summaryBudgetTokens && pairs.length + thinkings.length > 0) {
		// Segments that must never be dropped: pinned pairs
		const mustKeep = new Set(pinnedPairs);

		// Assign a "Jev score" to every segment
		const scored = [];
		for (const s of segs) {
			if (mustKeep.has(s)) continue; // pinned items are excluded from budget competition
			if (s.kind === "pair") {
				const d = decisions.get(s.idx) || {};
				// Mean of the call + result scores (a combined importance metric)
				const score = ((d.call ?? 0) + (d.res ?? 0)) / 2;
				const argsText = s.argsText || argsOneLine(s.args, Infinity);
				const resultLen = s.resultText?.length || 0;
				scored.push({ seg: s, score, chars: callTagLen(s) + resultLen });
			} else {
				// thinking / user / asst: thinking has a score, user has the highest priority
				const cat = catOf(s);
				if (cat === "user_text") {
					scored.push({ seg: s, score: 1.0, chars: s.text.length }); // user is mandatory
				} else if (cat === "thinking") {
					const tid = thinkingSegIds.get(s);
					const score = thinkingDecisions.get(tid) ?? 0.54;
					scored.push({ seg: s, score, chars: s.text.length });
				} else {
					// asst_text: medium priority (contains conclusions)
					scored.push({ seg: s, score: 0.6, chars: s.text.length });
				}
			}
		}
		// Length of the call tag
		function callTagLen(s) {
			const argsText = s.argsText || argsOneLine(s.args, Infinity);
			return `[${s.idx}] TOOL_CALL ${s.toolName}(${argsText})\n[${s.idx}] TOOL_RESULT: `.length;
		}

		// Find the maximum number that fits the budget (sort by score descending and greedily select)
		const sorted = [...scored].sort((a, b) => b.score - a.score);
		const budgetChars = summaryBudgetTokens * 4;
		const keepSet = new Set(mustKeep);
		// Account for the consumption of pinned pairs first
		let used = 0;
		for (const s of mustKeep) {
			used += (s.kind === "text" ? s.text.length : callTagLen(s) + (s.resultText?.length || 0)) + 50;
		}
		for (const item of sorted) {
			const cost = item.chars + 50; // +50 is the tag / newline overhead
			if (used + cost <= budgetChars) {
				keepSet.add(item.seg);
				used += cost;
			}
		}

		// Rebuild the summary using only what was selected within budget
		outParts.length = 0;
		for (const s of segs) {
			if (!keepSet.has(s)) continue;
			if (s.kind === "text") {
				const tag = s.role === "user" ? "[User]" : s.role === "assistant" ? "[Assistant]" : s.role === "assistant_thinking" ? "[Assistant thinking]" : "[Note]";
				outParts.push(`${tag}: ${s.text}`);
			} else {
				const d = decisions.get(s.idx) || { call: 0, res: 0 };
				const callTag = `[${s.idx}] TOOL_CALL ${s.toolName}(${s.argsText || argsOneLine(s.args, Infinity)})`;
				if (s.resultText === null) {
					outParts.push(`${callTag}\n[${s.idx}] TOOL_RESULT: (no result recorded)`);
				} else if (d.res >= keepThreshold) {
					outParts.push(`${callTag}\n[${s.idx}] TOOL_RESULT: ${s.resultText}`);
				} else {
					const tt = truncateHeadTail(s.resultText, Math.min(headApplied, 600), truncateHeadRatio);
					outParts.push(`${callTag}\n[${s.idx}] TOOL_RESULT: ${tt ?? s.resultText}`);
				}
			}
		}

		// Re-account the after sizes
		for (const k of CATEGORY_KEYS) afterComp[k] = { chars: 0, count: 0 };
		for (const s of segs) {
			if (!keepSet.has(s)) continue;
			if (s.kind === "text") {
				const cat = catOf(s);
				const len = s.text.length + 20;
				afterComp[cat].chars += len;
				afterComp[cat].count += 1;
			} else {
				const argsText = s.argsText || argsOneLine(s.args, Infinity);
				afterComp.toolcall_args.chars += argsText.length + 40;
				afterComp.toolcall_args.count += 1;
				afterComp.tool_result.chars += s.resultText?.length || 0;
			}
		}
		// Recompute the decision counts too
		let newKeep = 0, newTrunc = 0, newDrop = 0;
		for (const s of segs) {
			if (s.kind !== "pair") continue;
			const d = decisions.get(s.idx) || {};
			if (keepSet.has(s)) { if (d.res >= keepThreshold) newKeep++; else newTrunc++; }
			else newDrop++;
		}
		counts = { ...counts, keep: newKeep, truncate: newTrunc, drop: newDrop };
	}

	// Per-category breakdown of Jev's judgments (score distribution and counts)
	const decisionsByCat = {
		tool_result: { keep: 0, truncate: 0, drop: 0, scores: [] },
		toolcall_args: { keep: 0, drop: 0, scores: [] },
		thinking: { keep: 0, drop: 0, scores: [] },
	};
	for (const p of pairs) {
		const d = decisions.get(p.idx) || {};
		if (typeof d.res === "number") {
			const b = decisionsByCat.tool_result;
			if (d.res >= keepThreshold) b.keep++; else if (d.call >= keepThreshold) b.truncate++; else b.drop++;
			b.scores.push(d.res);
		}
		if (typeof d.args === "number") {
			const b = decisionsByCat.toolcall_args;
			if (d.args >= keepThreshold) b.keep++; else b.drop++;
			b.scores.push(d.args);
		}
	}
	for (const t of thinkings) {
		const sc = thinkingDecisions.get(t.id);
		if (typeof sc === "number") {
			const b = decisionsByCat.thinking;
			if (sc >= keepThreshold) b.keep++; else b.drop++;
			b.scores.push(sc);
		}
	}
	const stat = (arr) => arr.length ? { min: Math.min(...arr), avg: arr.reduce((a,b)=>a+b,0)/arr.length, max: Math.max(...arr) } : null;
	const decisionsByCategory = {};
	for (const [k,v] of Object.entries(decisionsByCat)) decisionsByCategory[k] = { keep: v.keep, truncate: v.truncate, drop: v.drop, scoreStats: stat(v.scores) };

	// --- Assemble the summary (= the pruned verbatim archive) ---
	const header = [
		"<jev-pruned-context>",
		"This is NOT a summary. It is the earlier conversation with stale tool calls/results removed by a scoring model (Jev).",
		"Everything shown is verbatim except tool results marked as truncated. If you need a dropped tool's content, re-run the tool.",
		"",
	];
	const footer = [];
	if (dropped.length) {
		footer.push("", "## Dropped tool calls (judged stale)", ...dropped);
	}
	if (fileOps) {
		const fmt = (label, list) => (list && list.length ? [`${label}: ${list.join(", ")}`] : []);
		const f = [
			"",
			"## Files touched in the pruned span",
			...fmt("read", fileOps.read),
			...fmt("written", fileOps.written),
			...fmt("edited", fileOps.edited),
		];
		if (f.length > 2) footer.push(...f);
	}
	footer.push("", "</jev-pruned-context>");

	const bodyParts = [];
	if (previousSummary) {
		bodyParts.push("## Previous context (from earlier compaction, verbatim)", "", previousSummary, "");
	}
	if (outParts.length) {
		bodyParts.push("## Pruned transcript", "", ...outParts);
	} else if (!previousSummary) {
		bodyParts.push("(nothing survived pruning)");
	}

	const summary = [...header, ...bodyParts, ...footer].join("\n");

	// --- Statistics ---
	const estTokensAfter = estimateTokens(summary);
	// Breakdown of the body actually sent to Jev (to verify "why total input is lower than default")
	const requestBreakdown = {
		totalChars: usage.requestChars,
		stateChars: usage.stateChars,
		questionChars: usage.questionChars,
		questions: decisions.size * 2,
	};
	const stats = {
		model: resolvedModel,
		stage,
		stateTokens,
		requests: usage.requests,
		pairsPerBatch,
		questionsPerBatch: pairsPerBatch * 2,
		inputBudget,
		safetyRatio,
		batches: batches.length,
		decisions: counts,
		truncate: { headChars: headApplied, requestedHeadChars: truncateHeadChars, headRatio: truncateHeadRatio, summaryShrunk },
		policy: { thinkingBudgetChars, argsBudgetChars, removed: policyRemoved },
		profile: { preserveRecentMessages, disableBudgetSelection, askThinking, askArgs, truncateHeadChars, truncateHeadRatio },
		pinned: { pairs: pinnedPairs.length, candidates: pairs.length },
		probabilities: { call: probStat("call"), res: probStat("res") },
		chars: { before: charsBefore, after: summary.length },
		estTokens: { summarizedBefore: estTokensBefore, summary: estTokensAfter },
		composition: {
			before: beforeComp,
			after: afterComp,
			reductionPct: Object.fromEntries(CATEGORY_KEYS.map((k) => {
				const b = beforeComp[k]?.chars || 0;
				const a = afterComp[k]?.chars || 0;
				return [k, b > 0 ? Math.round((1 - a / b) * 1000) / 10 : null];
			})),
		},
		decisionsByCategory,
		reduction: estTokensBefore > 0 ? 1 - estTokensAfter / estTokensBefore : 0,
		jevUsage: usage,
		requestBreakdown,
		elapsedMs: Date.now() - started,
	};

	return { summary, stats };
}
