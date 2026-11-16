#!/usr/bin/env node
/**
 * VideoGameQA-Bench runner: scores our judging approach on a public video-glitch benchmark.
 * https://asgaardlab.github.io/videogameqa-bench/ (data: huggingface.co/datasets/taesiri/VideoGameQA-Bench)
 *
 * Tasks
 *   glitch  "Video Glitch Detection": 1,000 clips (500 glitch / 500 clean), answer true/false.
 *   niah    "Needle in a Haystack": 100 Unity clips (~50s), one planted glitch; answer its time in seconds.
 *
 * Setups (compared on the same clips, same model)
 *   baseline   One call with the dataset's own prompt. This is how the paper scored models.
 *   loop       Our loop's judging: two independent judges, a third call only when they disagree,
 *              majority vote. For niah the judges also get the times our frame-diff check flagged.
 *   framediff  niah only, no model: guess the time of the biggest sudden change between frames.
 *
 * Usage
 *   node scripts/bench/videogameqa.mjs --task niah --limit 20 --dry-run
 *   node scripts/bench/videogameqa.mjs --task niah --limit 20 --setups baseline,loop,framediff
 *   node scripts/bench/videogameqa.mjs --task glitch --limit 100 --repeats 3 --budget 30
 *   node scripts/bench/videogameqa.mjs --run <name>        # resume / re-summarize a run
 *
 * Results go to runs/bench/videogameqa/<run>/ (results.jsonl, summary.json, summary.md).
 * Re-running with the same --run skips clips already scored, so a crash or 429 never loses work.
 * ORCH_FAKE=1 runs everything offline with fake answers (no downloads, no API key).
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { callModel, uploadMediaFile } from '../lib/orchestrator.mjs';
import { toWireParts } from '../loop/wire.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BENCH_DIR = process.env.VGQA_DIR || path.join(ROOT, 'runs/bench/videogameqa');
const DATASET = 'taesiri/VideoGameQA-Bench';
const ROWS_API = `https://datasets-server.huggingface.co/rows?dataset=${DATASET}&config=default&split=test`;
const FILE_BASE = `https://huggingface.co/datasets/${DATASET}/resolve/main/`;

export const TASKS = {
  glitch: {
    category: 'Video Glitch Detection',
    // Accuracy (%) from the paper's leaderboard, for context only (different model versions).
    published: { 'gemini-2.5-pro': 78.1, 'gemini-2.5-flash': 64.7, 'gemini-2.0-flash': 54.5 },
  },
  niah: {
    category: 'Needle in a Haystack',
    published: { 'gemini-2.5-pro': 34.0, 'gemini-2.5-flash': 35.0, 'gemini-2.0-flash': 36.0 },
  },
};
const NIAH_TOLERANCES = [1, 2, 5]; // seconds; the paper's headline uses 5s
const UPLOAD_TTL_MS = 40 * 3600 * 1000; // Gemini Files API keeps uploads ~48h

const SCHEMAS = {
  glitch: {
    type: 'object',
    properties: { reasoning: { type: 'string' }, glitch_detected: { type: 'boolean' } },
    required: ['glitch_detected'],
  },
  niah: {
    type: 'object',
    properties: {
      reasoning: { type: 'string' },
      glitch_detected: { type: 'boolean' },
      timestamp: { type: 'integer' },
    },
    required: ['glitch_detected', 'timestamp'],
  },
};

// ---------- pure helpers (unit tested) ----------

const sha = (s) => crypto.createHash('sha1').update(String(s)).digest('hex');

/** Truth for a dataset row: { glitch: bool, timestamp: number|null }. */
export function parseTruth(row) {
  const gt = JSON.parse(row.ground_truth);
  return { glitch: gt.glitch_detected === true, timestamp: Number.isFinite(gt.timestamp) ? gt.timestamp : null };
}

/** Normalizes a model answer (parsed object or raw text) to { glitch, timestamp } or null. */
export function parseAnswer(output, rawText = '') {
  let obj = output && typeof output === 'object' && 'glitch_detected' in output ? output : null;
  if (!obj) {
    const text = typeof output === 'string' ? output : rawText || output?.notes || '';
    const m = String(text).match(/\{[\s\S]*\}/);
    if (m) {
      try {
        obj = JSON.parse(m[0]);
      } catch {
        obj = null;
      }
    }
  }
  if (!obj || !('glitch_detected' in obj)) return null;
  const g = obj.glitch_detected;
  const ts = Number(obj.timestamp);
  return {
    glitch: g === true || g === 'true',
    timestamp: Number.isFinite(ts) ? ts : null,
    reasoning: typeof obj.reasoning === 'string' ? obj.reasoning.slice(0, 400) : undefined,
  };
}

/** Deterministic sample. For glitch, half glitch / half clean so accuracy isn't skewed. */
export function pickSample(rows, task, limit) {
  const own = rows
    .filter((r) => (r.question_categories || []).includes(TASKS[task].category))
    .sort((a, b) => sha(a.custom_id).localeCompare(sha(b.custom_id)));
  if (!limit || limit >= own.length) return own;
  if (task !== 'glitch') return own.slice(0, limit);
  const pos = own.filter((r) => parseTruth(r).glitch);
  const neg = own.filter((r) => !parseTruth(r).glitch);
  const half = Math.ceil(limit / 2);
  return [...pos.slice(0, half), ...neg.slice(0, limit - half)];
}

/** Majority vote over answers; timestamp = median of the answers that saw a glitch. */
export function majority(answers) {
  const valid = answers.filter(Boolean);
  if (valid.length === 0) return null;
  const yes = valid.filter((a) => a.glitch);
  const glitch = yes.length * 2 > valid.length;
  const times = yes.map((a) => a.timestamp).filter((t) => t !== null).sort((x, y) => x - y);
  const timestamp = times.length ? times[Math.floor((times.length - 1) / 2)] : null;
  return { glitch, timestamp };
}

/** True when two judges agree (same verdict and, for niah, times within 2s). */
export function judgesAgree(a, b, task) {
  if (!a || !b || a.glitch !== b.glitch) return false;
  if (task !== 'niah' || !a.glitch) return true;
  return a.timestamp !== null && b.timestamp !== null && Math.abs(a.timestamp - b.timestamp) <= 2;
}

/** Collapses per-cell frame-diff spikes into the top-k distinct moments (seconds). */
export function topSpikeTimes(spikes, k = 3, mergeWithinSec = 1) {
  const byFrame = new Map();
  for (const s of spikes || []) {
    const score = Math.max(s.maxCellFrac || 0, s.d || 0);
    const prev = byFrame.get(s.frameIndex);
    if (!prev || score > prev.score) byFrame.set(s.frameIndex, { t: s.t, score });
  }
  const picked = [];
  for (const c of [...byFrame.values()].sort((a, b) => b.score - a.score)) {
    if (picked.every((p) => Math.abs(p.t - c.t) > mergeWithinSec)) picked.push(c);
    if (picked.length >= k) break;
  }
  return picked.map((p) => Number(p.t.toFixed(1)));
}

/** Scores one prediction against the truth. */
export function scoreOne(task, truth, pred) {
  if (!pred) return { correct: false, error: true };
  if (task === 'glitch') return { correct: pred.glitch === truth.glitch };
  const absErr = pred.glitch && pred.timestamp !== null ? Math.abs(pred.timestamp - truth.timestamp) : null;
  const within = Object.fromEntries(NIAH_TOLERANCES.map((s) => [`${s}s`, absErr !== null && absErr <= s]));
  return { correct: within['5s'], absErr, within };
}

/** Summary per setup from result lines. */
export function summarize(task, results) {
  const out = {};
  for (const setup of [...new Set(results.map((r) => r.setup))]) {
    const rs = results.filter((r) => r.setup === setup && !r.error);
    const errors = results.filter((r) => r.setup === setup && r.error).length;
    const cost = results.filter((r) => r.setup === setup).reduce((s, r) => s + (r.costUsd || 0), 0);
    const s = { n: rs.length, errors, costUsd: Number(cost.toFixed(4)) };
    if (task === 'glitch') {
      const tp = rs.filter((r) => r.truth.glitch && r.pred.glitch).length;
      const tn = rs.filter((r) => !r.truth.glitch && !r.pred.glitch).length;
      const fp = rs.filter((r) => !r.truth.glitch && r.pred.glitch).length;
      const fn = rs.filter((r) => r.truth.glitch && !r.pred.glitch).length;
      Object.assign(s, {
        accuracy: pct(tp + tn, rs.length),
        glitchesCaught: pct(tp, tp + fn),
        falseAlarms: pct(fp, fp + tn),
        tp, tn, fp, fn,
      });
    } else {
      for (const t of NIAH_TOLERANCES) s[`accuracyWithin${t}s`] = pct(rs.filter((r) => r.score.within?.[`${t}s`]).length, rs.length);
      const errs = rs.map((r) => r.score.absErr).filter((e) => e !== null);
      s.saidGlitch = pct(rs.filter((r) => r.pred.glitch).length, rs.length);
      s.medianErrorSec = errs.length ? errs.sort((a, b) => a - b)[Math.floor(errs.length / 2)] : null;
    }
    // Repeat consistency: share of clips where every repeat gave the same answer.
    const byClip = new Map();
    for (const r of rs) byClip.set(r.id, [...(byClip.get(r.id) || []), r]);
    const multi = [...byClip.values()].filter((v) => v.length > 1);
    if (multi.length) {
      const key = (r) => (task === 'glitch' ? String(r.pred.glitch) : String(r.score.within?.['5s']));
      s.repeatAgreement = pct(multi.filter((v) => new Set(v.map(key)).size === 1).length, multi.length);
    }
    out[setup] = s;
  }
  return out;
}

const pct = (a, b) => (b ? Number(((100 * a) / b).toFixed(1)) : null);

// ---------- I/O ----------

/** fetch with retries on network errors, 429 and 5xx (Hugging Face returns occasional 502s). */
async function fetchRetry(url, tries = 6) {
  for (let i = 0; ; i++) {
    try {
      const res = await fetch(url);
      if (res.ok || (res.status < 500 && res.status !== 429) || i >= tries - 1) return res;
    } catch (err) {
      if (i >= tries - 1) throw err;
    }
    await new Promise((r) => setTimeout(r, Math.min(30_000, 1000 * 2 ** i)));
  }
}

async function fetchRows() {
  const cache = path.join(BENCH_DIR, 'index.json');
  if (fs.existsSync(cache)) return JSON.parse(fs.readFileSync(cache, 'utf8'));
  const rows = [];
  for (let offset = 0; ; offset += 100) {
    const res = await fetchRetry(`${ROWS_API}&offset=${offset}&length=100`);
    if (!res.ok) throw new Error(`Dataset index fetch failed: HTTP ${res.status}`);
    const page = await res.json();
    rows.push(...page.rows.map((r) => r.row));
    if (rows.length >= page.num_rows_total || page.rows.length === 0) break;
  }
  fs.mkdirSync(BENCH_DIR, { recursive: true });
  fs.writeFileSync(cache, JSON.stringify(rows));
  return rows;
}

/** Runs fn once per key at a time, so parallel setups on the same clip share one download/upload. */
const inflight = new Map();
function once(key, fn) {
  if (!inflight.has(key)) inflight.set(key, fn().finally(() => inflight.delete(key)));
  return inflight.get(key);
}

/** Repo path of a row's video: media_path ./video/<task>/<id>/x.mp4 lives at videos/<id>/x.mp4. */
const repoVideoPath = (row) => `videos/${row.custom_id}/${path.basename(row.media_path[0])}`;

function ensureVideo(row) {
  return once(`video:${row.custom_id}`, () => downloadVideo(row));
}

async function downloadVideo(row) {
  const file = path.join(BENCH_DIR, 'videos', `${row.custom_id}.mp4`);
  if (fs.existsSync(file)) return file;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const res = await fetchRetry(FILE_BASE + repoVideoPath(row));
  if (!res.ok) throw new Error(`Video download failed for ${row.custom_id}: HTTP ${res.status}`);
  await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(`${file}.part`));
  fs.renameSync(`${file}.part`, file);
  return file;
}

function loadJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return fallback;
  }
}

/** Uploads each clip once and reuses the handle until it expires. */
function ensureUpload(row, file, fake) {
  return once(`upload:${row.custom_id}`, () => uploadOnce(row, file, fake));
}

async function uploadOnce(row, file, fake) {
  const cacheFile = path.join(BENCH_DIR, 'uploads.json');
  const cache = loadJson(cacheFile, {});
  const hit = cache[row.custom_id];
  if (!fake && hit && Date.now() - hit.ts < UPLOAD_TTL_MS) return hit.uri;
  const up = await uploadMediaFile({ filePath: file, mimeType: 'video/mp4', pollTimeoutMs: 600_000 });
  if (!fake) {
    const fresh = loadJson(cacheFile, {});
    fresh[row.custom_id] = { uri: up.uri, ts: Date.now() };
    fs.writeFileSync(cacheFile, JSON.stringify(fresh, null, 2));
  }
  return up.uri;
}

/** Runs our frame-diff check over the clip (4 fps, small frames) and returns the top moments. */
function frameDiffCandidates(row, file) {
  return once(`framediff:${row.custom_id}`, () => computeFrameDiff(row, file));
}

async function computeFrameDiff(row, file) {
  const cacheFile = path.join(BENCH_DIR, 'framediff', `${row.custom_id}.json`);
  const cached = loadJson(cacheFile, null);
  if (cached) return cached.times;
  const { analyzeFrameSequence } = await import('../loop/capture.mjs');
  const fps = 4;
  const dir = fs.mkdtempSync(path.join(BENCH_DIR, 'tmp-frames-'));
  try {
    execFileSync('ffmpeg', ['-v', 'error', '-i', file, '-vf', `fps=${fps},scale=320:-2`, path.join(dir, 'f-%04d.png')]);
    const frames = fs.readdirSync(dir).filter((f) => f.endsWith('.png')).sort().map((f) => path.join(dir, f));
    const res = await analyzeFrameSequence(frames, { fps, grid: 8 });
    const times = topSpikeTimes(res.spikes, 3);
    fs.mkdirSync(path.dirname(cacheFile), { recursive: true });
    fs.writeFileSync(cacheFile, JSON.stringify({ times, spikeCount: res.spikes.length }));
    return times;
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

// ---------- model calls ----------

function fakeAnswer(task, row, role) {
  const h = parseInt(sha(`${row.custom_id}:${role}`).slice(0, 8), 16);
  const truth = parseTruth(row);
  const right = h % 10 < 7; // fake judges are right ~70% of the time
  return {
    reasoning: 'fake',
    glitch_detected: right ? truth.glitch : !truth.glitch,
    ...(task === 'niah' ? { timestamp: right ? truth.timestamp : (h % 45) + 1 } : {}),
  };
}

async function ask(ctx, row, uri, role, extraText = '') {
  const { task, opts, state, tracePath, fake } = ctx;
  const res = await callModel({
    role,
    model: opts.model,
    thinking: opts.thinking,
    taskId: `vgqa:${task}:${row.custom_id}:${role}`,
    input: toWireParts([
      { type: 'video', uri, mime_type: 'video/mp4', resolution: opts.resolution },
      { type: 'text', text: row.question + extraText },
    ]),
    schema: SCHEMAS[task],
    maxRetries: 5,
    retryBaseMs: 2000,
    pollTimeoutMs: 300_000,
    budgetUsd: opts.budget,
    state,
    tracePath,
    rootDir: ROOT,
    ...(fake ? { fakeHandler: async () => ({ output: fakeAnswer(task, row, role), inputTokens: 9000, outputTokens: 600 }) } : {}),
  });
  return { role, answer: parseAnswer(res.output, res.rawText), costUsd: res.costUsd, interactionId: res.interactionId };
}

async function runSetup(ctx, setup, row) {
  const { task, fake } = ctx;
  if (setup === 'framediff') {
    if (fake) return { pred: { glitch: true, timestamp: parseTruth(row).timestamp }, calls: [] };
    const times = await frameDiffCandidates(row, await ensureVideo(row));
    return { pred: { glitch: times.length > 0, timestamp: times[0] ?? null }, calls: [], hints: times };
  }
  const file = fake ? `${row.custom_id}.mp4` : await ensureVideo(row);
  const uri = await ensureUpload(row, file, fake);
  if (setup === 'baseline') {
    const c = await ask(ctx, row, uri, 'baseline');
    return { pred: c.answer, calls: [c] };
  }
  // loop: two judges, a tie-break only when they disagree, majority vote
  let hint = '';
  let hints;
  if (task === 'niah' && !fake) {
    hints = await frameDiffCandidates(row, file);
    if (hints.length) {
      hint =
        `\n\nAn automatic frame-difference check flagged sudden visual changes at about ${hints.map((t) => `${t}s`).join(', ')}. ` +
        'Look closely at those moments first, but they can also be normal cuts or camera moves: report the glitch you actually see. ' +
        'Give "timestamp" in whole seconds from the start of the video.';
    }
  }
  const [a, b] = await Promise.all([ask(ctx, row, uri, 'judge_a', hint), ask(ctx, row, uri, 'judge_b', hint)]);
  const calls = [a, b];
  if (!judgesAgree(a.answer, b.answer, task)) calls.push(await ask(ctx, row, uri, 'judge_tiebreak', hint));
  return { pred: majority(calls.map((c) => c.answer)), calls, hints };
}

// ---------- runner ----------

async function pool(items, n, fn) {
  let i = 0;
  await Promise.all(
    Array.from({ length: Math.min(n, items.length) }, async () => {
      while (i < items.length) await fn(items[i++]);
    })
  );
}

function readResults(file) {
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
}

export async function runBenchmark(opts, { rows = null, log = console.log } = {}) {
  const fake = process.env.ORCH_FAKE === '1';
  const runDir = path.join(BENCH_DIR, opts.run);
  const metaFile = path.join(runDir, 'run.json');
  if (fs.existsSync(metaFile)) Object.assign(opts, { ...loadJson(metaFile, {}), ...opts.explicit });
  const { task } = opts;
  if (!TASKS[task]) throw new Error(`--task must be one of: ${Object.keys(TASKS).join(', ')}`);
  const setups = opts.setups.filter((s) => s !== 'framediff' || task === 'niah');

  const allRows = rows || (await fetchRows());
  const sample = pickSample(allRows, task, opts.limit);
  const jobs = [];
  for (let rep = 1; rep <= opts.repeats; rep++)
    for (const setup of setups) for (const row of sample) if (setup !== 'framediff' || rep === 1) jobs.push({ setup, row, rep });

  const estPerCall = task === 'niah' ? 0.05 : 0.025; // rough, Argon pricing; video length dominates
  const modelCalls = jobs.filter((j) => j.setup !== 'framediff').reduce((n, j) => n + (j.setup === 'loop' ? 2.3 : 1), 0);
  log(`VideoGameQA-Bench ${task}: ${sample.length} clips x setups [${setups}] x ${opts.repeats} repeat(s) = ${jobs.length} jobs`);
  log(`Rough estimate: ~${Math.round(modelCalls)} model calls, ~$${(modelCalls * estPerCall).toFixed(0)} (cap $${opts.budget}); download ~${((sample.length * (task === 'niah' ? 48 : 18)) / 1000).toFixed(1)} GB (cached)`);
  if (opts.dryRun) return null;

  fs.mkdirSync(runDir, { recursive: true });
  const { explicit, ...persist } = opts;
  fs.writeFileSync(metaFile, JSON.stringify(persist, null, 2));
  const resultsFile = path.join(runDir, 'results.jsonl');
  const done = new Set(readResults(resultsFile).filter((r) => !r.error).map((r) => `${r.setup}|${r.id}|${r.rep}`));
  const todo = jobs.filter((j) => !done.has(`${j.setup}|${j.row.custom_id}|${j.rep}`));
  if (done.size) log(`Resuming: ${done.size} already scored, ${todo.length} to go`);

  const ctx = { task, opts, fake, state: { totalCostUsd: 0, totalCalls: 0 }, tracePath: path.join(runDir, 'trace.jsonl') };
  let n = 0;
  let stop = false;
  await pool(todo, opts.concurrency, async ({ setup, row, rep }) => {
    if (stop) return;
    const truth = parseTruth(row);
    let line;
    try {
      const r = await runSetup(ctx, setup, row);
      line = { setup, id: row.custom_id, rep, truth, pred: r.pred, score: scoreOne(task, truth, r.pred), hints: r.hints, costUsd: r.calls.reduce((s, c) => s + (c.costUsd || 0), 0), calls: r.calls };
      if (!r.pred) line.error = 'unparseable answer';
    } catch (err) {
      if (err.code === 'SPEND_CAP_EXCEEDED') stop = true;
      line = { setup, id: row.custom_id, rep, truth, error: String(err.message || err).slice(0, 300) };
    }
    fs.appendFileSync(resultsFile, JSON.stringify(line) + '\n');
    n += 1;
    log(`[${n}/${todo.length}] ${setup} ${row.custom_id.slice(0, 8)} rep${rep}: ${line.error ? `ERROR ${line.error}` : line.score.correct ? 'correct' : 'wrong'} | spent $${ctx.state.totalCostUsd.toFixed(2)}`);
  });
  if (stop) log(`Stopped: budget cap $${opts.budget} reached. Re-run with --run ${opts.run} --budget <higher> to continue.`);

  return writeSummary(opts, runDir, readResults(resultsFile), log);
}

function writeSummary(opts, runDir, results, log) {
  // Only the latest line per job counts (errored lines are retried on resume).
  const latest = new Map();
  for (const r of results) latest.set(`${r.setup}|${r.id}|${r.rep}`, r);
  const summary = { task: opts.task, model: opts.model, thinking: opts.thinking, limit: opts.limit, repeats: opts.repeats, published: TASKS[opts.task].published, setups: summarize(opts.task, [...latest.values()]) };
  fs.writeFileSync(path.join(runDir, 'summary.json'), JSON.stringify(summary, null, 2));

  const metric = opts.task === 'glitch' ? 'accuracy' : 'accuracyWithin5s';
  const lines = [
    `# VideoGameQA-Bench: ${TASKS[opts.task].category}`,
    '',
    `Model ${opts.model} (thinking ${opts.thinking}), ${opts.limit || 'all'} clips, ${opts.repeats} repeat(s).`,
    '',
    '| Setup | Score | Details | Clips | Errors | Cost |',
    '|---|---|---|---|---|---|',
    ...Object.entries(summary.setups).map(([k, s]) => {
      const details =
        opts.task === 'glitch'
          ? `caught ${s.glitchesCaught}% of glitches, ${s.falseAlarms}% false alarms`
          : `within 1s ${s.accuracyWithin1s}%, 2s ${s.accuracyWithin2s}%, median error ${s.medianErrorSec ?? 'n/a'}s`;
      const rep = s.repeatAgreement !== undefined ? `, repeats agree ${s.repeatAgreement}%` : '';
      return `| ${k} | ${s[metric]}% | ${details}${rep} | ${s.n} | ${s.errors} | $${s.costUsd} |`;
    }),
    '',
    `Score = ${opts.task === 'glitch' ? 'accuracy (balanced glitch/clean)' : 'glitch time within 5s of the truth'}.`,
    `Published (paper, older models): ${Object.entries(summary.published).map(([m, v]) => `${m} ${v}%`).join(', ')}.`,
  ];
  fs.writeFileSync(path.join(runDir, 'summary.md'), lines.join('\n') + '\n');
  log('\n' + lines.join('\n'));
  return summary;
}

// ---------- CLI ----------

export function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) args[key] = true;
    else args[key] = argv[++i];
  }
  const explicit = {};
  const set = (k, v) => {
    if (args[k] !== undefined) explicit[k === 'dry-run' ? 'dryRun' : k] = v;
  };
  set('task', args.task);
  set('limit', Number(args.limit));
  set('repeats', Number(args.repeats));
  set('setups', String(args.setups || '').split(',').filter(Boolean));
  set('concurrency', Number(args.concurrency));
  set('budget', Number(args.budget));
  set('model', args.model);
  set('thinking', args.thinking);
  set('resolution', args.resolution);
  const defaults = { task: 'niah', limit: 20, repeats: 1, setups: ['baseline', 'loop', 'framediff'], concurrency: 2, budget: 25, model: 'gemini-4-argon', thinking: 'high', resolution: 'high' };
  const opts = { ...defaults, ...explicit, dryRun: Boolean(args['dry-run']) };
  opts.run = args.run || `${opts.task}-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}`;
  opts.explicit = explicit;
  return opts;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  runBenchmark(parseArgs(process.argv.slice(2))).catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
