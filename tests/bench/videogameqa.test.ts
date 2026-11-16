import { describe, it, expect, beforeAll } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'vgqa-test-'));
process.env.VGQA_DIR = tmp;

let mod: Record<string, any>;
beforeAll(async () => {
  mod = (await import('../../scripts/bench/videogameqa.mjs')) as Record<string, any>;
});

const glitchRow = (id: string, glitch: boolean) => ({
  custom_id: id,
  question_categories: ['Video Glitch Detection'],
  question: 'Is there a glitch?',
  ground_truth: JSON.stringify({ glitch_detected: glitch }),
  media_path: [`./video/video_glitch_detection/${id}/question_video_1.mp4`],
  media_type: 'videos',
});
const niahRow = (id: string, t: number) => ({
  custom_id: id,
  question_categories: ['Needle in a Haystack'],
  question: 'Find the glitch.',
  ground_truth: JSON.stringify({ glitch_detected: true, timestamp: t }),
  media_path: [`./video/niah/${id}/question_video_1.mp4`],
  media_type: 'videos',
});

describe('VideoGameQA-Bench runner', () => {
  it('parses answers from parsed objects, JSON in text, and rejects junk', () => {
    expect(mod.parseAnswer({ glitch_detected: true, timestamp: 12 })).toMatchObject({ glitch: true, timestamp: 12 });
    expect(mod.parseAnswer({ files: [], notes: 'x' }, '```json\n{"glitch_detected": "false"}\n```')).toMatchObject({ glitch: false });
    expect(mod.parseAnswer({ notes: 'no json here' })).toBeNull();
  });

  it('samples glitch clips balanced and deterministically', () => {
    const rows = [...Array(10)].map((_, i) => glitchRow(`g${i}`, i < 7));
    const a = mod.pickSample(rows, 'glitch', 4);
    expect(a.filter((r: any) => JSON.parse(r.ground_truth).glitch_detected)).toHaveLength(2);
    expect(mod.pickSample(rows, 'glitch', 4).map((r: any) => r.custom_id)).toEqual(a.map((r: any) => r.custom_id));
  });

  it('majority vote and agreement follow the loop rules', () => {
    const y = (t: number) => ({ glitch: true, timestamp: t });
    const n = { glitch: false, timestamp: null };
    expect(mod.majority([y(10), n, y(14)])).toEqual({ glitch: true, timestamp: 10 });
    expect(mod.majority([n, n, y(3)])).toMatchObject({ glitch: false });
    expect(mod.judgesAgree(y(10), y(11), 'niah')).toBe(true);
    expect(mod.judgesAgree(y(10), y(20), 'niah')).toBe(false);
    expect(mod.judgesAgree(y(10), y(20), 'glitch')).toBe(true);
  });

  it('collapses frame-diff spikes into distinct top moments', () => {
    const spikes = [
      { frameIndex: 40, t: 10.0, maxCellFrac: 0.5, d: 0.1 },
      { frameIndex: 41, t: 10.25, maxCellFrac: 0.4, d: 0.1 },
      { frameIndex: 80, t: 20.0, maxCellFrac: 0.3, d: 0.1 },
    ];
    expect(mod.topSpikeTimes(spikes, 3)).toEqual([10, 20]);
  });

  it('scores niah within tolerances', () => {
    const s = mod.scoreOne('niah', { glitch: true, timestamp: 20 }, { glitch: true, timestamp: 23 });
    expect(s).toMatchObject({ correct: true, absErr: 3, within: { '1s': false, '2s': false, '5s': true } });
    expect(mod.scoreOne('niah', { glitch: true, timestamp: 20 }, { glitch: false, timestamp: null }).correct).toBe(false);
  });

  it('runs end to end offline, writes a summary, and resumes without redoing work', async () => {
    process.env.ORCH_FAKE = '1';
    const rows = [...[...Array(6)].map((_, i) => glitchRow(`g${i}`, i % 2 === 0)), ...[...Array(3)].map((_, i) => niahRow(`n${i}`, 10 + i))];
    const opts = mod.parseArgs(['--task', 'glitch', '--limit', '6', '--repeats', '2', '--run', 'fake-glitch']);
    const summary = await mod.runBenchmark(opts, { rows, log: () => {} });
    expect(Object.keys(summary.setups).sort()).toEqual(['baseline', 'loop']); // framediff is niah-only
    expect(summary.setups.baseline.n).toBe(12);
    expect(summary.setups.baseline.repeatAgreement).toBeDefined();
    expect(fs.existsSync(path.join(tmp, 'fake-glitch', 'summary.md'))).toBe(true);

    const lines = fs.readFileSync(path.join(tmp, 'fake-glitch', 'results.jsonl'), 'utf8').trim().split('\n');
    const logs: string[] = [];
    await mod.runBenchmark(mod.parseArgs(['--run', 'fake-glitch']), { rows, log: (m: string) => logs.push(m) });
    expect(fs.readFileSync(path.join(tmp, 'fake-glitch', 'results.jsonl'), 'utf8').trim().split('\n')).toHaveLength(lines.length);
    expect(logs.join('\n')).toContain('Resuming');

    const niah = await mod.runBenchmark(mod.parseArgs(['--task', 'niah', '--limit', '3', '--run', 'fake-niah']), { rows, log: () => {} });
    expect(Object.keys(niah.setups).sort()).toEqual(['baseline', 'framediff', 'loop']);
    expect(niah.setups.loop.accuracyWithin5s).not.toBeNull();
  });
});
