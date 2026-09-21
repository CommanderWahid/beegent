/**
 * The pipeline as five steps, derived from the log lines the run already streams.
 *
 * Deliberately read off the log rather than pushed from the server: every line arrives over the
 * existing SSE stream, so this needs no new endpoint and no new protocol. The cost is that it
 * depends on log WORDING - reword a `_log.info` prefix in beegent/ and a step stops lighting up.
 * The patterns below are the whole of that coupling; keep them together.
 */

export type StageState = 'todo' | 'active' | 'done';

export interface Stage {
  id: string;
  name: string;
  /** An LLM role, so it thinks and it spends tokens. The others are plain deterministic code. */
  agent: boolean;
  state: StageState;
  /** A short measured aside, e.g. "route 2 of 3". Empty when there is nothing to say. */
  detail: string;
}

/** In the order a run performs them: the catalog is consulted before any model is called. */
const ORDER: { id: string; name: string; agent: boolean }[] = [
  { id: 'catalog', name: 'Catalog', agent: false },
  { id: 'plan', name: 'Plan', agent: true },
  { id: 'geofetch', name: 'Geofetch', agent: true },
  { id: 'gate', name: 'Gate', agent: false },
  { id: 'critic', name: 'Critic', agent: true },
];

/** First match wins, so the specific geofetch patterns must precede the general one. */
const MATCH: [RegExp, string][] = [
  [/^\[(catalog|memory)\]/, 'catalog'],
  [/^\[plan\]/, 'plan'],
  [/^\[step \d+\]/, 'geofetch'],
  [/^(fetch_page|web_search|probe_url)\b/, 'geofetch'],
  [/^\[geofetch\]/, 'geofetch'],
  [/^\[gate\]/, 'gate'],
  [/^\[critic\]/, 'critic'],
];

/** Extra detail for the steps that have something worth counting. */
const DETAIL: [RegExp, (m: RegExpMatchArray) => string][] = [
  [/^\[geofetch\] angle (\d+)\/(\d+)/, (m) => `route ${m[1]} of ${m[2]}`],
  [/^\[plan\] (\d+) angle/, (m) => `${m[1]} route(s)`],
];

export function initialStages(): Stage[] {
  return ORDER.map((s) => ({ ...s, state: 'todo' as StageState, detail: '' }));
}

/**
 * Fold one log line into the steps.
 *
 * Only the step that was running is marked done - a step that never ran stays `todo` rather than
 * being back-filled, so a catalog hit that answers outright does not pretend the planner ran.
 */
export function advanceStages(stages: Stage[], raw: string): Stage[] {
  const text = raw.trim();

  // A REPLAN starts the pipeline over, so the steps reset rather than growing a second column.
  // Iteration 1 is not a replan: resetting there would wipe the catalog step that just ran.
  const iteration = text.match(/^=== iteration (\d+)/);
  if (iteration) return Number(iteration[1]) > 1 ? initialStages() : stages;

  const hit = MATCH.find(([pattern]) => pattern.test(text));
  if (!hit) return stages;
  const at = stages.findIndex((s) => s.id === hit[1]);
  if (at < 0) return stages;

  let detail = '';
  for (const [pattern, format] of DETAIL) {
    const found = text.match(pattern);
    if (found) { detail = format(found); break; }
  }

  return stages.map((s, i) => {
    if (i === at) return { ...s, state: 'active', detail: detail || s.detail };
    return s.state === 'active' ? { ...s, state: 'done' } : s;
  });
}

/** The run is over: whatever was in flight finished. */
export function finishStages(stages: Stage[]): Stage[] {
  return stages.map((s) => (s.state === 'active' ? { ...s, state: 'done' } : s));
}
