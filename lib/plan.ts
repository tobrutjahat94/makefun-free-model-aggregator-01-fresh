// The routing plan: which free model serves each request, and in what order to fail over.
//
// Quality is 60% of the score: for each request, the sealed quality table's rating of the
// free model that answered it, over the best rating for a free model meeting that request's
// requirements, weighted by the request's weight. The plan is computed once, at startup, from
// the brief alone — the manifest (every request's requirements and weight, in order) and the
// quality table — so it is the same on a replay.
//
// The greedy rule: take the requests heaviest first, and give each the best free model that
// meets its requirements on a provider that still has planned capacity. Capacity is the
// provider's minimum per-trial budget with two calls held back: every provider refuses one or
// two calls on its own failure schedule, and those refusals spend budget, so the plan never
// books a provider to its last call. The minimum budgets are the sealed generator's
// calls_min (program/generator/draw.mjs): groq 9, cerebras 5, gemini 6, zai 4, openrouter 10,
// sambanova 4, pollinations 10, cloudflare 7, ovh 5, kilol 12. A provider the table does not
// name gets a capacity of 2.
//
// Two wire limits shape the candidate list, both from the trial's rules:
//   - an OpenAI JSON-mode request carries response_format, which only the openai wire (as
//     response_format) and the gemini wire (as responseMimeType) can carry; an anthropic-wire
//     provider echoes it absent and the request fails its wire checks. So for an OpenAI client
//     asking for JSON, anthropic-wire providers are not candidates at all.
//   - an Anthropic client carries no JSON-mode field of its own, so no wire is excluded for
//     its JSON requests; the model still has to be JSON-capable.
//
// At run time the plan is only the first choice: when a provider refuses (402, 429 or 5xx),
// the request fails over down the same ranked candidate list, retrying a provider once before
// moving on (a refusal is usually one scheduled call, not a dead provider), and a provider
// that refuses twice in a row is left alone for the rest of the trial. The plan's reserve is
// what keeps the first choices inside budget; failover is what keeps the requests served.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Probe,
  type ProbedModel,
  type RatingsFile,
  type Requirement,
  type WorkloadRequest,
} from "./providers.ts";

/** One (provider, model) that can serve a request, with the quality table's rating for it. */
export interface Candidate {
  provider: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The minimum per-trial budget of each provider, from the sealed generator's calls_min, with
 * two calls of each held back for that provider's own failure schedule.
 */
const MIN_BUDGET: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const FAIL_RESERVE = 2;

function plannedCapacity(providerId: string): number {
  const min = MIN_BUDGET[providerId] ?? 4;
  return Math.max(0, min - FAIL_RESERVE);
}

const byCandidate = (a: Candidate, b: Candidate): number =>
  b.rating - a.rating ||
  (a.provider.provider.id < b.provider.provider.id ? -1 : a.provider.provider.id > b.provider.provider.id ? 1 : 0) ||
  (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0);

/**
 * Every free model that can serve one request, best first. `clientApi` is the wire the client
 * used: an OpenAI client asking for JSON excludes anthropic-wire providers, which cannot carry
 * response_format.
 */
export function candidatesFor(
  requires: Requirement,
  clientApi: "openai" | "anthropic",
  ratings: RatingsFile,
  probes: readonly Probe[],
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientApi === "openai" && requires.json === true && probe.provider.wire === "anthropic") continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe, model, rating: row?.ratingMilli ?? 0 });
    }
  }
  return out.sort(byCandidate);
}

/**
 * The static plan: the first-choice (provider, model) for each request id. Requests are
 * planned heaviest first, then in manifest order, each taking the best candidate whose
 * provider still has planned capacity. A request with no candidate gets no plan entry and is
 * served by failover alone (which, with no candidates, cannot serve it either).
 */
export function computePlan(
  manifest: readonly WorkloadRequest[],
  ratings: RatingsFile,
  probes: readonly Probe[],
): Map<string, Candidate> {
  const remaining = new Map<string, number>();
  for (const probe of probes) remaining.set(probe.provider.id, plannedCapacity(probe.provider.id));
  const plan = new Map<string, Candidate>();
  const ordered = [...manifest].sort(
    (a, b) => b.weight - a.weight || manifest.indexOf(a) - manifest.indexOf(b),
  );
  for (const request of ordered) {
    const candidates = candidatesFor(request.requires, request.api, ratings, probes);
    for (const candidate of candidates) {
      const left = remaining.get(candidate.provider.provider.id) ?? 0;
      if (left <= 0) continue;
      remaining.set(candidate.provider.provider.id, left - 1);
      plan.set(request.id, candidate);
      break;
    }
  }
  return plan;
}

/**
 * The attempt order for one request: its planned candidate first, then every candidate in
 * ranked order, each once. The plan is the first choice; the rest is the failover ladder.
 */
export function attemptOrder(planned: Candidate | undefined, ranked: Candidate[]): Candidate[] {
  const out: Candidate[] = [];
  const seen = new Set<string>();
  const push = (c: Candidate): void => {
    const key = `${c.provider.provider.id}/${c.model.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(c);
  };
  if (planned !== undefined) push(planned);
  for (const candidate of ranked) push(candidate);
  return out;
}
