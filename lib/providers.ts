// free-model-aggregator-01: the brief, the catalog probe, and the catalog document.
//
// A trial hands this server a brief (the path in $MAKEFUN_PROMPT_FILE) carrying three JSON
// documents, found by their `schema` member; the prose around them does not matter:
//   - the provider directory: ten providers, each with an id, a name, a wire, an auth scheme,
//     a base path, a key variable and a key;
//   - the quality table: every free model's rating, context length and capabilities, plus each
//     request's best rating;
//   - the workload manifest: the twenty requests in order, each with its api, its stream flag,
//     its weight and its requirements.
//
// The catalog this server registers is built by probing every provider's own catalog route
// (GET <base>/v1/models, or /v1beta/models on the Gemini wire), with that provider's own
// auth header. The probe answers list free and paid models alike; only the free flag decides
// whether a model may serve a request, and the registered catalog must name every model with
// its true free flag, because the scorer compares it against the providers' true catalogs.

export type Json = unknown;

export function isRecord(value: Json): value is Record<string, Json> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function asString(value: Json): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: Json): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asBool(value: Json, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

/** One provider from the brief's directory. `path` is the provider's own base path. */
export interface ProviderInfo {
  id: string;
  name: string;
  wire: "openai" | "anthropic" | "gemini";
  auth: "bearer" | "x-api-key" | "x-goog-api-key" | "none";
  keyEnv: string;
  key: string;
  path: string;
}

/** One model as a provider's catalog route reports it, with its capabilities. */
export interface ProbedModel {
  id: string;
  free: boolean;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** One provider's probe result: the models its catalog route listed. */
export interface Probe {
  provider: ProviderInfo;
  models: ProbedModel[];
  ok: boolean;
}

export interface RatingRow {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  ratings: RatingRow[];
  best: { id: string; ratingMilli: number }[];
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadRequest {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderInfo[];
  ratings: RatingsFile;
  manifest: WorkloadRequest[];
}

/** The catalog document, exactly as the rules spell it. */
export interface CatalogDoc {
  providers: {
    id: string;
    base_url: string;
    key_env: string;
    wire: "openai" | "anthropic" | "gemini";
    models: { id: string; free: boolean; context_length: number | null }[];
  }[];
}

/**
 * The JSON value starting at `start`, found by balancing braces while honouring strings
 * and escapes. Returns the parsed value and the index just past it, or null.
 */
function jsonValueAt(text: string, start: number): { value: Json; end: number } | null {
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try {
          return { value: JSON.parse(text.slice(start, i + 1)), end: i + 1 };
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
export function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf("{"); start !== -1; start = text.indexOf("{", start + 1)) {
    const found = jsonValueAt(text, start);
    if (found === null) continue;
    if (isRecord(found.value) && typeof found.value.schema === "string") out.push(found.value);
    start = found.end - 1;
  }
  return out;
}

function parseProviders(doc: Json): ProviderInfo[] {
  if (!isRecord(doc) || !Array.isArray(doc.providers)) return [];
  const out: ProviderInfo[] = [];
  for (const raw of doc.providers) {
    if (!isRecord(raw)) continue;
    const id = asString(raw.id);
    const wire = asString(raw.wire);
    const auth = asString(raw.auth);
    if (id === null) continue;
    if (wire !== "openai" && wire !== "anthropic" && wire !== "gemini") continue;
    if (auth !== "bearer" && auth !== "x-api-key" && auth !== "x-goog-api-key" && auth !== "none") continue;
    out.push({
      id,
      name: asString(raw.name) ?? id,
      wire,
      auth,
      keyEnv: asString(raw.key_env) ?? "",
      key: asString(raw.key) ?? "",
      path: asString(raw.path) ?? `/p/${id}`,
    });
  }
  return out;
}

function parseRatings(doc: Json): RatingsFile {
  const ratings: RatingRow[] = [];
  const best: { id: string; ratingMilli: number }[] = [];
  if (!isRecord(doc)) return { ratings, best };
  if (Array.isArray(doc.ratings)) {
    for (const raw of doc.ratings) {
      if (!isRecord(raw)) continue;
      const model = asString(raw.model);
      const provider = asString(raw.provider);
      const ratingMilli = asNumber(raw.rating_milli);
      if (model === null || provider === null || ratingMilli === null) continue;
      ratings.push({
        model,
        provider,
        ratingMilli,
        contextLength: asNumber(raw.context_length),
        tools: asBool(raw.tools, false),
        vision: asBool(raw.vision, false),
        json: asBool(raw.json, false),
        free: asBool(raw.free, false),
      });
    }
  }
  if (Array.isArray(doc.best)) {
    for (const raw of doc.best) {
      if (!isRecord(raw)) continue;
      const id = asString(raw.id);
      const ratingMilli = asNumber(raw.rating_milli);
      if (id !== null && ratingMilli !== null) best.push({ id, ratingMilli });
    }
  }
  return { ratings, best };
}

function parseManifest(doc: Json): WorkloadRequest[] {
  const out: WorkloadRequest[] = [];
  if (!isRecord(doc) || !Array.isArray(doc.manifest)) return out;
  for (const raw of doc.manifest) {
    if (!isRecord(raw)) continue;
    const id = asString(raw.id);
    const api = asString(raw.api);
    if (id === null || (api !== "openai" && api !== "anthropic")) continue;
    const requires: Requirement = {};
    if (isRecord(raw.requires)) {
      if (raw.requires.tools === true) requires.tools = true;
      if (raw.requires.vision === true) requires.vision = true;
      if (raw.requires.json === true) requires.json = true;
      const minContext = asNumber(raw.requires.min_context);
      if (minContext !== null) requires.minContext = minContext;
    }
    out.push({
      id,
      api,
      stream: raw.stream === true,
      weight: asNumber(raw.weight) ?? 1,
      requires,
    });
  }
  return out;
}

/** Split the brief's prose into its three documents and read each in turn. */
export function parseBrief(text: string): Brief {
  const docs = documentsIn(text);
  let providers: ProviderInfo[] = [];
  let ratings: RatingsFile = { ratings: [], best: [] };
  let manifest: WorkloadRequest[] = [];
  for (const doc of docs) {
    if (!isRecord(doc) || typeof doc.schema !== "string") continue;
    if (doc.schema.endsWith("/providers/v1")) providers = parseProviders(doc);
    else if (doc.schema.endsWith("/ratings/v1")) ratings = parseRatings(doc);
    else if (doc.schema.endsWith("/workload/v1")) manifest = parseManifest(doc);
  }
  return { providers, ratings, manifest };
}

/** The auth header a provider's scheme names, or none when it takes no key. */
export function authHeaders(provider: ProviderInfo): Record<string, string> {
  if (provider.auth === "bearer") return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === "x-api-key") return { "x-api-key": provider.key };
  if (provider.auth === "x-goog-api-key") return { "x-goog-api-key": provider.key };
  return {};
}

function catalogModelsOf(provider: ProviderInfo, body: Json): ProbedModel[] {
  const out: ProbedModel[] = [];
  if (!isRecord(body)) return out;
  if (provider.wire === "gemini") {
    if (!Array.isArray(body.models)) return out;
    for (const raw of body.models) {
      if (!isRecord(raw)) continue;
      const name = asString(raw.name);
      if (name === null) continue;
      const id = name.startsWith("models/") ? name.slice("models/".length) : name;
      out.push({
        id,
        free: asBool(raw.free, false),
        contextLength: asNumber(raw.inputTokenLimit) ?? asNumber(raw.context_length),
        tools: asBool(raw.tools, false),
        vision: asBool(raw.vision, false),
        json: asBool(raw.json, false),
      });
    }
    return out;
  }
  if (!Array.isArray(body.data)) return out;
  for (const raw of body.data) {
    if (!isRecord(raw)) continue;
    const id = asString(raw.id);
    if (id === null) continue;
    out.push({
      id,
      free: asBool(raw.free, false),
      contextLength: asNumber(raw.context_length),
      tools: asBool(raw.tools, false),
      vision: asBool(raw.vision, false),
      json: asBool(raw.json, false),
    });
  }
  return out;
}

/** Probe one provider's catalog route. A catalog call spends no budget. */
export async function probeProvider(envUrl: string, provider: ProviderInfo): Promise<Probe> {
  const base = envUrl.replace(/\/+$/, "") + provider.path;
  const url = provider.wire === "gemini" ? `${base}/v1beta/models` : `${base}/v1/models`;
  try {
    const headers: Record<string, string> = authHeaders(provider);
    if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
    const res = await fetch(url, { headers });
    if (res.status < 200 || res.status >= 300) return { provider, models: [], ok: false };
    const body = (await res.json()) as Json;
    return { provider, models: catalogModelsOf(provider, body), ok: true };
  } catch {
    return { provider, models: [], ok: false };
  }
}

/** Probe every provider in the brief's order, one at a time. */
export async function probeAll(envUrl: string, providers: ProviderInfo[]): Promise<Probe[]> {
  const out: Probe[] = [];
  for (const provider of providers) out.push(await probeProvider(envUrl, provider));
  return out;
}

/** The catalog document from the probes, in the brief's provider order. */
export function buildCatalog(probes: readonly Probe[]): CatalogDoc {
  return {
    providers: probes.map((probe) => ({
      id: probe.provider.id,
      base_url: probe.provider.path,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models: probe.models.map((m) => ({ id: m.id, free: m.free, context_length: m.contextLength })),
    })),
  };
}

/** Every free model id the catalog registers, each once, in catalog order. */
export function freeModelIds(catalog: CatalogDoc): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free && !seen.has(model.id)) {
        seen.add(model.id);
        out.push(model.id);
      }
    }
  }
  return out;
}

/** The quality table's row for one provider's model, when the table has one. */
export function ratingOfModel(
  ratings: RatingsFile,
  providerId: string,
  modelId: string,
): RatingRow | null {
  for (const row of ratings.ratings) {
    if (row.provider === providerId && row.model === modelId) return row;
  }
  return null;
}

/** What a model can do: the quality table's row where it has one, else the catalog's flags. */
export function capabilitiesOf(
  model: ProbedModel,
  row: RatingRow | null,
): { tools: boolean; vision: boolean; json: boolean; contextLength: number | null } {
  if (row !== null) {
    return { tools: row.tools, vision: row.vision, json: row.json, contextLength: row.contextLength };
  }
  return { tools: model.tools, vision: model.vision, json: model.json, contextLength: model.contextLength };
}

/** A model is a legal pick only when the catalog says it is free and the table agrees. */
export function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (!model.free) return false;
  if (row !== null && !row.free) return false;
  return true;
}

/** Whether a model's capabilities meet one request's requirements. */
export function meetsRequirements(
  requires: Requirement,
  caps: { tools: boolean; vision: boolean; json: boolean; contextLength: number | null },
): boolean {
  if (requires.tools === true && !caps.tools) return false;
  if (requires.vision === true && !caps.vision) return false;
  if (requires.json === true && !caps.json) return false;
  if (requires.minContext !== undefined) {
    if (caps.contextLength === null || caps.contextLength < requires.minContext) return false;
  }
  return true;
}
