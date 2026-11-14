// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry serves the OpenAI Chat Completions API and the Anthropic Messages API over the
// ten simulated free-tier providers the trial gives it, in JSON and in server-sent events,
// and registers the catalog it read from them. Run it with no arguments inside a trial; run
// it with --list to print the catalog document and exit.
//
// A trial works like this: the sandbox sets MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE and
// MAKEFUN_TRIAL_SEED. The brief names the ten providers, the sealed quality table and the
// twenty requests. This server probes every provider's catalog route, computes the routing
// plan once from the brief alone, listens on a port the operating system picks, registers
// that port and its catalog with the env, and then answers the requests the env's client
// sends, one at a time, each with the best free model that can serve it, failing over when
// a provider refuses. While the client works it asks the env whether the workload is done,
// and when the env says it is, it exits 0.
//
// Every choice is a function of the brief, the requests and the providers' answers: no wall
// clock and no unseeded randomness, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import {
  buildCatalog,
  freeModelIds,
  parseBrief,
  probeAll,
  type Brief,
  type CatalogDoc,
  type Probe,
  type RatingsFile,
  type WorkloadRequest,
} from "./lib/providers.ts";
import { attemptOrder, candidatesFor, computePlan, type Candidate } from "./lib/plan.ts";
import {
  anthropicResponse,
  callProvider,
  openAIResponse,
  type CallOutcome,
} from "./lib/wire.ts";

const PROTOCOL_VERSION = 1;

/** Read the brief the trial hands this server, or null outside a trial. */
function readBrief(): Brief | null {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (file === undefined || file === "") return null;
  try {
    return parseBrief(readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

function envUrl(): string {
  return (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
}

async function postJson(url: string, body: unknown): Promise<{ status: number; text: string }> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return { status: res.status, text: await res.text() };
}

async function getJson(url: string): Promise<unknown | null> {
  try {
    const res = await fetch(url);
    if (res.status < 200 || res.status >= 300) return null;
    return (await res.json()) as unknown;
  } catch {
    return null;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function jsonBody(res: ServerResponse, status: number, value: unknown): void {
  const body = JSON.stringify(value);
  res.writeHead(status, { "content-type": "application/json", "content-length": String(Buffer.byteLength(body)) });
  res.end(body);
}

function readRequestBody(req: IncomingMessage): Promise<string | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 2 * 1024 * 1024) chunks.push(chunk);
    });
    req.on("end", () => resolve(size > 2 * 1024 * 1024 ? null : Buffer.concat(chunks).toString("utf8")));
    req.on("error", () => resolve(null));
  });
}

/**
 * The server's whole trial state: the brief, the probes, the catalog, the plan, and the
 * per-provider refusal record that failover reads. A provider that refuses twice in a row
 * is left alone for the rest of the trial.
 */
class Trial {
  readonly brief: Brief;
  readonly probes: Probe[];
  readonly ratings: RatingsFile;
  readonly catalog: CatalogDoc;
  readonly plan: Map<string, Candidate>;
  readonly consecutiveFails = new Map<string, number>();
  readonly dead = new Set<string>();
  served = 0;

  constructor(brief: Brief, probes: Probe[]) {
    this.brief = brief;
    this.probes = probes;
    this.ratings = brief.ratings;
    this.catalog = buildCatalog(probes);
    this.plan = computePlan(brief.manifest, brief.ratings, probes);
  }

  /** Which request the env's client is sending next, or null when it cannot be asked. */
  async nextRequestId(): Promise<string | null> {
    const work = await getJson(`${envUrl()}/work`);
    if (work !== null && typeof work === "object" && !Array.isArray(work)) {
      const next = (work as Record<string, unknown>).next;
      if (typeof next === "string") return next;
    }
    return null;
  }

  requestFor(id: string | null): WorkloadRequest | null {
    if (id !== null) {
      for (const request of this.brief.manifest) if (request.id === id) return request;
    }
    return this.brief.manifest[this.served] ?? null;
  }

  /** Serve one client request: the plan's first choice, then the failover ladder. */
  async serve(
    clientApi: "openai" | "anthropic",
    clientBody: Record<string, unknown>,
  ): Promise<{ status: number; headers: Record<string, string>; body: string }> {
    // The env is the source of truth for which request the client is sending: /work names
    // it, and the manifest order is only a fallback for when /work cannot be reached.
    // Trusting the counter alone would serve the wrong request after any stray POST.
    const nextId = await this.nextRequestId();
    const request = this.requestFor(nextId);
    this.served += 1;
    if (request === null) {
      return { status: 502, headers: { "content-type": "application/json" }, body: JSON.stringify({ error: "no request" }) };
    }
    const ranked = candidatesFor(request.requires, clientApi, this.ratings, this.probes);
    const order = attemptOrder(this.plan.get(request.id), ranked);
    for (const candidate of order) {
      const providerId = candidate.provider.provider.id;
      if (this.dead.has(providerId)) continue;
      for (let attempt = 0; attempt < 2; attempt++) {
        const outcome: CallOutcome = await callProvider(
          envUrl(),
          candidate.provider,
          candidate.model.id,
          clientApi,
          clientBody,
          request.stream,
        );
        if (outcome.ok) {
          this.consecutiveFails.set(providerId, 0);
          const answer = outcome.answer;
          return clientApi === "openai"
            ? openAIResponse(candidate.model.id, answer, request.stream)
            : anthropicResponse(candidate.model.id, answer, request.stream);
        }
        const fails = (this.consecutiveFails.get(providerId) ?? 0) + 1;
        this.consecutiveFails.set(providerId, fails);
        if (outcome.status === 400 || outcome.status === 401) {
          // A bad key or an unknown model is this provider's own fault, not a refusal to
          // retry: leave it alone and move on down the ladder.
          this.dead.add(providerId);
          break;
        }
        if (fails >= 2) {
          this.dead.add(providerId);
          break;
        }
        // Otherwise retry the same provider once: a refusal is usually one scheduled call,
        // not a dead provider.
      }
    }
    return {
      status: 502,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ error: "no provider could serve this request" }),
    };
  }

  /** The model list, in the wire the auth header names, listing exactly the free models. */
  modelsReply(req: IncomingMessage): { status: number; headers: Record<string, string>; body: string } {
    const ids = freeModelIds(this.catalog);
    const authorization = req.headers.authorization;
    if (typeof authorization === "string" && authorization !== "") {
      return {
        status: 200,
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          object: "list",
          data: ids.map((id) => ({ id, object: "model", created: 1767225600, owned_by: "free-model-aggregator-01", free: true })),
        }),
      };
    }
    return {
      status: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        data: ids.map((id) => ({
          type: "model",
          id,
          display_name: id,
          created_at: "2026-01-01T00:00:00Z",
          free: true,
        })),
        has_more: false,
      }),
    };
  }
}

async function register(port: number, catalog: CatalogDoc): Promise<boolean> {
  for (let attempt = 0; attempt < 50; attempt++) {
    try {
      const { status } = await postJson(`${envUrl()}/register`, {
        protocol_version: PROTOCOL_VERSION,
        port,
        catalog,
      });
      if (status >= 200 && status < 300) return true;
    } catch {
      // The env is not listening yet; wait and try again.
    }
    await sleep(100);
  }
  return false;
}

async function workIsDone(): Promise<boolean> {
  const work = await getJson(`${envUrl()}/work`);
  if (work !== null && typeof work === "object" && !Array.isArray(work)) {
    return (work as Record<string, unknown>).done === true;
  }
  return false;
}

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const listOnly = args.includes("--list");
  const brief = readBrief();
  if (brief === null) {
    if (listOnly) process.stdout.write(JSON.stringify({ providers: [] }));
    else process.stderr.write("free-model-aggregator-01: no brief: set MAKEFUN_PROMPT_FILE\n");
    return listOnly ? 0 : 1;
  }
  const url = envUrl();
  if (url === "") {
    if (listOnly) process.stdout.write(JSON.stringify({ providers: [] }));
    else process.stderr.write("free-model-aggregator-01: no env: set MAKEFUN_ENV_URL\n");
    return listOnly ? 0 : 1;
  }
  const probes = await probeAll(url, brief.providers);
  const trial = new Trial(brief, probes);
  if (listOnly) {
    process.stdout.write(JSON.stringify(trial.catalog));
    return 0;
  }
  const server = createServer((req, res) => {
    void (async () => {
      try {
        const method = req.method ?? "GET";
        const path = new URL(req.url ?? "/", "http://entry").pathname;
        if (method === "GET" && path === "/v1/models") {
          const reply = trial.modelsReply(req);
          res.writeHead(reply.status, reply.headers);
          res.end(reply.body);
          return;
        }
        if (method === "POST" && (path === "/v1/chat/completions" || path === "/v1/messages")) {
          const raw = await readRequestBody(req);
          if (raw === null) {
            jsonBody(res, 413, { error: "the request body is too large" });
            return;
          }
          let clientBody: unknown;
          try {
            clientBody = JSON.parse(raw);
          } catch {
            jsonBody(res, 400, { error: "the request body is not JSON" });
            return;
          }
          if (clientBody === null || typeof clientBody !== "object" || Array.isArray(clientBody)) {
            jsonBody(res, 400, { error: "the request body is not one JSON object" });
            return;
          }
          const reply = await trial.serve(
            path === "/v1/chat/completions" ? "openai" : "anthropic",
            clientBody as Record<string, unknown>,
          );
          res.writeHead(reply.status, reply.headers);
          res.end(reply.body);
          return;
        }
        jsonBody(res, 404, { error: `no route ${method} ${path}` });
      } catch (err) {
        try {
          jsonBody(res, 500, { error: err instanceof Error ? err.message : "internal error" });
        } catch {
          // The socket is already gone; there is nothing left to answer.
        }
      }
    })();
  });
  // Listen on a port the operating system picks: the env itself listens on 8080 in
  // process-driver trials, and an ephemeral port is never 8080, so this cannot collide
  // with it. The register call below tells the env the port that was actually picked.
  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen({ port: 0, host: "0.0.0.0" }, () => {
      const address = server.address();
      if (address === null || typeof address === "string") {
        reject(new Error("the entry's listener has no TCP address"));
        return;
      }
      resolve(address.port);
    });
  });
  if (!(await register(port, trial.catalog))) {
    process.stderr.write("free-model-aggregator-01: the env would not take the registration\n");
    server.close();
    return 1;
  }
  // Poll the env until the workload is done, then exit 0. The driver only says done once
  // every request has been sent and answered, so no request can still be in flight.
  const timer = setInterval(() => {
    void workIsDone().then((done) => {
      if (!done) return;
      clearInterval(timer);
      server.close(() => process.exit(0));
      setTimeout(() => process.exit(0), 500).unref();
    });
  }, 100);
  process.on("SIGTERM", () => {
    clearInterval(timer);
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 500).unref();
  });
  return 0;
}

main()
  .then((code) => {
    if (code !== 0) process.exit(code);
  })
  .catch((err: unknown) => {
    process.stderr.write(`free-model-aggregator-01: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
