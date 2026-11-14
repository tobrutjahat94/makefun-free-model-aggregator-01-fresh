# free-model-aggregator-01

One server, two APIs, ten free providers. It serves the OpenAI Chat Completions API
and the Anthropic Messages API — in JSON and in server-sent events — over the ten
simulated free-tier providers a trial gives it, routing each request to the best free
model that can serve it and failing over when a provider refuses.

## Run it

```sh
node main.ts            # inside a trial: reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog document and exit 0
```

No dependencies beyond the Node 22 standard library, and nothing is installed before
the server runs.

## How it routes

The brief carries the provider directory, the sealed quality table and the twenty
requests. At startup the server probes every provider's catalog route and computes
the routing plan once, from the brief alone, so a replay makes the same choices:

- Requests are planned heaviest first. Each takes the best free model that meets its
  requirements on a provider that still has planned capacity.
- Planned capacity is the provider's minimum per-trial budget with two calls held
  back, because every provider refuses one or two calls on its own failure schedule
  and those refusals spend budget.
- An OpenAI client asking for JSON never goes to an anthropic-wire provider: that
  wire cannot carry `response_format`, and the provider would echo it absent. An
  Anthropic client carries no JSON-mode field, so no wire is excluded for its JSON
  requests; the model still has to be JSON-capable.

At run time the plan is only the first choice. On a refusal (402, 429 or 5xx) the
request fails over down the same ranked candidate list, retrying a provider once
before moving on, and a provider that refuses twice in a row is left alone for the
rest of the trial.

## How it translates

Every request is passed through in the provider's wire untouched: the system prompt,
the user text (images included), the tools, the stop sequences, the max_tokens and
the JSON mode. The provider echoes what it parsed, and the client compares that echo
with what it sent, so a field rewritten or dropped would lose the request. The
provider's answer comes back in the wire the client used, streamed or not as asked,
with the provider's text whole, its usage counts as that wire reports them, and its
stop reason mapped to the client's normal stop.

## Layout

- `main.ts` — bootstrap, registration, the HTTP server and the work loop.
- `lib/providers.ts` — the brief, the catalog probe and the catalog document.
- `lib/plan.ts` — the routing plan and the failover ladder.
- `lib/wire.ts` — the client wires and the three upstream wires.

## Practice trials

`./run-trial.sh <n>` runs sealed practice trial `n` (0, 1 or 2) against the kit's
env and scorer and prints the score. It needs the kit at
`$MAKEFUN_KIT` (default: the free-model-aggregator-01 kit in this workspace).
