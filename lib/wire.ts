// The client wires and the three upstream wires.
//
// The trial's client calls this server as an OpenAI service (POST /v1/chat/completions) or
// an Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. This
// server answers from a provider on the wire that provider speaks — OpenAI, Anthropic or
// Gemini — and returns the provider's answer in the wire the client used.
//
// The one rule that shapes every translation: pass the request through untouched. The
// provider echoes what it parsed — the system text, the user text, the tool names, the
// max_tokens, the stop sequences, the image and the JSON mode — and the client compares that
// echo with what it sent. A field rewritten or dropped loses the request. So every
// translation below carries the same fields across, only re-expressed in the other wire's
// shape: the system prompt, the user text (images included), the tools, the stop sequences,
// the max_tokens and the JSON mode.
//
// A provider's answer is always plain text with a normal stop: its text is the whole
// `ANSWER ECHO...` line, its usage counts are the wire's own, and its stop reason maps to
// the client's normal stop (OpenAI `stop`, Anthropic `end_turn`). Streaming answers are
// reassembled into the client's event order: OpenAI chunks and `[DONE]`, Anthropic
// message_start through message_stop, each carrying the provider's usage whole.

import { authHeaders, isRecord, type Probe, type ProviderInfo } from "./providers.ts";

/** One provider answer, parsed out of its wire: the text and the usage it reported. */
export interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number };
}

/** A provider call's outcome: an answer, or a refusal with its status. */
export type CallOutcome =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number };

const FIXED_CREATED = 1767225600;

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? (value as Record<string, unknown>) : null;
}

/** Split a text into pieces the way the simulated providers stream it: 13 characters at a time. */
export function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [""] : pieces;
}

/* ------------------------------------------------------------------ */
/* Reading the client's request                                        */
/* ------------------------------------------------------------------ */

export interface NormalizedTool {
  name: string;
  description: string;
  inputSchema: unknown;
}

/** The client's tools as (name, description, input schema), whatever wire sent them. */
export function normalizeTools(body: Record<string, unknown>, clientApi: "openai" | "anthropic"): NormalizedTool[] {
  const out: NormalizedTool[] = [];
  if (!Array.isArray(body.tools)) return out;
  for (const raw of body.tools) {
    const tool = asRecord(raw);
    if (tool === null) continue;
    if (clientApi === "openai") {
      const fn = asRecord(tool.function);
      if (fn === null) continue;
      const name = asString(fn.name);
      if (name === null) continue;
      out.push({ name, description: asString(fn.description) ?? "", inputSchema: fn.parameters ?? {} });
    } else {
      const name = asString(tool.name);
      if (name === null) continue;
      out.push({ name, description: asString(tool.description) ?? "", inputSchema: tool.input_schema ?? {} });
    }
  }
  return out;
}

function textOfBlocks(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    const b = asRecord(block);
    if (b !== null && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

/** The client's system prompt, from its system field or its system messages. */
export function systemTextOf(body: Record<string, unknown>, clientApi: "openai" | "anthropic"): string {
  if (clientApi === "anthropic") return textOfBlocks(body.system);
  const parts: string[] = [];
  if (!Array.isArray(body.messages)) return "";
  for (const raw of body.messages) {
    const message = asRecord(raw);
    if (message !== null && message.role === "system") {
      const text = textOfBlocks(message.content);
      if (text !== "") parts.push(text);
    }
  }
  return parts.join("\n");
}

function maxTokensOf(body: Record<string, unknown>): number | null {
  return asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens);
}

function stopsOf(body: Record<string, unknown>, clientApi: "openai" | "anthropic"): string[] {
  const raw = clientApi === "openai" ? body.stop : body.stop_sequences;
  if (typeof raw === "string") return [raw];
  if (!Array.isArray(raw)) return [];
  return raw.filter((x): x is string => typeof x === "string");
}

function jsonModeOf(body: Record<string, unknown>, clientApi: "openai" | "anthropic"): boolean {
  if (clientApi !== "openai") return false;
  const format = asRecord(body.response_format);
  return format !== null && (format.type === "json_object" || format.type === "json_schema");
}

function temperatureOf(body: Record<string, unknown>): number | null {
  return asNumber(body.temperature);
}

/* ------------------------------------------------------------------ */
/* Translating messages between wires                                  */
/* ------------------------------------------------------------------ */

/** An OpenAI message list for the upstream call, from either client wire. */
export function toOpenAIMessages(
  body: Record<string, unknown>,
  clientApi: "openai" | "anthropic",
  systemText: string,
): Record<string, unknown>[] {
  if (clientApi === "openai") {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    return messages.filter((m): m is Record<string, unknown> => isRecord(m));
  }
  const out: Record<string, unknown>[] = [];
  if (systemText !== "") out.push({ role: "system", content: systemText });
  if (!Array.isArray(body.messages)) return out;
  for (const raw of body.messages) {
    const message = asRecord(raw);
    if (message === null) continue;
    if (message.role === "user") {
      const content = message.content;
      if (typeof content === "string") {
        out.push({ role: "user", content });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const blocks: Record<string, unknown>[] = [];
      const flushText = (): void => {
        if (blocks.length > 0) {
          out.push({ role: "user", content: blocks.splice(0, blocks.length) });
        }
      };
      for (const rawBlock of content) {
        const block = asRecord(rawBlock);
        if (block === null) continue;
        if (block.type === "text" && typeof block.text === "string") {
          blocks.push({ type: "text", text: block.text });
        } else if (block.type === "tool_result") {
          flushText();
          const text = textOfBlocks(block.content);
          out.push({ role: "tool", tool_call_id: asString(block.tool_use_id) ?? "", content: text });
        }
      }
      flushText();
    } else if (message.role === "assistant") {
      const content = message.content;
      const textParts: Record<string, unknown>[] = [];
      const toolCalls: Record<string, unknown>[] = [];
      if (typeof content === "string") {
        if (content !== "") textParts.push({ type: "text", text: content });
      } else if (Array.isArray(content)) {
        for (const rawBlock of content) {
          const block = asRecord(rawBlock);
          if (block === null) continue;
          if (block.type === "text" && typeof block.text === "string") {
            textParts.push({ type: "text", text: block.text });
          } else if (block.type === "tool_use") {
            toolCalls.push({
              id: asString(block.id) ?? "",
              type: "function",
              function: {
                name: asString(block.name) ?? "",
                arguments: JSON.stringify(block.input ?? {}),
              },
            });
          }
        }
      }
      const outMessage: Record<string, unknown> = { role: "assistant" };
      outMessage.content = textParts.length > 0 ? textParts : null;
      if (toolCalls.length > 0) outMessage.tool_calls = toolCalls;
      out.push(outMessage);
    }
  }
  return out;
}

/** An Anthropic message list for the upstream call, from either client wire. */
export function toAnthropicMessages(
  body: Record<string, unknown>,
  clientApi: "openai" | "anthropic",
): Record<string, unknown>[] {
  if (clientApi === "anthropic") {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    return messages.filter((m): m is Record<string, unknown> => isRecord(m));
  }
  const out: Record<string, unknown>[] = [];
  if (!Array.isArray(body.messages)) return out;
  for (const raw of body.messages) {
    const message = asRecord(raw);
    if (message === null) continue;
    if (message.role === "system") continue;
    if (message.role === "user") {
      const content = message.content;
      if (typeof content === "string") {
        out.push({ role: "user", content: [{ type: "text", text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const blocks: Record<string, unknown>[] = [];
      for (const rawBlock of content) {
        const block = asRecord(rawBlock);
        if (block === null) continue;
        if (block.type === "text" && typeof block.text === "string") {
          blocks.push({ type: "text", text: block.text });
        } else if (block.type === "image_url") {
          const imageUrl = asRecord(block.image_url);
          const url = imageUrl === null ? null : asString(imageUrl.url);
          if (url === null) continue;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) {
            blocks.push({
              type: "image",
              source: { type: "base64", media_type: data[1] === "" ? "image/png" : data[1], data: data[2] },
            });
          } else {
            blocks.push({ type: "image", source: { type: "url", url } });
          }
        }
      }
      if (blocks.length > 0) out.push({ role: "user", content: blocks });
    } else if (message.role === "assistant") {
      const content = message.content;
      const blocks: Record<string, unknown>[] = [];
      if (typeof content === "string") {
        if (content !== "") blocks.push({ type: "text", text: content });
      } else if (Array.isArray(content)) {
        for (const rawBlock of content) {
          const block = asRecord(rawBlock);
          if (block === null) continue;
          if (block.type === "text" && typeof block.text === "string") {
            blocks.push({ type: "text", text: block.text });
          }
        }
      }
      const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
      for (const rawCall of toolCalls) {
        const call = asRecord(rawCall);
        if (call === null) continue;
        const fn = asRecord(call.function);
        if (fn === null) continue;
        let input: unknown = {};
        try {
          input = JSON.parse(asString(fn.arguments) ?? "{}");
        } catch {
          input = {};
        }
        blocks.push({ type: "tool_use", id: asString(call.id) ?? "", name: asString(fn.name) ?? "", input });
      }
      out.push({ role: "assistant", content: blocks });
    } else if (message.role === "tool") {
      const text = textOfBlocks(message.content);
      out.push({
        role: "user",
        content: [{ type: "tool_result", tool_use_id: asString(message.tool_call_id) ?? "", content: text }],
      });
    }
  }
  return out;
}

/** Gemini contents for the upstream call, from either client wire. */
export function toGeminiContents(
  body: Record<string, unknown>,
  clientApi: "openai" | "anthropic",
): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  const pushParts = (role: "user" | "model", parts: Record<string, unknown>[]): void => {
    if (parts.length > 0) out.push({ role, parts });
  };
  if (clientApi === "anthropic") {
    if (!Array.isArray(body.messages)) return out;
    for (const raw of body.messages) {
      const message = asRecord(raw);
      if (message === null) continue;
      if (message.role === "user") {
        const parts = geminiPartsOfAnthropic(message.content);
        pushParts("user", parts);
      } else if (message.role === "assistant") {
        const parts = geminiPartsOfAnthropic(message.content);
        pushParts("model", parts);
      }
    }
    return out;
  }
  if (!Array.isArray(body.messages)) return out;
  for (const raw of body.messages) {
    const message = asRecord(raw);
    if (message === null) continue;
    if (message.role === "system") continue;
    if (message.role === "user") {
      const parts = geminiPartsOfOpenAI(message.content);
      pushParts("user", parts);
    } else if (message.role === "assistant") {
      const parts: Record<string, unknown>[] = [];
      const content = message.content;
      if (typeof content === "string") {
        if (content !== "") parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const rawBlock of content) {
          const block = asRecord(rawBlock);
          if (block !== null && block.type === "text" && typeof block.text === "string") {
            parts.push({ text: block.text });
          }
        }
      }
      const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
      for (const rawCall of toolCalls) {
        const call = asRecord(rawCall);
        if (call === null) continue;
        const fn = asRecord(call.function);
        if (fn === null) continue;
        let args: unknown = {};
        try {
          args = JSON.parse(asString(fn.arguments) ?? "{}");
        } catch {
          args = {};
        }
        parts.push({ functionCall: { name: asString(fn.name) ?? "", args } });
      }
      pushParts("model", parts);
    } else if (message.role === "tool") {
      const text = textOfBlocks(message.content);
      pushParts("user", [{ functionResponse: { name: "", response: { text } } }]);
    }
  }
  return out;
}

function geminiPartsOfAnthropic(content: unknown): Record<string, unknown>[] {
  const parts: Record<string, unknown>[] = [];
  if (typeof content === "string") {
    if (content !== "") parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const rawBlock of content) {
    const block = asRecord(rawBlock);
    if (block === null) continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ text: block.text });
    } else if (block.type === "image") {
      const source = asRecord(block.source);
      if (source === null) continue;
      if (source.type === "base64" && typeof source.data === "string") {
        parts.push({
          inlineData: { mimeType: asString(source.media_type) ?? "image/png", data: source.data },
        });
      } else if (source.type === "url" && typeof source.url === "string") {
        parts.push({ fileData: { fileUri: source.url } });
      }
    } else if (block.type === "tool_use") {
      parts.push({ functionCall: { name: asString(block.name) ?? "", args: block.input ?? {} } });
    } else if (block.type === "tool_result") {
      parts.push({
        functionResponse: { name: asString(block.tool_use_id) ?? "", response: { text: textOfBlocks(block.content) } },
      });
    }
  }
  return parts;
}

function geminiPartsOfOpenAI(content: unknown): Record<string, unknown>[] {
  const parts: Record<string, unknown>[] = [];
  if (typeof content === "string") {
    if (content !== "") parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const rawBlock of content) {
    const block = asRecord(rawBlock);
    if (block === null) continue;
    if (block.type === "text" && typeof block.text === "string") {
      parts.push({ text: block.text });
    } else if (block.type === "image_url") {
      const imageUrl = asRecord(block.image_url);
      const url = imageUrl === null ? null : asString(imageUrl.url);
      if (url === null) continue;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) {
        parts.push({
          inlineData: { mimeType: data[1] === "" ? "image/png" : data[1], data: data[2] },
        });
      } else {
        parts.push({ fileData: { fileUri: url } });
      }
    }
  }
  return parts;
}

/* ------------------------------------------------------------------ */
/* Building one upstream call                                          */
/* ------------------------------------------------------------------ */

/** The upstream request body for one provider, in that provider's wire. */
export function upstreamBody(
  probe: Probe,
  modelId: string,
  clientApi: "openai" | "anthropic",
  clientBody: Record<string, unknown>,
  stream: boolean,
): Record<string, unknown> {
  const systemText = systemTextOf(clientBody, clientApi);
  const tools = normalizeTools(clientBody, clientApi);
  const maxTokens = maxTokensOf(clientBody);
  const stops = stopsOf(clientBody, clientApi);
  const jsonMode = jsonModeOf(clientBody, clientApi);
  const temperature = temperatureOf(clientBody);
  const wire = probe.provider.wire;
  if (wire === "openai") {
    const body: Record<string, unknown> = {
      model: modelId,
      messages: toOpenAIMessages(clientBody, clientApi, systemText),
      stream,
    };
    if (maxTokens !== null) body.max_tokens = maxTokens;
    if (tools.length > 0) {
      body.tools = tools.map((t) => ({
        type: "function",
        function: { name: t.name, description: t.description, parameters: t.inputSchema },
      }));
      const choice = clientBody.tool_choice;
      if (choice !== undefined) {
        if (clientApi === "openai") body.tool_choice = choice;
        else if (isRecord(choice) && choice.type === "auto") body.tool_choice = "auto";
        else if (isRecord(choice) && choice.type === "none") body.tool_choice = "none";
        else if (isRecord(choice) && choice.type === "tool" && typeof choice.name === "string") {
          body.tool_choice = { type: "function", function: { name: choice.name } };
        }
      }
    }
    if (stops.length > 0) body.stop = stops;
    if (clientApi === "openai" && jsonMode) {
      const format = asRecord(clientBody.response_format);
      body.response_format = format !== null ? format : { type: "json_object" };
    }
    if (temperature !== null) body.temperature = temperature;
    if (stream) body.stream_options = { include_usage: true };
    return body;
  }
  if (wire === "anthropic") {
    const body: Record<string, unknown> = {
      model: modelId,
      messages: toAnthropicMessages(clientBody, clientApi),
      stream,
    };
    if (systemText !== "") body.system = systemText;
    if (maxTokens !== null) body.max_tokens = maxTokens;
    if (tools.length > 0) {
      body.tools = tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.inputSchema }));
      const choice = clientBody.tool_choice;
      if (choice !== undefined) {
        if (clientApi === "anthropic" && isRecord(choice)) body.tool_choice = choice;
        else if (choice === "auto") body.tool_choice = { type: "auto" };
        else if (choice === "none") body.tool_choice = { type: "none" };
        else if (isRecord(choice) && choice.type === "function") {
          const fn = asRecord(choice.function);
          if (fn !== null && typeof fn.name === "string") body.tool_choice = { type: "tool", name: fn.name };
        }
      }
    }
    if (stops.length > 0) body.stop_sequences = stops;
    if (temperature !== null) body.temperature = temperature;
    return body;
  }
  const generationConfig: Record<string, unknown> = {};
  if (maxTokens !== null) generationConfig.maxOutputTokens = maxTokens;
  if (stops.length > 0) generationConfig.stopSequences = stops;
  if (clientApi === "openai" && jsonMode) generationConfig.responseMimeType = "application/json";
  if (temperature !== null) generationConfig.temperature = temperature;
  const body: Record<string, unknown> = {
    contents: toGeminiContents(clientBody, clientApi),
    generationConfig,
  };
  if (systemText !== "") body.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.inputSchema,
        })),
      },
    ];
  }
  return body;
}

/** The upstream URL for one provider call, in that provider's wire. */
function upstreamUrl(
  envUrl: string,
  provider: ProviderInfo,
  modelId: string,
  stream: boolean,
): string {
  const base = envUrl.replace(/\/+$/, "") + provider.path;
  if (provider.wire === "openai") return `${base}/v1/chat/completions`;
  if (provider.wire === "anthropic") return `${base}/v1/messages`;
  const action = stream ? "streamGenerateContent" : "generateContent";
  const query = stream ? "?alt=sse" : "";
  return `${base}/v1beta/models/${encodeURIComponent(modelId)}:${action}${query}`;
}

/* ------------------------------------------------------------------ */
/* Reading one provider's answer                                       */
/* ------------------------------------------------------------------ */

function parseSseText(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const frame of normalized.split("\n\n")) {
    if (frame.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function parseOpenAIAnswer(body: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    const value = asRecord(safeJson(body));
    if (value === null) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = asRecord(choices[0]);
    const message = choice === null ? null : asRecord(choice.message);
    if (message === null) return null;
    const text = typeof message.content === "string" ? message.content : "";
    const usage = asRecord(value.usage);
    if (usage === null) return null;
    const input = asNumber(usage.prompt_tokens);
    const output = asNumber(usage.completion_tokens);
    if (input === null || output === null) return null;
    return { text, usage: { input, output } };
  }
  let text = "";
  let usage: { input: number; output: number } | null = null;
  for (const frame of parseSseText(body)) {
    if (frame.data.trim() === "[DONE]") continue;
    const chunk = asRecord(safeJson(frame.data));
    if (chunk === null) continue;
    const chunkUsage = asRecord(chunk.usage);
    if (chunkUsage !== null) {
      const input = asNumber(chunkUsage.prompt_tokens);
      const output = asNumber(chunkUsage.completion_tokens);
      if (input !== null && output !== null) usage = { input, output };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = asRecord(choices[0]);
    if (choice === null) continue;
    const delta = asRecord(choice.delta);
    if (delta !== null && typeof delta.content === "string") text += delta.content;
  }
  if (usage === null) return null;
  return { text, usage };
}

function parseAnthropicAnswer(body: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    const value = asRecord(safeJson(body));
    if (value === null) return null;
    const usage = asRecord(value.usage);
    if (usage === null) return null;
    const input = asNumber(usage.input_tokens);
    const output = asNumber(usage.output_tokens);
    if (input === null || output === null) return null;
    return { text: anthropicTextOf(value.content), usage: { input, output } };
  }
  let text = "";
  let input: number | null = null;
  let output: number | null = null;
  for (const frame of parseSseText(body)) {
    const data = asRecord(safeJson(frame.data));
    if (data === null) continue;
    if (frame.event === "message_start") {
      const message = asRecord(data.message);
      const usage = message === null ? null : asRecord(message.usage);
      if (usage !== null) input = asNumber(usage.input_tokens);
    } else if (frame.event === "content_block_delta") {
      const delta = asRecord(data.delta);
      if (delta !== null && delta.type === "text_delta" && typeof delta.text === "string") text += delta.text;
    } else if (frame.event === "message_delta") {
      const usage = asRecord(data.usage);
      if (usage !== null) output = asNumber(usage.output_tokens);
    }
  }
  if (input === null || output === null) return null;
  return { text, usage: { input, output } };
}

function anthropicTextOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    const b = asRecord(block);
    if (b !== null && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("");
}

function parseGeminiAnswer(body: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    const value = asRecord(safeJson(body));
    if (value === null) return null;
    return geminiAnswerOf(value);
  }
  let text = "";
  let usage: { input: number; output: number } | null = null;
  for (const frame of parseSseText(body)) {
    const value = asRecord(safeJson(frame.data));
    if (value === null) continue;
    // A streamed frame carries text in its parts; only the last frame carries the usage
    // counts, so the two are read apart: a frame with no usage still answers with text.
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = asRecord(candidates[0]);
    const content = candidate === null ? null : asRecord(candidate.content);
    const parts = content === null || !Array.isArray(content.parts) ? [] : content.parts;
    for (const raw of parts) {
      const part = asRecord(raw);
      if (part !== null && typeof part.text === "string") text += part.text;
    }
    const meta = asRecord(value.usageMetadata);
    if (meta !== null) {
      const input = asNumber(meta.promptTokenCount);
      const output = asNumber(meta.candidatesTokenCount);
      if (input !== null && output !== null) usage = { input, output };
    }
  }
  if (usage === null) return null;
  return { text, usage };
}

function geminiAnswerOf(value: Record<string, unknown>): ProviderAnswer | null {
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const candidate = asRecord(candidates[0]);
  if (candidate === null) return null;
  const content = asRecord(candidate.content);
  const parts = content === null || !Array.isArray(content.parts) ? [] : content.parts;
  const texts: string[] = [];
  for (const raw of parts) {
    const part = asRecord(raw);
    if (part !== null && typeof part.text === "string") texts.push(part.text);
  }
  const meta = asRecord(value.usageMetadata);
  if (meta === null) return null;
  const input = asNumber(meta.promptTokenCount);
  const output = asNumber(meta.candidatesTokenCount);
  if (input === null || output === null) return null;
  return { text: texts.join(""), usage: { input, output } };
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/**
 * Call one provider with one model and read its answer. A 2xx with a parsable answer is a
 * success; anything else is a refusal, and the caller decides whether to retry or move on.
 */
export async function callProvider(
  envUrl: string,
  probe: Probe,
  modelId: string,
  clientApi: "openai" | "anthropic",
  clientBody: Record<string, unknown>,
  stream: boolean,
): Promise<CallOutcome> {
  const provider = probe.provider;
  const url = upstreamUrl(envUrl, provider, modelId, stream);
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(provider) };
  if (provider.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(upstreamBody(probe, modelId, clientApi, clientBody, stream)),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  if (res.status < 200 || res.status >= 300) {
    try {
      await res.text();
    } catch {
      // The body is drained so the socket can close; its content does not matter.
    }
    return { ok: false, status: res.status };
  }
  const text = await res.text();
  const parsed =
    provider.wire === "openai"
      ? parseOpenAIAnswer(text, stream)
      : provider.wire === "anthropic"
        ? parseAnthropicAnswer(text, stream)
        : parseGeminiAnswer(text, stream);
  if (parsed === null) return { ok: false, status: 502 };
  return { ok: true, answer: parsed };
}

/* ------------------------------------------------------------------ */
/* Answering the client in its own wire                                */
/* ------------------------------------------------------------------ */

function openAIUsage(usage: { input: number; output: number }): Record<string, unknown> {
  return {
    prompt_tokens: usage.input,
    completion_tokens: usage.output,
    total_tokens: usage.input + usage.output,
  };
}

/** An OpenAI completion for one provider answer, JSON or streamed. */
export function openAIResponse(
  modelId: string,
  answer: ProviderAnswer,
  stream: boolean,
): { status: number; headers: Record<string, string>; body: string } {
  if (!stream) {
    return {
      status: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        id: "chatcmpl-sim",
        object: "chat.completion",
        created: FIXED_CREATED,
        model: modelId,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: answer.text },
            finish_reason: "stop",
            logprobs: null,
          },
        ],
        usage: openAIUsage(answer.usage),
      }),
    };
  }
  const chunk = (delta: Record<string, unknown>, finish: string | null): Record<string, unknown> => ({
    id: "chatcmpl-sim",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model: modelId,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [`data: ${JSON.stringify(chunk({ role: "assistant", content: "" }, null))}\n\n`];
  for (const piece of splitText(answer.text)) {
    frames.push(`data: ${JSON.stringify(chunk({ content: piece }, null))}\n\n`);
  }
  frames.push(`data: ${JSON.stringify(chunk({}, "stop"))}\n\n`);
  frames.push(
    `data: ${JSON.stringify({
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model: modelId,
      choices: [],
      usage: openAIUsage(answer.usage),
    })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  return {
    status: 200,
    headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" },
    body: frames.join(""),
  };
}

/** An Anthropic message for one provider answer, JSON or streamed. */
export function anthropicResponse(
  modelId: string,
  answer: ProviderAnswer,
  stream: boolean,
): { status: number; headers: Record<string, string>; body: string } {
  if (!stream) {
    return {
      status: 200,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model: modelId,
        content: [{ type: "text", text: answer.text }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
      }),
    };
  }
  const frame = (event: string, data: Record<string, unknown>): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const frames: string[] = [
    frame("message_start", {
      type: "message_start",
      message: {
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model: modelId,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
    frame("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      frame("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: piece },
      }),
    );
  }
  frames.push(
    frame("content_block_stop", { type: "content_block_stop", index: 0 }),
    frame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
    frame("message_stop", { type: "message_stop" }),
  );
  return {
    status: 200,
    headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" },
    body: frames.join(""),
  };
}
