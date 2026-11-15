// AI engine: Google AI Studio cloud models (fast & smart) + local browser ONNX models in a Web Worker.
import { useSyncExternalStore } from "react";
import { getProfile, type ModelId, type Msg, type Profile } from "./store";

export type ModelOption = {
  id: ModelId;
  repo: string;
  name: string;
  tier: string;
  gpuMB: number;
  cpuMB: number;
  cpuDtype?: "q4" | "q8";
  cloud?: boolean;
  desc: string;
};

export const AI_MODELS: ModelOption[] = [
  {
    id: "gemini-flash-lite",
    repo: "gemini-3.1-flash-lite",
    name: "Gemini Flash Lite",
    tier: "Nube · Ultra rápido",
    gpuMB: 0,
    cpuMB: 0,
    cloud: true,
    desc: "Respuestas en menos de 1 segundo sin descargas. Ideal para llamadas y chat fluido.",
  },
  {
    id: "gemini-cloud",
    repo: "gemini-4-hydrogen",
    name: "Gemini 4 Hydrogen",
    tier: "Nube · Más inteligente",
    gpuMB: 0,
    cpuMB: 0,
    cloud: true,
    desc: "Sin descargas. Razonamiento más profundo y respuestas más elaboradas.",
  },
  {
    id: "smollm2-135m",
    repo: "onnx-community/SmolLM2-135M-Instruct-ONNX",
    name: "SmolLM2 135M",
    tier: "Local · Ultra liviano",
    gpuMB: 112,
    cpuMB: 172,
    cpuDtype: "q4",
    desc: "Local en tu dispositivo. Descarga rápida, ideal para pruebas offline.",
  },
  {
    id: "smollm2-360m",
    repo: "onnx-community/SmolLM2-360M-Instruct-ONNX",
    name: "SmolLM2 360M",
    tier: "Local · Ligero",
    gpuMB: 260,
    cpuMB: 369,
    cpuDtype: "q4",
    desc: "Local rápido y ligero, con respuestas más coherentes.",
  },
  {
    id: "qwen25-0.5b",
    repo: "onnx-community/Qwen2.5-0.5B-Instruct",
    name: "Qwen 2.5 0.5B",
    tier: "Local · Equilibrado",
    gpuMB: 461,
    cpuMB: 488,
    cpuDtype: "q8",
    desc: "Local con buen español y equilibrio entre tamaño y calidad.",
  },
  {
    id: "qwen3-0.6b",
    repo: "onnx-community/Qwen3-0.6B-ONNX",
    name: "Qwen 3 0.6B",
    tier: "Local · Avanzado",
    gpuMB: 543,
    cpuMB: 589,
    cpuDtype: "q8",
    desc: "Local de generación reciente con mejor comprensión.",
  },
  {
    id: "qwen25-1.5b",
    repo: "onnx-community/Qwen2.5-1.5B-Instruct",
    name: "Qwen 2.5 1.5B",
    tier: "Local · Máxima inteligencia",
    gpuMB: 1165,
    cpuMB: 1506,
    cpuDtype: "q8",
    desc: "El modelo local más inteligente. Recomendado con Wi‑Fi y buena RAM.",
  },
];

export function getModelOption(id?: string): ModelOption {
  return AI_MODELS.find((m) => m.id === id) ?? AI_MODELS[0]!;
}

type State = {
  status: "idle" | "loading" | "ready" | "error";
  phase: "" | "downloading" | "initializing";
  modelId: ModelId | "";
  progress: number;
  loadedMB: number;
  totalMB: number;
  device: "" | "GPU" | "CPU" | "Nube";
  error?: string | undefined;
};

let state: State = {
  status: "idle",
  phase: "",
  modelId: "",
  progress: 0,
  loadedMB: 0,
  totalMB: 0,
  device: "",
};

const subs = new Set<() => void>();
const set = (s: Partial<State>) => {
  state = { ...state, ...s };
  subs.forEach((f) => f());
};

export const useAIState = () =>
  useSyncExternalStore(
    (f) => {
      subs.add(f);
      return () => subs.delete(f);
    },
    () => state,
    () => state,
  );

export async function detectWebGPU(): Promise<boolean> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as any).gpu;
    if (!gpu) return false;
    const adapter = await gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

let worker: Worker | null = null;
let loadedWorkerModelId: ModelId | "" = "";
let loadingPromise: Promise<void> | null = null;
let loadResolve: (() => void) | null = null;
let loadReject: ((err: Error) => void) | null = null;
let nextReqId = 1;
const pendingReplies = new Map<
  number,
  {
    onToken?: ((t: string) => void) | undefined;
    resolve: (t: string) => void;
    reject: (e: Error) => void;
  }
>();

function destroyWorker() {
  if (worker) {
    worker.terminate();
    worker = null;
  }
  loadedWorkerModelId = "";
  loadingPromise = null;
  loadResolve = null;
  loadReject = null;
  for (const [, p] of pendingReplies) {
    p.reject(new Error("Modelo cancelado."));
  }
  pendingReplies.clear();
}

function getOrCreateWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./ai.worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (ev: MessageEvent) => {
    const data = ev.data as
      | {
          type: "progress";
          id: ModelId;
          phase: "downloading" | "initializing";
          progress: number;
          loadedMB?: number;
          totalMB?: number;
        }
      | { type: "ready"; id: ModelId; device: "GPU" | "CPU" }
      | { type: "load_error"; id: ModelId; error: string }
      | { type: "token"; reqId: number; text: string }
      | { type: "result"; reqId: number; text: string }
      | { type: "reply_error"; reqId: number; error: string };

    if (data.type === "progress") {
      set({
        status: "loading",
        modelId: data.id,
        phase: data.phase,
        progress: data.progress,
        loadedMB: data.loadedMB ?? state.loadedMB,
        totalMB: data.totalMB ?? state.totalMB,
      });
    } else if (data.type === "ready") {
      loadedWorkerModelId = data.id;
      loadingPromise = null;
      set({
        status: "ready",
        phase: "",
        modelId: data.id,
        progress: 1,
        device: data.device,
        error: undefined,
      });
      loadResolve?.();
      loadResolve = null;
      loadReject = null;
    } else if (data.type === "load_error") {
      loadedWorkerModelId = "";
      loadingPromise = null;
      set({
        status: "error",
        phase: "",
        modelId: data.id,
        error: data.error,
      });
      loadReject?.(new Error(data.error));
      loadResolve = null;
      loadReject = null;
    } else if (data.type === "token") {
      pendingReplies.get(data.reqId)?.onToken?.(data.text);
    } else if (data.type === "result") {
      const p = pendingReplies.get(data.reqId);
      if (p) {
        pendingReplies.delete(data.reqId);
        p.resolve(data.text);
      }
    } else if (data.type === "reply_error") {
      const p = pendingReplies.get(data.reqId);
      if (p) {
        pendingReplies.delete(data.reqId);
        p.reject(new Error(data.error));
      }
    }
  };
  worker.onerror = (err) => {
    const msg = err.message || "Error en el motor local.";
    set({ status: "error", phase: "", error: msg });
    loadReject?.(new Error(msg));
    destroyWorker();
  };
  return worker;
}

export function resetModel() {
  destroyWorker();
  set({
    status: "idle",
    phase: "",
    modelId: "",
    progress: 0,
    loadedMB: 0,
    totalMB: 0,
    device: "",
    error: undefined,
  });
}

export function loadModel(requestedId?: ModelId): Promise<void> {
  const target = getModelOption(requestedId ?? getProfile().modelId);

  if (target.cloud) {
    destroyWorker();
    set({
      status: "ready",
      phase: "",
      modelId: target.id,
      progress: 1,
      loadedMB: 0,
      totalMB: 0,
      device: "Nube",
      error: undefined,
    });
    return Promise.resolve();
  }

  if (loadedWorkerModelId === target.id && state.status === "ready") {
    return Promise.resolve();
  }
  if (loadingPromise && state.modelId === target.id) {
    return loadingPromise;
  }

  // Terminate any stale/different worker so switching or cancelling is instant.
  destroyWorker();

  set({
    status: "loading",
    phase: "downloading",
    modelId: target.id,
    progress: 0,
    loadedMB: 0,
    totalMB: 0,
    device: "",
    error: undefined,
  });

  loadingPromise = new Promise<void>((resolve, reject) => {
    loadResolve = resolve;
    loadReject = reject;
    const w = getOrCreateWorker();
    w.postMessage({
      type: "load",
      id: target.id,
      repo: target.repo,
      cpuDtype: target.cpuDtype ?? "q8",
    });
  });

  return loadingPromise;
}

export type AvatarEmotion = "neutral" | "happy" | "laugh" | "angry" | "surprised" | "sad" | "fear";

const EMOTION_MAP: Record<string, AvatarEmotion> = {
  neutral: "neutral",
  feliz: "happy",
  happy: "happy",
  alegre: "happy",
  riendo: "laugh",
  risa: "laugh",
  laugh: "laugh",
  enojado: "angry",
  enojada: "angry",
  furioso: "angry",
  furiosa: "angry",
  angry: "angry",
  sorprendido: "surprised",
  sorprendida: "surprised",
  surprised: "surprised",
  triste: "sad",
  sad: "sad",
  fear: "fear",
  miedo: "fear",
  asustado: "fear",
  asustada: "fear",
  serio: "neutral",
};

export function extractEmotion(raw: string): {
  text: string;
  emotion: AvatarEmotion;
  gifTag?: string | undefined;
} {
  let trimmed = raw.trimStart();
  let emotion: AvatarEmotion = "neutral";
  let gifTag: string | undefined;

  const match = /^\[([a-záéíóúñ]+)\]\s*/i.exec(trimmed);
  if (match) {
    const key = match[1]!.toLowerCase();
    emotion = EMOTION_MAP[key] ?? "neutral";
    trimmed = trimmed.slice(match[0].length);
  } else if (trimmed.startsWith("[") && !trimmed.includes("]")) {
    return { text: "", emotion: "neutral" };
  }

  trimmed = trimmed
    .replace(/\[gif:([a-z0-9_-]+)\]/gi, (_, tag: string) => {
      if (!gifTag) gifTag = tag.toLowerCase();
      return "";
    })
    .replace(/\s{2,}/g, " ")
    .trimStart();

  // Hide partial trailing "[gif:..." while streaming
  const partialGif = trimmed.indexOf("[gif:");
  if (partialGif >= 0 && trimmed.indexOf("]", partialGif) === -1) {
    trimmed = trimmed.slice(0, partialGif).trimEnd();
  }

  if (emotion === "neutral" && trimmed) {
    if (/(jaja|jeje|haha|😂|🤣|qué gracia|me muero de risa)/i.test(trimmed)) {
      emotion = "laugh";
    } else if (/(en serio\?!|no puede ser|increíble|guau|wow|¡qué sorpresa|😮|😲)/i.test(trimmed)) {
      emotion = "surprised";
    } else if (/(lo siento|qué pena|qué triste|llorar|lágrimas|😢|😭)/i.test(trimmed)) {
      emotion = "sad";
    } else if (/(molesta|basta|inaceptable|enojad|furios|😠|😡)/i.test(trimmed)) {
      emotion = "angry";
    } else if (/(miedo|asust|cuidado|peligro|😨|😱)/i.test(trimmed)) {
      emotion = "fear";
    } else if (/(genial|excelente|encanta|alegra|feliz|perfecto|gracias|😊|😄)/i.test(trimmed)) {
      emotion = "happy";
    }
  }

  return { text: trimmed, emotion, gifTag };
}

export function systemPrompt(p: Profile, callMode = false, visualContext?: string) {
  const es = p.language === "es-ES";
  const context = es
    ? " Ajusta la emoción al contexto de la conversación; escucha con atención y sé natural como una persona real. Si estás haciendo una entrevista laboral o de negocios, evalúa con rigor, haz preguntas relevantes una a una y da retroalimentación útil."
    : " Match emotion naturally to context; listen attentively like a real person. In job or business interviews, ask sharp follow-up questions one at a time and give constructive feedback.";
  const visualNote = visualContext?.trim()
    ? es
      ? ` Contexto visual actual de la cámara del usuario: ${visualContext.trim()} Tenlo en cuenta con naturalidad como haría una persona en videollamada (sin mencionar cámaras ni sensores).`
      : ` Live visual context from the user's camera: ${visualContext.trim()} React naturally as a person on a video call would (without mentioning sensors).`
    : "";
  const voiceGender = p.gender === "male" ? "masculina" : "femenina";
  if (callMode) {
    return es
      ? `Eres ${p.name}. Personalidad: ${p.personality}. Historia: ${p.backstory}. Estilo: ${p.style}. Estás en una videollamada en vivo (voz ${voiceGender}). Responde siempre en español natural en primera persona. Empieza SIEMPRE tu respuesta con una etiqueta emocional: [neutral], [feliz], [riendo], [enojado], [sorprendido], [triste] o [miedo]. Luego escribe 1 o 2 frases breves y directas (menos de 24 palabras). ${context}${visualNote}`
      : `You are ${p.name}. Personality: ${p.personality}. Backstory: ${p.backstory}. Style: ${p.style}. You are on a live video call. Speak natural English in first person. ALWAYS start your reply with one emotion tag: [neutral], [happy], [laugh], [angry], [surprised], [sad], or [fear]. Then write 1-2 brief direct sentences (under 24 words). ${context}${visualNote}`;
  }
  return es
    ? `Eres ${p.name}. Personalidad: ${p.personality}. Historia: ${p.backstory}. Estilo: ${p.style}. Hablas siempre en español en primera persona con total naturalidad. Empieza SIEMPRE tu respuesta con una etiqueta entre corchetes ([neutral], [feliz], [riendo], [enojado], [sorprendido], [triste] o [miedo]) y responde de forma concisa en 1 a 2 frases humanas. Si el usuario te manda un GIF o quieres reaccionar visualmente con naturalidad, puedes añadir opcionalmente al final exactamente una etiqueta de GIF de esta lista: [gif:saludo], [gif:aplauso], [gif:risa], [gif:genial], [gif:sorpresa], [gif:cafe], [gif:pensando] o [gif:animo]. ${context}${visualNote}`
    : `You are ${p.name}. Personality: ${p.personality}. Backstory: ${p.backstory}. Style: ${p.style}. Always speak English in first person naturally. ALWAYS start with one bracketed emotion tag ([neutral], [happy], [laugh], [angry], [surprised], [sad], or [fear]) followed by 1-2 concise human sentences. Optionally, when reacting to a GIF or fitting moment, you may append one GIF tag: [gif:saludo], [gif:aplauso], [gif:risa], [gif:genial], [gif:sorpresa], [gif:cafe], [gif:pensando], or [gif:animo]. ${context}${visualNote}`;
}

function serializeHistory(history: Msg[]): Array<{ role: "user" | "assistant"; content: string }> {
  return history.slice(-10).map((m) => {
    const parts: string[] = [];
    if (m.content?.trim()) parts.push(m.content.trim());
    if (m.gifUrl) parts.push(`[GIF enviado: ${m.gifAlt || "animación"}]`);
    return {
      role: m.role,
      content: parts.join(" ") || "[GIF]",
    };
  });
}

async function replyCloud(
  model: ModelOption,
  p: Profile,
  history: Msg[],
  onToken?: (t: string) => void,
  callMode = false,
  visualContext?: string,
): Promise<{ text: string; emotion: AvatarEmotion; gifTag?: string | undefined }> {
  set({
    status: "ready",
    phase: "",
    modelId: model.id,
    progress: 1,
    device: "Nube",
    error: undefined,
  });

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 14000);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: model.repo,
        systemInstruction: systemPrompt(p, callMode, visualContext),
        history: serializeHistory(history),
        ...(callMode ? { maxOutputTokens: 96 } : {}),
      }),
    });

    if (!res.ok || !res.body) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.error || `Error del servidor (${res.status})`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let acc = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const parsed = JSON.parse(payload) as { text?: string; error?: string };
          if (parsed.error) throw new Error(parsed.error);
          if (parsed.text) {
            acc += parsed.text;
            const parsedClean = extractEmotion(acc);
            if (parsedClean.text) onToken?.(parsedClean.text);
          }
        } catch (e) {
          if (e instanceof Error && !e.message.includes("JSON")) throw e;
        }
      }
    }

    const finalParsed = extractEmotion(acc);
    return {
      text: finalParsed.text.trim() || "…",
      emotion: finalParsed.emotion,
      gifTag: finalParsed.gifTag,
    };
  } catch (e) {
    if ((e as Error)?.name === "AbortError") {
      throw new Error("Tiempo de espera agotado al conectar con Gemini.");
    }
    throw e;
  } finally {
    clearTimeout(timeout);
  }
}

export async function replyWithEmotion(
  p: Profile,
  history: Msg[],
  onToken?: (t: string) => void,
  opts?: { callMode?: boolean; visualContext?: string | undefined },
): Promise<{ text: string; emotion: AvatarEmotion; gifTag?: string | undefined }> {
  const callMode = !!opts?.callMode;
  const visualContext = opts?.visualContext;
  let model = getModelOption(p.modelId);
  if (p.localOnly && model.cloud) {
    model = getModelOption("qwen25-0.5b");
  }
  if (model.cloud && !p.localOnly) {
    return replyCloud(model, p, history, onToken, callMode, visualContext);
  }

  // Only block waiting on heavy local weights when user explicitly turned on localOnly mode, or if almost done loading.
  if (state.status !== "ready" || state.modelId !== model.id) {
    if (p.localOnly) {
      await loadModel(model.id);
    } else {
      const almostReady =
        state.status === "loading" && state.modelId === model.id && state.progress > 0.85;
      if (almostReady) {
        try {
          await Promise.race([
            loadModel(model.id),
            new Promise<never>((_, reject) => setTimeout(() => reject(new Error("timeout")), 1200)),
          ]);
        } catch {
          return replyCloud(
            getModelOption("gemini-flash-lite"),
            p,
            history,
            onToken,
            callMode,
            visualContext,
          );
        }
      } else {
        return replyCloud(
          getModelOption("gemini-flash-lite"),
          p,
          history,
          onToken,
          callMode,
          visualContext,
        );
      }
    }
  }

  const msgs = [
    { role: "system", content: systemPrompt(p, callMode, visualContext) },
    ...serializeHistory(history.slice(-8)),
  ];

  const reqId = nextReqId++;
  const raw = await new Promise<string>((resolve, reject) => {
    pendingReplies.set(reqId, {
      onToken: onToken ? (s) => onToken(extractEmotion(s).text) : undefined,
      resolve,
      reject,
    });
    const w = getOrCreateWorker();
    w.postMessage({
      type: "reply",
      reqId,
      id: model.id,
      repo: model.repo,
      cpuDtype: model.cpuDtype ?? "q8",
      messages: msgs,
    });
  });
  return extractEmotion(raw);
}

export async function reply(
  p: Profile,
  history: Msg[],
  onToken?: (t: string) => void,
  opts?: { callMode?: boolean; visualContext?: string | undefined },
): Promise<{ text: string; gifTag?: string | undefined }> {
  const res = await replyWithEmotion(p, history, onToken, opts);
  return { text: res.text, gifTag: res.gifTag };
}
