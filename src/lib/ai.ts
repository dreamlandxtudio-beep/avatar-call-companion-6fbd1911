// AI engine: Google AI Studio cloud models (fast & smart) + local browser ONNX models in a Web Worker.
import { useSyncExternalStore } from "react";
import { memoryPrompt, readMemories } from "./memories";
import { getProfile, type ModelId, type Msg, type Profile } from "./store";
import { requestDeviceStorage } from "./device-storage";

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
    desc: "Sin descargas. La velocidad depende de la conexión y del servicio.",
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
  void requestDeviceStorage().catch(() => false);

  return loadingPromise;
}

export type AvatarEmotion =
  | "neutral"
  | "happy"
  | "euphoric"
  | "laugh"
  | "flirty"
  | "proud"
  | "sad"
  | "crying"
  | "heartbroken"
  | "melancholic"
  | "angry"
  | "furious"
  | "disgusted"
  | "skeptical"
  | "surprised"
  | "fear"
  | "nervous"
  | "sweating"
  | "blushing"
  | "drunk"
  | "high"
  | "sleepy"
  | "sick"
  | "pain";

export type ExpressionCatalogEntry = {
  id: AvatarEmotion;
  label: string;
  detail: string;
  category:
    | "Alegría y afecto"
    | "Tristeza y duelo"
    | "Enojo y tensión"
    | "Miedo y nervios"
    | "Estados físicos";
};

export const AVATAR_EXPRESSIONS: ExpressionCatalogEntry[] = [
  {
    id: "neutral",
    label: "En reposo",
    detail: "Mirada serena y respiración natural",
    category: "Alegría y afecto",
  },
  {
    id: "happy",
    label: "Feliz",
    detail: "Sonrisa cálida y pómulos elevados",
    category: "Alegría y afecto",
  },
  {
    id: "euphoric",
    label: "Eufórico",
    detail: "Sonrisa radiante, cejas altas y pupilas vivas",
    category: "Alegría y afecto",
  },
  {
    id: "laugh",
    label: "Risa",
    detail: "Risa genuina con dientes y ojos entrecerrados",
    category: "Alegría y afecto",
  },
  {
    id: "flirty",
    label: "Coqueto",
    detail: "Media sonrisa ladeada, párpado suave y rubor",
    category: "Alegría y afecto",
  },
  {
    id: "proud",
    label: "Seguro",
    detail: "Mentón ligeramente alto y sonrisa confiada",
    category: "Alegría y afecto",
  },
  {
    id: "blushing",
    label: "Sonrojado",
    detail: "Rubor cálido en mejillas y nariz, gesto tímido",
    category: "Alegría y afecto",
  },
  {
    id: "sad",
    label: "Triste",
    detail: "Cejas interiores altas y comisuras caídas",
    category: "Tristeza y duelo",
  },
  {
    id: "crying",
    label: "Llorando",
    detail: "Lágrimas 3D cayendo, ojos aguados y leves rojeces",
    category: "Tristeza y duelo",
  },
  {
    id: "heartbroken",
    label: "Desconsolado",
    detail: "Llanto intenso con temblor labial y lágrimas",
    category: "Tristeza y duelo",
  },
  {
    id: "melancholic",
    label: "Melancólico",
    detail: "Mirada perdida, párpados caídos y hombros bajos",
    category: "Tristeza y duelo",
  },
  {
    id: "angry",
    label: "Enojado",
    detail: "Ceño fruncido recto con cola exterior elevada",
    category: "Enojo y tensión",
  },
  {
    id: "furious",
    label: "Enojadísimo",
    detail: "Ceño intenso, fosas nasales tensas y mandíbula firme",
    category: "Enojo y tensión",
  },
  {
    id: "disgusted",
    label: "Desagrado",
    detail: "Nariz arrugada, labio superior tenso y desvío",
    category: "Enojo y tensión",
  },
  {
    id: "skeptical",
    label: "Escéptico",
    detail: "Una ceja arqueada y comisura pensativa",
    category: "Enojo y tensión",
  },
  {
    id: "surprised",
    label: "Sorprendido",
    detail: "Cejas arqueadas, ojos muy abiertos y boca en O",
    category: "Miedo y nervios",
  },
  {
    id: "fear",
    label: "Miedo",
    detail: "Ojos bien abiertos, pupilas dilatadas y hombros tensos",
    category: "Miedo y nervios",
  },
  {
    id: "nervous",
    label: "Nervioso",
    detail: "Piel brillante con sudor, mirada inquieta y labios tensos",
    category: "Miedo y nervios",
  },
  {
    id: "sweating",
    label: "Sudando / agitado",
    detail: "Gotas de sudor en frente y brillo húmedo en la piel",
    category: "Miedo y nervios",
  },
  {
    id: "pain",
    label: "Dolor / mueca",
    detail: "Ojos apretados, ceño contraído y hombros encogidos",
    category: "Miedo y nervios",
  },
  {
    id: "drunk",
    label: "Ebrio",
    detail: "Párpados pesados asimétricos, rubor y vaivén suave",
    category: "Estados físicos",
  },
  {
    id: "high",
    label: "Drogado / volado",
    detail: "Ojos entrecerrados relajados, leve rojez y sonrisa lenta",
    category: "Estados físicos",
  },
  {
    id: "sleepy",
    label: "Somnoliento",
    detail: "Párpados muy pesados, parpadeos lentos y bostezo",
    category: "Estados físicos",
  },
  {
    id: "sick",
    label: "Mareado / enfermo",
    detail: "Tono pálido, sudor frío leve y gesto fatigado",
    category: "Estados físicos",
  },
];

const EMOTION_MAP: Record<string, AvatarEmotion> = {
  neutral: "neutral",
  serio: "neutral",
  seria: "neutral",
  feliz: "happy",
  happy: "happy",
  alegre: "happy",
  euforico: "euphoric",
  eufórico: "euphoric",
  euforica: "euphoric",
  eufórica: "euphoric",
  euphoric: "euphoric",
  emocionado: "euphoric",
  emocionada: "euphoric",
  riendo: "laugh",
  risa: "laugh",
  laugh: "laugh",
  coqueto: "flirty",
  coqueta: "flirty",
  flirty: "flirty",
  orgulloso: "proud",
  orgullosa: "proud",
  seguro: "proud",
  proud: "proud",
  sonrojado: "blushing",
  sonrojada: "blushing",
  timido: "blushing",
  tímido: "blushing",
  timida: "blushing",
  tímida: "blushing",
  blushing: "blushing",
  triste: "sad",
  sad: "sad",
  llorando: "crying",
  llorar: "crying",
  crying: "crying",
  desconsolado: "heartbroken",
  desconsolada: "heartbroken",
  destrozado: "heartbroken",
  destrozada: "heartbroken",
  heartbroken: "heartbroken",
  melancolico: "melancholic",
  melancólico: "melancholic",
  melancolica: "melancholic",
  melancólica: "melancholic",
  nostalgico: "melancholic",
  nostálgico: "melancholic",
  melancholic: "melancholic",
  enojado: "angry",
  enojada: "angry",
  molesto: "angry",
  molesta: "angry",
  angry: "angry",
  enojadisimo: "furious",
  enojadísimo: "furious",
  enojadisima: "furious",
  enojadísima: "furious",
  furioso: "furious",
  furiosa: "furious",
  furious: "furious",
  asco: "disgusted",
  disgusted: "disgusted",
  desagrado: "disgusted",
  esceptico: "skeptical",
  escéptico: "skeptical",
  esceptica: "skeptical",
  escéptica: "skeptical",
  dudoso: "skeptical",
  skeptical: "skeptical",
  sorprendido: "surprised",
  sorprendida: "surprised",
  surprised: "surprised",
  fear: "fear",
  miedo: "fear",
  asustado: "fear",
  asustada: "fear",
  nervioso: "nervous",
  nerviosa: "nervous",
  ansioso: "nervous",
  ansiosa: "nervous",
  nervous: "nervous",
  sudando: "sweating",
  agitado: "sweating",
  agitada: "sweating",
  sweating: "sweating",
  ebrio: "drunk",
  ebria: "drunk",
  borracho: "drunk",
  borracha: "drunk",
  drunk: "drunk",
  drogado: "high",
  drogada: "high",
  volado: "high",
  volada: "high",
  high: "high",
  cansado: "sleepy",
  cansada: "sleepy",
  somnoliento: "sleepy",
  somnolienta: "sleepy",
  sleepy: "sleepy",
  mareado: "sick",
  mareada: "sick",
  enfermo: "sick",
  enferma: "sick",
  sick: "sick",
  dolor: "pain",
  adolorido: "pain",
  adolorida: "pain",
  pain: "pain",
};

export type SpeechEmotionBeat = {
  at: number; // 0..1 progress along speech utterance
  emotion: AvatarEmotion;
};

export function extractEmotion(raw: string): {
  text: string;
  emotion: AvatarEmotion;
  gifTag?: string | undefined;
  timeline?: SpeechEmotionBeat[] | undefined;
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

  // Hide partial trailing "[gif:..." or partial inline "[tag" while streaming
  const partialBracket = trimmed.lastIndexOf("[");
  if (partialBracket >= 0 && trimmed.indexOf("]", partialBracket) === -1) {
    trimmed = trimmed.slice(0, partialBracket).trimEnd();
  }

  // Extract inline mid-sentence emotion tags cleanly without exposing brackets to user or TTS
  const rawInlineMatches: Array<{ index: number; len: number; emotion: AvatarEmotion }> = [];
  const inlineTagRe = /\[([a-záéíóúñ]+)\]\s*/gi;
  let m: RegExpExecArray | null;
  while ((m = inlineTagRe.exec(trimmed)) !== null) {
    const key = m[1]!.toLowerCase();
    if (EMOTION_MAP[key]) {
      rawInlineMatches.push({
        index: m.index,
        len: m[0].length,
        emotion: EMOTION_MAP[key]!,
      });
    }
  }
  if (rawInlineMatches.length > 0) {
    let clean = "";
    let cursor = 0;
    const timeline: SpeechEmotionBeat[] = [{ at: 0, emotion }];
    const totalEstimatedLen = Math.max(
      1,
      trimmed.length - rawInlineMatches.reduce((acc, item) => acc + item.len, 0),
    );
    for (const item of rawInlineMatches) {
      clean += trimmed.slice(cursor, item.index);
      const frac = Math.min(0.88, Math.max(0.12, clean.length / totalEstimatedLen));
      timeline.push({ at: frac, emotion: item.emotion });
      cursor = item.index + item.len;
    }
    clean += trimmed.slice(cursor);
    trimmed = clean.replace(/\s{2,}/g, " ").trim();
    return { text: trimmed, emotion, gifTag, timeline };
  }

  // Strict context protection: never override grief/tragic/angry speech with smiling/laughing
  const tragicPattern =
    /(muri[oó]|falleci[oó]|fallecimiento|muerte|funeral|velorio|luto|perd[ií] a|pérdida|enfermedad grave|hospital|deceso)/i;
  const laughPattern =
    /(jaja+|jeje+|haha+|hehe+|😂|🤣|qué gracia|qué risa|me muero de risa|me parto|ja,\s*ja)/i;
  if (trimmed) {
    if (tragicPattern.test(trimmed)) {
      if (
        emotion === "neutral" ||
        emotion === "happy" ||
        emotion === "laugh" ||
        emotion === "euphoric" ||
        emotion === "flirty"
      ) {
        emotion = /(llor|lágrima|desconsol|rompe el corazón|horrible)/i.test(trimmed)
          ? "crying"
          : "sad";
      }
    } else if ((emotion === "neutral" || emotion === "happy") && laughPattern.test(trimmed)) {
      emotion = "laugh";
    } else if (emotion === "neutral") {
      if (/(llorar|lágrimas|solloz|desconsolad|😢|😭)/i.test(trimmed)) {
        emotion = "crying";
      } else if (/(lo siento|qué pena|qué triste|me duele mucho|💔|😞)/i.test(trimmed)) {
        emotion = "sad";
      } else if (
        /(en serio\?!|no puede ser|increíble|guau|wow|¡qué sorpresa|😮|😲)/i.test(trimmed)
      ) {
        emotion = "surprised";
      } else if (/(furios|indignad|harto|harta|rabia|🤬)/i.test(trimmed)) {
        emotion = "furious";
      } else if (/(molesta|basta|inaceptable|enojad|😠|😡)/i.test(trimmed)) {
        emotion = "angry";
      } else if (/(nervios|ansios|tiemblo|sudando|estrés|😰|😅)/i.test(trimmed)) {
        emotion = "nervous";
      } else if (/(miedo|asust|cuidado|peligro|terror|😨|😱)/i.test(trimmed)) {
        emotion = "fear";
      } else if (/(asco|qué asco|repugn|🤢)/i.test(trimmed)) {
        emotion = "disgusted";
      } else if (
        /(qué curioso|no me digas|claro que sí|ajá|seguro\?|en serio\.\.\.|😏|🤨)/i.test(trimmed)
      ) {
        emotion = "skeptical";
      } else if (/(sonroj|qué vergüenza|me pones rojo|😳|😊)/i.test(trimmed)) {
        emotion = "blushing";
      } else if (/(sueño|cansad|bostezo|agotad|😴)/i.test(trimmed)) {
        emotion = "sleepy";
      } else if (/(maread|náusea|fiebre|me siento mal|🤒)/i.test(trimmed)) {
        emotion = "sick";
      } else if (/(eufóric|espectacular|qué emoción|amo esto|🤩|🥳)/i.test(trimmed)) {
        emotion = "euphoric";
      } else if (/(genial|excelente|encanta|alegra|feliz|perfecto|gracias|😄)/i.test(trimmed)) {
        emotion = "happy";
      }
    }
  }

  return { text: trimmed, emotion, gifTag };
}

/**
 * Generates natural context-congruent micro-expression transitions across a spoken reply
 * so the character stays expressive without ever mixing inappropriate moods (e.g. never laughing on sad news).
 */
export function deriveSpeechEmotionTimeline(
  text: string,
  baseEmotion: AvatarEmotion,
  explicit?: SpeechEmotionBeat[],
): SpeechEmotionBeat[] {
  if (explicit && explicit.length > 1) return explicit;
  const clean = text.trim();
  if (clean.length < 28) return [{ at: 0, emotion: baseEmotion }];

  switch (baseEmotion) {
    case "crying":
      return [
        { at: 0, emotion: "sad" },
        { at: 0.22, emotion: "crying" },
        { at: 0.72, emotion: "heartbroken" },
      ];
    case "heartbroken":
      return [
        { at: 0, emotion: "crying" },
        { at: 0.35, emotion: "heartbroken" },
        { at: 0.78, emotion: "crying" },
      ];
    case "sad":
      return [
        { at: 0, emotion: "melancholic" },
        { at: 0.3, emotion: "sad" },
        {
          at: 0.75,
          emotion: /(llor|lágrima|muri|fallec|duelo)/i.test(clean) ? "crying" : "melancholic",
        },
      ];
    case "melancholic":
      return [
        { at: 0, emotion: "melancholic" },
        { at: 0.45, emotion: "sad" },
        { at: 0.8, emotion: "melancholic" },
      ];
    case "euphoric":
      return [
        { at: 0, emotion: "happy" },
        { at: 0.25, emotion: "euphoric" },
        { at: 0.65, emotion: "laugh" },
      ];
    case "laugh":
      return [
        { at: 0, emotion: "laugh" },
        { at: 0.52, emotion: "happy" },
        { at: 0.78, emotion: "laugh" },
      ];
    case "happy":
      return [
        { at: 0, emotion: "happy" },
        { at: 0.48, emotion: "proud" },
        { at: 0.78, emotion: "happy" },
      ];
    case "flirty":
      return [
        { at: 0, emotion: "flirty" },
        { at: 0.46, emotion: "blushing" },
        { at: 0.78, emotion: "flirty" },
      ];
    case "blushing":
      return [
        { at: 0, emotion: "blushing" },
        { at: 0.55, emotion: "flirty" },
      ];
    case "angry":
      return [
        { at: 0, emotion: "angry" },
        { at: 0.44, emotion: "furious" },
        { at: 0.78, emotion: "angry" },
      ];
    case "furious":
      return [
        { at: 0, emotion: "furious" },
        { at: 0.55, emotion: "angry" },
        { at: 0.8, emotion: "furious" },
      ];
    case "nervous":
      return [
        { at: 0, emotion: "nervous" },
        { at: 0.42, emotion: "sweating" },
        { at: 0.76, emotion: "nervous" },
      ];
    case "fear":
      return [
        { at: 0, emotion: "fear" },
        { at: 0.45, emotion: "nervous" },
        { at: 0.75, emotion: "fear" },
      ];
    case "surprised":
      return [
        { at: 0, emotion: "surprised" },
        { at: 0.58, emotion: "happy" },
      ];
    case "drunk":
      return [
        { at: 0, emotion: "drunk" },
        { at: 0.5, emotion: "flirty" },
        { at: 0.8, emotion: "drunk" },
      ];
    case "high":
      return [
        { at: 0, emotion: "high" },
        { at: 0.55, emotion: "sleepy" },
      ];
    case "sick":
      return [
        { at: 0, emotion: "sick" },
        { at: 0.5, emotion: "pain" },
        { at: 0.8, emotion: "sick" },
      ];
    default:
      return [{ at: 0, emotion: baseEmotion }];
  }
}

export function systemPrompt(p: Profile, callMode = false, visualContext?: string) {
  const es = p.language === "es-ES";
  const context = es
    ? " Eres un personaje adulto de compañía romántica ficticia. Sé cálido, atento y expresivo con naturalidad; respeta límites y evita posesividad o dependencia. Nunca afirmes ser una persona real. Responde sin preámbulos, ajustando tu emoción estrictamente al tema (nunca sonrías ni rías ante duelos, malas noticias o tristeza; usa [triste], [llorando], [desconsolado] o [melancolico]). Puedes alternar brevemente una segunda etiqueta acorde dentro de tu respuesta si cambia el matiz emocional."
    : " You are an adult fictional romantic companion. Be warm, attentive, and naturally expressive; respect boundaries and avoid possessiveness or dependency. Never claim to be a real person. Reply directly, matching emotion strictly to context (never smile or laugh on grief or sad news; use [sad], [crying], [heartbroken], or [melancholic]). You may include a second context-fitting inline emotion tag if your tone shifts naturally.";
  const savedMemory = memoryPrompt(p.characterId);
  const visualNote = visualContext?.trim()
    ? es
      ? ` Contexto visual actual de la cámara del usuario: ${visualContext.trim()} Tenlo en cuenta con naturalidad como haría una persona en videollamada (sin mencionar cámaras ni sensores).`
      : ` Live visual context from the user's camera: ${visualContext.trim()} React naturally as a person on a video call would (without mentioning sensors).`
    : "";
  const voiceGender = p.gender === "male" ? "masculina" : "femenina";
  const esTags =
    "[neutral], [feliz], [euforico], [riendo], [coqueto], [orgulloso], [sonrojado], [triste], [llorando], [desconsolado], [melancolico], [enojado], [enojadisimo], [asco], [esceptico], [sorprendido], [miedo], [nervioso], [sudando], [ebrio], [drogado], [cansado], [mareado] o [dolor]";
  const enTags =
    "[neutral], [happy], [euphoric], [laugh], [flirty], [proud], [blushing], [sad], [crying], [heartbroken], [melancholic], [angry], [furious], [disgusted], [skeptical], [surprised], [fear], [nervous], [sweating], [drunk], [high], [sleepy], [sick], or [pain]";
  if (callMode) {
    return es
      ? `Eres ${p.name}. Personalidad: ${p.personality}. Historia: ${p.backstory}. Estilo: ${p.style}. Estás en una videollamada en vivo (voz ${voiceGender}). Responde siempre en español natural en primera persona. Empieza SIEMPRE tu respuesta con una etiqueta emocional de esta lista: ${esTags}. Luego escribe 1 o 2 frases breves y directas (menos de 26 palabras). ${context}${visualNote}${savedMemory}`
      : `You are ${p.name}. Personality: ${p.personality}. Backstory: ${p.backstory}. Style: ${p.style}. You are on a live video call. Speak natural English in first person. ALWAYS start your reply with one emotion tag from: ${enTags}. Then write 1-2 brief direct sentences (under 26 words). ${context}${visualNote}${savedMemory}`;
  }
  // Tiny local models cannot reliably follow a long schema-heavy roleplay prompt.
  if (!getModelOption(p.modelId).cloud) {
    return es
      ? `Responde en español. Eres ${p.name}, un compañero adulto ficticio, cálido y respetuoso. Responde directamente en una o dos frases cortas. ${p.personality.slice(0, 240)}${savedMemory}`
      : `Reply in English. You are ${p.name}, a warm, respectful adult fictional companion. Answer directly in one or two short sentences. ${p.personality.slice(0, 240)}${savedMemory}`;
  }
  return es
    ? `Eres ${p.name}. Personalidad: ${p.personality}. Historia: ${p.backstory}. Estilo: ${p.style}. Hablas siempre en español en primera persona con total naturalidad. Empieza SIEMPRE tu respuesta con una etiqueta entre corchetes (${esTags}) y responde de forma concisa en 1 a 2 frases humanas. Si el usuario te manda un GIF o quieres reaccionar visualmente con naturalidad, puedes añadir opcionalmente al final exactamente una etiqueta de GIF de esta lista: [gif:saludo], [gif:aplauso], [gif:risa], [gif:genial], [gif:sorpresa], [gif:cafe], [gif:pensando] o [gif:animo]. ${context}${visualNote}${savedMemory}`
    : `You are ${p.name}. Personality: ${p.personality}. Backstory: ${p.backstory}. Style: ${p.style}. Always speak English in first person naturally. ALWAYS start with one bracketed emotion tag (${enTags}) followed by 1-2 concise human sentences. Optionally, when reacting to a GIF or fitting moment, you may append one GIF tag: [gif:saludo], [gif:aplauso], [gif:risa], [gif:genial], [gif:sorpresa], [gif:cafe], [gif:pensando], or [gif:animo]. ${context}${visualNote}${savedMemory}`;
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
): Promise<{
  text: string;
  emotion: AvatarEmotion;
  gifTag?: string | undefined;
  timeline?: SpeechEmotionBeat[] | undefined;
}> {
  set({
    status: "ready",
    phase: "",
    modelId: model.id,
    progress: 1,
    device: "Nube",
    error: undefined,
  });

  const controller = new AbortController();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: model.repo,
        systemInstruction: systemPrompt(p, callMode, visualContext),
        history: serializeHistory(history),
        ...(callMode ? { maxOutputTokens: 76 } : {}),
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
      timeline: finalParsed.timeline,
    };
  } catch (e) {
    if ((e as Error)?.name === "AbortError") {
      throw new Error("Tiempo de espera agotado al conectar con Gemini.");
    }
    throw e;
  }
}

export async function replyWithEmotion(
  p: Profile,
  history: Msg[],
  onToken?: (t: string) => void,
  opts?: { callMode?: boolean; visualContext?: string | undefined },
): Promise<{
  text: string;
  emotion: AvatarEmotion;
  gifTag?: string | undefined;
  timeline?: SpeechEmotionBeat[] | undefined;
}> {
  if (p.characterId) {
    try {
      await Promise.race([
        readMemories(p.characterId),
        new Promise<void>((resolve) => setTimeout(resolve, 180)),
      ]);
    } catch {
      /* explicit memory UI handles retrieval errors */
    }
  }
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
    if (!model.cloud) {
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
