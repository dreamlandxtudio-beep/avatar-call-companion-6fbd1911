// Voices: Gemini Cloud TTS (Spanish & English, WAV + real AnalyserNode lip-sync), Kokoro-82M, and device voices.
// STT: MediaRecorder + Web Audio VAD -> /api/transcribe (Gemini 3.5 Transcribe), with browser SpeechRecognition fallback.
import type { VoiceChoice } from "./store";
import { dynamicSpeechTempo, type SpeechTempoProfile } from "./voice-prosody";
import { textToVisemeProtos } from "./phonemes";

export const GEMINI_VOICES = [
  { id: "Kore", gender: "female" as const, label: "Kore · mujer, cálida y natural" },
  { id: "Zephyr", gender: "female" as const, label: "Zephyr · mujer, alegre y brillante" },
  { id: "Aoede", gender: "female" as const, label: "Aoede · mujer, melodiosa y dulce" },
  { id: "Leda", gender: "female" as const, label: "Leda · mujer, juvenil y fresca" },
  { id: "Sulafat", gender: "female" as const, label: "Sulafat · mujer, serena y elegante" },
  { id: "Autonoe", gender: "female" as const, label: "Autonoe · mujer, brillante y viva" },
  { id: "Callirrhoe", gender: "female" as const, label: "Callirrhoe · mujer, suave y tranquila" },
  { id: "Despina", gender: "female" as const, label: "Despina · mujer, cálida y cercana" },
  { id: "Puck", gender: "male" as const, label: "Puck · hombre, joven y cercano" },
  { id: "Charon", gender: "male" as const, label: "Charon · hombre, sereno y grave" },
  { id: "Fenrir", gender: "male" as const, label: "Fenrir · hombre, enérgico y firme" },
  { id: "Orus", gender: "male" as const, label: "Orus · hombre, maduro y seguro" },
  { id: "Algenib", gender: "male" as const, label: "Algenib · hombre, profundo y claro" },
  { id: "Achird", gender: "male" as const, label: "Achird · hombre, amigable y ligero" },
  { id: "Gacrux", gender: "female" as const, label: "Gacrux · mujer, expresiva y clara" },
  { id: "Umbriel", gender: "male" as const, label: "Umbriel · hombre, pausado y reflexivo" },
];

export const KOKORO_VOICES = [
  { id: "af_heart", gender: "female" as const, label: "Heart · mujer, cálida (EN)" },
  { id: "af_bella", gender: "female" as const, label: "Bella · mujer, expresiva (EN)" },
  { id: "af_nova", gender: "female" as const, label: "Nova · mujer, moderna y clara (EN)" },
  { id: "af_sky", gender: "female" as const, label: "Sky · mujer, dulce y ligera (EN)" },
  { id: "af_sarah", gender: "female" as const, label: "Sarah · mujer, natural (EN)" },
  { id: "af_nicole", gender: "female" as const, label: "Nicole · mujer, susurrada (EN)" },
  { id: "af_aoede", gender: "female" as const, label: "Aoede · mujer, suave (EN)" },
  { id: "af_kore", gender: "female" as const, label: "Kore · mujer, equilibrada (EN)" },
  { id: "bf_emma", gender: "female" as const, label: "Emma · mujer británica (EN)" },
  { id: "bf_isabella", gender: "female" as const, label: "Isabella · mujer británica (EN)" },
  { id: "am_michael", gender: "male" as const, label: "Michael · hombre, grave (EN)" },
  { id: "am_fenrir", gender: "male" as const, label: "Fenrir · hombre, intenso (EN)" },
  { id: "am_adam", gender: "male" as const, label: "Adam · hombre, firme (EN)" },
  { id: "am_liam", gender: "male" as const, label: "Liam · hombre, joven (EN)" },
  { id: "bm_george", gender: "male" as const, label: "George · hombre británico (EN)" },
  { id: "bm_daniel", gender: "male" as const, label: "Daniel · hombre británico (EN)" },
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let kokoro: any = null;
let kLoading: Promise<void> | null = null;
export function loadKokoro(onProgress?: (p: number) => void) {
  if (kLoading) return kLoading;
  kLoading = (async () => {
    const { env } = await import("@huggingface/transformers");
    env.useBrowserCache = true;
    const { KokoroTTS } = await import("kokoro-js");
    kokoro = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
      dtype: "q8",
      device: "wasm",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      progress_callback: (p: any) => {
        if (p.status === "progress" && p.total) onProgress?.(p.loaded / p.total);
      },
    });
    // Load the voice bank too, not just the neural weights, so the first reply works offline.
    await kokoro.generate("Hello.", { voice: "af_heart" });
    await kokoro.generate("Hello.", { voice: "am_michael" });
  })().catch((e) => {
    kLoading = null;
    throw e;
  });
  return kLoading;
}

const FEMALE_VOICE_HINTS =
  /female|mujer|femenin|helena|elena|sofia|monica|mónica|paulina|laura|lucia|lucía|sabina|paloma|conchita|penelope|serena|aria|jenny|samantha|victoria|karen|ava|allison|google español|google us english/i;
const MALE_VOICE_HINTS =
  /male|hombre|masculin|jorge|diego|pablo|carlos|enrique|raul|raúl|alonso|mateo|daniel|alex|thomas|guy|ryan|christopher|davis/i;

export function systemVoices(lang: string, gender?: "female" | "male") {
  if (typeof speechSynthesis === "undefined") return [];
  const all = speechSynthesis.getVoices();
  const pre = lang.slice(0, 2).toLowerCase();
  const matching = all.filter((v) => v.lang.toLowerCase().startsWith(pre));
  const pool = matching.length > 0 ? matching : all;
  return [...pool].sort((a, b) => scoreVoice(b, lang, gender) - scoreVoice(a, lang, gender));
}

function scoreVoice(v: SpeechSynthesisVoice, lang: string, gender?: "female" | "male"): number {
  const name = v.name || "";
  let s = 0;
  if (/natural|neural|online.*natural|microsoft.*online|google|siri|premium|enhanced/i.test(name)) {
    s += 10;
  }
  if (/espeak|festival|mbrola|compact|android speech|piper/i.test(name)) {
    s -= 10;
  }
  if (v.lang.toLowerCase() === lang.toLowerCase()) {
    s += 3;
  } else if (v.lang.toLowerCase().startsWith(lang.slice(0, 2).toLowerCase())) {
    s += 1;
  }
  if (gender === "female") {
    if (FEMALE_VOICE_HINTS.test(name)) s += 4;
    if (MALE_VOICE_HINTS.test(name) && !FEMALE_VOICE_HINTS.test(name)) s -= 3;
  } else if (gender === "male") {
    if (MALE_VOICE_HINTS.test(name)) s += 4;
    if (FEMALE_VOICE_HINTS.test(name) && !MALE_VOICE_HINTS.test(name)) s -= 3;
  }
  if (!v.localService) s += 1;
  return s;
}

let ctx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let current: AudioBufferSourceNode | null = null;
let speakAbort: AbortController | null = null;
let fakeLevel = 0;
let fakeTimer: ReturnType<typeof setTimeout> | null = null;
let currentMicLevel = 0;

export type AudioVisemes = {
  level: number;
  open: number;
  wide: number;
  round: number;
  press?: number | undefined;
  pucker?: number | undefined;
  tongue?: number | undefined;
  labiodental?: number | undefined;
  stress?: number | undefined;
};

type VisemeCue = {
  tStart: number;
  tEnd: number;
  open: number;
  wide: number;
  round: number;
  press: number;
  pucker: number;
  tongue: number;
  labiodental: number;
  stress: number;
};

type VisemeTimeline = {
  cues: VisemeCue[];
  totalSec: number;
  wordStartCues: Array<{ charIdx: number; sec: number }>;
  startedAtMs: number;
  startedAudioTime: number;
  useAudioClock: boolean;
  rate: number;
};

let activeTimeline: VisemeTimeline | null = null;

function detectAudioSpeechWindow(buf: AudioBuffer): { startSec: number; endSec: number } {
  const dur = Math.max(0.4, buf.duration || 1);
  if (buf.numberOfChannels === 0 || buf.length < 64) {
    return { startSec: 0, endSec: dur };
  }
  const ch = buf.getChannelData(0);
  const sr = buf.sampleRate || 24000;
  const step = Math.max(128, Math.round(sr * 0.015));
  let peak = 0;
  const energies: number[] = [];
  for (let i = 0; i < ch.length; i += step) {
    const end = Math.min(ch.length, i + step);
    let s = 0;
    for (let j = i; j < end; j++) {
      const v = ch[j]!;
      s += v * v;
    }
    const rms = Math.sqrt(s / Math.max(1, end - i));
    if (rms > peak) peak = rms;
    energies.push(rms);
  }
  const thresh = Math.max(0.012, peak * 0.085);
  let firstIdx = 0;
  let lastIdx = Math.max(0, energies.length - 1);
  while (firstIdx < energies.length && energies[firstIdx]! < thresh) firstIdx++;
  while (lastIdx > firstIdx && energies[lastIdx]! < thresh) lastIdx--;
  if (firstIdx >= lastIdx) {
    return { startSec: 0, endSec: dur };
  }
  const startSec = Math.max(0, (firstIdx * step) / sr - 0.03);
  const endSec = Math.min(dur, Math.max(startSec + 0.3, ((lastIdx + 1) * step) / sr + 0.05));
  return { startSec, endSec };
}

export function buildVisemeTimeline(
  text: string,
  estimatedSec: number,
  rate = 1,
  speechWindow?: { startSec: number; endSec: number },
): VisemeTimeline {
  const { protos, wordStartChars } = textToVisemeProtos(text);
  const cues: VisemeCue[] = [];
  const wordStartCues: Array<{ charIdx: number; sec: number }> = wordStartChars.map((charIdx) => ({
    charIdx,
    sec: 0,
  }));

  let totalW = 0;
  for (const p of protos) totalW += p.weight;
  const totalDur = Math.max(0.4, estimatedSec);
  const winStart = Math.max(0, speechWindow?.startSec ?? 0);
  const winEnd = Math.min(totalDur, Math.max(winStart + 0.35, speechWindow?.endSec ?? totalDur));
  const activeSpan = Math.max(0.35, winEnd - winStart);

  if (winStart > 0.01) {
    cues.push({
      tStart: 0,
      tEnd: winStart,
      open: 0,
      wide: 0,
      round: 0,
      press: 0.12,
      pucker: 0,
      tongue: 0,
      labiodental: 0,
      stress: 0,
    });
  }

  let curT = winStart;
  let wPtr = 0;
  for (const p of protos) {
    const dt = (p.weight / Math.max(1, totalW)) * activeSpan;
    while (wPtr < wordStartCues.length && wordStartCues[wPtr]!.charIdx <= p.charIdx) {
      wordStartCues[wPtr]!.sec = curT;
      wPtr++;
    }
    cues.push({
      tStart: curT,
      tEnd: curT + dt,
      open: p.open,
      wide: p.wide,
      round: p.round,
      press: p.press,
      pucker: p.pucker,
      tongue: p.tongue,
      labiodental: p.labiodental,
      stress: p.stress,
    });
    curT += dt;
  }

  if (winEnd < totalDur - 0.01) {
    cues.push({
      tStart: winEnd,
      tEnd: totalDur,
      open: 0,
      wide: 0,
      round: 0,
      press: 0.12,
      pucker: 0,
      tongue: 0,
      labiodental: 0,
      stress: 0,
    });
  }

  return {
    cues,
    totalSec: totalDur,
    wordStartCues,
    startedAtMs: typeof performance !== "undefined" ? performance.now() : 0,
    startedAudioTime: ctx ? ctx.currentTime : 0,
    useAudioClock: false,
    rate,
  };
}

export function sampleVisemeTimeline(
  tl: VisemeTimeline | null,
  elapsedOverrideSec?: number,
): Omit<VisemeCue, "tStart" | "tEnd"> | null {
  if (!tl || tl.cues.length === 0) return null;
  const elapsed =
    elapsedOverrideSec !== undefined
      ? elapsedOverrideSec
      : tl.useAudioClock && ctx
        ? Math.max(0, ctx.currentTime - tl.startedAudioTime)
        : typeof performance !== "undefined"
          ? Math.max(0, (performance.now() - tl.startedAtMs) * 0.001)
          : 0;
  if (elapsed >= tl.totalSec + 0.35) return null;

  const t = Math.min(tl.totalSec - 0.001, Math.max(0, elapsed));
  // Binary search active cue
  let lo = 0;
  let hi = tl.cues.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (tl.cues[mid]!.tEnd < t) lo = mid + 1;
    else hi = mid;
  }
  const cur = tl.cues[lo]!;
  const next = tl.cues[Math.min(tl.cues.length - 1, lo + 1)]!;
  const prev = tl.cues[Math.max(0, lo - 1)]!;
  const span = Math.max(0.015, cur.tEnd - cur.tStart);
  const f = Math.max(0, Math.min(1, (t - cur.tStart) / span));
  // Crisp coarticulation preserving phoneme distinctness at center of cue
  const neighbor = f < 0.28 ? prev : f > 0.72 ? next : cur;
  const wN = f < 0.28 ? (0.28 - f) * 1.1 : f > 0.72 ? (f - 0.72) * 1.1 : 0;
  const wC = 1 - wN;

  return {
    open: cur.open * wC + neighbor.open * wN,
    wide: cur.wide * wC + neighbor.wide * wN,
    round: cur.round * wC + neighbor.round * wN,
    press: cur.press * wC + neighbor.press * wN,
    pucker: cur.pucker * wC + neighbor.pucker * wN,
    tongue: cur.tongue * wC + neighbor.tongue * wN,
    labiodental: cur.labiodental * wC + neighbor.labiodental * wN,
    stress: cur.stress * wC + neighbor.stress * wN,
  };
}

let timeBuf: Uint8Array<ArrayBuffer> | null = null;
let freqBuf: Uint8Array<ArrayBuffer> | null = null;

export function audioLevel(): number {
  if (analyser && current) {
    if (!timeBuf || timeBuf.length !== analyser.fftSize) {
      timeBuf = new Uint8Array(analyser.fftSize);
    }
    analyser.getByteTimeDomainData(timeBuf);
    let s = 0;
    for (const x of timeBuf) s += ((x - 128) / 128) ** 2;
    return Math.min(1, Math.sqrt(s / timeBuf.length) * 5.2);
  }
  return fakeLevel;
}

export function audioVisemes(): AudioVisemes {
  const cue = sampleVisemeTimeline(activeTimeline);

  if (analyser && current && ctx) {
    const level = audioLevel();
    if (!freqBuf || freqBuf.length !== analyser.frequencyBinCount) {
      freqBuf = new Uint8Array(analyser.frequencyBinCount);
    }
    analyser.getByteFrequencyData(freqBuf);
    const binHz = ctx.sampleRate / analyser.fftSize;
    const avgRange = (hzLo: number, hzHi: number) => {
      const i0 = Math.max(1, Math.floor(hzLo / binHz));
      const i1 = Math.min(freqBuf!.length - 1, Math.ceil(hzHi / binHz));
      let sum = 0;
      let count = 0;
      for (let i = i0; i <= i1; i++) {
        sum += freqBuf![i]!;
        count++;
      }
      return count > 0 ? sum / (count * 255) : 0;
    };

    const low = avgRange(120, 420);
    const mid = avgRange(420, 1250);
    const high = avgRange(1250, 3400);
    const sib = avgRange(3400, 6800);

    // Acoustic RMS drop triggers brief bilabial seal (P/B/M) when text timeline agrees or sound pauses briefly
    const silenceDip = level < 0.065 ? (0.065 - level) / 0.065 : 0;
    const acousticOpen = Math.min(1, level * 0.52 + mid * 0.88 + low * 0.34 - sib * 0.25);
    const acousticWide = Math.min(
      1,
      Math.max(0, (high * 1.48 - low * 0.28 + sib * 0.35) * (0.4 + level)),
    );
    const acousticRound = Math.min(1, Math.max(0, (low * 1.22 - high * 0.62) * (0.4 + level)));

    if (cue) {
      const voiceGate = Math.min(1, level * 2.4);
      // Preserve bilabial seal (P/B/M) strongly when cue calls for it instead of blurring it with acoustic bleed
      const press =
        cue.press > 0.45
          ? Math.max(cue.press * 0.92, silenceDip * cue.press)
          : Math.max(cue.press * 0.7, silenceDip * 0.25);
      const labiodental = cue.labiodental * (0.55 + 0.45 * voiceGate);
      const open = Math.max(
        0,
        (cue.open * (0.45 + 0.55 * voiceGate) + acousticOpen * 0.42) *
          (1 - press * 0.88) *
          (1 - labiodental * 0.55),
      );
      const wide = Math.min(1, cue.wide * (0.52 + 0.48 * voiceGate) + acousticWide * 0.35);
      const round = Math.min(1, cue.round * (0.56 + 0.44 * voiceGate) + acousticRound * 0.34);
      const pucker = Math.min(1, round * 0.48 + cue.pucker * (0.5 + 0.5 * voiceGate));
      const tongue = Math.min(1, cue.tongue * (0.42 + 0.58 * voiceGate));
      return {
        level,
        open,
        wide,
        round,
        press,
        pucker,
        tongue,
        labiodental,
        stress: cue.stress * voiceGate,
      };
    }

    return {
      level,
      open: Math.max(0, acousticOpen),
      wide: acousticWide,
      round: acousticRound,
      press: silenceDip * 0.35,
      pucker: acousticRound * 0.65,
      tongue: sib > 0.22 ? 0.22 : 0,
      labiodental: 0,
      stress: Math.min(1, level * 0.7),
    };
  }

  if (cue && fakeLevel > 0.01) {
    const dyn = 0.75 + Math.min(0.4, fakeLevel * 0.45);
    return {
      level: Math.max(fakeLevel, cue.open * 0.85),
      open: Math.min(1, cue.open * dyn * (1 - cue.press * 0.88) * (1 - cue.labiodental * 0.55)),
      wide: Math.min(1, cue.wide * dyn),
      round: Math.min(1, cue.round * dyn),
      press: cue.press,
      pucker: cue.pucker,
      tongue: cue.tongue,
      labiodental: cue.labiodental,
      stress: cue.stress,
    };
  }

  const t = typeof performance !== "undefined" ? performance.now() * 0.001 : 0;
  return {
    level: fakeLevel,
    open: fakeLevel,
    wide: fakeLevel > 0.05 ? (0.5 + 0.5 * Math.sin(t * 7.3)) * fakeLevel * 0.7 : 0,
    round: fakeLevel > 0.05 ? (0.5 + 0.5 * Math.cos(t * 5.7)) * fakeLevel * 0.55 : 0,
    press: 0,
    pucker: 0,
    tongue: 0,
    labiodental: 0,
    stress: 0,
  };
}

export function micLevel(): number {
  return currentMicLevel;
}

export function unlockAudio() {
  if (typeof window === "undefined") return null;
  if (!ctx) {
    const AC =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
    analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    analyser.connect(ctx.destination);
  }
  if (ctx.state === "suspended") {
    ctx.resume().catch(() => {});
  }
  try {
    const silent = ctx.createBuffer(1, 1, 22050);
    const src = ctx.createBufferSource();
    src.buffer = silent;
    src.connect(ctx.destination);
    src.start();
  } catch {
    /* ignore */
  }
  if (typeof speechSynthesis !== "undefined") {
    speechSynthesis.getVoices();
  }
  return ctx;
}

export function stopSpeaking() {
  speakAbort?.abort();
  speakAbort = null;
  activeTimeline = null;
  try {
    current?.stop();
  } catch {
    /* */
  }
  current = null;
  fakeLevel = 0;
  if (fakeTimer) {
    clearTimeout(fakeTimer);
    fakeTimer = null;
  }
  if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
}

async function playAudioBuffer(
  buf: AudioBuffer,
  c: AudioContext,
  signal: AbortSignal,
  spokenText?: string,
  playbackRate = 1,
): Promise<boolean> {
  if (signal.aborted) return false;
  if (c.state === "suspended") {
    await Promise.race([c.resume().catch(() => {}), new Promise<void>((r) => setTimeout(r, 450))]);
  }
  if (c.state !== "running") return false;
  const clampedRate = Math.max(0.92, Math.min(1.14, playbackRate));
  const src = c.createBufferSource();
  src.buffer = buf;
  src.playbackRate.value = clampedRate;
  src.connect(analyser!);
  current = src;
  const effectiveDuration = (buf.duration || 3) / clampedRate;
  if (spokenText) {
    const rawWindow = detectAudioSpeechWindow(buf);
    const speechWindow = {
      startSec: rawWindow.startSec / clampedRate,
      endSec: rawWindow.endSec / clampedRate,
    };
    const tl = buildVisemeTimeline(spokenText, effectiveDuration, clampedRate, speechWindow);
    tl.useAudioClock = true;
    tl.startedAudioTime = c.currentTime;
    activeTimeline = tl;
  }
  const maxMs = Math.min(25000, Math.max(1500, Math.round(effectiveDuration * 1000) + 1200));
  await new Promise<void>((resolve) => {
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      clearTimeout(watchdog);
      signal.removeEventListener("abort", onAbort);
      if (current === src) {
        current = null;
        activeTimeline = null;
      }
      resolve();
    };
    const onAbort = () => {
      try {
        src.stop();
      } catch {
        /* */
      }
      done();
    };
    const watchdog = setTimeout(() => {
      try {
        src.stop();
      } catch {
        /* */
      }
      done();
    }, maxMs);
    signal.addEventListener("abort", onAbort, { once: true });
    src.onended = done;
    try {
      src.start();
    } catch {
      done();
    }
  });
  return true;
}

const ttsCache = new Map<string, ArrayBuffer>();

function decodePcm16Fallback(c: AudioContext, arr: ArrayBuffer): AudioBuffer | null {
  if (arr.byteLength < 64) return null;
  const view = new DataView(arr);
  const isRiff =
    arr.byteLength >= 44 &&
    view.getUint8(0) === 0x52 &&
    view.getUint8(1) === 0x49 &&
    view.getUint8(2) === 0x46 &&
    view.getUint8(3) === 0x46;
  const offset = isRiff ? 44 : 0;
  const sampleRate = isRiff ? Math.max(8000, Math.min(48000, view.getUint32(24, true))) : 24000;
  const byteLen = (arr.byteLength - offset) & ~1;
  if (byteLen < 32) return null;
  const int16 = new Int16Array(arr, offset, byteLen / 2);
  const buf = c.createBuffer(1, int16.length, sampleRate);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < int16.length; i++) {
    ch[i] = int16[i]! / 32768;
  }
  return buf;
}

async function speakViaGemini(
  clean: string,
  voiceId: string,
  lang: string,
  signal: AbortSignal,
  emotion?: string,
  tempoProfile?: SpeechTempoProfile,
): Promise<boolean> {
  const tempoTag = tempoProfile?.tempo ?? "normal";
  const cacheKey = `${voiceId}|${lang}|${emotion ?? ""}|${tempoTag}|${clean}`;
  let arr = ttsCache.get(cacheKey);
  if (!arr) {
    const timeout = new AbortController();

    const onParentAbort = () => timeout.abort();
    signal.addEventListener("abort", onParentAbort, { once: true });

    try {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: timeout.signal,
        body: JSON.stringify({
          text: clean,
          voiceName: voiceId || "Kore",
          language: lang,
          tempo: tempoTag,
          ...(emotion ? { emotion } : {}),
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Servicio de voz: ${res.status}`);
      }
      arr = await res.arrayBuffer();
      if (arr.byteLength >= 64 && ttsCache.size < 24) {
        ttsCache.set(cacheKey, arr);
      }
    } catch (error) {
      if (signal.aborted) return true;
      throw error;
    } finally {
      signal.removeEventListener("abort", onParentAbort);
    }
  }

  if (signal.aborted || !arr || arr.byteLength < 64) return signal.aborted;
  const c = unlockAudio();
  if (!c) return false;

  // Subtle acoustic tempo micro-nudge preserves natural vocal pitch while making fast/slow cues unmistakably distinct
  const playbackRate = tempoProfile
    ? tempoProfile.tempo === "fast"
      ? 1.07
      : tempoProfile.tempo === "brisk"
        ? 1.03
        : tempoProfile.tempo === "slow"
          ? 0.95
          : 1
    : 1;

  try {
    const decoded = await c.decodeAudioData(arr.slice(0));
    return await playAudioBuffer(decoded, c, signal, clean, playbackRate);
  } catch {
    const fallbackBuf = decodePcm16Fallback(c, arr);
    if (!fallbackBuf) return false;
    return await playAudioBuffer(fallbackBuf, c, signal, clean, playbackRate);
  }
}

export function defaultGeminiVoiceForGender(gender?: "female" | "male"): string {
  return gender === "male" ? "Charon" : "Kore";
}

/**
 * Synthesizes realistic aspirated vocal laughter pulses ("ha-ha-ha-ha") routed through the active
 * AnalyserNode + viseme timeline so audible laughter drives authentic lip/jaw bounce.
 */
export async function playVocalLaugh(gender?: "female" | "male", bursts = 4): Promise<void> {
  if (typeof window === "undefined") return;
  const c = unlockAudio();
  if (!c) return;
  if (c.state === "suspended") {
    await Promise.race([c.resume().catch(() => {}), new Promise<void>((r) => setTimeout(r, 250))]);
  }
  if (c.state !== "running") return;

  stopSpeaking();
  const ctrl = new AbortController();
  speakAbort = ctrl;

  const count = Math.max(3, Math.min(6, Math.round(bursts)));
  const sr = c.sampleRate || 24000;
  const beatDur = 0.21;
  const totalDur = count * beatDur + 0.26;
  const len = Math.max(1024, Math.floor(totalDur * sr));

  let audioBuf: AudioBuffer | null = null;
  if (typeof OfflineAudioContext !== "undefined") {
    const off = new OfflineAudioContext(1, len, sr);
    const baseF0 = gender === "male" ? 128 : 218;
    const masterGain = off.createGain();
    masterGain.gain.value = 0.56;
    masterGain.connect(off.destination);

    // Parallel formant filters modeling open warm /a/ -> /eh/ vocal tract resonance
    const f1 = off.createBiquadFilter();
    f1.type = "bandpass";
    f1.frequency.value = gender === "male" ? 680 : 760;
    f1.Q.value = 4.6;

    const f2 = off.createBiquadFilter();
    f2.type = "bandpass";
    f2.frequency.value = gender === "male" ? 1180 : 1340;
    f2.Q.value = 6.2;

    const f1Gain = off.createGain();
    f1Gain.gain.value = 0.85;
    const f2Gain = off.createGain();
    f2Gain.gain.value = 0.42;
    f1.connect(f1Gain);
    f2.connect(f2Gain);
    f1Gain.connect(masterGain);
    f2Gain.connect(masterGain);

    // Soft breath aspiration noise buffer
    const noiseLen = Math.floor(0.28 * sr);
    const noiseBuf = off.createBuffer(1, noiseLen, sr);
    const nData = noiseBuf.getChannelData(0);
    let lp = 0;
    for (let i = 0; i < noiseLen; i++) {
      lp = lp * 0.72 + (Math.random() * 2 - 1) * 0.28;
      nData[i] = lp;
    }

    for (let b = 0; b < count; b++) {
      const start = 0.03 + b * beatDur;
      const pulseDur = 0.145;
      const decay = 1 - b * 0.12;
      const freq = baseF0 * (1 - b * 0.055);

      const osc = off.createOscillator();
      const sub = off.createOscillator();
      osc.type = "sawtooth";
      sub.type = "triangle";
      osc.frequency.setValueAtTime(freq * 1.07, start);
      osc.frequency.exponentialRampToValueAtTime(Math.max(70, freq * 0.86), start + pulseDur);
      sub.frequency.setValueAtTime(freq * 1.07, start);
      sub.frequency.exponentialRampToValueAtTime(Math.max(70, freq * 0.86), start + pulseDur);

      const env = off.createGain();
      env.gain.setValueAtTime(0.0001, start + 0.014);
      env.gain.linearRampToValueAtTime(0.26 * decay, start + 0.042);
      env.gain.exponentialRampToValueAtTime(0.001, start + pulseDur);

      osc.connect(env);
      sub.connect(env);
      env.connect(f1);
      env.connect(f2);

      // Breath "h" onset burst right before voiced vowel of each "ha!"
      const noise = off.createBufferSource();
      noise.buffer = noiseBuf;
      const nFilter = off.createBiquadFilter();
      nFilter.type = "bandpass";
      nFilter.frequency.value = 1450;
      nFilter.Q.value = 1.6;
      const nEnv = off.createGain();
      nEnv.gain.setValueAtTime(0.0001, start);
      nEnv.gain.linearRampToValueAtTime(0.22 * decay, start + 0.018);
      nEnv.gain.exponentialRampToValueAtTime(0.001, start + 0.068);
      noise.connect(nFilter);
      nFilter.connect(nEnv);
      nEnv.connect(masterGain);

      osc.start(start + 0.01);
      sub.start(start + 0.01);
      osc.stop(start + pulseDur + 0.02);
      sub.stop(start + pulseDur + 0.02);
      noise.start(start);
      noise.stop(start + 0.08);
    }

    audioBuf = await off.startRendering();
  }

  if (!audioBuf || ctrl.signal.aborted) return;
  const laughText = Array.from({ length: count }, () => "ja").join(" ");
  await playAudioBuffer(audioBuf, c, ctrl.signal, laughText);
}

export function isSpeaking(): boolean {
  return !!(current || fakeTimer || speakAbort);
}

/**
 * Synthesizes tremulous soft crying sobs and shaky inhales routed through the active AnalyserNode.
 */
export async function playVocalCry(gender?: "female" | "male", sobs = 2): Promise<void> {
  if (typeof window === "undefined") return;
  const c = unlockAudio();
  if (!c) return;
  if (c.state === "suspended") {
    await Promise.race([c.resume().catch(() => {}), new Promise<void>((r) => setTimeout(r, 250))]);
  }
  if (c.state !== "running") return;

  stopSpeaking();
  const ctrl = new AbortController();
  speakAbort = ctrl;

  const count = Math.max(2, Math.min(4, Math.round(sobs)));
  const sr = c.sampleRate || 24000;
  const beatDur = 0.34;
  const totalDur = count * beatDur + 0.32;
  const len = Math.max(1024, Math.floor(totalDur * sr));

  let audioBuf: AudioBuffer | null = null;
  if (typeof OfflineAudioContext !== "undefined") {
    const off = new OfflineAudioContext(1, len, sr);
    const baseF0 = gender === "male" ? 132 : 242;
    const masterGain = off.createGain();
    masterGain.gain.value = 0.44;
    masterGain.connect(off.destination);

    const f1 = off.createBiquadFilter();
    f1.type = "bandpass";
    f1.frequency.value = gender === "male" ? 410 : 480;
    f1.Q.value = 4.2;

    const f2 = off.createBiquadFilter();
    f2.type = "bandpass";
    f2.frequency.value = gender === "male" ? 960 : 1120;
    f2.Q.value = 5.0;

    f1.connect(masterGain);
    f2.connect(masterGain);

    const nLen = Math.floor(0.32 * sr);
    const noiseBuf = off.createBuffer(1, nLen, sr);
    const nData = noiseBuf.getChannelData(0);
    let lp = 0;
    for (let i = 0; i < nLen; i++) {
      lp = lp * 0.78 + (Math.random() * 2 - 1) * 0.22;
      nData[i] = lp;
    }

    for (let b = 0; b < count; b++) {
      const start = 0.04 + b * beatDur;
      const osc = off.createOscillator();
      osc.type = "triangle";
      const freq = baseF0 * (1 - b * 0.04);
      osc.frequency.setValueAtTime(freq * 1.06, start);
      osc.frequency.linearRampToValueAtTime(freq * 0.84, start + 0.22);

      const vib = off.createOscillator();
      vib.frequency.value = 6.2;
      const vibGain = off.createGain();
      vibGain.gain.value = freq * 0.055;
      vib.connect(vibGain);
      vibGain.connect(osc.frequency);

      const env = off.createGain();
      env.gain.setValueAtTime(0.0001, start);
      env.gain.linearRampToValueAtTime(0.22, start + 0.06);
      env.gain.exponentialRampToValueAtTime(0.001, start + 0.23);

      osc.connect(env);
      env.connect(f1);
      env.connect(f2);
      osc.start(start);
      vib.start(start);
      osc.stop(start + 0.25);
      vib.stop(start + 0.25);

      // Shaky breathy inhale sniff right after sob pulse
      const breathStart = start + 0.18;
      const noise = off.createBufferSource();
      noise.buffer = noiseBuf;
      const bf = off.createBiquadFilter();
      bf.type = "bandpass";
      bf.frequency.value = 1850;
      bf.Q.value = 2.4;
      const nEnv = off.createGain();
      nEnv.gain.setValueAtTime(0.0001, breathStart);
      nEnv.gain.linearRampToValueAtTime(0.14, breathStart + 0.04);
      nEnv.gain.exponentialRampToValueAtTime(0.001, breathStart + 0.14);
      noise.connect(bf);
      bf.connect(nEnv);
      nEnv.connect(masterGain);
      noise.start(breathStart);
      noise.stop(breathStart + 0.16);
    }

    audioBuf = await off.startRendering();
  }

  if (!audioBuf || ctrl.signal.aborted) return;
  const sobText = Array.from({ length: count }, () => "uh").join(" ");
  await playAudioBuffer(audioBuf, c, ctrl.signal, sobText);
}

/**
 * Synthesizes a soft realistic breathy exhale sigh ("ahhh").
 */
export async function playVocalSigh(gender?: "female" | "male"): Promise<void> {
  if (typeof window === "undefined") return;
  const c = unlockAudio();
  if (!c) return;
  if (c.state === "suspended") {
    await Promise.race([c.resume().catch(() => {}), new Promise<void>((r) => setTimeout(r, 250))]);
  }
  if (c.state !== "running") return;

  stopSpeaking();
  const ctrl = new AbortController();
  speakAbort = ctrl;

  const sr = c.sampleRate || 24000;
  const totalDur = 0.78;
  const len = Math.max(1024, Math.floor(totalDur * sr));

  let audioBuf: AudioBuffer | null = null;
  if (typeof OfflineAudioContext !== "undefined") {
    const off = new OfflineAudioContext(1, len, sr);
    const masterGain = off.createGain();
    masterGain.gain.value = 0.4;
    masterGain.connect(off.destination);

    const noiseBuf = off.createBuffer(1, len, sr);
    const nData = noiseBuf.getChannelData(0);
    let lp = 0;
    for (let i = 0; i < len; i++) {
      lp = lp * 0.76 + (Math.random() * 2 - 1) * 0.24;
      nData[i] = lp;
    }

    const noise = off.createBufferSource();
    noise.buffer = noiseBuf;
    const tract = off.createBiquadFilter();
    tract.type = "bandpass";
    tract.frequency.setValueAtTime(gender === "male" ? 720 : 940, 0);
    tract.frequency.exponentialRampToValueAtTime(gender === "male" ? 460 : 580, totalDur - 0.06);
    tract.Q.value = 2.0;

    const nEnv = off.createGain();
    nEnv.gain.setValueAtTime(0.0001, 0.02);
    nEnv.gain.linearRampToValueAtTime(0.32, 0.14);
    nEnv.gain.exponentialRampToValueAtTime(0.001, totalDur - 0.04);

    noise.connect(tract);
    tract.connect(nEnv);
    nEnv.connect(masterGain);

    const hum = off.createOscillator();
    hum.type = "triangle";
    const baseF0 = gender === "male" ? 115 : 198;
    hum.frequency.setValueAtTime(baseF0, 0.03);
    hum.frequency.exponentialRampToValueAtTime(baseF0 * 0.72, 0.56);
    const hFilter = off.createBiquadFilter();
    hFilter.type = "lowpass";
    hFilter.frequency.value = 480;
    const hEnv = off.createGain();
    hEnv.gain.setValueAtTime(0.0001, 0.03);
    hEnv.gain.linearRampToValueAtTime(0.09, 0.12);
    hEnv.gain.exponentialRampToValueAtTime(0.001, 0.58);
    hum.connect(hFilter);
    hFilter.connect(hEnv);
    hEnv.connect(masterGain);

    noise.start(0);
    hum.start(0.03);
    noise.stop(totalDur);
    hum.stop(0.6);

    audioBuf = await off.startRendering();
  }

  if (!audioBuf || ctrl.signal.aborted) return;
  await playAudioBuffer(audioBuf, c, ctrl.signal, "aaah");
}

/**
 * Synthesizes a quick startled breath intake gasp for surprise or fear.
 */
export async function playVocalGasp(gender?: "female" | "male"): Promise<void> {
  if (typeof window === "undefined") return;
  const c = unlockAudio();
  if (!c) return;
  if (c.state === "suspended") {
    await Promise.race([c.resume().catch(() => {}), new Promise<void>((r) => setTimeout(r, 250))]);
  }
  if (c.state !== "running") return;

  stopSpeaking();
  const ctrl = new AbortController();
  speakAbort = ctrl;

  const sr = c.sampleRate || 24000;
  const totalDur = 0.34;
  const len = Math.max(1024, Math.floor(totalDur * sr));

  let audioBuf: AudioBuffer | null = null;
  if (typeof OfflineAudioContext !== "undefined") {
    const off = new OfflineAudioContext(1, len, sr);
    const masterGain = off.createGain();
    masterGain.gain.value = 0.45;
    masterGain.connect(off.destination);

    const noiseBuf = off.createBuffer(1, len, sr);
    const nData = noiseBuf.getChannelData(0);
    let lp = 0;
    for (let i = 0; i < len; i++) {
      lp = lp * 0.65 + (Math.random() * 2 - 1) * 0.35;
      nData[i] = lp;
    }

    const noise = off.createBufferSource();
    noise.buffer = noiseBuf;
    const bf = off.createBiquadFilter();
    bf.type = "bandpass";
    bf.frequency.setValueAtTime(gender === "male" ? 1050 : 1280, 0.01);
    bf.frequency.exponentialRampToValueAtTime(gender === "male" ? 1850 : 2250, 0.24);
    bf.Q.value = 2.6;

    const env = off.createGain();
    env.gain.setValueAtTime(0.0001, 0.01);
    env.gain.linearRampToValueAtTime(0.36, 0.09);
    env.gain.exponentialRampToValueAtTime(0.001, 0.29);

    noise.connect(bf);
    bf.connect(env);
    env.connect(masterGain);
    noise.start(0.01);
    noise.stop(totalDur);

    audioBuf = await off.startRendering();
  }

  if (!audioBuf || ctrl.signal.aborted) return;
  await playAudioBuffer(audioBuf, c, ctrl.signal, "oh");
}

export async function playEmotionVocalCue(
  emotion: string,
  gender?: "female" | "male",
): Promise<void> {
  if (emotion === "laugh") {
    await playVocalLaugh(gender, 3);
  } else if (emotion === "crying" || emotion === "heartbroken") {
    await playVocalCry(gender, 2);
  } else if (emotion === "surprised" || emotion === "fear") {
    await playVocalGasp(gender);
  } else if (emotion === "sad" || emotion === "melancholic" || emotion === "sleepy") {
    await playVocalSigh(gender);
  }
}

export async function speak(
  text: string,
  voice: VoiceChoice,
  lang: string,
  emotion?: string,
  opts?: { localOnly?: boolean | undefined; gender?: "female" | "male" | undefined },
): Promise<void> {
  stopSpeaking();
  const clean = text
    .replace(/[*_#`]/g, "")
    .replace(/\[[^\]]*\]/g, "")
    .replace(/\([^)]*\)/g, "")
    .trim();
  if (!clean) return;

  const ctrl = new AbortController();
  speakAbort = ctrl;

  const tempo = dynamicSpeechTempo(clean, emotion);

  if (voice.engine === "gemini" && !opts?.localOnly) {
    try {
      const geminiVoiceId = voice.id || defaultGeminiVoiceForGender(opts?.gender);
      const ok = await speakViaGemini(clean, geminiVoiceId, lang, ctrl.signal, emotion, tempo);
      if (ok || ctrl.signal.aborted) return;
    } catch (e) {
      if (ctrl.signal.aborted) return;
      console.warn("Gemini TTS unavailable, falling back to device speech:", e);
    }
  }

  if (voice.engine === "kokoro") {
    try {
      await loadKokoro();
      if (ctrl.signal.aborted) return;
      const kokoroVoiceId = voice.id || (opts?.gender === "male" ? "am_michael" : "af_heart");
      const a = await kokoro.generate(clean, {
        voice: kokoroVoiceId,
        speed: tempo.rate,
      });
      if (ctrl.signal.aborted) return;
      const c = unlockAudio();
      if (c) {
        const buf = c.createBuffer(1, a.audio.length, a.sampling_rate);
        buf.copyToChannel(a.audio, 0);
        await playAudioBuffer(buf, c, ctrl.signal, clean);
        return;
      }
    } catch (e) {
      throw new Error(
        `La voz Kokoro no pudo prepararse. La voz del sistema sigue disponible. ${e instanceof Error ? e.message : ""}`,
      );
    }
  }

  if (ctrl.signal.aborted) return;

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    let started = false;

    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      ctrl.signal.removeEventListener("abort", onAbort);
      activeTimeline = null;
      fakeLevel = 0;
      if (fakeTimer) {
        clearTimeout(fakeTimer);
        fakeTimer = null;
      }
      clearTimeout(noStartWatchdog);
      if (error) reject(error);
      else resolve();
    };

    const onAbort = () => {
      if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
      finish();
    };
    ctrl.signal.addEventListener("abort", onAbort, { once: true });

    const estWords = Math.max(1, clean.split(/\s+/).length);
    const estDuration = Math.max(0.8, (clean.length / 13.5 + estWords * 0.04) / tempo.rate);

    const startCadence = () => {
      if (settled) return;
      activeTimeline = buildVisemeTimeline(clean, estDuration, tempo.rate);
      const tick = () => {
        if (settled) return;
        const sampled = sampleVisemeTimeline(activeTimeline);
        fakeLevel = sampled ? Math.max(0.08, sampled.open * 0.82 + 0.14) : 0.35;
        fakeTimer = setTimeout(tick, 45);
      };
      tick();
    };

    // Never stall forever when device speech synthesis cannot start audio.
    const noStartWatchdog = setTimeout(() => {
      if (!started && !settled) {
        if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
        finish();
      }
    }, 3200);

    if (typeof speechSynthesis === "undefined") {
      finish(new Error("Este navegador no dispone de voz del sistema."));
      return;
    }

    const u = new SpeechSynthesisUtterance(clean);
    u.lang = lang;
    u.rate = tempo.rate;
    u.pitch = tempo.pitch;
    const v =
      (voice.engine === "system" && voice.id
        ? speechSynthesis.getVoices().find((x) => x.voiceURI === voice.id)
        : undefined) || systemVoices(lang, opts?.gender)[0];
    if (v) u.voice = v;

    u.onstart = () => {
      started = true;
      clearTimeout(noStartWatchdog);
      startCadence();
    };
    u.onboundary = (ev) => {
      if (!activeTimeline || typeof ev.charIndex !== "number") return;
      // Align viseme timeline phase dynamically to exact spoken word boundary
      let matchedSec: number | null = null;
      for (const w of activeTimeline.wordStartCues) {
        if (w.charIdx <= ev.charIndex) matchedSec = w.sec;
        else break;
      }
      if (matchedSec !== null) {
        activeTimeline.startedAtMs = performance.now() - matchedSec * 1000;
      }
    };
    u.onend = () => finish();
    u.onerror = (event) => {
      if (ctrl.signal.aborted || event.error === "canceled" || event.error === "interrupted")
        finish();
      else
        finish(
          new Error(
            "No se pudo reproducir la voz del dispositivo. Prueba otra voz en Personalizar.",
          ),
        );
    };

    try {
      speechSynthesis.speak(u);
    } catch {
      finish(
        new Error("El navegador bloqueó la voz. Pulsa Probar voz o vuelve a iniciar la llamada."),
      );
    }
  });
}

type SpeechRecCtor = new () => {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onresult: ((e: any) => void) | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onerror: ((e: any) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  abort: () => void;
};

const SR = (): SpeechRecCtor | null => {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecCtor;
    webkitSpeechRecognition?: SpeechRecCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
};

export const canListen = () => {
  if (typeof window === "undefined") return false;
  const hasMedia =
    typeof navigator !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia &&
    typeof MediaRecorder !== "undefined";
  return hasMedia || !!SR();
};

export type ListenResult = {
  text: string;
  fatalError?: string | undefined;
  error?: string | undefined;
};

export type ListenOptions = {
  onInterim?: (t: string) => void;
  onStatus?: (s: "listening" | "speech" | "transcribing") => void;
  onLevel?: (lv: number) => void;
  onLaugh?: () => void;
  localOnly?: boolean | undefined;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let activeRec: any = null;
let activeStream: MediaStream | null = null;
let activeFinishFn: (() => void) | null = null;
let activeAbortFn: (() => void) | null = null;

function pickRecorderMime(): string {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
    return "";
  }
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4;codecs=mp4a.40.2",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
}

async function getMicStream(): Promise<{ stream?: MediaStream; fatalError?: string }> {
  if (activeStream) {
    const live = activeStream.getAudioTracks().some((t) => t.readyState === "live");
    if (live) return { stream: activeStream };
    activeStream = null;
  }

  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    return { fatalError: "not-supported" };
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    activeStream = stream;
    return { stream };
  } catch (err) {
    const name = err instanceof Error ? err.name : "";
    if (
      name !== "NotAllowedError" &&
      name !== "PermissionDeniedError" &&
      name !== "SecurityError"
    ) {
      try {
        const plainStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        activeStream = plainStream;
        return { stream: plainStream };
      } catch (retryErr) {
        const retryName = retryErr instanceof Error ? retryErr.name : "";
        if (
          retryName === "NotAllowedError" ||
          retryName === "PermissionDeniedError" ||
          retryName === "SecurityError"
        ) {
          return { fatalError: "not-allowed" };
        }
        return { fatalError: "audio-capture" };
      }
    }
    return { fatalError: "not-allowed" };
  }
}

export async function warmMic(): Promise<string | undefined> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) return undefined;
  const { fatalError } = await getMicStream();
  return fatalError;
}

/**
 * Acquires microphone and camera independently in parallel inside the same user gesture
 * so a missing, busy, or denied camera never prevents microphone use (and vice versa).
 */
export async function acquireCallMedia(
  wantMic: boolean,
  wantCam: boolean,
): Promise<{
  micError?: string;
  camStream: MediaStream | null;
  camError?: "not-allowed" | "not-found" | "in-use" | "error";
}> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    return {
      ...(wantMic ? { micError: "not-supported" } : {}),
      camStream: null,
      ...(wantCam ? { camError: "not-found" as const } : {}),
    };
  }

  const micPromise = wantMic
    ? getMicStream()
    : Promise.resolve<{ stream?: MediaStream; fatalError?: string }>({});

  const camPromise = wantCam
    ? (async (): Promise<{
        camStream: MediaStream | null;
        camError?: "not-allowed" | "not-found" | "in-use" | "error";
      }> => {
        const attempts: MediaStreamConstraints[] = [
          {
            video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false,
          },
          { video: { facingMode: "user" }, audio: false },
          { video: true, audio: false },
        ];
        let lastErr: unknown;
        for (const c of attempts) {
          try {
            const stream = await navigator.mediaDevices.getUserMedia(c);
            return { camStream: stream };
          } catch (err) {
            lastErr = err;
            const name = err instanceof Error ? err.name : "";
            if (
              name === "NotAllowedError" ||
              name === "PermissionDeniedError" ||
              name === "SecurityError"
            ) {
              return { camStream: null, camError: "not-allowed" };
            }
            if (name === "NotFoundError" || name === "DevicesNotFoundError") {
              return { camStream: null, camError: "not-found" };
            }
          }
        }
        const name = lastErr instanceof Error ? lastErr.name : "";
        return {
          camStream: null,
          camError: name === "NotReadableError" || name === "TrackStartError" ? "in-use" : "error",
        };
      })()
    : Promise.resolve<{
        camStream: MediaStream | null;
        camError?: "not-allowed" | "not-found" | "in-use" | "error";
      }>({ camStream: null });

  const [micRes, camRes] = await Promise.all([micPromise, camPromise]);

  return {
    ...(micRes.fatalError ? { micError: micRes.fatalError } : {}),
    camStream: camRes.camStream,
    ...(camRes.camError ? { camError: camRes.camError } : {}),
  };
}

type SpeechRecInstance = InstanceType<SpeechRecCtor> & { stop?: () => void };

function listenBrowserSpeech(lang: string, onInterim?: (t: string) => void): Promise<ListenResult> {
  return new Promise((resolve) => {
    const R = SR();
    if (!R) return resolve({ text: "", fatalError: "not-supported" });
    try {
      const rec = new R() as SpeechRecInstance;
      activeRec = rec;
      rec.lang = lang;
      rec.interimResults = true;
      rec.continuous = false;
      let final = "";
      let fatalError: string | undefined;
      let done = false;
      let endSilenceTimer: ReturnType<typeof setTimeout> | null = null;
      const finish = () => {
        if (done) return;
        done = true;
        if (endSilenceTimer) clearTimeout(endSilenceTimer);
        activeRec = null;
        resolve({ text: final.trim(), fatalError });
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      rec.onresult = (e: any) => {
        let t = "";
        for (const r of e.results) t += r[0].transcript;
        final = t;
        onInterim?.(t);
        if (endSilenceTimer) clearTimeout(endSilenceTimer);
        if (t.trim()) {
          // If the speaker pauses for ~550ms after interim words, finalize immediately instead of stalling
          endSilenceTimer = setTimeout(() => {
            try {
              rec.stop?.();
            } catch {
              finish();
            }
            setTimeout(finish, 320);
          }, 550);
        }
      };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      rec.onerror = (e: any) => {
        const err = String(e?.error || "");
        if (
          err === "not-allowed" ||
          err === "service-not-allowed" ||
          err === "audio-capture" ||
          err === "network"
        ) {
          fatalError = err;
        }
        finish();
      };
      rec.onend = finish;
      rec.start();
    } catch {
      resolve({ text: "", fatalError: "start-failed" });
    }
  });
}

let cloudSttBlockedUntil = 0;

async function transcribeBlob(
  blob: Blob,
  lang: string,
  signal: AbortSignal,
): Promise<{ text: string; error?: string }> {
  const form = new FormData();
  const mime = (blob.type || "").toLowerCase();
  const ext = mime.includes("mp4") ? "mp4" : mime.includes("ogg") ? "ogg" : "webm";
  form.append("audio", blob, `speech.${ext}`);
  form.append("language", lang);

  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), 10000);
  const onParentAbort = () => timeout.abort();
  signal.addEventListener("abort", onParentAbort, { once: true });

  try {
    const res = await fetch("/api/transcribe", {
      method: "POST",
      body: form,
      signal: timeout.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return { text: "", error: err.error || `Error al transcribir (${res.status})` };
    }
    const data = (await res.json()) as { text?: string };
    return { text: (data.text ?? "").trim() };
  } catch (e) {
    if (signal.aborted) return { text: "" };
    return { text: "", error: e instanceof Error ? e.message : "Error de red al transcribir" };
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", onParentAbort);
  }
}

async function runBrowserSpeechFallback(lang: string, opts: ListenOptions): Promise<ListenResult> {
  opts.onStatus?.("listening");
  const srRes = await listenBrowserSpeech(lang, opts.onInterim);
  if (srRes.fatalError === "not-allowed" || srRes.fatalError === "service-not-allowed") {
    const micCheck = await getMicStream();
    if (micCheck.stream) {
      return { text: "", fatalError: "speech-api-blocked" };
    }
  }
  return srRes;
}

export async function listen(
  lang: string,
  onInterimOrOpts: ((t: string) => void) | ListenOptions,
): Promise<ListenResult> {
  const opts: ListenOptions =
    typeof onInterimOrOpts === "function" ? { onInterim: onInterimOrOpts } : onInterimOrOpts;

  const hasBrowserSR = !!SR();
  if ((opts.localOnly || Date.now() < cloudSttBlockedUntil) && hasBrowserSR) {
    return runBrowserSpeechFallback(lang, opts);
  }

  if (typeof navigator === "undefined" || typeof MediaRecorder === "undefined") {
    return listenBrowserSpeech(lang, opts.onInterim);
  }

  const { stream, fatalError } = await getMicStream();
  if (!stream) {
    return { text: "", fatalError: fatalError || "audio-capture" };
  }

  // Ensure AudioContext is awake before attaching analyser (otherwise getByteTimeDomainData reads flat 128 silence)
  const c = unlockAudio();
  if (c && c.state === "suspended") {
    await Promise.race([c.resume().catch(() => {}), new Promise<void>((r) => setTimeout(r, 400))]);
  }

  opts.onStatus?.("listening");

  return new Promise<ListenResult>((resolve) => {
    const mimeType = pickRecorderMime();
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    } catch {
      if (hasBrowserSR) {
        void runBrowserSpeechFallback(lang, opts).then(resolve);
      } else {
        resolve({ text: "", fatalError: "not-supported" });
      }
      return;
    }

    const chunks: BlobPart[] = [];
    const abortCtrl = new AbortController();
    let aborted = false;
    let manualFinish = false;
    let speechDetected = false;
    let speechStartedAt = 0;
    let speechFrames = 0;
    let resumeVoiceFrames = 0;
    let silenceMs = 0;
    let noiseFloor = 0.005;
    let peakSpeechEnergy = 0;
    const recentEnergies: number[] = [];

    let sourceNode: MediaStreamAudioSourceNode | null = null;
    let micAnalyser: AnalyserNode | null = null;
    if (c) {
      try {
        if (c.state === "suspended") void c.resume().catch(() => {});
        sourceNode = c.createMediaStreamSource(stream);
        micAnalyser = c.createAnalyser();
        micAnalyser.fftSize = 256;
        sourceNode.connect(micAnalyser);
      } catch {
        micAnalyser = null;
      }
    }

    const sampleBuf = micAnalyser ? new Uint8Array(micAnalyser.fftSize) : null;

    const cleanupAudioNodes = () => {
      clearInterval(vadTimer);
      clearTimeout(idleTimer);
      clearTimeout(maxTimer);
      currentMicLevel = 0;
      opts.onLevel?.(0);
      try {
        sourceNode?.disconnect();
      } catch {
        /* */
      }
      if (activeFinishFn === finishNow) activeFinishFn = null;
      if (activeAbortFn === abortNow) activeAbortFn = null;
    };

    const stopRecorder = () => {
      if (recorder.state === "recording") {
        try {
          recorder.stop();
        } catch {
          /* */
        }
      }
    };

    const finishNow = () => {
      manualFinish = true;
      stopRecorder();
    };

    const abortNow = () => {
      aborted = true;
      abortCtrl.abort();
      cleanupAudioNodes();
      stopRecorder();
    };

    activeFinishFn = finishNow;
    activeAbortFn = abortNow;

    let laughBursts = 0;
    let inLaughPulse = false;
    let lastLaughPulseAt = 0;
    let laughEmitted = false;
    let prevVadEnergy = 0;

    const vadTimer = setInterval(() => {
      if (aborted || recorder.state !== "recording") return;
      let rms = 0;
      let deltaRms = 0;
      if (micAnalyser && sampleBuf) {
        micAnalyser.getByteTimeDomainData(sampleBuf);
        let sum = 0;
        let diffSum = 0;
        let prev = (sampleBuf[0]! - 128) / 128;
        for (let i = 0; i < sampleBuf.length; i++) {
          const n = (sampleBuf[i]! - 128) / 128;
          sum += n * n;
          const d = n - prev;
          diffSum += d * d;
          prev = n;
        }
        rms = Math.sqrt(sum / sampleBuf.length);
        deltaRms = Math.sqrt(diffSum / sampleBuf.length);
      }
      const speechEnergy = rms * 0.55 + deltaRms * 0.9;
      const level = Math.min(1, speechEnergy * 7.5);
      currentMicLevel = level;
      opts.onLevel?.(level);

      recentEnergies.push(speechEnergy);
      if (recentEnergies.length > 14) recentEnergies.shift();

      // Detect distinct rhythmic vocal bursts characteristic of human chuckles/laughter (ha-ha-ha-ha)
      const pulseHi = Math.max(noiseFloor * 3.8, 0.044);
      const pulseLo = pulseHi * 0.52;
      const nowMs = Date.now();
      if (
        !inLaughPulse &&
        speechEnergy > pulseHi &&
        speechEnergy > prevVadEnergy * 1.24 &&
        nowMs - lastLaughPulseAt >= 120
      ) {
        inLaughPulse = true;
        lastLaughPulseAt = nowMs;
        laughBursts++;
        if (!laughEmitted && laughBursts >= 4 && speechDetected && nowMs - speechStartedAt < 2200) {
          laughEmitted = true;
          opts.onLaugh?.();
        }
      } else if (inLaughPulse && speechEnergy < pulseLo) {
        inLaughPulse = false;
      }
      prevVadEnergy = speechEnergy;

      if (!speechDetected) {
        if (speechEnergy < noiseFloor) {
          noiseFloor = noiseFloor * 0.6 + speechEnergy * 0.4;
        } else {
          noiseFloor = noiseFloor * 0.92 + Math.min(speechEnergy, 0.028) * 0.08;
        }
        noiseFloor = Math.min(0.028, Math.max(0.0045, noiseFloor));
      }

      // Sensitive adaptive VAD threshold that tolerates natural speech pauses without cutting off words
      const startThreshold = Math.max(0.009, noiseFloor * 1.75);
      const keepThreshold = speechDetected
        ? Math.max(0.0085, noiseFloor * 1.6, Math.min(0.032, peakSpeechEnergy * 0.24))
        : startThreshold;

      if (speechEnergy > keepThreshold) {
        if (!speechDetected) {
          speechFrames++;
          if (speechFrames >= 2) {
            speechDetected = true;
            speechStartedAt = Date.now();
            peakSpeechEnergy = speechEnergy;
            silenceMs = 0;
            clearTimeout(idleTimer);
            opts.onStatus?.("speech");
          }
        } else {
          peakSpeechEnergy = Math.max(peakSpeechEnergy * 0.985, speechEnergy);
          resumeVoiceFrames++;
          if (resumeVoiceFrames >= 1) {
            silenceMs = 0;
          }
        }
      } else {
        speechFrames = 0;
        resumeVoiceFrames = 0;
        if (speechDetected) {
          silenceMs += 50;
          const elapsed = Date.now() - speechStartedAt;
          const maxRecent = recentEnergies.length > 0 ? Math.max(...recentEnergies) : speechEnergy;
          const quietWindow = recentEnergies.length >= 14 && maxRecent < keepThreshold;
          if ((silenceMs >= 750 || quietWindow) && elapsed >= 340) {
            stopRecorder();
          }
        }
      }
    }, 50);

    // If user hasn't started speaking after 9s, recycle cleanly without burning a transcription call
    const idleTimer = setTimeout(() => {
      if (!speechDetected && !manualFinish && !aborted) {
        stopRecorder();
      }
    }, 9000);

    // Max single turn length (16s)
    const maxTimer = setTimeout(() => {
      if (!aborted) {
        manualFinish = speechDetected || manualFinish;
        stopRecorder();
      }
    }, 16000);

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = async () => {
      cleanupAudioNodes();
      if (aborted) {
        resolve({ text: "" });
        return;
      }

      const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" });
      if ((!speechDetected && !manualFinish) || blob.size < 350) {
        resolve({ text: "" });
        return;
      }

      opts.onStatus?.("transcribing");
      const { text, error } = await transcribeBlob(blob, lang, abortCtrl.signal);
      if (aborted) {
        resolve({ text: "" });
        return;
      }
      if (error && hasBrowserSR) {
        // Cloud STT failed (e.g., invalid/missing key or network issue): transparently activate browser SpeechRecognition
        cloudSttBlockedUntil = Date.now() + 90_000;
        const srRes = await runBrowserSpeechFallback(lang, opts);
        if (!srRes.fatalError) {
          resolve(srRes);
          return;
        }
      }
      if (text) opts.onInterim?.(text);
      resolve({ text, error });
    };

    recorder.onerror = () => {
      cleanupAudioNodes();
      if (hasBrowserSR) {
        void runBrowserSpeechFallback(lang, opts).then(resolve);
      } else {
        resolve({ text: "", fatalError: "audio-capture" });
      }
    };

    try {
      recorder.start(200);
    } catch {
      cleanupAudioNodes();
      if (hasBrowserSR) {
        void runBrowserSpeechFallback(lang, opts).then(resolve);
      } else {
        resolve({ text: "", fatalError: "start-failed" });
      }
    }
  });
}

export function finishListening() {
  activeFinishFn?.();
}

/**
 * Monitors active echo-cancelled mic input while the avatar is speaking.
 * Calls `onUserSpeech` as soon as sustained user speech is detected so the avatar stops immediately.
 */
export function startBargeInMonitor(onUserSpeech: () => void): () => void {
  if (typeof window === "undefined" || !activeStream) return () => {};
  const live = activeStream.getAudioTracks().some((t) => t.readyState === "live");
  if (!live) return () => {};
  const c = unlockAudio();
  if (!c) return () => {};

  let sourceNode: MediaStreamAudioSourceNode | null = null;
  let micAnalyser: AnalyserNode | null = null;
  let stopped = false;

  try {
    sourceNode = c.createMediaStreamSource(activeStream);
    micAnalyser = c.createAnalyser();
    micAnalyser.fftSize = 256;
    sourceNode.connect(micAnalyser);
  } catch {
    return () => {};
  }

  const buf = new Uint8Array(micAnalyser.fftSize);
  let speechFrames = 0;
  let ticks = 0;
  let bargeFloor = 0.01;

  const timer = setInterval(() => {
    if (stopped || !micAnalyser) return;
    ticks++;

    micAnalyser.getByteTimeDomainData(buf);
    let sum = 0;
    for (const x of buf) {
      const n = (x - 128) / 128;
      sum += n * n;
    }
    const rms = Math.sqrt(sum / buf.length);

    // First ~200ms calibrates echo-cancelled speaker bleed floor so user speech cuts off avatar immediately
    if (ticks <= 5) {
      bargeFloor = bargeFloor * 0.65 + Math.min(rms, 0.026) * 0.35;
      return;
    }

    const bargeThreshold = Math.max(0.032, bargeFloor * 2.5);
    if (rms > bargeThreshold) {
      speechFrames++;
      if (speechFrames >= 2) {
        stopped = true;
        clearInterval(timer);
        try {
          sourceNode?.disconnect();
        } catch {
          /* */
        }
        onUserSpeech();
      }
    } else {
      bargeFloor = bargeFloor * 0.92 + Math.min(rms, 0.022) * 0.08;
      speechFrames = Math.max(0, speechFrames - 1);
    }
  }, 40);

  return () => {
    stopped = true;
    clearInterval(timer);
    try {
      sourceNode?.disconnect();
    } catch {
      /* */
    }
  };
}

export function stopListening(releaseMic = true) {
  activeAbortFn?.();
  activeAbortFn = null;
  activeFinishFn = null;
  currentMicLevel = 0;
  try {
    activeRec?.abort();
  } catch {
    /* */
  }
  activeRec = null;
  if (releaseMic && activeStream) {
    for (const track of activeStream.getTracks()) {
      try {
        track.stop();
      } catch {
        /* */
      }
    }
    activeStream = null;
  }
}
