// Voices: Gemini Cloud TTS (Spanish & English, WAV + real AnalyserNode lip-sync), Kokoro-82M, and device voices.
// STT: MediaRecorder + Web Audio VAD -> /api/transcribe (Gemini 3.5 Transcribe), with browser SpeechRecognition fallback.
import type { VoiceChoice } from "./store";

export const GEMINI_VOICES = [
  { id: "Kore", gender: "female" as const, label: "Kore · mujer, cálida y natural" },
  { id: "Zephyr", gender: "female" as const, label: "Zephyr · mujer, alegre y brillante" },
  { id: "Puck", gender: "male" as const, label: "Puck · hombre, joven y cercano" },
  { id: "Charon", gender: "male" as const, label: "Charon · hombre, sereno y grave" },
  { id: "Fenrir", gender: "male" as const, label: "Fenrir · hombre, enérgico y firme" },
];

export const KOKORO_VOICES = [
  { id: "af_heart", label: "Heart · mujer, cálida (EN)" },
  { id: "af_bella", label: "Bella · mujer, expresiva (EN)" },
  { id: "af_nicole", label: "Nicole · mujer, susurrada (EN)" },
  { id: "am_michael", label: "Michael · hombre, grave (EN)" },
  { id: "am_fenrir", label: "Fenrir · hombre, intenso (EN)" },
  { id: "bf_emma", label: "Emma · mujer británica (EN)" },
  { id: "bm_george", label: "George · hombre británico (EN)" },
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let kokoro: any = null;
let kLoading: Promise<void> | null = null;
export function loadKokoro(onProgress?: (p: number) => void) {
  if (kLoading) return kLoading;
  kLoading = (async () => {
    const { KokoroTTS } = await import("kokoro-js");
    kokoro = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
      dtype: "q8",
      device: "wasm",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      progress_callback: (p: any) => {
        if (p.status === "progress" && p.total) onProgress?.(p.loaded / p.total);
      },
    });
  })().catch((e) => {
    kLoading = null;
    throw e;
  });
  return kLoading;
}

export function systemVoices(lang: string) {
  if (typeof speechSynthesis === "undefined") return [];
  const all = speechSynthesis.getVoices();
  const pre = lang.slice(0, 2);
  return all.filter((v) => v.lang.startsWith(pre)).sort((a, b) => score(b) - score(a));
}
const score = (v: SpeechSynthesisVoice) =>
  (/natural|neural|premium|enhanced|google|siri/i.test(v.name) ? 2 : 0) + (v.localService ? 0 : 1);

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
};

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
    const mid = avgRange(420, 1200);
    const high = avgRange(1200, 3400);

    const open = Math.min(1, level * 0.55 + mid * 0.85 + low * 0.35);
    const wide = Math.min(1, Math.max(0, (high * 1.45 - low * 0.25) * (0.4 + level)));
    const round = Math.min(1, Math.max(0, (low * 1.15 - high * 0.65) * (0.4 + level)));
    return { level, open, wide, round };
  }

  const t = typeof performance !== "undefined" ? performance.now() * 0.001 : 0;
  return {
    level: fakeLevel,
    open: fakeLevel,
    wide: fakeLevel > 0.05 ? (0.5 + 0.5 * Math.sin(t * 7.3)) * fakeLevel * 0.7 : 0,
    round: fakeLevel > 0.05 ? (0.5 + 0.5 * Math.cos(t * 5.7)) * fakeLevel * 0.55 : 0,
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
): Promise<boolean> {
  if (signal.aborted) return false;
  if (c.state === "suspended") {
    await Promise.race([c.resume().catch(() => {}), new Promise<void>((r) => setTimeout(r, 450))]);
  }
  if (c.state !== "running") return false;
  const src = c.createBufferSource();
  src.buffer = buf;
  src.connect(analyser!);
  current = src;
  const maxMs = Math.min(25000, Math.max(1500, Math.round((buf.duration || 3) * 1000) + 1200));
  await new Promise<void>((resolve) => {
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      clearTimeout(watchdog);
      signal.removeEventListener("abort", onAbort);
      if (current === src) current = null;
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
): Promise<boolean> {
  const cacheKey = `${voiceId}|${lang}|${emotion ?? ""}|${clean}`;
  let arr = ttsCache.get(cacheKey);
  if (!arr) {
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), 5200);
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
          ...(emotion ? { emotion } : {}),
        }),
      });
      if (!res.ok) return false;
      arr = await res.arrayBuffer();
      if (arr.byteLength >= 64 && ttsCache.size < 24) {
        ttsCache.set(cacheKey, arr);
      }
    } catch {
      return signal.aborted;
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", onParentAbort);
    }
  }

  if (signal.aborted || !arr || arr.byteLength < 64) return signal.aborted;
  const c = unlockAudio();
  if (!c) return false;

  try {
    const decoded = await c.decodeAudioData(arr.slice(0));
    return await playAudioBuffer(decoded, c, signal);
  } catch {
    const fallbackBuf = decodePcm16Fallback(c, arr);
    if (!fallbackBuf) return false;
    return await playAudioBuffer(fallbackBuf, c, signal);
  }
}

export async function speak(
  text: string,
  voice: VoiceChoice,
  lang: string,
  emotion?: string,
  opts?: { localOnly?: boolean | undefined },
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

  if (voice.engine === "gemini" && !opts?.localOnly) {
    const ok = await speakViaGemini(clean, voice.id || "Kore", lang, ctrl.signal, emotion);
    if (ok || ctrl.signal.aborted) return;
  }

  if (voice.engine === "kokoro") {
    try {
      await loadKokoro();
      if (ctrl.signal.aborted) return;
      const a = await kokoro.generate(clean, { voice: voice.id || "af_heart" });
      if (ctrl.signal.aborted) return;
      const c = unlockAudio();
      if (c) {
        const buf = c.createBuffer(1, a.audio.length, a.sampling_rate);
        buf.copyToChannel(a.audio, 0);
        await playAudioBuffer(buf, c, ctrl.signal);
        return;
      }
    } catch (e) {
      console.warn("Kokoro failed, falling back", e);
    }
  }

  if (ctrl.signal.aborted) return;

  await new Promise<void>((resolve) => {
    let settled = false;
    let started = false;
    const estMs = Math.min(12000, Math.max(1800, clean.length * 75));

    const finish = () => {
      if (settled) return;
      settled = true;
      ctrl.signal.removeEventListener("abort", onAbort);
      fakeLevel = 0;
      if (fakeTimer) {
        clearTimeout(fakeTimer);
        fakeTimer = null;
      }
      clearTimeout(noStartWatchdog);
      clearTimeout(maxWatchdog);
      resolve();
    };

    const onAbort = () => {
      if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
      finish();
    };
    ctrl.signal.addEventListener("abort", onAbort, { once: true });

    const startCadence = () => {
      if (settled) return;
      const tick = () => {
        if (settled) return;
        fakeLevel = 0.25 + Math.random() * 0.65;
        fakeTimer = setTimeout(tick, 90);
      };
      tick();
    };

    // If speechSynthesis is unavailable or silently blocked (iframe/headless/no voices), animate mouth briefly and resolve so calls never hang.
    const noStartWatchdog = setTimeout(() => {
      if (!started && !settled) {
        startCadence();
        setTimeout(finish, Math.min(4500, estMs));
      }
    }, 650);

    const maxWatchdog = setTimeout(finish, estMs + 2500);

    if (typeof speechSynthesis === "undefined") {
      startCadence();
      setTimeout(finish, Math.min(4000, estMs));
      return;
    }

    const u = new SpeechSynthesisUtterance(clean);
    u.lang = lang;
    const v =
      speechSynthesis.getVoices().find((x) => x.voiceURI === voice.id) || systemVoices(lang)[0];
    if (v) u.voice = v;

    u.onstart = () => {
      started = true;
      clearTimeout(noStartWatchdog);
      startCadence();
    };
    u.onend = finish;
    u.onerror = () => {
      if (!started) {
        startCadence();
        setTimeout(finish, Math.min(3500, estMs));
      } else {
        finish();
      }
    };

    try {
      speechSynthesis.speak(u);
    } catch {
      startCadence();
      setTimeout(finish, Math.min(3500, estMs));
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
      name === "NotAllowedError" ||
      name === "PermissionDeniedError" ||
      name === "SecurityError"
    ) {
      return { fatalError: "not-allowed" };
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return { fatalError: "audio-capture" };
    }
    return { fatalError: "audio-capture" };
  }
}

export async function warmMic(): Promise<string | undefined> {
  if (typeof navigator === "undefined" || typeof MediaRecorder === "undefined") return undefined;
  const { fatalError } = await getMicStream();
  return fatalError;
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

async function transcribeBlob(
  blob: Blob,
  lang: string,
  signal: AbortSignal,
): Promise<{ text: string; error?: string }> {
  const form = new FormData();
  form.append("audio", blob, "speech.webm");
  form.append("language", lang);

  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), 6500);
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

export async function listen(
  lang: string,
  onInterimOrOpts: ((t: string) => void) | ListenOptions,
): Promise<ListenResult> {
  const opts: ListenOptions =
    typeof onInterimOrOpts === "function" ? { onInterim: onInterimOrOpts } : onInterimOrOpts;

  if (opts.localOnly && SR()) {
    opts.onStatus?.("listening");
    return listenBrowserSpeech(lang, opts.onInterim);
  }

  if (typeof navigator === "undefined" || typeof MediaRecorder === "undefined") {
    return listenBrowserSpeech(lang, opts.onInterim);
  }

  const { stream, fatalError } = await getMicStream();
  if (!stream) {
    return { text: "", fatalError: fatalError || "audio-capture" };
  }

  opts.onStatus?.("listening");

  return new Promise<ListenResult>((resolve) => {
    const mimeType = pickRecorderMime();
    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    } catch {
      resolve({ text: "", fatalError: "not-supported" });
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
    let noiseFloor = 0.008;
    let peakSpeechEnergy = 0;
    const recentEnergies: number[] = [];

    let sourceNode: MediaStreamAudioSourceNode | null = null;
    let micAnalyser: AnalyserNode | null = null;
    const c = unlockAudio();
    if (c) {
      try {
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
      // Emphasize loud vocal formants & immediately reject ambient background noise
      const speechEnergy = rms * 0.55 + deltaRms * 0.9;
      const level = Math.min(1, speechEnergy * 7.5);
      currentMicLevel = level;
      opts.onLevel?.(level);

      recentEnergies.push(speechEnergy);
      if (recentEnergies.length > 5) recentEnergies.shift();

      if (!speechDetected) {
        if (speechEnergy < noiseFloor) {
          noiseFloor = noiseFloor * 0.6 + speechEnergy * 0.4;
        } else {
          noiseFloor = noiseFloor * 0.92 + Math.min(speechEnergy, 0.028) * 0.08;
        }
        noiseFloor = Math.min(0.028, Math.max(0.005, noiseFloor));
      }

      // Require clear loud voice to start & keep recording; respond immediately when voice stops
      const startThreshold = Math.max(0.025, noiseFloor * 2.5);
      const keepThreshold = speechDetected
        ? Math.max(0.026, noiseFloor * 2.6, Math.min(0.075, peakSpeechEnergy * 0.44))
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
          if (resumeVoiceFrames >= 2) {
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
          const quietWindow = recentEnergies.length >= 5 && maxRecent < keepThreshold;
          if ((silenceMs >= 220 || quietWindow) && elapsed >= 180) {
            stopRecorder();
          }
        }
      }
    }, 50);

    // If user hasn't started speaking after 8s, recycle cleanly without burning a transcription call
    const idleTimer = setTimeout(() => {
      if (!speechDetected && !manualFinish && !aborted) {
        stopRecorder();
      }
    }, 8000);

    // Max single turn length (14s)
    const maxTimer = setTimeout(() => {
      if (!aborted) {
        manualFinish = speechDetected || manualFinish;
        stopRecorder();
      }
    }, 14000);

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
      if ((!speechDetected && !manualFinish) || blob.size < 600) {
        resolve({ text: "" });
        return;
      }

      opts.onStatus?.("transcribing");
      const { text, error } = await transcribeBlob(blob, lang, abortCtrl.signal);
      if (aborted) {
        resolve({ text: "" });
        return;
      }
      if (text) opts.onInterim?.(text);
      resolve({ text, error });
    };

    recorder.onerror = () => {
      cleanupAudioNodes();
      resolve({ text: "", fatalError: "audio-capture" });
    };

    try {
      recorder.start(200);
    } catch {
      cleanupAudioNodes();
      resolve({ text: "", fatalError: "start-failed" });
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
