import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import {
  ArrowLeft,
  Check,
  Keyboard,
  Mic,
  MicOff,
  Phone,
  PhoneOff,
  SendHorizontal,
  Video,
  VideoOff,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { addMessage, getMessages, getProfile, setProfile, useProfile } from "@/lib/store";
import {
  extractEmotion,
  getModelOption,
  loadModel,
  replyWithEmotion,
  resetModel,
  useAIState,
  type AvatarEmotion,
} from "@/lib/ai";
import {
  canListen,
  finishListening,
  listen,
  speak,
  startBargeInMonitor,
  stopListening,
  stopSpeaking,
  unlockAudio,
  warmMic,
} from "@/lib/voice";
import {
  analyzeUserVideoFrame,
  canUseCamera,
  ensureUserFaceTracker,
  startUserCamera,
  stopUserCamera,
  type UserFaceObservation,
} from "@/lib/userCamera";
import { Button } from "@/components/ui/button";
import { AnimatedAvatar } from "@/components/AnimatedAvatar";
import { AvatarImg } from "@/components/AvatarImg";

export const Route = createFileRoute("/call")({
  head: () => ({
    meta: [
      { title: "Videollamada — Yume" },
      { name: "description", content: "Videollamada realista en tiempo real en Yume." },
      { property: "og:title", content: "Videollamada — Yume" },
      {
        property: "og:description",
        content: "Videollamada realista en tiempo real en Yume.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Call,
});

type Phase = "ringing" | "listening" | "speaking" | "paused";
type ListenStage = "listening" | "speech" | "transcribing";

function micErrorText(code: string) {
  if (code === "not-allowed") {
    return "Permiso de micrófono bloqueado. Actívalo o usa el teclado.";
  }
  if (code === "audio-capture") {
    return "No se detectó micrófono. Puedes escribir para hablar.";
  }
  return "Micrófono no disponible: escribe para hablar.";
}

function Call() {
  const p = useProfile();
  const ai = useAIState();
  const nav = useNavigate();
  const [started, setStarted] = useState(false);
  const [phase, setPhase] = useState<Phase>("ringing");
  const [listenStage, setListenStage] = useState<ListenStage>("listening");
  const [micVol, setMicVol] = useState(0);
  const [muted, setMuted] = useState(false);
  const [textMode, setTextMode] = useState(false);
  const [micNote, setMicNote] = useState("");
  const [secs, setSecs] = useState(0);
  const [typed, setTyped] = useState("");
  const [sending, setSending] = useState(false);
  const [listenOk, setListenOk] = useState(true);
  const [camOk, setCamOk] = useState(true);
  const [camActive, setCamActive] = useState(false);
  const [camNote, setCamNote] = useState("");
  const [userFace, setUserFace] = useState<UserFaceObservation | undefined>(undefined);
  const [aiEmotion, setAiEmotion] = useState<AvatarEmotion>("neutral");

  const alive = useRef(true);
  const callTokenRef = useRef(0);
  const loopActiveRef = useRef(false);
  const mutedRef = useRef(false);
  const textModeRef = useRef(false);
  const userVideoRef = useRef<HTMLVideoElement>(null);
  const camStreamRef = useRef<MediaStream | null>(null);
  const userFaceRef = useRef<UserFaceObservation | undefined>(undefined);
  const isRinging = phase === "ringing";
  const mm = `${Math.floor(secs / 60)}:${(secs % 60).toString().padStart(2, "0")}`;

  const stopCamera = () => {
    stopUserCamera(camStreamRef.current);
    camStreamRef.current = null;
    if (userVideoRef.current) {
      userVideoRef.current.srcObject = null;
    }
    userFaceRef.current = undefined;
    setUserFace(undefined);
    setCamActive(false);
  };

  useEffect(() => {
    alive.current = true;
    setListenOk(canListen());
    setCamOk(canUseCamera());
    return () => {
      alive.current = false;
      callTokenRef.current += 1;
      loopActiveRef.current = false;
      stopListening(true);
      stopSpeaking();
      stopCamera();
    };
  }, []);

  useEffect(() => {
    if (!started || isRinging) return;
    const i = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(i);
  }, [started, isRinging]);

  // Attach webcam stream to <video> and run throttled client-side MediaPipe facial expression/gaze detection
  useEffect(() => {
    if (!camActive) return;
    const stream = camStreamRef.current;
    const video = userVideoRef.current;
    if (!stream || !video) return;

    video.srcObject = stream;
    void video.play().catch(() => {});

    let cancel = false;
    let raf = 0;
    let lastAnalyze = 0;
    let lastUiSync = 0;

    void ensureUserFaceTracker().then((ready) => {
      if (cancel || !ready) return;
      const tick = (now: number) => {
        if (cancel) return;
        raf = requestAnimationFrame(tick);
        if (now - lastAnalyze < 90) return;
        lastAnalyze = now;
        const obs = analyzeUserVideoFrame(video, now);
        if (!obs) return;
        userFaceRef.current = obs;
        if (now - lastUiSync >= 180) {
          lastUiSync = now;
          setUserFace(obs);
        }
      };
      raf = requestAnimationFrame(tick);
    });

    return () => {
      cancel = true;
      cancelAnimationFrame(raf);
    };
  }, [camActive]);

  const requestCamera = async () => {
    if (!canUseCamera()) {
      setCamOk(false);
      return;
    }
    if (camStreamRef.current) {
      setCamActive(true);
      return;
    }
    const res = await startUserCamera();
    if (!alive.current) {
      stopUserCamera(res.stream);
      return;
    }
    if (res.stream) {
      camStreamRef.current = res.stream;
      setCamNote("");
      setCamActive(true);
    } else if (res.error === "not-allowed") {
      setCamNote("Cámara sin permiso · solo audio");
      setCamActive(false);
    } else if (res.error === "not-found") {
      setCamOk(false);
      setCamActive(false);
    }
  };

  const toggleCamera = async () => {
    if (camActive) {
      stopCamera();
      setCamNote("");
    } else {
      await requestCamera();
    }
  };

  const isCallValid = (token: number) => alive.current && callTokenRef.current === token;

  const speakWithBargeIn = async (
    text: string,
    token: number,
    emotion: AvatarEmotion,
  ): Promise<boolean> => {
    let bargeTriggered = false;
    const stopBargeIn =
      !mutedRef.current && !textModeRef.current && canListen()
        ? startBargeInMonitor(() => {
            bargeTriggered = true;
            stopSpeaking();
          })
        : () => {};
    try {
      const cur = getProfile();
      await speak(text, cur.voice, cur.language, emotion, { localOnly: cur.localOnly });
    } catch {
      // Continuar si el audio está restringido
    } finally {
      stopBargeIn();
    }
    return isCallValid(token) && bargeTriggered;
  };

  const respond = async (heard: string, token: number) => {
    addMessage({ role: "user", content: heard });
    let r = "";
    let em: AvatarEmotion = "neutral";
    try {
      const visualContext = userFaceRef.current?.detected
        ? userFaceRef.current.contextPrompt
        : undefined;
      const res = await replyWithEmotion(
        getProfile(),
        getMessages(),
        (partial) => {
          if (!isCallValid(token)) return;
          const { emotion } = extractEmotion(partial);
          if (emotion !== "neutral") {
            setAiEmotion(emotion);
          }
        },
        { callMode: true, visualContext },
      );
      r = res.text;
      em = res.emotion;
    } catch {
      r = "Perdona, intenta decírmelo de nuevo.";
    }
    if (!isCallValid(token)) return;
    addMessage({ role: "assistant", content: r });
    setAiEmotion(em);
    setPhase("speaking");
    await speakWithBargeIn(r, token, em);
  };

  const loop = async (token: number) => {
    if (loopActiveRef.current) return;
    loopActiveRef.current = true;
    try {
      while (isCallValid(token)) {
        if (mutedRef.current || !canListen() || textModeRef.current) {
          setPhase("paused");
          return;
        }
        setPhase("listening");
        setListenStage("listening");
        setMicVol(0);
        const cur = getProfile();
        const res = await listen(cur.language, {
          localOnly: cur.localOnly,
          onStatus: (s) => {
            if (isCallValid(token)) setListenStage(s);
          },
          onLevel: (lv) => {
            if (isCallValid(token)) setMicVol(lv);
          },
        });
        if (!isCallValid(token)) return;

        if (res.fatalError) {
          textModeRef.current = true;
          setTextMode(true);
          setMicNote(micErrorText(res.fatalError));
          setPhase("paused");
          return;
        }
        if (res.error) {
          setMicNote(res.error);
          await new Promise((r) => setTimeout(r, 250));
          continue;
        }
        if (!res.text) {
          await new Promise((r) => setTimeout(r, 80));
          continue;
        }
        setMicNote("");
        await respond(res.text, token);
      }
    } finally {
      loopActiveRef.current = false;
    }
  };

  const start = async (preferTextMode = false) => {
    if (started) return;
    const token = ++callTokenRef.current;
    alive.current = true;
    setStarted(true);
    setSecs(0);
    setMicNote("");
    if (preferTextMode) {
      textModeRef.current = true;
      setTextMode(true);
    }
    unlockAudio();
    loadModel(getProfile().modelId).catch(() => {});

    // Request camera permission in parallel so avatar can mirror & understand user expressions
    void requestCamera();

    if (!preferTextMode && canListen()) {
      const micErr = await warmMic();
      if (!isCallValid(token)) return;
      if (micErr) {
        textModeRef.current = true;
        setTextMode(true);
        setMicNote(micErrorText(micErr));
      }
    }

    setPhase("ringing");
    await new Promise((r) => setTimeout(r, 80));
    if (!isCallValid(token)) return;

    const cur = getProfile();
    const es = cur.language === "es-ES";
    const isInterview =
      cur.characterId === "sofia" ||
      cur.characterId === "daniel" ||
      /entrevista/i.test(cur.personality + " " + cur.backstory);
    const hi = es
      ? isInterview
        ? `Hola, soy ${cur.name}. Gracias por conectarte, cuando quieras empezamos.`
        : `Hola, ¿cómo estás?`
      : isInterview
        ? `Hi, I'm ${cur.name}. Thanks for joining, let's get started whenever you're ready.`
        : "Hey! Good to see you.";
    const greetingEmotion: AvatarEmotion = isInterview ? "neutral" : "happy";
    setAiEmotion(greetingEmotion);
    setPhase("speaking");
    await speakWithBargeIn(hi, token, greetingEmotion);
    if (!isCallValid(token)) return;

    if (preferTextMode || !canListen()) {
      textModeRef.current = true;
      setTextMode(true);
      if (!canListen()) {
        setMicNote("Tu navegador no permite micrófono: escribe para hablar.");
      }
      setPhase("paused");
      return;
    }

    loop(token);
  };

  const toggleMute = () => {
    const m = !muted;
    setMuted(m);
    mutedRef.current = m;
    if (m) {
      stopListening(true);
      if (phase === "listening") setPhase("paused");
    } else if (!textModeRef.current && (phase === "paused" || !loopActiveRef.current)) {
      setMicNote("");
      loop(callTokenRef.current);
    }
  };

  const toggleKeyboard = () => {
    const next = !textMode;
    setTextMode(next);
    textModeRef.current = next;
    if (next) {
      stopListening(true);
      if (phase !== "speaking") {
        setPhase("paused");
      }
    } else if (!mutedRef.current && canListen()) {
      setMicNote("");
      if (phase === "paused") {
        loop(callTokenRef.current);
      }
    }
  };

  const hang = () => {
    alive.current = false;
    callTokenRef.current += 1;
    loopActiveRef.current = false;
    stopListening(true);
    stopSpeaking();
    stopCamera();
    void nav({ to: "/" });
  };

  const submitTyped = async (e: React.FormEvent) => {
    e.preventDefault();
    const t = typed.trim();
    if (!t || sending) return;
    const token = callTokenRef.current;
    setTyped("");
    setSending(true);
    stopListening(false);
    stopSpeaking();
    try {
      await respond(t, token);
      if (!isCallValid(token)) return;
      if (!mutedRef.current && !textModeRef.current && canListen()) {
        loop(token);
      } else {
        setPhase("paused");
      }
    } finally {
      setSending(false);
    }
  };

  const listenLabel =
    listenStage === "transcribing"
      ? "Procesando…"
      : listenStage === "speech"
        ? "Te escucha…"
        : "Habla ahora…";

  const status = !started
    ? "En espera"
    : {
        ringing: "Conectando…",
        listening: listenLabel,
        speaking: "Hablando",
        paused: muted ? "Micrófono apagado" : "Modo texto",
      }[phase];

  return (
    <main className="mx-auto grid h-dvh w-full max-w-5xl grid-rows-[56px_minmax(0,1fr)_136px] overflow-hidden bg-background">
      <header className="flex h-14 items-center justify-between gap-3 border-b border-border px-3 sm:px-5">
        <div className="flex min-w-0 items-center gap-2.5">
          <Link
            to="/"
            aria-label="Volver"
            onClick={() => {
              stopListening(true);
              stopSpeaking();
              stopCamera();
            }}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-foreground transition-colors hover:bg-secondary"
          >
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <AvatarImg src={p.avatar} name={p.name} className="h-8 w-8 shrink-0 text-xs" />
          <div className="min-w-0 text-left">
            <h1 className="truncate text-sm font-semibold leading-tight text-foreground">
              {p.name}
            </h1>
            <p className="truncate text-xs text-muted-foreground tabular-nums">
              {started && !isRinging ? `${status} · ${mm}` : status}
            </p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {ai.status === "loading" && (
            <span className="truncate text-xs text-muted-foreground tabular-nums">
              Cargando {getModelOption(ai.modelId).name} {Math.round(ai.progress * 100)}%
            </span>
          )}
          {ai.status === "error" && (
            <span className="truncate text-xs text-destructive">Error de modelo</span>
          )}
          {(ai.status === "loading" || ai.status === "error") && !p.localOnly && (
            <Button
              variant="secondary"
              size="sm"
              type="button"
              onClick={() => {
                setProfile({ modelId: "gemini-flash-lite", localOnly: false });
                resetModel();
                loadModel("gemini-flash-lite").catch(() => {});
              }}
              className="h-7 whitespace-nowrap shrink-0 rounded-md px-2.5 text-xs"
            >
              Usar rápido
            </Button>
          )}
          {ai.status !== "loading" && ai.status !== "error" && (
            <span className="hidden text-xs text-muted-foreground sm:inline">
              {camActive && userFace?.detected
                ? userFace.labelEs
                : p.localOnly
                  ? "Modo local · cero datos"
                  : "Estudio en vivo"}
            </span>
          )}
        </div>
      </header>

      <div className="min-h-0 overflow-hidden px-3 py-2 sm:px-6 sm:py-3">
        <div className="relative h-full w-full overflow-hidden rounded-2xl border border-border bg-[#0a0a0a]">
          <AnimatedAvatar
            src={p.avatar}
            name={p.name}
            active={phase === "speaking"}
            emotion={aiEmotion}
            listening={phase === "listening"}
            engine={p.avatarEngine ?? "mediapipe-mesh"}
            liveAvatarUrl={p.liveAvatarUrl}
            localOnly={p.localOnly}
            fit="contain"
            userFace={userFace}
            className="h-full w-full"
          />

          {camActive && (
            <div className="pointer-events-none absolute bottom-3 right-3 h-28 w-20 overflow-hidden rounded-xl border border-border bg-black/90 shadow-lg sm:h-36 sm:w-28">
              <video
                ref={userVideoRef}
                playsInline
                muted
                autoPlay
                className="h-full w-full -scale-x-100 object-cover"
              />
            </div>
          )}
        </div>
      </div>

      <section className="flex h-[136px] flex-col justify-center gap-2 border-t border-border bg-background px-4 text-center">
        <div className="flex h-11 items-center justify-center">
          {phase === "listening" ? (
            <div className="flex items-center justify-center gap-3">
              <div className="flex h-5 items-center gap-1" aria-hidden="true">
                {[0.35, 0.7, 1, 0.7, 0.35].map((w, idx) => (
                  <span
                    key={idx}
                    className="w-1.5 rounded-full bg-foreground/85 transition-all duration-75"
                    style={{
                      height: `${Math.round(5 + Math.min(1, micVol * 1.4) * 13 * w)}px`,
                    }}
                  />
                ))}
              </div>
              {listenStage === "speech" && (
                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  onClick={finishListening}
                  className="h-8 whitespace-nowrap shrink-0 rounded-full px-3 text-xs font-medium"
                >
                  <Check className="h-3.5 w-3.5" />
                  <span>Enviar ahora</span>
                </Button>
              )}
              {(micNote || camNote) && (
                <span className="max-w-xs truncate text-xs text-muted-foreground">
                  {micNote || camNote}
                </span>
              )}
            </div>
          ) : started && phase === "paused" ? (
            <form className="flex w-full max-w-xl items-center gap-2" onSubmit={submitTyped}>
              <input
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                placeholder={micNote || "Escribe lo que quieres decir…"}
                disabled={sending}
                className="h-10 min-w-0 flex-1 rounded-full border border-border bg-card px-4 text-sm text-foreground outline-none ring-ring focus:ring-2 disabled:opacity-60"
              />
              <Button
                variant="default"
                type="submit"
                disabled={!typed.trim() || sending}
                className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full p-0"
                aria-label="Enviar"
              >
                <SendHorizontal className="h-4 w-4" />
              </Button>
            </form>
          ) : (
            <p className="max-w-lg truncate text-xs text-muted-foreground">
              {micNote ||
                camNote ||
                (!started
                  ? "Pulsa llamar para iniciar con voz y cámara, o usa el teclado"
                  : phase === "speaking"
                    ? "Habla en cualquier momento para intervenir"
                    : "Conectando estudio en vivo…")}
            </p>
          )}
        </div>

        <div className="flex h-14 items-center justify-center">
          {!started ? (
            <div className="flex items-center justify-center gap-7">
              <Button
                variant="ghost"
                type="button"
                onClick={hang}
                className="flex h-12 w-12 items-center justify-center rounded-full bg-destructive text-destructive-foreground hover:bg-destructive/90"
                aria-label="Cancelar"
              >
                <PhoneOff className="h-5 w-5" />
              </Button>
              <Button
                variant="default"
                type="button"
                onClick={() => void start(false)}
                className="shadow-glow flex h-14 w-14 items-center justify-center rounded-full"
                aria-label="Llamar"
              >
                <Phone className="h-6 w-6" />
              </Button>
              <Button
                variant="secondary"
                type="button"
                onClick={() => void start(true)}
                className="flex h-12 w-12 items-center justify-center rounded-full"
                aria-label="Teclado"
              >
                <Keyboard className="h-5 w-5" />
              </Button>
            </div>
          ) : (
            <div className="mx-auto flex w-fit items-center gap-3.5 rounded-full border border-border bg-card px-5 py-1.5">
              <Button
                variant="ghost"
                type="button"
                onClick={toggleMute}
                disabled={!listenOk}
                className={`flex h-10 w-10 items-center justify-center rounded-full p-0 ${
                  muted
                    ? "bg-foreground text-background hover:bg-foreground/90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                } disabled:opacity-40`}
                aria-label="Silenciar"
              >
                {muted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                type="button"
                onClick={() => void toggleCamera()}
                disabled={!camOk}
                className={`flex h-10 w-10 items-center justify-center rounded-full p-0 ${
                  camActive
                    ? "bg-foreground text-background hover:bg-foreground/90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                } disabled:opacity-40`}
                aria-label={camActive ? "Apagar cámara" : "Encender cámara"}
              >
                {camActive ? <Video className="h-4 w-4" /> : <VideoOff className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                type="button"
                onClick={toggleKeyboard}
                className={`flex h-10 w-10 items-center justify-center rounded-full p-0 ${
                  textMode
                    ? "bg-primary text-primary-foreground hover:bg-primary/90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                }`}
                aria-label="Teclado"
              >
                <Keyboard className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                type="button"
                onClick={hang}
                className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive p-0 text-destructive-foreground hover:bg-destructive/90"
                aria-label="Colgar"
              >
                <PhoneOff className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
