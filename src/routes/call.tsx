import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import {
  Aperture,
  ArrowLeft,
  AudioLines,
  Check,
  FlipHorizontal2,
  Keyboard,
  Maximize2,
  Mic,
  MicOff,
  Phone,
  PhoneOff,
  RefreshCw,
  ScanFace,
  SendHorizontal,
  Video,
  VideoOff,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  addMessage,
  findCharacterById,
  getMessages,
  getProfile,
  setProfile,
  useProfile,
} from "@/lib/store";
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
  acquireCallMedia,
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
      { title: "Videollamada — YUME" },
      { name: "description", content: "Videollamada realista en tiempo real en YUME." },
      { property: "og:title", content: "Videollamada — YUME" },
      {
        property: "og:description",
        content: "Videollamada realista en tiempo real en YUME.",
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
  if (code === "speech-api-blocked") {
    return "El dictado local del navegador está bloqueado en esta ventana.";
  }
  if (code === "not-allowed") {
    return "Permiso de micrófono bloqueado en el navegador. Actívalo o escribe abajo.";
  }
  if (code === "audio-capture") {
    return "No se detectó micrófono disponible. Puedes escribir para hablar.";
  }
  return "Micrófono no disponible por ahora. Puedes escribir o reintentar.";
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
  const [micErrorCode, setMicErrorCode] = useState("");
  const [micNote, setMicNote] = useState("");
  const [secs, setSecs] = useState(0);
  const [typed, setTyped] = useState("");
  const [caption, setCaption] = useState("");
  const [sending, setSending] = useState(false);
  const [camActive, setCamActive] = useState(false);
  const [camNote, setCamNote] = useState("");
  const [inCallMirror, setInCallMirror] = useState(false);
  const [showMesh, setShowMesh] = useState(false);
  const [userFace, setUserFace] = useState<UserFaceObservation | undefined>(undefined);
  const [aiEmotion, setAiEmotion] = useState<AvatarEmotion>("neutral");

  const alive = useRef(true);
  const callTokenRef = useRef(0);
  const loopActiveRef = useRef(false);
  const mutedRef = useRef(false);
  const textModeRef = useRef(false);
  const userVideoRef = useRef<HTMLVideoElement>(null);
  const camStreamRef = useRef<MediaStream | null>(null);
  const camRequestRef = useRef<Promise<boolean> | null>(null);
  const userFaceRef = useRef<UserFaceObservation | undefined>(undefined);
  const recentUserLaughRef = useRef(0);
  const lastCameraLaughRef = useRef(0);
  const laughResetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isRinging = phase === "ringing";
  const mm = `${Math.floor(secs / 60)}:${(secs % 60).toString().padStart(2, "0")}`;

  const triggerUserLaughReaction = () => {
    recentUserLaughRef.current = Date.now();
    setAiEmotion("laugh");
    if (laughResetTimerRef.current) clearTimeout(laughResetTimerRef.current);
    laughResetTimerRef.current = setTimeout(() => {
      if (alive.current) {
        setAiEmotion((cur) => (cur === "laugh" ? "happy" : cur));
      }
    }, 2600);
  };

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
    // Enable user's camera by default on video call entry
    if (canUseCamera()) {
      void requestCamera();
    }
    return () => {
      alive.current = false;
      callTokenRef.current += 1;
      loopActiveRef.current = false;
      if (laughResetTimerRef.current) clearTimeout(laughResetTimerRef.current);
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
        if (now - lastAnalyze < 85) return;
        lastAnalyze = now;
        const obs = analyzeUserVideoFrame(video, now);
        if (!obs) return;
        userFaceRef.current = obs;
        if (now - lastUiSync >= 140) {
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

  // Camera laughter detection triggers natural contagious avatar laugh when not mid-reply
  useEffect(() => {
    if (!camActive || !userFace?.detected) return;
    if (userFace.emotion === "laugh" && phase !== "speaking") {
      const now = Date.now();
      if (now - lastCameraLaughRef.current > 3200) {
        lastCameraLaughRef.current = now;
        triggerUserLaughReaction();
      }
    }
  }, [camActive, userFace?.detected, userFace?.emotion, phase]);

  const requestCamera = async (): Promise<boolean> => {
    if (camRequestRef.current) return camRequestRef.current;
    const run = (async (): Promise<boolean> => {
      if (!canUseCamera()) {
        setCamNote("Este navegador no permite acceso a la cámara.");
        return false;
      }
      if (camStreamRef.current) {
        const live = camStreamRef.current.getVideoTracks().some((t) => t.readyState === "live");
        if (live) {
          setCamActive(true);
          setCamNote("");
          return true;
        }
        stopUserCamera(camStreamRef.current);
        camStreamRef.current = null;
      }
      const res = await startUserCamera();
      if (!alive.current) {
        stopUserCamera(res.stream);
        return false;
      }
      if (res.stream) {
        camStreamRef.current = res.stream;
        setCamNote("");
        setCamActive(true);
        return true;
      }
      if (res.error === "not-allowed") {
        setCamNote("Cámara bloqueada: permite el acceso en tu navegador.");
      } else if (res.error === "in-use") {
        setCamNote("La cámara está ocupada por otra pestaña o app.");
      } else {
        setCamNote("No se detectó cámara disponible. Puedes volver a intentarlo.");
      }
      setCamActive(false);
      return false;
    })();
    camRequestRef.current = run;
    try {
      return await run;
    } finally {
      if (camRequestRef.current === run) {
        camRequestRef.current = null;
      }
    }
  };

  const toggleCamera = async () => {
    if (camActive) {
      stopCamera();
      setInCallMirror(false);
      setCamNote("");
    } else {
      await requestCamera();
    }
  };

  const toggleInCallMirror = async () => {
    if (inCallMirror) {
      setInCallMirror(false);
      return;
    }
    setInCallMirror(true);
    if (!camActive) {
      const ok = await requestCamera();
      if (!ok && alive.current) {
        setInCallMirror(false);
      }
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
      await speak(text, cur.voice, cur.language, emotion, {
        localOnly: cur.localOnly,
        gender: cur.gender,
      });
    } catch (error) {
      if (isCallValid(token))
        setMicNote(error instanceof Error ? error.message : "La voz no está disponible.");
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
      const userLaughed =
        Date.now() - recentUserLaughRef.current < 5000 ||
        /\b(jaja+|jeje+|haha+|hihi+|jaj)\b/i.test(heard);
      const camCtx = userFaceRef.current?.detected ? userFaceRef.current.contextPrompt : undefined;
      const visualContext = [
        camCtx,
        userLaughed
          ? "El usuario se está riendo; responde con humor natural y sonríe o ríe si encaja."
          : undefined,
      ]
        .filter(Boolean)
        .join(" ");
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
        { callMode: true, visualContext: visualContext || undefined },
      );
      r = res.text;
      em = res.emotion;
    } catch (error) {
      if (!isCallValid(token)) return;
      setMicNote(error instanceof Error ? error.message : "No se pudo obtener una respuesta.");
      setPhase("paused");
      textModeRef.current = true;
      setTextMode(true);
      return;
    }
    if (!isCallValid(token)) return;
    addMessage({ role: "assistant", content: r });
    setCaption(r);
    setAiEmotion(em);
    setPhase("speaking");
    await speakWithBargeIn(r, token, em);
  };

  const loop = async (token: number) => {
    if (loopActiveRef.current) return;
    loopActiveRef.current = true;
    let consecutiveSttErrors = 0;
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
          onLaugh: () => {
            if (isCallValid(token)) triggerUserLaughReaction();
          },
        });
        if (!isCallValid(token)) return;

        if (res.fatalError) {
          textModeRef.current = true;
          setTextMode(true);
          setMicErrorCode(res.fatalError);
          setMicNote(micErrorText(res.fatalError));
          setPhase("paused");
          return;
        }
        if (res.error) {
          consecutiveSttErrors++;
          setMicNote(res.error);
          if (consecutiveSttErrors >= 2) {
            textModeRef.current = true;
            setTextMode(true);
            setPhase("paused");
            return;
          }
          await new Promise((r) => setTimeout(r, 550));
          continue;
        }
        consecutiveSttErrors = 0;
        if (!res.text) {
          await new Promise((r) => setTimeout(r, 80));
          continue;
        }
        setMicErrorCode("");
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
    setMicErrorCode("");
    setMicNote("");
    if (preferTextMode) {
      textModeRef.current = true;
      setTextMode(true);
    } else {
      mutedRef.current = false;
      setMuted(false);
      textModeRef.current = false;
      setTextMode(false);
    }
    unlockAudio();
    loadModel(getProfile().modelId).catch(() => {});

    // Acquire microphone and camera independently in parallel inside this user gesture
    const wantMic = !preferTextMode && canListen();
    const wantCam = !camStreamRef.current && !camRequestRef.current && canUseCamera();
    if (wantMic || wantCam) {
      const media = await acquireCallMedia(wantMic, wantCam);
      if (!isCallValid(token)) {
        stopUserCamera(media.camStream);
        return;
      }
      if (wantMic && media.micError) {
        textModeRef.current = true;
        setTextMode(true);
        setMicErrorCode(media.micError);
        setMicNote(micErrorText(media.micError));
      }
      if (media.camStream) {
        camStreamRef.current = media.camStream;
        setCamNote("");
        setCamActive(true);
      } else if (media.camError === "not-allowed") {
        setCamNote("Cámara sin permiso · toca Vídeo para reintentar");
      } else if (media.camError === "in-use") {
        setCamNote("Cámara ocupada por otra aplicación");
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
    setCaption(hi);
    setAiEmotion(greetingEmotion);
    setPhase("speaking");
    await speakWithBargeIn(hi, token, greetingEmotion);
    if (!isCallValid(token)) return;

    if (preferTextMode || textModeRef.current || mutedRef.current || !canListen()) {
      if (preferTextMode || !canListen()) {
        textModeRef.current = true;
        setTextMode(true);
      }
      if (!canListen()) {
        setMicNote("Tu navegador no permite micrófono: escribe para hablar.");
      }
      setPhase("paused");
      return;
    }

    loop(token);
  };

  const toggleMute = async () => {
    if (!started) {
      if (mutedRef.current) {
        setMicNote("");
        setMicErrorCode("");
        const err = await warmMic();
        if (!alive.current) return;
        if (err) {
          setMicErrorCode(err);
          setMicNote(micErrorText(err));
          return;
        }
        mutedRef.current = false;
        setMuted(false);
      } else {
        mutedRef.current = true;
        setMuted(true);
      }
      return;
    }

    const wantsEnableMic = muted || textModeRef.current;
    if (!wantsEnableMic) {
      setMuted(true);
      mutedRef.current = true;
      stopListening(true);
      if (phase === "listening") setPhase("paused");
      return;
    }

    setMicErrorCode("");
    setMicNote("");
    if (!canListen()) {
      setMicNote("Tu navegador no permite micrófono: escribe para hablar.");
      return;
    }
    const micErr = await warmMic();
    if (!alive.current) return;
    if (micErr) {
      setMuted(false);
      mutedRef.current = false;
      textModeRef.current = true;
      setTextMode(true);
      setMicErrorCode(micErr);
      setMicNote(micErrorText(micErr));
      setPhase("paused");
      return;
    }

    setMuted(false);
    mutedRef.current = false;
    textModeRef.current = false;
    setTextMode(false);
    if (phase === "paused" || !loopActiveRef.current) {
      loop(callTokenRef.current);
    }
  };

  const retryMic = async () => {
    setMicErrorCode("");
    setMicNote("");
    const err = await warmMic();
    if (!alive.current) return;
    if (err) {
      setMicErrorCode(err);
      setMicNote(micErrorText(err));
      return;
    }
    setMuted(false);
    mutedRef.current = false;
    setTextMode(false);
    textModeRef.current = false;
    if (started && (phase === "paused" || !loopActiveRef.current)) {
      loop(callTokenRef.current);
    }
  };

  const switchToSmartVoiceAndRetry = async () => {
    setProfile({ localOnly: false });
    await retryMic();
  };

  const toggleKeyboard = async () => {
    if (!started) {
      await start(true);
      return;
    }
    const next = !textMode;
    setTextMode(next);
    textModeRef.current = next;
    if (next) {
      stopListening(true);
      if (phase !== "speaking") {
        setPhase("paused");
      }
    } else if (!mutedRef.current && canListen()) {
      setMicErrorCode("");
      setMicNote("");
      const micErr = await warmMic();
      if (!alive.current) return;
      if (micErr) {
        textModeRef.current = true;
        setTextMode(true);
        setMicErrorCode(micErr);
        setMicNote(micErrorText(micErr));
        setPhase("paused");
        return;
      }
      if (phase === "paused" || !loopActiveRef.current) {
        loop(callTokenRef.current);
      }
    }
  };

  const hang = () => {
    alive.current = false;
    callTokenRef.current += 1;
    loopActiveRef.current = false;
    if (laughResetTimerRef.current) clearTimeout(laughResetTimerRef.current);
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
    ? inCallMirror
      ? "Espejo de prueba activo"
      : muted
        ? "Micrófono silenciado"
        : "En espera · voz lista"
    : {
        ringing: "Conectando…",
        listening: listenLabel,
        speaking: "Hablando",
        paused: muted ? "Micrófono apagado" : "Modo texto",
      }[phase];

  const isMicInactive = muted || textMode;

  return (
    <main className="mx-auto grid h-dvh w-full max-w-6xl grid-rows-[60px_minmax(0,1fr)_136px] overflow-hidden bg-background">
      <header className="flex h-[60px] items-center justify-between gap-2 border-b border-border px-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-2.5 sm:gap-3">
          <Link
            to="/"
            aria-label="Volver"
            onClick={() => {
              stopListening(true);
              stopSpeaking();
              stopCamera();
            }}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-border/60 bg-card text-foreground transition-colors hover:bg-secondary"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <AvatarImg src={p.avatar} name={p.name} className="h-9 w-9 shrink-0 text-xs" />
          <div className="min-w-0 text-left">
            <h1 className="truncate text-sm font-black leading-tight text-foreground">{p.name}</h1>
            <p className="truncate text-xs font-bold text-muted-foreground tabular-nums">
              {started && !isRinging ? `${status} · ${mm}` : status}
            </p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
          {ai.status === "loading" && (
            <span className="hidden truncate text-xs text-muted-foreground tabular-nums md:inline">
              Cargando {getModelOption(ai.modelId).name} {Math.round(ai.progress * 100)}%
            </span>
          )}
          {ai.status === "error" && (
            <span className="hidden truncate text-xs text-destructive md:inline">
              Error de modelo
            </span>
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
              className="h-8 whitespace-nowrap shrink-0 rounded-lg px-2.5 text-xs"
            >
              Usar rápido
            </Button>
          )}
          {ai.status !== "loading" && ai.status !== "error" && (
            <span className="hidden text-xs font-bold text-muted-foreground lg:inline">
              {camActive && userFace?.detected
                ? userFace.labelEs
                : p.localOnly
                  ? "Modo local · cero datos"
                  : "IA en tiempo real"}
            </span>
          )}
          {!p.localOnly && (
            <Button
              type="button"
              variant={p.voice.engine === "gemini" ? "default" : "secondary"}
              size="sm"
              aria-pressed={p.voice.engine === "gemini"}
              onClick={() => {
                const char = findCharacterById(p.characterId);
                const genderVoiceId = p.gender === "male" ? "Charon" : "Kore";
                const studioVoice =
                  char?.gender === p.gender && char?.studioVoice
                    ? char.studioVoice
                    : { engine: "gemini" as const, id: genderVoiceId };
                const nextVoice =
                  p.voice.engine === "gemini" ? { engine: "system" as const, id: "" } : studioVoice;
                setProfile({ voice: nextVoice });
              }}
              title="Alternar entre voz del dispositivo y voz neuronal natural"
              className="h-8 whitespace-nowrap shrink-0 rounded-xl px-2.5 sm:px-3 text-xs font-extrabold"
            >
              <AudioLines className="h-3.5 w-3.5" />
              <span className="hidden md:inline">
                {p.voice.engine === "gemini"
                  ? `Voz natural · ${p.voice.id || (p.gender === "male" ? "Charon" : "Kore")}`
                  : "Voz local"}
              </span>
            </Button>
          )}
          <Button
            type="button"
            variant={p.blurredBackground ? "default" : "secondary"}
            size="sm"
            aria-pressed={!!p.blurredBackground}
            onClick={() => setProfile({ blurredBackground: !p.blurredBackground })}
            title="Activar o desactivar fondo desenfocado"
            className="h-8 whitespace-nowrap shrink-0 rounded-xl px-2.5 sm:px-3 text-xs font-extrabold"
          >
            <Aperture className="h-3.5 w-3.5" />
            <span className="hidden md:inline">Fondo blur</span>
          </Button>
          <Button
            type="button"
            variant={inCallMirror ? "default" : "secondary"}
            size="sm"
            aria-pressed={inCallMirror}
            onClick={() => void toggleInCallMirror()}
            title="Probar avatar en modo espejo aquí mismo"
            className="h-8 whitespace-nowrap shrink-0 rounded-xl px-2.5 sm:px-3 text-xs font-extrabold"
          >
            <FlipHorizontal2 className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Espejo</span>
          </Button>
          <Button
            type="button"
            variant={showMesh ? "default" : "secondary"}
            size="sm"
            aria-pressed={showMesh}
            onClick={() => setShowMesh((v) => !v)}
            title="Mostrar u ocultar malla facial"
            className="h-8 whitespace-nowrap shrink-0 rounded-xl px-2.5 text-xs font-extrabold"
          >
            <ScanFace className="h-3.5 w-3.5" />
            <span className="hidden xl:inline">Malla</span>
          </Button>
          <Link
            to="/mirror"
            onClick={() => {
              stopListening(true);
              stopSpeaking();
              stopCamera();
            }}
            title="Abrir estudio Espejo lado a lado"
            aria-label="Abrir Espejo completo"
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-border/60 bg-secondary text-foreground transition-colors hover:bg-secondary/80"
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </Link>
        </div>
      </header>

      <div className="min-h-0 overflow-hidden px-3 py-2.5 sm:px-6 sm:py-3">
        <div className="relative h-full w-full overflow-hidden rounded-2xl border border-border bg-card/30 shadow-lg">
          <AnimatedAvatar
            src={p.avatar}
            name={p.name}
            active={phase === "speaking"}
            emotion={aiEmotion}
            listening={phase === "listening"}
            showMesh={showMesh}
            mirrorMode={inCallMirror}
            engine={p.avatarEngine ?? "mediapipe-mesh"}
            liveAvatarUrl={p.liveAvatarUrl}
            localOnly={p.localOnly}
            blurredBackground={p.blurredBackground}
            fit="contain"
            userFace={userFace}
            className="h-full w-full"
          />

          {inCallMirror && (
            <div className="pointer-events-none absolute left-3 top-3 rounded-lg border border-primary/40 bg-background/85 px-2.5 py-1 text-xs font-extrabold text-primary backdrop-blur-sm">
              Espejo en vivo {camActive && userFace?.detected ? `· ${userFace.labelEs}` : ""}
            </div>
          )}

          {caption && (
            <p className="call-caption" aria-live="polite">
              {caption}
            </p>
          )}
          {camActive && (
            <div className="pointer-events-none absolute bottom-3 right-3 h-28 w-20 overflow-hidden rounded-xl border border-border bg-background/90 shadow-lg sm:h-36 sm:w-28">
              <video
                ref={userVideoRef}
                playsInline
                muted
                autoPlay
                onLoadedMetadata={(e) => {
                  void e.currentTarget.play().catch(() => {});
                }}
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
                    className="w-1.5 rounded-full bg-primary transition-all duration-75"
                    style={{
                      height: `${Math.round(5 + Math.min(1, micVol * 1.4) * 13 * w)}px`,
                    }}
                  />
                ))}
              </div>
              {listenStage !== "transcribing" && (
                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  onClick={finishListening}
                  className="h-9 whitespace-nowrap shrink-0 rounded-xl px-3 text-xs font-extrabold"
                >
                  <Check className="h-3.5 w-3.5" />
                  <span>Enviar ahora</span>
                </Button>
              )}
              {(micNote || camNote) && (
                <span className="max-w-md truncate text-xs text-muted-foreground">
                  {micNote || camNote}
                </span>
              )}
            </div>
          ) : started && phase === "paused" ? (
            <div className="flex w-full max-w-2xl items-center justify-center gap-2">
              <form className="flex min-w-0 flex-1 items-center gap-2" onSubmit={submitTyped}>
                <input
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  placeholder="Escribe lo que quieres decir…"
                  disabled={sending}
                  className="h-10 min-w-0 flex-1 rounded-xl border border-border bg-card px-4 text-sm text-foreground outline-none ring-ring focus:ring-2 disabled:opacity-60"
                />
                <Button
                  variant="default"
                  type="submit"
                  disabled={!typed.trim() || sending}
                  className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl p-0"
                  aria-label="Enviar"
                >
                  <SendHorizontal className="h-4 w-4" />
                </Button>
              </form>
              {micErrorCode === "speech-api-blocked" ? (
                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  onClick={() => void switchToSmartVoiceAndRetry()}
                  className="h-10 shrink-0 rounded-xl px-3 text-xs font-extrabold"
                >
                  <Mic className="h-3.5 w-3.5" />
                  <span>Activar voz IA</span>
                </Button>
              ) : (
                micNote && (
                  <Button
                    variant="secondary"
                    size="sm"
                    type="button"
                    onClick={() => void retryMic()}
                    className="h-10 shrink-0 rounded-xl px-3 text-xs font-extrabold"
                  >
                    <RefreshCw className="h-3.5 w-3.5" />
                    <span className="hidden sm:inline">Reintentar mic</span>
                  </Button>
                )
              )}
            </div>
          ) : (
            <div className="flex max-w-full items-center justify-center gap-2">
              <p
                role={micNote ? "status" : undefined}
                className="max-w-lg truncate text-xs font-bold text-muted-foreground"
              >
                {micNote ||
                  camNote ||
                  (!started
                    ? inCallMirror
                      ? "Modo espejo listo · muévete frente a la cámara o pulsa Llamar"
                      : "Micrófono y cámara listos · pulsa el teléfono verde para hablar"
                    : phase === "speaking"
                      ? "Habla en cualquier momento para intervenir"
                      : "Conectando llamada…")}
              </p>
              {micNote && (
                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  onClick={() => void retryMic()}
                  className="h-7 shrink-0 rounded-lg px-2.5 text-xs font-bold"
                >
                  Reintentar
                </Button>
              )}
              {!micNote && camNote && (
                <Button
                  variant="secondary"
                  size="sm"
                  type="button"
                  onClick={() => void requestCamera()}
                  className="h-7 shrink-0 rounded-lg px-2.5 text-xs font-bold"
                >
                  Reintentar vídeo
                </Button>
              )}
            </div>
          )}
        </div>

        <div className="flex h-14 items-center justify-center">
          {!started ? (
            <div className="flex items-center justify-center gap-3 sm:gap-4">
              <Button
                variant="ghost"
                type="button"
                onClick={hang}
                className="flex h-11 w-11 items-center justify-center rounded-full bg-destructive text-destructive-foreground hover:bg-destructive/90"
                aria-label="Cancelar"
              >
                <PhoneOff className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                type="button"
                onClick={() => void toggleMute()}
                className={`flex h-11 w-11 items-center justify-center rounded-full p-0 ${
                  muted
                    ? "bg-foreground text-background hover:bg-foreground/90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                }`}
                aria-label={muted ? "Activar micrófono" : "Silenciar micrófono"}
                title={muted ? "Micrófono silenciado" : "Micrófono activo"}
              >
                {muted ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
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
                variant="ghost"
                type="button"
                onClick={() => void toggleCamera()}
                className={`flex h-11 w-11 items-center justify-center rounded-full p-0 ${
                  camActive
                    ? "bg-primary text-primary-foreground hover:bg-primary/90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                }`}
                aria-label={camActive ? "Apagar cámara" : "Encender cámara"}
                title={camActive ? "Cámara encendida" : "Probar cámara"}
              >
                {camActive ? <Video className="h-4 w-4" /> : <VideoOff className="h-4 w-4" />}
              </Button>
              <Button
                variant="secondary"
                type="button"
                onClick={() => void start(true)}
                className="flex h-11 w-11 items-center justify-center rounded-full"
                aria-label="Escribir en llamada"
                title="Empezar escribiendo"
              >
                <Keyboard className="h-4 w-4" />
              </Button>
            </div>
          ) : (
            <div className="mx-auto flex w-fit items-center gap-2.5 sm:gap-3 rounded-2xl border border-border bg-card px-4 py-1.5">
              <Button
                variant="ghost"
                type="button"
                onClick={() => void toggleMute()}
                className={`flex h-11 w-11 items-center justify-center rounded-xl p-0 ${
                  isMicInactive
                    ? "bg-foreground text-background hover:bg-foreground/90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                }`}
                aria-label={isMicInactive ? "Activar micrófono" : "Silenciar"}
              >
                {isMicInactive ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                type="button"
                onClick={() => void toggleCamera()}
                className={`flex h-11 w-11 items-center justify-center rounded-xl p-0 ${
                  camActive
                    ? "bg-primary text-primary-foreground hover:bg-primary/90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                }`}
                aria-label={camActive ? "Apagar cámara" : "Encender cámara"}
              >
                {camActive ? <Video className="h-4 w-4" /> : <VideoOff className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                type="button"
                onClick={() => void toggleInCallMirror()}
                className={`flex h-11 w-11 items-center justify-center rounded-xl p-0 ${
                  inCallMirror
                    ? "bg-primary text-primary-foreground hover:bg-primary/90"
                    : "bg-secondary text-foreground hover:bg-secondary/80"
                }`}
                aria-label="Alternar modo espejo"
                title="Probar avatar en modo espejo"
              >
                <FlipHorizontal2 className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                type="button"
                onClick={() => void toggleKeyboard()}
                className={`flex h-11 w-11 items-center justify-center rounded-xl p-0 ${
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
                className="flex h-11 w-11 items-center justify-center rounded-xl bg-destructive p-0 text-destructive-foreground hover:bg-destructive/90"
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
