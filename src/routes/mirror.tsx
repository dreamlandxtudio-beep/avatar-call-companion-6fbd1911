import { createFileRoute, Link } from "@tanstack/react-router";
import {
  Aperture,
  ArrowLeft,
  Camera,
  CameraOff,
  Droplets,
  ScanFace,
  SlidersHorizontal,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { AnimatedAvatar } from "@/components/AnimatedAvatar";
import { UserFaceMeshOverlay } from "@/components/UserFaceMeshOverlay";
import { Button } from "@/components/ui/button";
import { AVATAR_EXPRESSIONS, type AvatarEmotion } from "@/lib/ai";
import { setProfile, useProfile } from "@/lib/store";
import { playEmotionVocalCue, speak, stopSpeaking } from "@/lib/voice";
import {
  analyzeUserVideoFrame,
  ensureUserFaceTracker,
  startUserCamera,
  stopUserCamera,
  type UserFaceObservation,
} from "@/lib/userCamera";

export const Route = createFileRoute("/mirror")({
  head: () => ({
    meta: [
      { title: "Espejo de expresiones — YUME" },
      {
        name: "description",
        content: "Prueba la mirada, los párpados y los gestos de tu personaje con tu cámara.",
      },
      { property: "og:title", content: "Espejo de expresiones — YUME" },
      { property: "og:description", content: "Tu personaje y tus gestos, lado a lado." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Mirror,
});

const SAMPLE_SPEECH_ES: Partial<Record<AvatarEmotion, string>> = {
  neutral: "Hola, aquí estoy contigo probando cada movimiento facial con calma.",
  happy: "¡Qué alegría verte hoy! Me encanta cómo se siente conversar así contigo.",
  euphoric: "¡Esto es increíble, no sabes lo emocionado que me siento ahora mismo!",
  laugh: "¡Jajaja, qué ocurrencia! De verdad me hiciste reír muchísimo con eso.",
  smirk: "Qué curioso… casi parece que me estás ocultando algo interesante.",
  compassionate: "Ven acá, todo va a estar bien. Estoy contigo pase lo que pase.",
  intense: "Mírame bien… sabes perfectamente que no voy a apartar la vista de ti.",
  bitingLip: "No me hagas mirarte así, que luego me cuesta disimular lo que quiero.",
  sarcastic: "Ah, claro… seguro que fue una coincidencia inocente, ¿verdad?",
  eyeroll: "En serio no puedo creer que vuelvas a decir eso otra vez.",
  sad: "Lo siento mucho… me pesa verte pasar por un momento tan difícil.",
  crying: "Perdóname… se me llenan los ojos de lágrimas al escuchar todo esto.",
  heartbroken: "Me rompe el corazón saber lo que pasó, estoy aquí contigo en este duelo.",
  angry: "¡Ya basta! No me parece justo cómo están tratando esta situación.",
  furious: "¡Es completamente inaceptable, me hierve la sangre pensar en eso!",
  surprised: "¡¿En serio?! No puedo creer lo que me estás contando ahora mismo.",
  nervous: "Ay… estoy un poco nervioso, siento que me sudan hasta las sienes.",
  sleepy: "Qué sueño tengo… apenas puedo mantener los párpados abiertos.",
};

function Mirror() {
  const p = useProfile();
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const generation = useRef(0);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [face, setFace] = useState<UserFaceObservation>();
  const [mode, setMode] = useState<"mirror" | "subtle">("mirror");
  const [showMesh, setShowMesh] = useState(false);
  const [speakingTest, setSpeakingTest] = useState(false);
  const [pose, setPose] = useState<AvatarEmotion>("neutral");
  const [strength, setStrength] = useState(0.8);
  const [swallowSignal, setSwallowSignal] = useState<{ seq: number; strength: number } | undefined>(
    undefined,
  );
  const triggerSwallow = (swStrength: number) => {
    setSwallowSignal((prev) => ({ seq: (prev?.seq ?? 0) + 1, strength: swStrength }));
  };
  const stop = () => {
    generation.current++;
    stopUserCamera(stream.current);
    stream.current = null;
    if (video.current) {
      video.current.srcObject = null;
    }
    setRunning(false);
    setLoading(false);
    setFace(undefined);
  };
  useEffect(
    () => () => {
      generation.current++;
      speechGen.current++;
      stopUserCamera(stream.current);
      stopSpeaking();
    },
    [],
  );
  const start = async () => {
    const id = ++generation.current;
    setLoading(true);
    setError("");
    const result = await startUserCamera();
    if (id !== generation.current) {
      stopUserCamera(result.stream);
      return;
    }
    if (!result.stream) {
      setError(
        result.error === "not-allowed"
          ? "Permite el acceso a la cámara en tu navegador para probar el espejo."
          : result.error === "in-use"
            ? "La cámara está ocupada por otra pestaña o aplicación."
            : "No se encontró una cámara disponible.",
      );
      setLoading(false);
      return;
    }
    stream.current = result.stream;
    setRunning(true);
    if (video.current) {
      video.current.srcObject = result.stream;
      await video.current.play().catch(() => {});
    }
    const ready = await ensureUserFaceTracker();
    if (id !== generation.current) return;
    setLoading(false);
    if (!ready) {
      setError(
        "Cámara encendida, pero el seguimiento facial está cargando o bloqueado; prueba las expresiones manuales abajo.",
      );
    }
  };
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    let last = 0;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (document.hidden || now - last < 33 || !video.current) return;
      last = now;
      const observation = analyzeUserVideoFrame(video.current, now, { includeMesh: showMesh });
      if (observation) setFace(observation);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, showMesh]);
  const speechGen = useRef(0);
  const toggleSpeechTest = async () => {
    if (speakingTest) {
      speechGen.current++;
      stopSpeaking();
      setSpeakingTest(false);
      return;
    }
    const token = ++speechGen.current;
    setSpeakingTest(true);
    try {
      if (
        pose === "laugh" ||
        pose === "crying" ||
        pose === "heartbroken" ||
        pose === "sad" ||
        pose === "melancholic" ||
        pose === "sleepy" ||
        pose === "surprised" ||
        pose === "fear"
      ) {
        await playEmotionVocalCue(pose, p.gender).catch(() => {});
        if (token !== speechGen.current) return;
      }
      const sample =
        SAMPLE_SPEECH_ES[pose] ??
        `Probando la expresión ${AVATAR_EXPRESSIONS.find((e) => e.id === pose)?.label ?? pose} al hablar con naturalidad.`;
      await speak(sample, p.voice, p.language, pose, {
        localOnly: p.localOnly,
        gender: p.gender,
      });
    } catch {
      /* voice fallback ignored */
    } finally {
      if (token === speechGen.current) {
        setSpeakingTest(false);
      }
    }
  };
  const adjusted = face
    ? {
        ...face,
        gazeX: face.gazeX * strength,
        gazeY: face.gazeY * strength,
        headYaw: face.headYaw * strength,
        headPitch: face.headPitch * strength,
        headRoll: (face.headRoll ?? 0) * strength,
      }
    : undefined;
  const activeEntry = AVATAR_EXPRESSIONS.find((e) => e.id === pose) ?? AVATAR_EXPRESSIONS[0]!;
  const categories = Array.from(new Set(AVATAR_EXPRESSIONS.map((e) => e.category)));
  return (
    <main className="mx-auto max-w-6xl px-4 py-5 pb-16 sm:px-8">
      <header className="mb-5 flex items-center gap-3 border-b border-border pb-4">
        <Link
          to="/chat"
          aria-label="Volver al personaje"
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-border/60 bg-card text-foreground transition-colors hover:bg-secondary"
        >
          <ArrowLeft className="h-4 w-4" />
        </Link>
        <div className="min-w-0 flex-1">
          <h1 className="text-xl font-black">Espejo de expresiones</h1>
          <p className="text-sm text-muted-foreground">{p.name}</p>
        </div>
        <Link
          to="/profile"
          aria-label="Personalizar"
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-border/60 bg-card text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
        >
          <SlidersHorizontal className="h-4 w-4" />
        </Link>
      </header>
      <div className="mirror-stage">
        <section>
          <div className="mirror-portrait relative border border-border">
            <video
              ref={video}
              muted
              autoPlay
              playsInline
              onLoadedMetadata={(e) => {
                void e.currentTarget.play().catch(() => {});
              }}
              className="h-full w-full -scale-x-100 object-cover"
            />
            <UserFaceMeshOverlay
              meshPoints={face?.meshPoints}
              visible={running && showMesh && !!face?.detected}
              videoSize={{
                width: video.current?.videoWidth || 640,
                height: video.current?.videoHeight || 480,
              }}
            />
            {!running && !loading && (
              <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
                <Camera className="h-10 w-10" />
              </div>
            )}
          </div>
          <h2 className="mt-3 text-sm font-bold">Tú</h2>
          <p className="min-h-10 text-xs text-muted-foreground">
            {face?.labelEs ??
              (loading
                ? "Preparando cámara…"
                : running
                  ? "Cámara activa · buscando rostro…"
                  : "Cámara apagada")}
          </p>
        </section>
        <section>
          <div className="mirror-portrait relative border border-border shadow-md">
            <AnimatedAvatar
              src={p.avatar}
              name={p.name}
              active={speakingTest}
              emotion={pose}
              showMesh={showMesh}
              blurredBackground={p.blurredBackground}
              animationStyle="human"
              userFace={adjusted}
              mirrorMode={mode === "mirror"}
              swallowSignal={swallowSignal}
              fit="cover"
              className="h-full w-full"
            />
          </div>
          <h2 className="mt-3 text-sm font-bold">{p.name}</h2>
          <p className="text-xs text-muted-foreground">
            {mode === "mirror" ? "Espejo directo" : "Gestos sutiles"} · {activeEntry.label}
          </p>
        </section>
      </div>
      <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-border pt-5">
        <Button
          onClick={running || loading ? stop : () => void start()}
          className="h-11 whitespace-nowrap shrink-0 rounded-xl"
        >
          {running || loading ? <CameraOff /> : <Camera />}
          {running || loading ? "Apagar cámara" : "Activar cámara"}
        </Button>
        <Button
          variant={speakingTest ? "default" : "outline"}
          aria-pressed={speakingTest}
          onClick={() => void toggleSpeechTest()}
          className="h-11 whitespace-nowrap shrink-0 rounded-xl"
        >
          {speakingTest ? <VolumeX /> : <Volume2 />}
          {speakingTest ? "Detener voz" : "Probar habla"}
        </Button>
        <Button
          variant="outline"
          type="button"
          onClick={() => triggerSwallow(1.0)}
          className="h-11 whitespace-nowrap shrink-0 rounded-xl"
        >
          <Droplets />
          Tragar saliva
        </Button>
        <Button
          variant="outline"
          type="button"
          onClick={() => triggerSwallow(1.45)}
          className="h-11 whitespace-nowrap shrink-0 rounded-xl"
        >
          <Droplets />
          Tragar fuerte
        </Button>
        <div
          className="flex gap-1 rounded-lg border border-border bg-card p-1"
          aria-label="Modo de imitación"
        >
          <button
            type="button"
            onClick={() => setMode("mirror")}
            className={`whitespace-nowrap shrink-0 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === "mirror"
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Espejo
          </button>
          <button
            type="button"
            onClick={() => setMode("subtle")}
            className={`whitespace-nowrap shrink-0 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              mode === "subtle"
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Sutil
          </button>
        </div>
        <Button
          variant={showMesh ? "default" : "outline"}
          aria-pressed={showMesh}
          onClick={() => setShowMesh(!showMesh)}
          className="whitespace-nowrap shrink-0 rounded-lg"
        >
          <ScanFace />
          Malla facial
        </Button>
        <Button
          variant={p.blurredBackground ? "default" : "outline"}
          aria-pressed={!!p.blurredBackground}
          onClick={() => setProfile({ blurredBackground: !p.blurredBackground })}
          className="whitespace-nowrap shrink-0 rounded-lg"
        >
          <Aperture />
          Fondo desenfocado
        </Button>
        <label className="flex items-center gap-3 text-sm">
          Intensidad
          <input
            aria-label="Intensidad de imitación"
            type="range"
            min="0.2"
            max="1"
            step="0.1"
            value={strength}
            onChange={(e) => setStrength(Number(e.target.value))}
            className="accent-primary"
          />
        </label>
      </div>
      <section className="mt-6 border-t border-border pt-5" aria-label="Laboratorio de expresiones">
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <h2 className="text-base font-black text-foreground">
              Estudio de expresiones y estados
            </h2>
            <p className="text-xs text-muted-foreground">
              Activa cualquier gesto facial o estado físico para probar párpados, cejas, dientes,
              lágrimas y sudor.
            </p>
          </div>
          <span className="text-xs font-bold text-primary">
            Activo: {activeEntry.label} — {activeEntry.detail}
          </span>
        </div>
        <div className="space-y-4">
          {categories.map((cat) => (
            <div key={cat}>
              <h3 className="mb-2 text-xs font-extrabold uppercase tracking-wider text-muted-foreground">
                {cat}
              </h3>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
                {AVATAR_EXPRESSIONS.filter((e) => e.category === cat).map((entry) => {
                  const selected = pose === entry.id;
                  return (
                    <button
                      key={entry.id}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => setPose(entry.id)}
                      className={`flex flex-col items-start rounded-xl border px-3 py-2.5 text-left transition-colors ${
                        selected
                          ? "border-primary bg-primary/12 text-foreground shadow-sm"
                          : "border-border/70 bg-card/70 text-foreground/90 hover:border-border hover:bg-secondary/60"
                      }`}
                    >
                      <span className="text-xs font-extrabold leading-snug">{entry.label}</span>
                      <span className="mt-0.5 text-[11px] font-medium leading-tight text-muted-foreground">
                        {entry.detail}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </section>
      {error && (
        <p role="alert" className="mt-4 text-sm text-destructive">
          {error}
        </p>
      )}
      {face?.detected && (
        <div className="mt-6 grid grid-cols-2 gap-4 rounded-2xl border border-border bg-card/60 p-4 sm:grid-cols-4">
          {[
            ["Sonrisa", face.smile],
            ["Boca", face.mouthOpen],
            ["Párpados", face.blink ?? 0],
            ["Ceño", face.browFurrow],
            ["Cejas", face.browRaise],
            ["Comisuras bajas", face.mouthFrown ?? 0],
            ["Mirada", Math.abs(face.gazeX)],
            ["Nariz", face.noseSneer ?? 0],
          ].map(([label, value]) => {
            const pct = Math.round(Math.max(0, Math.min(1, Number(value))) * 100);
            return (
              <div key={String(label)} className="text-xs text-muted-foreground">
                <div className="flex items-center justify-between">
                  <span>{label}</span>
                  <span className="tabular-nums text-foreground/80">{pct}%</span>
                </div>
                <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full rounded-full bg-primary transition-all duration-100"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </main>
  );
}
