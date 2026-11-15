import { createFileRoute, Link } from "@tanstack/react-router";
import { Aperture, ArrowLeft, Camera, CameraOff, SlidersHorizontal, ScanFace } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { AnimatedAvatar } from "@/components/AnimatedAvatar";
import { Button } from "@/components/ui/button";
import { setProfile, useProfile } from "@/lib/store";
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
  const [style, setStyle] = useState<"auto" | "human" | "anime">(p.animationStyle ?? "auto");
  const [pose, setPose] = useState<"neutral" | "happy" | "laugh" | "angry" | "sad" | "surprised">(
    "neutral",
  );
  const [strength, setStrength] = useState(0.8);
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
      stopUserCamera(stream.current);
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
      if (document.hidden || now - last < 80 || !video.current) return;
      last = now;
      const observation = analyzeUserVideoFrame(video.current, now);
      if (observation) setFace(observation);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running]);
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
              active={false}
              emotion={pose}
              showMesh={showMesh}
              blurredBackground={p.blurredBackground}
              animationStyle={style}
              userFace={adjusted}
              mirrorMode={mode === "mirror"}
              fit="cover"
              className="h-full w-full"
            />
          </div>
          <h2 className="mt-3 text-sm font-bold">{p.name}</h2>
          <p className="text-xs text-muted-foreground">
            {mode === "mirror" ? "Espejo directo" : "Gestos sutiles"}
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
        <label className="flex items-center gap-2 text-sm">
          Retrato
          <select
            aria-label="Tipo de animación"
            className="rounded-lg border border-input bg-card px-2.5 py-1.5 text-sm outline-none"
            value={style}
            onChange={(e) => setStyle(e.target.value as typeof style)}
          >
            <option value="auto">Automático</option>
            <option value="human">Persona</option>
            <option value="anime">Anime / dibujo</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm">
          Expresión
          <select
            aria-label="Probar expresión"
            className="rounded-lg border border-input bg-card px-2.5 py-1.5 text-sm outline-none"
            value={pose}
            onChange={(e) => setPose(e.target.value as typeof pose)}
          >
            <option value="neutral">En reposo</option>
            <option value="happy">Sonrisa</option>
            <option value="laugh">Risa</option>
            <option value="angry">Ceño fruncido</option>
            <option value="sad">Comisuras bajas</option>
            <option value="surprised">Sorpresa</option>
          </select>
        </label>
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
