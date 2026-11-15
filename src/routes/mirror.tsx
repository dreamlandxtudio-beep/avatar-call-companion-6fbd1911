import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowLeft, Camera, CameraOff, SlidersHorizontal } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { AnimatedAvatar } from "@/components/AnimatedAvatar";
import { Button } from "@/components/ui/button";
import { useProfile } from "@/lib/store";
import { analyzeUserVideoFrame, ensureUserFaceTracker, startUserCamera, stopUserCamera, type UserFaceObservation } from "@/lib/userCamera";

export const Route = createFileRoute("/mirror")({
  head: () => ({ meta: [
    { title: "Espejo de expresiones — YUME" },
    { name: "description", content: "Prueba la mirada, los párpados y los gestos de tu personaje con tu cámara." },
    { property: "og:title", content: "Espejo de expresiones — YUME" },
    { property: "og:description", content: "Tu personaje y tus gestos, lado a lado." },
    { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary" },
  ] }),
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
  const [strength, setStrength] = useState(0.8);
  const stop = () => { generation.current++; stopUserCamera(stream.current); stream.current = null; setRunning(false); setLoading(false); setFace(undefined); };
  useEffect(() => () => { generation.current++; stopUserCamera(stream.current); }, []);
  const start = async () => {
    const id = ++generation.current;
    setLoading(true); setError("");
    const result = await startUserCamera();
    if (id !== generation.current) { stopUserCamera(result.stream); return; }
    if (!result.stream) { setError(result.error === "not-allowed" ? "Permite el acceso a la cámara para probar el espejo." : "No se encontró una cámara disponible."); setLoading(false); return; }
    stream.current = result.stream;
    if (video.current) { video.current.srcObject = result.stream; await video.current.play().catch(() => {}); }
    const ready = await ensureUserFaceTracker();
    if (id !== generation.current) return;
    setLoading(false);
    if (!ready) { stopUserCamera(result.stream); stream.current = null; setError("No se pudo cargar el seguimiento facial. Comprueba la conexión y vuelve a probar."); return; }
    setRunning(true);
  };
  useEffect(() => {
    if (!running) return;
    let raf = 0; let last = 0;
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
  const adjusted = face ? { ...face,
    gazeX: face.gazeX * strength, gazeY: face.gazeY * strength,
    headYaw: face.headYaw * strength, headPitch: face.headPitch * strength,
    headRoll: (face.headRoll ?? 0) * strength,
  } : undefined;
  return <main className="mx-auto max-w-6xl px-4 py-5 sm:px-8">
    <header className="mb-5 flex items-center gap-3">
      <Button asChild variant="ghost" size="icon" aria-label="Volver al personaje"><Link to="/chat"><ArrowLeft /></Link></Button>
      <div className="min-w-0 flex-1"><h1 className="text-xl font-extrabold">Espejo de expresiones</h1><p className="text-sm text-muted-foreground">{p.name}</p></div>
      <Button asChild variant="ghost" size="icon" aria-label="Personalizar"><Link to="/profile"><SlidersHorizontal /></Link></Button>
    </header>
    <div className="mirror-stage">
      <section><div className="mirror-portrait relative"><video ref={video} muted playsInline className="h-full w-full -scale-x-100 object-cover" />{!running && !loading && <div className="absolute inset-0 flex items-center justify-center text-muted-foreground"><Camera className="h-10 w-10" /></div>}</div><h2 className="mt-3 text-sm font-bold">Tú</h2><p className="min-h-10 text-xs text-muted-foreground">{face?.labelEs ?? (loading ? "Preparando cámara…" : running ? "Buscando rostro…" : "Cámara apagada")}</p></section>
      <section><div className="mirror-portrait"><AnimatedAvatar src={p.avatar} name={p.name} active={false} userFace={adjusted} mirrorMode={mode === "mirror"} fit="cover" className="h-full w-full" /></div><h2 className="mt-3 text-sm font-bold">{p.name}</h2><p className="text-xs text-muted-foreground">{mode === "mirror" ? "Espejo" : "Gestos sutiles"}</p></section>
    </div>
    <div className="mt-5 flex flex-wrap items-center gap-4 border-t border-border pt-5">
      <Button onClick={running || loading ? stop : () => void start()}>{running || loading ? <CameraOff /> : <Camera />}{running || loading ? "Apagar cámara" : "Activar cámara"}</Button>
      <div className="flex gap-1 rounded-full bg-secondary p-1" aria-label="Modo de imitación">
        <Button size="sm" variant={mode === "mirror" ? "default" : "ghost"} onClick={() => setMode("mirror")}>Espejo</Button>
        <Button size="sm" variant={mode === "subtle" ? "default" : "ghost"} onClick={() => setMode("subtle")}>Sutil</Button>
      </div>
      <label className="flex items-center gap-3 text-sm">Intensidad<input aria-label="Intensidad de imitación" type="range" min="0.2" max="1" step="0.1" value={strength} onChange={e => setStrength(Number(e.target.value))} /></label>
    </div>
    {error && <p role="alert" className="mt-4 text-sm text-destructive">{error}</p>}
    {face?.detected && <div className="mt-6 grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-4">{[["Sonrisa", face.smile], ["Boca", face.mouthOpen], ["Párpados", face.blink ?? 0], ["Ceño", face.browFurrow], ["Cejas", face.browRaise], ["Comisuras bajas", face.mouthFrown ?? 0], ["Mirada", Math.abs(face.gazeX)], ["Nariz", face.noseSneer ?? 0]].map(([label, value]) => <label key={String(label)} className="text-xs text-muted-foreground">{label}<progress className="mt-1 block h-2 w-full" max="1" value={Number(value)} /></label>)}</div>}
  </main>;
}