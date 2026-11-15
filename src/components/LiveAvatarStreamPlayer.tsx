import { ExternalLink, RefreshCw, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { setProfile } from "@/lib/store";
import { Button } from "./ui/button";

export function LiveAvatarStreamPlayer({
  serverUrl,
  name,
  fallbackPhoto,
}: {
  serverUrl?: string | undefined;
  active: boolean;
  name: string;
  fallbackPhoto: string;
}) {
  const [status, setStatus] = useState<"checking" | "connected" | "offline">("checking");
  const [detail, setDetail] = useState("");

  let safeUrl: string | undefined;
  try {
    const parsed = new URL(serverUrl ?? "");
    if (["http:", "https:"].includes(parsed.protocol)) safeUrl = parsed.href;
  } catch {
    safeUrl = undefined;
  }

  const probe = async () => {
    if (!safeUrl) {
      setStatus("offline");
      setDetail("Configura una URL válida en Personalizar.");
      return;
    }
    setStatus("checking");
    try {
      const res = await fetch(`/api/liveavatar/status?url=${encodeURIComponent(safeUrl)}`);
      const data = (await res.json()) as { ok?: boolean; error?: string };
      if (data.ok) {
        setStatus("connected");
        setDetail("Servidor externo accesible en tu equipo con GPU.");
      } else {
        setStatus("offline");
        setDetail(data.error || "No responde el servidor externo.");
      }
    } catch {
      setStatus("offline");
      setDetail("No se pudo comprobar el estado del servidor.");
    }
  };

  useEffect(() => {
    void probe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safeUrl]);

  return (
    <div className="absolute inset-0 flex flex-col bg-background">
      <img src={fallbackPhoto} alt={name} className="min-h-0 flex-1 object-cover object-top" />
      <div className="space-y-3 border-t border-border bg-card/90 p-4 text-center backdrop-blur-md">
        <p className="text-sm font-medium text-foreground">
          Estudio externo ·{" "}
          <span className={status === "connected" ? "text-accent" : "text-muted-foreground"}>
            {status === "checking"
              ? "Comprobando…"
              : status === "connected"
                ? "Disponible"
                : "Desconectado"}
          </span>
        </p>
        <p className="text-xs text-muted-foreground">
          {detail ||
            "Usa nuestro motor 3D GPU integrado para animar ojos, cabello, cuello y manos al instante."}
        </p>
        <div className="flex flex-wrap justify-center gap-2">
          <Button onClick={() => setProfile({ avatarEngine: "mediapipe-mesh" })}>
            <Sparkles />
            Usar Avatar 3D GPU
          </Button>
          <Button variant="secondary" onClick={probe}>
            <RefreshCw />
            Reintentar
          </Button>
          {safeUrl && (
            <Button asChild variant="outline">
              <a href={safeUrl} target="_blank" rel="noreferrer">
                <ExternalLink />
                Abrir estudio
              </a>
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
