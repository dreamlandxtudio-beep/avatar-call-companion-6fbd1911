import { ExternalLink, RefreshCw, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { setProfile } from "@/lib/store";
import { Button } from "./ui/button";

export function LiveAvatarStreamPlayer({
  serverUrl,
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
    setStatus("offline");
    setDetail("Falta conectar una sesión de vídeo del servidor GPU; una URL de estudio no genera vídeo en esta llamada.");
  };

  useEffect(() => {
    void probe();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safeUrl]);

  return (
    <div className="absolute inset-x-0 bottom-0 flex flex-col">
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
            "Animación local activa; vídeo LiveAvatar no conectado."}
        </p>
        <div className="flex flex-wrap justify-center gap-2">
          <Button onClick={() => setProfile({ avatarEngine: "mediapipe-mesh" })}>
            <Sparkles />
            Usar animación local
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
