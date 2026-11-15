import { useEffect, useState } from "react";
import { Check, Cpu, Download, HardDrive, RefreshCw } from "lucide-react";
import { Button } from "./ui/button";
import { AI_MODELS, detectWebGPU, loadModel, useAIState } from "@/lib/ai";
import { setProfile, useProfile, type ModelId } from "@/lib/store";
import { deviceStorageInfo, requestDeviceStorage } from "@/lib/device-storage";
import { loadKokoro } from "@/lib/voice";

export function DeviceDownloads() {
  const profile = useProfile();
  const ai = useAIState();
  const [hasGPU, setHasGPU] = useState(false);
  const [info, setInfo] = useState({ usedMB: 0, persistent: false });
  const [voiceProgress, setVoiceProgress] = useState<number | null>(null);
  const [voiceReady, setVoiceReady] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    detectWebGPU().then(setHasGPU);
  }, []);

  const refresh = () =>
    void deviceStorageInfo()
      .then(setInfo)
      .catch(() => {});
  useEffect(refresh, [ai.status, voiceReady]);

  const localModels = AI_MODELS.filter((m) => !m.cloud);

  const activateModel = async (id: ModelId) => {
    setError("");
    try {
      await requestDeviceStorage();
      await loadModel(id);
      setProfile({ modelId: id });
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo descargar el modelo.");
    }
  };

  const downloadVoice = async () => {
    setError("");
    setVoiceProgress(0);
    try {
      await requestDeviceStorage();
      await loadKokoro(setVoiceProgress);
      setVoiceReady(true);
      refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo descargar la voz.");
    } finally {
      setVoiceProgress(null);
    }
  };

  return (
    <section
      className="space-y-4 border-t border-border py-6"
      aria-label="Modelos locales descargables"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <HardDrive className="h-5 w-5 text-primary" />
          <div>
            <h2 className="text-lg font-black">Modelos locales de IA</h2>
            <p className="text-xs text-muted-foreground">
              {info.usedMB} MB guardados · {hasGPU ? "WebGPU disponible" : "Modo CPU"} ·{" "}
              {info.persistent ? "Almacenamiento protegido" : "Caché del navegador"}
            </p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Actualizar almacenamiento"
          onClick={refresh}
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
      </div>

      <div className="grid gap-2.5 sm:grid-cols-2">
        {localModels.map((m) => {
          const isReady = ai.status === "ready" && ai.modelId === m.id;
          const isLoading = ai.status === "loading" && ai.modelId === m.id;
          const isSelected = profile.modelId === m.id;
          const mb = hasGPU ? m.gpuMB : m.cpuMB;

          return (
            <div
              key={m.id}
              className={`flex flex-col justify-between gap-3 rounded-xl border p-3.5 transition-colors ${
                isSelected || isReady
                  ? "border-primary/70 bg-card"
                  : "border-border bg-card/60 hover:border-border/90"
              }`}
            >
              <div>
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <Cpu className="h-4 w-4 shrink-0 text-primary" />
                    <span className="truncate text-sm font-black text-foreground">{m.name}</span>
                  </div>
                  <span className="shrink-0 font-mono text-[11px] font-bold text-muted-foreground tabular-nums">
                    ~{mb} MB
                  </span>
                </div>
                <p className="mt-1 text-[11px] font-bold text-primary">{m.tier}</p>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{m.desc}</p>
              </div>

              {isLoading && (
                <div className="space-y-1">
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-150"
                      style={{ width: `${Math.round(ai.progress * 100)}%` }}
                    />
                  </div>
                  <p className="text-[11px] font-bold text-muted-foreground tabular-nums">
                    {ai.phase === "initializing" ? "Inicializando…" : "Descargando…"}{" "}
                    {Math.round(ai.progress * 100)}%
                  </p>
                </div>
              )}

              <Button
                type="button"
                variant={isReady ? "secondary" : "default"}
                size="sm"
                disabled={ai.status === "loading"}
                onClick={() => void activateModel(m.id)}
                className="h-10 w-full rounded-xl text-xs font-extrabold"
              >
                {isReady ? (
                  <>
                    <Check className="h-3.5 w-3.5 text-primary" />
                    <span>Listo ({ai.device || "Local"})</span>
                  </>
                ) : isLoading ? (
                  <span>Descargando {Math.round(ai.progress * 100)}%</span>
                ) : (
                  <>
                    <Download className="h-3.5 w-3.5" />
                    <span>Descargar y usar</span>
                  </>
                )}
              </Button>
            </div>
          );
        })}
      </div>

      <div className="flex flex-wrap items-center gap-3 pt-1">
        <Button
          variant="secondary"
          disabled={voiceProgress !== null}
          onClick={() => void downloadVoice()}
          className="h-10 rounded-xl text-xs font-extrabold"
        >
          <Download className="h-4 w-4" />
          {voiceProgress !== null
            ? `Descargando voces Kokoro ${Math.round(voiceProgress * 100)}%`
            : voiceReady
              ? "Banco de voces Kokoro listo"
              : "Descargar voces offline Kokoro · EN"}
        </Button>
        {voiceReady && (
          <Button
            variant="outline"
            onClick={() =>
              setProfile({
                language: "en-US",
                voice: {
                  engine: "kokoro",
                  id: profile.gender === "male" ? "am_michael" : "af_heart",
                },
              })
            }
            className="h-10 rounded-xl text-xs font-extrabold"
          >
            Usar voz Kokoro ahora
          </Button>
        )}
      </div>

      <p className="text-xs text-muted-foreground">
        Primera descarga con internet. Luego el navegador guarda los pesos en tu dispositivo para
        chatear y hablar sin enviar datos fuera.
      </p>
      {error && (
        <p role="alert" className="text-xs font-bold text-destructive">
          {error}
        </p>
      )}
    </section>
  );
}
