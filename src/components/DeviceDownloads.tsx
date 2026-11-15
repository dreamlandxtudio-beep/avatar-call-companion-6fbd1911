import { useEffect, useState } from "react";
import { Download, HardDrive, RefreshCw } from "lucide-react";
import { Button } from "./ui/button";
import { AI_MODELS, loadModel, useAIState } from "@/lib/ai";
import { setProfile, useProfile, type ModelId } from "@/lib/store";
import { deviceStorageInfo, requestDeviceStorage } from "@/lib/device-storage";
import { loadKokoro } from "@/lib/voice";

export function DeviceDownloads() {
  const profile = useProfile(); const ai = useAIState();
  const [selected, setSelected] = useState<ModelId>("qwen25-0.5b");
  const [info, setInfo] = useState({ usedMB: 0, persistent: false });
  const [voiceProgress, setVoiceProgress] = useState<number | null>(null);
  const [voiceReady, setVoiceReady] = useState(false);
  const [error, setError] = useState("");
  const refresh = () => void deviceStorageInfo().then(setInfo).catch(() => {});
  useEffect(refresh, [ai.status, voiceReady]);
  const activate = async () => {
    setError("");
    try { await requestDeviceStorage(); await loadModel(selected); setProfile({ modelId: selected }); refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : "No se pudo descargar el modelo."); }
  };
  const downloadVoice = async () => {
    setError(""); setVoiceProgress(0);
    try { await requestDeviceStorage(); await loadKokoro(setVoiceProgress); setVoiceReady(true); refresh(); }
    catch (err) { setError(err instanceof Error ? err.message : "No se pudo descargar la voz."); }
    finally { setVoiceProgress(null); }
  };
  return <section className="space-y-4 border-t border-border py-6" aria-label="Descargas del dispositivo">
    <div className="flex items-center gap-3"><HardDrive className="h-5 w-5 text-primary" /><h2 className="text-lg font-extrabold">En tu dispositivo</h2><Button variant="ghost" size="icon" aria-label="Actualizar almacenamiento" onClick={refresh}><RefreshCw className="h-4 w-4" /></Button></div>
    <p className="text-xs text-muted-foreground">{info.usedMB} MB guardados · {info.persistent ? "Almacenamiento protegido" : "El navegador puede liberar las descargas si falta espacio"}</p>
    <div className="flex flex-wrap gap-2"><select aria-label="Modelo para descargar" className="min-w-0 flex-1 rounded-xl border border-input bg-card px-3 py-2 text-sm" value={selected} onChange={e => setSelected(e.target.value as ModelId)}>{AI_MODELS.filter(m => !m.cloud).map(m => <option key={m.id} value={m.id}>{m.name} · ~{m.cpuMB} MB</option>)}</select><Button disabled={ai.status === "loading"} onClick={() => void activate()}><Download className="h-4 w-4" />{ai.status === "loading" ? `${Math.round(ai.progress * 100)}%` : "Descargar y usar"}</Button></div>
    {ai.status === "ready" && ai.device !== "Nube" && <p role="status" className="text-sm text-accent">{AI_MODELS.find(m => m.id === ai.modelId)?.name} listo · {ai.device}</p>}
    <div className="flex flex-wrap items-center gap-3"><Button variant="secondary" disabled={voiceProgress !== null} onClick={() => void downloadVoice()}><Download className="h-4 w-4" />{voiceProgress !== null ? `Voz ${Math.round(voiceProgress * 100)}%` : voiceReady ? "Voz local lista" : "Descargar voz Kokoro · EN"}</Button>{voiceReady && <Button variant="outline" onClick={() => setProfile({ language: "en-US", voice: { engine: "kokoro", id: profile.gender === "male" ? "am_michael" : "af_heart" } })}>Usar voz en inglés</Button>}</div>
    <p className="text-xs text-muted-foreground">Primera descarga con internet. Después se reutilizan los archivos guardados; al abrir la app se vuelve a preparar el modelo.</p>
    {error && <p role="alert" className="text-xs text-destructive">{error}</p>}
  </section>;
}