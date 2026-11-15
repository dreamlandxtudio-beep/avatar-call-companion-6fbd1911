import { useEffect, useState } from "react";
import {
  AI_MODELS,
  detectWebGPU,
  getModelOption,
  loadModel,
  resetModel,
  useAIState,
} from "@/lib/ai";
import { setProfile, useProfile, type ModelId } from "@/lib/store";

export function ModelBanner() {
  const s = useAIState();
  const p = useProfile();
  const [hasGPU, setHasGPU] = useState(true);

  useEffect(() => {
    detectWebGPU().then(setHasGPU);
  }, []);

  const selected = getModelOption(
    p.localOnly && getModelOption(p.modelId).cloud ? "qwen25-0.5b" : p.modelId,
  );
  if (selected.cloud) return null;

  const estMB = hasGPU ? selected.gpuMB : selected.cpuMB;
  const isDifferentLoaded = s.status === "ready" && s.modelId !== selected.id;

  if (s.status === "ready" && !isDifferentLoaded) return null;

  const onSelect = (id: ModelId) => {
    const opt = getModelOption(id);
    setProfile({
      modelId: id,
      ...(opt.cloud ? { localOnly: false } : {}),
    });
    if (s.status === "loading" || s.status === "error") {
      resetModel();
    }
  };

  return (
    <div className="mx-3 mt-2 rounded-2xl border border-border bg-card p-3 text-sm">
      {(s.status === "idle" || isDifferentLoaded) && (
        <div className="space-y-2.5">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs text-muted-foreground tabular-nums">
              IA gratuita en tu dispositivo · ~{estMB} MB
            </span>
            <button
              onClick={() => loadModel(selected.id).catch(() => {})}
              className="whitespace-nowrap shrink-0 rounded-full bg-primary px-3.5 py-1.5 text-xs font-medium text-primary-foreground"
            >
              Activar
            </button>
          </div>
          <select
            aria-label="Seleccionar modelo de IA"
            value={selected.id}
            onChange={(e) => onSelect(e.target.value as ModelId)}
            className="w-full rounded-xl bg-secondary px-3 py-2 text-xs outline-none ring-ring focus:ring-2"
          >
            {AI_MODELS.filter((m) => !p.localOnly || !m.cloud).map((m) => (
              <option key={m.id} value={m.id}>
                {m.tier}: {m.name} {m.cloud ? "" : `(~${hasGPU ? m.gpuMB : m.cpuMB} MB)`}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-muted-foreground">{selected.desc}</p>
        </div>
      )}

      {s.status === "loading" && (
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
            <span className="truncate">
              {s.phase === "initializing"
                ? `Inicializando ${getModelOption(s.modelId).name}…`
                : `Descargando ${getModelOption(s.modelId).name}…`}
            </span>
            <span className="shrink-0 tabular-nums">
              {s.totalMB > 0 ? `${s.loadedMB}/${s.totalMB} MB · ` : ""}
              {Math.round(s.progress * 100)}%
            </span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-all"
              style={{ width: `${Math.round(s.progress * 100)}%` }}
            />
          </div>
          <div className="flex items-center justify-between gap-2 pt-0.5">
            <select
              aria-label="Cambiar a otro modelo"
              value={selected.id}
              onChange={(e) => {
                const next = e.target.value as ModelId;
                onSelect(next);
                loadModel(next).catch(() => {});
              }}
              className="min-w-0 flex-1 rounded-lg bg-secondary px-2.5 py-1 text-[11px] outline-none"
            >
              {AI_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  Cambiar: {m.tier} ({m.name}
                  {m.cloud ? "" : ` · ~${hasGPU ? m.gpuMB : m.cpuMB} MB`})
                </option>
              ))}
            </select>
            <button
              onClick={() => resetModel()}
              className="whitespace-nowrap shrink-0 rounded-full bg-secondary px-2.5 py-1 text-[11px] text-muted-foreground"
            >
              Cancelar
            </button>
          </div>
        </div>
      )}

      {s.status === "error" && (
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-destructive">
              No se pudo cargar {getModelOption(s.modelId).name}. Prueba Gemini (Nube) o uno más
              liviano.
            </span>
            <button
              onClick={() => loadModel(selected.id).catch(() => {})}
              className="whitespace-nowrap shrink-0 rounded-full bg-secondary px-3 py-1.5 text-xs"
            >
              Reintentar
            </button>
          </div>
          <select
            aria-label="Elegir otro modelo de IA"
            value={selected.id}
            onChange={(e) => {
              const next = e.target.value as ModelId;
              onSelect(next);
              loadModel(next).catch(() => {});
            }}
            className="w-full rounded-xl bg-secondary px-3 py-2 text-xs outline-none"
          >
            {AI_MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.tier}: {m.name} {m.cloud ? "" : `(~${hasGPU ? m.gpuMB : m.cpuMB} MB)`}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
