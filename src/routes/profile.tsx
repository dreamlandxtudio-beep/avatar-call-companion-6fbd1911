import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowLeft,
  AudioLines,
  ImagePlus,
  RefreshCw,
  ShieldCheck,
  Trash2,
  UserPlus,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  applyAvatarPhoto,
  clearMessages,
  createCustomCharacter,
  deleteCustomCharacter,
  resizeImage,
  selectCharacter,
  setLocalOnlyMode,
  setProfile,
  useCustomCharacters,
  useProfile,
  type ModelId,
} from "@/lib/store";
import {
  AI_MODELS,
  detectWebGPU,
  getModelOption,
  loadModel,
  resetModel,
  useAIState,
} from "@/lib/ai";
import {
  GEMINI_VOICES,
  KOKORO_VOICES,
  loadKokoro,
  speak,
  systemVoices,
  unlockAudio,
} from "@/lib/voice";
import { AppSidebar } from "@/components/AppSidebar";
import { Button } from "@/components/ui/button";
import { CHARACTERS } from "@/lib/characters";
import { AvatarImg } from "@/components/AvatarImg";

export const Route = createFileRoute("/profile")({
  head: () => ({
    meta: [
      { title: "Personalizar — Yume" },
      {
        name: "description",
        content: "Crea avatares permanentes, elige voz y activa el modo 100% local en Yume.",
      },
      { property: "og:title", content: "Personalizar — Yume" },
      {
        property: "og:description",
        content: "Crea avatares permanentes, elige voz y activa el modo 100% local en Yume.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: ProfilePage,
});

const PRESETS = [
  {
    label: "Entrevistador",
    personality:
      "Profesional, perceptivo y sereno. Evalúa respuestas con rigor, hace preguntas relevantes y transmite confianza.",
    style: "Ejecutivo natural, articulado y directo, como en una videollamada real de trabajo",
  },
  {
    label: "Cercana",
    personality: "Cálida, dulce y atenta, siempre escucha con empatía",
    style: "Natural y cercana, usa expresiones afectuosas sin exagerar",
  },
  {
    label: "Ingenioso",
    personality: "Bromista, rápido de mente y con chispa creativa",
    style: "Ágil, responde con humor sutil y preguntas curiosas",
  },
  {
    label: "Sereno",
    personality: "Tranquilo, observador y buen consejero",
    style: "Habla con calma y responde de forma directa",
  },
];

const field =
  "w-full rounded-xl border border-border bg-card px-4 py-3 outline-none ring-ring focus:ring-2";

function ProfilePage() {
  const p = useProfile();
  const customChars = useCustomCharacters();
  const ai = useAIState();
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [kProg, setKProg] = useState<number | null>(null);
  const [hasGPU, setHasGPU] = useState(true);
  const [confirmClear, setConfirmClear] = useState(false);
  const [testingUrl, setTestingUrl] = useState(false);
  const [urlStatus, setUrlStatus] = useState<"idle" | "online" | "offline">("idle");
  const [urlMessage, setUrlMessage] = useState("");

  useEffect(() => {
    detectWebGPU().then(setHasGPU);
  }, []);

  useEffect(() => {
    const up = () => setVoices(systemVoices(p.language));
    up();
    if (typeof window !== "undefined" && "speechSynthesis" in window && window.speechSynthesis) {
      window.speechSynthesis.addEventListener?.("voiceschanged", up);
      return () => window.speechSynthesis.removeEventListener?.("voiceschanged", up);
    }
    return undefined;
  }, [p.language]);

  const checkLiveAvatarUrl = async () => {
    const target = (p.liveAvatarUrl ?? "http://localhost:7860").trim();
    setTestingUrl(true);
    setUrlStatus("idle");
    setUrlMessage("");
    try {
      const res = await fetch(`/api/liveavatar/status?url=${encodeURIComponent(target)}`);
      const data = (await res.json()) as { ok?: boolean; error?: string };
      if (data.ok) {
        setUrlStatus("online");
        setUrlMessage("Servidor externo accesible en tu equipo.");
      } else {
        setUrlStatus("offline");
        setUrlMessage(data.error ? `Sin respuesta: ${data.error}` : "Servidor no responde");
      }
    } catch (err) {
      setUrlStatus("offline");
      setUrlMessage(err instanceof Error ? err.message : "Error de conexión");
    } finally {
      setTestingUrl(false);
    }
  };

  const pickVoice = async (v: string) => {
    const [engine, id] = v.split("|") as ["gemini" | "kokoro" | "system", string];
    setProfile({
      voice: { engine, id },
      ...(engine === "gemini" ? { localOnly: false } : {}),
    });
    if (engine === "kokoro") {
      setKProg(0);
      try {
        await loadKokoro((x) => setKProg(x));
      } finally {
        setKProg(null);
      }
    }
  };

  const testVoice = () => {
    unlockAudio();
    void speak(
      p.language === "es-ES"
        ? `Hola, soy ${p.name}. ¿Qué tal suena mi voz?`
        : `Hi, I'm ${p.name}. How does my voice sound?`,
      p.voice,
      p.language,
      undefined,
      { localOnly: p.localOnly },
    );
  };

  const selectedModel = getModelOption(
    p.localOnly && getModelOption(p.modelId).cloud ? "qwen25-0.5b" : p.modelId,
  );
  const isSelectedReady =
    !!selectedModel.cloud || (ai.status === "ready" && ai.modelId === selectedModel.id);
  const isSelectedLoading = ai.status === "loading" && ai.modelId === selectedModel.id;

  const pickModel = (id: ModelId) => {
    const opt = getModelOption(id);
    setProfile({
      modelId: id,
      ...(opt.cloud ? { localOnly: false } : {}),
    });
    if (ai.status === "loading" || ai.status === "error") {
      resetModel();
    }
  };

  const gender = p.gender ?? "female";
  const allChars = [...customChars, ...CHARACTERS];
  const isCustomActive = customChars.some((c) => c.id === p.characterId);

  return (
    <div className="app-layout">
      <AppSidebar current="profile" />
      <main className="profile-content">
        <header className="flex items-center justify-between gap-3 border-b border-border pb-3">
          <div className="flex items-center gap-3">
            <Link
              to="/"
              aria-label="Volver"
              className="flex min-h-[40px] min-w-[40px] items-center justify-center rounded-full hover:bg-secondary md:hidden"
            >
              <ArrowLeft className="h-5 w-5" />
            </Link>
            <h1 className="text-lg font-bold">Personalizar y crear</h1>
          </div>
          <Button
            type="button"
            size="sm"
            onClick={() => createCustomCharacter()}
            className="whitespace-nowrap shrink-0"
          >
            <UserPlus className="h-4 w-4" />
            <span>Nuevo personaje</span>
          </Button>
        </header>

        <div className="mt-5 flex flex-wrap gap-2">
          {allChars.map((character) => (
            <div key={character.id} className="inline-flex items-center">
              <Button
                variant={p.characterId === character.id ? "default" : "outline"}
                className="whitespace-nowrap shrink-0"
                onClick={() => selectCharacter(character.id)}
              >
                <AvatarImg src={character.avatar} name={character.name} className="h-5 w-5" />
                {character.name}
              </Button>
            </div>
          ))}
        </div>

        <div className="mt-6 flex flex-col items-center">
          <label className="group relative block w-fit cursor-pointer">
            <AvatarImg
              src={p.avatar}
              name={p.name}
              className="h-32 w-32 text-4xl shadow-lg ring-2 ring-border"
            />
            <span className="absolute bottom-0 right-0 rounded-full bg-primary p-2.5 text-primary-foreground shadow-md transition-transform group-hover:scale-105">
              <ImagePlus className="h-4 w-4" />
            </span>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (f) {
                  const dataUrl = await resizeImage(f);
                  await applyAvatarPhoto(dataUrl, !isCustomActive);
                }
                e.target.value = "";
              }}
            />
          </label>
          <p className="mt-2.5 text-xs text-muted-foreground">
            {isCustomActive
              ? "Toca la foto para cambiar la imagen de este personaje"
              : "Toca la foto para crear un nuevo personaje guardado con tu imagen"}
          </p>
          {isCustomActive && p.characterId && (
            <Button
              variant="ghost"
              size="sm"
              type="button"
              onClick={() => deleteCustomCharacter(p.characterId!)}
              className="mt-2 text-xs text-destructive hover:bg-destructive/10"
            >
              <Trash2 className="h-3.5 w-3.5" />
              Eliminar este personaje guardado
            </Button>
          )}
        </div>

        <div className="mt-6 rounded-xl border border-border bg-card p-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-start gap-3">
              <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
              <div>
                <p className="text-sm font-semibold text-foreground">
                  Modo 100% local (cero consumo de datos)
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Expresiones faciales, IA conversacional, voz y micrófono operan solo en tu
                  navegador sin enviar audio ni mensajes a la nube.
                </p>
              </div>
            </div>
            <Button
              type="button"
              variant={p.localOnly ? "default" : "secondary"}
              onClick={() => {
                const next = !p.localOnly;
                setLocalOnlyMode(next);
                resetModel();
              }}
              className="shrink-0"
            >
              {p.localOnly ? "Activado" : "Activar"}
            </Button>
          </div>
        </div>

        <div className="mt-6 space-y-5">
          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">Nombre</label>
            <input
              className={field}
              value={p.name}
              onChange={(e) => setProfile({ name: e.target.value })}
            />
          </div>

          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">Género y voz</label>
            <div className="mb-2.5 grid grid-cols-2 gap-2">
              {(
                [
                  ["female", "Mujer"],
                  ["male", "Hombre"],
                ] as const
              ).map(([g, label]) => (
                <Button
                  variant="ghost"
                  key={g}
                  type="button"
                  onClick={() => setProfile({ gender: g })}
                  className={`rounded-xl py-2.5 text-sm font-medium transition-colors ${
                    gender === g ? "bg-primary text-primary-foreground" : "bg-card text-foreground"
                  }`}
                >
                  {label}
                </Button>
              ))}
            </div>
            <div className="flex gap-2">
              <select
                className={field}
                value={`${p.voice.engine}|${p.voice.id}`}
                onChange={(e) => void pickVoice(e.target.value)}
              >
                {!p.localOnly && (
                  <optgroup label="Voces naturales · Gemini">
                    {GEMINI_VOICES.map((v) => (
                      <option key={v.id} value={`gemini|${v.id}`}>
                        {v.label}
                      </option>
                    ))}
                  </optgroup>
                )}
                <optgroup label="Voces locales · Kokoro (EN)">
                  {KOKORO_VOICES.map((v) => (
                    <option key={v.id} value={`kokoro|${v.id}`}>
                      {v.label}
                    </option>
                  ))}
                </optgroup>
                <optgroup label="Voces locales del dispositivo">
                  <option value="system|">Automática (sin datos)</option>
                  {voices.map((v) => (
                    <option key={v.voiceURI} value={`system|${v.voiceURI}`}>
                      {v.name}
                    </option>
                  ))}
                </optgroup>
              </select>
              <Button
                variant="ghost"
                type="button"
                onClick={testVoice}
                className="rounded-xl bg-secondary px-4"
                aria-label="Probar voz"
              >
                <AudioLines className="h-4 w-4" />
              </Button>
            </div>
            {kProg !== null && (
              <p className="mt-1.5 text-xs text-primary">
                Descargando voz… {Math.round(kProg * 100)}%
              </p>
            )}
          </div>

          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">Personalidad</label>
            <div className="mb-2 flex flex-wrap gap-2">
              {PRESETS.map((x) => (
                <Button
                  variant="ghost"
                  key={x.label}
                  type="button"
                  onClick={() => setProfile({ personality: x.personality, style: x.style })}
                  className={`rounded-full px-3 py-1 text-xs ${
                    p.personality === x.personality
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary"
                  }`}
                >
                  {x.label}
                </Button>
              ))}
            </div>
            <textarea
              rows={2}
              className={field}
              value={p.personality}
              onChange={(e) => setProfile({ personality: e.target.value })}
            />
          </div>

          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">
              Historia / Contexto
            </label>
            <textarea
              rows={2}
              className={field}
              value={p.backstory}
              onChange={(e) => setProfile({ backstory: e.target.value })}
            />
          </div>

          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">Forma de hablar</label>
            <input
              className={field}
              value={p.style}
              onChange={(e) => setProfile({ style: e.target.value })}
            />
          </div>

          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">Idioma</label>
            <div className="grid grid-cols-2 gap-2">
              {(
                [
                  ["es-ES", "Español"],
                  ["en-US", "English"],
                ] as const
              ).map(([v, l]) => (
                <Button
                  variant="ghost"
                  key={v}
                  type="button"
                  onClick={() => setProfile({ language: v })}
                  className={`rounded-xl py-2.5 text-sm font-medium ${
                    p.language === v ? "bg-primary text-primary-foreground" : "bg-card"
                  }`}
                >
                  {l}
                </Button>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-sm text-muted-foreground">Modelo de IA</label>
            <div className="flex gap-2">
              <select
                className={field}
                value={selectedModel.id}
                onChange={(e) => pickModel(e.target.value as ModelId)}
              >
                {AI_MODELS.filter((m) => !p.localOnly || !m.cloud).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.tier} · {m.name} {m.cloud ? "" : `(~${hasGPU ? m.gpuMB : m.cpuMB} MB)`}
                  </option>
                ))}
              </select>
              {!selectedModel.cloud && (
                <Button
                  variant="ghost"
                  type="button"
                  onClick={() =>
                    isSelectedLoading ? resetModel() : loadModel(selectedModel.id).catch(() => {})
                  }
                  disabled={isSelectedReady}
                  className="shrink-0 rounded-xl bg-primary px-4 text-sm font-medium text-primary-foreground disabled:opacity-50"
                >
                  {isSelectedReady ? "Listo" : isSelectedLoading ? "Cancelar" : "Cargar"}
                </Button>
              )}
            </div>
          </div>

          <details className="rounded-xl border border-border bg-card/60 p-3.5 text-xs">
            <summary className="cursor-pointer font-medium text-muted-foreground">
              Motor de animación facial
            </summary>
            <div className="mt-3 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant="ghost"
                  type="button"
                  onClick={() => setProfile({ avatarEngine: "mediapipe-mesh" })}
                  className={`rounded-lg border px-3 py-2 text-left ${
                    (p.avatarEngine ?? "mediapipe-mesh") === "mediapipe-mesh"
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border text-muted-foreground"
                  }`}
                >
                  Motor local WebGL2
                </Button>
                <Button
                  variant="ghost"
                  type="button"
                  onClick={() => setProfile({ avatarEngine: "liveavatar" })}
                  className={`rounded-lg border px-3 py-2 text-left ${
                    p.avatarEngine === "liveavatar"
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border text-muted-foreground"
                  }`}
                >
                  Estudio externo
                </Button>
              </div>
              {p.avatarEngine === "liveavatar" && (
                <div className="space-y-2">
                  <p className="text-muted-foreground">
                    LiveAvatar o FasterLivePortrait requieren un servidor externo con GPU dedicada.
                  </p>
                  <div className="flex gap-2">
                    <input
                      type="url"
                      className="flex-1 rounded-lg border border-input bg-background px-3 py-1.5 text-xs outline-none ring-ring focus:ring-1"
                      value={p.liveAvatarUrl ?? "http://localhost:7860"}
                      placeholder="http://localhost:7860"
                      onChange={(e) => setProfile({ liveAvatarUrl: e.target.value })}
                    />
                    <Button
                      variant="ghost"
                      type="button"
                      onClick={() => void checkLiveAvatarUrl()}
                      disabled={testingUrl}
                      className="flex items-center gap-1 rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
                    >
                      <RefreshCw className={`h-3 w-3 ${testingUrl ? "animate-spin" : ""}`} />
                      Probar
                    </Button>
                  </div>
                  {urlStatus !== "idle" && (
                    <p className={urlStatus === "online" ? "text-primary" : "text-destructive"}>
                      {urlMessage}
                    </p>
                  )}
                </div>
              )}
            </div>
          </details>

          {confirmClear ? (
            <div className="flex gap-2">
              <Button
                variant="ghost"
                type="button"
                onClick={() => {
                  clearMessages();
                  setConfirmClear(false);
                }}
                className="flex-1 rounded-xl bg-destructive py-3 text-sm font-semibold text-destructive-foreground"
              >
                Sí, borrar conversación
              </Button>
              <Button
                variant="ghost"
                type="button"
                onClick={() => setConfirmClear(false)}
                className="rounded-xl border border-input bg-background px-4 py-3 text-sm"
              >
                Cancelar
              </Button>
            </div>
          ) : (
            <Button
              variant="ghost"
              type="button"
              onClick={() => setConfirmClear(true)}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-destructive/40 py-3 text-sm font-medium text-destructive hover:bg-destructive/10"
            >
              <Trash2 className="h-4 w-4" />
              Borrar conversación
            </Button>
          )}
        </div>
      </main>
    </div>
  );
}
