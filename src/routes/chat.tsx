import { createFileRoute, Link } from "@tanstack/react-router";
import {
  ArrowLeft,
  AudioLines,
  ImagePlus,
  SendHorizontal,
  SlidersHorizontal,
  SmilePlus,
  Video,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  addMessage,
  findCharacterById,
  getMessages,
  getProfile,
  resizeImage,
  setProfile,
  useMessages,
  useProfile,
  type ModelId,
} from "@/lib/store";
import { AI_MODELS, getModelOption, reply, resetModel } from "@/lib/ai";
import { getReactionGifs, readGifFileAsDataUrl, resolveCharacterGif } from "@/lib/gifs";
import { speak, unlockAudio } from "@/lib/voice";
import { AvatarImg } from "@/components/AvatarImg";
import { AppSidebar } from "@/components/AppSidebar";
import { Button } from "@/components/ui/button";
import { ModelBanner } from "@/components/ModelBanner";

export const Route = createFileRoute("/chat")({
  head: () => ({
    meta: [
      { title: "Conversación — Yume" },
      { name: "description", content: "Chatea y prepara entrevistas con tu avatar en Yume." },
      { property: "og:title", content: "Conversación — Yume" },
      {
        property: "og:description",
        content: "Chatea y prepara entrevistas con tu avatar en Yume.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Chat,
});

function Chat() {
  const p = useProfile();
  const msgs = useMessages();
  const [text, setText] = useState("");
  const [typing, setTyping] = useState<string | null>(null);
  const [gifOpen, setGifOpen] = useState(false);
  const [gifUrlInput, setGifUrlInput] = useState("");
  const [pendingGif, setPendingGif] = useState<{ url: string; alt: string } | null>(null);
  const end = useRef<HTMLDivElement>(null);

  const reactionGifs = useMemo(() => (gifOpen ? getReactionGifs() : []), [gifOpen]);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs.length, typing]);

  const sendMessage = async (raw: string, gif?: { url: string; alt: string } | null) => {
    const t = raw.trim();
    const attached = gif ?? pendingGif;
    if ((!t && !attached) || typing !== null) return;
    setText("");
    setPendingGif(null);
    setGifOpen(false);

    addMessage({
      role: "user",
      content: t,
      ...(attached ? { gifUrl: attached.url, gifAlt: attached.alt } : {}),
    });
    setTyping("");
    try {
      const res = await reply(getProfile(), getMessages(), (s) => setTyping(s));
      const matchedGif = resolveCharacterGif(res.gifTag);
      addMessage({
        role: "assistant",
        content: res.text,
        ...(matchedGif ? { gifUrl: matchedGif.url, gifAlt: matchedGif.alt } : {}),
      });
    } catch (e) {
      const detail = e instanceof Error ? e.message : "";
      addMessage({
        role: "assistant",
        content: detail
          ? `(Error al responder: ${detail})`
          : "(No pude responder: prueba elegir otro modelo arriba o en Personalizar.)",
      });
    } finally {
      setTyping(null);
    }
  };

  const activeModel = getModelOption(
    p.localOnly && getModelOption(p.modelId).cloud ? "qwen25-0.5b" : p.modelId,
  );
  const charMeta = findCharacterById(p.characterId);
  const starters = charMeta?.starters ?? [
    `Hola ${p.name}, ¿podemos empezar una entrevista práctica?`,
    "Cuéntame sobre ti.",
    "Conversemos sobre un proyecto reciente.",
  ];

  return (
    <div className="app-layout">
      <AppSidebar current="chat" />
      <main className="conversation-page">
        <header className="flex min-h-[60px] items-center gap-2.5 border-b border-border px-3 py-2">
          <Link
            to="/"
            aria-label="Volver"
            className="flex min-h-[40px] min-w-[40px] items-center justify-center rounded-full text-foreground hover:bg-secondary/60 md:hidden"
          >
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <AvatarImg src={p.avatar} name={p.name} className="h-9 w-9 shrink-0 text-sm" />
          <div className="min-w-0 flex-1 pl-0.5">
            <div className="truncate text-sm font-semibold leading-tight text-foreground">
              {p.name}
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {typing !== null ? (
                <span className="text-primary">escribiendo…</span>
              ) : (
                <select
                  aria-label="Modelo de IA"
                  value={activeModel.id}
                  onChange={(e) => {
                    const id = e.target.value as ModelId;
                    const chosen = getModelOption(id);
                    setProfile({
                      modelId: id,
                      ...(chosen.cloud ? { localOnly: false } : {}),
                    });
                    resetModel();
                  }}
                  className="max-w-full truncate bg-transparent text-xs text-muted-foreground outline-none hover:text-foreground"
                >
                  {AI_MODELS.filter((m) => !p.localOnly || !m.cloud).map((m) => (
                    <option key={m.id} value={m.id} className="bg-card text-foreground">
                      {m.name} ({m.tier})
                    </option>
                  ))}
                </select>
              )}
              {p.localOnly && (
                <span className="shrink-0 text-[10px] font-medium text-primary">· 100% Local</span>
              )}
            </div>
          </div>
          <Button
            asChild
            variant="secondary"
            size="sm"
            className="whitespace-nowrap shrink-0 rounded-full"
          >
            <Link to="/call">
              <Video className="h-4 w-4 text-foreground" />
              <span>Llamar</span>
            </Link>
          </Button>
          <Link
            to="/profile"
            aria-label="Personalizar"
            className="flex min-h-[40px] min-w-[40px] items-center justify-center rounded-full text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
          >
            <SlidersHorizontal className="h-5 w-5" />
          </Link>
        </header>

        <ModelBanner />

        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-5">
          {msgs.length === 0 && (
            <div className="mx-auto flex max-w-md flex-col items-center py-10 text-center">
              <AvatarImg
                src={p.avatar}
                name={p.name}
                className="h-24 w-24 shadow-md ring-1 ring-border"
              />
              <h1 className="mt-4 text-2xl font-semibold text-foreground">{p.name}</h1>
              {charMeta?.roleLabel && (
                <p className="mt-1 text-xs font-medium uppercase tracking-wider text-primary">
                  {charMeta.roleLabel}
                </p>
              )}
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{p.personality}</p>

              <div className="mt-6 flex w-full flex-col gap-2">
                {starters.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => void sendMessage(prompt)}
                    className="rounded-xl border border-border bg-card px-4 py-2.5 text-left text-sm text-foreground transition-colors hover:border-primary/45 hover:bg-secondary/60"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {msgs.map((m, idx) => (
            <div
              key={`${m.at}-${idx}`}
              className={`flex items-end gap-2.5 ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {m.role === "assistant" && (
                <AvatarImg
                  src={p.avatar}
                  name={p.name}
                  className="mb-1 h-8 w-8 shrink-0 text-[10px]"
                />
              )}
              <div
                className={`max-w-[80%] px-4 py-2.5 text-[15px] leading-relaxed ${
                  m.role === "user"
                    ? "rounded-2xl rounded-br-sm bg-primary text-primary-foreground"
                    : "rounded-2xl rounded-bl-sm border border-border bg-card text-foreground"
                }`}
              >
                {m.gifUrl && (
                  <img
                    src={m.gifUrl}
                    alt={m.gifAlt || "GIF"}
                    referrerPolicy="no-referrer"
                    className={`${m.content ? "mb-2" : ""} max-h-44 w-auto rounded-xl border border-border/40 object-contain`}
                  />
                )}
                {m.content && <p className="whitespace-pre-wrap">{m.content}</p>}
                <div
                  className={`mt-1 flex items-center justify-end gap-2 text-[10px] ${
                    m.role === "user" ? "text-primary-foreground/70" : "text-muted-foreground"
                  }`}
                >
                  {m.role === "assistant" && m.content && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6"
                      type="button"
                      aria-label="Escuchar"
                      onClick={() => {
                        unlockAudio();
                        void speak(m.content, p.voice, p.language, undefined, {
                          localOnly: p.localOnly,
                        });
                      }}
                    >
                      <AudioLines className="h-3.5 w-3.5" />
                    </Button>
                  )}
                  <span className="tabular-nums">
                    {new Date(m.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
              </div>
            </div>
          ))}

          {typing !== null && (
            <div className="flex items-end gap-2.5">
              <AvatarImg
                src={p.avatar}
                name={p.name}
                className="mb-1 h-8 w-8 shrink-0 text-[10px]"
              />
              <div className="max-w-[80%] rounded-2xl rounded-bl-sm border border-border bg-card px-4 py-2.5 text-[15px] text-foreground">
                {typing || (
                  <span className="flex gap-1 py-1">
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        className="animate-dot h-2 w-2 rounded-full bg-muted-foreground"
                        style={{ animationDelay: `${i * 0.2}s` }}
                      />
                    ))}
                  </span>
                )}
              </div>
            </div>
          )}
          <div ref={end} />
        </div>

        {gifOpen && (
          <div className="border-t border-border bg-card/95 p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-foreground">Enviar GIF</span>
              <div className="flex items-center gap-2">
                <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-border bg-background px-2.5 py-1 text-xs font-medium text-foreground hover:border-primary/50">
                  <ImagePlus className="h-3.5 w-3.5 text-primary" />
                  <span>Subir GIF</span>
                  <input
                    type="file"
                    accept="image/gif,image/webp,image/*"
                    className="hidden"
                    onChange={async (e) => {
                      const f = e.target.files?.[0];
                      e.target.value = "";
                      if (!f) return;
                      const url =
                        f.type === "image/gif"
                          ? await readGifFileAsDataUrl(f)
                          : await resizeImage(f, 480);
                      if (text.trim()) {
                        setPendingGif({ url, alt: f.name || "GIF enviado" });
                        setGifOpen(false);
                      } else {
                        void sendMessage("", { url, alt: f.name || "GIF enviado" });
                      }
                    }}
                  />
                </label>
                <button
                  type="button"
                  onClick={() => setGifOpen(false)}
                  aria-label="Cerrar GIFs"
                  className="rounded-lg p-1 text-muted-foreground hover:bg-secondary hover:text-foreground"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
              {reactionGifs.map((g) => (
                <button
                  key={g.id}
                  type="button"
                  onClick={() => {
                    if (text.trim()) {
                      setPendingGif({ url: g.url, alt: g.alt });
                      setGifOpen(false);
                    } else {
                      void sendMessage("", { url: g.url, alt: g.alt });
                    }
                  }}
                  className="group flex flex-col items-center gap-1 rounded-xl border border-border bg-background p-1.5 transition-colors hover:border-primary/60"
                >
                  <img src={g.url} alt={g.alt} className="h-12 w-16 rounded-lg object-cover" />
                  <span className="text-[10px] font-medium text-muted-foreground group-hover:text-foreground">
                    {g.label}
                  </span>
                </button>
              ))}
            </div>

            <div className="mt-2.5 flex gap-2">
              <input
                type="url"
                value={gifUrlInput}
                onChange={(e) => setGifUrlInput(e.target.value)}
                placeholder="Pega la URL de un GIF (https://...)"
                className="h-9 flex-1 rounded-lg border border-border bg-background px-3 text-xs text-foreground outline-none ring-ring focus:ring-1"
              />
              <Button
                type="button"
                size="sm"
                variant="secondary"
                disabled={!gifUrlInput.trim()}
                onClick={() => {
                  const u = gifUrlInput.trim();
                  if (!u) return;
                  setGifUrlInput("");
                  if (text.trim()) {
                    setPendingGif({ url: u, alt: "GIF externo" });
                    setGifOpen(false);
                  } else {
                    void sendMessage("", { url: u, alt: "GIF externo" });
                  }
                }}
              >
                Usar URL
              </Button>
            </div>
          </div>
        )}

        {pendingGif && (
          <div className="flex items-center gap-2 border-t border-border bg-card/70 px-3 py-2">
            <img
              src={pendingGif.url}
              alt={pendingGif.alt}
              referrerPolicy="no-referrer"
              className="h-12 w-16 rounded-lg border border-border object-cover"
            />
            <span className="flex-1 truncate text-xs text-muted-foreground">GIF adjunto</span>
            <button
              type="button"
              onClick={() => setPendingGif(null)}
              aria-label="Quitar GIF"
              className="rounded-lg p-1 text-muted-foreground hover:bg-secondary hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            void sendMessage(text);
          }}
          className="flex items-center gap-2 border-t border-border p-3"
        >
          <Button
            type="button"
            variant={gifOpen ? "default" : "ghost"}
            size="icon"
            onClick={() => setGifOpen((v) => !v)}
            aria-label="Abrir GIFs"
            className="h-11 w-11 shrink-0 rounded-full"
          >
            <SmilePlus className="h-5 w-5" />
          </Button>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={`Escribe a ${p.name}…`}
            className="h-11 flex-1 rounded-full border border-border bg-card px-4 text-sm outline-none ring-ring focus:ring-2"
          />
          <Button
            type="submit"
            aria-label="Enviar"
            className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full"
            disabled={(!text.trim() && !pendingGif) || typing !== null}
          >
            <SendHorizontal className="h-5 w-5" />
          </Button>
        </form>
      </main>
    </div>
  );
}
