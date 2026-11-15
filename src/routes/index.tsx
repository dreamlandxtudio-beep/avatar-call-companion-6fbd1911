import { createFileRoute, Link } from "@tanstack/react-router";
import {
  AudioLines,
  Check,
  ImagePlus,
  MessageSquareText,
  ScanFace,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Video,
} from "lucide-react";
import { useRef, useState } from "react";
import {
  applyAvatarPhoto,
  deleteCustomCharacter,
  findCharacterById,
  resizeImage,
  selectCharacter,
  useCustomCharacters,
  useMessages,
  useProfile,
} from "@/lib/store";
import { CHARACTERS } from "@/lib/characters";

import { AvatarImg } from "@/components/AvatarImg";
import { AppSidebar } from "@/components/AppSidebar";
import { BrandLogo } from "@/components/BrandLogo";
import { Button } from "@/components/ui/button";
import { speak, unlockAudio } from "@/lib/voice";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "YUME — personajes de IA, anime y videollamadas" },
      {
        name: "description",
        content:
          "Chatea y haz videollamadas en vivo con personajes de IA, anime, rol y retratos realistas.",
      },
      { property: "og:title", content: "YUME — personajes de IA, anime y videollamadas" },
      {
        property: "og:description",
        content:
          "Conversaciones divertidas, rol, anime y videollamadas con personajes de IA en tiempo real.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Home,
});

type FilterMode = "all" | "anime" | "realistas" | "rol" | "creativa" | "custom";

function Home() {
  const p = useProfile();
  const customChars = useCustomCharacters();
  const messages = useMessages();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<FilterMode>("all");
  const [uploadError, setUploadError] = useState("");
  const [uploading, setUploading] = useState(false);
  const upload = useRef<HTMLInputElement>(null);

  const allCharacters = [...customChars, ...CHARACTERS];
  const activeCharacter = findCharacterById(p.characterId) ?? allCharacters[0];

  const characters = allCharacters.filter((character) => {
    const matchesQuery =
      `${character.name} ${character.roleLabel} ${character.personality} ${character.tags.join(" ")}`
        .toLowerCase()
        .includes(query.toLowerCase());
    if (!matchesQuery) return false;
    if (filter === "anime") return character.tags.includes("Anime");
    if (filter === "realistas") return character.tags.includes("Realistas");
    if (filter === "rol")
      return character.tags.includes("Rol") || character.tags.includes("Misterio");
    if (filter === "creativa") return character.tags.includes("Creativa");
    if (filter === "custom") return !!character.isCustom;
    return true;
  });

  const filterOptions: { id: FilterMode; label: string }[] = [
    { id: "all", label: "Para ti" },
    { id: "anime", label: "Anime" },
    { id: "realistas", label: "Realistas" },
    { id: "rol", label: "Rol y aventura" },
    { id: "creativa", label: "Creativa" },
    ...(customChars.length > 0 ? [{ id: "custom" as const, label: "Mis personajes" }] : []),
  ];

  const previewVoice = (
    name: string,
    voice: { engine: "gemini" | "kokoro" | "system"; id: string },
    gender: "female" | "male",
  ) => {
    unlockAudio();
    void speak(`Hola, soy ${name}. ¿De qué quieres hablar hoy?`, voice, p.language, undefined, {
      localOnly: p.localOnly,
      gender,
    }).catch((error) =>
      setUploadError(error instanceof Error ? error.message : "No se pudo reproducir la voz."),
    );
  };

  return (
    <div className="app-layout">
      <AppSidebar current="discover" />

      <main className="discovery-content">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-4">
          <div className="flex min-w-0 flex-1 items-center gap-2.5 sm:flex-initial">
            <Link to="/" className="shrink-0 md:hidden">
              <BrandLogo />
            </Link>
            <label className="flex h-11 min-w-0 flex-1 items-center gap-2.5 rounded-xl border border-border bg-card px-3.5 sm:w-80 sm:flex-none">
              <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
              <input
                aria-label="Buscar personajes"
                placeholder="Buscar personajes, anime, rol…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="min-w-0 w-full bg-transparent text-sm font-semibold text-foreground outline-none placeholder:font-semibold placeholder:text-muted-foreground"
              />
            </label>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <Button
              asChild
              variant="secondary"
              size="sm"
              className="hidden h-10 whitespace-nowrap shrink-0 rounded-xl px-3.5 font-extrabold sm:inline-flex"
            >
              <Link to="/mirror">
                <ScanFace className="h-4 w-4" />
                <span>Espejo</span>
              </Link>
            </Button>
            <Button
              asChild
              size="sm"
              className="h-10 whitespace-nowrap shrink-0 rounded-xl px-3.5 font-extrabold sm:px-4"
            >
              <Link to="/call">
                <Video className="h-4 w-4" />
                <span>Videollamada</span>
              </Link>
            </Button>
            <Button
              asChild
              variant="secondary"
              size="icon"
              className="h-10 w-10 shrink-0 rounded-xl"
              aria-label="Personalizar"
            >
              <Link to="/profile">
                <SlidersHorizontal className="h-4 w-4" />
              </Link>
            </Button>
          </div>
        </header>

        <section
          className="mt-5 grid items-stretch gap-5 rounded-2xl border border-border bg-card/75 p-4 sm:mt-6 sm:gap-6 sm:p-7 lg:grid-cols-[1.25fr_0.75fr]"
          aria-label="Personaje destacado"
        >
          <div className="flex flex-col justify-between gap-5 sm:gap-6">
            <div>
              <div className="inline-flex items-center gap-1.5 text-xs font-extrabold uppercase tracking-wider text-primary">
                <Sparkles className="h-3.5 w-3.5" />
                <span>Personajes de IA en vivo</span>
              </div>
              <h1 className="mt-2 text-2xl font-black tracking-tight text-foreground sm:mt-2.5 sm:text-4xl">
                Chatea, rolea y llama a {activeCharacter ? activeCharacter.name : p.name}
              </h1>
              <p className="mt-2 text-sm font-bold text-muted-foreground">
                {activeCharacter?.roleLabel ?? "Tu personaje personalizado"}
                {" · "}
                {p.localOnly ? "Modo local privado" : "Voz y rostro en tiempo real"}
              </p>
              <p className="mt-3 max-w-xl text-sm font-semibold leading-relaxed text-foreground/90 sm:text-base">
                {activeCharacter?.tagline || p.personality}
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2.5">
                <Button
                  asChild
                  className="h-11 whitespace-nowrap shrink-0 rounded-xl px-5 text-sm font-extrabold"
                >
                  <Link to="/chat">
                    <MessageSquareText className="h-4 w-4" />
                    <span>Chatear ahora</span>
                  </Link>
                </Button>
                <Button
                  asChild
                  variant="secondary"
                  className="h-11 whitespace-nowrap shrink-0 rounded-xl px-5 text-sm font-extrabold"
                >
                  <Link to="/call">
                    <Video className="h-4 w-4" />
                    <span>Videollamada</span>
                  </Link>
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => upload.current?.click()}
                  disabled={uploading}
                  className="h-11 whitespace-nowrap shrink-0 rounded-xl px-4 text-sm font-extrabold"
                >
                  <ImagePlus className="h-4 w-4" />
                  <span>{uploading ? "Procesando…" : "Crear con una foto"}</span>
                </Button>
              </div>
              {uploadError && (
                <p role="alert" className="text-xs font-bold text-destructive">
                  {uploadError}
                </p>
              )}
            </div>
          </div>

          <div className="relative min-h-[240px] overflow-hidden rounded-2xl border border-border bg-background sm:min-h-[290px]">
            <img
              src={p.avatar}
              alt={p.name}
              onError={(e) => {
                const imgEl = e.currentTarget;
                if (imgEl.dataset.fallbackApplied) return;
                imgEl.dataset.fallbackApplied = "1";
                imgEl.src = "/avatars/default.jpg";
              }}
              className="h-full max-h-[320px] w-full object-cover object-[50%_22%]"
            />
            <div
              className="pointer-events-none absolute inset-0 bg-gradient-to-t from-background/90 via-background/15 to-transparent"
              aria-hidden="true"
            />
            <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-3 p-4">
              <div className="min-w-0">
                <p className="truncate text-base font-black text-foreground">{p.name}</p>
                <p className="truncate text-xs font-bold text-muted-foreground">
                  {activeCharacter?.roleLabel ?? "Listo para hablar"}
                </p>
              </div>
              <button
                type="button"
                onClick={() => previewVoice(p.name, p.voice, p.gender ?? "female")}
                aria-label={`Escuchar a ${p.name}`}
                className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-border/80 bg-background/85 text-foreground backdrop-blur-md transition-colors hover:bg-secondary"
              >
                <AudioLines className="h-4 w-4" />
              </button>
            </div>
          </div>
        </section>

        <input
          ref={upload}
          type="file"
          accept="image/*"
          aria-label="Subir retrato"
          className="hidden"
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setUploading(true);
            setUploadError("");
            try {
              await applyAvatarPhoto(await resizeImage(file), true);
            } catch {
              setUploadError("No se pudo abrir esta imagen. Prueba otro archivo.");
            } finally {
              setUploading(false);
              e.target.value = "";
            }
          }}
        />

        <section className="mt-8" aria-label="Explorar personajes">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-baseline gap-3">
              <h2 className="text-2xl font-black text-foreground">Explorar personajes</h2>
              <span className="font-mono text-xs font-bold text-muted-foreground tabular-nums">
                {characters.length}
              </span>
            </div>
          </div>

          <div
            className="filter-tabs-scroll mt-4 sm:flex-wrap"
            role="tablist"
            aria-label="Categorías de personajes"
          >
            {filterOptions.map((opt) => {
              const active = filter === opt.id;
              return (
                <button
                  key={opt.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setFilter(opt.id)}
                  className={`min-h-[40px] whitespace-nowrap shrink-0 rounded-xl px-4 py-2 text-xs font-extrabold transition-all ${
                    active
                      ? "bg-primary text-primary-foreground btn-metallic"
                      : "border border-border bg-card text-foreground/85 btn-metallic-subtle hover:bg-secondary hover:text-foreground"
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>

          <div className="character-grid mt-6">
            {characters.map((character) => {
              const isSelected = p.characterId === character.id;
              return (
                <article
                  key={character.id}
                  className={`group flex min-w-0 flex-col overflow-hidden rounded-2xl border bg-card transition-colors ${
                    isSelected
                      ? "border-primary ring-1 ring-primary/40"
                      : "border-border hover:border-primary/45"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => selectCharacter(character.id)}
                    aria-label={`Seleccionar a ${character.name}`}
                    className="relative block w-full overflow-hidden text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <img
                      src={character.avatar}
                      alt={character.name}
                      onError={(e) => {
                        const imgEl = e.currentTarget;
                        if (imgEl.dataset.fallbackApplied) return;
                        imgEl.dataset.fallbackApplied = "1";
                        imgEl.src = "/avatars/default.jpg";
                      }}
                      className="character-cover transition-transform duration-300 group-hover:scale-[1.03]"
                    />
                    <div
                      className="pointer-events-none absolute inset-x-0 bottom-0 h-20 bg-gradient-to-t from-card via-card/30 to-transparent"
                      aria-hidden="true"
                    />
                    {isSelected && (
                      <span className="absolute right-2.5 top-2.5 flex h-7 w-7 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
                        <Check className="h-4 w-4 stroke-[2.5]" />
                      </span>
                    )}
                  </button>

                  <div className="flex min-w-0 flex-1 flex-col justify-between p-3.5">
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <button
                          type="button"
                          onClick={() => selectCharacter(character.id)}
                          className="min-w-0 flex-1 text-left focus-visible:outline-none"
                        >
                          <h3 className="truncate text-base font-black text-foreground transition-colors hover:text-primary">
                            {character.name}
                          </h3>
                          <p className="truncate text-xs font-bold text-muted-foreground">
                            {character.roleLabel}
                          </p>
                        </button>

                        <div className="flex shrink-0 items-center gap-0.5">
                          <button
                            type="button"
                            className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                            title={`Escuchar a ${character.name}`}
                            aria-label={`Escuchar a ${character.name}`}
                            onClick={() =>
                              previewVoice(character.name, character.voice, character.gender)
                            }
                          >
                            <AudioLines className="h-4 w-4" />
                          </button>
                          {character.isCustom && (
                            <button
                              type="button"
                              className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-destructive/15 hover:text-destructive"
                              title={`Eliminar a ${character.name}`}
                              aria-label={`Eliminar a ${character.name}`}
                              onClick={() => deleteCustomCharacter(character.id)}
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          )}
                        </div>
                      </div>

                      <p className="mt-2 line-clamp-2 text-xs font-semibold leading-relaxed text-muted-foreground">
                        {character.tagline}
                      </p>
                    </div>

                    <div className="mt-3.5 grid grid-cols-2 gap-2">
                      <Button
                        asChild
                        size="sm"
                        className="h-9 whitespace-nowrap shrink-0 rounded-xl px-2.5 text-xs font-extrabold"
                      >
                        <Link to="/call" onClick={() => selectCharacter(character.id)}>
                          <Video className="h-3.5 w-3.5" />
                          <span>Llamar</span>
                        </Link>
                      </Button>
                      <Button
                        asChild
                        variant="secondary"
                        size="sm"
                        className="h-9 whitespace-nowrap shrink-0 rounded-xl px-2.5 text-xs font-extrabold"
                      >
                        <Link to="/chat" onClick={() => selectCharacter(character.id)}>
                          <MessageSquareText className="h-3.5 w-3.5" />
                          <span>Chatear</span>
                        </Link>
                      </Button>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>

          {characters.length === 0 && (
            <p className="py-12 text-center text-sm font-bold text-muted-foreground">
              No hay personajes que coincidan con tu búsqueda.
            </p>
          )}
        </section>

        <section className="mt-10">
          <h2 className="mb-3 font-mono text-xs font-bold uppercase tracking-widest text-muted-foreground">
            Continuar conversación
          </h2>
          <Link
            to="/chat"
            className="flex items-center gap-4 rounded-2xl border border-border bg-card p-4 transition-colors hover:border-primary/45 hover:bg-secondary/60"
          >
            <AvatarImg src={p.avatar} name={p.name} className="h-12 w-12 shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-black text-foreground">{p.name}</p>
              <p className="mt-0.5 truncate text-xs font-semibold text-muted-foreground">
                {messages.at(-1)?.content ?? "¿De qué quieres hablar hoy?"}
              </p>
            </div>
            <MessageSquareText className="h-4 w-4 shrink-0 text-primary" />
          </Link>
        </section>
      </main>
    </div>
  );
}
