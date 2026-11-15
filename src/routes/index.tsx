import { createFileRoute, Link } from "@tanstack/react-router";
import {
  AudioLines,
  Check,
  ImagePlus,
  MessageSquareText,
  Search,
  SlidersHorizontal,
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
import { AnimatedAvatar } from "@/components/AnimatedAvatar";
import { AvatarImg } from "@/components/AvatarImg";
import { AppSidebar } from "@/components/AppSidebar";
import { BrandLogo } from "@/components/BrandLogo";
import { Button } from "@/components/ui/button";
import { speak, unlockAudio } from "@/lib/voice";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Yume — videollamadas y entrevistas realistas con IA" },
      {
        name: "description",
        content:
          "Simula entrevistas de trabajo, practica reuniones reales o conversa con personajes y fotos animadas localmente.",
      },
      { property: "og:title", content: "Yume — videollamadas y entrevistas realistas con IA" },
      {
        property: "og:description",
        content:
          "Elige un entrevistador realista, personaje o sube una foto para animarla con fidelidad fotográfica y cero consumo de datos en modo local.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Home,
});

const CATEGORIES = ["Todos", "Entrevistas", "Realistas", "Personalizado", "Anime", "Creativa"];

function Home() {
  const p = useProfile();
  const customChars = useCustomCharacters();
  const messages = useMessages();
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("Todos");
  const [uploadError, setUploadError] = useState("");
  const [uploading, setUploading] = useState(false);
  const upload = useRef<HTMLInputElement>(null);

  const allCharacters = [...customChars, ...CHARACTERS];

  const characters = allCharacters.filter((character) => {
    const matchesQuery = `${character.name} ${character.roleLabel} ${character.personality}`
      .toLowerCase()
      .includes(query.toLowerCase());
    const matchesCategory = category === "Todos" || character.tags.includes(category);
    return matchesQuery && matchesCategory;
  });

  const featured = findCharacterById(p.characterId) ?? allCharacters[0]!;

  return (
    <div className="app-layout">
      <AppSidebar current="discover" />

      <main className="discovery-content">
        <header className="flex items-center justify-between gap-4 border-b border-border pb-4">
          <Link to="/" className="md:hidden">
            <BrandLogo />
          </Link>
          <p className="hidden text-sm text-muted-foreground md:block">
            Estudio facial local · entrevistas y personajes
          </p>
          <div className="flex items-center gap-2">
            <Button asChild size="sm" className="whitespace-nowrap shrink-0">
              <Link to="/call">
                <Video className="h-4 w-4" />
                <span>Videollamada</span>
              </Link>
            </Button>
            <Button asChild variant="secondary" size="icon" aria-label="Personalizar">
              <Link to="/profile">
                <SlidersHorizontal className="h-4 w-4" />
              </Link>
            </Button>
          </div>
        </header>

        <section
          aria-label="Estudio activo"
          className="mt-6 grid items-stretch gap-6 rounded-2xl border border-border bg-card p-5 lg:grid-cols-[1.05fr_0.95fr] lg:p-6"
        >
          <div className="flex flex-col justify-between gap-5">
            <div>
              <div className="flex items-center justify-between gap-3">
                <p className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
                  Perfil en escena · {featured.roleLabel}
                </p>
                <Button
                  variant="ghost"
                  size="icon"
                  title={`Escuchar a ${featured.name}`}
                  aria-label={`Escuchar a ${featured.name}`}
                  onClick={() => {
                    unlockAudio();
                    void speak(
                      `Hola, soy ${featured.name}. ¿Sobre qué te gustaría conversar hoy?`,
                      featured.voice,
                      p.language,
                      undefined,
                      { localOnly: p.localOnly },
                    );
                  }}
                >
                  <AudioLines className="h-4 w-4 text-muted-foreground" />
                </Button>
              </div>

              <h1 className="mt-2 text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
                {featured.name}
              </h1>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
                {featured.tagline}
              </p>
              <p className="mt-3 text-xs text-muted-foreground/85">{featured.tags.join(" · ")}</p>
            </div>

            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2.5">
                <Button asChild className="whitespace-nowrap shrink-0">
                  <Link to="/call" onClick={() => selectCharacter(featured.id)}>
                    <Video className="h-4 w-4" />
                    <span>Iniciar videollamada</span>
                  </Link>
                </Button>
                <Button asChild variant="secondary" className="whitespace-nowrap shrink-0">
                  <Link to="/chat" onClick={() => selectCharacter(featured.id)}>
                    <MessageSquareText className="h-4 w-4" />
                    <span>Abrir chat</span>
                  </Link>
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  disabled={uploading}
                  onClick={() => upload.current?.click()}
                  className="whitespace-nowrap shrink-0"
                >
                  <ImagePlus className="h-4 w-4" />
                  <span>{uploading ? "Procesando…" : "Subir retrato"}</span>
                </Button>
              </div>
              {uploadError && (
                <p role="alert" className="text-xs text-destructive">
                  {uploadError}
                </p>
              )}
            </div>
          </div>

          <div className="relative h-[280px] w-full overflow-hidden rounded-xl border border-border bg-[#0a0a0a] sm:h-[340px] lg:h-[360px]">
            <AnimatedAvatar
              src={featured.avatar}
              name={featured.name}
              active={false}
              engine={p.avatarEngine ?? "mediapipe-mesh"}
              liveAvatarUrl={p.liveAvatarUrl}
              localOnly={p.localOnly}
              fit="contain"
              className="h-full w-full"
            />
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

        <section className="mt-9" aria-label="Reparto disponible">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-foreground">Reparto disponible</h2>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Selecciona cualquier perfil para cargarlo en escena o iniciar sesión
              </p>
            </div>
            <label className="flex h-9 w-full items-center gap-2 rounded-lg border border-border bg-card px-3 sm:w-72">
              <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
              <input
                aria-label="Buscar personajes"
                placeholder="Filtrar por nombre o rol…"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="min-w-0 w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              />
            </label>
          </div>

          <div
            className="mt-5 flex items-center gap-1 overflow-x-auto border-b border-border pb-0"
            role="tablist"
            aria-label="Categorías"
          >
            {CATEGORIES.map((tag) => {
              const active = category === tag;
              return (
                <button
                  key={tag}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setCategory(tag)}
                  className={`whitespace-nowrap shrink-0 border-b-2 px-3 py-2 text-xs font-medium transition-colors ${
                    active
                      ? "border-foreground text-foreground"
                      : "border-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {tag}
                </button>
              );
            })}
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {characters.map((character) => {
              const isSelected = p.characterId === character.id;
              return (
                <article
                  key={character.id}
                  className={`flex min-w-0 items-center gap-3.5 rounded-xl border bg-card p-3.5 transition-colors ${
                    isSelected ? "border-foreground/50" : "border-border hover:border-foreground/25"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => selectCharacter(character.id)}
                    className="relative h-20 w-16 shrink-0 overflow-hidden rounded-lg border border-border bg-[#0a0a0a]"
                    aria-label={`Seleccionar a ${character.name}`}
                  >
                    <img
                      src={character.avatar}
                      alt={character.name}
                      className="h-full w-full object-cover object-top"
                    />
                    {isSelected && (
                      <span
                        title="Perfil en escena"
                        className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded bg-background/85 text-foreground"
                      >
                        <Check className="h-3 w-3" />
                      </span>
                    )}
                  </button>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <button
                        type="button"
                        onClick={() => selectCharacter(character.id)}
                        className="min-w-0 text-left"
                      >
                        <h3 className="truncate text-base font-semibold text-foreground">
                          {character.name}
                        </h3>
                        <p className="truncate text-xs text-muted-foreground">
                          {character.roleLabel} · {character.tags.join(" · ")}
                        </p>
                      </button>
                      <div className="flex shrink-0 items-center gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          title={`Escuchar a ${character.name}`}
                          aria-label={`Escuchar a ${character.name}`}
                          onClick={() => {
                            unlockAudio();
                            void speak(
                              `Hola, soy ${character.name}. ¿Cómo va tu día?`,
                              character.voice,
                              p.language,
                              undefined,
                              { localOnly: p.localOnly },
                            );
                          }}
                        >
                          <AudioLines className="h-4 w-4 text-muted-foreground" />
                        </Button>
                        {character.isCustom && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                            title={`Eliminar a ${character.name}`}
                            aria-label={`Eliminar a ${character.name}`}
                            onClick={() => deleteCustomCharacter(character.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>

                    <p className="mt-1.5 line-clamp-1 text-xs text-muted-foreground">
                      {character.tagline}
                    </p>

                    <div className="mt-2.5 flex items-center gap-2">
                      <Button asChild size="sm" className="h-8 whitespace-nowrap shrink-0 px-3">
                        <Link to="/call" onClick={() => selectCharacter(character.id)}>
                          <Video className="h-3.5 w-3.5" />
                          <span>Llamar</span>
                        </Link>
                      </Button>
                      <Button
                        asChild
                        variant="secondary"
                        size="sm"
                        className="h-8 whitespace-nowrap shrink-0 px-3"
                      >
                        <Link to="/chat" onClick={() => selectCharacter(character.id)}>
                          <MessageSquareText className="h-3.5 w-3.5" />
                          <span>Chat</span>
                        </Link>
                      </Button>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>

          {characters.length === 0 && (
            <p className="py-12 text-center text-sm text-muted-foreground">
              No hay personajes que coincidan con tu búsqueda.
            </p>
          )}
        </section>

        <section className="mt-9 pb-4">
          <h2 className="mb-3 font-mono text-xs uppercase tracking-widest text-muted-foreground">
            Última sesión
          </h2>
          <Link
            to="/chat"
            className="flex items-center gap-4 rounded-xl border border-border bg-card p-3.5 transition-colors hover:bg-secondary/60"
          >
            <AvatarImg src={p.avatar} name={p.name} className="h-11 w-11 shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-foreground">{p.name}</p>
              <p className="mt-0.5 truncate text-xs text-muted-foreground">
                {messages.at(-1)?.content ?? "¿De qué quieres hablar hoy?"}
              </p>
            </div>
            <MessageSquareText className="h-4 w-4 shrink-0 text-muted-foreground" />
          </Link>
        </section>
      </main>
    </div>
  );
}
