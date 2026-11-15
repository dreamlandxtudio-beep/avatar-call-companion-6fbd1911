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
      { title: "YUME — personajes y conexiones románticas" },
      {
        name: "description",
        content:
          "Descubre compañeros adultos realistas y anime para conversaciones, romance ficticio y videollamadas.",
      },
      { property: "og:title", content: "YUME — personajes y conexiones románticas" },
      {
        property: "og:description",
        content:
          "Conecta con personajes adultos de personalidad propia, voz y recuerdos que tú eliges guardar.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Home,
});

const CATEGORIES = ["Todos", "Romance", "Realistas", "Anime", "Creativa", "Personalizado"];

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


  return (
    <div className="app-layout">
      <AppSidebar current="discover" />

      <main className="discovery-content">
        <header className="flex items-center justify-between gap-4 border-b border-border pb-4">
          <Link to="/" className="md:hidden">
            <BrandLogo />
          </Link>
          <p className="hidden text-sm text-muted-foreground md:block">
            Tus personajes
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

        <section className="border-b border-border py-7" aria-label="Descubrir personajes">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div><p className="text-sm font-semibold text-primary">Un mundo de conexiones</p><h1 className="mt-2 text-3xl font-black">Encuentra a tu persona favorita</h1></div>
            <Button variant="secondary" onClick={() => upload.current?.click()} disabled={uploading}><ImagePlus className="h-4 w-4" />{uploading ? "Procesando…" : "Crear con una foto"}</Button>
          </div>
          {uploadError && <p role="alert" className="mt-3 text-xs text-destructive">{uploadError}</p>}
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

        <section className="mt-9" aria-label="Para ti">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-foreground">Para ti</h2>
              <p className="mt-0.5 text-xs text-muted-foreground">
                
              </p>
            </div>
            <label className="flex h-9 w-full items-center gap-2 rounded-lg border border-border bg-card px-3 sm:w-72">
              <Search className="h-4 w-4 shrink-0 text-muted-foreground" />
              <input
                aria-label="Buscar personajes"
                placeholder="Busca tu próxima conexión…"
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
                <Button
                  variant="ghost"
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
                </Button>
              );
            })}
          </div>

          <div className="character-grid mt-5">
            {characters.map((character) => {
              const isSelected = p.characterId === character.id;
              return (
                <article
                  key={character.id}
                  className={`min-w-0 overflow-hidden rounded-2xl border bg-card transition-colors ${isSelected ? "border-primary/60" : "border-border hover:border-primary/40"}`}
                >
                  <Button type="button" variant="ghost" onClick={() => selectCharacter(character.id)} className="relative block h-auto w-full overflow-hidden rounded-none p-0" aria-label={`Seleccionar a ${character.name}`}>
                    <img src={character.avatar} alt={character.name} className="character-cover" />
                    {isSelected && <span className="absolute right-3 top-3 rounded-full bg-primary p-1 text-primary-foreground"><Check className="h-3 w-3" /></span>}
                  </Button>
                  <div className="min-w-0 p-3.5">
                    <div className="flex items-center justify-between gap-2">
                      <Button
                        variant="ghost"
                        type="button"
                        onClick={() => selectCharacter(character.id)}
                        className="h-auto min-w-0 justify-start p-0 text-left"
                      >
                        <div className="min-w-0"><h3 className="truncate text-base font-extrabold text-foreground">
                          {character.name}
                        </h3>
                        <p className="truncate text-xs text-muted-foreground">
                          {character.roleLabel} · {character.tags.join(" · ")}
                        </p>
                        </div>
                      </Button>
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
                            ).catch(error => setUploadError(error instanceof Error ? error.message : "No se pudo reproducir la voz."));
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

                    <p className="mt-1.5 line-clamp-2 min-h-8 text-xs text-muted-foreground">
                      {character.tagline}
                    </p>

                    <div className="mt-2.5 flex flex-wrap items-center gap-2">
                      <Button asChild size="sm" className="h-8 whitespace-nowrap shrink-0 px-2.5">
                        <Link to="/call" onClick={() => selectCharacter(character.id)}>
                          <Video className="h-3.5 w-3.5" />
                          <span>Llamar</span>
                        </Link>
                      </Button>
                      <Button
                        asChild
                        variant="secondary"
                        size="sm"
                        className="h-8 whitespace-nowrap shrink-0 px-2.5"
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
