import { Link, useNavigate } from "@tanstack/react-router";
import {
  Clapperboard,
  MessageSquareText,
  ScanFace,
  SlidersHorizontal,
  UserPlus,
  Video,
} from "lucide-react";
import { CHARACTERS } from "@/lib/characters";
import {
  createCustomCharacter,
  selectCharacter,
  useCustomCharacters,
  useProfile,
} from "@/lib/store";
import { AvatarImg } from "./AvatarImg";
import { BrandLogo } from "./BrandLogo";
import { Button } from "./ui/button";

export function AppSidebar({ current }: { current: "discover" | "chat" | "profile" }) {
  const p = useProfile();
  const customChars = useCustomCharacters();
  const navigate = useNavigate();

  const onCreateNew = () => {
    createCustomCharacter();
    void navigate({ to: "/profile" });
  };

  return (
    <>
      <aside className="app-sidebar">
        <div className="px-2">
          <Link to="/" className="inline-flex items-center">
            <BrandLogo />
          </Link>
        </div>

        <Button
          type="button"
          variant="default"
          onClick={onCreateNew}
          className="mt-6 w-full justify-start whitespace-nowrap rounded-xl font-extrabold"
        >
          <UserPlus className="h-4 w-4" />
          <span>Crear personaje</span>
        </Button>

        <nav className="mt-5 space-y-1" aria-label="Navegación principal">
          <Link
            to="/"
            className={`flex h-9 w-full items-center gap-2.5 rounded-xl px-3 text-sm font-extrabold transition-colors ${
              current === "discover"
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:bg-secondary/70 hover:text-foreground"
            }`}
          >
            <Clapperboard className="h-4 w-4 shrink-0" />
            <span className="truncate">Descubrir</span>
          </Link>
          <Link
            to="/chat"
            className={`flex h-9 w-full items-center gap-2.5 rounded-xl px-3 text-sm font-extrabold transition-colors ${
              current === "chat"
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:bg-secondary/70 hover:text-foreground"
            }`}
          >
            <MessageSquareText className="h-4 w-4 shrink-0" />
            <span className="truncate">Chatear</span>
          </Link>
          <Link
            to="/call"
            className="flex h-9 w-full items-center gap-2.5 rounded-xl px-3 text-sm font-extrabold text-muted-foreground transition-colors hover:bg-secondary/70 hover:text-foreground"
          >
            <Video className="h-4 w-4 shrink-0" />
            <span className="truncate">Videollamada</span>
          </Link>
          <Link
            to="/mirror"
            className="flex h-9 w-full items-center gap-2.5 rounded-xl px-3 text-sm font-extrabold text-muted-foreground transition-colors hover:bg-secondary/70 hover:text-foreground"
          >
            <ScanFace className="h-4 w-4 shrink-0" />
            <span className="truncate">Espejo</span>
          </Link>
        </nav>

        <div className="mt-6 flex min-h-0 flex-1 flex-col overflow-hidden">
          {customChars.length > 0 && (
            <div className="mb-4">
              <p className="px-2 font-mono text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                Mis personajes
              </p>
              <div className="mt-1.5 max-h-36 space-y-0.5 overflow-y-auto pr-1">
                {customChars.map((ch) => {
                  const active = p.characterId === ch.id;
                  return (
                    <button
                      key={ch.id}
                      type="button"
                      onClick={() => {
                        selectCharacter(ch.id);
                        if (current === "discover") {
                          void navigate({ to: "/chat" });
                        }
                      }}
                      className={`flex w-full items-center gap-2.5 rounded-xl px-2 py-2 text-left transition-colors ${
                        active
                          ? "bg-secondary text-foreground"
                          : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                      }`}
                    >
                      <AvatarImg src={ch.avatar} name={ch.name} className="h-8 w-8 shrink-0" />
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-extrabold text-foreground">{ch.name}</p>
                        <p className="truncate text-[11px] font-bold text-muted-foreground">
                          {ch.roleLabel}
                        </p>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <p className="px-2 font-mono text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            Personajes
          </p>
          <div className="mt-1.5 flex-1 space-y-0.5 overflow-y-auto pr-1">
            {CHARACTERS.map((ch) => {
              const active = p.characterId === ch.id;
              return (
                <button
                  key={ch.id}
                  type="button"
                  onClick={() => {
                    selectCharacter(ch.id);
                    if (current === "discover") {
                      void navigate({ to: "/chat" });
                    }
                  }}
                  className={`flex w-full items-center gap-2.5 rounded-xl px-2 py-2 text-left transition-colors ${
                    active
                      ? "bg-secondary text-foreground"
                      : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                  }`}
                >
                  <AvatarImg src={ch.avatar} name={ch.name} className="h-8 w-8 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-extrabold text-foreground">{ch.name}</p>
                    <p className="truncate text-[11px] font-bold text-muted-foreground">
                      {ch.roleLabel}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="mt-4 space-y-1 border-t border-sidebar-border pt-3">
          <Link
            to="/account"
            className="flex h-9 w-full items-center rounded-xl px-3 text-sm font-extrabold text-muted-foreground transition-colors hover:bg-secondary/70 hover:text-foreground"
          >
            Mi cuenta
          </Link>
          <Link
            to="/profile"
            className={`flex h-9 w-full items-center gap-2.5 rounded-xl px-3 text-sm font-extrabold transition-colors ${
              current === "profile"
                ? "bg-secondary text-foreground"
                : "text-muted-foreground hover:bg-secondary/70 hover:text-foreground"
            }`}
          >
            <SlidersHorizontal className="h-4 w-4 shrink-0" />
            <span>Personalizar</span>
          </Link>
        </div>
      </aside>

      {current !== "chat" && (
        <nav className="mobile-bottom-nav md:hidden" aria-label="Navegación móvil">
          <Link
            to="/"
            className={`flex h-11 flex-col items-center justify-center gap-0.5 text-[11px] font-extrabold ${
              current === "discover" ? "text-primary" : "text-muted-foreground"
            }`}
          >
            <Clapperboard className="h-4 w-4" />
            <span>Descubrir</span>
          </Link>
          <Link
            to="/chat"
            className="flex h-11 flex-col items-center justify-center gap-0.5 text-[11px] font-extrabold text-muted-foreground"
          >
            <MessageSquareText className="h-4 w-4" />
            <span>Chat</span>
          </Link>
          <Link
            to="/call"
            className="flex h-11 flex-col items-center justify-center gap-0.5 text-[11px] font-extrabold text-muted-foreground"
          >
            <Video className="h-4 w-4" />
            <span>Llamar</span>
          </Link>
          <Link
            to="/mirror"
            className="flex h-11 flex-col items-center justify-center gap-0.5 text-[11px] font-extrabold text-muted-foreground"
          >
            <ScanFace className="h-4 w-4" />
            <span>Espejo</span>
          </Link>
          <Link
            to="/profile"
            className={`flex h-11 flex-col items-center justify-center gap-0.5 text-[11px] font-extrabold ${
              current === "profile" ? "text-primary" : "text-muted-foreground"
            }`}
          >
            <SlidersHorizontal className="h-4 w-4" />
            <span>Ajustes</span>
          </Link>
        </nav>
      )}
    </>
  );
}
