import { Link, useNavigate } from "@tanstack/react-router";
import { Clapperboard, MessageSquareText, SlidersHorizontal, UserPlus, Video } from "lucide-react";
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
        className="mt-6 w-full justify-start whitespace-nowrap rounded-xl font-medium"
      >
        <UserPlus className="h-4 w-4" />
        <span>Crear personaje</span>
      </Button>

      <nav className="mt-5 space-y-1" aria-label="Navegación principal">
        <Button
          asChild
          variant={current === "discover" ? "secondary" : "ghost"}
          className="w-full justify-start whitespace-nowrap rounded-lg"
        >
          <Link to="/">
            <Clapperboard className="h-4 w-4" />
            <span>Estudio</span>
          </Link>
        </Button>
        <Button
          asChild
          variant={current === "chat" ? "secondary" : "ghost"}
          className="w-full justify-start whitespace-nowrap rounded-lg"
        >
          <Link to="/chat">
            <MessageSquareText className="h-4 w-4" />
            <span>Conversación</span>
          </Link>
        </Button>
        <Button
          asChild
          variant="ghost"
          className="w-full justify-start whitespace-nowrap rounded-lg"
        >
          <Link to="/call">
            <Video className="h-4 w-4" />
            <span>Videollamada</span>
          </Link>
        </Button>
      </nav>

      <div className="mt-6 flex min-h-0 flex-1 flex-col overflow-hidden">
        {customChars.length > 0 && (
          <div className="mb-4">
            <p className="px-2 font-mono text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
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
                    className={`flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition-colors ${
                      active
                        ? "bg-secondary text-foreground ring-1 ring-foreground/25"
                        : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                    }`}
                  >
                    <AvatarImg src={ch.avatar} name={ch.name} className="h-8 w-8 shrink-0" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-foreground">{ch.name}</p>
                      <p className="truncate text-[11px] text-muted-foreground">{ch.roleLabel}</p>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <p className="px-2 font-mono text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          Reparto
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
                className={`flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition-colors ${
                  active
                    ? "bg-secondary text-foreground ring-1 ring-foreground/25"
                    : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
                }`}
              >
                <AvatarImg src={ch.avatar} name={ch.name} className="h-8 w-8 shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-foreground">{ch.name}</p>
                  <p className="truncate text-[11px] text-muted-foreground">{ch.roleLabel}</p>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-4 border-t border-sidebar-border pt-3">
        <Button
          asChild
          variant={current === "profile" ? "secondary" : "ghost"}
          className="w-full justify-start whitespace-nowrap rounded-lg"
        >
          <Link to="/profile">
            <SlidersHorizontal className="h-4 w-4" />
            <span>Personalizar</span>
          </Link>
        </Button>
      </div>
    </aside>
  );
}
