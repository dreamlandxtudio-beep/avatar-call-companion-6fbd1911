import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { lovable } from "@/integrations/lovable";
import { Button } from "@/components/ui/button";
import { clearMemoryCache } from "@/lib/memories";
export const Route = createFileRoute("/account")({
  head: () => ({
    meta: [
      { title: "Tu cuenta — YUME" },
      { name: "description", content: "Accede a tus recuerdos privados en YUME." },
      { property: "og:title", content: "Tu cuenta — YUME" },
      { property: "og:description", content: "Tu cuenta y recuerdos privados." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
  component: Account,
});
function Account() {
  const [mode, setMode] = useState<"login" | "signup" | "reset" | "password">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [signedIn, setSignedIn] = useState(false);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    void supabase.auth.getSession().then(({ data }) => setSignedIn(!!data.session));
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      setSignedIn(!!session);
      if (event === "PASSWORD_RECOVERY") setMode("password");
    });
    return () => subscription.unsubscribe();
  }, []);
  return (
    <main className="profile-content">
      <Button asChild variant="ghost">
        <Link to="/profile">
          <ArrowLeft />
          Volver
        </Link>
      </Button>
      <h1 className="my-6 text-3xl font-semibold">Tu cuenta</h1>
      {signedIn && mode !== "password" ? (
        <div className="space-y-4">
          <p>Sesión iniciada. Tus recuerdos son privados.</p>
          <Button asChild>
            <Link to="/profile">Ver mis recuerdos</Link>
          </Button>
          <Button
            variant="outline"
            onClick={async () => {
              await supabase.auth.signOut();
              clearMemoryCache();
            }}
          >
            Cerrar sesión
          </Button>
        </div>
      ) : (
        <>
          <form
            className="auth-form"
            onSubmit={async (e) => {
              e.preventDefault();
              setBusy(true);
              setNotice("");
              try {
                const result =
                  mode === "signup"
                    ? await supabase.auth.signUp({
                        email,
                        password,
                        options: { emailRedirectTo: window.location.origin + "/account" },
                      })
                    : mode === "reset"
                      ? await supabase.auth.resetPasswordForEmail(email, {
                          redirectTo: window.location.origin + "/account",
                        })
                      : mode === "password"
                        ? await supabase.auth.updateUser({ password })
                        : await supabase.auth.signInWithPassword({ email, password });
                if (result.error) throw result.error;
                if (mode === "signup" || mode === "reset")
                  setNotice("Revisa tu correo para continuar.");
                if (mode === "password") {
                  setMode("login");
                  setNotice("Contraseña actualizada.");
                }
              } catch (e) {
                setNotice(e instanceof Error ? e.message : "No se pudo completar.");
              } finally {
                setBusy(false);
              }
            }}
          >
            {mode !== "password" && (
              <input
                className="memory-input"
                type="email"
                aria-label="Correo electrónico"
                placeholder="Correo electrónico"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            )}
            {mode !== "reset" && (
              <input
                className="memory-input"
                type="password"
                aria-label="Contraseña"
                placeholder="Contraseña"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                minLength={8}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            )}
            <Button disabled={busy} type="submit">
              {mode === "signup"
                ? "Crear cuenta"
                : mode === "reset"
                  ? "Enviar enlace"
                  : mode === "password"
                    ? "Guardar contraseña"
                    : "Iniciar sesión"}
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                setMode(mode === "signup" ? "login" : "signup");
                setNotice("");
              }}
            >
              {mode === "signup" ? "Ya tengo cuenta" : "Crear cuenta"}
            </Button>
            <Button type="button" variant="ghost" onClick={() => setMode("reset")}>
              Olvidé mi contraseña
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={async () => {
                const result = await lovable.auth.signInWithOAuth("google", {
                  redirect_uri: window.location.origin + "/account",
                });
                if (result.error) setNotice(result.error.message);
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  fill="#4285F4"
                  d="M23.49 12.27c0-.79-.07-1.54-.19-2.27H12v4.51h6.47c-.29 1.48-1.14 2.73-2.4 3.58v3h3.86c2.26-2.09 3.56-5.17 3.56-8.82z"
                />
                <path
                  fill="#34A853"
                  d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.86-3c-1.08.72-2.45 1.16-4.07 1.16-3.13 0-5.78-2.11-6.73-4.96H1.29v3.09C3.26 21.3 7.31 24 12 24z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.27 14.29c-.25-.72-.38-1.49-.38-2.29s.14-1.57.38-2.29V6.62H1.29C.47 8.24 0 10.06 0 12s.47 3.76 1.29 5.38l3.98-3.09z"
                />
                <path
                  fill="#EA4335"
                  d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.31 0 3.26 2.7 1.29 6.62l3.98 3.09c.95-2.85 3.6-4.96 6.73-4.96z"
                />
              </svg>
              Continuar con Google
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={async () => {
                const result = await lovable.auth.signInWithOAuth("lovable", {
                  redirect_uri: window.location.origin + "/account",
                });
                if (result.error) setNotice(result.error.message);
              }}
            >
              Continuar con Lovable
            </Button>
          </form>
        </>
      )}
      {notice && (
        <p role="status" className="mt-4 text-sm">
          {notice}
        </p>
      )}
    </main>
  );
}
