import { createFileRoute } from "@tanstack/react-router";
export const Route = createFileRoute("/api/liveavatar/status")({ server: { handlers: { GET: () => Response.json({ ok: false, error: "Falta conectar una API de inferencia GPU con sesiones de vídeo; una página accesible no verifica LiveAvatar." }) } } });
