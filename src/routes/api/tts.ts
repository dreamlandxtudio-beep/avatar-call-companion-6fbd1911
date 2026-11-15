import { createFileRoute } from "@tanstack/react-router";
export const Route = createFileRoute("/api/tts")({
 server: { handlers: { POST: async ({ request }) => {
  const { handleApiTts } = await import("@/server/source-api");
  return handleApiTts(request);
 } } }
});
