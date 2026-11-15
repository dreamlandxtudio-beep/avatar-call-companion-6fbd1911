import { createFileRoute } from "@tanstack/react-router";
export const Route = createFileRoute("/api/face-landmarks")({
 server: { handlers: { POST: async ({ request }) => {
  const { handleApiFaceLandmarks } = await import("@/server/source-api");
  return handleApiFaceLandmarks(request);
 } } }
});
