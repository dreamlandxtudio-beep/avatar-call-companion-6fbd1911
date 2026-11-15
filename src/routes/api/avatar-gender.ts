import { createFileRoute } from "@tanstack/react-router";
export const Route = createFileRoute("/api/avatar-gender")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const { handleApiAvatarGender } = await import("@/server/source-api");
        return handleApiAvatarGender(request);
      },
    },
  },
});
