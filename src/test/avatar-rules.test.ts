import { describe, expect, it, vi } from "vitest";
import { blinkClosure } from "@/lib/avatar-motion";
import { providerError } from "@/lib/provider-error";
const auth = vi.hoisted(() => ({ getUser: vi.fn() }));
const from = vi.hoisted(() => vi.fn());
vi.mock("@/integrations/supabase/client", () => ({ supabase: { auth, from } }));
import { clearMemoryCache, memoryPrompt, readMemories, saveMemory } from "@/lib/memories";
import { CHARACTERS } from "@/lib/characters";
import { getProfile, selectCharacter } from "@/lib/store";

describe("Avatar and private memory rules", () => {
  it("preserves each character portrait when selected instead of falling back to a single default", () => {
    const avatars = new Set(CHARACTERS.map((c) => c.avatar));
    expect(avatars.size).toBe(CHARACTERS.length);
    for (const char of CHARACTERS) {
      selectCharacter(char.id);
      const profile = getProfile();
      expect(profile.characterId).toBe(char.id);
      expect(profile.avatar).toBe(char.avatar);
    }
  });
  it("fully closes the eyelids during a blink", () => {
    expect(blinkClosure(0.4)).toBe(1);
    expect(blinkClosure(1)).toBe(0);
  });
  it("does not retain private memories after signing out", async () => {
    clearMemoryCache();
    auth.getUser.mockResolvedValueOnce({ data: { user: { id: "owner" } } });
    from.mockReturnValue({
      select: () => ({
        eq: () => ({
          order: async () => ({
            data: [{ id: "memory", character_id: "sofia", content: "Me llamo Alex" }],
            error: null,
          }),
        }),
      }),
    });
    await readMemories("sofia");
    expect(memoryPrompt("sofia")).toContain("Me llamo Alex");
    expect(memoryPrompt("daniel")).toBe("");
    auth.getUser.mockResolvedValueOnce({ data: { user: null } });
    await readMemories("sofia");
    expect(memoryPrompt("sofia")).toBe("");
  });
  it("requires a signed-in owner before saving a memory", async () => {
    auth.getUser.mockResolvedValueOnce({ data: { user: null } });
    await expect(saveMemory("sofia", "Mi nombre es Alex")).rejects.toThrow("Inicia sesión");
  });
  it("preserves the provider quota status instead of masking it as a server failure", () => {
    expect(
      providerError(new Error(JSON.stringify({ error: { code: 429, message: "Quota exceeded" } }))),
    ).toEqual({
      status: 429,
      message:
        "Google ha agotado la cuota de voz. Revisa el límite y la facturación de Google AI Studio.",
    });
  });
});
