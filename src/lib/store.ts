import { useSyncExternalStore } from "react";
import { CHARACTERS, characterById, type CharacterDef } from "./characters";

export type VoiceChoice = { engine: "gemini" | "kokoro" | "system"; id: string };
export type AvatarGender = "female" | "male";
export type ModelId =
  | "gemini-flash-lite"
  | "gemini-cloud"
  | "smollm2-135m"
  | "smollm2-360m"
  | "qwen25-0.5b"
  | "qwen3-0.6b"
  | "qwen25-1.5b";

export type AvatarEngine = "mediapipe-mesh" | "liveavatar";

export type Profile = {
  characterId?: string | undefined;
  name: string;
  avatar: string | null; // data URL or relative photo URL
  gender?: AvatarGender | undefined;
  personality: string;
  backstory: string;
  style: string;
  language: "es-ES" | "en-US";
  voice: VoiceChoice;
  modelId: ModelId;
  avatarEngine?: AvatarEngine | undefined;
  liveAvatarUrl?: string | undefined;
  localOnly?: boolean | undefined;
  configVersion?: number | undefined;
};
export type Msg = {
  role: "user" | "assistant";
  content: string;
  gifUrl?: string | undefined;
  gifAlt?: string | undefined;
  at: number;
};

const FEMALE_VOICES = new Set([
  "Kore",
  "Zephyr",
  "af_heart",
  "af_bella",
  "af_nicole",
  "bf_emma",
  "Aoede",
  "Leda",
  "Sulafat",
]);
const MALE_VOICES = new Set([
  "Puck",
  "Charon",
  "Fenrir",
  "am_michael",
  "am_fenrir",
  "bm_george",
  "Orus",
  "Algenib",
  "Achird",
]);

export function defaultVoiceForGender(
  gender: AvatarGender,
  current?: VoiceChoice,
  localOnly?: boolean,
): VoiceChoice {
  if (localOnly && current?.engine === "gemini") {
    return { engine: "system", id: "" };
  }
  if (localOnly && !current) {
    return { engine: "system", id: "" };
  }
  if (gender === "male") {
    if (current && (current.engine === "system" || MALE_VOICES.has(current.id))) return current;
    return { engine: "gemini", id: "Charon" };
  }
  if (current && (current.engine === "system" || FEMALE_VOICES.has(current.id))) return current;
  return { engine: "gemini", id: "Kore" };
}

const VOICE_REMAP: Record<string, string> = {
  Orus: "Charon",
  Algenib: "Fenrir",
  Achird: "Puck",
  Aoede: "Zephyr",
  Leda: "Kore",
  Sulafat: "Kore",
};

const DEFAULT_CHAR = CHARACTERS[0]!;
const CONFIG_VERSION = 10;
const CUSTOM_CHARS_KEY = "amiga.customCharacters";

const DEFAULT: Profile = {
  characterId: DEFAULT_CHAR.id,
  name: DEFAULT_CHAR.name,
  avatar: DEFAULT_CHAR.avatar,
  gender: DEFAULT_CHAR.gender,
  personality: DEFAULT_CHAR.personality,
  backstory: DEFAULT_CHAR.backstory,
  style: DEFAULT_CHAR.style,
  language: "es-ES",
  voice: DEFAULT_CHAR.voice,
  modelId: "gemini-flash-lite",
  avatarEngine: "mediapipe-mesh",
  liveAvatarUrl: "http://localhost:7860",
  localOnly: false,
  configVersion: CONFIG_VERSION,
};

const read = <T>(k: string, d: T): T => {
  if (typeof window === "undefined") return d;
  try {
    const v = localStorage.getItem(k);
    return v ? ({ ...(d as object), ...JSON.parse(v) } as T) : d;
  } catch {
    return d;
  }
};

function readCustomCharacters(): CharacterDef[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(CUSTOM_CHARS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((c) => ({ ...c, isCustom: true })) as CharacterDef[];
  } catch {
    return [];
  }
}

function saveCustomCharacters(list: CharacterDef[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(CUSTOM_CHARS_KEY, JSON.stringify(list));
  } catch {
    /* ignore quota errors */
  }
}

let profile: Profile = DEFAULT;
let customCharacters: CharacterDef[] = [];
let messages: Msg[] = [];
let loaded = false;
const subs = new Set<() => void>();
const emit = () => subs.forEach((f) => f());

function ensure() {
  if (loaded || typeof window === "undefined") return;
  loaded = true;
  customCharacters = readCustomCharacters();
  profile = read("amiga.profile", DEFAULT);

  // Migrate legacy transient "custom" avatar into customCharacters so it is never lost
  if (
    profile.characterId === "custom" &&
    profile.avatar &&
    !customCharacters.some((c) => c.avatar === profile.avatar)
  ) {
    const migrated: CharacterDef = {
      id: `custom-${Date.now()}`,
      name: profile.name || "Nuevo personaje",
      avatar: profile.avatar,
      gender: profile.gender ?? "female",
      roleLabel: "Personaje personalizado",
      tagline: "Avatar creado a partir de tu imagen.",
      personality: profile.personality,
      backstory: profile.backstory,
      style: profile.style,
      voice: profile.voice,
      tags: ["Personalizado", "Entrevistas"],
      starters: [
        "Empecemos una simulación de entrevista laboral.",
        "Hablemos sobre un proyecto reciente.",
        "¿Cómo te gustaría que conversemos hoy?",
      ],
      isCustom: true,
    };
    customCharacters = [migrated, ...customCharacters];
    saveCustomCharacters(customCharacters);
    profile.characterId = migrated.id;
  }

  if (!profile.configVersion || profile.configVersion < CONFIG_VERSION) {
    const isBuiltInNonCustom = !profile.characterId || !!characterById(profile.characterId);
    const shouldResetDefault =
      !profile.name ||
      profile.name === "Luna" ||
      profile.name === "Alma" ||
      profile.name === "Vesper" ||
      (isBuiltInNonCustom && profile.characterId === "rei");

    if (shouldResetDefault) {
      profile = {
        ...DEFAULT,
        language: profile.language ?? "es-ES",
        avatarEngine: profile.avatarEngine ?? "mediapipe-mesh",
        liveAvatarUrl: profile.liveAvatarUrl ?? "http://localhost:7860",
        localOnly: profile.localOnly ?? false,
        configVersion: CONFIG_VERSION,
      };
    } else {
      const gender: AvatarGender =
        profile.gender ?? (profile.avatar?.includes("male.jpg") ? "male" : "female");
      const remappedVoice =
        profile.voice?.engine === "gemini" && VOICE_REMAP[profile.voice.id]
          ? { engine: "gemini" as const, id: VOICE_REMAP[profile.voice.id]! }
          : profile.voice;
      profile = {
        ...profile,
        characterId: profile.characterId ?? DEFAULT_CHAR.id,
        gender,
        modelId: profile.modelId ?? "gemini-flash-lite",
        avatarEngine: profile.avatarEngine ?? "mediapipe-mesh",
        liveAvatarUrl: profile.liveAvatarUrl ?? "http://localhost:7860",
        localOnly: profile.localOnly ?? false,
        voice: defaultVoiceForGender(gender, remappedVoice, profile.localOnly),
        configVersion: CONFIG_VERSION,
      };
    }
    try {
      localStorage.setItem("amiga.profile", JSON.stringify(profile));
    } catch {
      /* ignore */
    }
  }
  try {
    messages = JSON.parse(
      localStorage.getItem(messageKey()) || localStorage.getItem("amiga.messages") || "[]",
    );
  } catch {
    messages = [];
  }
}

function messageKey() {
  return `amiga.messages.${profile.characterId ?? "default"}`;
}

export function getCustomCharacters(): CharacterDef[] {
  ensure();
  return customCharacters;
}

export function getAllCharacters(): CharacterDef[] {
  ensure();
  return [...customCharacters, ...CHARACTERS];
}

export function findCharacterById(id?: string): CharacterDef | undefined {
  if (!id) return undefined;
  ensure();
  return customCharacters.find((c) => c.id === id) ?? characterById(id);
}

export function selectCharacter(id: string) {
  ensure();
  const ch = findCharacterById(id);
  if (!ch) return;
  const nextVoice =
    profile.localOnly && ch.voice.engine === "gemini"
      ? { engine: "system" as const, id: "" }
      : ch.voice;
  profile = {
    ...profile,
    characterId: ch.id,
    name: ch.name,
    avatar: ch.avatar,
    gender: ch.gender,
    personality: ch.personality,
    backstory: ch.backstory,
    style: ch.style,
    voice: nextVoice,
  };
  try {
    localStorage.setItem("amiga.profile", JSON.stringify(profile));
    messages = JSON.parse(localStorage.getItem(messageKey()) || "[]");
  } catch {
    messages = [];
  }
  emit();
}

export function getProfile() {
  ensure();
  return profile;
}

export function setProfile(p: Partial<Profile>) {
  ensure();
  const nextLocalOnly = p.localOnly ?? profile.localOnly ?? false;
  const nextGender = p.gender ?? profile.gender ?? "female";
  const nextVoice =
    p.gender && !p.voice
      ? defaultVoiceForGender(nextGender, profile.voice, nextLocalOnly)
      : (p.voice ?? profile.voice);
  profile = {
    ...profile,
    ...p,
    localOnly: nextLocalOnly,
    gender: nextGender,
    voice: nextVoice,
  };

  // Sync edits back into customCharacters if active character is user-created
  if (profile.characterId && customCharacters.some((c) => c.id === profile.characterId)) {
    customCharacters = customCharacters.map((c) =>
      c.id === profile.characterId
        ? {
            ...c,
            name: profile.name,
            avatar: profile.avatar ?? c.avatar,
            gender: nextGender,
            personality: profile.personality,
            backstory: profile.backstory,
            style: profile.style,
            voice: nextVoice,
          }
        : c,
    );
    saveCustomCharacters(customCharacters);
  }

  try {
    localStorage.setItem("amiga.profile", JSON.stringify(profile));
  } catch {
    /* ignore */
  }
  emit();
}

export function setLocalOnlyMode(enabled: boolean) {
  ensure();
  const updates: Partial<Profile> = { localOnly: enabled };
  if (enabled) {
    if (profile.modelId === "gemini-flash-lite" || profile.modelId === "gemini-cloud") {
      updates.modelId = "qwen25-0.5b";
    }
    if (profile.voice.engine === "gemini") {
      updates.voice = { engine: "system", id: "" };
    }
  }
  setProfile(updates);
}

export function createCustomCharacter(
  overrides?: Partial<CharacterDef> & { avatar?: string | null },
): CharacterDef {
  ensure();
  const id = `custom-${Date.now()}`;
  const index = customCharacters.length + 1;
  const gender: AvatarGender = overrides?.gender ?? "female";
  const defaultAvatar = overrides?.avatar ?? CHARACTERS[0]!.avatar;
  const voice: VoiceChoice =
    overrides?.voice ?? defaultVoiceForGender(gender, undefined, profile.localOnly);

  const created: CharacterDef = {
    id,
    name: overrides?.name?.trim() || `Personaje ${index}`,
    avatar: defaultAvatar,
    gender,
    roleLabel: overrides?.roleLabel || "Personaje personalizado · Entrevistas",
    tagline: overrides?.tagline || "Avatar creado desde tu propia imagen.",
    personality:
      overrides?.personality ||
      "Profesional, atento, cercano y con expresión natural. Escucha con atención y responde de forma clara.",
    backstory:
      overrides?.backstory ||
      "Personaje personalizado listo para entrevistas profesionales, prácticas o conversación natural.",
    style:
      overrides?.style ||
      "Conversación fluida, natural y directa en primera persona, sin sonar robótico.",
    voice,
    tags: overrides?.tags ?? ["Personalizado", "Entrevistas", "Realistas"],
    starters: overrides?.starters ?? [
      "Empecemos una simulación de entrevista laboral.",
      "Cuéntame sobre ti y tu enfoque profesional.",
      "Conversemos sobre un tema de actualidad.",
    ],
    isCustom: true,
  };

  customCharacters = [created, ...customCharacters];
  saveCustomCharacters(customCharacters);
  selectCharacter(created.id);
  return created;
}

export function deleteCustomCharacter(id: string) {
  ensure();
  const nextList = customCharacters.filter((c) => c.id !== id);
  if (nextList.length === customCharacters.length) return;
  customCharacters = nextList;
  saveCustomCharacters(customCharacters);
  try {
    localStorage.removeItem(`amiga.messages.${id}`);
  } catch {
    /* ignore */
  }
  if (profile.characterId === id) {
    selectCharacter(CHARACTERS[0]!.id);
  } else {
    emit();
  }
}

export async function applyAvatarPhoto(dataUrl: string, createNewIfBuiltIn = true) {
  ensure();
  const activeIsCustom =
    !!profile.characterId && customCharacters.some((c) => c.id === profile.characterId);

  let targetId = profile.characterId;
  if (!activeIsCustom && createNewIfBuiltIn) {
    const created = createCustomCharacter({
      avatar: dataUrl,
      name: `Avatar ${customCharacters.length + 1}`,
      gender: profile.gender ?? "female",
    });
    targetId = created.id;
  } else {
    setProfile({ avatar: dataUrl });
  }

  if (typeof window === "undefined" || !dataUrl.startsWith("data:") || profile.localOnly) {
    return;
  }

  try {
    const res = await fetch("/api/avatar-gender", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataUrl }),
    });
    if (!res.ok) return;
    const data = (await res.json()) as { gender?: AvatarGender };
    if (data.gender === "male" || data.gender === "female") {
      if (getProfile().characterId === targetId && getProfile().avatar === dataUrl) {
        setProfile({
          gender: data.gender,
          voice: defaultVoiceForGender(data.gender, getProfile().voice, getProfile().localOnly),
        });
      }
    }
  } catch {
    /* ignore */
  }
}

export function getMessages() {
  ensure();
  return messages;
}

export function addMessage(m: Omit<Msg, "at">) {
  ensure();
  messages = [...messages, { ...m, at: Date.now() }];
  try {
    localStorage.setItem(messageKey(), JSON.stringify(messages.slice(-200)));
  } catch {
    /* ignore */
  }
  emit();
}

export function clearMessages() {
  ensure();
  messages = [];
  try {
    localStorage.removeItem(messageKey());
    localStorage.removeItem("amiga.messages");
  } catch {
    /* ignore */
  }
  emit();
}

const sub = (f: () => void) => {
  subs.add(f);
  return () => subs.delete(f);
};
export const useProfile = () => useSyncExternalStore(sub, getProfile, () => DEFAULT);
const EMPTY_MSGS: Msg[] = [];
export const useMessages = () => useSyncExternalStore(sub, getMessages, () => EMPTY_MSGS);
const EMPTY_CUSTOM: CharacterDef[] = [];
export const useCustomCharacters = () =>
  useSyncExternalStore(sub, getCustomCharacters, () => EMPTY_CUSTOM);

export async function resizeImage(file: File, max = 720): Promise<string> {
  // Keep compact animated GIFs intact so GIF avatars stay animated
  if (file.type === "image/gif" && file.size <= 1_200_000) {
    return await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () =>
        typeof reader.result === "string"
          ? resolve(reader.result)
          : reject(new Error("No se pudo leer el GIF"));
      reader.onerror = () => reject(new Error("No se pudo leer el GIF"));
      reader.readAsDataURL(file);
    });
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  const s = Math.min(1, max / Math.max(img.width, img.height));
  const c = document.createElement("canvas");
  c.width = Math.round(img.width * s);
  c.height = Math.round(img.height * s);
  const context = c.getContext("2d");
  if (!context) {
    URL.revokeObjectURL(url);
    throw new Error("No se pudo leer la foto");
  }
  context.drawImage(img, 0, 0, c.width, c.height);
  URL.revokeObjectURL(url);
  return c.toDataURL("image/jpeg", 0.88);
}
