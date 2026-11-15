import { GoogleGenAI, ThinkingLevel, Type } from "@google/genai";

export type ChatHistoryMsg = {
  role: "user" | "assistant";
  content: string;
};

export type CloudModelName = "gemini-3.1-flash-lite" | "gemini-4-hydrogen";

let client: GoogleGenAI | null = null;
let betaClient: GoogleGenAI | null = null;

function getClient() {
  const apiKey = process.env["GEMINI_API_KEY"]?.trim();
  if (!apiKey) {
    throw new Error("Falta configurar GEMINI_API_KEY en Settings > Secrets.");
  }
  if (!client) {
    client = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return client;
}

function getBetaClient() {
  const apiKey = process.env["GEMINI_API_KEY"]?.trim();
  if (!apiKey) {
    throw new Error("Falta configurar GEMINI_API_KEY en Settings > Secrets.");
  }
  if (!betaClient) {
    betaClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        apiVersion: "v1beta",
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return betaClient;
}

function normalizeContents(history: ChatHistoryMsg[]) {
  const cleaned = history
    .filter((m) => m.content && m.content.trim().length > 0 && !m.content.startsWith("(Error"))
    .slice(-14);

  // Drop leading assistant turns so the conversation starts cleanly with a user turn.
  while (cleaned.length > 0 && cleaned[0]?.role === "assistant") {
    cleaned.shift();
  }

  // Merge consecutive messages from the same role so turns strictly alternate user/model.
  const merged: Array<{ role: "user" | "model"; parts: Array<{ text: string }> }> = [];
  for (const m of cleaned) {
    const role = m.role === "assistant" ? "model" : "user";
    const text = m.content.trim();
    const prev = merged.at(-1);
    if (prev && prev.role === role) {
      prev.parts[0]!.text += `\n${text}`;
    } else {
      merged.push({ role, parts: [{ text }] });
    }
  }

  return merged;
}

export async function streamGeminiChat(opts: {
  model?: string | undefined;
  systemInstruction: string;
  history: ChatHistoryMsg[];
  maxOutputTokens?: number | undefined;
}) {
  const ai = getClient();
  const model: CloudModelName =
    opts.model === "gemini-4-hydrogen" ? "gemini-4-hydrogen" : "gemini-3.1-flash-lite";

  const contents = normalizeContents(opts.history);
  if (contents.length === 0) {
    throw new Error("Conversation history cannot be empty.");
  }

  const thinkingLevel =
    model === "gemini-3.1-flash-lite" ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW;

  return ai.models.generateContentStream({
    model,
    contents,
    config: {
      systemInstruction: opts.systemInstruction,
      thinkingConfig: { thinkingLevel },
      ...(opts.maxOutputTokens ? { maxOutputTokens: opts.maxOutputTokens } : {}),
    },
  });
}

export async function transcribeAudio(opts: {
  mimeType: string;
  data: string;
  language?: string;
}): Promise<string> {
  const ai = getClient();
  const cleanMime = (opts.mimeType || "audio/webm").split(";")[0]?.trim() || "audio/webm";
  const isEs = (opts.language ?? "es-ES").startsWith("es");
  const contents = {
    parts: [
      {
        inlineData: {
          mimeType: cleanMime,
          data: opts.data,
        },
      },
      {
        text: isEs
          ? "Transcribe exactamente lo que dice la persona en este audio. Devuelve únicamente el texto hablado, sin comillas ni explicaciones. Si solo hay silencio o ruido sin palabras, responde vacío."
          : "Transcribe exactly what the speaker says in this audio. Return only the spoken words without quotes or commentary. If there is only silence or background noise with no speech, return empty.",
      },
    ],
  };

  let rawText = "";
  try {
    const response = await Promise.race([
      ai.models.generateContent({
        model: "gemini-3.5-transcribe",
        contents,
      }),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("TRANSCRIBE_TIMEOUT")), 1500),
      ),
    ]);
    rawText = response.text ?? "";
  } catch {
    // Fast fallback to gemini-3.1-flash-lite with minimal thinking
    const fallback = await ai.models.generateContent({
      model: "gemini-3.1-flash-lite",
      contents,
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
      },
    });
    rawText = fallback.text ?? "";
  }

  const text = rawText.trim();
  if (/^(\[?(silencio|silence|inaudible|noise|ruido|empty)\]?|\.{1,3})$/i.test(text)) {
    return "";
  }
  return text;
}

export type GeminiVoiceName = "Kore" | "Zephyr" | "Puck" | "Charon" | "Fenrir";
const VALID_VOICES = new Set<GeminiVoiceName>(["Kore", "Zephyr", "Puck", "Charon", "Fenrir"]);
const VOICE_ALIAS: Record<string, GeminiVoiceName> = {
  Orus: "Charon",
  Algenib: "Fenrir",
  Achird: "Puck",
  Aoede: "Zephyr",
  Leda: "Kore",
  Sulafat: "Kore",
};

function normalizeVoiceName(raw?: string): GeminiVoiceName {
  if (raw && VALID_VOICES.has(raw as GeminiVoiceName)) return raw as GeminiVoiceName;
  if (raw && VOICE_ALIAS[raw]) return VOICE_ALIAS[raw]!;
  return "Kore";
}

function ttsStyleForEmotion(emotion: string | undefined, isEs: boolean, custom?: string): string {
  if (custom?.trim()) return custom.trim();
  if (isEs) {
    switch (emotion) {
      case "angry":
        return "Voz humana real, molesta y firme en español natural";
      case "laugh":
        return "Voz humana riendo con alegría espontánea en español natural";
      case "happy":
        return "Voz humana cálida, cercana y sonriente en español natural";
      case "surprised":
        return "Voz humana sorprendida y expresiva en español natural";
      case "sad":
        return "Voz humana suave y empática en español natural";
      default:
        return "Voz humana realista y conversacional en español natural";
    }
  }
  switch (emotion) {
    case "angry":
      return "Realistic firm angry human voice";
    case "laugh":
      return "Realistic joyful laughing human voice";
    case "happy":
      return "Realistic warm cheerful human voice";
    case "surprised":
      return "Realistic surprised expressive human voice";
    case "sad":
      return "Realistic soft empathetic human voice";
    default:
      return "Realistic natural conversational human voice";
  }
}

function ensureWavBuffer(raw: Buffer, mimeType?: string): Buffer {
  if (raw.length >= 12 && raw.subarray(0, 4).toString("ascii") === "RIFF") {
    return raw;
  }
  const rateMatch = /rate=(\d+)/i.exec(mimeType ?? "");
  const sampleRate = rateMatch ? Math.max(8000, Math.min(48000, Number(rateMatch[1]))) : 24000;
  const header = Buffer.alloc(44);
  const dataSize = raw.length;
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + dataSize, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(dataSize, 40);
  return Buffer.concat([header, raw]);
}

export async function synthesizeSpeech(opts: {
  text: string;
  voiceName?: string | undefined;
  language?: string | undefined;
  style?: string | undefined;
  emotion?: string | undefined;
}): Promise<Buffer> {
  const ai = getClient();
  const beta = getBetaClient();
  const isEs = (opts.language ?? "es-ES").startsWith("es");
  const voiceName = normalizeVoiceName(opts.voiceName);

  const style = ttsStyleForEmotion(opts.emotion, isEs, opts.style);
  const cleanText = opts.text.trim();
  const expressiveText =
    opts.emotion === "laugh" && !cleanText.includes("<laugh>") ? `<laugh> ${cleanText}` : cleanText;

  const callTts = async (
    clientRef: GoogleGenAI,
    model: "gemini-3.8-flash-lite-tts" | "gemini-3.8-flash-tts",
    text: string,
    withMetadata: boolean,
  ) => {
    const part = withMetadata ? { text, speechMetadata: { style } } : { text };
    const response = await clientRef.models.generateContent({
      model,
      contents: [{ role: "user", parts: [part] }],
      config: {
        responseModalities: ["AUDIO"],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName },
          },
        },
      },
    });
    const inline = response.candidates?.[0]?.content?.parts?.[0]?.inlineData;
    const base64Audio = inline?.data;
    if (!base64Audio) {
      throw new Error("No se recibió audio del servicio de voz.");
    }
    return ensureWavBuffer(Buffer.from(base64Audio, "base64"), inline?.mimeType);
  };

  try {
    return await Promise.race([
      callTts(ai, "gemini-3.8-flash-lite-tts", expressiveText, true),
      new Promise<never>((_, reject) => setTimeout(() => reject(new Error("TTS_TIMEOUT")), 3200)),
    ]);
  } catch {
    try {
      return await Promise.race([
        callTts(beta, "gemini-3.8-flash-lite-tts", cleanText, false),
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error("TTS_TIMEOUT")), 3200)),
      ]);
    } catch {
      return await callTts(ai, "gemini-3.8-flash-tts", cleanText, false);
    }
  }
}

export type VisionHandKeypoints = {
  wrist: [number, number];
  palmCenter: [number, number];
  fingerTips: [number, number];
  radius: number;
};

export type VisionFaceKeypoints = {
  detected: boolean;
  rightEye: [number, number];
  leftEye: [number, number];
  rightPupil?: [number, number] | undefined;
  leftPupil?: [number, number] | undefined;
  noseTip: [number, number];
  mouthCenter: [number, number];
  mouthLeft: [number, number];
  mouthRight: [number, number];
  chin: [number, number];
  forehead: [number, number];
  rightEar?: [number, number] | undefined;
  leftEar?: [number, number] | undefined;
  hairTop?: [number, number] | undefined;
  hairLeft?: [number, number] | undefined;
  hairRight?: [number, number] | undefined;
  neckBase?: [number, number] | undefined;
  rightShoulder?: [number, number] | undefined;
  leftShoulder?: [number, number] | undefined;
  hands?: VisionHandKeypoints[] | undefined;
  stylized: boolean;
};

export async function detectFaceKeypointsWithGemini(opts: {
  mimeType: string;
  data: string;
}): Promise<VisionFaceKeypoints | null> {
  const ai = getClient();
  const ptSchema = {
    type: Type.OBJECT,
    properties: {
      x: {
        type: Type.NUMBER,
        description: "Horizontal pixel coordinate scaled 0 (left) to 1000 (right)",
      },
      y: {
        type: Type.NUMBER,
        description: "Vertical pixel coordinate scaled 0 (top) to 1000 (bottom)",
      },
    },
    required: ["x", "y"],
  };

  const handSchema = {
    type: Type.OBJECT,
    properties: {
      wrist: ptSchema,
      palmCenter: ptSchema,
      fingerTips: ptSchema,
      radiusNorm: {
        type: Type.NUMBER,
        description: "Approximate hand radius relative to image width (0.04 to 0.22)",
      },
    },
    required: ["wrist", "palmCenter", "fingerTips", "radiusNorm"],
  };

  const response = await ai.models.generateContent({
    model: "gemini-3.1-flash-lite",
    contents: {
      parts: [
        {
          inlineData: {
            mimeType: opts.mimeType || "image/jpeg",
            data: opts.data,
          },
        },
        {
          text: "Analyze the main character's anatomy in this image (real photo, 3D render, anime, or illustration) for 3D avatar rigging. Return coordinates normalized 0..1000 (x: 0 left to 1000 right, y: 0 top to 1000 bottom): eyes and pupils, nose tip, mouth corners and center, chin, forehead hairline, ears (or temple edge if covered by hair), outer hair crown/left/right envelope, neck base (collarbone center), right/left shoulders, and any visible hands (wrist, palm center, fingertips). If no face exists, set detected=false.",
        },
      ],
    },
    config: {
      thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          detected: { type: Type.BOOLEAN },
          stylized: {
            type: Type.BOOLEAN,
            description: "True if anime, cartoon, drawing, or illustration",
          },
          eyeLeftSideOfImage: ptSchema,
          eyeRightSideOfImage: ptSchema,
          pupilLeftSideOfImage: ptSchema,
          pupilRightSideOfImage: ptSchema,
          noseTip: ptSchema,
          mouthLeftCornerInImage: ptSchema,
          mouthRightCornerInImage: ptSchema,
          mouthCenter: ptSchema,
          chinBottom: ptSchema,
          foreheadTop: ptSchema,
          earLeftSideOfImage: ptSchema,
          earRightSideOfImage: ptSchema,
          hairTop: ptSchema,
          hairLeftSide: ptSchema,
          hairRightSide: ptSchema,
          neckBase: ptSchema,
          shoulderLeftSideOfImage: ptSchema,
          shoulderRightSideOfImage: ptSchema,
          hands: {
            type: Type.ARRAY,
            items: handSchema,
            description: "All clearly visible hands in the portrait (empty array if none)",
          },
        },
        required: [
          "detected",
          "stylized",
          "eyeLeftSideOfImage",
          "eyeRightSideOfImage",
          "noseTip",
          "mouthLeftCornerInImage",
          "mouthRightCornerInImage",
          "mouthCenter",
          "chinBottom",
          "foreheadTop",
        ],
      },
    },
  });

  const raw = response.text?.trim();
  if (!raw) return null;
  const parsed = JSON.parse(raw) as {
    detected?: boolean;
    stylized?: boolean;
    eyeLeftSideOfImage?: { x: number; y: number };
    eyeRightSideOfImage?: { x: number; y: number };
    pupilLeftSideOfImage?: { x: number; y: number };
    pupilRightSideOfImage?: { x: number; y: number };
    noseTip?: { x: number; y: number };
    mouthLeftCornerInImage?: { x: number; y: number };
    mouthRightCornerInImage?: { x: number; y: number };
    mouthCenter?: { x: number; y: number };
    chinBottom?: { x: number; y: number };
    foreheadTop?: { x: number; y: number };
    earLeftSideOfImage?: { x: number; y: number };
    earRightSideOfImage?: { x: number; y: number };
    hairTop?: { x: number; y: number };
    hairLeftSide?: { x: number; y: number };
    hairRightSide?: { x: number; y: number };
    neckBase?: { x: number; y: number };
    shoulderLeftSideOfImage?: { x: number; y: number };
    shoulderRightSideOfImage?: { x: number; y: number };
    hands?: Array<{
      wrist?: { x: number; y: number };
      palmCenter?: { x: number; y: number };
      fingerTips?: { x: number; y: number };
      radiusNorm?: number;
    }>;
  };
  if (!parsed.detected) return null;
  const norm = (p?: { x: number; y: number }): [number, number] | null =>
    p && Number.isFinite(p.x) && Number.isFinite(p.y)
      ? [Math.min(0.98, Math.max(0.02, p.x / 1000)), Math.min(0.98, Math.max(0.02, p.y / 1000))]
      : null;

  const rightEye = norm(parsed.eyeLeftSideOfImage);
  const leftEye = norm(parsed.eyeRightSideOfImage);
  const noseTip = norm(parsed.noseTip);
  const mouthLeft = norm(parsed.mouthLeftCornerInImage);
  const mouthRight = norm(parsed.mouthRightCornerInImage);
  const mouthCenter = norm(parsed.mouthCenter);
  const chin = norm(parsed.chinBottom);
  const forehead = norm(parsed.foreheadTop);
  if (
    !rightEye ||
    !leftEye ||
    !noseTip ||
    !mouthLeft ||
    !mouthRight ||
    !mouthCenter ||
    !chin ||
    !forehead
  ) {
    return null;
  }

  const hands: VisionHandKeypoints[] = [];
  if (Array.isArray(parsed.hands)) {
    for (const h of parsed.hands.slice(0, 2)) {
      const wrist = norm(h.wrist);
      const palmCenter = norm(h.palmCenter);
      const fingerTips = norm(h.fingerTips);
      if (wrist && palmCenter && fingerTips) {
        hands.push({
          wrist,
          palmCenter,
          fingerTips,
          radius: Math.min(0.24, Math.max(0.04, h.radiusNorm ?? 0.09)),
        });
      }
    }
  }

  return {
    detected: true,
    stylized: !!parsed.stylized,
    rightEye,
    leftEye,
    rightPupil: norm(parsed.pupilLeftSideOfImage) ?? undefined,
    leftPupil: norm(parsed.pupilRightSideOfImage) ?? undefined,
    noseTip,
    mouthCenter,
    mouthLeft,
    mouthRight,
    chin,
    forehead,
    rightEar: norm(parsed.earLeftSideOfImage) ?? undefined,
    leftEar: norm(parsed.earRightSideOfImage) ?? undefined,
    hairTop: norm(parsed.hairTop) ?? undefined,
    hairLeft: norm(parsed.hairLeftSide) ?? undefined,
    hairRight: norm(parsed.hairRightSide) ?? undefined,
    neckBase: norm(parsed.neckBase) ?? undefined,
    rightShoulder: norm(parsed.shoulderLeftSideOfImage) ?? undefined,
    leftShoulder: norm(parsed.shoulderRightSideOfImage) ?? undefined,
    hands: hands.length > 0 ? hands : undefined,
  };
}

export async function detectAvatarGenderWithGemini(opts: {
  mimeType: string;
  data: string;
}): Promise<"male" | "female"> {
  try {
    const ai = getClient();
    const response = await ai.models.generateContent({
      model: "gemini-3.1-flash-lite",
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: opts.mimeType || "image/jpeg",
              data: opts.data,
            },
          },
          {
            text: "Look at the main person or character in this portrait. Should their speaking voice be male or female? Reply strictly in JSON.",
          },
        ],
      },
      config: {
        thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            gender: {
              type: Type.STRING,
              description: "Strictly 'male' or 'female'",
            },
          },
          required: ["gender"],
        },
      },
    });
    const parsed = JSON.parse(response.text?.trim() || "{}") as { gender?: string };
    return parsed.gender?.toLowerCase() === "male" ? "male" : "female";
  } catch {
    return "female";
  }
}
