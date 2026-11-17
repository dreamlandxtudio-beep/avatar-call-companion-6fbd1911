import { z } from "zod";
import { providerError } from "@/lib/provider-error";
import {
  detectAvatarGenderWithGemini,
  detectFaceKeypointsWithGemini,
  streamGeminiChat,
  synthesizeSpeech,
  transcribeAudio,
  type ChatHistoryMsg,
} from "./gemini";
async function readBody(request: Request) {
  const raw = await request.text();
  if (raw.length > 4_000_000) throw new Error("Solicitud demasiado grande.");
  return z.record(z.unknown()).parse(JSON.parse(raw));
}
export async function handleApiChat(request: Request): Promise<Response> {
  try {
    const body = (await readBody(request)) as {
      model?: string;
      systemInstruction?: string;
      history?: ChatHistoryMsg[];
      maxOutputTokens?: number;
    };
    const model = typeof body.model === "string" ? body.model : undefined;
    const systemInstruction = String(body.systemInstruction ?? "").trim();
    const history = Array.isArray(body.history) ? body.history : [];
    const maxOutputTokens =
      typeof body.maxOutputTokens === "number" && body.maxOutputTokens > 0
        ? body.maxOutputTokens
        : undefined;

    const stream = await streamGeminiChat({ model, systemInstruction, history, maxOutputTokens });
    const encoder = new TextEncoder();

    const readable = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of stream) {
            const text = chunk.text ?? "";
            if (text) {
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text })}\n\n`));
            }
          }
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
        } catch (err) {
          const { message } = providerError(err);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ error: message })}\n\n`));
          controller.close();
        }
      },
    });

    return new Response(readable, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
      },
    });
  } catch (err) {
    const { message, status } = providerError(err);
    return new Response(JSON.stringify({ error: message }), {
      status,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
}

export async function handleApiTranscribe(request: Request): Promise<Response> {
  try {
    const contentType = request.headers.get("content-type") ?? "";
    let mimeType = "audio/webm";
    let base64Data = "";
    let language = "es-ES";

    if (contentType.includes("multipart/form-data")) {
      const form = await request.formData();
      const audio = form.get("audio");
      const lang = form.get("language");
      if (typeof lang === "string" && lang.trim()) language = lang.trim();
      if (audio instanceof Blob) {
        mimeType = audio.type || mimeType;
        const arr = new Uint8Array(await audio.arrayBuffer());
        base64Data = Buffer.from(arr).toString("base64");
      }
    } else {
      const body = (await readBody(request)) as {
        mimeType?: string;
        data?: string;
        language?: string;
      };
      if (typeof body.mimeType === "string") mimeType = body.mimeType;
      if (typeof body.data === "string") base64Data = body.data;
      if (typeof body.language === "string") language = body.language;
    }

    if (!base64Data) {
      return new Response(JSON.stringify({ text: "" }), {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }

    const text = await transcribeAudio({ mimeType, data: base64Data, language });
    return new Response(JSON.stringify({ text }), {
      status: 200,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  } catch (err) {
    const { message, status } = providerError(err);
    return new Response(JSON.stringify({ error: message }), {
      status,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
}

export async function handleApiTts(request: Request): Promise<Response> {
  try {
    const body = (await readBody(request)) as {
      text?: string;
      voiceName?: string;
      language?: string;
      style?: string;
      emotion?: string;
      tempo?: string;
    };
    const text = String(body.text ?? "").trim();
    if (!text) {
      return new Response(JSON.stringify({ error: "Texto vacío." }), {
        status: 400,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }

    const wav = await synthesizeSpeech({
      text,
      voiceName: typeof body.voiceName === "string" ? body.voiceName : undefined,
      language: typeof body.language === "string" ? body.language : undefined,
      style: typeof body.style === "string" ? body.style : undefined,
      emotion: typeof body.emotion === "string" ? body.emotion : undefined,
      tempo: typeof body.tempo === "string" ? body.tempo : undefined,
    });

    return new Response(new Uint8Array(wav), {
      status: 200,
      headers: {
        "Content-Type": "audio/wav",
        "Cache-Control": "no-store",
      },
    });
  } catch (err) {
    const { message, status } = providerError(err);
    return new Response(JSON.stringify({ error: message }), {
      status,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
}

export async function handleApiFaceLandmarks(request: Request): Promise<Response> {
  try {
    const body = (await readBody(request)) as { dataUrl?: string };
    const dataUrl = String(body.dataUrl ?? "");
    const m = /^data:([^;]+);base64,(.+)$/.exec(dataUrl);
    if (!m) {
      return new Response(JSON.stringify({ detected: false }), {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }
    const result = await detectFaceKeypointsWithGemini({
      mimeType: m[1] ?? "image/jpeg",
      data: m[2] ?? "",
    });
    return new Response(JSON.stringify(result ?? { detected: false }), {
      status: 200,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  } catch {
    return new Response(JSON.stringify({ detected: false }), {
      status: 200,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
}

export async function handleApiAvatarGender(request: Request): Promise<Response> {
  try {
    const body = (await readBody(request)) as { dataUrl?: string };
    const dataUrl = String(body.dataUrl ?? "");
    const m = /^data:([^;]+);base64,(.+)$/.exec(dataUrl);
    if (!m) {
      return new Response(JSON.stringify({ gender: "female" }), {
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8" },
      });
    }
    const gender = await detectAvatarGenderWithGemini({
      mimeType: m[1] ?? "image/jpeg",
      data: m[2] ?? "",
    });
    return new Response(JSON.stringify({ gender }), {
      status: 200,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  } catch {
    return new Response(JSON.stringify({ gender: "female" }), {
      status: 200,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
}
