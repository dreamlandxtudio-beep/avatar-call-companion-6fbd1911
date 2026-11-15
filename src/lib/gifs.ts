// Pure TypeScript animated GIF89a + LZW generator & reaction GIF catalog (100% offline/reliable, zero broken URLs).

export type ReactionGif = {
  id: string;
  label: string;
  alt: string;
  url: string;
};

// Minimal crisp GIF89a LZW encoder for 8-color / 16-color indexed RGBA animation frames
function encodeGifFrames(
  width: number,
  height: number,
  framesRgba: Uint8ClampedArray[],
  delayCs = 9,
): string {
  // Build quantized 16-color global palette (3-bit RGB-ish / curated palette extracted cleanly)
  const palette: number[] = [
    0x12,
    0x12,
    0x12, // 0: deep charcoal bg
    0x1e,
    0x1e,
    0x1e, // 1: card surface
    0xd4,
    0xd4,
    0xd4, // 2: light silver primary
    0xe5,
    0xe5,
    0xe5, // 3: bright silver glow
    0xfa,
    0xfa,
    0xfa, // 4: crisp white
    0xfd,
    0xe0,
    0x47, // 5: warm gold
    0xfb,
    0x92,
    0x3c, // 6: warm orange
    0xf4,
    0x3f,
    0x5e, // 7: rose red
    0xa3,
    0xa3,
    0xa3, // 8: neutral silver
    0xa7,
    0x8b,
    0xfa, // 9: soft violet
    0x8a,
    0x8a,
    0x8a, // 10: muted gray
    0x52,
    0x52,
    0x52, // 11: dark gray
    0x85,
    0x4d,
    0x0e, // 12: espresso brown
    0xfd,
    0xba,
    0x74, // 13: peach skin
    0x22,
    0xc5,
    0x5e, // 14: emerald
    0x2b,
    0x2b,
    0x2b, // 15: charcoal slate
  ];

  const nearestColor = (r: number, g: number, b: number): number => {
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < 16; i++) {
      const pr = palette[i * 3]!;
      const pg = palette[i * 3 + 1]!;
      const pb = palette[i * 3 + 2]!;
      const d = (r - pr) * (r - pr) + (g - pg) * (g - pg) + (b - pb) * (b - pb);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return best;
  };

  const out: number[] = [];
  const pushStr = (s: string) => {
    for (let i = 0; i < s.length; i++) out.push(s.charCodeAt(i));
  };
  const pushU16 = (v: number) => {
    out.push(v & 0xff, (v >> 8) & 0xff);
  };

  // Header
  pushStr("GIF89a");
  pushU16(width);
  pushU16(height);
  // Global Color Table Flag (1), Color Res (7 -> 3), Sort (0), Size of GCT (3 -> 2^(3+1) = 16 colors)
  out.push(0xf3, 0x00, 0x00);
  for (let i = 0; i < palette.length; i++) out.push(palette[i]!);

  // Netscape 2.0 infinite loop extension
  out.push(0x21, 0xff, 0x0b);
  pushStr("NETSCAPE2.0");
  out.push(0x03, 0x01, 0x00, 0x00, 0x00);

  const lzwCompress = (indices: Uint8Array): number[] => {
    const minCodeSize = 4;
    const clearCode = 1 << minCodeSize;
    const eoiCode = clearCode + 1;
    const bytes: number[] = [minCodeSize];
    let subBlock: number[] = [];

    let bitBuf = 0;
    let bitCount = 0;
    const writeBits = (code: number, size: number) => {
      bitBuf |= code << bitCount;
      bitCount += size;
      while (bitCount >= 8) {
        subBlock.push(bitBuf & 0xff);
        bitBuf >>= 8;
        bitCount -= 8;
        if (subBlock.length === 255) {
          bytes.push(255, ...subBlock);
          subBlock = [];
        }
      }
    };

    let codeSize = minCodeSize + 1;
    let nextCode = eoiCode + 1;
    let dict = new Map<string, number>();

    const resetDict = () => {
      dict = new Map();
      codeSize = minCodeSize + 1;
      nextCode = eoiCode + 1;
    };

    writeBits(clearCode, codeSize);
    resetDict();

    let prefix = String(indices[0] ?? 0);
    for (let i = 1; i < indices.length; i++) {
      const k = String(indices[i]!);
      const combined = `${prefix},${k}`;
      if (dict.has(combined)) {
        prefix = combined;
      } else {
        const outCode = prefix.includes(",") ? dict.get(prefix)! : Number(prefix);
        writeBits(outCode, codeSize);
        if (nextCode < 4096) {
          dict.set(combined, nextCode++);
          if (nextCode > 1 << codeSize && codeSize < 12) {
            codeSize++;
          }
        } else {
          writeBits(clearCode, codeSize);
          resetDict();
        }
        prefix = k;
      }
    }

    const lastCode = prefix.includes(",") ? dict.get(prefix)! : Number(prefix);
    writeBits(lastCode, codeSize);
    writeBits(eoiCode, codeSize);
    if (bitCount > 0) subBlock.push(bitBuf & 0xff);
    if (subBlock.length > 0) bytes.push(subBlock.length, ...subBlock);
    bytes.push(0x00);
    return bytes;
  };

  for (const rgba of framesRgba) {
    // Graphic Control Extension
    out.push(0x21, 0xf9, 0x04, 0x04);
    pushU16(delayCs);
    out.push(0x00, 0x00);

    // Image Descriptor
    out.push(0x2c);
    pushU16(0);
    pushU16(0);
    pushU16(width);
    pushU16(height);
    out.push(0x00);

    const indices = new Uint8Array(width * height);
    for (let p = 0; p < indices.length; p++) {
      const off = p * 4;
      indices[p] = nearestColor(rgba[off]!, rgba[off + 1]!, rgba[off + 2]!);
    }
    const lzw = lzwCompress(indices);
    for (const b of lzw) out.push(b);
  }

  out.push(0x3b);
  const bin = new Uint8Array(out);
  let str = "";
  for (let i = 0; i < bin.length; i++) str += String.fromCharCode(bin[i]!);
  return `data:image/gif;base64,${btoa(str)}`;
}

type DrawFn = (ctx: CanvasRenderingContext2D, w: number, h: number, t: number) => void;

const REACTION_SPECS: Array<{
  id: string;
  label: string;
  alt: string;
  draw: DrawFn;
}> = [
  {
    id: "saludo",
    label: "Saludo",
    alt: "GIF animado saludando",
    draw: (ctx, w, h, t) => {
      const ang = Math.sin(t * Math.PI * 2) * 0.42;
      ctx.save();
      ctx.translate(w * 0.5, h * 0.58);
      ctx.rotate(ang);
      ctx.fillStyle = "#fdba74";
      ctx.beginPath();
      ctx.roundRect(-12, -16, 24, 26, 10);
      ctx.fill();
      for (const fx of [-9, -3, 3, 9]) {
        ctx.beginPath();
        ctx.roundRect(fx - 2.5, -28, 5, 16, 2.5);
        ctx.fill();
      }
      ctx.restore();
      ctx.fillStyle = "#e5e5e5";
      const s = 3 + Math.abs(Math.sin(t * Math.PI * 2)) * 3;
      ctx.beginPath();
      ctx.arc(w * 0.24, h * 0.25, s, 0, Math.PI * 2);
      ctx.arc(w * 0.78, h * 0.24, s * 0.8, 0, Math.PI * 2);
      ctx.fill();
    },
  },
  {
    id: "aplauso",
    label: "Aplauso",
    alt: "GIF animado de aplausos",
    draw: (ctx, w, h, t) => {
      const clap = Math.abs(Math.sin(t * Math.PI * 4));
      const sep = 4 + clap * 14;
      ctx.fillStyle = "#fdba74";
      ctx.save();
      ctx.translate(w * 0.5 - sep, h * 0.55);
      ctx.rotate(-0.28 + (1 - clap) * 0.2);
      ctx.beginPath();
      ctx.roundRect(-9, -18, 18, 30, 8);
      ctx.fill();
      ctx.restore();
      ctx.save();
      ctx.translate(w * 0.5 + sep, h * 0.55);
      ctx.rotate(0.28 - (1 - clap) * 0.2);
      ctx.beginPath();
      ctx.roundRect(-9, -18, 18, 30, 8);
      ctx.fill();
      ctx.restore();
      if (clap < 0.35) {
        ctx.strokeStyle = "#fde047";
        ctx.lineWidth = 2.5;
        for (let i = 0; i < 5; i++) {
          const a = -Math.PI * 0.8 + i * 0.4;
          ctx.beginPath();
          ctx.moveTo(w * 0.5 + Math.cos(a) * 10, h * 0.32 + Math.sin(a) * 10);
          ctx.lineTo(w * 0.5 + Math.cos(a) * 20, h * 0.32 + Math.sin(a) * 20);
          ctx.stroke();
        }
      }
    },
  },
  {
    id: "risa",
    label: "Risa",
    alt: "GIF animado riendo",
    draw: (ctx, w, h, t) => {
      const dy = Math.sin(t * Math.PI * 4) * 3.5;
      ctx.save();
      ctx.translate(w * 0.5, h * 0.52 + dy);
      ctx.fillStyle = "#fde047";
      ctx.beginPath();
      ctx.arc(0, 0, 22, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#10161c";
      ctx.lineWidth = 2.8;
      for (const ex of [-8, 8]) {
        ctx.beginPath();
        ctx.arc(ex, -4, 4, Math.PI, 0);
        ctx.stroke();
      }
      ctx.fillStyle = "#10161c";
      ctx.beginPath();
      ctx.arc(0, 4, 11, 0, Math.PI);
      ctx.fill();
      ctx.fillStyle = "#f43f5e";
      ctx.beginPath();
      ctx.arc(0, 12, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    },
  },
  {
    id: "genial",
    label: "Genial",
    alt: "GIF animado diciendo genial",
    draw: (ctx, w, h, t) => {
      const sc = 0.92 + Math.sin(t * Math.PI * 2) * 0.12;
      ctx.save();
      ctx.translate(w * 0.5, h * 0.5);
      ctx.scale(sc, sc);
      ctx.fillStyle = "#d4d4d4";
      ctx.beginPath();
      ctx.arc(0, 0, 22, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#121212";
      ctx.lineWidth = 4.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(-9, 1);
      ctx.lineTo(-2, 8);
      ctx.lineTo(10, -7);
      ctx.stroke();
      ctx.restore();
    },
  },
  {
    id: "sorpresa",
    label: "Sorpresa",
    alt: "GIF animado de sorpresa",
    draw: (ctx, w, h, t) => {
      const pulse = 1 + Math.sin(t * Math.PI * 2) * 0.14;
      ctx.save();
      ctx.translate(w * 0.5, h * 0.5);
      ctx.scale(pulse, pulse);
      ctx.fillStyle = "#fb923c";
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2 + t * 0.8;
        ctx.beginPath();
        ctx.moveTo(Math.cos(a) * 14, Math.sin(a) * 14);
        ctx.lineTo(Math.cos(a + 0.18) * 25, Math.sin(a + 0.18) * 25);
        ctx.lineTo(Math.cos(a + 0.36) * 14, Math.sin(a + 0.36) * 14);
        ctx.fill();
      }
      ctx.fillStyle = "#f8fafc";
      ctx.beginPath();
      ctx.roundRect(-3, -12, 6, 15, 3);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(0, 9, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    },
  },
  {
    id: "cafe",
    label: "Café",
    alt: "GIF animado de café caliente",
    draw: (ctx, w, h, t) => {
      ctx.fillStyle = "#f8fafc";
      ctx.beginPath();
      ctx.roundRect(w * 0.33, h * 0.42, 30, 24, 6);
      ctx.fill();
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.arc(w * 0.65, h * 0.54, 6.5, -Math.PI * 0.5, Math.PI * 0.5);
      ctx.stroke();
      ctx.fillStyle = "#854d0e";
      ctx.fillRect(w * 0.36, h * 0.45, 24, 5);
      ctx.strokeStyle = "#d4d4d4";
      ctx.lineWidth = 2.5;
      for (const sx of [w * 0.41, w * 0.5, w * 0.58]) {
        const wave = Math.sin(t * Math.PI * 2 + sx) * 3;
        ctx.beginPath();
        ctx.moveTo(sx, h * 0.38);
        ctx.quadraticCurveTo(sx + wave, h * 0.28, sx - wave, h * 0.18);
        ctx.stroke();
      }
    },
  },
  {
    id: "pensando",
    label: "Pensando",
    alt: "GIF animado pensando",
    draw: (ctx, w, h, t) => {
      ctx.fillStyle = "#1e1e1e";
      ctx.beginPath();
      ctx.roundRect(w * 0.22, h * 0.3, w * 0.56, h * 0.38, 12);
      ctx.fill();
      ctx.strokeStyle = "#d4d4d4";
      ctx.lineWidth = 2;
      ctx.stroke();
      for (let i = 0; i < 3; i++) {
        const jump = Math.max(0, Math.sin(t * Math.PI * 2 - i * 1.1)) * 6;
        ctx.fillStyle = i === 1 ? "#fafafa" : "#a3a3a3";
        ctx.beginPath();
        ctx.arc(w * (0.37 + i * 0.13), h * 0.5 - jump, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    },
  },
  {
    id: "animo",
    label: "Ánimo",
    alt: "GIF animado de ánimo",
    draw: (ctx, w, h, t) => {
      const beat = 1 + Math.pow(Math.abs(Math.sin(t * Math.PI * 2)), 2) * 0.22;
      ctx.save();
      ctx.translate(w * 0.5, h * 0.52);
      ctx.scale(beat, beat);
      ctx.fillStyle = "#f43f5e";
      ctx.beginPath();
      ctx.moveTo(0, 14);
      ctx.bezierCurveTo(-22, 0, -20, -20, 0, -8);
      ctx.bezierCurveTo(20, -20, 22, 0, 0, 14);
      ctx.fill();
      ctx.restore();
    },
  },
];

let cachedPack: ReactionGif[] | null = null;

export function getReactionGifs(): ReactionGif[] {
  if (cachedPack) return cachedPack;
  if (typeof document === "undefined") return [];
  const W = 96;
  const H = 72;
  const FRAMES = 8;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [];

  cachedPack = REACTION_SPECS.map((spec) => {
    const frames: Uint8ClampedArray[] = [];
    for (let f = 0; f < FRAMES; f++) {
      ctx.fillStyle = "#121212";
      ctx.fillRect(0, 0, W, H);
      spec.draw(ctx, W, H, f / FRAMES);
      frames.push(ctx.getImageData(0, 0, W, H).data);
    }
    return {
      id: spec.id,
      label: spec.label,
      alt: spec.alt,
      url: encodeGifFrames(W, H, frames, 10),
    };
  });
  return cachedPack;
}

export function resolveCharacterGif(keyword?: string): ReactionGif | undefined {
  if (!keyword) return undefined;
  const k = keyword.trim().toLowerCase();
  const list = getReactionGifs();
  return list.find((g) => g.id === k || g.label.toLowerCase() === k);
}

export function readGifFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("No se pudo leer el GIF"));
    };
    reader.onerror = () => reject(new Error("No se pudo leer el GIF"));
    reader.readAsDataURL(file);
  });
}
