import { resolveVisionFileset } from "./vision-runtime";
// Face, iris, hand & portrait segmentation detection with MediaPipe + Gemini Vision anatomical fallback.
import { fitCanonicalLandmarks, type FaceKeypoints, type HandKeypoints } from "./canonicalFace";

export type { HandKeypoints };

export type Axis2D = {
  ux: number;
  uy: number;
  vx: number;
  vy: number;
};

export type FaceColors = {
  skin: [number, number, number];
  lip: [number, number, number];
  eyelid: [number, number, number];
  enamel: [number, number, number];
  shirt: [number, number, number];
  scleraRight: [number, number, number];
  scleraLeft: [number, number, number];
};

export type EyeCurveProfile = {
  upper: [number, number, number, number];
  lower: [number, number, number, number];
  halfW: number;
  scaleY: number;
};

export type FacePts = {
  landmarks: Array<[number, number, number]>;
  aspect: number;
  faceTris: Uint16Array;
  ovalLoop: number[];
  center: [number, number];
  faceAxis: Axis2D;
  mouthAxis: Axis2D;
  leftEyeAxis: Axis2D;
  rightEyeAxis: Axis2D;
  faceW: number;
  faceH: number;
  mouthW: number;
  mouthH: number;
  jawSpan: number;
  restMouthRatio: number;
  hasVisibleTeeth: boolean;
  isMouthParted: boolean;
  restTeethDepth: number;
  yaw: number;
  stylized: boolean;
  rightPupil: [number, number];
  leftPupil: [number, number];
  rightEyeCenter: [number, number];
  leftEyeCenter: [number, number];
  rightEyeRadius: [number, number];
  leftEyeRadius: [number, number];
  rightEyeCurve: EyeCurveProfile;
  leftEyeCurve: EyeCurveProfile;
  rightIrisRadius: number;
  leftIrisRadius: number;
  rightEar: [number, number];
  leftEar: [number, number];
  hairTop: [number, number];
  hairLeft: [number, number];
  hairRight: [number, number];
  neckBase: [number, number];
  rightShoulder: [number, number];
  leftShoulder: [number, number];
  hands: HandKeypoints[];
  colors: FaceColors;
  segMask?: Float32Array | undefined;
  segWidth?: number | undefined;
  segHeight?: number | undefined;
};

export const INNER_UPPER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];
export const INNER_LOWER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];
export const INNER_LIP_LOOP = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
];

export const OUTER_UPPER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291];
export const OUTER_LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291];

export const RIGHT_EYE_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133];
export const RIGHT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133];
export const LEFT_EYE_UPPER = [362, 398, 384, 385, 386, 387, 388, 466, 263];
export const LEFT_EYE_LOWER = [362, 382, 381, 380, 374, 373, 390, 249, 263];

export const RIGHT_EYE_LOOP = [
  33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
];
export const LEFT_EYE_LOOP = [
  362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,
];

const cache = new Map<string, FacePts | null>();
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let lm: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let fd: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let hl: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let seg: any = null;
let cachedFaceTris: Uint16Array | null = null;
let cachedOvalLoop: number[] | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildCanonicalTopology(FaceLandmarkerCls: any) {
  if (!cachedFaceTris) {
    const adj = new Map<number, Set<number>>();
    const edges = FaceLandmarkerCls.FACE_LANDMARKS_TESSELATION as Array<{
      start: number;
      end: number;
    }>;
    for (const { start, end } of edges) {
      if (!adj.has(start)) adj.set(start, new Set());
      if (!adj.has(end)) adj.set(end, new Set());
      adj.get(start)!.add(end);
      adj.get(end)!.add(start);
    }
    const tris: number[] = [];
    for (const [u, nu] of adj) {
      for (const v of nu) {
        if (v <= u) continue;
        const nv = adj.get(v)!;
        for (const w of nu) {
          if (w <= v) continue;
          if (nv.has(w)) tris.push(u, v, w);
        }
      }
    }
    cachedFaceTris = new Uint16Array(tris);
  }
  if (!cachedOvalLoop) {
    const ovalEdges = FaceLandmarkerCls.FACE_LANDMARKS_FACE_OVAL as Array<{
      start: number;
      end: number;
    }>;
    cachedOvalLoop = ovalEdges.map((e) => e.start);
  }
  return { faceTris: cachedFaceTris, ovalLoop: cachedOvalLoop };
}

function makeAxis(ax: number, ay: number, bx: number, by: number, aspect: number): Axis2D {
  const dx = (bx - ax) * aspect;
  const dy = by - ay;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  let vx = -uy;
  let vy = ux;
  if (vy < 0) {
    vx = -vx;
    vy = -vy;
  }
  return { ux, uy, vx, vy };
}

function extractEyeCurveProfile(
  landmarks: Array<[number, number, number]>,
  upperIndices: number[],
  lowerIndices: number[],
  axis: Axis2D,
  aspect: number,
): { center: [number, number]; curve: EyeCurveProfile } {
  const c0 = landmarks[upperIndices[0]!]!;
  const c1 = landmarks[upperIndices[upperIndices.length - 1]!]!;
  const midX = (c0[0] + c1[0]) * 0.5;
  const midY = (c0[1] + c1[1]) * 0.5;
  const halfW = Math.max(0.008, Math.hypot((c1[0] - c0[0]) * aspect, c1[1] - c0[1]) * 0.5);

  const projectPts = (idxs: number[]) =>
    idxs
      .map((idx) => {
        const pt = landmarks[idx]!;
        const dx = (pt[0] - midX) * aspect;
        const dy = pt[1] - midY;
        const t = (dx * axis.ux + dy * axis.uy) / halfW;
        const v = dx * axis.vx + dy * axis.vy;
        return { t, v };
      })
      .sort((a, b) => a.t - b.t);

  const up = projectPts(upperIndices);
  const lo = projectPts(lowerIndices);

  const evalSorted = (arr: Array<{ t: number; v: number }>, qt: number): number => {
    if (arr.length === 0) return 0;
    if (qt <= arr[0]!.t) return arr[0]!.v;
    if (qt >= arr[arr.length - 1]!.t) return arr[arr.length - 1]!.v;
    for (let i = 0; i < arr.length - 1; i++) {
      const a = arr[i]!;
      const b = arr[i + 1]!;
      if (qt >= a.t && qt <= b.t) {
        const f = (qt - a.t) / Math.max(1e-5, b.t - a.t);
        return a.v + (b.v - a.v) * f;
      }
    }
    return 0;
  };

  const samples = [-0.6, -0.2, 0.2, 0.6];
  const rawUpper = samples.map((qt) => Math.min(-0.001, evalSorted(up, qt)));
  const rawLower = samples.map((qt) => Math.max(0.001, evalSorted(lo, qt)));

  let maxAbs = 0.006;
  for (const v of rawUpper) maxAbs = Math.max(maxAbs, Math.abs(v));
  for (const v of rawLower) maxAbs = Math.max(maxAbs, Math.abs(v));

  return {
    center: [midX, midY],
    curve: {
      upper: [
        rawUpper[0]! / maxAbs,
        rawUpper[1]! / maxAbs,
        rawUpper[2]! / maxAbs,
        rawUpper[3]! / maxAbs,
      ],
      lower: [
        rawLower[0]! / maxAbs,
        rawLower[1]! / maxAbs,
        rawLower[2]! / maxAbs,
        rawLower[3]! / maxAbs,
      ],
      halfW,
      scaleY: maxAbs,
    },
  };
}

function pointInPoly(px: number, py: number, poly: Array<[number, number]>): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i]![0];
    const yi = poly[i]![1];
    const xj = poly[j]![0];
    const yj = poly[j]![1];
    const intersect = yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

type MouthAnalysis = {
  colors: FaceColors;
  hasVisibleTeeth: boolean;
  isMouthParted: boolean;
  restTeethDepth: number;
};

function analyzePhotoFeatures(
  img: HTMLImageElement,
  pts: Array<[number, number, number]>,
  restMouthRatio: number,
  rightPupil: [number, number],
  leftPupil: [number, number],
  rightEyeRx: number,
  leftEyeRx: number,
): MouthAnalysis {
  const fallback: FaceColors = {
    skin: [215, 175, 155],
    lip: [165, 92, 92],
    eyelid: [195, 155, 138],
    enamel: [232, 226, 218],
    shirt: [45, 68, 88],
    scleraRight: [242, 240, 238],
    scleraLeft: [242, 240, 238],
  };

  try {
    const S = 224;
    const c = document.createElement("canvas");
    c.width = S;
    c.height = S;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      return {
        colors: fallback,
        hasVisibleTeeth: restMouthRatio > 0.08,
        isMouthParted: restMouthRatio > 0.08,
        restTeethDepth: Math.abs(pts[14]![1] - pts[13]![1]),
      };
    }
    ctx.drawImage(img, 0, 0, S, S);
    const data = ctx.getImageData(0, 0, S, S).data;

    const avgAt = (indices: number[]): [number, number, number] => {
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (const idx of indices) {
        const pt = pts[idx];
        if (!pt) continue;
        const px = Math.min(S - 1, Math.max(0, Math.round(pt[0] * (S - 1))));
        const py = Math.min(S - 1, Math.max(0, Math.round(pt[1] * (S - 1))));
        const off = (py * S + px) * 4;
        r += data[off]!;
        g += data[off + 1]!;
        b += data[off + 2]!;
        n++;
      }
      return n > 0 ? [r / n, g / n, b / n] : fallback.skin;
    };

    const sampleEyeSclera = (
      eyeLoop: number[],
      pupil: [number, number],
      rx: number,
    ): [number, number, number] => {
      const imgW = img.naturalWidth || img.width || 512;
      const imgH = img.naturalHeight || img.height || 512;
      let eMinX = 1;
      let eMaxX = 0;
      let eMinY = 1;
      let eMaxY = 0;
      for (const idx of eyeLoop) {
        const pt = pts[idx];
        if (!pt) continue;
        if (pt[0] < eMinX) eMinX = pt[0];
        if (pt[0] > eMaxX) eMaxX = pt[0];
        if (pt[1] < eMinY) eMinY = pt[1];
        if (pt[1] > eMaxY) eMaxY = pt[1];
      }
      const ew = Math.max(0.012, eMaxX - eMinX);
      const eh = Math.max(0.008, eMaxY - eMinY);
      const cropW = 64;
      const cropH = 32;
      const ec = document.createElement("canvas");
      ec.width = cropW;
      ec.height = cropH;
      const ectx = ec.getContext("2d", { willReadFrequently: true });
      if (!ectx) return [245, 245, 243];
      ectx.drawImage(img, eMinX * imgW, eMinY * imgH, ew * imgW, eh * imgH, 0, 0, cropW, cropH);
      const edata = ectx.getImageData(0, 0, cropW, cropH).data;
      const eyePoly: Array<[number, number]> = eyeLoop.map((idx) => [
        ((pts[idx]![0] - eMinX) / ew) * cropW,
        ((pts[idx]![1] - eMinY) / eh) * cropH,
      ]);

      const candidates: Array<{ score: number; rgb: [number, number, number] }> = [];
      for (let py = 0; py < cropH; py++) {
        for (let px = 0; px < cropW; px++) {
          if (!pointInPoly(px + 0.5, py + 0.5, eyePoly)) continue;
          const nx = eMinX + ((px + 0.5) / cropW) * ew;
          const ny = eMinY + ((py + 0.5) / cropH) * eh;
          // Exclude central pupil/iris region so we measure genuine sclera white
          const dPupil = Math.hypot(nx - pupil[0], ny - pupil[1]);
          if (dPupil < rx * 0.42) continue;

          const off = (py * cropW + px) * 4;
          const r = edata[off]!;
          const g = edata[off + 1]!;
          const b = edata[off + 2]!;
          const luma = 0.299 * r + 0.587 * g + 0.114 * b;
          const maxC = Math.max(r, g, b);
          const minC = Math.min(r, g, b);
          const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
          const redness = r - (g + b) * 0.5;
          if (luma < 142 || sat > 0.19 || redness > 9) continue;
          const score = luma - redness * 3.2 - sat * 110;
          candidates.push({ score, rgb: [r, g, b] });
        }
      }

      if (candidates.length === 0) {
        return [238, 236, 234];
      }
      candidates.sort((a, b) => b.score - a.score);
      // Sample the natural mid-upper clean sclera band (26%-64% percentile)
      const startIdx = Math.min(candidates.length - 1, Math.floor(candidates.length * 0.26));
      const endIdx = Math.min(
        candidates.length,
        Math.max(startIdx + 4, Math.floor(candidates.length * 0.64)),
      );
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let count = 0;
      for (let i = startIdx; i < endIdx; i++) {
        const cRGB = candidates[i]!.rgb;
        sr += cRGB[0];
        sg += cRGB[1];
        sb += cRGB[2];
        count++;
      }
      const n = Math.max(1, count);
      const avgR = sr / n;
      const avgG = sg / n;
      const avgB = sb / n;
      const cleanAvg = (avgG + avgB) * 0.5;
      return [
        Math.min(246, Math.round(Math.min(avgR, cleanAvg + 2))),
        Math.min(246, Math.round(avgG)),
        Math.min(246, Math.round(avgB)),
      ];
    };

    const skin = avgAt([118, 347, 50, 280, 195]);
    const lip = avgAt([0, 17, 37, 267, 84, 314]);
    const eyelid = avgAt([29, 30, 259, 260, 27, 257]);
    const forehead = avgAt([10, 151, 9]);
    const scleraRight = sampleEyeSclera(RIGHT_EYE_LOOP, rightPupil, rightEyeRx);
    const scleraLeft = sampleEyeSclera(LEFT_EYE_LOOP, leftPupil, leftEyeRx);

    let shirtR = 0;
    let shirtG = 0;
    let shirtB = 0;
    let shirtN = 0;
    for (let py = Math.round(S * 0.78); py < S; py += 4) {
      for (let px = Math.round(S * 0.25); px < Math.round(S * 0.75); px += 4) {
        const off = (py * S + px) * 4;
        shirtR += data[off]!;
        shirtG += data[off + 1]!;
        shirtB += data[off + 2]!;
        shirtN++;
      }
    }
    const shirt: [number, number, number] =
      shirtN > 0 ? [shirtR / shirtN, shirtG / shirtN, shirtB / shirtN] : [45, 68, 88];

    const imgW = img.naturalWidth || img.width || 512;
    const imgH = img.naturalHeight || img.height || 512;
    let minX = 1;
    let maxX = 0;
    let minY = 1;
    let maxY = 0;
    for (const idx of INNER_LIP_LOOP) {
      const pt = pts[idx]!;
      if (pt[0] < minX) minX = pt[0];
      if (pt[0] > maxX) maxX = pt[0];
      if (pt[1] < minY) minY = pt[1];
      if (pt[1] > maxY) maxY = pt[1];
    }

    const padX = Math.max(0.015, (maxX - minX) * 0.2);
    const padY = Math.max(0.015, (maxY - minY) * 0.35);
    const rx0 = Math.max(0, minX - padX);
    const ry0 = Math.max(0, minY - padY);
    const rw = Math.max(0.02, Math.min(1 - rx0, maxX - minX + padX * 2));
    const rh = Math.max(0.02, Math.min(1 - ry0, maxY - minY + padY * 2));

    const MW = 96;
    const MH = 48;
    const mc = document.createElement("canvas");
    mc.width = MW;
    mc.height = MH;
    const mctx = mc.getContext("2d", { willReadFrequently: true });

    let hasVisibleTeeth = false;
    let isMouthParted = restMouthRatio > 0.085;
    let teethR = 0;
    let teethG = 0;
    let teethB = 0;
    let teethCount = 0;
    let maxTeethNormY = 0;

    if (mctx) {
      mctx.drawImage(img, rx0 * imgW, ry0 * imgH, rw * imgW, rh * imgH, 0, 0, MW, MH);
      const mData = mctx.getImageData(0, 0, MW, MH).data;
      const poly: Array<[number, number]> = INNER_LIP_LOOP.map((idx) => [
        ((pts[idx]![0] - rx0) / rw) * MW,
        ((pts[idx]![1] - ry0) / rh) * MH,
      ]);

      const lipLuma = lip[0] * 0.299 + lip[1] * 0.587 + lip[2] * 0.114;
      let insideCount = 0;
      let darkCavityCount = 0;

      for (let y = 0; y < MH; y++) {
        for (let x = 0; x < MW; x++) {
          if (!pointInPoly(x + 0.5, y + 0.5, poly)) continue;
          insideCount++;
          const off = (y * MW + x) * 4;
          const r = mData[off]!;
          const g = mData[off + 1]!;
          const b = mData[off + 2]!;
          const luma = 0.299 * r + 0.587 * g + 0.114 * b;
          const maxC = Math.max(r, g, b);
          const minC = Math.min(r, g, b);
          const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
          const redness = r - (g + b) * 0.5;

          if (luma > Math.max(115, lipLuma * 1.08) && sat < 0.36 && redness < 32) {
            teethCount++;
            teethR += r;
            teethG += g;
            teethB += b;
            const normY = (y / MH) * rh + ry0 - pts[13]![1];
            if (normY > maxTeethNormY) maxTeethNormY = normY;
          } else if (luma < lipLuma * 0.58) {
            darkCavityCount++;
          }
        }
      }

      const teethFrac = insideCount > 0 ? teethCount / insideCount : 0;
      const cavityFrac = insideCount > 0 ? darkCavityCount / insideCount : 0;
      hasVisibleTeeth = restMouthRatio > 0.055 && (teethCount >= 20 || teethFrac >= 0.16);
      isMouthParted = hasVisibleTeeth || cavityFrac >= 0.2 || restMouthRatio > 0.085;
    }

    const skinLuma =
      (skin[0] * 0.299 +
        skin[1] * 0.587 +
        skin[2] * 0.114 +
        forehead[0] * 0.299 +
        forehead[1] * 0.587 +
        forehead[2] * 0.114) /
      2;
    const scale = Math.min(0.98, Math.max(0.76, 0.72 + (skinLuma / 255) * 0.28));
    const sampledEnamel: [number, number, number] =
      hasVisibleTeeth && teethCount > 0
        ? [teethR / teethCount, teethG / teethCount, teethB / teethCount]
        : [
            Math.min(244, Math.round(236 * scale + (skin[0] - skin[2]) * 0.06)),
            Math.min(240, Math.round(232 * scale + (skin[1] - skin[2]) * 0.04)),
            Math.min(234, Math.round(224 * scale)),
          ];
    const enamelLuma =
      sampledEnamel[0] * 0.299 + sampledEnamel[1] * 0.587 + sampledEnamel[2] * 0.114;
    const boost = enamelLuma < 200 ? 205 / Math.max(80, enamelLuma) : 1;
    const enamel: [number, number, number] = [
      Math.min(246, Math.round(sampledEnamel[0] * boost)),
      Math.min(242, Math.round(sampledEnamel[1] * boost)),
      Math.min(236, Math.round(sampledEnamel[2] * boost)),
    ];

    const lipGapY = Math.max(0.01, Math.abs(pts[14]![1] - pts[13]![1]));
    const restTeethDepth = Math.max(lipGapY, maxTeethNormY);

    return {
      colors: { skin, lip, eyelid, enamel, shirt, scleraRight, scleraLeft },
      hasVisibleTeeth,
      isMouthParted,
      restTeethDepth,
    };
  } catch {
    return {
      colors: fallback,
      hasVisibleTeeth: restMouthRatio > 0.08,
      isMouthParted: restMouthRatio > 0.08,
      restTeethDepth: Math.abs(pts[14]![1] - pts[13]![1]),
    };
  }
}

let tfliteFilterInstalled = false;
function installTfliteConsoleFilter() {
  if (tfliteFilterInstalled || typeof window === "undefined") return;
  tfliteFilterInstalled = true;
  const origError = console.error.bind(console);
  console.error = (...args: unknown[]) => {
    const first = typeof args[0] === "string" ? args[0] : "";
    if (
      /^INFO:\s/i.test(first) ||
      /Created TensorFlow Lite XNNPACK delegate/i.test(first) ||
      /^I\d{4}\s+.*(tflite|xnnpack|mediapipe|face_landmarker|face_detector|hand_landmarker|image_segmenter)/i.test(
        first,
      )
    ) {
      console.info(...args);
      return;
    }
    origError(...args);
  };
}

let visionPromise: Promise<{
  landmarker: unknown;
  detector: unknown;
  handLandmarker: unknown;
  segmenter: unknown;
}> | null = null;

let visionRetryAfter = 0;

async function getVisionModels() {
  if (lm && fd) {
    return { landmarker: lm, detector: fd, handLandmarker: hl, segmenter: seg };
  }
  if (Date.now() < visionRetryAfter) {
    throw new Error("Vision model cooldown");
  }
  installTfliteConsoleFilter();
  const { FaceDetector, FaceLandmarker, HandLandmarker, ImageSegmenter } =
    await import("@mediapipe/tasks-vision");

  const initFromBase = async () => {
    const fs = await resolveVisionFileset();
    const [landmarker, detector, handLandmarker, segmenter] = await Promise.all([
      FaceLandmarker.createFromOptions(fs, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "CPU",
        },
        runningMode: "IMAGE",
        numFaces: 1,
        minFaceDetectionConfidence: 0.12,
        minFacePresenceConfidence: 0.12,
        minTrackingConfidence: 0.12,
      }),
      FaceDetector.createFromOptions(fs, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
          delegate: "CPU",
        },
        runningMode: "IMAGE",
        minDetectionConfidence: 0.15,
      }).catch(() => null),
      HandLandmarker.createFromOptions(fs, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "CPU",
        },
        runningMode: "IMAGE",
        numHands: 2,
        minHandDetectionConfidence: 0.2,
        minHandPresenceConfidence: 0.2,
      }).catch(() => null),
      ImageSegmenter.createFromOptions(fs, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
          delegate: "CPU",
        },
        runningMode: "IMAGE",
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      }).catch(() => null),
    ]);
    return { landmarker, detector, handLandmarker, segmenter };
  };

  if (!visionPromise) {
    visionPromise = Promise.race([
      initFromBase(),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Vision model timeout")), 5500),
      ),
    ]).catch((err) => {
      visionPromise = null;
      visionRetryAfter = Date.now() + 60000;
      throw err;
    });
  }
  const res = await visionPromise;
  lm = res.landmarker;
  fd = res.detector;
  hl = res.handLandmarker;
  seg = res.segmenter;
  return res;
}

function toDetectionCanvas(img: HTMLImageElement, maxDim = 640): HTMLCanvasElement {
  const w = img.naturalWidth || img.width || 512;
  const h = img.naturalHeight || img.height || 512;
  const scale = Math.min(1, maxDim / Math.max(w, h));
  const c = document.createElement("canvas");
  c.width = Math.max(1, Math.round(w * scale));
  c.height = Math.max(1, Math.round(h * scale));
  const ctx = c.getContext("2d");
  ctx?.drawImage(img, 0, 0, c.width, c.height);
  return c;
}

async function detectWithGeminiFallback(
  img: HTMLImageElement,
  key: string,
): Promise<FaceKeypoints | null> {
  const storageKey = "yume.fk.v2." + key;
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      const parsed = JSON.parse(saved) as FaceKeypoints;
      if (parsed?.rightEye && parsed?.leftEye && parsed?.mouthCenter) return parsed;
    }
  } catch {
    /* ignore */
  }

  try {
    const c = toDetectionCanvas(img, 512);
    const dataUrl = c.toDataURL("image/jpeg", 0.85);
    const res = await fetch("/api/face-landmarks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataUrl }),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as { detected?: boolean } & Partial<FaceKeypoints>;
    if (!data.detected || !data.rightEye || !data.leftEye || !data.mouthCenter || !data.noseTip) {
      return null;
    }
    const kp: FaceKeypoints = {
      rightEye: data.rightEye,
      leftEye: data.leftEye,
      rightPupil: data.rightPupil,
      leftPupil: data.leftPupil,
      noseTip: data.noseTip,
      mouthCenter: data.mouthCenter,
      mouthLeft: data.mouthLeft,
      mouthRight: data.mouthRight,
      chin: data.chin,
      forehead: data.forehead,
      rightEar: data.rightEar,
      leftEar: data.leftEar,
      hairTop: data.hairTop,
      hairLeft: data.hairLeft,
      hairRight: data.hairRight,
      neckBase: data.neckBase,
      rightShoulder: data.rightShoulder,
      leftShoulder: data.leftShoulder,
      hands: data.hands,
      stylized: !!data.stylized,
    };
    try {
      localStorage.setItem(storageKey, JSON.stringify(kp));
    } catch {
      /* ignore */
    }
    return kp;
  } catch {
    return null;
  }
}

function buildFacePts(
  img: HTMLImageElement,
  landmarks: Array<[number, number, number]>,
  faceTris: Uint16Array,
  ovalLoop: number[],
  aspect: number,
  stylized: boolean,
  extras?: {
    kp?: FaceKeypoints | undefined;
    hands?: HandKeypoints[] | undefined;
    segMask?: Float32Array | undefined;
    segWidth?: number | undefined;
    segHeight?: number | undefined;
  },
): FacePts {
  const distA = (a: number, b: number) => {
    const pa = landmarks[a]!;
    const pb = landmarks[b]!;
    return Math.hypot((pb[0] - pa[0]) * aspect, pb[1] - pa[1]);
  };

  const mouthW = Math.max(0.05, distA(61, 291));
  const restGap = distA(13, 14);
  const mouthH = Math.max(0.014, distA(0, 17));
  const restMouthRatio = stylized ? 0 : restGap / mouthW;

  const dLeft = distA(1, 234);
  const dRight = distA(1, 454);
  const yaw = (dLeft - dRight) / Math.max(0.05, dLeft + dRight);

  const faceAxis = makeAxis(
    landmarks[33]![0],
    landmarks[33]![1],
    landmarks[263]![0],
    landmarks[263]![1],
    aspect,
  );
  const mouthAxis = makeAxis(
    landmarks[61]![0],
    landmarks[61]![1],
    landmarks[291]![0],
    landmarks[291]![1],
    aspect,
  );
  const rightEyeAxis = makeAxis(
    landmarks[33]![0],
    landmarks[33]![1],
    landmarks[133]![0],
    landmarks[133]![1],
    aspect,
  );
  const leftEyeAxis = makeAxis(
    landmarks[362]![0],
    landmarks[362]![1],
    landmarks[263]![0],
    landmarks[263]![1],
    aspect,
  );

  const rEyeShape = extractEyeCurveProfile(
    landmarks,
    RIGHT_EYE_UPPER,
    RIGHT_EYE_LOWER,
    rightEyeAxis,
    aspect,
  );
  const lEyeShape = extractEyeCurveProfile(
    landmarks,
    LEFT_EYE_UPPER,
    LEFT_EYE_LOWER,
    leftEyeAxis,
    aspect,
  );

  const faceW = Math.max(0.12, distA(234, 454));
  const faceH = Math.max(0.15, distA(10, 152));
  const mouthMidY = (landmarks[13]![1] + landmarks[14]![1]) / 2;
  const mouthMidX = (landmarks[61]![0] + landmarks[291]![0]) / 2;
  const jawSpan = Math.max(
    0.08,
    Math.hypot((landmarks[152]![0] - mouthMidX) * aspect, landmarks[152]![1] - mouthMidY),
  );

  // Extract eye centers & iris radii (supports MediaPipe 478 iris landmarks 468..477 when present)
  const rightEyeCenter: [number, number] =
    landmarks.length >= 478
      ? [landmarks[468]![0], landmarks[468]![1]]
      : (extras?.kp?.rightPupil ?? rEyeShape.center);
  const leftEyeCenter: [number, number] =
    landmarks.length >= 478
      ? [landmarks[473]![0], landmarks[473]![1]]
      : (extras?.kp?.leftPupil ?? lEyeShape.center);

  const rIrisBase =
    landmarks.length >= 478
      ? Math.max(
          0.006,
          Math.hypot(
            (landmarks[469]![0] - landmarks[471]![0]) * aspect,
            landmarks[469]![1] - landmarks[471]![1],
          ) * 0.52,
        )
      : Math.max(0.008, rEyeShape.curve.halfW * (stylized ? 0.56 : 0.48));
  const lIrisBase =
    landmarks.length >= 478
      ? Math.max(
          0.006,
          Math.hypot(
            (landmarks[474]![0] - landmarks[476]![0]) * aspect,
            landmarks[474]![1] - landmarks[476]![1],
          ) * 0.52,
        )
      : Math.max(0.008, lEyeShape.curve.halfW * (stylized ? 0.56 : 0.48));
  const rIrisR = stylized ? Math.max(rIrisBase, rEyeShape.curve.halfW * 0.54) : rIrisBase;
  const lIrisR = stylized ? Math.max(lIrisBase, lEyeShape.curve.halfW * 0.54) : lIrisBase;

  const rEyeRx = Math.max(0.012, rEyeShape.curve.halfW / aspect);
  const rEyeRy = Math.max(0.007, rEyeShape.curve.scaleY * (stylized ? 1.38 : 1.0));
  const lEyeRx = Math.max(0.012, lEyeShape.curve.halfW / aspect);
  const lEyeRy = Math.max(0.007, lEyeShape.curve.scaleY * (stylized ? 1.38 : 1.0));

  const chinPt = landmarks[152]!;
  const foreheadPt = landmarks[10]!;
  const rCheekEdge = landmarks[234]!;
  const lCheekEdge = landmarks[454]!;

  const rightEar: [number, number] = extras?.kp?.rightEar ?? [
    Math.max(0.02, rCheekEdge[0] - (faceW * 0.04) / aspect),
    rCheekEdge[1],
  ];
  const leftEar: [number, number] = extras?.kp?.leftEar ?? [
    Math.min(0.98, lCheekEdge[0] + (faceW * 0.04) / aspect),
    lCheekEdge[1],
  ];
  const hairTop: [number, number] = extras?.kp?.hairTop ?? [
    foreheadPt[0],
    Math.max(0.02, foreheadPt[1] - faceH * 0.34),
  ];
  const hairLeft: [number, number] = extras?.kp?.hairLeft ?? [
    Math.max(0.02, rCheekEdge[0] - (faceW * 0.16) / aspect),
    (foreheadPt[1] + chinPt[1]) * 0.5,
  ];
  const hairRight: [number, number] = extras?.kp?.hairRight ?? [
    Math.min(0.98, lCheekEdge[0] + (faceW * 0.16) / aspect),
    (foreheadPt[1] + chinPt[1]) * 0.5,
  ];
  const neckBase: [number, number] = extras?.kp?.neckBase ?? [
    chinPt[0],
    Math.min(0.96, chinPt[1] + faceH * 0.34),
  ];
  const rightShoulder: [number, number] = extras?.kp?.rightShoulder ?? [
    Math.max(0.04, neckBase[0] - (faceW * 0.92) / aspect),
    Math.min(0.98, neckBase[1] + faceH * 0.22),
  ];
  const leftShoulder: [number, number] = extras?.kp?.leftShoulder ?? [
    Math.min(0.96, neckBase[0] + (faceW * 0.92) / aspect),
    Math.min(0.98, neckBase[1] + faceH * 0.22),
  ];

  const hands: HandKeypoints[] = extras?.hands ?? extras?.kp?.hands ?? [];
  const analysis = analyzePhotoFeatures(
    img,
    landmarks,
    restMouthRatio,
    rightEyeCenter,
    leftEyeCenter,
    rEyeRx,
    lEyeRx,
  );

  return {
    landmarks,
    aspect,
    faceTris,
    ovalLoop,
    center: [landmarks[1]![0], landmarks[1]![1]],
    faceAxis,
    mouthAxis,
    leftEyeAxis,
    rightEyeAxis,
    faceW,
    faceH,
    mouthW,
    mouthH,
    jawSpan,
    restMouthRatio,
    hasVisibleTeeth: analysis.hasVisibleTeeth,
    isMouthParted: analysis.isMouthParted,
    restTeethDepth: analysis.restTeethDepth,
    yaw,
    stylized,
    rightPupil: rightEyeCenter,
    leftPupil: leftEyeCenter,
    rightEyeCenter: rEyeShape.center,
    leftEyeCenter: lEyeShape.center,
    rightEyeRadius: [rEyeRx, rEyeRy],
    leftEyeRadius: [lEyeRx, lEyeRy],
    rightEyeCurve: rEyeShape.curve,
    leftEyeCurve: lEyeShape.curve,
    rightIrisRadius: rIrisR,
    leftIrisRadius: lIrisR,
    rightEar,
    leftEar,
    hairTop,
    hairLeft,
    hairRight,
    neckBase,
    rightShoulder,
    leftShoulder,
    hands,
    colors: analysis.colors,
    segMask: extras?.segMask,
    segWidth: extras?.segWidth,
    segHeight: extras?.segHeight,
  };
}

export async function detectFace(
  img: HTMLImageElement,
  key: string,
  opts?: { localOnly?: boolean },
): Promise<FacePts | null> {
  if (cache.has(key)) return cache.get(key)!;
  try {
    installTfliteConsoleFilter();
    const { FaceLandmarker } = await import("@mediapipe/tasks-vision");
    const { faceTris, ovalLoop } = buildCanonicalTopology(FaceLandmarker);
    const aspect =
      img.naturalWidth > 0 && img.naturalHeight > 0 ? img.naturalWidth / img.naturalHeight : 1;

    let landmarker: unknown = null;
    let detector: unknown = null;
    let handLandmarker: unknown = null;
    let segmenter: unknown = null;
    try {
      const models = await getVisionModels();
      landmarker = models.landmarker;
      detector = models.detector;
      handLandmarker = models.handLandmarker;
      segmenter = models.segmenter;
    } catch (modelErr) {
      console.warn("MediaPipe vision models unavailable, using fallback keypoints", modelErr);
    }

    const detCanvas = toDetectionCanvas(img, 640);

    // Stage 1: Direct MediaPipe 468/478-landmark detection
    let rawLandmarks: Array<{ x: number; y: number; z?: number }> | undefined;
    if (landmarker) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const r = (landmarker as any).detect(detCanvas);
        rawLandmarks = r?.faceLandmarks?.[0] as
          Array<{ x: number; y: number; z?: number }> | undefined;
      } catch {
        rawLandmarks = undefined;
      }
    }

    let landmarks: Array<[number, number, number]> | null =
      rawLandmarks && rawLandmarks.length >= 468
        ? rawLandmarks.map((pt) => [pt.x, pt.y, pt.z ?? 0])
        : null;
    let stylized = false;
    let visionKp: FaceKeypoints | undefined;

    // Detect hands on real photos via MediaPipe HandLandmarker
    const mpHands: HandKeypoints[] = [];
    if (handLandmarker) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const hRes = (handLandmarker as any).detect(detCanvas);
        const hList = hRes?.landmarks as Array<Array<{ x: number; y: number }>> | undefined;
        if (hList) {
          for (const h of hList.slice(0, 2)) {
            if (h.length >= 21) {
              const wrist = h[0]!;
              const palm = h[9]!;
              const tip = h[12]!;
              const span = Math.hypot((tip.x - wrist.x) * aspect, tip.y - wrist.y);
              mpHands.push({
                wrist: [wrist.x, wrist.y],
                palmCenter: [palm.x, palm.y],
                fingerTips: [tip.x, tip.y],
                radius: Math.min(0.22, Math.max(0.05, span * 0.72)),
              });
            }
          }
        }
      } catch {
        /* ignore hand detection error */
      }
    }

    // Run portrait selfie segmenter for crisp foreground/background separation
    let segMask: Float32Array | undefined;
    let segWidth: number | undefined;
    let segHeight: number | undefined;
    if (segmenter) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const sRes = (segmenter as any).segment(detCanvas);
        const confMask = sRes?.confidenceMasks?.[0];
        if (confMask) {
          const raw = confMask.getAsFloat32Array() as Float32Array;
          segMask = new Float32Array(raw);
          segWidth = confMask.width as number;
          segHeight = confMask.height as number;
          confMask.close?.();
        }
        sRes?.close?.();
      } catch {
        /* ignore segmentation error */
      }
    }

    // Stage 2: Tight crop via FaceDetector + rotated multi-angle fallback for tilted/varied perspectives
    if (!landmarks && detector && landmarker) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const detRes = (detector as any).detect(detCanvas);
        const det = detRes?.detections?.[0];
        const bb = det?.boundingBox;
        if (bb && bb.width > 8 && bb.height > 8) {
          const size = Math.max(bb.width, bb.height) * 1.75;
          const cx = bb.originX + bb.width * 0.5;
          const cy = bb.originY + bb.height * 0.5;
          const x0 = Math.max(0, cx - size * 0.5);
          const y0 = Math.max(0, cy - size * 0.5);
          const x1 = Math.min(detCanvas.width, cx + size * 0.5);
          const y1 = Math.min(detCanvas.height, cy + size * 0.5);
          const cw = Math.max(16, x1 - x0);
          const ch = Math.max(16, y1 - y0);

          const crop = document.createElement("canvas");
          crop.width = 448;
          crop.height = 448;
          crop.getContext("2d")?.drawImage(detCanvas, x0, y0, cw, ch, 0, 0, 448, 448);

          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const cropRes = (landmarker as any).detect(crop);
          rawLandmarks = cropRes?.faceLandmarks?.[0];
          if (rawLandmarks && rawLandmarks.length >= 468) {
            landmarks = rawLandmarks.map((pt) => [
              (x0 + pt.x * cw) / detCanvas.width,
              (y0 + pt.y * ch) / detCanvas.height,
              ((pt.z ?? 0) * cw) / detCanvas.width,
            ]);
          } else if (det.keypoints?.length >= 4) {
            const kps = det.keypoints as Array<{ x: number; y: number }>;
            const rightEye: [number, number] = [kps[0]!.x, kps[0]!.y];
            const leftEye: [number, number] = [kps[1]!.x, kps[1]!.y];
            const noseTip: [number, number] = [kps[2]!.x, kps[2]!.y];
            const mouthCenter: [number, number] = [kps[3]!.x, kps[3]!.y];
            const rightCheek: [number, number] | undefined = kps[4]
              ? [kps[4].x, kps[4].y]
              : undefined;
            const leftCheek: [number, number] | undefined = kps[5]
              ? [kps[5].x, kps[5].y]
              : undefined;
            visionKp = {
              rightEye,
              leftEye,
              noseTip,
              mouthCenter,
              rightCheek,
              leftCheek,
              stylized: true,
            };
            landmarks = fitCanonicalLandmarks(visionKp, aspect);
            stylized = true;
          }
        }
      } catch {
        /* ignore detector fallback error */
      }
    }

    // Stage 2b: Rotated canvas attempt (±22°) so heavily tilted heads still lock locally
    if (!landmarks && landmarker) {
      const sq = Math.max(detCanvas.width, detCanvas.height);
      const rotC = document.createElement("canvas");
      rotC.width = sq;
      rotC.height = sq;
      const rCtx = rotC.getContext("2d");
      if (rCtx) {
        for (const deg of [-22, 22]) {
          const rad = (deg * Math.PI) / 180;
          rCtx.clearRect(0, 0, sq, sq);
          rCtx.save();
          rCtx.translate(sq * 0.5, sq * 0.5);
          rCtx.rotate(rad);
          rCtx.drawImage(detCanvas, -detCanvas.width * 0.5, -detCanvas.height * 0.5);
          rCtx.restore();
          try {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const rr = (landmarker as any).detect(rotC);
            const rPts = rr?.faceLandmarks?.[0] as
              Array<{ x: number; y: number; z?: number }> | undefined;
            if (rPts && rPts.length >= 468) {
              const cos = Math.cos(-rad);
              const sin = Math.sin(-rad);
              landmarks = rPts.map((pt) => {
                const rx = pt.x * sq - sq * 0.5;
                const ry = pt.y * sq - sq * 0.5;
                const ox = rx * cos - ry * sin + detCanvas.width * 0.5;
                const oy = rx * sin + ry * cos + detCanvas.height * 0.5;
                return [
                  Math.min(0.995, Math.max(0.005, ox / detCanvas.width)),
                  Math.min(0.995, Math.max(0.005, oy / detCanvas.height)),
                  pt.z ?? 0,
                ];
              });
              break;
            }
          } catch {
            /* ignore rotation check */
          }
        }
      }
    }

    // Stage 3: Anime / 2D Illustration / Full-Anatomy Gemini Vision fallback (skipped in localOnly mode)
    if (!landmarks && !opts?.localOnly) {
      const kp = await detectWithGeminiFallback(img, key);
      if (kp) {
        visionKp = kp;
        landmarks = fitCanonicalLandmarks(kp, aspect);
        stylized = kp.stylized ?? true;
      }
    }

    // Stage 4: Deterministic centered portrait rig fallback so WebGL2 animation still starts cleanly
    if (!landmarks) {
      const kp: FaceKeypoints = {
        rightEye: [0.41, 0.38],
        leftEye: [0.59, 0.38],
        noseTip: [0.5, 0.46],
        mouthCenter: [0.5, 0.56],
        chin: [0.5, 0.68],
        forehead: [0.5, 0.22],
        stylized: /anime/i.test(key),
      };
      visionKp = kp;
      landmarks = fitCanonicalLandmarks(kp, aspect);
      stylized = !!kp.stylized;
    }

    // Use MediaPipe hands if detected, otherwise Gemini Vision hands
    const finalHands = mpHands.length > 0 ? mpHands : visionKp?.hands;
    // Discard segmentation mask if it's almost empty (<8% foreground) as happens on some anime drawings
    let usableSegMask = segMask;
    if (usableSegMask && segWidth && segHeight) {
      let sum = 0;
      for (let i = 0; i < usableSegMask.length; i++) {
        if (usableSegMask[i]! > 0.5) sum++;
      }
      const frac = sum / usableSegMask.length;
      if (frac < 0.08 || frac > 0.96) {
        usableSegMask = undefined;
      }
    }

    const out = buildFacePts(img, landmarks, faceTris, ovalLoop, aspect, stylized, {
      kp: visionKp,
      hands: finalHands,
      segMask: usableSegMask,
      segWidth: usableSegMask ? segWidth : undefined,
      segHeight: usableSegMask ? segHeight : undefined,
    });
    cache.set(key, out);
    return out;
  } catch (e) {
    console.warn("face detect failed", e);
    lm = null;
    fd = null;
    hl = null;
    seg = null;
    visionPromise = null;
    return null;
  }
}
