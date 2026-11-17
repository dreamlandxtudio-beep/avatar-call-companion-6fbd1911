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
  lowerEyelid: [number, number, number];
  enamel: [number, number, number];
  shirt: [number, number, number];
  scleraRight: [number, number, number];
  scleraLeft: [number, number, number];
  scleraRightNeg: [number, number, number];
  scleraRightPos: [number, number, number];
  scleraLeftNeg: [number, number, number];
  scleraLeftPos: [number, number, number];
};

export type EyeCurveProfile = {
  upper: [number, number, number, number];
  lower: [number, number, number, number];
  upperArch: [number, number, number, number, number, number, number, number];
  lowerArch: [number, number, number, number, number, number, number, number];
  halfW: number;
  scaleY: number;
  outerReach: number;
  outerWingY: number;
  browFloorEy?: number | undefined;
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
  restMouthTopY: number;
  restMouthBotY: number;
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
  lowerNeck: [number, number];
  adamApple: [number, number];
  rightTrapezius: [number, number];
  leftTrapezius: [number, number];
  rightShoulder: [number, number];
  leftShoulder: [number, number];
  rightCheekPos: [number, number];
  leftCheekPos: [number, number];
  rightInnerCanthus: [number, number];
  leftInnerCanthus: [number, number];
  hasTrapezius?: boolean | undefined;
  neckInflectionY?: number | undefined;
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

  const samples = [-0.74, -0.25, 0.25, 0.74];
  const samples8 = [-0.82, -0.58, -0.34, -0.11, 0.11, 0.34, 0.58, 0.82];
  const rawUpper = samples.map((qt) => Math.min(-0.001, evalSorted(up, qt)));
  const rawLower = samples.map((qt) => Math.max(0.001, evalSorted(lo, qt)));
  const rawUpper8 = samples8.map((qt) => Math.min(-0.001, evalSorted(up, qt)));
  const rawLower8 = samples8.map((qt) => Math.max(0.001, evalSorted(lo, qt)));

  let maxAbs = 0.006;
  for (const v of rawUpper8) maxAbs = Math.max(maxAbs, Math.abs(v));
  for (const v of rawLower8) maxAbs = Math.max(maxAbs, Math.abs(v));

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
      upperArch: [
        rawUpper8[0]! / maxAbs,
        rawUpper8[1]! / maxAbs,
        rawUpper8[2]! / maxAbs,
        rawUpper8[3]! / maxAbs,
        rawUpper8[4]! / maxAbs,
        rawUpper8[5]! / maxAbs,
        rawUpper8[6]! / maxAbs,
        rawUpper8[7]! / maxAbs,
      ],
      lowerArch: [
        rawLower8[0]! / maxAbs,
        rawLower8[1]! / maxAbs,
        rawLower8[2]! / maxAbs,
        rawLower8[3]! / maxAbs,
        rawLower8[4]! / maxAbs,
        rawLower8[5]! / maxAbs,
        rawLower8[6]! / maxAbs,
        rawLower8[7]! / maxAbs,
      ],
      halfW,
      scaleY: maxAbs,
      outerReach: 1.26,
      outerWingY: 0,
      browFloorEy: -0.78,
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
  rightPupil: [number, number];
  leftPupil: [number, number];
  rightIrisRadius: number;
  leftIrisRadius: number;
  rightEyeCurve: EyeCurveProfile;
  leftEyeCurve: EyeCurveProfile;
};

function analyzePhotoFeatures(
  img: HTMLImageElement,
  pts: Array<[number, number, number]>,
  restMouthRatio: number,
  rightPupil: [number, number],
  leftPupil: [number, number],
  rightEyeCenter: [number, number],
  leftEyeCenter: [number, number],
  rightEyeAxis: Axis2D,
  leftEyeAxis: Axis2D,
  rightEyeCurve: EyeCurveProfile,
  leftEyeCurve: EyeCurveProfile,
  rightIrisRadius: number,
  leftIrisRadius: number,
  aspect: number,
): MouthAnalysis {
  const fallback: FaceColors = {
    skin: [215, 175, 155],
    lip: [165, 92, 92],
    eyelid: [195, 155, 138],
    lowerEyelid: [198, 160, 142],
    enamel: [232, 226, 218],
    shirt: [45, 68, 88],
    scleraRight: [212, 206, 200],
    scleraLeft: [212, 206, 200],
    scleraRightNeg: [204, 198, 192],
    scleraRightPos: [216, 210, 204],
    scleraLeftNeg: [216, 210, 204],
    scleraLeftPos: [204, 198, 192],
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
        rightPupil,
        leftPupil,
        rightIrisRadius,
        leftIrisRadius,
        rightEyeCurve,
        leftEyeCurve,
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

    const sampleEyeScleraSides = (
      eyeLoop: number[],
      eyeCenter: [number, number],
      pupil: [number, number],
      axis: Axis2D,
      irisRad: number,
    ): {
      mid: [number, number, number];
      neg: [number, number, number];
      pos: [number, number, number];
      refinedPupil: [number, number];
      refinedIrisRad: number;
    } => {
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
      const cropW = 80;
      const cropH = 40;
      const ec = document.createElement("canvas");
      ec.width = cropW;
      ec.height = cropH;
      const ectx = ec.getContext("2d", { willReadFrequently: true });
      if (!ectx) {
        return {
          mid: [212, 206, 200],
          neg: [208, 202, 196],
          pos: [214, 208, 202],
          refinedPupil: pupil,
          refinedIrisRad: irisRad * 1.15,
        };
      }
      ectx.drawImage(img, eMinX * imgW, eMinY * imgH, ew * imgW, eh * imgH, 0, 0, cropW, cropH);
      const edata = ectx.getImageData(0, 0, cropW, cropH).data;
      const eyePoly: Array<[number, number]> = eyeLoop.map((idx) => [
        ((pts[idx]![0] - eMinX) / ew) * cropW,
        ((pts[idx]![1] - eMinY) / eh) * cropH,
      ]);

      // Pass 1: Identify clean outer sclera reference color (supports white, warm, or stylized tinted sclera)
      const eyePixels: Array<{
        nx: number;
        ny: number;
        fy: number;
        relX: number;
        relY: number;
        luma: number;
        r: number;
        g: number;
        b: number;
      }> = [];
      const eyeHalfW = Math.max(0.01, ew * aspect * 0.5);
      for (let py = 0; py < cropH; py++) {
        const fy = (py + 0.5) / cropH;
        if (fy < 0.16 || fy > 0.9) continue;
        for (let px = 0; px < cropW; px++) {
          if (!pointInPoly(px + 0.5, py + 0.5, eyePoly)) continue;
          const nx = eMinX + ((px + 0.5) / cropW) * ew;
          const ny = eMinY + fy * eh;
          const off = (py * cropW + px) * 4;
          const r = edata[off]!;
          const g = edata[off + 1]!;
          const b = edata[off + 2]!;
          const luma = 0.299 * r + 0.587 * g + 0.114 * b;
          const relX = (nx - eyeCenter[0]) * aspect * axis.ux + (ny - eyeCenter[1]) * axis.uy;
          const relY = (nx - eyeCenter[0]) * aspect * axis.vx + (ny - eyeCenter[1]) * axis.vy;
          eyePixels.push({ nx, ny, fy, relX, relY, luma, r, g, b });
        }
      }

      let refinedPupil: [number, number] = [pupil[0], pupil[1]];
      let refinedIrisRad = irisRad * 1.06;

      // Helper to sample bilinear RGB inside the eye crop for radial edge-jump scanning
      const sampleEyeCrop = (
        nx: number,
        ny: number,
      ): { luma: number; r: number; g: number; b: number } => {
        const px = Math.min(cropW - 1, Math.max(0, Math.round(((nx - eMinX) / ew) * (cropW - 1))));
        const py = Math.min(cropH - 1, Math.max(0, Math.round(((ny - eMinY) / eh) * (cropH - 1))));
        const off = (py * cropW + px) * 4;
        const r = edata[off]!;
        const g = edata[off + 1]!;
        const b = edata[off + 2]!;
        return { r, g, b, luma: 0.299 * r + 0.587 * g + 0.114 * b };
      };

      if (eyePixels.length >= 24) {
        // Estimate baseline sclera signature from bright lateral zones (excluding outer corners & center iris)
        const latBright = eyePixels
          .filter((p) => {
            const d0 = Math.hypot((p.nx - pupil[0]) * aspect, p.ny - pupil[1]);
            return (
              p.fy >= 0.34 &&
              p.fy <= 0.74 &&
              d0 >= irisRad * 1.14 &&
              Math.abs(p.relX) <= eyeHalfW * 0.78
            );
          })
          .sort((a, b) => a.luma - b.luma);
        const scleraRef =
          latBright.length >= 4
            ? latBright[Math.floor(latBright.length * 0.58)]!
            : (() => {
                const sorted = [...eyePixels].sort((a, b) => a.luma - b.luma);
                return sorted[Math.floor(sorted.length * 0.78)]!;
              })();

        const colDist = (r: number, g: number, b: number) =>
          Math.hypot(r - scleraRef.r, g - scleraRef.g, b - scleraRef.b);
        const scleraLuma = scleraRef.luma;

        // Pass 1b: Re-center cornea by bilateral color-difference weighting inside iris window (avoids upper lashes)
        let sumW = 0;
        let sumX = 0;
        let sumY = 0;
        for (const p of eyePixels) {
          if (p.fy < 0.28 || p.fy > 0.84) continue;
          const d0 = Math.hypot((p.nx - pupil[0]) * aspect, p.ny - pupil[1]);
          if (d0 > irisRad * 1.25) continue;
          const dc = colDist(p.r, p.g, p.b);
          const dl = Math.max(0, scleraLuma - p.luma);
          if (dc < 24 && dl < 22) continue;
          const w = dc * 0.7 + dl * 1.1;
          sumW += w * w;
          sumX += p.nx * w * w;
          sumY += p.ny * w * w;
        }
        if (sumW > 0) {
          const cx = sumX / sumW;
          const cy = sumY / sumW;
          const shiftDist = Math.hypot((cx - pupil[0]) * aspect, cy - pupil[1]);
          const maxShift = irisRad * 0.28;
          if (shiftDist > 1e-6) {
            const clampScale = Math.min(1, maxShift / shiftDist);
            refinedPupil = [
              pupil[0] + (cx - pupil[0]) * clampScale,
              pupil[1] + (cy - pupil[1]) * clampScale,
            ];
          }
        }

        // Pass 1c: Scan radially outward from pupil along clean lower/horizontal angles to find sharpest iris->sclera color jump & symmetry
        const scanRadiusAtAngle = (angRad: number): number | null => {
          const ux = Math.cos(angRad);
          const uy = Math.sin(angRad);
          const minR = Math.max(0.002, irisRad * 0.55);
          const maxR = Math.min(eyeHalfW * 0.85, irisRad * 1.56);
          const step = Math.max(0.0008, (maxR - minR) / 16);
          let bestR = -1;
          let bestJump = 18;
          for (let r = minR; r <= maxR - step; r += step) {
            const inPt = sampleEyeCrop(
              refinedPupil[0] + (ux * (r - step * 0.7)) / aspect,
              refinedPupil[1] + uy * (r - step * 0.7),
            );
            const outPt = sampleEyeCrop(
              refinedPupil[0] + (ux * (r + step * 0.7)) / aspect,
              refinedPupil[1] + uy * (r + step * 0.7),
            );
            const inToSclera = colDist(inPt.r, inPt.g, inPt.b);
            const outToSclera = colDist(outPt.r, outPt.g, outPt.b);
            const stepGrad = Math.hypot(outPt.r - inPt.r, outPt.g - inPt.g, outPt.b - inPt.b);
            const jump = (inToSclera - outToSclera) * 1.15 + stepGrad * 0.65;
            if (inToSclera > 24 && jump > bestJump) {
              bestJump = jump;
              bestR = r;
            }
          }
          return bestR > 0 ? bestR : null;
        };

        const leftAngles = [Math.PI, Math.PI - 0.28, Math.PI + 0.26];
        const rightAngles = [0, 0.28, -0.26];
        const downAngles = [Math.PI * 0.5, Math.PI * 0.5 - 0.35, Math.PI * 0.5 + 0.35];
        const leftHits = leftAngles.map(scanRadiusAtAngle).filter((v): v is number => v !== null);
        const rightHits = rightAngles.map(scanRadiusAtAngle).filter((v): v is number => v !== null);
        const downHits = downAngles.map(scanRadiusAtAngle).filter((v): v is number => v !== null);

        if (leftHits.length > 0 && rightHits.length > 0) {
          const avgL = leftHits.reduce((a, b) => a + b, 0) / leftHits.length;
          const avgR = rightHits.reduce((a, b) => a + b, 0) / rightHits.length;
          const symShift = Math.max(-irisRad * 0.22, Math.min(irisRad * 0.22, (avgR - avgL) * 0.5));
          refinedPupil = [refinedPupil[0] + symShift / aspect, refinedPupil[1]];
          const symRad = (avgL + avgR) * 0.5;
          const allHits = [...leftHits, ...rightHits, ...downHits].sort((a, b) => a - b);
          const midHit =
            allHits[Math.min(allHits.length - 1, Math.floor(allHits.length * 0.65))] ?? symRad;
          refinedIrisRad = Math.min(
            eyeHalfW * 0.62,
            irisRad * 1.06,
            Math.max(irisRad * 0.94, Math.max(symRad, midHit) * 1.02),
          );
        } else {
          const allHits = [...leftHits, ...rightHits, ...downHits].sort((a, b) => a - b);
          if (allHits.length >= 2) {
            const hitR = allHits[Math.min(allHits.length - 1, Math.floor(allHits.length * 0.65))]!;
            refinedIrisRad = Math.min(
              eyeHalfW * 0.62,
              irisRad * 1.06,
              Math.max(irisRad * 0.94, hitR * 1.02),
            );
          }
        }
      }

      type ScleraCand = { score: number; rgb: [number, number, number]; relX: number };
      const summarize = (list: ScleraCand[], fallbackColor: [number, number, number]) => {
        if (list.length === 0) return fallbackColor;
        const sorted = [...list].sort((a, b) => a.score - b.score);
        // Take middle neutral sclera band (35%-75%) to preserve exact portrait luminance without washing out
        const sIdx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.35));
        const eIdx = Math.min(sorted.length, Math.max(sIdx + 1, Math.floor(sorted.length * 0.75)));
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let cnt = 0;
        for (let i = sIdx; i < eIdx; i++) {
          const cRGB = sorted[i]!.rgb;
          sr += cRGB[0];
          sg += cRGB[1];
          sb += cRGB[2];
          cnt++;
        }
        const n = Math.max(1, cnt);
        return [Math.round(sr / n), Math.round(sg / n), Math.round(sb / n)] as [
          number,
          number,
          number,
        ];
      };

      // Preserve the portrait's real sclera brightness & tone (never bleach shaded eyes whitish), only calming abnormal caruncle redness
      const cleanScleraNatural = (rgb: [number, number, number]): [number, number, number] => {
        const rednessExcess = Math.max(0, rgb[0] - Math.max(rgb[1], rgb[2]) - 12);
        if (rednessExcess <= 0) {
          return [Math.round(rgb[0]), Math.round(rgb[1]), Math.round(rgb[2])];
        }
        const nr = Math.max(0, rgb[0] - rednessExcess * 0.35);
        const ng = Math.min(255, rgb[1] + rednessExcess * 0.45);
        const nb = Math.min(255, rgb[2] + rednessExcess * 0.42);
        return [Math.round(nr), Math.round(ng), Math.round(nb)];
      };

      const sampleScleraFromRadius = (
        cPupil: [number, number],
        cRad: number,
      ): {
        mid: [number, number, number];
        neg: [number, number, number];
        pos: [number, number, number];
      } => {
        const negCands: ScleraCand[] = [];
        const posCands: ScleraCand[] = [];
        const allCands: ScleraCand[] = [];
        const fallbackCands: ScleraCand[] = [];

        for (const p of eyePixels) {
          const dPupil = Math.hypot((p.nx - cPupil[0]) * aspect, p.ny - cPupil[1]);
          if (dPupil < cRad * 1.18) continue;
          if (p.fy < 0.36 || p.fy > 0.72) continue;
          if (Math.abs(p.relX) > eyeHalfW * 0.72) continue;

          const maxC = Math.max(p.r, p.g, p.b);
          const minC = Math.min(p.r, p.g, p.b);
          const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
          const redness = p.r - (p.g + p.b) * 0.5;
          if (p.luma >= 64 && redness <= 22) {
            fallbackCands.push({
              score: p.luma - sat * 45 - Math.max(0, redness) * 1.4,
              rgb: [p.r, p.g, p.b],
              relX: p.relX,
            });
          }

          if (p.luma < 86 || sat > 0.24 || redness > 16) continue;
          const item: ScleraCand = {
            score: p.luma - Math.max(0, redness) * 1.6 - sat * 60,
            rgb: [p.r, p.g, p.b],
            relX: p.relX,
          };
          allCands.push(item);
          if (p.relX < -cRad * 0.25) negCands.push(item);
          else if (p.relX > cRad * 0.25) posCands.push(item);
        }

        const sourcePool = allCands.length >= 4 ? allCands : fallbackCands;
        const mid = cleanScleraNatural(summarize(sourcePool, [214, 210, 206]));
        const midLuma = 0.299 * mid[0] + 0.587 * mid[1] + 0.114 * mid[2];
        const rawNeg = negCands.length >= 3 ? cleanScleraNatural(summarize(negCands, mid)) : mid;
        const rawPos = posCands.length >= 3 ? cleanScleraNatural(summarize(posCands, mid)) : mid;

        // Enforce gentle homogeneity across left/right sclera halves without jarring jumps
        const anchorSide = (side: [number, number, number]): [number, number, number] => {
          const sLuma = 0.299 * side[0] + 0.587 * side[1] + 0.114 * side[2];
          const lift = sLuma > 1 ? Math.max(0.95, Math.min(1.05, midLuma / sLuma)) : 1;
          const adj: [number, number, number] = [
            Math.min(252, side[0] * lift),
            Math.min(252, side[1] * lift),
            Math.min(252, side[2] * lift),
          ];
          return [
            Math.round(mid[0] * 0.55 + adj[0] * 0.45),
            Math.round(mid[1] * 0.55 + adj[1] * 0.45),
            Math.round(mid[2] * 0.55 + adj[2] * 0.45),
          ];
        };

        return { mid, neg: anchorSide(rawNeg), pos: anchorSide(rawPos) };
      };

      const scleraEst = sampleScleraFromRadius(refinedPupil, refinedIrisRad);
      const { mid, neg, pos } = scleraEst;
      return { mid, neg, pos, refinedPupil, refinedIrisRad };
    };

    const skin = avgAt([118, 347, 50, 280, 195]);
    const skinRefLuma = 0.299 * skin[0] + 0.587 * skin[1] + 0.114 * skin[2];

    const refineEyeAndSample = (
      eyeLoop: number[],
      eyeCenter: [number, number],
      pupil: [number, number],
      axis: Axis2D,
      curve: EyeCurveProfile,
      irisRad: number,
      outerSign: -1 | 1,
    ): {
      mid: [number, number, number];
      neg: [number, number, number];
      pos: [number, number, number];
      refinedPupil: [number, number];
      refinedIrisRad: number;
      refinedCurve: EyeCurveProfile;
      upperLidRgb: [number, number, number] | null;
      lowerLidRgb: [number, number, number] | null;
    } => {
      const baseRes = sampleEyeScleraSides(eyeLoop, eyeCenter, pupil, axis, irisRad);
      const imgW = img.naturalWidth || img.width || 512;
      const imgH = img.naturalHeight || img.height || 512;
      const halfW = curve.halfW;
      const scaleY = curve.scaleY;

      // Bounding region anchored beyond outer commissure so extended lash/eyeliner wings are detected cleanly
      const padNormX = (halfW * 1.65) / aspect;
      const padNormYTop = scaleY * 2.4;
      const padNormYBot = scaleY * 1.8;
      const rx0 = Math.max(0, eyeCenter[0] - padNormX);
      const rx1 = Math.min(1, eyeCenter[0] + padNormX);
      const ry0 = Math.max(0, eyeCenter[1] - padNormYTop);
      const ry1 = Math.min(1, eyeCenter[1] + padNormYBot);
      const rw = Math.max(0.012, rx1 - rx0);
      const rh = Math.max(0.012, ry1 - ry0);

      const CW = 112;
      const CH = 72;
      const cc = document.createElement("canvas");
      cc.width = CW;
      cc.height = CH;
      const cctx = cc.getContext("2d", { willReadFrequently: true });
      if (!cctx) {
        return {
          ...baseRes,
          refinedCurve: curve,
          upperLidRgb: null,
          lowerLidRgb: null,
        };
      }
      cctx.drawImage(img, rx0 * imgW, ry0 * imgH, rw * imgW, rh * imgH, 0, 0, CW, CH);
      const cdata = cctx.getImageData(0, 0, CW, CH).data;

      const sampleCrop = (
        nx: number,
        ny: number,
      ): { r: number; g: number; b: number; luma: number } => {
        const px = Math.min(CW - 1, Math.max(0, Math.round(((nx - rx0) / rw) * (CW - 1))));
        const py = Math.min(CH - 1, Math.max(0, Math.round(((ny - ry0) / rh) * (CH - 1))));
        const off = (py * CW + px) * 4;
        const r = cdata[off]!;
        const g = cdata[off + 1]!;
        const b = cdata[off + 2]!;
        return { r, g, b, luma: 0.299 * r + 0.587 * g + 0.114 * b };
      };

      const sampleEyCoord = (ex: number, ey: number) => {
        const dx = axis.ux * (ex * halfW) + axis.vx * (ey * scaleY);
        const dy = axis.uy * (ex * halfW) + axis.vy * (ey * scaleY);
        return sampleCrop(eyeCenter[0] + dx / aspect, eyeCenter[1] + dy);
      };

      // Project lower eyebrow landmarks into eye-local ey so tarsal lid sampling strictly avoids eyebrow hairs
      const browBotIndices = outerSign < 0 ? [46, 53, 52, 65, 55] : [276, 283, 282, 295, 285];
      let lowestBrowEy = -1.65;
      for (const bIdx of browBotIndices) {
        const bp = pts[bIdx];
        if (!bp) continue;
        const bdx = (bp[0] - eyeCenter[0]) * aspect;
        const bdy = bp[1] - eyeCenter[1];
        const bEy = (bdx * axis.vx + bdy * axis.vy) / Math.max(0.004, scaleY);
        if (bEy > lowestBrowEy) lowestBrowEy = bEy;
      }
      const browFloorEy = Math.max(-1.6, Math.min(-0.46, lowestBrowEy + 0.16));

      // Corner-to-corner 8-column color-edge refinement for upper lash margin & lower lid rim
      const cols = [-0.82, -0.58, -0.34, -0.11, 0.11, 0.34, 0.58, 0.82];
      const detectedUp = [...curve.upperArch];
      const detectedLo = [...curve.lowerArch];
      const upperSkinCands: Array<{ luma: number; rgb: [number, number, number] }> = [];
      const lowerSkinCands: Array<{ luma: number; rgb: [number, number, number] }> = [];

      for (let ci = 0; ci < cols.length; ci++) {
        const ex = cols[ci]!;
        const hintUp = curve.upperArch[ci]!;
        const hintLo = curve.lowerArch[ci]!;

        // 1) Upper lid boundary: strongest dark-above -> brighter-below transition OR sharp vertical intensity contrast near hintUp
        let bestUp = hintUp;
        let bestUpScore = -1e9;
        for (let ey = hintUp - 0.38; ey <= Math.min(-0.08, hintUp + 0.34); ey += 0.04) {
          const above = sampleEyCoord(ex, ey - 0.14);
          const edge = sampleEyCoord(ex, ey);
          const below = sampleEyCoord(ex, ey + 0.16);
          const contrast =
            below.luma - edge.luma * 0.65 - Math.max(0, above.luma - below.luma) * 0.15;
          const lashDarkness = Math.max(0, skinRefLuma - edge.luma) * 0.45;
          const prior = Math.abs(ey - hintUp) * 95;
          const score = contrast + lashDarkness - prior;
          if (score > bestUpScore) {
            bestUpScore = score;
            bestUp = ey;
          }
        }
        detectedUp[ci] = Math.min(-0.08, Math.max(hintUp - 0.28, Math.min(hintUp + 0.24, bestUp)));

        // 2) Lower lid rim: brighter-interior -> lid-border/skin transition near hintLo
        let bestLo = hintLo;
        let bestLoScore = -1e9;
        for (let ey = Math.max(0.08, hintLo - 0.32); ey <= hintLo + 0.36; ey += 0.04) {
          const above = sampleEyCoord(ex, ey - 0.16);
          const edge = sampleEyCoord(ex, ey);
          const below = sampleEyCoord(ex, ey + 0.18);
          const edgeJump =
            Math.abs(above.luma - edge.luma) * 0.65 + (below.luma - edge.luma) * 0.45;
          const prior = Math.abs(ey - hintLo) * 105;
          const score = edgeJump - prior;
          if (score > bestLoScore) {
            bestLoScore = score;
            bestLo = ey;
          }
        }
        detectedLo[ci] = Math.max(0.08, Math.min(hintLo + 0.28, Math.max(hintLo - 0.24, bestLo)));

        // 3) Sample pure tarsal upper-lid skin strictly above lash root AND strictly below eyebrow hairs
        if (Math.abs(ex) <= 0.64) {
          const tarsalNearY = detectedUp[ci]! - 0.18;
          const tarsalCeilY = Math.max(detectedUp[ci]! - 0.58, browFloorEy + 0.06);
          const spanStart = Math.min(tarsalNearY, tarsalCeilY);
          const spanEnd = Math.max(tarsalNearY, tarsalCeilY + 0.04);
          for (let frac = 0.2; frac <= 0.85; frac += 0.22) {
            const sampleY = spanStart + (spanEnd - spanStart) * frac;
            const sp = sampleEyCoord(ex, sampleY);
            const chromaDist =
              Math.abs(sp.r - skin[0]) + Math.abs(sp.g - skin[1]) + Math.abs(sp.b - skin[2]);
            if (sp.luma >= Math.max(66, skinRefLuma * 0.72) && chromaDist < 125) {
              upperSkinCands.push({ luma: sp.luma, rgb: [sp.r, sp.g, sp.b] });
            }
          }
          for (let dy = 0.22; dy <= 0.62; dy += 0.16) {
            const sp = sampleEyCoord(ex, detectedLo[ci]! + dy);
            if (sp.luma >= Math.max(62, skinRefLuma * 0.68)) {
              lowerSkinCands.push({ luma: sp.luma, rgb: [sp.r, sp.g, sp.b] });
            }
          }
        }
      }

      // Smooth adjacent arch samples gently so curvature from left corner to right corner is continuously arched
      const smoothArch = (
        arr: number[],
        isUpper: boolean,
      ): [number, number, number, number, number, number, number, number] => {
        const out = [...arr];
        for (let i = 1; i < out.length - 1; i++) {
          out[i] = arr[i - 1]! * 0.24 + arr[i]! * 0.52 + arr[i + 1]! * 0.24;
        }
        out[0] = arr[0]! * 0.72 + arr[1]! * 0.28;
        out[7] = arr[7]! * 0.72 + arr[6]! * 0.28;
        return out.map((v) => (isUpper ? Math.min(-0.06, v) : Math.max(0.06, v))) as [
          number,
          number,
          number,
          number,
          number,
          number,
          number,
          number,
        ];
      };

      const upperArch = smoothArch(detectedUp, true);
      const lowerArch = smoothArch(detectedLo, false);

      const medianRgb = (
        list: Array<{ luma: number; rgb: [number, number, number] }>,
      ): [number, number, number] | null => {
        if (list.length < 3) return null;
        const sorted = [...list].sort((a, b) => a.luma - b.luma);
        const sIdx = Math.floor(sorted.length * 0.32);
        const eIdx = Math.max(sIdx + 1, Math.floor(sorted.length * 0.76));
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let cnt = 0;
        for (let i = sIdx; i < eIdx; i++) {
          const c = sorted[i]!.rgb;
          sr += c[0];
          sg += c[1];
          sb += c[2];
          cnt++;
        }
        return [Math.round(sr / cnt), Math.round(sg / cnt), Math.round(sb / cnt)];
      };

      // Scan past the outer commissure (outerSign * 0.96 .. 1.44) to detect the darkest outer lash/eyeliner wing + default margin
      let outerReach = 1.24;
      let outerWingY = 0;
      let bestWingScore = 12;
      for (let ext = 1.02; ext <= 1.44; ext += 0.08) {
        const exProbe = outerSign * ext;
        let colDarkScore = 0;
        let colBestY = 0;
        for (let ey = -0.34; ey <= 0.24; ey += 0.06) {
          const sp = sampleEyCoord(exProbe, ey);
          const dark = Math.max(0, skinRefLuma - sp.luma);
          if (dark > colDarkScore) {
            colDarkScore = dark;
            colBestY = ey;
          }
        }
        if (colDarkScore > Math.max(22, skinRefLuma * 0.24)) {
          outerReach = Math.max(outerReach, Math.min(1.46, ext + 0.1));
          if (colDarkScore > bestWingScore) {
            bestWingScore = colDarkScore;
            outerWingY = Math.max(-0.18, Math.min(0.14, colBestY));
          }
        }
      }

      return {
        ...baseRes,
        refinedCurve: {
          ...curve,
          upper: [upperArch[0], upperArch[2], upperArch[5], upperArch[7]],
          lower: [lowerArch[0], lowerArch[2], lowerArch[5], lowerArch[7]],
          upperArch,
          lowerArch,
          outerReach,
          outerWingY,
          browFloorEy,
        },
        upperLidRgb: medianRgb(upperSkinCands),
        lowerLidRgb: medianRgb(lowerSkinCands),
      };
    };

    const lip = avgAt([0, 17, 37, 267, 84, 314]);
    const rawEyelid = avgAt([29, 30, 259, 260, 27, 257]);
    const forehead = avgAt([10, 151, 9]);
    const rSclera = refineEyeAndSample(
      RIGHT_EYE_LOOP,
      rightEyeCenter,
      rightPupil,
      rightEyeAxis,
      rightEyeCurve,
      rightIrisRadius,
      -1,
    );
    const lSclera = refineEyeAndSample(
      LEFT_EYE_LOOP,
      leftEyeCenter,
      leftPupil,
      leftEyeAxis,
      leftEyeCurve,
      leftIrisRadius,
      1,
    );
    const avgLid = (
      a: [number, number, number] | null,
      b: [number, number, number] | null,
      fb: [number, number, number],
    ): [number, number, number] => {
      if (a && b) {
        return [
          Math.round((a[0] + b[0]) * 0.5),
          Math.round((a[1] + b[1]) * 0.5),
          Math.round((a[2] + b[2]) * 0.5),
        ];
      }
      return a ?? b ?? fb;
    };
    const eyelid = avgLid(rSclera.upperLidRgb, lSclera.upperLidRgb, [
      Math.round(skin[0] * 0.58 + rawEyelid[0] * 0.42),
      Math.round(skin[1] * 0.58 + rawEyelid[1] * 0.42),
      Math.round(skin[2] * 0.58 + rawEyelid[2] * 0.42),
    ]);
    const lowerEyelid = avgLid(rSclera.lowerLidRgb, lSclera.lowerLidRgb, skin);
    const scleraRight = rSclera.mid;
    const scleraLeft = lSclera.mid;
    const scleraRightNeg = rSclera.neg;
    const scleraRightPos = rSclera.pos;
    const scleraLeftNeg = lSclera.neg;
    const scleraLeftPos = lSclera.pos;

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
      colors: {
        skin,
        lip,
        eyelid,
        lowerEyelid,
        enamel,
        shirt,
        scleraRight,
        scleraLeft,
        scleraRightNeg,
        scleraRightPos,
        scleraLeftNeg,
        scleraLeftPos,
      },
      hasVisibleTeeth,
      isMouthParted,
      restTeethDepth,
      rightPupil: rSclera.refinedPupil,
      leftPupil: lSclera.refinedPupil,
      rightIrisRadius: rSclera.refinedIrisRad,
      leftIrisRadius: lSclera.refinedIrisRad,
      rightEyeCurve: rSclera.refinedCurve,
      leftEyeCurve: lSclera.refinedCurve,
    };
  } catch {
    return {
      colors: fallback,
      hasVisibleTeeth: restMouthRatio > 0.08,
      isMouthParted: restMouthRatio > 0.08,
      restTeethDepth: Math.abs(pts[14]![1] - pts[13]![1]),
      rightPupil,
      leftPupil,
      rightIrisRadius: rightIrisRadius * 1.15,
      leftIrisRadius: leftIrisRadius * 1.15,
      rightEyeCurve,
      leftEyeCurve,
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
  if (lm) {
    return { landmarker: lm, detector: fd, handLandmarker: hl, segmenter: seg };
  }
  if (Date.now() < visionRetryAfter) {
    throw new Error("Vision model cooldown");
  }
  installTfliteConsoleFilter();
  const { FaceDetector, FaceLandmarker, HandLandmarker, ImageSegmenter } =
    await import("@mediapipe/tasks-vision");

  const withSoftTimeout = <T>(p: Promise<T>, ms: number): Promise<T | null> =>
    Promise.race([p.catch(() => null), new Promise<null>((r) => setTimeout(() => r(null), ms))]);

  const initFromBase = async () => {
    const fs = await resolveVisionFileset();
    const landmarkerPromise = FaceLandmarker.createFromOptions(fs, {
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
    });
    const [landmarker, detector, handLandmarker, segmenter] = await Promise.all([
      landmarkerPromise,
      withSoftTimeout(
        FaceDetector.createFromOptions(fs, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate: "CPU",
          },
          runningMode: "IMAGE",
          minDetectionConfidence: 0.15,
        }),
        6000,
      ),
      withSoftTimeout(
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
        }),
        6000,
      ),
      withSoftTimeout(
        ImageSegmenter.createFromOptions(fs, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            delegate: "CPU",
          },
          runningMode: "IMAGE",
          outputConfidenceMasks: true,
          outputCategoryMask: false,
        }),
        6000,
      ),
    ]);
    return { landmarker, detector, handLandmarker, segmenter };
  };

  if (!visionPromise) {
    visionPromise = Promise.race([
      initFromBase(),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Vision model timeout")), 15000),
      ),
    ]).catch((err) => {
      visionPromise = null;
      visionRetryAfter = Date.now() + 20000;
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

  const mouthW = Math.max(0.04, distA(61, 291));
  const restGap = distA(13, 14);
  const mouthH = Math.max(0.012, distA(0, 17));
  const restMouthRatio = restGap / mouthW;

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

  // Detect laryngeal prominence (Adam's apple) along exposed neck midline below chin
  const detectAdamApple = (): [number, number] => {
    const midX = chinPt[0] * 0.42 + neckBase[0] * 0.58;
    const fallbackY =
      chinPt[1] + Math.min(faceH * 0.24, Math.max(faceH * 0.14, (neckBase[1] - chinPt[1]) * 0.54));
    try {
      const imgW = img.naturalWidth || img.width || 512;
      const imgH = img.naturalHeight || img.height || 512;
      const bandW = Math.max(0.024, (faceW * 0.14) / aspect);
      const yTop = Math.min(0.94, chinPt[1] + faceH * 0.08);
      const yBot = Math.min(
        0.98,
        Math.max(yTop + 0.04, Math.min(neckBase[1], chinPt[1] + faceH * 0.42)),
      );
      const x0 = Math.max(0, midX - bandW);
      const x1 = Math.min(1, midX + bandW);
      const rw = Math.max(0.02, x1 - x0);
      const rh = Math.max(0.02, yBot - yTop);
      const NW = 32;
      const NH = 48;
      const nc = document.createElement("canvas");
      nc.width = NW;
      nc.height = NH;
      const nctx = nc.getContext("2d", { willReadFrequently: true });
      if (!nctx) return [midX, fallbackY];
      nctx.drawImage(img, x0 * imgW, yTop * imgH, rw * imgW, rh * imgH, 0, 0, NW, NH);
      const nd = nctx.getImageData(0, 0, NW, NH).data;
      const skin = analysis.colors.skin;
      const rowStats: Array<{ midLuma: number; flankLuma: number; skinOk: boolean }> = [];
      for (let py = 0; py < NH; py++) {
        let midSum = 0;
        let midN = 0;
        let flankSum = 0;
        let flankN = 0;
        let skinDiffSum = 0;
        for (let px = 0; px < NW; px++) {
          const o = (py * NW + px) * 4;
          const r = nd[o]!;
          const g = nd[o + 1]!;
          const b = nd[o + 2]!;
          const lum = 0.299 * r + 0.587 * g + 0.114 * b;
          const fx = (px + 0.5) / NW;
          if (fx >= 0.36 && fx <= 0.64) {
            midSum += lum;
            midN++;
            skinDiffSum += Math.abs(r - skin[0]) + Math.abs(g - skin[1]) + Math.abs(b - skin[2]);
          } else if (fx <= 0.22 || fx >= 0.78) {
            flankSum += lum;
            flankN++;
          }
        }
        const avgDiff = midN > 0 ? skinDiffSum / midN : 999;
        rowStats.push({
          midLuma: midN > 0 ? midSum / midN : 0,
          flankLuma: flankN > 0 ? flankSum / flankN : 0,
          skinOk: avgDiff < 135,
        });
      }
      let bestPy = -1;
      let bestScore = 3.5;
      for (let py = 5; py < NH - 6; py++) {
        if (!rowStats[py]!.skinOk || !rowStats[py + 3]!.skinOk) continue;
        const ridge = rowStats[py]!.midLuma - rowStats[py]!.flankLuma;
        const shadowDrop = rowStats[py]!.midLuma - rowStats[py + 4]!.midLuma;
        const centerPrior = 1 - Math.abs(py / NH - 0.48) * 0.55;
        const score = (Math.max(0, ridge) * 0.75 + Math.max(0, shadowDrop) * 1.15) * centerPrior;
        if (score > bestScore) {
          bestScore = score;
          bestPy = py;
        }
      }
      if (bestPy >= 0) {
        return [midX, yTop + ((bestPy + 0.5) / NH) * rh];
      }
    } catch {
      /* fallback below */
    }
    return [midX, fallbackY];
  };
  const adamApple = detectAdamApple();

  // 5-point anatomical structure: lowerNeck + rightTrapezius + leftTrapezius + rightShoulder + leftShoulder
  // Finds vertical-to-diagonal inflection point so vertical neck lines stay undeformed and trapezius is disabled when out-of-frame or occluded.
  const detect5PointNeckShoulderYoke = (): {
    lowerNeck: [number, number];
    rightTrapezius: [number, number];
    leftTrapezius: [number, number];
    rightShoulder: [number, number];
    leftShoulder: [number, number];
    hasTrapezius: boolean;
    neckInflectionY: number;
  } => {
    const defLowerY = Math.min(0.95, Math.max(neckBase[1], chinPt[1] + faceH * 0.34));
    const defLowerNeck: [number, number] = [neckBase[0], defLowerY];
    const defRightTrap: [number, number] = [
      Math.max(0.05, defLowerNeck[0] - (faceW * 0.56) / aspect),
      Math.min(0.96, defLowerY + faceH * 0.11),
    ];
    const defLeftTrap: [number, number] = [
      Math.min(0.95, defLowerNeck[0] + (faceW * 0.56) / aspect),
      Math.min(0.96, defLowerY + faceH * 0.11),
    ];
    const defRightShoulder: [number, number] = extras?.kp?.rightShoulder ?? [
      Math.max(0.04, defLowerNeck[0] - (faceW * 1.02) / aspect),
      Math.min(0.98, defLowerY + faceH * 0.24),
    ];
    const defLeftShoulder: [number, number] = extras?.kp?.leftShoulder ?? [
      Math.min(0.96, defLowerNeck[0] + (faceW * 1.02) / aspect),
      Math.min(0.98, defLowerY + faceH * 0.24),
    ];

    const segMask = extras?.segMask;
    const sw = extras?.segWidth;
    const sh = extras?.segHeight;
    const roomBelowChin = 0.98 - chinPt[1];
    if (roomBelowChin < faceH * 0.22) {
      return {
        lowerNeck: defLowerNeck,
        rightTrapezius: defRightTrap,
        leftTrapezius: defLeftTrap,
        rightShoulder: defRightShoulder,
        leftShoulder: defLeftShoulder,
        hasTrapezius: false,
        neckInflectionY: defLowerY,
      };
    }

    if (!segMask || !sw || !sh) {
      return {
        lowerNeck: defLowerNeck,
        rightTrapezius: defRightTrap,
        leftTrapezius: defLeftTrap,
        rightShoulder: defRightShoulder,
        leftShoulder: defLeftShoulder,
        hasTrapezius: roomBelowChin >= faceH * 0.34,
        neckInflectionY: defLowerY,
      };
    }

    const rowSpan = (ny: number): { leftX: number; rightX: number; width: number } | null => {
      const sy = Math.max(0, Math.min(sh - 1, Math.round(ny * (sh - 1))));
      const midPx = Math.max(0, Math.min(sw - 1, Math.round(neckBase[0] * (sw - 1))));
      let lx = midPx;
      let rx = midPx;
      while (lx > 0 && (segMask[sy * sw + (lx - 1)] ?? 0) > 0.45) lx--;
      while (rx < sw - 1 && (segMask[sy * sw + (rx + 1)] ?? 0) > 0.45) rx++;
      if (rx - lx < sw * 0.04) return null;
      return {
        leftX: lx / (sw - 1),
        rightX: rx / (sw - 1),
        width: ((rx - lx) / (sw - 1)) * aspect,
      };
    };

    const rows: Array<{ ny: number; leftX: number; rightX: number; width: number }> = [];
    const scanStart = Math.min(0.94, chinPt[1] + faceH * 0.06);
    const scanEnd = 0.975;
    for (let ny = scanStart; ny <= scanEnd; ny += 0.012) {
      const sp = rowSpan(ny);
      if (sp) rows.push({ ny, ...sp });
    }
    if (rows.length < 4) {
      return {
        lowerNeck: defLowerNeck,
        rightTrapezius: defRightTrap,
        leftTrapezius: defLeftTrap,
        rightShoulder: defRightShoulder,
        leftShoulder: defLeftShoulder,
        hasTrapezius: false,
        neckInflectionY: Math.min(0.94, chinPt[1] + faceH * 0.24),
      };
    }

    // 1) Find narrowest vertical neck waist in upper cervical zone
    let minIdx = 0;
    let minNeckWidth = rows[0]!.width;
    const upperBoundY = Math.min(0.94, chinPt[1] + faceH * 0.32);
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i]!;
      if (r.ny > upperBoundY) break;
      if (r.width < minNeckWidth) {
        minNeckWidth = r.width;
        minIdx = i;
      }
    }

    // 2) Scan downward from neck waist for inflection where vertical neck line begins flaring into trapezius
    let maxLowerW = minNeckWidth;
    for (let i = minIdx; i < rows.length; i++) {
      if (rows[i]!.width > maxLowerW) maxLowerW = rows[i]!.width;
    }

    const neckTooWideOrHidden = minNeckWidth > faceW * 1.18 && maxLowerW < minNeckWidth * 1.24;
    const noFlare = maxLowerW < minNeckWidth * 1.22;
    let inflectionIdx = -1;
    if (!neckTooWideOrHidden && !noFlare) {
      for (let i = Math.max(1, minIdx); i < rows.length - 1; i++) {
        const prev = rows[i - 1]!;
        const curr = rows[i]!;
        const next = rows[i + 1]!;
        const dy = Math.max(0.004, next.ny - prev.ny);
        const slope = (next.width - prev.width) / dy;
        const leftFlare = prev.leftX - next.leftX > (faceW * 0.007) / aspect;
        const rightFlare = next.rightX - prev.rightX > (faceW * 0.007) / aspect;
        if (
          curr.width >= minNeckWidth * 1.03 &&
          slope > faceW * 0.24 &&
          (leftFlare || rightFlare)
        ) {
          inflectionIdx = i;
          break;
        }
      }
    }

    const hasTrapezius = inflectionIdx >= 0 && rows.length - inflectionIdx >= 3;
    if (!hasTrapezius) {
      return {
        lowerNeck: defLowerNeck,
        rightTrapezius: defRightTrap,
        leftTrapezius: defLeftTrap,
        rightShoulder: defRightShoulder,
        leftShoulder: defLeftShoulder,
        hasTrapezius: false,
        neckInflectionY: Math.min(
          0.94,
          Math.max(rows[minIdx]!.ny, chinPt[1] + faceH * 0.22),
          chinPt[1] + faceH * 0.28,
        ),
      };
    }

    const vertEndRow = rows[Math.max(minIdx, inflectionIdx - 1)]!;
    const neckInflectionY = vertEndRow.ny;
    const lowerNeckX = (vertEndRow.leftX + vertEndRow.rightX) * 0.5;
    const lowerNeck: [number, number] = [lowerNeckX, Math.min(0.95, neckInflectionY)];

    // Find shoulder acromion row where outward flare reaches outer width before bottom edge
    let shIdx = rows.length - 1;
    const targetShoulderW = minNeckWidth + (maxLowerW - minNeckWidth) * 0.88;
    for (let i = inflectionIdx + 1; i < rows.length; i++) {
      if (rows[i]!.width >= targetShoulderW || rows[i]!.ny >= neckInflectionY + faceH * 0.24) {
        shIdx = i;
        break;
      }
    }
    const shRow = rows[shIdx]!;
    const trapIdx = Math.min(
      shIdx,
      Math.max(inflectionIdx + 1, Math.round((inflectionIdx + shIdx) * 0.45)),
    );
    const trapRow = rows[trapIdx]!;

    const neckColHalfX = Math.max((minNeckWidth * 0.5) / aspect, (faceW * 0.28) / aspect);
    const rightTrapezius: [number, number] =
      trapRow.leftX < lowerNeckX - neckColHalfX * 1.04
        ? [trapRow.leftX, trapRow.ny]
        : [Math.min(lowerNeckX - neckColHalfX * 1.05, defRightTrap[0]), trapRow.ny];
    const leftTrapezius: [number, number] =
      trapRow.rightX > lowerNeckX + neckColHalfX * 1.04
        ? [trapRow.rightX, trapRow.ny]
        : [Math.max(lowerNeckX + neckColHalfX * 1.05, defLeftTrap[0]), trapRow.ny];

    const rightShoulder: [number, number] =
      extras?.kp?.rightShoulder ??
      (shRow.leftX < rightTrapezius[0] - (faceW * 0.06) / aspect
        ? [shRow.leftX, shRow.ny]
        : [Math.min(rightTrapezius[0] - (faceW * 0.16) / aspect, defRightShoulder[0]), shRow.ny]);
    const leftShoulder: [number, number] =
      extras?.kp?.leftShoulder ??
      (shRow.rightX > leftTrapezius[0] + (faceW * 0.06) / aspect
        ? [shRow.rightX, shRow.ny]
        : [Math.max(leftTrapezius[0] + (faceW * 0.16) / aspect, defLeftShoulder[0]), shRow.ny]);

    return {
      lowerNeck,
      rightTrapezius,
      leftTrapezius,
      rightShoulder,
      leftShoulder,
      hasTrapezius: true,
      neckInflectionY,
    };
  };

  const {
    lowerNeck,
    rightTrapezius,
    leftTrapezius,
    rightShoulder,
    leftShoulder,
    hasTrapezius,
    neckInflectionY,
  } = detect5PointNeckShoulderYoke();

  const hands: HandKeypoints[] = extras?.hands ?? extras?.kp?.hands ?? [];
  const analysis = analyzePhotoFeatures(
    img,
    landmarks,
    restMouthRatio,
    rightEyeCenter,
    leftEyeCenter,
    rEyeShape.center,
    lEyeShape.center,
    rightEyeAxis,
    leftEyeAxis,
    rEyeShape.curve,
    lEyeShape.curve,
    rIrisR,
    lIrisR,
    aspect,
  );

  const restCenterY = (landmarks[13]![1] + landmarks[14]![1]) * 0.5;
  const restCenterX = (landmarks[61]![0] + landmarks[291]![0]) * 0.5;
  const projMouthY = (idx: number) => {
    const pt = landmarks[idx]!;
    return (
      ((pt[0] - restCenterX) * aspect * mouthAxis.vx + (pt[1] - restCenterY) * mouthAxis.vy) /
      mouthW
    );
  };
  const restMouthTopY = Math.min(-0.004, projMouthY(13));
  const restMouthBotY = Math.max(0.004, projMouthY(14));

  const rCheekAnchor = landmarks[50] ?? landmarks[118] ?? landmarks[234]!;
  const rCheekTop = landmarks[117] ?? rCheekAnchor;
  const rightCheekPos: [number, number] = [
    (rCheekAnchor[0] + rCheekTop[0]) * 0.5,
    (rCheekAnchor[1] + rCheekTop[1]) * 0.5,
  ];
  const lCheekAnchor = landmarks[280] ?? landmarks[347] ?? landmarks[454]!;
  const lCheekTop = landmarks[346] ?? lCheekAnchor;
  const leftCheekPos: [number, number] = [
    (lCheekAnchor[0] + lCheekTop[0]) * 0.5,
    (lCheekAnchor[1] + lCheekTop[1]) * 0.5,
  ];
  const rightInnerCanthus: [number, number] = [landmarks[133]![0], landmarks[133]![1]];
  const leftInnerCanthus: [number, number] = [landmarks[362]![0], landmarks[362]![1]];

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
    restMouthTopY,
    restMouthBotY,
    hasVisibleTeeth: analysis.hasVisibleTeeth,
    isMouthParted: analysis.isMouthParted,
    restTeethDepth: analysis.restTeethDepth,
    yaw,
    stylized,
    rightPupil: analysis.rightPupil,
    leftPupil: analysis.leftPupil,
    rightEyeCenter: rEyeShape.center,
    leftEyeCenter: lEyeShape.center,
    rightEyeRadius: [rEyeRx, rEyeRy],
    leftEyeRadius: [lEyeRx, lEyeRy],
    rightEyeCurve: analysis.rightEyeCurve,
    leftEyeCurve: analysis.leftEyeCurve,
    rightIrisRadius: analysis.rightIrisRadius,
    leftIrisRadius: analysis.leftIrisRadius,
    rightEar,
    leftEar,
    hairTop,
    hairLeft,
    hairRight,
    neckBase,
    lowerNeck,
    adamApple,
    rightTrapezius,
    leftTrapezius,
    rightShoulder,
    leftShoulder,
    rightCheekPos,
    leftCheekPos,
    rightInnerCanthus,
    leftInnerCanthus,
    hasTrapezius,
    neckInflectionY,
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
  opts?: { localOnly?: boolean; styleHint?: "auto" | "human" | "anime" },
): Promise<FacePts | null> {
  const styleTag = opts?.styleHint && opts.styleHint !== "auto" ? `:${opts.styleHint}` : "";
  const cacheKey = `${key}${styleTag}`;
  if (cache.has(cacheKey)) return cache.get(cacheKey)!;
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

    const detCanvas = toDetectionCanvas(img, 1024);

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
    let stylized = opts?.styleHint === "anime";
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
    // Discard segmentation mask if almost empty (<8%), otherwise snap soft border probabilities tightly to real image edges
    let usableSegMask = segMask;
    if (usableSegMask && segWidth && segHeight) {
      let sum = 0;
      for (let i = 0; i < usableSegMask.length; i++) {
        if (usableSegMask[i]! > 0.5) sum++;
      }
      const frac = sum / usableSegMask.length;
      if (frac < 0.08 || frac > 0.96) {
        usableSegMask = undefined;
      } else {
        const detCtx = detCanvas.getContext("2d", { willReadFrequently: true });
        if (detCtx) {
          const detW = detCanvas.width;
          const detH = detCanvas.height;
          const detData = detCtx.getImageData(0, 0, detW, detH).data;
          const snapped = new Float32Array(usableSegMask);
          const sampleDetRgb = (sx: number, sy: number) => {
            const px = Math.min(detW - 1, Math.max(0, Math.round((sx / segWidth!) * (detW - 1))));
            const py = Math.min(detH - 1, Math.max(0, Math.round((sy / segHeight!) * (detH - 1))));
            const o = (py * detW + px) * 4;
            return [detData[o]!, detData[o + 1]!, detData[o + 2]!] as const;
          };
          for (let y = 2; y < segHeight - 2; y++) {
            for (let x = 2; x < segWidth - 2; x++) {
              const idx = y * segWidth + x;
              const p = usableSegMask[idx]!;
              if (p <= 0.04 || p >= 0.96) continue;
              const [r, g, b] = sampleDetRgb(x, y);
              let fgW = 0;
              let fgSim = 0;
              let bgW = 0;
              let bgSim = 0;
              for (const [dx, dy] of [
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1],
                [-2, 0],
                [2, 0],
                [0, -2],
                [0, 2],
                [-1, -1],
                [1, 1],
                [-2, 2],
                [2, -2],
              ] as const) {
                const np = usableSegMask[(y + dy) * segWidth + (x + dx)]!;
                const [nr, ng, nb] = sampleDetRgb(x + dx, y + dy);
                const diff = Math.abs(r - nr) + Math.abs(g - ng) + Math.abs(b - nb);
                const sim = Math.exp(-diff / 36);
                if (np > 0.72) {
                  fgW += 1;
                  fgSim += sim;
                } else if (np < 0.24) {
                  bgW += 1;
                  bgSim += sim;
                }
              }
              if (fgW > 0 && bgW > 0) {
                const fAvg = fgSim / fgW;
                const bAvg = bgSim / bgW;
                const guide = fAvg / Math.max(1e-4, fAvg + bAvg);
                snapped[idx] = Math.max(0, Math.min(1, p * 0.35 + guide * 0.65));
              }
            }
          }
          usableSegMask = snapped;
        }
      }
    }

    stylized = false;

    const out = buildFacePts(img, landmarks, faceTris, ovalLoop, aspect, stylized, {
      kp: visionKp,
      hands: finalHands,
      segMask: usableSegMask,
      segWidth: usableSegMask ? segWidth : undefined,
      segHeight: usableSegMask ? segHeight : undefined,
    });
    cache.set(cacheKey, out);
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
