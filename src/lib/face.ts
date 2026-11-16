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

      // Pass 1: Estimate sclera reference luma and refine exact cornea center & outer limbus radius
      const eyeLumas: number[] = [];
      const eyePixels: Array<{
        nx: number;
        ny: number;
        fy: number;
        luma: number;
        r: number;
        g: number;
        b: number;
      }> = [];
      for (let py = 0; py < cropH; py++) {
        const fy = (py + 0.5) / cropH;
        if (fy < 0.2 || fy > 0.88) continue;
        for (let px = 0; px < cropW; px++) {
          if (!pointInPoly(px + 0.5, py + 0.5, eyePoly)) continue;
          const nx = eMinX + ((px + 0.5) / cropW) * ew;
          const ny = eMinY + fy * eh;
          const off = (py * cropW + px) * 4;
          const r = edata[off]!;
          const g = edata[off + 1]!;
          const b = edata[off + 2]!;
          const luma = 0.299 * r + 0.587 * g + 0.114 * b;
          eyeLumas.push(luma);
          eyePixels.push({ nx, ny, fy, luma, r, g, b });
        }
      }

      let refinedPupil: [number, number] = [pupil[0], pupil[1]];
      let refinedIrisRad = irisRad * 1.16;

      if (eyeLumas.length >= 20) {
        const sortedL = [...eyeLumas].sort((a, b) => a - b);
        const scleraLuma =
          sortedL[Math.min(sortedL.length - 1, Math.floor(sortedL.length * 0.78))]!;
        const darkCut = scleraLuma * 0.76;

        let sumW = 0;
        let sumX = 0;
        let sumY = 0;
        const eyeHalfW = Math.max(0.01, ew * aspect * 0.5);
        for (const p of eyePixels) {
          if (p.fy < 0.26 || p.fy > 0.84) continue;
          // Exclude inner/outer eye corners (caruncle/shadow) so dark corner pixels never pull pupils cross-eyed
          const relCenter = Math.abs(
            (p.nx - eyeCenter[0]) * aspect * axis.ux + (p.ny - eyeCenter[1]) * axis.uy,
          );
          if (relCenter > eyeHalfW * 0.56) continue;
          const d0 = Math.hypot((p.nx - pupil[0]) * aspect, p.ny - pupil[1]);
          if (d0 > irisRad * 1.18 || p.luma >= darkCut) continue;
          const w = Math.pow(scleraLuma - p.luma, 2);
          sumW += w;
          sumX += p.nx * w;
          sumY += p.ny * w;
        }
        if (sumW > 0) {
          const cx = sumX / sumW;
          const cy = sumY / sumW;
          const shiftDist = Math.hypot((cx - pupil[0]) * aspect, cy - pupil[1]);
          const maxShift = irisRad * 0.16;
          if (shiftDist > 1e-6) {
            const clampScale = Math.min(1, maxShift / shiftDist);
            refinedPupil = [
              pupil[0] + (cx - pupil[0]) * clampScale,
              pupil[1] + (cy - pupil[1]) * clampScale,
            ];
          }
        }

        // Measure crisp inner-limbus cornea radius without grabbing surrounding lashes/sclera
        const limbusCut = scleraLuma * 0.81;
        const radialDists: number[] = [];
        for (const p of eyePixels) {
          if (p.fy < 0.3 || p.fy > 0.82) continue;
          const d = Math.hypot((p.nx - refinedPupil[0]) * aspect, p.ny - refinedPupil[1]);
          if (d <= irisRad * 1.24 && p.luma < limbusCut) {
            radialDists.push(d);
          }
        }
        if (radialDists.length >= 12) {
          radialDists.sort((a, b) => a - b);
          const p85 =
            radialDists[Math.min(radialDists.length - 1, Math.floor(radialDists.length * 0.85))]!;
          refinedIrisRad = Math.min(eyeHalfW * 0.54, Math.max(irisRad * 0.98, p85));
        }
      }

      type ScleraCand = { score: number; rgb: [number, number, number] };
      const negCands: ScleraCand[] = [];
      const posCands: ScleraCand[] = [];
      const allCands: ScleraCand[] = [];
      const fallbackCands: ScleraCand[] = [];
      const eyeHalfW = Math.max(0.01, ew * aspect * 0.5);

      for (const p of eyePixels) {
        // Aspect-corrected cornea + limbus exclusion across full clean exposed sclera band
        const dPupil = Math.hypot((p.nx - refinedPupil[0]) * aspect, p.ny - refinedPupil[1]);
        if (dPupil < refinedIrisRad * 1.18) continue;
        if (p.fy < 0.36 || p.fy > 0.72) continue;

        const relX = (p.nx - eyeCenter[0]) * aspect * axis.ux + (p.ny - eyeCenter[1]) * axis.uy;
        if (Math.abs(relX) > eyeHalfW * 0.78) continue;

        const maxC = Math.max(p.r, p.g, p.b);
        const minC = Math.min(p.r, p.g, p.b);
        const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
        const redness = p.r - (p.g + p.b) * 0.5;
        if (p.luma >= 45) {
          fallbackCands.push({
            score: p.luma - sat * 18,
            rgb: [p.r, p.g, p.b],
          });
        }

        // Rank brightest clean neutral sclera highest, rejecting upper-lid shadow, lash edge and caruncle red
        if (p.luma < 92 || sat > 0.3 || redness > 24) continue;
        const item: ScleraCand = {
          score: p.luma - redness * 1.25 - sat * 48,
          rgb: [p.r, p.g, p.b],
        };
        allCands.push(item);
        if (relX < -refinedIrisRad * 0.14) negCands.push(item);
        else if (relX > refinedIrisRad * 0.14) posCands.push(item);
      }

      const summarize = (list: ScleraCand[], fallbackColor: [number, number, number]) => {
        if (list.length === 0) return fallbackColor;
        const sorted = [...list].sort((a, b) => a.score - b.score);
        // Clean upper-bright percentile (64%-92%) picks pure sclera without shadow bleed or highlight clipping
        const sIdx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.64));
        const eIdx = Math.min(sorted.length, Math.max(sIdx + 1, Math.floor(sorted.length * 0.92)));
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

      const sourcePool = allCands.length >= 4 ? allCands : fallbackCands;
      const mid = summarize(sourcePool, [224, 220, 216]);
      const midLuma = 0.299 * mid[0] + 0.587 * mid[1] + 0.114 * mid[2];
      const rawNeg = negCands.length >= 3 ? summarize(negCands, mid) : mid;
      const rawPos = posCands.length >= 3 ? summarize(posCands, mid) : mid;
      // Preserve authentic directional light/shadow difference between inner & outer sclera while blocking noisy extremes
      const anchorSide = (side: [number, number, number]): [number, number, number] => {
        const sLuma = 0.299 * side[0] + 0.587 * side[1] + 0.114 * side[2];
        const lift = sLuma > 1 ? Math.max(0.82, Math.min(1.08, (midLuma * 0.98) / sLuma)) : 1;
        const clampedS =
          Math.max(0.84 * midLuma, Math.min(1.08 * midLuma, sLuma)) / Math.max(1, sLuma);
        const factor = sLuma < midLuma * 0.8 || sLuma > midLuma * 1.12 ? lift : clampedS;
        const adj: [number, number, number] = [
          Math.min(255, side[0] * factor),
          Math.min(255, side[1] * factor),
          Math.min(255, side[2] * factor),
        ];
        return [
          Math.round(mid[0] * 0.25 + adj[0] * 0.75),
          Math.round(mid[1] * 0.25 + adj[1] * 0.75),
          Math.round(mid[2] * 0.25 + adj[2] * 0.75),
        ];
      };
      const neg = anchorSide(rawNeg);
      const pos = anchorSide(rawPos);
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

      // Bounding region anchored cleanly to left/right commissures with vertical margin for upper tarsal skin & lower lid
      const padNormX = (halfW * 1.25) / aspect;
      const padNormYTop = scaleY * 2.4;
      const padNormYBot = scaleY * 1.8;
      const rx0 = Math.max(0, eyeCenter[0] - padNormX);
      const rx1 = Math.min(1, eyeCenter[0] + padNormX);
      const ry0 = Math.max(0, eyeCenter[1] - padNormYTop);
      const ry1 = Math.min(1, eyeCenter[1] + padNormYBot);
      const rw = Math.max(0.012, rx1 - rx0);
      const rh = Math.max(0.012, ry1 - ry0);

      const CW = 96;
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

        // 3) Sample clean tarsal upper lid skin strictly ABOVE lash root (excluding black mascara/lashes)
        if (Math.abs(ex) <= 0.68) {
          for (let dy = 0.42; dy <= 1.05; dy += 0.18) {
            const sp = sampleEyCoord(ex, detectedUp[ci]! - dy);
            if (sp.luma >= Math.max(62, skinRefLuma * 0.68)) {
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

      return {
        ...baseRes,
        refinedCurve: {
          ...curve,
          upper: [upperArch[0], upperArch[2], upperArch[5], upperArch[7]],
          lower: [lowerArch[0], lowerArch[2], lowerArch[5], lowerArch[7]],
          upperArch,
          lowerArch,
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
    );
    const lSclera = refineEyeAndSample(
      LEFT_EYE_LOOP,
      leftEyeCenter,
      leftPupil,
      leftEyeAxis,
      leftEyeCurve,
      leftIrisRadius,
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
              if (p <= 0.08 || p >= 0.92) continue;
              const [r, g, b] = sampleDetRgb(x, y);
              let fgW = 0;
              let fgSim = 0;
              let bgW = 0;
              let bgSim = 0;
              for (const [dx, dy] of [
                [-3, 0],
                [3, 0],
                [0, -3],
                [0, 3],
                [-2, -2],
                [2, 2],
              ] as const) {
                const np = usableSegMask[(y + dy) * segWidth + (x + dx)]!;
                const [nr, ng, nb] = sampleDetRgb(x + dx, y + dy);
                const diff = Math.abs(r - nr) + Math.abs(g - ng) + Math.abs(b - nb);
                const sim = Math.exp(-diff / 48);
                if (np > 0.8) {
                  fgW += 1;
                  fgSim += sim;
                } else if (np < 0.2) {
                  bgW += 1;
                  bgSim += sim;
                }
              }
              if (fgW > 0 && bgW > 0) {
                const fAvg = fgSim / fgW;
                const bAvg = bgSim / bgW;
                const guide = fAvg / Math.max(1e-4, fAvg + bAvg);
                snapped[idx] = Math.max(0, Math.min(1, p * 0.45 + guide * 0.55));
              }
            }
          }
          usableSegMask = snapped;
        }
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
