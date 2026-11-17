import Delaunator from "delaunator";
import { CANONICAL_FACE_XYZ } from "@/lib/canonicalFace";
import {
  INNER_LOWER_LIP,
  INNER_LIP_LOOP,
  INNER_UPPER_LIP,
  LEFT_EYE_LOOP,
  OUTER_LOWER_LIP,
  OUTER_UPPER_LIP,
  RIGHT_EYE_LOOP,
  type FacePts,
} from "@/lib/face";

// Vertex attribute stride: 20 floats per vertex
// [0..3]   aPosUv:   vec4(nx, ny, uvx, uvy)
// [4..7]   aHeadRig: vec4(headW, torsoW, jawW, depthZ)
// [8..11]  aFaceRig: vec4(upperLipW, cornerW, cheekW, noseW)
// [12..15] aBrowRig: vec4(innerBrowW, outerBrowW, eyeOrbitW, sideSign)
// [16..19] aHandRig: vec4(hand0W, hand1W, mouthLipT, mouthX)
export const AVATAR_VERTEX_STRIDE = 20;

export type BuiltAvatarMeshes = {
  bgVerts: Float32Array;
  bgIdx: Uint16Array;
  faceVerts: Float32Array;
  faceIdx: Uint16Array;
  wireIdx: Uint16Array;
  mouthVerts: Float32Array;
  mouthIdx: Uint16Array;
  neckPivot: [number, number];
  jawPivot: [number, number];
  mouthCenter: [number, number];
};

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0 || 1e-6)));
  return t * t * (3 - 2 * t);
}

function pointToSegmentDist(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  aspect: number,
): number {
  const dx = (bx - ax) * aspect;
  const dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-8) return Math.hypot((px - ax) * aspect, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * aspect * dx + (py - ay) * dy) / lenSq));
  return Math.hypot((px - (ax + t * (bx - ax))) * aspect, py - (ay + t * (by - ay)));
}

/**
 * Generates three GPU-ready canvases from the portrait:
 * 1) `fgMaskCanvas`: Soft cutout alpha mask isolating head, hair, ears, neck & torso.
 * 2) `bgCanvas`: Inpainted background behind the cutout silhouette so 3D turns never leave dark seams.
 * 3) `bgBlurCanvas`: Smooth optical bokeh version of `bgCanvas`.
 */
export function buildPortraitLayers(
  img: HTMLImageElement,
  face: FacePts,
): {
  fgMaskCanvas: HTMLCanvasElement;
  bgCanvas: HTMLCanvasElement;
  bgBlurCanvas: HTMLCanvasElement;
} {
  const srcW = img.naturalWidth || img.width || 512;
  const srcH = img.naturalHeight || img.height || 512;
  const maxDim = 448;
  const scale = Math.min(1, maxDim / Math.max(srcW, srcH));
  const W = Math.max(64, Math.round(srcW * scale));
  const H = Math.max(64, Math.round(srcH * scale));
  const aspect = face.aspect;

  const fgMaskCanvas = document.createElement("canvas");
  fgMaskCanvas.width = W;
  fgMaskCanvas.height = H;
  const maskCtx = fgMaskCanvas.getContext("2d", { willReadFrequently: true });

  const bgCanvas = document.createElement("canvas");
  bgCanvas.width = W;
  bgCanvas.height = H;
  const bgCtx = bgCanvas.getContext("2d", { willReadFrequently: true });

  const bgBlurCanvas = document.createElement("canvas");
  bgBlurCanvas.width = W;
  bgBlurCanvas.height = H;
  const blurCtx = bgBlurCanvas.getContext("2d");

  if (!maskCtx || !bgCtx) {
    return { fgMaskCanvas, bgCanvas, bgBlurCanvas };
  }

  bgCtx.drawImage(img, 0, 0, W, H);
  const bgImgData = bgCtx.getImageData(0, 0, W, H);
  const maskImgData = maskCtx.createImageData(W, H);

  const hasSegMask = !!(face.segMask && face.segWidth && face.segHeight);
  const sampleSeg = (nx: number, ny: number): number => {
    const mask = face.segMask;
    const sw = face.segWidth;
    const sh = face.segHeight;
    if (!mask || !sw || !sh) return 0;
    const fx = Math.max(0, Math.min(sw - 1, nx * (sw - 1)));
    const fy = Math.max(0, Math.min(sh - 1, ny * (sh - 1)));
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(sw - 1, x0 + 1);
    const y1 = Math.min(sh - 1, y0 + 1);
    const tx = fx - x0;
    const ty = fy - y0;
    const v00 = mask[y0 * sw + x0] ?? 0;
    const v10 = mask[y0 * sw + x1] ?? 0;
    const v01 = mask[y1 * sw + x0] ?? 0;
    const v11 = mask[y1 * sw + x1] ?? 0;
    return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
  };

  const pts = face.landmarks;
  const chin = pts[152] ?? [face.center[0], face.center[1] + face.faceH * 0.5];
  const crownTopY = Math.min(face.hairTop[1], (pts[10]?.[1] ?? face.center[1]) - face.faceH * 0.34);
  const headCenterX = (face.hairLeft[0] + face.hairRight[0] + face.center[0] * 2) * 0.25;
  const headCenterY = (crownTopY + chin[1]) * 0.5;
  const headRx = Math.max(
    face.faceW * 0.72,
    Math.abs(face.hairRight[0] - face.hairLeft[0]) * aspect * 0.58,
  );
  const headRy = Math.max(face.faceH * 0.68, (chin[1] - crownTopY) * 0.55);
  const innerFaceRx = face.faceW * 0.48;
  const innerFaceRy = face.faceH * 0.5;

  const alphaArr = new Float32Array(W * H);

  for (let py = 0; py < H; py++) {
    const ny = (py + 0.5) / H;
    for (let px = 0; px < W; px++) {
      const nx = (px + 0.5) / W;

      // Protect core facial interior (eyes, nose, cheeks, mouth) so landmarks never disappear
      const coreDist = Math.hypot(
        ((nx - face.center[0]) * aspect) / Math.max(0.05, innerFaceRx),
        (ny - face.center[1]) / Math.max(0.05, innerFaceRy),
      );
      const faceCore = 1 - smoothstep(0.78, 0.96, coreDist);

      let fg: number;
      if (hasSegMask) {
        const seg = sampleSeg(nx, ny);
        const segSmooth = smoothstep(0.24, 0.64, seg);
        fg = Math.max(faceCore, segSmooth);
      } else {
        const headDist = Math.hypot(
          ((nx - headCenterX) * aspect) / Math.max(0.08, headRx),
          (ny - headCenterY) / Math.max(0.08, headRy),
        );
        const headHalo = 1 - smoothstep(0.92, 1.04, headDist);
        let bodyMask = 0;
        if (ny > chin[1] - face.faceH * 0.12) {
          const neckHalfW =
            face.faceW *
            (0.46 +
              smoothstep(chin[1], Math.min(1, face.neckBase[1] + face.faceH * 0.3), ny) * 0.85);
          const lat = Math.abs((nx - face.neckBase[0]) * aspect) / Math.max(0.08, neckHalfW);
          bodyMask =
            smoothstep(chin[1] - face.faceH * 0.12, chin[1] + face.faceH * 0.06, ny) *
            (1 - smoothstep(0.86, 1.02, lat));
        }
        fg = Math.max(faceCore, headHalo, bodyMask);
      }

      alphaArr[py * W + px] = Math.max(0, Math.min(1, fg));
    }
  }

  for (let i = 0; i < W * H; i++) {
    const fg = alphaArr[i]!;
    const aByte = Math.round(fg * 255);
    const o = i * 4;
    maskImgData.data[o] = aByte;
    maskImgData.data[o + 1] = aByte;
    maskImgData.data[o + 2] = aByte;
    maskImgData.data[o + 3] = 255;
  }
  maskCtx.putImageData(maskImgData, 0, 0);

  // Dilate occlusion mask 3px around foreground contour so turning reveals clean background without silhouette rings
  const holeMask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let inHole = false;
      for (let dy = -3; dy <= 3 && !inHole; dy++) {
        const sy = y + dy;
        if (sy < 0 || sy >= H) continue;
        for (let dx = -3; dx <= 3; dx++) {
          const sx = x + dx;
          if (sx < 0 || sx >= W) continue;
          if (dx * dx + dy * dy <= 10 && alphaArr[sy * W + sx]! >= 0.08) {
            inHole = true;
            break;
          }
        }
      }
      if (inHole) holeMask[y * W + x] = 1;
    }
  }

  // Alpha-normalized multi-scale push-pull inpainting
  type Level = { w: number; h: number; rgba: Float32Array };
  const baseRgba = new Float32Array(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const o = i * 4;
    if (!holeMask[i]) {
      baseRgba[o] = bgImgData.data[o]!;
      baseRgba[o + 1] = bgImgData.data[o + 1]!;
      baseRgba[o + 2] = bgImgData.data[o + 2]!;
      baseRgba[o + 3] = 1;
    }
  }

  const levels: Level[] = [{ w: W, h: H, rgba: baseRgba }];
  let cw = W;
  let ch = H;
  while (cw > 2 || ch > 2) {
    const nw = Math.max(1, Math.ceil(cw / 2));
    const nh = Math.max(1, Math.ceil(ch / 2));
    const prev = levels[levels.length - 1]!;
    const nextRgba = new Float32Array(nw * nh * 4);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let sw = 0;
        for (let dy = 0; dy < 2; dy++) {
          const sy = Math.min(prev.h - 1, y * 2 + dy);
          for (let dx = 0; dx < 2; dx++) {
            const sx = Math.min(prev.w - 1, x * 2 + dx);
            const so = (sy * prev.w + sx) * 4;
            const w = prev.rgba[so + 3]!;
            if (w > 0) {
              sr += prev.rgba[so]! * w;
              sg += prev.rgba[so + 1]! * w;
              sb += prev.rgba[so + 2]! * w;
              sw += w;
            }
          }
        }
        if (sw > 0) {
          const no = (y * nw + x) * 4;
          nextRgba[no] = sr / sw;
          nextRgba[no + 1] = sg / sw;
          nextRgba[no + 2] = sb / sw;
          nextRgba[no + 3] = Math.min(1, sw / 4);
        }
      }
    }
    levels.push({ w: nw, h: nh, rgba: nextRgba });
    cw = nw;
    ch = nh;
    if (cw === 1 && ch === 1) break;
  }

  for (let li = levels.length - 2; li >= 0; li--) {
    const cur = levels[li]!;
    const coarse = levels[li + 1]!;
    for (let y = 0; y < cur.h; y++) {
      const cy = Math.min(coarse.h - 1, (y + 0.5) * 0.5 - 0.5);
      const y0 = Math.max(0, Math.floor(cy));
      const y1 = Math.min(coarse.h - 1, y0 + 1);
      const ty = Math.max(0, cy - y0);
      for (let x = 0; x < cur.w; x++) {
        const co = (y * cur.w + x) * 4;
        if (cur.rgba[co + 3]! >= 0.999) continue;
        const cx = Math.min(coarse.w - 1, (x + 0.5) * 0.5 - 0.5);
        const x0 = Math.max(0, Math.floor(cx));
        const x1 = Math.min(coarse.w - 1, x0 + 1);
        const tx = Math.max(0, cx - x0);

        const sampleCoarse = (px: number, py: number, cIdx: number) =>
          coarse.rgba[(py * coarse.w + px) * 4 + cIdx]!;
        const w00 = sampleCoarse(x0, y0, 3);
        const w10 = sampleCoarse(x1, y0, 3);
        const w01 = sampleCoarse(x0, y1, 3);
        const w11 = sampleCoarse(x1, y1, 3);
        const k00 = (1 - tx) * (1 - ty) * w00;
        const k10 = tx * (1 - ty) * w10;
        const k01 = (1 - tx) * ty * w01;
        const k11 = tx * ty * w11;
        const kSum = k00 + k10 + k01 + k11;
        if (kSum <= 1e-6) continue;

        const cr =
          (sampleCoarse(x0, y0, 0) * k00 +
            sampleCoarse(x1, y0, 0) * k10 +
            sampleCoarse(x0, y1, 0) * k01 +
            sampleCoarse(x1, y1, 0) * k11) /
          kSum;
        const cg =
          (sampleCoarse(x0, y0, 1) * k00 +
            sampleCoarse(x1, y0, 1) * k10 +
            sampleCoarse(x0, y1, 1) * k01 +
            sampleCoarse(x1, y1, 1) * k11) /
          kSum;
        const cb =
          (sampleCoarse(x0, y0, 2) * k00 +
            sampleCoarse(x1, y0, 2) * k10 +
            sampleCoarse(x0, y1, 2) * k01 +
            sampleCoarse(x1, y1, 2) * k11) /
          kSum;

        const wCur = cur.rgba[co + 3]!;
        cur.rgba[co] = cur.rgba[co]! * wCur + cr * (1 - wCur);
        cur.rgba[co + 1] = cur.rgba[co + 1]! * wCur + cg * (1 - wCur);
        cur.rgba[co + 2] = cur.rgba[co + 2]! * wCur + cb * (1 - wCur);
        cur.rgba[co + 3] = 1;
      }
    }
  }

  const filled = levels[0]!.rgba;
  for (let pass = 0; pass < 2; pass++) {
    const copy = new Float32Array(filled);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const idx = y * W + x;
        if (!holeMask[idx]) continue;
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let sw = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nIdx = (y + dy) * W + (x + dx);
            const w = holeMask[nIdx] ? 1.0 : 1.5;
            const o = nIdx * 4;
            sr += copy[o]! * w;
            sg += copy[o + 1]! * w;
            sb += copy[o + 2]! * w;
            sw += w;
          }
        }
        const o = idx * 4;
        filled[o] = sr / sw;
        filled[o + 1] = sg / sw;
        filled[o + 2] = sb / sw;
      }
    }
  }

  for (let i = 0; i < W * H; i++) {
    const o = i * 4;
    if (holeMask[i]) {
      bgImgData.data[o] = Math.round(Math.max(0, Math.min(255, filled[o]!)));
      bgImgData.data[o + 1] = Math.round(Math.max(0, Math.min(255, filled[o + 1]!)));
      bgImgData.data[o + 2] = Math.round(Math.max(0, Math.min(255, filled[o + 2]!)));
    }
    bgImgData.data[o + 3] = 255;
  }
  bgCtx.putImageData(bgImgData, 0, 0);

  if (blurCtx) {
    const pad = 20;
    const tmp = document.createElement("canvas");
    tmp.width = W + pad * 2;
    tmp.height = H + pad * 2;
    const tCtx = tmp.getContext("2d");
    if (tCtx) {
      tCtx.drawImage(bgCanvas, 0, 0, W, H, pad, pad, W, H);
      tCtx.drawImage(bgCanvas, 0, 0, W, 1, pad, 0, W, pad);
      tCtx.drawImage(bgCanvas, 0, H - 1, W, 1, pad, H + pad, W, pad);
      tCtx.drawImage(bgCanvas, 0, 0, 1, H, 0, pad, pad, H);
      tCtx.drawImage(bgCanvas, W - 1, 0, 1, H, W + pad, pad, pad, H);
      tCtx.drawImage(bgCanvas, 0, 0, 1, 1, 0, 0, pad, pad);
      tCtx.drawImage(bgCanvas, W - 1, 0, 1, 1, W + pad, 0, pad, pad);
      tCtx.drawImage(bgCanvas, 0, H - 1, 1, 1, 0, H + pad, pad, pad);
      tCtx.drawImage(bgCanvas, W - 1, H - 1, 1, 1, W + pad, H + pad, pad, pad);
      blurCtx.filter = "blur(14px)";
      blurCtx.drawImage(tmp, pad, pad, W, H, 0, 0, W, H);
      blurCtx.filter = "none";
    } else {
      blurCtx.drawImage(bgCanvas, 0, 0);
    }
  }

  return { fgMaskCanvas, bgCanvas, bgBlurCanvas };
}

/**
 * Builds the continuous Delaunay portrait mesh, full-screen background quad,
 * oral interior strip, and face-only wireframe indices.
 */
export function buildAvatarMeshes(face: FacePts): BuiltAvatarMeshes {
  const aspect = Math.max(0.25, face.aspect || 1);
  const pts = face.landmarks;
  const point = (i: number): [number, number, number] =>
    pts[i] ?? [face.center[0], face.center[1], 0];

  const chin = point(152);
  const noseTip = point(1);
  const mouthCenter: [number, number] = [
    (point(61)[0] + point(291)[0]) * 0.5,
    (point(13)[1] + point(14)[1]) * 0.5,
  ];
  const neckPivot: [number, number] = [face.neckBase[0], face.neckBase[1]];
  const jawPivot: [number, number] = [
    (point(234)[0] + point(454)[0]) * 0.5,
    (point(234)[1] + point(454)[1]) * 0.5,
  ];

  const STRIDE = AVATAR_VERTEX_STRIDE;

  // Static full-frame quad for background pass (uPass = -1.0)
  const bgVerts = new Float32Array(4 * STRIDE);
  const quadCoords: Array<[number, number]> = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
  ];
  for (let i = 0; i < 4; i++) {
    const off = i * STRIDE;
    bgVerts[off] = quadCoords[i]![0];
    bgVerts[off + 1] = quadCoords[i]![1];
    bgVerts[off + 2] = quadCoords[i]![0];
    bgVerts[off + 3] = quadCoords[i]![1];
    bgVerts[off + 7] = 0.0;
  }
  const bgIdx = new Uint16Array([0, 2, 1, 1, 2, 3]);

  // Unified cranium plateau covering face, ears & hair smoothly
  const hairSpanHalfW = Math.max(
    face.faceW * 0.85,
    Math.abs(face.hairRight[0] - face.center[0]) * aspect,
    Math.abs(face.hairLeft[0] - face.center[0]) * aspect,
    Math.abs(face.rightEar[0] - face.center[0]) * aspect,
    Math.abs(face.leftEar[0] - face.center[0]) * aspect,
  );
  const craniumX = (face.center[0] * 2 + face.rightEar[0] + face.leftEar[0]) * 0.25;
  const craniumY = (face.hairTop[1] + chin[1]) * 0.42;
  const radX = Math.max(face.faceW * 1.45, hairSpanHalfW * 1.65);
  const radY = Math.max(face.faceH * 1.28, (chin[1] - face.hairTop[1]) * 0.96);
  const neckFadeStartY = chin[1] + face.faceH * 0.08;
  const neckFadeEndY = Math.min(
    0.96,
    Math.max(face.neckBase[1] + face.faceH * 0.04, chin[1] + face.faceH * 0.42),
  );

  const sampleHairSeg = (nx: number, ny: number): number => {
    const mask = face.segMask;
    const sw = face.segWidth;
    const sh = face.segHeight;
    if (!mask || !sw || !sh) return 0;
    const sx = Math.max(0, Math.min(sw - 1, Math.round(nx * (sw - 1))));
    const sy = Math.max(0, Math.min(sh - 1, Math.round(ny * (sh - 1))));
    return mask[sy * sw + sx] ?? 0;
  };

  const computeHeadWeight = (nx: number, ny: number): number => {
    const rx = ((nx - craniumX) * aspect) / Math.max(0.08, radX);
    const ry = (ny - craniumY) / Math.max(0.08, radY);
    const oval =
      ny <= chin[1]
        ? 1 - smoothstep(1.32, 2.05, Math.hypot(rx, ry))
        : 1 - smoothstep(1.12, 1.85, Math.hypot(rx, ry));
    const segHair =
      ny <= chin[1] + face.faceH * 0.05 ? smoothstep(0.22, 0.58, sampleHairSeg(nx, ny)) : 0;
    const belowChinT = ny > chin[1] ? smoothstep(chin[1], face.neckBase[1], ny) : 0;
    const neckAxisX = face.center[0] + belowChinT * (face.neckBase[0] - face.center[0]);
    const neckLatFade =
      1 - smoothstep(face.faceW * 0.4, face.faceW * 0.66, Math.abs((nx - neckAxisX) * aspect));
    const lateralMask = 1 - belowChinT * (1 - neckLatFade);
    const radial = Math.max(oval, segHair) * lateralMask;
    const neckFade = ny > neckFadeStartY ? 1 - smoothstep(neckFadeStartY, neckFadeEndY, ny) : 1;
    return radial * neckFade;
  };

  const computeEllipsoidDepth = (nx: number, ny: number): number => {
    const relX = (nx - face.center[0]) * aspect;
    const relY = ny - face.center[1];
    const u =
      (relX * face.faceAxis.ux + relY * face.faceAxis.uy) / Math.max(0.06, face.faceW * 0.68);
    const v =
      (relX * face.faceAxis.vx + relY * face.faceAxis.vy) / Math.max(0.06, face.faceH * 0.62);
    const r2 = Math.min(0.95, u * u * 0.72 + v * v * 0.52);
    return Math.sqrt(Math.max(0.05, 1 - r2)) * 0.76 + 0.16;
  };

  const shoulderHalfSpan = Math.max(
    face.faceW * 1.42,
    Math.hypot(
      (face.rightShoulder[0] - face.neckBase[0]) * aspect,
      face.rightShoulder[1] - face.neckBase[1],
    ) * 1.35,
    Math.hypot(
      (face.leftShoulder[0] - face.neckBase[0]) * aspect,
      face.leftShoulder[1] - face.neckBase[1],
    ) * 1.35,
  );
  const computeShoulderRidge = (nx: number, ny: number): number => {
    if (ny < chin[1] + face.faceH * 0.04) return 0;
    const dRight = pointToSegmentDist(
      nx,
      ny,
      face.neckBase[0],
      face.neckBase[1],
      face.rightShoulder[0],
      face.rightShoulder[1],
      aspect,
    );
    const dLeft = pointToSegmentDist(
      nx,
      ny,
      face.neckBase[0],
      face.neckBase[1],
      face.leftShoulder[0],
      face.leftShoulder[1],
      aspect,
    );
    const nearClavicle =
      1 - smoothstep(face.faceW * 0.14, face.faceW * 0.62, Math.min(dRight, dLeft));
    const aboveChestBottom = smoothstep(chin[1] + 0.04 * face.faceH, face.neckBase[1], ny);
    return Math.max(0, Math.min(1, nearClavicle * aboveChestBottom));
  };

  const lipBotY = point(17)[1];
  const jawCenterY = (lipBotY + chin[1]) * 0.52;
  const jawRx = Math.max(0.03, face.faceW * 0.4);
  const jawRy = Math.max(face.faceH * 0.18, (chin[1] - mouthCenter[1]) * 1.16);
  const computeMandibleWeight = (nx: number, ny: number, mpY?: number): number => {
    const eR = Math.hypot(((nx - mouthCenter[0]) * aspect) / jawRx, (ny - jawCenterY) / jawRy);
    const ovalBell = 1 - smoothstep(0.28, 1.06, eR);
    const vertUp =
      mpY !== undefined
        ? smoothstep(-face.mouthH * 0.1, face.mouthH * 0.35, mpY)
        : smoothstep(mouthCenter[1] - 0.01 * face.faceH, lipBotY, ny);
    return ovalBell * vertUp;
  };

  // 1) Dense 56x56 Outer Portrait Grid
  const GX = 56;
  const GY = 56;
  const baseVerts = new Float32Array((GX + 1) * (GY + 1) * STRIDE);
  const hand0 = face.hands[0];
  const hand1 = face.hands[1];

  let ptr = 0;
  for (let iy = 0; iy <= GY; iy++) {
    const ny = iy / GY;
    const borderY = smoothstep(0.0, 0.025, ny) * smoothstep(1.0, 0.975, ny);
    for (let ix = 0; ix <= GX; ix++) {
      const nx = ix / GX;
      const borderX = smoothstep(0.0, 0.025, nx) * smoothstep(1.0, 0.975, nx);
      const borderMask = borderX * borderY;

      const headW = computeHeadWeight(nx, ny) * borderMask;
      const underChinJawW = computeMandibleWeight(nx, ny) * borderMask;
      const shoulderRidgeW = computeShoulderRidge(nx, ny) * borderMask;

      const torsoW =
        Math.max(
          shoulderRidgeW * 0.92,
          smoothstep(chin[1] + 0.06 * face.faceH, neckFadeEndY + 0.04 * face.faceH, ny) *
            (1 -
              smoothstep(
                shoulderHalfSpan * 0.72,
                Math.min(0.98, shoulderHalfSpan * 1.18),
                Math.abs((nx - neckPivot[0]) * aspect),
              )),
        ) * borderMask;

      const depthZ = computeEllipsoidDepth(nx, ny);

      let h0W = 0;
      if (hand0) {
        const d0 = pointToSegmentDist(
          nx,
          ny,
          hand0.wrist[0],
          hand0.wrist[1],
          hand0.fingerTips[0],
          hand0.fingerTips[1],
          aspect,
        );
        h0W =
          (1 - smoothstep(hand0.radius * 0.25, hand0.radius * 1.35, d0)) * borderMask * (1 - headW);
      }
      let h1W = 0;
      if (hand1) {
        const d1 = pointToSegmentDist(
          nx,
          ny,
          hand1.wrist[0],
          hand1.wrist[1],
          hand1.fingerTips[0],
          hand1.fingerTips[1],
          aspect,
        );
        h1W =
          (1 - smoothstep(hand1.radius * 0.25, hand1.radius * 1.35, d1)) * borderMask * (1 - headW);
      }

      baseVerts[ptr++] = nx;
      baseVerts[ptr++] = ny;
      baseVerts[ptr++] = nx;
      baseVerts[ptr++] = ny;
      baseVerts[ptr++] = headW;
      baseVerts[ptr++] = torsoW;
      baseVerts[ptr++] = underChinJawW;
      baseVerts[ptr++] = depthZ;
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = h0W;
      baseVerts[ptr++] = h1W;
      baseVerts[ptr++] = 0; // mouthLipT strictly unused outside mouthVerts
      baseVerts[ptr++] = 0;
    }
  }

  // 2) High-Precision 468-Landmark Face Mesh
  const numPts = Math.min(468, pts.length);
  const faceVerts = new Float32Array(numPts * STRIDE);

  // Normalize facial depth across landmark Z + canonical Z for organic 3D rotation
  let minZ = Infinity;
  let maxZ = -Infinity;
  const rawZ = new Float32Array(numPts);
  for (let i = 0; i < numPts; i++) {
    const ptZ = pts[i]?.[2] ?? 0;
    const canZ = -((CANONICAL_FACE_XYZ[i * 3 + 2] ?? 4) - 4) * 0.015;
    const zVal = Math.abs(ptZ) > 1e-4 ? -(ptZ * 0.72 + canZ * 0.28) : -canZ;
    rawZ[i] = zVal;
    if (zVal < minZ) minZ = zVal;
    if (zVal > maxZ) maxZ = zVal;
  }
  const zSpan = Math.max(0.02, maxZ - minZ);

  const ovalSet = new Set(face.ovalLoop);
  const innerUpperSet = new Set(INNER_UPPER_LIP);
  const innerLowerSet = new Set(INNER_LOWER_LIP);
  const innerLipSet = new Set(INNER_LIP_LOOP);
  const outerUpperSet = new Set(OUTER_UPPER_LIP);
  const outerLowerSet = new Set(OUTER_LOWER_LIP);
  const eyeLoopSet = new Set([...RIGHT_EYE_LOOP, ...LEFT_EYE_LOOP]);

  const rightBrowIndices = [70, 63, 105, 66, 107, 46, 53, 52, 65, 55];
  const leftBrowIndices = [300, 293, 334, 296, 336, 276, 283, 282, 295, 285];
  const rightBrowSet = new Set(rightBrowIndices);
  const leftBrowSet = new Set(leftBrowIndices);

  const browBounds = (idxs: number[], eyeTopY: number) => {
    let minY = 1;
    let maxY = 0;
    let minX = 1;
    let maxX = 0;
    for (const idx of idxs) {
      const p = point(idx);
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
    }
    return { minY, maxY, minX, maxX, eyeTopY };
  };
  const rBrowBox = browBounds(rightBrowIndices, point(159)[1]);
  const lBrowBox = browBounds(leftBrowIndices, point(386)[1]);

  const rNoseWing = pts[49] ?? pts[102] ?? noseTip;
  const lNoseWing = pts[279] ?? pts[331] ?? noseTip;
  const rCheekPt = pts[50] ?? pts[118] ?? point(205);
  const lCheekPt = pts[280] ?? pts[347] ?? point(425);
  const rCorner = point(61);
  const lCorner = point(291);
  const innerMouthHalfW = Math.max(
    0.02,
    Math.hypot((point(308)[0] - point(78)[0]) * aspect, point(308)[1] - point(78)[1]) * 0.5,
  );
  const innerMouthMidX = (point(78)[0] + point(308)[0]) * 0.5;
  const innerMouthMidY = (point(78)[1] + point(308)[1]) * 0.5;

  const dist2 = (ax: number, ay: number, bx: number, by: number) =>
    Math.hypot((ax - bx) * aspect, ay - by);

  const projMouth = (pt: [number, number, number]) => {
    const dx = (pt[0] - innerMouthMidX) * aspect;
    const dy = pt[1] - innerMouthMidY;
    return {
      x: (dx * face.mouthAxis.ux + dy * face.mouthAxis.uy) / innerMouthHalfW,
      y: dx * face.mouthAxis.vx + dy * face.mouthAxis.vy,
    };
  };
  const buildLipCurve = (idxs: number[]) =>
    idxs.map((idx) => projMouth(point(idx))).sort((a, b) => a.x - b.x);
  const evalLipCurve = (curve: Array<{ x: number; y: number }>, qx: number): number => {
    if (curve.length === 0) return 0;
    if (qx <= curve[0]!.x) return curve[0]!.y;
    if (qx >= curve[curve.length - 1]!.x) return curve[curve.length - 1]!.y;
    for (let k = 0; k < curve.length - 1; k++) {
      const a = curve[k]!;
      const b = curve[k + 1]!;
      if (qx >= a.x && qx <= b.x) {
        const f = (qx - a.x) / Math.max(1e-5, b.x - a.x);
        return a.y + (b.y - a.y) * f;
      }
    }
    return 0;
  };
  const outerUpCurve = buildLipCurve(OUTER_UPPER_LIP);
  const innerUpCurve = buildLipCurve(INNER_UPPER_LIP);
  const innerLoCurve = buildLipCurve(INNER_LOWER_LIP);
  const outerLoCurve = buildLipCurve(OUTER_LOWER_LIP);
  const mouthMidSlitY = (idxsX: number) =>
    (evalLipCurve(innerUpCurve, idxsX) + evalLipCurve(innerLoCurve, idxsX)) * 0.5;

  const cornerLipSet = new Set([61, 78, 291, 308]);
  let fPtr = 0;
  for (let i = 0; i < numPts; i++) {
    const pt = point(i);
    const nx = pt[0];
    const ny = pt[1];
    const isOval = ovalSet.has(i);
    const isEyeAp = eyeLoopSet.has(i);

    const localScale = isOval ? 0.82 : 1.0;
    const relFaceX =
      (nx - noseTip[0]) * aspect * face.faceAxis.ux + (ny - noseTip[1]) * face.faceAxis.uy;
    const side = Math.max(-1, Math.min(1, relFaceX / (face.faceW * 0.22)));

    const mp = projMouth(pt);
    const mouthX = Math.max(-1, Math.min(1, mp.x));
    const lipCenterArch = Math.max(0, Math.cos(mouthX * Math.PI * 0.48));
    const yOuterUp = evalLipCurve(outerUpCurve, mouthX);
    const yInnerUp = evalLipCurve(innerUpCurve, mouthX);
    const ySlit = mouthMidSlitY(mouthX);
    const yOuterLo = evalLipCurve(outerLoCurve, mouthX);

    const inMouthSpan = Math.abs(mp.x) <= 1.15;
    const isCornerVert = cornerLipSet.has(i);
    const isExplicitUpper = innerUpperSet.has(i) || outerUpperSet.has(i);
    const isExplicitLower = innerLowerSet.has(i) || outerLowerSet.has(i);
    const inUpperVermilion =
      inMouthSpan &&
      !isCornerVert &&
      !isExplicitLower &&
      mp.y >= yOuterUp - 0.003 * face.faceH &&
      mp.y <= ySlit;
    const inLowerVermilion =
      inMouthSpan &&
      !isCornerVert &&
      !isExplicitUpper &&
      mp.y >= ySlit &&
      mp.y <= yOuterLo + 0.003 * face.faceH;

    let jawW = computeMandibleWeight(nx, ny, mp.y - ySlit);

    let upperLipW =
      inMouthSpan && mp.y < ySlit
        ? (1 - smoothstep(0.0, face.mouthH * 0.95, Math.max(0, yOuterUp - mp.y))) *
          (0.36 + 0.64 * lipCenterArch) *
          localScale
        : 0;

    if (isCornerVert) {
      jawW = 0.36 * localScale;
      upperLipW = 0.24 * localScale;
    } else if (isExplicitUpper || inUpperVermilion) {
      const cornerTaper = smoothstep(0.76, 1.0, Math.abs(mouthX));
      jawW = 0.36 * localScale * cornerTaper * 0.45;
      upperLipW = (0.24 + 0.76 * lipCenterArch) * localScale;
    } else if (isExplicitLower || inLowerVermilion) {
      jawW = (0.36 + 0.64 * lipCenterArch) * localScale;
      upperLipW = 0.24 * localScale * smoothstep(0.76, 1.0, Math.abs(mouthX)) * 0.45;
    } else if (mp.y <= yInnerUp) {
      jawW = 0;
    }

    const dMouth = dist2(nx, ny, mouthCenter[0], mouthCenter[1]);
    const dCorners = Math.min(
      dist2(nx, ny, rCorner[0], rCorner[1]),
      dist2(nx, ny, lCorner[0], lCorner[1]),
    );
    let cornerW = (1 - smoothstep(0.0, face.mouthW * 0.54, dCorners)) * localScale;
    if (
      innerLipSet.has(i) ||
      outerUpperSet.has(i) ||
      outerLowerSet.has(i) ||
      inUpperVermilion ||
      inLowerVermilion
    ) {
      cornerW = Math.max(
        cornerW,
        (1 - smoothstep(0.0, face.mouthW * 0.62, dMouth)) * 0.85 * localScale,
      );
    }

    const dCheek = Math.min(
      dist2(nx, ny, rCheekPt[0], rCheekPt[1]),
      dist2(nx, ny, lCheekPt[0], lCheekPt[1]),
      dist2(nx, ny, point(116)[0], point(116)[1]),
      dist2(nx, ny, point(345)[0], point(345)[1]),
      dist2(nx, ny, point(205)[0], point(205)[1]),
      dist2(nx, ny, point(425)[0], point(425)[1]),
    );
    const cheekW =
      (1 - smoothstep(0.0, face.faceW * 0.42, dCheek)) *
      smoothstep(
        Math.min(face.rightEyeCenter[1], face.leftEyeCenter[1]) - 0.02 * face.faceH,
        mouthCenter[1] + 0.03 * face.faceH,
        ny,
      ) *
      localScale *
      (isEyeAp ? 0 : 1);

    const dNose = Math.min(
      dist2(nx, ny, rNoseWing[0], rNoseWing[1]),
      dist2(nx, ny, lNoseWing[0], lNoseWing[1]),
      dist2(nx, ny, point(102)[0], point(102)[1]),
      dist2(nx, ny, point(331)[0], point(331)[1]),
      dist2(nx, ny, point(98)[0], point(98)[1]),
      dist2(nx, ny, point(327)[0], point(327)[1]),
      dist2(nx, ny, noseTip[0], noseTip[1]),
      dist2(nx, ny, point(4)[0], point(4)[1]),
      dist2(nx, ny, point(5)[0], point(5)[1]),
      dist2(nx, ny, point(195)[0], point(195)[1]),
      dist2(nx, ny, point(197)[0], point(197)[1]),
      dist2(nx, ny, point(6)[0], point(6)[1]),
    );
    const noseW = (1 - smoothstep(0.0, face.faceW * 0.24, dNose)) * localScale;

    let innerBrowW = 0;
    let outerBrowW = 0;
    const isRBrow = rightBrowSet.has(i);
    const isLBrow = leftBrowSet.has(i);
    const box = relFaceX < 0 ? rBrowBox : lBrowBox;
    const xRatio = Math.min(1, Math.abs(relFaceX) / Math.max(0.02, face.faceW * 0.36));

    if (isRBrow || isLBrow) {
      innerBrowW = (1 - smoothstep(0.18, 0.58, xRatio)) * 0.96;
      outerBrowW = smoothstep(0.16, 0.58, xRatio) * 0.96;
    } else if (
      ny < box.eyeTopY - face.faceH * 0.01 &&
      ny > box.minY - face.faceH * 0.13 &&
      nx > box.minX - face.faceW * 0.08 &&
      nx < box.maxX + face.faceW * 0.08
    ) {
      let browBand = 0;
      if (ny >= box.minY && ny <= box.maxY) {
        browBand = 1.0;
      } else if (ny < box.minY) {
        browBand = 1 - smoothstep(0.0, face.faceH * 0.12, box.minY - ny);
      } else {
        browBand = 1 - smoothstep(0.0, Math.max(0.012, box.eyeTopY - box.maxY), ny - box.maxY);
      }
      innerBrowW = browBand * (1 - smoothstep(0.18, 0.58, xRatio)) * 0.9 * localScale;
      outerBrowW = browBand * smoothstep(0.16, 0.58, xRatio) * 0.9 * localScale;
    }

    let eyeOrbitW = 0;
    if (!isEyeAp && !isOval) {
      const dR = dist2(nx, ny, face.rightPupil[0], face.rightPupil[1]);
      const dL = dist2(nx, ny, face.leftPupil[0], face.leftPupil[1]);
      const rSpan = face.rightEyeRadius[0] * aspect;
      const lSpan = face.leftEyeRadius[0] * aspect;
      const rLowerBias = ny >= face.rightEyeCenter[1] - 0.005 * face.faceH ? 1.0 : 0.25;
      const lLowerBias = ny >= face.leftEyeCenter[1] - 0.005 * face.faceH ? 1.0 : 0.25;
      eyeOrbitW = Math.max(
        (1 - smoothstep(rSpan * 0.85, rSpan * 2.25, dR)) *
          smoothstep(rSpan * 0.45, rSpan * 0.9, dR) *
          rLowerBias,
        (1 - smoothstep(lSpan * 0.85, lSpan * 2.25, dL)) *
          smoothstep(lSpan * 0.45, lSpan * 0.9, dL) *
          lLowerBias,
      );
    }

    const borderMask =
      smoothstep(0.0, 0.018, nx) *
      smoothstep(1.0, 0.982, nx) *
      smoothstep(0.0, 0.018, ny) *
      smoothstep(1.0, 0.982, ny);
    const headW =
      (ny <= chin[1] ? Math.max(0.96, computeHeadWeight(nx, ny)) : computeHeadWeight(nx, ny)) *
      borderMask;

    const normLandmarkZ = Math.max(0, Math.min(1, (rawZ[i]! - minZ) / zSpan));
    const ellipsoidZ = computeEllipsoidDepth(nx, ny);
    const depthZ = isOval
      ? ellipsoidZ * 0.65 + normLandmarkZ * 0.35
      : normLandmarkZ * 0.65 + ellipsoidZ * 0.35;

    faceVerts[fPtr++] = nx;
    faceVerts[fPtr++] = ny;
    faceVerts[fPtr++] = nx;
    faceVerts[fPtr++] = ny;
    faceVerts[fPtr++] = headW;
    faceVerts[fPtr++] = 0;
    faceVerts[fPtr++] = jawW;
    faceVerts[fPtr++] = depthZ;
    faceVerts[fPtr++] = upperLipW;
    faceVerts[fPtr++] = cornerW;
    faceVerts[fPtr++] = cheekW;
    faceVerts[fPtr++] = noseW;
    faceVerts[fPtr++] = innerBrowW;
    faceVerts[fPtr++] = outerBrowW;
    faceVerts[fPtr++] = eyeOrbitW;
    faceVerts[fPtr++] = side;
    faceVerts[fPtr++] = 0;
    faceVerts[fPtr++] = 0;
    faceVerts[fPtr++] = 0; // mouthLipT only on mouthVerts
    faceVerts[fPtr++] = mouthX;
  }

  // 3) Watertight Multi-Row Oral Interior Bridging INNER_UPPER_LIP <-> INNER_LOWER_LIP
  const M_COLS = INNER_UPPER_LIP.length;
  const M_ROWS = 7;
  const mouthVerts = new Float32Array(M_COLS * M_ROWS * STRIDE);
  let mPtr = 0;
  for (let r = 0; r < M_ROWS; r++) {
    const t = r / (M_ROWS - 1);
    for (let k = 0; k < M_COLS; k++) {
      const upIdx = INNER_UPPER_LIP[k] ?? 0;
      const loIdx = INNER_LOWER_LIP[k] ?? 0;
      const upOff = upIdx * STRIDE;
      const loOff = loIdx * STRIDE;
      for (let f = 0; f < STRIDE; f++) {
        mouthVerts[mPtr + f] =
          (faceVerts[upOff + f] ?? 0) * (1 - t) + (faceVerts[loOff + f] ?? 0) * t;
      }
      mouthVerts[mPtr + 18] = t; // mouthLipT
      mPtr += STRIDE;
    }
  }

  const mouthIdxArr: number[] = [];
  for (let r = 0; r < M_ROWS - 1; r++) {
    for (let k = 0; k < M_COLS - 1; k++) {
      const a = r * M_COLS + k;
      const b = a + 1;
      const c = a + M_COLS;
      const d = c + 1;
      mouthIdxArr.push(a, c, b, b, c, d);
    }
  }

  // 4) Unified Whole-Portrait Delaunay Triangulation
  const inside = (x: number, y: number, loop: number[]) => {
    let result = false;
    for (let i = 0, j = loop.length - 1; i < loop.length; j = i++) {
      const a = pts[loop[i] ?? 0];
      const b = pts[loop[j] ?? 0];
      if (!a || !b) continue;
      if (a[1] > y !== b[1] > y && x < ((b[0] - a[0]) * (y - a[1])) / (b[1] - a[1]) + a[0]) {
        result = !result;
      }
    }
    return result;
  };

  const minGridDist = face.faceW * 0.036;
  const combined = Array.from(faceVerts);

  for (let i = 0; i < baseVerts.length; i += STRIDE) {
    const x = baseVerts[i] ?? 0;
    const y = baseVerts[i + 1] ?? 0;
    if (inside(x, y, face.ovalLoop)) continue;
    let tooClose = false;
    for (const oi of face.ovalLoop) {
      const op = pts[oi];
      if (op && Math.hypot((op[0] - x) * aspect, op[1] - y) < minGridDist) {
        tooClose = true;
        break;
      }
    }
    if (tooClose) continue;
    for (let f = 0; f < STRIDE; f++) {
      combined.push(baseVerts[i + f] ?? 0);
    }
  }

  const coordinates: [number, number][] = [];
  for (let i = 0; i < combined.length; i += STRIDE) {
    coordinates.push([combined[i] ?? 0, combined[i + 1] ?? 0]);
  }
  const triangulation = Delaunator.from(coordinates);
  const continuous: number[] = [];
  const wire: number[] = [];
  const rightEyeSet = new Set(RIGHT_EYE_LOOP);
  const leftEyeSet = new Set(LEFT_EYE_LOOP);

  for (let i = 0; i < triangulation.triangles.length; i += 3) {
    const a = triangulation.triangles[i];
    const b = triangulation.triangles[i + 1];
    const c = triangulation.triangles[i + 2];
    if (a === undefined || b === undefined || c === undefined) continue;
    continuous.push(a, b, c);
    if (a < numPts && b < numPts && c < numPts) {
      const inREye = rightEyeSet.has(a) && rightEyeSet.has(b) && rightEyeSet.has(c);
      const inLEye = leftEyeSet.has(a) && leftEyeSet.has(b) && leftEyeSet.has(c);
      if (!inREye && !inLEye) {
        wire.push(a, b, b, c, c, a);
      }
    }
  }

  return {
    bgVerts,
    bgIdx,
    faceVerts: new Float32Array(combined),
    faceIdx: new Uint16Array(continuous),
    wireIdx: new Uint16Array(wire),
    mouthVerts,
    mouthIdx: new Uint16Array(mouthIdxArr),
    neckPivot,
    jawPivot,
    mouthCenter,
  };
}
