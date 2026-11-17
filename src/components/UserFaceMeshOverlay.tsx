import { useEffect, useRef } from "react";
import {
  INNER_LIP_LOOP,
  LEFT_EYE_LOOP,
  OUTER_LOWER_LIP,
  OUTER_UPPER_LIP,
  RIGHT_EYE_LOOP,
} from "@/lib/face";

// Canonical MediaPipe FaceMesh contour chains for live wireframe display over user camera (ONLY when Espejo + Malla facial are both active).
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

const RIGHT_BROW_UPPER = [70, 63, 105, 66, 107];
const RIGHT_BROW_LOWER = [46, 53, 52, 65, 55];
const LEFT_BROW_UPPER = [336, 296, 334, 293, 300];
const LEFT_BROW_LOWER = [285, 295, 282, 283, 276];

const NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2];
const NOSE_ALAR = [129, 49, 131, 134, 51, 5, 281, 363, 360, 279, 358];

// Cross-facial triangulation structural ribs (forehead, cheeks, temples, jawline)
const STRUCTURAL_CHAINS: number[][] = [
  [10, 151, 9, 8, 168],
  [54, 104, 68, 71, 21],
  [284, 333, 298, 301, 251],
  [107, 9, 336],
  [70, 127, 234, 116, 117, 118, 119, 120, 121, 128, 196],
  [300, 356, 454, 345, 346, 347, 348, 349, 350, 357, 419],
  [234, 50, 205, 206, 216, 61],
  [454, 280, 425, 426, 436, 291],
  [58, 135, 192, 207, 187, 123, 116],
  [288, 364, 416, 427, 411, 352, 345],
  [136, 194, 204, 211, 170, 140],
  [365, 418, 424, 431, 395, 369],
  [152, 175, 199, 200, 18, 17],
  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
];

const OUTER_LIP_LOOP = [
  ...OUTER_UPPER_LIP,
  ...OUTER_LOWER_LIP.slice(1, OUTER_LOWER_LIP.length - 1).reverse(),
];

export function UserFaceMeshOverlay({
  meshPoints,
  visible,
  videoSize,
}: {
  meshPoints?: Array<[number, number]> | undefined;
  visible: boolean;
  videoSize?: { width: number; height: number } | undefined;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = Math.max(16, Math.round(rect.width));
    const cssH = Math.max(16, Math.round(rect.height));
    const pxW = Math.round(cssW * dpr);
    const pxH = Math.round(cssH * dpr);
    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW;
      canvas.height = pxH;
    }

    ctx.clearRect(0, 0, pxW, pxH);
    if (!visible || !meshPoints || meshPoints.length < 468) return;

    const vw = Math.max(1, videoSize?.width || 640);
    const vh = Math.max(1, videoSize?.height || 480);
    const scale = Math.max(pxW / vw, pxH / vh);
    const drawnW = vw * scale;
    const drawnH = vh * scale;
    const offX = (pxW - drawnW) * 0.5;
    const offY = (pxH - drawnH) * 0.5;

    const mapPt = (idx: number): [number, number] | null => {
      const pt = meshPoints[idx];
      if (!pt) return null;
      return [offX + pt[0] * drawnW, offY + pt[1] * drawnH];
    };

    const strokeChain = (chain: number[], closed = false) => {
      if (chain.length < 2) return;
      ctx.beginPath();
      let started = false;
      for (const idx of chain) {
        const p = mapPt(idx);
        if (!p) continue;
        if (!started) {
          ctx.moveTo(p[0], p[1]);
          started = true;
        } else {
          ctx.lineTo(p[0], p[1]);
        }
      }
      if (closed) ctx.closePath();
      ctx.stroke();
    };

    const connectPairs = (aList: number[], bList: number[]) => {
      const n = Math.min(aList.length, bList.length);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const a = mapPt(aList[i]!);
        const b = mapPt(bList[i]!);
        if (!a || !b) continue;
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();
    };

    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    // Subtle soft glow + crisp hairline matching avatar mesh overlay
    ctx.strokeStyle = "rgba(247, 250, 255, 0.34)";
    ctx.lineWidth = Math.max(0.85, dpr * 0.85);

    strokeChain(FACE_OVAL, true);
    strokeChain(RIGHT_BROW_UPPER, false);
    strokeChain(RIGHT_BROW_LOWER, false);
    strokeChain(LEFT_BROW_UPPER, false);
    strokeChain(LEFT_BROW_LOWER, false);
    connectPairs(RIGHT_BROW_UPPER, RIGHT_BROW_LOWER);
    connectPairs(LEFT_BROW_UPPER, LEFT_BROW_LOWER);

    strokeChain(RIGHT_EYE_LOOP, true);
    strokeChain(LEFT_EYE_LOOP, true);
    strokeChain(NOSE_BRIDGE, false);
    strokeChain(NOSE_ALAR, false);
    strokeChain(OUTER_LIP_LOOP, true);
    strokeChain(INNER_LIP_LOOP, true);
    connectPairs(OUTER_UPPER_LIP, [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]);
    connectPairs(OUTER_LOWER_LIP, [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]);

    ctx.strokeStyle = "rgba(247, 250, 255, 0.22)";
    for (const chain of STRUCTURAL_CHAINS) {
      strokeChain(chain, false);
    }

    ctx.restore();
  }, [meshPoints, visible, videoSize?.width, videoSize?.height]);

  if (!visible) return null;

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 h-full w-full -scale-x-100"
    />
  );
}
