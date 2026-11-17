import { describe, expect, it } from "vitest";
import { fitCanonicalLandmarks } from "@/lib/canonicalFace";
import { AVATAR_VERTEX_STRIDE, buildAvatarMeshes } from "@/lib/avatarMesh";
import { createExpressionEngine, stepExpressionEngine } from "@/lib/expressionEngine";
import type { EyeCurveProfile, FacePts } from "@/lib/face";

function createTestFacePts(stylized = false): FacePts {
  const landmarks = fitCanonicalLandmarks(
    {
      rightEye: [0.42, 0.36],
      leftEye: [0.58, 0.36],
      noseTip: [0.5, 0.46],
      mouthCenter: [0.5, 0.56],
      chin: [0.5, 0.68],
      forehead: [0.5, 0.21],
      stylized,
    },
    1,
  );
  const eyeCurve: EyeCurveProfile = {
    upper: [-0.28, -0.45, -0.45, -0.28],
    lower: [0.22, 0.35, 0.35, 0.22],
    upperArch: [-0.18, -0.32, -0.44, -0.48, -0.48, -0.42, -0.3, -0.16],
    lowerArch: [0.14, 0.26, 0.34, 0.36, 0.36, 0.32, 0.24, 0.14],
    halfW: 0.032,
    scaleY: 0.018,
    outerReach: 1.26,
    outerWingY: 0,
    browFloorEy: -0.78,
  };
  return {
    landmarks,
    aspect: 1,
    faceTris: new Uint16Array([0, 1, 2]),
    ovalLoop: [
      10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
      148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ],
    center: [0.5, 0.45],
    faceAxis: { ux: 1, uy: 0, vx: 0, vy: 1 },
    mouthAxis: { ux: 1, uy: 0, vx: 0, vy: 1 },
    leftEyeAxis: { ux: 1, uy: 0, vx: 0, vy: 1 },
    rightEyeAxis: { ux: 1, uy: 0, vx: 0, vy: 1 },
    faceW: 0.34,
    faceH: 0.46,
    mouthW: 0.12,
    mouthH: 0.04,
    jawSpan: 0.14,
    restMouthRatio: 0.02,
    restMouthTopY: -0.02,
    restMouthBotY: 0.02,
    hasVisibleTeeth: false,
    isMouthParted: false,
    restTeethDepth: 0.01,
    yaw: 0,
    stylized,
    rightPupil: [0.42, 0.36],
    leftPupil: [0.58, 0.36],
    rightEyeCenter: [0.42, 0.36],
    leftEyeCenter: [0.58, 0.36],
    rightEyeRadius: [0.032, 0.018],
    leftEyeRadius: [0.032, 0.018],
    rightEyeCurve: eyeCurve,
    leftEyeCurve: eyeCurve,
    rightIrisRadius: 0.014,
    leftIrisRadius: 0.014,
    rightEar: [0.3, 0.42],
    leftEar: [0.7, 0.42],
    hairTop: [0.5, 0.1],
    hairLeft: [0.28, 0.4],
    hairRight: [0.72, 0.4],
    neckBase: [0.5, 0.82],
    lowerNeck: [0.5, 0.78],
    adamApple: [0.5, 0.74],
    rightTrapezius: [0.36, 0.84],
    leftTrapezius: [0.64, 0.84],
    rightShoulder: [0.22, 0.92],
    leftShoulder: [0.78, 0.92],
    rightCheekPos: [0.41, 0.5],
    leftCheekPos: [0.59, 0.5],
    rightInnerCanthus: [0.452, 0.36],
    leftInnerCanthus: [0.548, 0.36],
    hasTrapezius: true,
    neckInflectionY: 0.78,
    hands: [],
    colors: {
      skin: [215, 175, 155],
      lip: [165, 92, 92],
      eyelid: [195, 155, 138],
      lowerEyelid: [198, 160, 142],
      enamel: [232, 226, 218],
      shirt: [45, 68, 88],
      scleraRight: [224, 220, 216],
      scleraLeft: [224, 220, 216],
      scleraRightNeg: [222, 218, 214],
      scleraRightPos: [226, 222, 218],
      scleraLeftNeg: [226, 222, 218],
      scleraLeftPos: [222, 218, 214],
    },
  };
}

describe("Avatar mesh rigging and 3D kinematics", () => {
  it("builds a continuous Delaunay mesh with finite 3D depth and zero attribute aliasing", () => {
    const face = createTestFacePts(false);
    const built = buildAvatarMeshes(face);

    expect(built.faceVerts.length).toBeGreaterThan(468 * AVATAR_VERTEX_STRIDE);
    expect(built.faceIdx.length).toBeGreaterThan(1500);
    expect(built.wireIdx.length).toBeGreaterThan(600);
    expect(built.mouthVerts.length).toBe(11 * 7 * AVATAR_VERTEX_STRIDE);
    expect(built.mouthIdx.length).toBeGreaterThan(100);

    // Every float across faceVerts and mouthVerts must be finite
    for (let i = 0; i < built.faceVerts.length; i++) {
      expect(Number.isFinite(built.faceVerts[i])).toBe(true);
    }

    // Check first 468 face landmark vertices: depthZ (off+7) in [0,1], mouthLipT (off+18) === 0
    for (let v = 0; v < 468; v++) {
      const off = v * AVATAR_VERTEX_STRIDE;
      const depthZ = built.faceVerts[off + 7]!;
      const mouthLipT = built.faceVerts[off + 18]!;
      expect(depthZ).toBeGreaterThanOrEqual(0);
      expect(depthZ).toBeLessThanOrEqual(1);
      expect(mouthLipT).toBe(0);
    }

    // Check oral interior vertices carry mouthLipT in [0, 1]
    for (let v = 0; v < 11 * 7; v++) {
      const off = v * AVATAR_VERTEX_STRIDE;
      const t = built.mouthVerts[off + 18]!;
      expect(t).toBeGreaterThanOrEqual(0);
      expect(t).toBeLessThanOrEqual(1);
    }
  });

  it("fits canonical landmarks with non-zero 3D depth across facial contours", () => {
    const pts = fitCanonicalLandmarks(
      {
        rightEye: [0.41, 0.37],
        leftEye: [0.59, 0.37],
        noseTip: [0.5, 0.47],
        mouthCenter: [0.5, 0.58],
        stylized: true,
      },
      0.8,
    );
    expect(pts).toHaveLength(468);
    const noseZ = pts[1]![2];
    const cheekZ = pts[234]![2];
    expect(noseZ).not.toBe(cheekZ);
  });

  it("animates speech, natural symmetric brows, and head sway smoothly over time", () => {
    const state = createExpressionEngine(0);
    let last = stepExpressionEngine(state, 16, {
      active: true,
      listening: false,
      emotion: "happy",
      visemes: { level: 0.72, open: 0.58, wide: 0.32, round: 0.14 },
      hasVisibleTeeth: false,
      restMouthRatio: 0.02,
      animeStyle: false,
    });

    for (let t = 50; t <= 1200; t += 33) {
      last = stepExpressionEngine(state, t, {
        active: true,
        listening: false,
        emotion: "happy",
        visemes: { level: 0.72, open: 0.58, wide: 0.32, round: 0.14 },
        hasVisibleTeeth: false,
        restMouthRatio: 0.02,
        animeStyle: false,
      });
    }

    expect(last.open).toBeGreaterThan(0.25);
    expect(last.smile).toBeGreaterThan(0.3);
    expect(Math.abs(last.browAsym)).toBeLessThan(0.25);
    expect(Math.abs(last.headYaw) + Math.abs(last.headPitch)).toBeGreaterThan(0.001);
  });
});
