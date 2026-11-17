import { useEffect, useRef, useState } from "react";
import Delaunator from "delaunator";
import { findCharacterById, getProfile } from "@/lib/store";
import {
  detectFace,
  INNER_LIP_LOOP,
  INNER_LOWER_LIP,
  INNER_UPPER_LIP,
  LEFT_EYE_LOOP,
  OUTER_LOWER_LIP,
  OUTER_UPPER_LIP,
  RIGHT_EYE_LOOP,
  type FacePts,
} from "@/lib/face";
import { createExpressionEngine, stepExpressionEngine } from "@/lib/expressionEngine";
import type { AvatarEmotion } from "@/lib/ai";
import { audioVisemes } from "@/lib/voice";
import { LiveAvatarStreamPlayer } from "./LiveAvatarStreamPlayer";
import { portraitRenderBudget } from "@/lib/avatar-device";
import type { AvatarEngine } from "@/lib/store";
import type { UserFaceObservation } from "@/lib/userCamera";
import teethTextureUrl from "@/assets/images/teeth_mouth_cavity_1794754794898.jpg";

export type { AvatarEmotion };
export type AvatarGesture = "idle" | "talk";

const DEFAULT_FALLBACK_PHOTO = "/avatars/default.jpg";

// Vertex layout (20 floats per vertex):
// aPosUv:    vec4(nx, ny, uvx, uvy)        [photo normalized 0..1]
// aHeadRig:  vec4(headW, torsoW, jawW, edgeFade)
// aFaceRig:  vec4(upperLipW, cornerW, cheekW, noseW)
// aBrowRig:  vec4(innerBrowW, outerBrowW, eyeOrbitW, sideSign)
// aHandRig:  vec4(hand0W, hand1W, depthZ / mouthLipT, mouthX)

const VERT_SRC = `#version 300 es
precision highp float;

in vec4 aPosUv;
in vec4 aHeadRig;
in vec4 aFaceRig;
in vec4 aBrowRig;
in vec4 aHandRig;

uniform vec2 uViewSize;
uniform vec2 uImgSize;
uniform vec2 uCoverOffset;
uniform vec2 uCoverSize;

uniform vec2 uNeckPivot;
uniform vec2 uJawPivot;
uniform vec2 uMouthCenter;
uniform vec2 uFaceCenter;
uniform vec4 uFaceAxis;   // xy = right unit vector, zw = down unit vector
uniform vec4 uMouthAxis;  // xy = mouth right unit vector, zw = mouth down unit vector
uniform float uFaceW;
uniform float uFaceH;
uniform float uMouthW;
uniform float uAspect;

uniform float uOpen;
uniform float uWide;
uniform float uRound;
uniform float uLipPress;
uniform float uLabiodental;
uniform float uUpperLipLift;
uniform float uLowerLipDrop;
uniform float uLipBite;
uniform float uSmile;
uniform float uSmileR;
uniform float uSmileL;
uniform float uMouthSkewX;
uniform float uCheekLift;
uniform float uInnerBrowY;
uniform float uInnerBrowPinch;
uniform float uOuterBrowY;
uniform float uBrowAsym;
uniform float uSquint;
uniform float uEyeWide;
uniform float uNoseFlare;

uniform vec2 uHeadRot;      // x: yaw-parallax, y: pitch-parallax
uniform float uHeadRoll;
uniform vec2 uNeckRot;      // decoupled cervical neck yaw/pitch
uniform float uNeckRoll;
uniform float uTorsoYaw;
uniform float uForwardLean;
uniform vec2 uHeadTranslate;
uniform float uTorsoBreath;
uniform float uTorsoSwayX;
uniform float uShoulderTilt;
uniform float uShoulderShrug;
uniform vec2 uHand0Delta;
uniform vec2 uHand1Delta;
uniform vec2 uHairDelta;
uniform float uSwallow;
uniform vec2 uAdamAppleCoord;
uniform float uIsBasePass;
uniform float uPass;

out vec2 vUv;
out float vEdgeFade;
out float vMouthLipT;
out float vMouthX;
out vec2 vMouthPlane;
out float vJawDropMouthW;
out float vJawBoneDropMouthW;
out vec2 vRigidMouth;
out vec2 vFaceCoord;

vec2 addAxisDelta(vec2 p, vec4 axis, vec2 deltaAspect, float aspect) {
  return p + vec2(
    (axis.x * deltaAspect.x + axis.z * deltaAspect.y) / aspect,
    axis.y * deltaAspect.x + axis.w * deltaAspect.y
  );
}

void main() {
  vUv = aPosUv.zw;
  vEdgeFade = aHeadRig.w;
  vMouthLipT = aHandRig.z;
  vMouthX = aHandRig.w;

  vec2 p = aPosUv.xy;

  // Static background reconstruction quad
  if (uPass < -0.5) {
    vec2 pxBg = uCoverOffset + p * uCoverSize;
    vec2 ndcBg = vec2((pxBg.x / max(1.0, uViewSize.x)) * 2.0 - 1.0, 1.0 - (pxBg.y / max(1.0, uViewSize.y)) * 2.0);
    gl_Position = vec4(ndcBg, 0.0, 1.0);
    vMouthPlane = vec2(0.0);
    vJawDropMouthW = 0.0;
    vJawBoneDropMouthW = 0.0;
    vRigidMouth = vec2(0.0);
    vFaceCoord = vec2(0.0);
    return;
  }

  float aspect = max(0.25, uAspect);
  float headW = aHeadRig.x;
  float torsoW = aHeadRig.y;
  float jawW = aHeadRig.z;
  float side = aBrowRig.w;

  // Compute rest-frame normalized face coordinates for consistent dynamic expression effects
  vec2 faceRel = vec2((aPosUv.x - uFaceCenter.x) * aspect, aPosUv.y - uFaceCenter.y);
  vFaceCoord = vec2(
    dot(faceRel, uFaceAxis.xy) / max(0.05, uFaceW * 0.5),
    dot(faceRel, uFaceAxis.zw) / max(0.05, uFaceH * 0.5)
  );

  // 1) Expressive brow shaping: organic curved motion + glabella pinch (never stretches brows or pulls eyelids!)
  float asym = side * uBrowAsym;
  float browLift = clamp((aBrowRig.x * (uInnerBrowY + asym * 0.42) + aBrowRig.y * (uOuterBrowY + asym)) * 0.026, -0.030, 0.034) * uFaceH;
  float furrowDrop = aBrowRig.x * max(0.0, uInnerBrowPinch) * 0.015 * uFaceH;
  float browPinch = -side * aBrowRig.x * uInnerBrowPinch * 0.030 * uFaceW;
  float orbitLift = aBrowRig.z * (uSquint * 0.010 - uEyeWide * 0.006) * uFaceH;
  p = addAxisDelta(p, uFaceAxis, vec2(browPinch, -(browLift - furrowDrop + orbitLift)), aspect);

  // 2) Natural nostril flare & disgust upper-nose contraction
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(
      side * aFaceRig.w * max(0.0, uNoseFlare) * 0.009 * uFaceW,
      -aFaceRig.w * max(0.0, uNoseFlare) * 0.0095 * uFaceH
    ),
    aspect
  );

  // 3) Gentle zygomatic cheek lift preserving rounded oval cheekbones (no cat-cheek pinching)
  float sideSmile = mix(uSmileR, uSmileL, smoothstep(-0.35, 0.35, side));
  float laughCheek = uCheekLift * 0.0105 + max(0.0, sideSmile) * 0.0065;
  float cheekSpread = side * aFaceRig.z * (max(0.0, sideSmile) * 0.0032 + uCheekLift * 0.0025) * uFaceW;
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(cheekSpread, -aFaceRig.z * laughCheek * uFaceH),
    aspect
  );

  // 4) Independent upper/lower lip control, wide opening, side-to-side skew, lip-bite & mandible drop
  vec2 mouthRel = vec2((p.x - uMouthCenter.x) * aspect, p.y - uMouthCenter.y);
  float mSign = clamp(dot(mouthRel, uMouthAxis.xy) / max(0.01, uMouthW * 0.46), -1.0, 1.0);
  float localSmile = mix(uSmileR, uSmileL, smoothstep(-0.35, 0.35, mSign));
  float lipSealZone = 1.0 - smoothstep(uMouthW * 0.32, uMouthW * 0.86, length(mouthRel));
  float cornerTaper = 1.0 - 0.16 * pow(abs(mSign), 2.2);
  float centerArch = 1.0 - smoothstep(0.18, 0.92, abs(mSign));
  float closedTeethPart = clamp((max(uSmileR, uSmileL) - 0.32) * 1.6, 0.0, 1.0) * (1.0 - smoothstep(0.06, 0.20, uOpen));

  float upperLift = aFaceRig.x * cornerTaper * (
    uOpen * 0.015 + max(0.0, localSmile) * 0.016 + uUpperLipLift * 0.026 + closedTeethPart * 0.008 * centerArch + uRound * 0.005 - uLipPress * 0.009
  ) * uMouthW;
  float cornerWide = mSign * aFaceRig.y * (uWide * 0.019 + localSmile * 0.023 - uRound * 0.028 - uLipPress * 0.009) * uMouthW;
  float cornerLift = abs(mSign) * aFaceRig.y * (localSmile * 0.023 + uWide * 0.005 - uRound * 0.006 + mSign * uMouthSkewX * 0.010) * uMouthW;
  float cornerJawFollow = aFaceRig.y * (1.0 - aFaceRig.x) * (uOpen * 0.010 + uLowerLipDrop * 0.008) * uFaceH;
  float mouthLatSkew = uMouthSkewX * 0.115 * uMouthW * clamp(aFaceRig.y * 1.05 + aFaceRig.x * 0.85 + jawW * 0.55, 0.0, 1.25);

  float jawBoneDrop = uOpen * 0.075 * uFaceH;
  float jawOpenDrop = (uOpen * 0.078 + uLowerLipDrop * 0.028) * uFaceH;
  float lowerSmilePart = closedTeethPart * centerArch * (1.0 - aFaceRig.x) * 0.011 * uFaceH * jawW;
  float jawPressSeal = -jawW * lipSealZone * (uLipPress * 0.009 + uLabiodental * 0.011 + uLipBite * 0.016) * uFaceH;
  float jawDrop = jawOpenDrop * jawW + lowerSmilePart + jawPressSeal;
  float jawInwardTuck = -mSign * jawW * (uRound * 0.005 + uLabiodental * 0.003 + uLipBite * 0.004) * uFaceW;
  p = addAxisDelta(
    p,
    uMouthAxis,
    vec2(
      cornerWide + jawInwardTuck + mouthLatSkew,
      jawDrop + cornerJawFollow - upperLift - cornerLift
    ),
    aspect
  );

  // Cranial mouth coordinates & decoupled mandible bone drop so teeth stay anchored on the skull while lips slide over them
  vec2 postMouthRel = vec2((p.x - uMouthCenter.x) * aspect, p.y - uMouthCenter.y);
  vMouthPlane = vec2(
    dot(postMouthRel, uMouthAxis.xy) / max(0.01, uMouthW),
    dot(postMouthRel, uMouthAxis.zw) / max(0.01, uMouthW)
  );
  vJawDropMouthW = jawOpenDrop / max(0.01, uMouthW);
  vJawBoneDropMouthW = jawBoneDrop / max(0.01, uMouthW);
  vec2 rigidMouthRel = vec2((aPosUv.x - uMouthCenter.x) * aspect, aPosUv.y - uMouthCenter.y);
  vRigidMouth = vec2(
    dot(rigidMouthRel, uMouthAxis.xy) / max(0.01, uMouthW),
    dot(rigidMouthRel, uMouthAxis.zw) / max(0.01, uMouthW)
  );

  // 5) Decoupled cranium + cervical neck 3D rotation preserving full face shape (zero spaghetti stretching),
  // subtle interior turn & realistic near-ear profile depth
  float neckBand = smoothstep(0.06, 0.56, headW) * (1.0 - smoothstep(0.68, 0.96, headW));
  float craniumCore = smoothstep(0.42, 0.96, headW);
  if (headW > 0.001) {
    vec2 skullPivot = mix(uNeckPivot, uFaceCenter, 0.38);
    vec2 rel = vec2((p.x - skullPivot.x) * aspect, p.y - skullPivot.y);
    float activeRoll = uHeadRoll * craniumCore + uNeckRoll * neckBand;
    float cRoll = cos(activeRoll);
    float sRoll = sin(activeRoll);
    vec2 r = vec2(rel.x * cRoll - rel.y * sRoll, rel.x * sRoll + rel.y * cRoll);

    float hx = clamp(vFaceCoord.x, -1.25, 1.25);
    float hy = clamp(vFaceCoord.y, -1.25, 1.25);
    float faceR = length(vec2(hx * 0.95, hy * 0.85));
    float innerMask = 1.0 - smoothstep(0.28, 0.96, faceR);

    float noseBump = exp(-(hx * hx * 7.5 + hy * hy * 4.5)) * 0.065 * uFaceW;
    float r2 = (hx * 0.32) * (hx * 0.32) + (hy * 0.26) * (hy * 0.26);
    float skullZ = sqrt(max(0.16, 1.0 - min(0.84, r2))) * uFaceW * 0.34 + noseBump;
    float pivotZ = uFaceW * 0.14;
    float dz = skullZ - pivotZ;

    float yawRad = clamp(uHeadRot.x * 0.24 * craniumCore + uNeckRot.x * 0.14 * neckBand, -0.22, 0.22) * headW;
    float pitchRad = clamp(uHeadRot.y * 0.20 * craniumCore + uNeckRot.y * 0.10 * neckBand, -0.18, 0.18) * headW;

    float cosY = cos(yawRad);
    float sinY = sin(yawRad);
    // Shift interior features (nose, eyes, brows, mouth) only gently so features stay centered cleanly in the face oval
    float rotX = r.x * mix(1.0, cosY, 0.08 * craniumCore) + dz * sinY * mix(0.10, 0.38, innerMask);
    r.x = rotX;
    r.y = r.y - dz * sin(pitchRad) * mix(0.16, 0.42, innerMask);

    // Profile near-ear & outer-rim depth illusion: near ear opens outward & scales subtly larger while far ear tucks gently behind
    float earZone = smoothstep(0.74, 1.08, abs(vFaceCoord.x)) *
      smoothstep(-0.36, 0.00, vFaceCoord.y) *
      (1.0 - smoothstep(0.28, 0.62, vFaceCoord.y)) * craniumCore;
    float earNearness = -sign(vFaceCoord.x) * sinY;
    float nearAmt = clamp(earNearness * 2.2, 0.0, 1.0);
    float farAmt = clamp(-earNearness * 2.2, 0.0, 1.0);
    float earShift = sign(vFaceCoord.x) * (nearAmt * 0.020 - farAmt * 0.012) * uFaceW * earZone;
    float earScale = 1.0 + (nearAmt * 0.080 - farAmt * 0.050) * earZone;
    vec2 earAnchor = vec2(sign(vFaceCoord.x) * uFaceW * 0.42, uFaceCenter.y - skullPivot.y);
    r = mix(r, earAnchor + (r - earAnchor) * earScale, earZone);

    p.x = skullPivot.x + (r.x + earShift) / aspect + uHeadTranslate.x * headW;
    p.y = skullPivot.y + r.y + uHeadTranslate.y * headW;
  }

  // 6) 5-point anatomical lowerNeck + bilateral trapezius + shoulders moving together symmetrically (never opposite deformation)
  if (uPass < 1.5) {
    float shoulderRidgeW = aHandRig.z;
    float yokeW = clamp(max(torsoW, shoulderRidgeW) * (1.0 - craniumCore * 0.92), 0.0, 1.0);
    float shrugShift = clamp(uShoulderShrug, -0.6, 1.0) * 0.010 * uFaceH;
    float torsoYawShiftX = clamp(uTorsoYaw, -0.5, 0.5) * 0.011 * uFaceW * yokeW;
    float hyoidY = exp(-pow((vFaceCoord.y - uAdamAppleCoord.y) / 0.24, 2.0));
    float hyoidX = 1.0 - smoothstep(0.14, 0.44, abs(vFaceCoord.x - uAdamAppleCoord.x));
    float hyoidBand = hyoidY * hyoidX;
    float swallowLift = clamp(uSwallow, 0.0, 1.0) * hyoidBand * 0.0115 * uFaceH;
    p.x += (uTorsoSwayX * 0.0042 * uFaceW * yokeW + torsoYawShiftX) / aspect;
    p.y += -swallowLift - (uTorsoBreath * 0.0048 * uFaceH + shrugShift) * yokeW;
  }

  if (headW > 0.001) {
    float leanVal = clamp(uForwardLean, -0.35, 0.55);
    float craniumZoom = 1.0 + leanVal * headW * 0.022;
    vec2 leanAnchor = mix(uMouthCenter, uFaceCenter, 0.58);
    vec2 relLean = vec2((p.x - leanAnchor.x) * aspect, p.y - leanAnchor.y) * craniumZoom;
    p.x = leanAnchor.x + relLean.x / aspect;
    p.y = leanAnchor.y + relLean.y + leanVal * headW * 0.004 * uFaceH;
  }

  // 7) Hair stays strictly locked to cranium (zero detached hair warping)
  if (uPass < 1.5 && length(uHairDelta) > 0.0001 && p.y < uFaceCenter.y && headW > 0.5) {
    p += uHairDelta * 0.0;
  }

  // 8) Subtle hand micro-gestures when visible
  p += uHand0Delta * aHandRig.x;
  p += uHand1Delta * aHandRig.y;

  vec2 px = uCoverOffset + p * uCoverSize;
  vec2 ndc = vec2((px.x / max(1.0, uViewSize.x)) * 2.0 - 1.0, 1.0 - (px.y / max(1.0, uViewSize.y)) * 2.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

uniform sampler2D uPhotoTex;
uniform sampler2D uTeethTex;
uniform sampler2D uBgTex;
uniform sampler2D uFgMaskTex;
uniform sampler2D uBgBlurTex;
uniform sampler2D uEyeAtlasTex;
uniform vec2 uBgTexel;
uniform float uBlurredBg;
uniform float uPass; // -1.0 = reconstructed background, 1.0 = cutout portrait mesh, 2.0 = oral interior

uniform vec2 uREyeCenter;
uniform vec2 uREyePupil;
uniform vec4 uREyeAxis;
uniform vec4 uREyeScale; // xy = halfW, scaleY; z = outerReach; w = outerWingY
uniform vec4 uREyeUpperA;
uniform vec4 uREyeUpperB;
uniform vec4 uREyeLowerA;
uniform vec4 uREyeLowerB;
uniform float uRIrisRad;

uniform vec2 uLEyeCenter;
uniform vec2 uLEyePupil;
uniform vec4 uLEyeAxis;
uniform vec4 uLEyeScale; // xy = halfW, scaleY; z = outerReach; w = outerWingY
uniform vec4 uLEyeUpperA;
uniform vec4 uLEyeUpperB;
uniform vec4 uLEyeLowerA;
uniform vec4 uLEyeLowerB;
uniform float uLIrisRad;

uniform vec2 uGaze;
uniform float uBlink;
uniform float uBlinkR;
uniform float uBlinkL;
uniform float uSquint;
uniform float uEyeWide;
uniform float uAspect;
uniform float uYaw;
uniform float uStylized;
uniform float uWireframe;
uniform float uFaceH;
uniform float uMouthW;

uniform float uTime;
uniform float uOpen;
uniform float uWide;
uniform float uRound;
uniform float uLipPress;
uniform float uUpperLipLift;
uniform float uLowerLipDrop;
uniform float uLipBite;
uniform float uSmile;
uniform float uSmileR;
uniform float uSmileL;
uniform float uCheekLift;
uniform float uInnerBrowY;
uniform float uInnerBrowPinch;
uniform float uOuterBrowY;
uniform float uEyeRoll;
uniform float uNoseFlare;
uniform float uTongueOut;
uniform float uLidDroop;
uniform float uTearAmount;
uniform float uEyeWater;
uniform float uEyeRedness;
uniform float uPupilDilate;
uniform float uSweatAmount;
uniform float uBlushAmount;
uniform float uSwallow;
uniform vec2 uAdamAppleCoord;
uniform vec2 uHeadRot;
uniform vec4 uFaceAxis;
uniform float uFaceW;
uniform vec4 uCheekCenters;
uniform vec4 uTearDucts;
uniform float uHasVisibleTeeth;
uniform float uRestMouthTopY;
uniform float uRestMouthBotY;
uniform vec3 uLipColor;
uniform vec3 uEnamelColor;
uniform vec3 uEyelidColor;
uniform vec3 uLowerEyelidColor;
uniform vec3 uRScleraNeg;
uniform vec3 uRScleraPos;
uniform vec3 uLScleraNeg;
uniform vec3 uLScleraPos;

in vec2 vUv;
in float vEdgeFade;
in float vMouthLipT;
in float vMouthX;
in vec2 vMouthPlane;
in float vJawDropMouthW;
in float vJawBoneDropMouthW;
in vec2 vRigidMouth;
in vec2 vFaceCoord;

out vec4 fragColor;

float evalLidCurveOuter(float t, vec4 a, vec4 b, float outerSign, float outerReach, float outerWingY, float wingPad) {
  float sEx = t * outerSign;
  float x = clamp(t, -1.0, 1.0);
  float y = 0.0;
  if (x <= -0.82) {
    y = mix(0.03, a.x, smoothstep(-1.0, -0.82, x));
  } else if (x <= -0.58) {
    y = mix(a.x, a.y, smoothstep(-0.82, -0.58, x));
  } else if (x <= -0.34) {
    y = mix(a.y, a.z, smoothstep(-0.58, -0.34, x));
  } else if (x <= -0.11) {
    y = mix(a.z, a.w, smoothstep(-0.34, -0.11, x));
  } else if (x <= 0.11) {
    y = mix(a.w, b.x, smoothstep(-0.11, 0.11, x));
  } else if (x <= 0.34) {
    y = mix(b.x, b.y, smoothstep(0.11, 0.34, x));
  } else if (x <= 0.58) {
    y = mix(b.y, b.z, smoothstep(0.34, 0.58, x));
  } else if (x <= 0.82) {
    y = mix(b.z, b.w, smoothstep(0.58, 0.82, x));
  } else {
    y = mix(b.w, 0.05, smoothstep(0.82, 1.0, x));
  }

  // Smooth continuous parabolic descent into a level/slightly-lower outer commissure (zero straight-line edges)
  float outerCornerY = clamp(max(0.05, outerWingY * 0.65 + 0.04), 0.04, 0.12);
  if (sEx > 0.34) {
    float archPeak = outerSign > 0.0 ? b.x : a.w;
    float paraSpan = max(0.50, outerReach - 0.34);
    float pFrac = clamp((sEx - 0.34) / paraSpan, 0.0, 1.0);
    float pBend = pFrac * pFrac * (1.25 - 0.25 * pFrac);
    float parabolicArch = mix(archPeak, outerCornerY + wingPad * (1.0 - pow(pFrac, 0.85)), pBend);
    if (sEx <= 0.82) {
      float blendOuter = smoothstep(0.34, 0.82, sEx);
      y = mix(y, parabolicArch, blendOuter * 0.52);
    } else {
      float outerKnot = outerSign > 0.0 ? b.w : a.x;
      float wingSpan = max(0.18, outerReach - 0.82);
      float wFrac = clamp((sEx - 0.82) / wingSpan, 0.0, 1.0);
      float wCurve = smoothstep(0.0, 1.0, wFrac);
      y = mix(mix(outerKnot, parabolicArch, 0.52), outerCornerY + wingPad * (1.0 - wCurve), wCurve);
    }
  }
  return y;
}

vec3 sampleCrispTeeth(vec2 uv) {
  return texture(uTeethTex, uv).rgb;
}

vec3 sampleBlurredBg(vec2 uv) {
  vec3 baseBlur = texture(uBgBlurTex, uv).rgb;
  vec2 rad = uBgTexel * 5.0;
  vec2 ring[8] = vec2[8](
    vec2( 1.000,  0.000), vec2( 0.707,  0.707), vec2( 0.000,  1.000), vec2(-0.707,  0.707),
    vec2(-1.000,  0.000), vec2(-0.707, -0.707), vec2( 0.000, -1.000), vec2( 0.707, -0.707)
  );
  vec3 bokeh = baseBlur * 0.36;
  for (int i = 0; i < 8; i++) {
    bokeh += texture(uBgBlurTex, clamp(uv + ring[i] * rad, 0.001, 0.999)).rgb * 0.08;
  }
  return bokeh * 0.95;
}

// Photorealistic human eye: verified homogeneous sclera (zero double-cornea), 100% original iris color, full outer eyeliner-wing closure
vec3 renderHumanEye(
  vec3 baseRgb,
  vec2 eyeCenter,
  vec2 eyePupil,
  vec4 eyeAxis,
  vec4 eyeScale,
  vec4 upperA,
  vec4 upperB,
  vec4 lowerA,
  vec4 lowerB,
  float irisRadNorm,
  vec3 scleraNegColor,
  vec3 scleraPosColor,
  float eyeBlink,
  float outerSign,
  float aspect
) {
  vec2 dUv = vec2((vUv.x - eyeCenter.x) * aspect, vUv.y - eyeCenter.y);
  float halfW = max(0.005, eyeScale.x);
  float scaleY = max(0.004, eyeScale.y);
  float outerReach = clamp(eyeScale.z, 1.18, 1.50);
  float outerWingY = eyeScale.w;

  float ex = dot(dUv, eyeAxis.xy) / halfW;
  float ey = dot(dUv, eyeAxis.zw) / scaleY;

  float rawTopY = evalLidCurveOuter(ex, upperA, upperB, outerSign, outerReach, outerWingY, -0.05);
  float rawBotY = evalLidCurveOuter(ex, lowerA, lowerB, outerSign, outerReach, outerWingY, 0.05);

  // Reach all the way past the outer eye commissure & detected eyeliner/lash wing
  bool isOuterSide = (ex * outerSign) > 0.0;
  float maxReach = isOuterSide ? outerReach : 1.06;
  float exNorm = abs(ex) / maxReach;
  float cornerEnv = 1.0 - smoothstep(0.84, 1.0, exNorm);
  if (cornerEnv < 0.002) return baseRgb;

  float effectiveTopBase = rawTopY;
  float effectiveBotBase = rawBotY;

  float effectiveBlink = max(eyeBlink, clamp(uLidDroop, 0.0, 0.85) * 0.68);
  float blinkClose = clamp(effectiveBlink - uEyeWide * 0.05, 0.0, 1.0);
  float squintClose = clamp(uSquint * 0.24, 0.0, 0.36);
  float lidAction = max(blinkClose, squintClose);

  float paraCurve = 1.0 - min(1.0, exNorm * exNorm);
  float meetArch = 0.045 * paraCurve * blinkClose;
  float outerCloseBias = smoothstep(0.28, outerReach * 0.86, ex * outerSign);
  float meet = mix(effectiveTopBase, effectiveBotBase, mix(0.86, 1.0, outerCloseBias)) + meetArch;
  float topY = mix(effectiveTopBase, meet, blinkClose);
  float squintBot = mix(effectiveBotBase, mix(effectiveBotBase, effectiveTopBase, 0.22), squintClose);
  float botY = mix(squintBot, meet + 0.006, blinkClose);

  float feather = max(0.022, fwidth(ey) * 1.3);
  float rawAperture = cornerEnv * smoothstep(effectiveTopBase - feather, effectiveTopBase + feather, ey) *
    (1.0 - smoothstep(effectiveBotBase - feather, effectiveBotBase + feather, ey));
  float aperture = cornerEnv * smoothstep(topY - feather, topY + feather, ey) *
    (1.0 - smoothstep(botY - feather, botY + feather, ey));

  // Strictly bound upperCrease below eyebrow hairs so blinking never drags or darkens raised eyebrows onto the eyelid
  float archEnv = 1.0 - 0.04 * min(1.0, exNorm * exNorm);
  float upperCrease = effectiveTopBase - 1.12 * archEnv;
  float lowerCrease = effectiveBotBase + 0.84 * archEnv;

  float lidTopFeather = smoothstep(upperCrease - 0.12, upperCrease + 0.26, ey);
  float lidBotFeather = 1.0 - smoothstep(lowerCrease - 0.20, lowerCrease + 0.16, ey);
  float lidZoneMask = cornerEnv * lidTopFeather * lidBotFeather;
  float lidMovingMask = smoothstep(0.004, 0.040, lidAction);

  // Defined eyeliner & lash contour reaching cleanly across the outer eye wing
  float linerSpan = isOuterSide
    ? (1.0 - smoothstep(0.84, outerReach * 0.98, ex * outerSign)) * cornerEnv
    : (1.0 - smoothstep(0.76, 1.04, -ex * outerSign)) * cornerEnv;
  float upperLinerBand = exp(-pow((ey - (topY - 0.04)) / 0.095, 2.0)) * linerSpan;
  float outerWingBoost = mix(0.78, 1.22, smoothstep(0.15, 1.08, ex * outerSign));
  float lowerLinerMask = smoothstep(-0.20, outerReach * 0.78, ex * outerSign) * linerSpan;
  float lowerLinerBand = exp(-pow((ey - (botY + 0.05)) / 0.090, 2.0)) * lowerLinerMask;
  float contourMask = clamp(max(upperLinerBand * outerWingBoost, lowerLinerBand * 0.42), 0.0, 1.0);

  float effectMask = max(max(rawAperture, lidZoneMask * lidMovingMask), contourMask * 0.72);
  if (effectMask < 0.002) return baseRgb;

  float sourceY;
  if (ey <= topY) {
    float tUp = clamp((ey - upperCrease) / max(0.04, topY - upperCrease), 0.0, 1.0);
    sourceY = mix(upperCrease, effectiveTopBase - 0.024 * lidMovingMask, pow(tUp, 1.02));
  } else if (ey >= botY) {
    float tDn = clamp((ey - botY) / max(0.04, lowerCrease - botY), 0.0, 1.0);
    sourceY = mix(effectiveBotBase + 0.016 * lidMovingMask, lowerCrease, tDn);
  } else {
    float tAp = clamp((ey - topY) / max(0.025, botY - topY), 0.0, 1.0);
    sourceY = mix(effectiveTopBase - 0.024 * lidMovingMask, effectiveBotBase + 0.016 * lidMovingMask, tAp);
  }

  vec2 sourceDelta = eyeAxis.xy * (ex * halfW) + eyeAxis.zw * (sourceY * scaleY);
  vec2 sourceUv = eyeCenter + vec2(sourceDelta.x / aspect, sourceDelta.y);
  vec3 photoSkin = texture(uPhotoTex, clamp(sourceUv, 0.001, 0.999)).rgb;

  // Reject any dark eyebrow hairs/shadow near upperCrease during blink so raised eyebrows never darken the descending lid
  float refLidLum = max(0.18, dot(uEyelidColor, vec3(0.299, 0.587, 0.114)));
  float sampleLum = dot(photoSkin, vec3(0.299, 0.587, 0.114));
  float upperSampleFrac = smoothstep(effectiveTopBase - 0.25, upperCrease, sourceY);
  float darkBrowReject = (1.0 - smoothstep(refLidLum * 0.62, refLidLum * 0.90, sampleLum)) * upperSampleFrac * lidMovingMask;
  photoSkin = mix(photoSkin, uEyelidColor, clamp(darkBrowReject, 0.0, 0.94));

  float crispLashEdge = exp(-pow((ey - topY) / 0.068, 2.0)) * (0.34 + 0.24 * blinkClose) * linerSpan * outerWingBoost;
  float domeHighlight = exp(-pow(ex / 0.78, 2.0)) *
    exp(-pow((ey - mix(effectiveTopBase, topY, 0.52)) / 0.52, 2.0)) *
    blinkClose * 0.018;
  vec3 linerInk = vec3(0.055, 0.040, 0.036);
  vec3 sculptedLid = mix(clamp(photoSkin + vec3(domeHighlight), 0.0, 1.0), linerInk, clamp(crispLashEdge, 0.0, 0.72));

  // Split-sclera square atlas: left/right sampled light sclera gradient + extracted crisp original cornea
  float shiftEx = clamp(uGaze.x * 0.40, -0.44, 0.44);
  float shiftEy = clamp(-(uGaze.y * 0.66 + uEyeRoll * 1.10), -1.75, 1.75);
  float moveActive = smoothstep(0.012, 0.075, length(vec2(shiftEx, shiftEy)));

  float atlasLocalU = clamp(((ex - shiftEx) / 1.9) * 0.5 + 0.5, 0.005, 0.995);
  float atlasU = (outerSign < 0.0 ? 0.0 : 0.5) + atlasLocalU * 0.5;
  float atlasV = clamp(((ey - shiftEy) / 2.9) * 0.5 + 0.5, 0.005, 0.995);
  vec3 atlasEye = texture(uEyeAtlasTex, vec2(atlasU, atlasV)).rgb;

  // Keep stationary inner lacrimal caruncle anchored smoothly in the inner eye corner
  float innerCornerAnchor = smoothstep(0.62, 0.88, -ex * outerSign);
  float upperLidShadow = mix(0.96, 1.0, smoothstep(topY, topY + 0.16, ey));
  vec3 eyeCol = mix(baseRgb, atlasEye * upperLidShadow, moveActive * (1.0 - innerCornerAnchor * 0.92));

  vec2 dPupil = vec2((eyePupil.x - eyeCenter.x) * aspect, eyePupil.y - eyeCenter.y);
  float pupilEx = dot(dPupil, eyeAxis.xy) / halfW;
  float pupilEy = dot(dPupil, eyeAxis.zw) / scaleY;
  float lenNew = length(vec2((ex - shiftEx - pupilEx) * halfW, (ey - shiftEy - pupilEy) * scaleY));
  float baseIrisR = max(0.002, irisRadNorm);
  float drawR = baseIrisR * 0.96;
  float eraseR = baseIrisR * 1.14;

  float redAmt = clamp(uEyeRedness, 0.0, 1.0);
  if (redAmt > 0.02) {
    float irisArea = 1.0 - smoothstep(drawR * 0.92, eraseR * 1.04, lenNew);
    float scleralArea = (1.0 - irisArea) * aperture;
    float outerRimBias = smoothstep(0.45, 0.95, abs(ex));
    vec3 irritated = eyeCol * mix(vec3(1.0), vec3(1.02, 0.88, 0.87), redAmt * 0.24 * outerRimBias);
    eyeCol = mix(eyeCol, clamp(irritated, 0.0, 1.0), scleralArea * 0.38);
  }

  float waterAmt = clamp(uEyeWater, 0.0, 1.0);
  if (waterAmt > 0.01) {
    float rimPool = exp(-pow((ey - (botY - 0.14)) / 0.09, 2.0)) *
      (1.0 - smoothstep(0.68, 0.96, abs(ex))) * aperture;
    eyeCol += vec3(0.18, 0.20, 0.24) * rimPool * waterAmt * 0.60;
  }

  float openInside = aperture * (1.0 - smoothstep(0.76, 0.92, blinkClose));
  vec3 result = mix(sculptedLid, eyeCol, openInside);

  // Subtle defined upper & lower outer eyeliner contour around the open eye rim
  float openContour = clamp((upperLinerBand * outerWingBoost * 0.48 + lowerLinerBand * 0.26) * (1.0 - openInside * 0.55), 0.0, 0.62);
  result = mix(result, linerInk, openContour);

  return mix(baseRgb, result, effectMask);
}

// Dynamic facial creases, glabella "11" wrinkles, disgust nose-bridge & upper-nasolabial sneer folds, crow's feet & forehead lines
vec3 applyExpressionWrinkles(vec3 col, vec2 fc, vec2 rm) {
  float absX = abs(fc.x);
  float skinMask = (1.0 - smoothstep(0.84, 1.08, absX)) * (1.0 - smoothstep(0.88, 1.10, abs(fc.y)));
  if (skinMask < 0.01) return col;

  float luma = dot(col, vec3(0.299, 0.587, 0.114));
  float skinToneMask = smoothstep(0.14, 0.30, luma);
  if (skinToneMask < 0.01) return col;

  // 1) Nasolabial folds (zygomatic smile & upper-lip disgust sneer arch around/above nostrils)
  float sideSmile = fc.x < 0.0 ? max(0.0, uSmileR) : max(0.0, uSmileL);
  float noseScrunchAct = smoothstep(0.12, 0.82, max(0.0, uNoseFlare));
  float foldWeight = clamp(0.14 + sideSmile * 0.72 + uCheekLift * 0.54 + noseScrunchAct * 0.58, 0.0, 1.35);
  float nasoCurveX = 0.22 + smoothstep(-0.04, 0.54, fc.y) * 0.36;
  float nasoVert = smoothstep(-0.08, 0.10, fc.y) * (1.0 - smoothstep(0.46, 0.64, fc.y));
  float nasoDist = abs(absX - nasoCurveX);
  float nasoShadow = exp(-pow(nasoDist / 0.048, 2.0)) * nasoVert * foldWeight;

  // 2) Crow's-feet outer-eye smile crinkles & lower-lid smile crease
  float squintAct = smoothstep(0.12, 0.82, max(0.18, uSquint + uCheekLift * 0.42 + sideSmile * 0.30));
  float crowZone = smoothstep(0.50, 0.68, absX) * (1.0 - smoothstep(0.90, 1.04, absX)) *
    exp(-pow((fc.y + 0.16) / 0.15, 2.0));
  float crowRays = 0.5 + 0.5 * cos((fc.y + 0.16) * 46.0 - (absX - 0.64) * 18.0);
  float crowShadow = crowZone * crowRays * squintAct * 0.75;

  float underEyeCrease = exp(-pow((absX - 0.36) / 0.18, 2.0)) *
    exp(-pow((fc.y + 0.05) / 0.042, 2.0)) * (sideSmile * 0.45 + uCheekLift * 0.50);

  // 3) Glabella "11" anger creases between brows + horizontal bunny nose-bridge scrunch creases
  float furrowAct = smoothstep(0.04, 0.72, uInnerBrowPinch);
  float glabellaZone = exp(-pow((absX - 0.068) / 0.032, 2.0)) *
    smoothstep(-0.58, -0.42, fc.y) * (1.0 - smoothstep(-0.20, -0.08, fc.y));
  float glabellaShadow = glabellaZone * furrowAct * 1.48;

  float bridgeZone = (1.0 - smoothstep(0.06, 0.22, absX)) *
    smoothstep(-0.34, -0.24, fc.y) * (1.0 - smoothstep(-0.04, 0.05, fc.y));
  float bridgeLines = 0.5 + 0.5 * cos((fc.y + 0.18) * 58.0);
  float bunnySideZone = exp(-pow((absX - 0.14) / 0.05, 2.0)) *
    exp(-pow((fc.y + 0.06) / 0.10, 2.0));
  float bunnyShadow = (bridgeZone * bridgeLines * 0.90 + bunnySideZone * 0.80) * noseScrunchAct;

  // 4) Expressive forehead lines on brow lift & central supra-brow anger tension creases
  float browHigh = smoothstep(0.18, 0.85, (uInnerBrowY + uOuterBrowY) * 0.5);
  float foreheadZone = (1.0 - smoothstep(0.28, 0.68, absX)) *
    smoothstep(-0.88, -0.74, fc.y) * (1.0 - smoothstep(-0.50, -0.40, fc.y));
  float foreheadBands = 0.5 + 0.5 * cos((fc.y + 0.66) * 44.0);
  float frownArchZone = (1.0 - smoothstep(0.08, 0.34, absX)) *
    smoothstep(-0.74, -0.60, fc.y) * (1.0 - smoothstep(-0.46, -0.36, fc.y));
  float frownArchLines = 0.5 + 0.5 * cos((fc.y + 0.56) * 52.0);
  float foreheadShadow = foreheadZone * foreheadBands * browHigh * 0.68 + frownArchZone * frownArchLines * furrowAct * 0.85;

  // 5) Mouth corner dimple shading
  float restSlitY = (uRestMouthTopY + uRestMouthBotY) * 0.5;
  float cornerDimple = exp(-pow((abs(rm.x) - 0.96) / 0.12, 2.0)) *
    exp(-pow((rm.y - restSlitY) / 0.075, 2.0)) * (0.20 + uSmile * 0.32 + uLipPress * 0.28);

  float totalCrease = clamp(
    (nasoShadow * 0.12 + crowShadow * 0.10 + underEyeCrease * 0.08 + glabellaShadow * 0.22 +
     bunnyShadow * 0.13 + foreheadShadow * 0.13 + cornerDimple * 0.10) * skinMask * skinToneMask,
    0.0,
    0.36
  );
  vec3 shaded = col * mix(vec3(1.0), vec3(0.66, 0.54, 0.50), totalCrease);
  return clamp(shaded, 0.0, 1.0);
}

// Organic irregular skin moisture/tear droplet helper (for sweat beads & subtle tear micro-beads)
vec3 blendSubtleLiquidTrail(
  vec3 col,
  vec2 fc,
  vec2 startFc,
  float prog,
  float sideSign,
  float alpha,
  bool isTear
) {
  if (alpha < 0.01) return col;
  float lenRun = isTear ? 0.52 : 0.26;
  float curY = startFc.y + prog * lenRun;
  float curX = startFc.x + (isTear ? sin(prog * 7.5 + sideSign) * 0.022 + sideSign * prog * 0.045 : 0.0);

  // Subtle wet dampened skin path left above the sliding tear
  float relYTrack = fc.y - startFc.y;
  if (relYTrack >= -0.02 && relYTrack <= prog * lenRun) {
    float tFrac = clamp(relYTrack / max(0.01, lenRun), 0.0, 1.0);
    float trackCenterX = startFc.x + (isTear ? sin(tFrac * 7.5 + sideSign) * 0.022 + sideSign * tFrac * 0.045 : 0.0);
    float distTrack = abs(fc.x - trackCenterX);
    float fadeTail = smoothstep(0.0, 0.18, tFrac) * smoothstep(-0.02, 0.06, prog * lenRun - relYTrack);
    float sheenLine = exp(-pow(distTrack / 0.011, 2.0)) * fadeTail;
    col = mix(col, col * vec3(0.94, 0.95, 0.98) + vec3(0.065, 0.075, 0.095), sheenLine * alpha * 0.45);
  }

  // Organic elongated micro-drop head at the leading edge (subtle refractive wet edge, zero big 2D circles)
  vec2 dHead = (fc - vec2(curX, curY)) / (isTear ? vec2(0.018, 0.029) : vec2(0.022, 0.032));
  float taper = 1.0 - clamp(-dHead.y * 0.35, -0.18, 0.40);
  float r = length(vec2(dHead.x / taper, dHead.y));
  if (r > 1.35) return col;

  float dropMask = (1.0 - smoothstep(0.75, 1.02, r)) * alpha;
  float shMask = (1.0 - smoothstep(0.35, 1.25, length(dHead - vec2(0.16, 0.30)))) * smoothstep(0.55, 1.08, r);
  col = mix(col, col * 0.80, shMask * alpha * 0.38);

  vec3 wetTint = mix(col * 0.96, vec3(0.88, 0.93, 0.98), 0.16 + pow(r, 2.0) * 0.20);
  float fineRim = exp(-pow(length(dHead - vec2(-0.24, -0.30)) / 0.24, 2.0)) * 0.32;
  return mix(col, clamp(wetTint + vec3(fineRim), 0.0, 1.0), dropMask * 0.72);
}

// Physical overlays: true cheekbone blush (or pale chill), temple/forehead sweat, and natural lacrimal-to-cheek tears
vec3 applyPhysicalEffects(vec3 col, vec2 fc) {
  float faceBound = (1.0 - smoothstep(0.92, 1.18, abs(fc.x))) * (1.0 - smoothstep(0.92, 1.18, abs(fc.y)));
  if (faceBound < 0.01) return col;

  // 1) Natural skin-matched blush across cheekbones & upper nasal bridge (never below nostrils or neon pink)
  if (abs(uBlushAmount) > 0.01) {
    vec2 rCheekPos = uCheekCenters.xy;
    vec2 lCheekPos = uCheekCenters.zw;
    float bridgeY = (rCheekPos.y + lCheekPos.y) * 0.5 - 0.05;
    float cheekR = exp(-(pow((fc.x - rCheekPos.x) / 0.25, 2.0) + pow((fc.y - rCheekPos.y) / 0.15, 2.0)));
    float cheekL = exp(-(pow((fc.x - lCheekPos.x) / 0.25, 2.0) + pow((fc.y - lCheekPos.y) / 0.15, 2.0)));
    float noseBridge = exp(-(pow(fc.x / 0.24, 2.0) + pow((fc.y - bridgeY) / 0.10, 2.0))) * 0.62;
    float aboveMouthCut = 1.0 - smoothstep(0.06, 0.20, fc.y);
    float blushZone = clamp((cheekL + cheekR + noseBridge) * aboveMouthCut * faceBound, 0.0, 1.0);
    if (uBlushAmount > 0.0) {
      vec3 naturalRosy = vec3(min(1.0, col.r * 1.07 + 0.032), col.g * 0.88, col.b * 0.87);
      col = mix(col, naturalRosy, blushZone * clamp(uBlushAmount, 0.0, 1.0) * 0.46);
    } else {
      float paleAmt = clamp(-uBlushAmount, 0.0, 1.0);
      float lum = dot(col, vec3(0.299, 0.587, 0.114));
      vec3 paleCol = clamp(vec3(lum * 0.95 + 0.055, lum * 0.98 + 0.05, lum * 1.02 + 0.05), 0.0, 1.0);
      col = mix(col, mix(col, paleCol, 0.68), faceBound * paleAmt * 0.78);
    }
  }

  // 2) Sweat sheen & fine beads strictly on upper forehead & outer temples (never inside eye box)
  if (uSweatAmount > 0.01) {
    float sw = clamp(uSweatAmount, 0.0, 1.0);
    float eyeBoxClear = smoothstep(0.46, 0.64, length(vec2(fc.x / 0.58, (fc.y + 0.24) / 0.26)));
    float foreheadBand = smoothstep(-0.46, -0.58, fc.y) * (1.0 - smoothstep(-0.88, -0.98, fc.y));
    float templeBand = smoothstep(0.56, 0.70, abs(fc.x)) * (1.0 - smoothstep(-0.18, -0.02, fc.y));
    float browSheen = exp(-(pow(fc.x / 0.52, 2.0) + pow((fc.y + 0.64) / 0.16, 2.0))) * foreheadBand;
    float templeSheen = exp(-(pow((abs(fc.x) - 0.66) / 0.14, 2.0) + pow((fc.y + 0.42) / 0.16, 2.0))) * templeBand;
    float gloss = clamp((browSheen + templeSheen * 0.85) * faceBound * eyeBoxClear, 0.0, 1.0);
    col = clamp(col + vec3(0.15, 0.16, 0.18) * gloss * sw, 0.0, 1.0);

    for (int sIdx = 0; sIdx < 3; sIdx++) {
      float fi = float(sIdx);
      float prog = fract(uTime * (0.14 + fi * 0.04) + fi * 0.37);
      float fade = smoothstep(0.0, 0.14, prog) * (1.0 - smoothstep(0.76, 1.0, prog));
      float sx = sIdx == 0 ? 0.68 : (sIdx == 1 ? -0.66 : 0.24);
      float startY = sIdx == 2 ? -0.78 : -0.60;
      col = blendSubtleLiquidTrail(col, fc, vec2(sx, startY), prog, sx < 0.0 ? -1.0 : 1.0, sw * fade * 0.78 * eyeBoxClear, false);
    }
  }

  // 3) Realistic crying flush + glistening tears emerging from real inner tear ducts (uTearDucts) down along the nose wing
  if (uTearAmount > 0.01) {
    float tr = clamp(uTearAmount, 0.0, 1.0);
    vec2 rDuct = uTearDucts.xy;
    vec2 lDuct = uTearDucts.zw;
    float rRimWet = exp(-(pow((fc.x - (rDuct.x - 0.12)) / 0.18, 2.0) + pow((fc.y - (rDuct.y + 0.06)) / 0.08, 2.0)));
    float lRimWet = exp(-(pow((fc.x - (lDuct.x + 0.12)) / 0.18, 2.0) + pow((fc.y - (lDuct.y + 0.06)) / 0.08, 2.0)));
    col = mix(col, col * vec3(1.04, 0.82, 0.83) + vec3(0.05, 0.02, 0.02), clamp((rRimWet + lRimWet) * tr * 0.55, 0.0, 0.55));

    for (int side = 0; side < 2; side++) {
      float sSign = side == 0 ? -1.0 : 1.0;
      vec2 duct = side == 0 ? rDuct : lDuct;
      for (int d = 0; d < 2; d++) {
        float fd = float(d);
        float speed = 0.16 + fd * 0.05;
        float phase = float(side) * 0.37 + fd * 0.51;
        float prog = fract(uTime * speed + phase);
        float fade = smoothstep(0.0, 0.14, prog) * (1.0 - smoothstep(0.74, 1.0, prog));
        col = blendSubtleLiquidTrail(col, fc, duct + vec2(sSign * 0.03 * fd, 0.03), prog, sSign, tr * fade * 0.88, true);
      }
    }
  }

  return col;
}

vec3 whitenToothEnamel(vec3 rawTex, float brighten) {
  float lum = dot(rawTex, vec3(0.299, 0.587, 0.114));
  vec3 neutralWhite = mix(rawTex, vec3(lum * 1.015, lum * 1.005, lum * 0.985), 0.72);
  float enamelWeight = smoothstep(0.24, 0.72, lum);
  vec3 scaleCol = neutralWhite * mix(0.86, brighten, enamelWeight);
  return clamp(scaleCol, vec3(0.0), vec3(0.97));
}

void main() {
  if (uPass < -0.5) {
    vec2 uvBg = clamp(vUv, 0.001, 0.999);
    vec3 bgCol = uBlurredBg > 0.5 ? sampleBlurredBg(uvBg) : texture(uBgTex, uvBg).rgb;
    fragColor = vec4(bgCol, 1.0);
    return;
  }

  vec2 uvSafe = clamp(vUv, 0.001, 0.999);
  vec4 photoCol = texture(uPhotoTex, uvSafe);
  vec3 col = photoCol.rgb;
  float fgAlpha = texture(uFgMaskTex, uvSafe).r;

  if (uPass > 1.5) {
    // Watertight photorealistic oral cavity: very dark red interior flesh, perspective tongue, and cleanly SEPARATED white photographic upper/lower teeth
    float closedBite = smoothstep(0.34, 0.78, uSmile) * (1.0 - smoothstep(0.06, 0.18, uOpen));
    float activeOpen = max(uOpen, max(uUpperLipLift * 0.48, max(uLowerLipDrop * 0.42, uTongueOut * 0.34)));
    float openMask = max(
      closedBite * 0.96,
      uHasVisibleTeeth > 0.5
        ? smoothstep(0.014, 0.082, activeOpen)
        : smoothstep(0.006, 0.040, activeOpen)
    );
    if (openMask <= 0.002) {
      discard;
    }

    float absX = abs(vMouthX);
    float cornerCurve = 0.08 * pow(absX, 1.8);
    float rimAA = max(0.018, fwidth(vMouthLipT) * 1.4);
    float topRim = smoothstep(cornerCurve, cornerCurve + rimAA, vMouthLipT);
    // Keep oral cavity & teeth strictly inside the opaque lower lip border (zero translucent teeth showing through lower lip)
    float botCut = 0.89 - cornerCurve * 0.65;
    float botRim = smoothstep(botCut, botCut - rimAA, vMouthLipT);
    float sideAlmondFade = 1.0 - smoothstep(0.84, 0.99, absX);
    float lipRimFeather = topRim * botRim * sideAlmondFade;

    // Skull-space spatial mouth coordinates (teeth stay fixed on maxilla while lips shift laterally)
    float x = vMouthPlane.x;
    float y = vMouthPlane.y;
    float jawY = y - vJawBoneDropMouthW * 0.80;
    float ax = clamp(x - uYaw * 0.075, -0.46, 0.46);
    float absAx = abs(ax);
    float archCurve = min(1.0, absAx / 0.44);
    float toothTexU = clamp(0.5 + ax * 1.05, 0.06, 0.94);
    float uCurve = pow(abs(toothTexU - 0.5) / 0.44, 1.8);

    // 1) Deep very-dark-red flesh interior cavity (never skin tone!)
    vec2 cavUv = vec2(clamp(0.5 + ax * 0.52, 0.22, 0.78), mix(0.36, 0.50, clamp(vMouthLipT, 0.0, 1.0)));
    vec3 rawCav = sampleCrispTeeth(cavUv);
    float cavDepth = 1.0 - 0.56 * (1.0 - abs(vMouthLipT - 0.44) * 1.25) * (1.0 - absX * 0.65);
    vec3 deepRedFlesh = vec3(0.048, 0.005, 0.007) + vec3(rawCav.r * 0.088, rawCav.g * 0.012, rawCav.b * 0.014);
    vec3 oralCol = clamp(deepRedFlesh * clamp(cavDepth, 0.28, 1.0), vec3(0.018, 0.002, 0.003), vec3(0.145, 0.022, 0.028));

    // 2) Upper teeth rigidly fixed to cranial maxilla (NEVER deformed by jaw drop or lateral lip shift)
    float pitchPersp = -uHeadRot.y * 0.007;
    float midRestY = (uRestMouthTopY + uRestMouthBotY) * 0.5;
    float upperEdgeOpen = uRestMouthTopY + 0.032 - 0.040 * ax * ax + pitchPersp;
    float upperEdgeSmile = midRestY + 0.003 - 0.026 * ax * ax + pitchPersp;
    float upperEdgeY = mix(upperEdgeOpen, upperEdgeSmile, closedBite);
    float upperCrownH = mix(0.138, 0.106, archCurve);
    float upperGumY = upperEdgeY - upperCrownH;
    float upperFrac = clamp((y - upperGumY) / max(0.022, upperCrownH), 0.0, 1.0);
    float upperVTop = mix(0.122, 0.195, uCurve);
    float upperVBot = mix(0.295, 0.325, uCurve);
    float upperTexV = (y < upperGumY)
      ? mix(0.065, upperVTop, clamp(1.0 - (upperGumY - y) / 0.055, 0.0, 1.0))
      : mix(upperVTop, upperVBot, upperFrac);

    vec3 rawUpper = sampleCrispTeeth(vec2(toothTexU, upperTexV));
    vec3 whiteUpper = whitenToothEnamel(rawUpper, 1.14);
    float archDepthShade = 1.0 - 0.34 * pow(archCurve, 1.35);
    float upperLipShadow = mix(0.72, 1.0, smoothstep(0.0, 0.45, upperFrac));
    float upperGumMask = y < upperGumY ? smoothstep(upperGumY + 0.004, upperGumY - 0.018, y) : 0.0;
    vec3 upperGumCol = clamp(rawUpper * vec3(0.88, 0.54, 0.56), vec3(0.18, 0.06, 0.07), vec3(0.66, 0.32, 0.35));
    vec3 upperToothCol = clamp(mix(whiteUpper, upperGumCol, upperGumMask) * archDepthShade * upperLipShadow, 0.0, 1.0);

    // 3) Lower teeth rigidly fixed to dropping mandible (jawY), natural taller proportions & cleanly separated
    float lowerTexU = clamp(0.5 + ax * 0.96, 0.08, 0.92);
    float lowerUCurve = pow(abs(lowerTexU - 0.5) / 0.42, 1.8);
    float lowerRestTipY = uRestMouthBotY - 0.014 - 0.028 * ax * ax;
    float lowerSmileTipY = upperEdgeY + 0.011;
    float lowerTipLocalY = mix(lowerRestTipY, lowerSmileTipY, closedBite);
    float lowerCrownH = mix(0.116, 0.090, archCurve);
    float lowerGumLocalY = lowerTipLocalY + lowerCrownH;
    float lowerFrac = clamp((jawY - lowerTipLocalY) / max(0.020, lowerCrownH), 0.0, 1.0);
    float lowerVTip = mix(0.750, 0.665, lowerUCurve);
    float lowerVGum = mix(0.900, 0.845, lowerUCurve);
    float lowerTexV = (jawY > lowerGumLocalY)
      ? mix(lowerVGum, 0.96, clamp((jawY - lowerGumLocalY) / 0.050, 0.0, 1.0))
      : mix(lowerVTip, lowerVGum, lowerFrac);

    vec3 rawLower = sampleCrispTeeth(vec2(lowerTexU, lowerTexV));
    vec3 whiteLower = whitenToothEnamel(rawLower, 1.07);
    float lowerLipShadow = mix(1.0, 0.74, smoothstep(0.68, 1.0, lowerFrac));
    float lowerGumMask = jawY > lowerGumLocalY ? smoothstep(lowerGumLocalY - 0.004, lowerGumLocalY + 0.018, jawY) : 0.0;
    vec3 lowerGumCol = clamp(rawLower * vec3(0.86, 0.52, 0.54), vec3(0.16, 0.05, 0.06), vec3(0.62, 0.29, 0.32));
    vec3 lowerToothCol = clamp(mix(whiteLower, lowerGumCol, lowerGumMask) * archDepthShade * lowerLipShadow, 0.0, 1.0);

    // 4) Perspective-tracked real photographic tongue seated in the mandible behind lower teeth (clearly visible on wide mouth openings)
    float tongueAdv = clamp(uTongueOut, 0.0, 1.0);
    float speechTongueWave = activeOpen > 0.05 ? sin(uTime * 3.8 + uRound * 2.0) * 0.005 : 0.0;
    float openTongueRise = smoothstep(0.08, 0.72, activeOpen) * 0.068;
    float tongueLift = clamp(uRound, 0.0, 1.0) * 0.008 + tongueAdv * 0.036 + speechTongueWave;
    float tonguePerspX = x - uYaw * 0.15;
    float tongueY = jawY + uHeadRot.y * 0.010;
    float tongueTopY = lowerTipLocalY - 0.024 - openTongueRise + 0.11 * tonguePerspX * tonguePerspX - tongueLift;
    float tongueBotY = lowerGumLocalY + 0.062;
    float tongueLatMask = 1.0 - smoothstep(0.26 + tongueAdv * 0.06, 0.46 + tongueAdv * 0.08, abs(tonguePerspX));
    float tongueVertMask =
      smoothstep(tongueTopY - 0.012, tongueTopY + 0.010, tongueY) *
      (1.0 - smoothstep(tongueBotY - 0.010, tongueBotY + 0.012, tongueY));
    float tongueShow = max(smoothstep(0.06, 0.20, activeOpen), smoothstep(0.03, 0.16, tongueAdv));
    float tongueMask = tongueLatMask * tongueVertMask * tongueShow * (1.0 - closedBite * 0.92);
    float tNormY = clamp((tongueY - tongueTopY) / max(0.020, tongueBotY - tongueTopY), 0.0, 1.0);
    vec2 tongueUv = vec2(
      clamp(0.5 + tonguePerspX * 0.56, 0.24, 0.76),
      mix(0.535, 0.725, tNormY)
    );
    vec3 rawTongue = sampleCrispTeeth(tongueUv);
    float wideOpenLight = 1.0 + smoothstep(0.28, 0.85, activeOpen) * 0.22;
    float tongueDepthShade = mix(0.54, 1.04, smoothstep(0.0, 0.48, tNormY)) * (1.0 - 0.20 * smoothstep(0.20, 0.46, abs(tonguePerspX))) * wideOpenLight;
    vec3 tongueCol = clamp(
      mix(rawTongue, vec3(rawTongue.r * 1.12, rawTongue.g * 0.84, rawTongue.b * 0.84), 0.35) * tongueDepthShade,
      vec3(0.06, 0.014, 0.018),
      vec3(0.86, 0.50, 0.52)
    );

    float yFeather = max(0.005, fwidth(y) * 1.6);
    float upperZone =
      (1.0 - smoothstep(upperEdgeY - yFeather, upperEdgeY + yFeather, y)) *
      (1.0 - smoothstep(0.34, 0.44, absAx));

    float lowerShowGate = max(
      max(smoothstep(0.045, 0.16, activeOpen), smoothstep(0.035, 0.12, uLowerLipDrop)),
      closedBite * 0.94
    );
    // Strict vertical separation gap below upperEdgeY so upper & lower teeth NEVER fuse or overlap
    float clearBelowUpper = smoothstep(upperEdgeY + 0.004, upperEdgeY + 0.011, y);
    float lowerZone =
      smoothstep(lowerTipLocalY - yFeather, lowerTipLocalY + yFeather, jawY) *
      clearBelowUpper *
      lowerShowGate *
      (1.0 - smoothstep(0.31, 0.43, absAx));

    // Keep lower teeth cleanly inside the lower labial vestibule so they sit BEHIND the opaque lower lip
    float lowerEdgeGuard = 1.0 - smoothstep(botCut - 0.11, botCut - 0.02, vMouthLipT);

    oralCol = mix(oralCol, tongueCol, tongueMask * (1.0 - tongueAdv * 0.12));
    oralCol = mix(oralCol, upperToothCol, upperZone * 0.97);
    oralCol = mix(oralCol, lowerToothCol, lowerZone * lowerEdgeGuard * 0.95 * (1.0 - tongueAdv * 0.75) * (1.0 - clamp(uLipBite, 0.0, 1.0) * 0.85));
    if (tongueAdv > 0.06) {
      oralCol = mix(oralCol, tongueCol, tongueMask * smoothstep(0.06, 0.38, tongueAdv) * 0.94);
    }

    // Crisp dark occlusal shadow line when teeth meet on a closed smile so upper & lower rows remain distinctly separated
    if (closedBite > 0.08) {
      float biteGap = exp(-pow((y - (upperEdgeY + 0.005)) / 0.0032, 2.0));
      oralCol = mix(oralCol, vec3(0.025, 0.004, 0.006), biteGap * closedBite * 0.84 * (1.0 - smoothstep(0.32, 0.43, absAx)));
    }

    if (uLipBite > 0.05) {
      float biteBand = smoothstep(0.54, 0.92, vMouthLipT) * (1.0 - smoothstep(0.28, 0.56, absX));
      vec3 tuckedLipCol = mix(uLipColor * 0.84, upperToothCol, smoothstep(0.54, 0.70, vMouthLipT) * 0.45);
      oralCol = mix(oralCol, tuckedLipCol, biteBand * clamp(uLipBite, 0.0, 1.0) * 0.82);
    }

    float topAO = mix(0.62, 1.0, smoothstep(0.0, 0.14, vMouthLipT));
    float botAO = mix(0.64, 1.0, smoothstep(botCut, botCut - 0.16, vMouthLipT));
    float sideAO = 1.0 - 0.86 * smoothstep(0.46, 0.88, absX);
    float cavityLight = mix(0.80, 1.0, max(smoothstep(0.02, 0.22, activeOpen), closedBite));
    vec3 buccalShadow = clamp(mix(vec3(0.025, 0.004, 0.006), uLipColor * 0.24, 0.28), vec3(0.015, 0.002, 0.004), vec3(0.14, 0.05, 0.05));
    oralCol = mix(buccalShadow, oralCol, topAO * botAO * sideAO * cavityLight);

    // Homogeneous natural lip-rim edge shading matching real lip pigment into oral cavity depth
    float rimSoft = smoothstep(0.0, 0.22, topRim * botRim * sideAlmondFade);
    vec3 rimHomogCol = clamp(mix(vec3(0.045, 0.008, 0.012), uLipColor * 0.34, 0.35), vec3(0.02, 0.004, 0.006), vec3(0.20, 0.07, 0.07));
    oralCol = mix(rimHomogCol, oralCol, rimSoft);

    fragColor = vec4(oralCol, openMask * lipRimFeather * fgAlpha);
    return;
  }

  // Diagnostic face lines: thin, soft, nearly white so eye detail remains crisp underneath
  if (uWireframe > 0.5) {
    fragColor = vec4(0.97, 0.98, 1.0, 0.30);
    return;
  }

  col = applyExpressionWrinkles(col, vFaceCoord, vRigidMouth);
  col = renderHumanEye(
    col,
    uREyeCenter,
    uREyePupil,
    uREyeAxis,
    uREyeScale,
    uREyeUpperA,
    uREyeUpperB,
    uREyeLowerA,
    uREyeLowerB,
    uRIrisRad,
    uRScleraNeg,
    uRScleraPos,
    uBlinkR,
    -1.0,
    uAspect
  );
  col = renderHumanEye(
    col,
    uLEyeCenter,
    uLEyePupil,
    uLEyeAxis,
    uLEyeScale,
    uLEyeUpperA,
    uLEyeUpperB,
    uLEyeLowerA,
    uLEyeLowerB,
    uLIrisRad,
    uLScleraNeg,
    uLScleraPos,
    uBlinkL,
    1.0,
    uAspect
  );

  col = applyPhysicalEffects(col, vFaceCoord);

  // Subtle homogeneous mouth-rim harmonizing smoothing along outer lip corners when opening/stretching mouth
  float mouthEdgeHarm = exp(-pow((abs(vRigidMouth.x) - 0.88) / 0.22, 2.0)) *
    exp(-pow((vRigidMouth.y - (uRestMouthTopY + uRestMouthBotY) * 0.5) / 0.14, 2.0));
  float lipToneMatch = clamp(0.14 * mouthEdgeHarm * smoothstep(0.08, 0.55, uOpen + uWide * 0.5), 0.0, 0.18);
  col = mix(col, mix(col, uLipColor * 0.82, 0.45), lipToneMatch);

  // Subtle larynx-centered throat swallow shadow wave when swallowing saliva
  if (uSwallow > 0.01) {
    float applePosY = uAdamAppleCoord.y - uSwallow * 0.14;
    float swBand = exp(-pow((vFaceCoord.y - applePosY) / 0.15, 2.0)) *
      (1.0 - smoothstep(0.10, 0.34, abs(vFaceCoord.x - uAdamAppleCoord.x)));
    col *= 1.0 - swBand * clamp(uSwallow, 0.0, 1.0) * 0.095;
  }

  // Crisp tight edge feather without a wide blurry halo moat against detailed backgrounds
  if (fgAlpha > 0.02 && fgAlpha < 0.26) {
    vec3 bgApprox = uBlurredBg > 0.5 ? sampleBlurredBg(uvSafe) : texture(uBgTex, uvSafe).rgb;
    float fringe = 1.0 - smoothstep(0.02, 0.26, fgAlpha);
    col = mix(col, bgApprox, fringe * 0.28);
  }

  fragColor = vec4(col, fgAlpha);
}
`;

type BuiltMeshes = {
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

function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0 || 1e-6)));
  return t * t * (3 - 2 * t);
}

/**
 * Generates two GPU-ready canvases from the portrait:
 * 1) `fgMaskCanvas`: Soft cutout alpha mask isolating head, hair, ears, neck & torso.
 * 2) `bgCanvas`: Background with the person cleanly erased & smoothly reconstructed behind them
 *    so moving head/hair/ears never expose black gaps or ghost duplicate features.
 */
function buildPortraitLayers(
  img: HTMLImageElement,
  face: FacePts,
): {
  fgMaskCanvas: HTMLCanvasElement;
  bgCanvas: HTMLCanvasElement;
  bgBlurCanvas: HTMLCanvasElement;
} {
  const srcW = img.naturalWidth || img.width || 512;
  const srcH = img.naturalHeight || img.height || 512;
  const maxDim = 512;
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

  // Anatomical head+hair ellipse and neck/torso silhouette
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
  const segArr = new Float32Array(W * H);

  for (let py = 0; py < H; py++) {
    const ny = (py + 0.5) / H;
    for (let px = 0; px < W; px++) {
      const nx = (px + 0.5) / W;

      // Protect only true interior face region so eyes/nose/mouth never drop out; contour follows seg edge tightly
      const coreDist = Math.hypot(
        ((nx - face.center[0]) * aspect) / Math.max(0.05, innerFaceRx),
        (ny - face.center[1]) / Math.max(0.05, innerFaceRy),
      );
      const faceCore = 1 - smoothstep(0.76, 0.94, coreDist);

      // Always build geometric head+hair+neck+torso envelope so segmentation dropouts never leave arms/hair in background
      const headDist = Math.hypot(
        ((nx - headCenterX) * aspect) / Math.max(0.08, headRx),
        (ny - headCenterY) / Math.max(0.08, headRy),
      );
      const headHalo = 1 - smoothstep(0.92, 1.04, headDist);
      let bodyMask = 0;
      if (ny > chin[1] - face.faceH * 0.14) {
        const neckHalfW =
          face.faceW *
          (0.48 +
            smoothstep(chin[1], Math.min(1, face.neckBase[1] + face.faceH * 0.34), ny) * 0.92);
        const lat = Math.abs((nx - face.neckBase[0]) * aspect) / Math.max(0.08, neckHalfW);
        bodyMask =
          smoothstep(chin[1] - face.faceH * 0.14, chin[1] + face.faceH * 0.06, ny) *
          (1 - smoothstep(0.84, 1.04, lat));
      }
      const geomEnvelope = Math.max(faceCore, headHalo * 0.85, bodyMask * 0.82);

      let fg: number;
      if (hasSegMask) {
        const seg = sampleSeg(nx, ny);
        segArr[py * W + px] = Math.max(seg, geomEnvelope * 0.72);
        const segTight = smoothstep(0.24, 0.5, seg);
        fg = Math.max(faceCore, segTight, bodyMask * 0.75);
      } else {
        fg = Math.max(faceCore, headHalo, bodyMask);
        segArr[py * W + px] = fg;
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

  // Dilate 9px around every foreground & outer hair fringe pixel so zero flyaway hairs ever stick in background
  const holeMask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let inHole = false;
      for (let dy = -9; dy <= 9 && !inHole; dy++) {
        const sy = y + dy;
        if (sy < 0 || sy >= H) continue;
        for (let dx = -9; dx <= 9; dx++) {
          const sx = x + dx;
          if (sx < 0 || sx >= W) continue;
          const idx = sy * W + sx;
          if (dx * dx + dy * dy <= 85 && (alphaArr[idx]! >= 0.05 || segArr[idx]! >= 0.1)) {
            inHole = true;
            break;
          }
        }
      }
      if (inHole) holeMask[y * W + x] = 1;
    }
  }

  // Alpha-normalized multi-scale push-pull inpainting (zero dark premultiplication bleed or hair halo)
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

  // Pull valid surrounding background colors smoothly inward into the foreground hole
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

  // Smooth strictly INSIDE occluded holeMask pixels; NEVER blur pristine background outside silhouette
  const filled = levels[0]!.rgba;
  for (let pass = 0; pass < 4; pass++) {
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

  // Fast 2-pass separable Gaussian/box bokeh directly on pixel buffer so ALL models blur consistently
  if (blurCtx) {
    const src = bgImgData.data;
    const tmpBuf = new Float32Array(W * H * 3);
    const outBuf = new Float32Array(W * H * 3);
    for (let i = 0; i < W * H; i++) {
      tmpBuf[i * 3] = src[i * 4]!;
      tmpBuf[i * 3 + 1] = src[i * 4 + 1]!;
      tmpBuf[i * 3 + 2] = src[i * 4 + 2]!;
    }
    const rad = Math.max(6, Math.round(Math.min(W, H) * 0.026));
    const win = rad * 2 + 1;
    for (let pass = 0; pass < 2; pass++) {
      const inB = tmpBuf;
      const midB = outBuf;
      // Horizontal pass (tmpBuf -> outBuf)
      for (let y = 0; y < H; y++) {
        let sr = 0;
        let sg = 0;
        let sb = 0;
        for (let k = -rad; k <= rad; k++) {
          const sx = Math.max(0, Math.min(W - 1, k));
          const o = (y * W + sx) * 3;
          sr += inB[o]!;
          sg += inB[o + 1]!;
          sb += inB[o + 2]!;
        }
        for (let x = 0; x < W; x++) {
          const o = (y * W + x) * 3;
          midB[o] = sr / win;
          midB[o + 1] = sg / win;
          midB[o + 2] = sb / win;
          const addX = Math.min(W - 1, x + rad + 1);
          const remX = Math.max(0, x - rad);
          const ao = (y * W + addX) * 3;
          const ro = (y * W + remX) * 3;
          sr += inB[ao]! - inB[ro]!;
          sg += inB[ao + 1]! - inB[ro + 1]!;
          sb += inB[ao + 2]! - inB[ro + 2]!;
        }
      }
      // Vertical pass (outBuf -> tmpBuf)
      const dstB = tmpBuf;
      for (let x = 0; x < W; x++) {
        let sr = 0;
        let sg = 0;
        let sb = 0;
        for (let k = -rad; k <= rad; k++) {
          const sy = Math.max(0, Math.min(H - 1, k));
          const o = (sy * W + x) * 3;
          sr += midB[o]!;
          sg += midB[o + 1]!;
          sb += midB[o + 2]!;
        }
        for (let y = 0; y < H; y++) {
          const o = (y * W + x) * 3;
          dstB[o] = sr / win;
          dstB[o + 1] = sg / win;
          dstB[o + 2] = sb / win;
          const addY = Math.min(H - 1, y + rad + 1);
          const remY = Math.max(0, y - rad);
          const ao = (addY * W + x) * 3;
          const ro = (remY * W + x) * 3;
          sr += midB[ao]! - midB[ro]!;
          sg += midB[ao + 1]! - midB[ro + 1]!;
          sb += midB[ao + 2]! - midB[ro + 2]!;
        }
      }
    }
    const blurImg = blurCtx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      blurImg.data[i * 4] = Math.round(Math.max(0, Math.min(255, tmpBuf[i * 3]!)));
      blurImg.data[i * 4 + 1] = Math.round(Math.max(0, Math.min(255, tmpBuf[i * 3 + 1]!)));
      blurImg.data[i * 4 + 2] = Math.round(Math.max(0, Math.min(255, tmpBuf[i * 3 + 2]!)));
      blurImg.data[i * 4 + 3] = 255;
    }
    blurCtx.putImageData(blurImg, 0, 0);
  }

  return { fgMaskCanvas, bgCanvas, bgBlurCanvas };
}

function evalLidCurveJS(
  t: number,
  arch: [number, number, number, number, number, number, number, number],
): number {
  const x = Math.max(-1, Math.min(1, t));
  if (x <= -0.82) return arch[0] * smoothstep(-1.0, -0.82, x);
  if (x <= -0.58) return arch[0] + (arch[1] - arch[0]) * smoothstep(-0.82, -0.58, x);
  if (x <= -0.34) return arch[1] + (arch[2] - arch[1]) * smoothstep(-0.58, -0.34, x);
  if (x <= -0.11) return arch[2] + (arch[3] - arch[2]) * smoothstep(-0.34, -0.11, x);
  if (x <= 0.11) return arch[3] + (arch[4] - arch[3]) * smoothstep(-0.11, 0.11, x);
  if (x <= 0.34) return arch[4] + (arch[5] - arch[4]) * smoothstep(0.11, 0.34, x);
  if (x <= 0.58) return arch[5] + (arch[6] - arch[5]) * smoothstep(0.34, 0.58, x);
  if (x <= 0.82) return arch[6] + (arch[7] - arch[6]) * smoothstep(0.58, 0.82, x);
  return arch[7] * (1 - smoothstep(0.82, 1.0, x));
}

/**
/**
 * Builds an offscreen 2x1 eye-cutout atlas (right eye tile + left eye tile):
 * - Samples ONLY light, clean left-of-cornea and right-of-cornea sclera tones (never caruncle red or cornea edge).
 * - Fills the full eyeball square with Left Sclera Color on the left, Right Sclera Color on the right, and a smooth
 *   horizontal gradient blending both across the center.
 * - Extracts ONLY the unmodified circular cornea+iris+pupil (completing any lid-clipped iris arc symmetrically)
 *   and places it centered cleanly inside that square.
 */
function buildEyeCutoutAtlas(img: HTMLImageElement, face: FacePts): HTMLCanvasElement {
  const TILE_W = 192;
  const TILE_H = 192;
  const EX_SPAN = 1.9;
  const EY_SPAN = 2.9;

  const atlas = document.createElement("canvas");
  atlas.width = TILE_W * 2;
  atlas.height = TILE_H;
  const aCtx = atlas.getContext("2d");
  if (!aCtx) return atlas;

  const natW = img.naturalWidth || img.width || 512;
  const natH = img.naturalHeight || img.height || 512;
  const sampleScale = Math.min(1, 1024 / Math.max(natW, natH));
  const sw = Math.max(64, Math.round(natW * sampleScale));
  const sh = Math.max(64, Math.round(natH * sampleScale));
  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = sw;
  srcCanvas.height = sh;
  const sCtx = srcCanvas.getContext("2d", { willReadFrequently: true });
  if (!sCtx) return atlas;
  sCtx.drawImage(img, 0, 0, sw, sh);
  const srcPx = sCtx.getImageData(0, 0, sw, sh).data;
  const aspect = face.aspect;

  const sampleBilinear = (nx: number, ny: number): [number, number, number] => {
    const fx = Math.max(0, Math.min(sw - 1, nx * (sw - 1)));
    const fy = Math.max(0, Math.min(sh - 1, ny * (sh - 1)));
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(sw - 1, x0 + 1);
    const y1 = Math.min(sh - 1, y0 + 1);
    const tx = fx - x0;
    const ty = fy - y0;
    const o00 = (y0 * sw + x0) * 4;
    const o10 = (y0 * sw + x1) * 4;
    const o01 = (y1 * sw + x0) * 4;
    const o11 = (y1 * sw + x1) * 4;
    const w00 = (1 - tx) * (1 - ty);
    const w10 = tx * (1 - ty);
    const w01 = (1 - tx) * ty;
    const w11 = tx * ty;
    return [
      srcPx[o00]! * w00 + srcPx[o10]! * w10 + srcPx[o01]! * w01 + srcPx[o11]! * w11,
      srcPx[o00 + 1]! * w00 + srcPx[o10 + 1]! * w10 + srcPx[o01 + 1]! * w01 + srcPx[o11 + 1]! * w11,
      srcPx[o00 + 2]! * w00 + srcPx[o10 + 2]! * w10 + srcPx[o01 + 2]! * w01 + srcPx[o11 + 2]! * w11,
    ];
  };

  const renderTile = (
    offsetX: number,
    eyeCenter: [number, number],
    eyePupil: [number, number],
    eyeAxis: FacePts["rightEyeAxis"],
    curve: FacePts["rightEyeCurve"],
    irisRad: number,
    outerSign: -1 | 1,
    fallbackNeg: [number, number, number],
    fallbackPos: [number, number, number],
  ) => {
    const halfW = Math.max(0.005, curve.halfW);
    const scaleY = Math.max(0.004, curve.scaleY);
    const dpX = (eyePupil[0] - eyeCenter[0]) * aspect;
    const dpY = eyePupil[1] - eyeCenter[1];
    const pupilEx = (dpX * eyeAxis.ux + dpY * eyeAxis.uy) / halfW;
    const pupilEy = (dpX * eyeAxis.vx + dpY * eyeAxis.vy) / scaleY;
    const irisExRad = Math.max(0.18, irisRad / halfW);

    const toNormPhoto = (ex: number, ey: number): [number, number] => {
      const dx = eyeAxis.ux * (ex * halfW) + eyeAxis.vx * (ey * scaleY);
      const dy = eyeAxis.uy * (ex * halfW) + eyeAxis.vy * (ey * scaleY);
      return [eyeCenter[0] + dx / aspect, eyeCenter[1] + dy];
    };

    // Sample clean light sclera tones strictly left-of-cornea & right-of-cornea (never cornea edge, lashes, or red caruncle)
    type ScleraSample = { score: number; r: number; g: number; b: number; lum: number };
    const leftCands: ScleraSample[] = [];
    const rightCands: ScleraSample[] = [];

    for (let ey = -0.85; ey <= 0.85; ey += 0.05) {
      for (let ex = -0.78; ex <= 0.78; ex += 0.04) {
        const sEx = ex * outerSign;
        // Skip inner corner caruncle zone & outer corner lash wing
        if (sEx < -0.62 || sEx > 0.76) continue;
        const topY = evalLidCurveJS(ex, curve.upperArch);
        const botY = evalLidCurveJS(ex, curve.lowerArch);
        const safeTop = topY + Math.max(0.18, (botY - topY) * 0.22);
        const safeBot = botY - Math.max(0.15, (botY - topY) * 0.18);
        if (ey < safeTop || ey > safeBot) continue;

        const [nx, ny] = toNormPhoto(ex, ey);
        const distP = Math.hypot((nx - eyePupil[0]) * aspect, ny - eyePupil[1]);
        if (distP < irisRad * 1.28) continue;

        const [r, g, b] = sampleBilinear(nx, ny);
        if (r - g > 16 && r - b > 22) continue;
        const lum = r * 0.299 + g * 0.587 + b * 0.114;
        if (lum < 72) continue;
        const maxC = Math.max(r, g, b);
        const minC = Math.min(r, g, b);
        const sat = maxC > 0 ? (maxC - minC) / maxC : 0;
        const redness = Math.max(0, r - (g + b) * 0.5);
        const cand: ScleraSample = { score: lum - sat * 45 - redness * 1.25, r, g, b, lum };

        if (ex <= pupilEx - irisExRad * 1.15) {
          leftCands.push(cand);
        } else if (ex >= pupilEx + irisExRad * 1.15) {
          rightCands.push(cand);
        }
      }
    }

    const pickLightSclera = (
      cands: ScleraSample[],
      fb: [number, number, number],
    ): [number, number, number] => {
      if (cands.length < 3) return fb;
      const sorted = [...cands].sort((a, b) => a.score - b.score);
      const start = Math.floor(sorted.length * 0.55);
      const end = Math.max(start + 1, Math.floor(sorted.length * 0.9));
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let n = 0;
      for (let i = start; i < end; i++) {
        sr += sorted[i]!.r;
        sg += sorted[i]!.g;
        sb += sorted[i]!.b;
        n++;
      }
      return n > 0 ? [sr / n, sg / n, sb / n] : fb;
    };

    const rawLeft = pickLightSclera(leftCands, fallbackNeg);
    const rawRight = pickLightSclera(rightCands, fallbackPos);
    const lLum = rawLeft[0] * 0.299 + rawLeft[1] * 0.587 + rawLeft[2] * 0.114;
    const rLum = rawRight[0] * 0.299 + rawRight[1] * 0.587 + rawRight[2] * 0.114;
    const maxLum = Math.max(lLum, rLum, 130);
    // Keep both left & right sclera clean, light and naturally matched without harsh contrast seam
    const stabilizeSide = (
      rgb: [number, number, number],
      lum: number,
    ): [number, number, number] => {
      const lift = lum < maxLum * 0.92 ? (maxLum * 0.94) / Math.max(40, lum) : 1;
      const clampedLift = Math.min(1.24, Math.max(1.0, lift));
      return [
        Math.min(250, rgb[0] * clampedLift),
        Math.min(249, rgb[1] * clampedLift),
        Math.min(248, rgb[2] * clampedLift),
      ];
    };
    const leftColor = stabilizeSide(rawLeft, lLum);
    const rightColor = stabilizeSide(rawRight, rLum);

    const tileImg = aCtx.createImageData(TILE_W, TILE_H);
    for (let ly = 0; ly < TILE_H; ly++) {
      const ey = ((ly + 0.5) / TILE_H - 0.5) * 2 * EY_SPAN;
      for (let lx = 0; lx < TILE_W; lx++) {
        const ex = ((lx + 0.5) / TILE_W - 0.5) * 2 * EX_SPAN;
        const o4 = (ly * TILE_W + lx) * 4;

        // Smooth horizontal gradient across central cornea zone blending left sclera into right sclera
        const gradT = smoothstep(-irisExRad * 1.1, irisExRad * 1.1, ex - pupilEx);
        let sr = leftColor[0] * (1 - gradT) + rightColor[0] * gradT;
        let sg = leftColor[1] * (1 - gradT) + rightColor[1] * gradT;
        let sb = leftColor[2] * (1 - gradT) + rightColor[2] * gradT;

        // Extract ONLY unmodified circular cornea + iris + pupil (completing any lid-clipped iris arc symmetrically)
        const [nx, ny] = toNormPhoto(ex, ey);
        const distP = Math.hypot((nx - eyePupil[0]) * aspect, ny - eyePupil[1]);
        const irisAlpha = 1 - smoothstep(irisRad * 0.86, irisRad * 0.99, distP);
        if (irisAlpha > 0.002) {
          const topY = evalLidCurveJS(ex, curve.upperArch);
          const botY = evalLidCurveJS(ex, curve.lowerArch);
          const safeTop = topY + Math.max(0.14, (botY - topY) * 0.14);
          const safeBot = botY - Math.max(0.1, (botY - topY) * 0.1);
          let sampleEy = ey;
          if (ey < safeTop) {
            sampleEy = Math.min(safeBot, pupilEy + Math.abs(pupilEy - ey));
          } else if (ey > safeBot) {
            sampleEy = Math.max(safeTop, pupilEy - Math.abs(ey - pupilEy));
          }
          const [ix, iy] = toNormPhoto(ex, sampleEy);
          const [ir, ig, ib] = sampleBilinear(ix, iy);
          sr = sr * (1 - irisAlpha) + ir * irisAlpha;
          sg = sg * (1 - irisAlpha) + ig * irisAlpha;
          sb = sb * (1 - irisAlpha) + ib * irisAlpha;
        }

        tileImg.data[o4] = Math.round(Math.max(0, Math.min(255, sr)));
        tileImg.data[o4 + 1] = Math.round(Math.max(0, Math.min(255, sg)));
        tileImg.data[o4 + 2] = Math.round(Math.max(0, Math.min(255, sb)));
        tileImg.data[o4 + 3] = 255;
      }
    }
    aCtx.putImageData(tileImg, offsetX, 0);
  };

  renderTile(
    0,
    face.rightEyeCenter,
    face.rightPupil,
    face.rightEyeAxis,
    face.rightEyeCurve,
    face.rightIrisRadius,
    -1,
    face.colors.scleraRightNeg,
    face.colors.scleraRightPos,
  );
  renderTile(
    TILE_W,
    face.leftEyeCenter,
    face.leftPupil,
    face.leftEyeAxis,
    face.leftEyeCurve,
    face.leftIrisRadius,
    1,
    face.colors.scleraLeftNeg,
    face.colors.scleraLeftPos,
  );
  return atlas;
}

function buildAvatarMeshes(face: FacePts): BuiltMeshes {
  const aspect = face.aspect;
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

  const STRIDE = 20;

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
    bgVerts[off + 7] = 1;
  }
  const bgIdx = new Uint16Array([0, 2, 1, 1, 2, 3]);

  // Solid unified cranium plateau covering 100% of face, ears & full hair silhouette evenly so hair never warps or detaches
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
  // Stop cranium/neck turn cleanly ABOVE lowerNeck so head turns never twist or pinch the trapezius or shoulders
  const neckFadeEndY = Math.min(
    0.96,
    Math.max(chin[1] + face.faceH * 0.22, face.lowerNeck[1] - face.faceH * 0.02),
  );

  // Full segmented hair & cranium silhouette locks solidly to headW=1.0 so hair stays 100% consistent with head
  const sampleHairSegMax = (nx: number, ny: number): number => {
    const mask = face.segMask;
    const sw = face.segWidth;
    const sh = face.segHeight;
    if (!mask || !sw || !sh) return 0;
    const sx = Math.max(0, Math.min(sw - 1, Math.round(nx * (sw - 1))));
    const sy = Math.max(0, Math.min(sh - 1, Math.round(ny * (sh - 1))));
    let best = mask[sy * sw + sx] ?? 0;
    for (const [dx, dy] of [
      [-4, 0],
      [4, 0],
      [0, -4],
      [0, 4],
      [-2, -2],
      [2, 2],
    ] as const) {
      const mx = Math.max(0, Math.min(sw - 1, sx + dx));
      const my = Math.max(0, Math.min(sh - 1, sy + dy));
      best = Math.max(best, mask[my * sw + mx] ?? 0);
    }
    return best;
  };
  const computeHeadWeight = (nx: number, ny: number): number => {
    const rx = ((nx - craniumX) * aspect) / Math.max(0.08, radX);
    const ry = (ny - craniumY) / Math.max(0.08, radY);
    const oval =
      ny <= chin[1]
        ? 1 - smoothstep(1.38, 2.15, Math.hypot(rx, ry))
        : 1 - smoothstep(1.16, 1.92, Math.hypot(rx, ry));
    const latOffset = Math.abs((nx - face.center[0]) * aspect);
    const hairYMax =
      latOffset > face.faceW * 0.24 ? chin[1] + face.faceH * 0.3 : chin[1] + face.faceH * 0.06;
    const upperHairSeg = ny <= hairYMax && sampleHairSegMax(nx, ny) > 0.14 ? 1 : 0;
    const radial = Math.max(oval, upperHairSeg);
    const neckFade = ny > neckFadeStartY ? 1 - smoothstep(neckFadeStartY, neckFadeEndY, ny) : 1;
    return radial * neckFade;
  };

  // 5-point anatomical yoke: lowerNeck -> rightTrapezius -> rightShoulder & lowerNeck -> leftTrapezius -> leftShoulder
  const shoulderHalfSpan = Math.max(
    face.faceW * 1.42,
    Math.hypot(
      (face.rightShoulder[0] - face.lowerNeck[0]) * aspect,
      face.rightShoulder[1] - face.lowerNeck[1],
    ) * 1.35,
    Math.hypot(
      (face.leftShoulder[0] - face.lowerNeck[0]) * aspect,
      face.leftShoulder[1] - face.lowerNeck[1],
    ) * 1.35,
  );
  const computeShoulderRidge = (nx: number, ny: number): number => {
    if (ny < chin[1] + face.faceH * 0.06) return 0;
    const dRightTrap = pointToSegmentDist(
      nx,
      ny,
      face.lowerNeck[0],
      face.lowerNeck[1],
      face.rightTrapezius[0],
      face.rightTrapezius[1],
      aspect,
    );
    const dRightOuter = pointToSegmentDist(
      nx,
      ny,
      face.rightTrapezius[0],
      face.rightTrapezius[1],
      face.rightShoulder[0],
      face.rightShoulder[1],
      aspect,
    );
    const dLeftTrap = pointToSegmentDist(
      nx,
      ny,
      face.lowerNeck[0],
      face.lowerNeck[1],
      face.leftTrapezius[0],
      face.leftTrapezius[1],
      aspect,
    );
    const dLeftOuter = pointToSegmentDist(
      nx,
      ny,
      face.leftTrapezius[0],
      face.leftTrapezius[1],
      face.leftShoulder[0],
      face.leftShoulder[1],
      aspect,
    );
    const nearYoke =
      1 -
      smoothstep(
        face.faceW * 0.24,
        face.faceW * 0.86,
        Math.min(dRightTrap, dRightOuter, dLeftTrap, dLeftOuter),
      );
    const belowUpperNeck = smoothstep(
      Math.max(chin[1] + 0.06 * face.faceH, face.lowerNeck[1] - 0.08 * face.faceH),
      face.lowerNeck[1] + 0.04 * face.faceH,
      ny,
    );
    return Math.max(0, Math.min(1, nearYoke * belowUpperNeck));
  };

  // Pure rounded elliptical mandible weight so jaw & chin deform smoothly in a curved oval arc
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

  // 1) Dense 52 x 52 Outer Portrait Grid with Smooth Continuous Weights
  const GX = 52;
  const GY = 52;
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
          shoulderRidgeW * 0.85,
          smoothstep(chin[1] + 0.06 * face.faceH, neckFadeEndY + 0.04 * face.faceH, ny) *
            (1 -
              smoothstep(
                shoulderHalfSpan * 0.72,
                Math.min(0.98, shoulderHalfSpan * 1.18),
                Math.abs((nx - neckPivot[0]) * aspect),
              )),
        ) * borderMask;

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
      baseVerts[ptr++] = 1;
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
      baseVerts[ptr++] = shoulderRidgeW;
      baseVerts[ptr++] = 0;
    }
  }

  // 2) High-Precision 468-Landmark Face Mesh
  const numPts = Math.min(468, pts.length);
  const faceVerts = new Float32Array(numPts * STRIDE);

  const ovalSet = new Set(face.ovalLoop);
  const innerUpperSet = new Set(INNER_UPPER_LIP);
  const innerLowerSet = new Set(INNER_LOWER_LIP);
  const innerLipSet = new Set(INNER_LIP_LOOP);
  const outerUpperSet = new Set(OUTER_UPPER_LIP);
  const outerLowerSet = new Set(OUTER_LOWER_LIP);
  const eyeLoopSet = new Set([...RIGHT_EYE_LOOP, ...LEFT_EYE_LOOP]);

  // Whole brow ribbons (both upper & lower brow borders move together uniformly so brows never thicken)
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
    const edgeFade = 1.0;

    // Smooth oval transition so inner mesh blends seamlessly into hair/ears/cheeks without edge creases
    const localScale = isOval ? 0.82 : 1.0;
    const relFaceX =
      (nx - noseTip[0]) * aspect * face.faceAxis.ux + (ny - noseTip[1]) * face.faceAxis.uy;
    const side = Math.max(-1, Math.min(1, relFaceX / (face.faceW * 0.22)));

    const mp = projMouth(pt);
    const mouthX = Math.max(-1, Math.min(1, mp.x));
    const lipCenterArch = Math.max(0.18, Math.cos(mouthX * Math.PI * 0.44));
    const yOuterUp = evalLipCurve(outerUpCurve, mouthX);
    const yInnerUp = evalLipCurve(innerUpCurve, mouthX);
    const yInnerLo = evalLipCurve(innerLoCurve, mouthX);
    const yOuterLo = evalLipCurve(outerLoCurve, mouthX);
    const ySlit = mouthMidSlitY(mouthX);

    const inMouthSpan = Math.abs(mp.x) <= 1.26;
    const isCornerVert = cornerLipSet.has(i);
    const isExplicitUpper = innerUpperSet.has(i) || outerUpperSet.has(i);
    const isExplicitLower = innerLowerSet.has(i) || outerLowerSet.has(i);
    const inUpperVermilion =
      inMouthSpan &&
      !isCornerVert &&
      !isExplicitLower &&
      mp.y >= yOuterUp - 0.004 * face.faceH &&
      mp.y <= ySlit;
    const inLowerVermilion =
      inMouthSpan &&
      !isCornerVert &&
      !isExplicitUpper &&
      mp.y >= ySlit &&
      mp.y <= yOuterLo + 0.004 * face.faceH;

    let jawW = computeMandibleWeight(nx, ny, mp.y - ySlit);

    let upperLipW =
      inMouthSpan && mp.y < ySlit
        ? (1 - smoothstep(0.0, face.mouthH * 0.95, Math.max(0, yOuterUp - mp.y))) *
          (0.36 + 0.64 * lipCenterArch) *
          localScale
        : 0;

    if (isCornerVert) {
      jawW = 0.24 * localScale;
      upperLipW = 0.22 * localScale;
    } else if (isExplicitUpper || inUpperVermilion) {
      jawW = 0.08 * (1 - lipCenterArch) * localScale;
      upperLipW = (0.34 + 0.66 * lipCenterArch) * localScale;
    } else if (isExplicitLower || inLowerVermilion) {
      // Smooth rounded arch plateau across lower lip vermilion tapering cleanly into both corners
      jawW = Math.max(0.28, 0.3 + 0.7 * lipCenterArch) * localScale;
      upperLipW = 0.06 * (1 - lipCenterArch) * localScale;
    } else if (mp.y <= yInnerUp) {
      jawW = 0;
    }

    const dMouth = dist2(nx, ny, mouthCenter[0], mouthCenter[1]);
    const dCorners = Math.min(
      dist2(nx, ny, rCorner[0], rCorner[1]),
      dist2(nx, ny, lCorner[0], lCorner[1]),
    );
    let cornerW = (1 - smoothstep(0.0, face.mouthW * 0.72, dCorners)) * localScale;
    if (
      innerLipSet.has(i) ||
      outerUpperSet.has(i) ||
      outerLowerSet.has(i) ||
      inUpperVermilion ||
      inLowerVermilion
    ) {
      cornerW = Math.max(
        cornerW,
        (1 - smoothstep(0.0, face.mouthW * 0.76, dMouth)) * 0.9 * localScale,
      );
    }

    // Expanded 3D malar cheekbone + zygomatic arch + nasolabial plane for organic facial unity
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

    const rNoseWing = pts[49] ?? pts[102] ?? noseTip;
    const lNoseWing = pts[279] ?? pts[331] ?? noseTip;
    const rNoseBridgeSide = pts[48] ?? pts[198] ?? rNoseWing;
    const lNoseBridgeSide = pts[278] ?? pts[420] ?? lNoseWing;
    const dWing = Math.min(
      dist2(nx, ny, rNoseWing[0], rNoseWing[1]),
      dist2(nx, ny, lNoseWing[0], lNoseWing[1]),
      dist2(nx, ny, rNoseBridgeSide[0], rNoseBridgeSide[1]),
      dist2(nx, ny, lNoseBridgeSide[0], lNoseBridgeSide[1]),
    );
    const noseW = (1 - smoothstep(0.0, face.faceW * 0.17, dWing)) * localScale;

    // Uniform eyebrow strip plateau: tarsal eyelid crease (ny >= lidCreaseY) stays 0.0, entire brow & immediate sub-brow skin stays flat 1.0 (zero hair stretching)
    const upperLidProtectSet = new Set([
      247, 30, 29, 27, 28, 56, 190, 467, 260, 259, 257, 258, 286, 414,
    ]);
    let innerBrowW = 0;
    let outerBrowW = 0;
    const isRBrow = rightBrowSet.has(i);
    const isLBrow = leftBrowSet.has(i);
    const xRatio = Math.min(1, Math.abs(relFaceX) / Math.max(0.02, face.faceW * 0.36));
    const isRightSide = relFaceX < 0;
    const evalArcYAtX = (idxs: number[], targetX: number): number => {
      const sorted = idxs.map((k) => point(k)).sort((a, b) => a[0] - b[0]);
      if (sorted.length === 0) return ny;
      if (targetX <= sorted[0]![0]) return sorted[0]![1];
      if (targetX >= sorted[sorted.length - 1]![0]) return sorted[sorted.length - 1]![1];
      for (let k = 0; k < sorted.length - 1; k++) {
        const a = sorted[k]!;
        const b = sorted[k + 1]!;
        if (targetX >= a[0] && targetX <= b[0]) {
          const f = (targetX - a[0]) / Math.max(1e-5, b[0] - a[0]);
          return a[1] + (b[1] - a[1]) * f;
        }
      }
      return sorted[0]![1];
    };
    const browTopPts = isRightSide ? [70, 63, 105, 66, 107] : [336, 296, 334, 293, 300];
    const browBotPts = isRightSide ? [46, 53, 52, 65, 55] : [285, 295, 282, 283, 276];
    const upperEyePts = isRightSide
      ? [33, 246, 161, 160, 159, 158, 157, 173, 133]
      : [362, 398, 384, 385, 386, 387, 388, 466, 263];
    const activeBox = isRightSide ? rBrowBox : lBrowBox;
    const browSpanX = Math.max(0.03, activeBox.maxX - activeBox.minX);
    const horizDistOutside = Math.max(0, activeBox.minX - nx, nx - activeBox.maxX);
    const horizEnv = 1 - smoothstep(browSpanX * 0.04, browSpanX * 0.34, horizDistOutside);

    if (!isEyeAp && (isRBrow || isLBrow || (!upperLidProtectSet.has(i) && horizEnv > 0.005))) {
      if (isRBrow || isLBrow) {
        innerBrowW = (1 - smoothstep(0.18, 0.58, xRatio)) * 0.96;
        outerBrowW = smoothstep(0.16, 0.58, xRatio) * 0.96;
      } else {
        const browTopY = evalArcYAtX(browTopPts, nx);
        const browBotY = Math.max(browTopY + face.faceH * 0.008, evalArcYAtX(browBotPts, nx));
        const lidEdgeY = evalArcYAtX(upperEyePts, nx);
        const infraGap = Math.max(face.faceH * 0.035, lidEdgeY - browBotY);
        const subBrowBufferY = browBotY + Math.max(face.faceH * 0.014, infraGap * 0.3);
        const lidCreaseY = lidEdgeY - Math.max(face.faceH * 0.018, infraGap * 0.28);
        const hairPad = face.faceH * 0.014;

        let vertEnv = 0;
        if (ny < lidCreaseY) {
          if (ny > subBrowBufferY) {
            vertEnv = smoothstep(lidCreaseY, subBrowBufferY, ny);
          } else if (ny >= browTopY - hairPad) {
            vertEnv = 1.0;
          } else {
            vertEnv =
              1 - smoothstep(browTopY - hairPad, browTopY - hairPad - face.faceH * 0.16, ny);
          }
        }
        const browBand = vertEnv * horizEnv;
        innerBrowW = browBand * (1 - smoothstep(0.18, 0.58, xRatio)) * 0.96 * localScale;
        outerBrowW = browBand * smoothstep(0.16, 0.58, xRatio) * 0.96 * localScale;
      }
    }

    // Lower orbital rim & crow's-feet skin rises strongly during squint / Duchenne laugh
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
      smoothstep(0.0, 0.06, nx) *
      smoothstep(1.0, 0.94, nx) *
      smoothstep(0.0, 0.06, ny) *
      smoothstep(1.0, 0.94, ny);
    const headW =
      (ny <= chin[1] ? Math.max(0.96, computeHeadWeight(nx, ny)) : computeHeadWeight(nx, ny)) *
      borderMask;

    faceVerts[fPtr++] = nx;
    faceVerts[fPtr++] = ny;
    faceVerts[fPtr++] = nx;
    faceVerts[fPtr++] = ny;
    faceVerts[fPtr++] = headW;
    faceVerts[fPtr++] = 0;
    faceVerts[fPtr++] = jawW;
    faceVerts[fPtr++] = edgeFade;
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
    faceVerts[fPtr++] = 0; // aHandRig.z carries shoulderRidgeW when uPass < 1.5; mouthVerts overrides with t
    faceVerts[fPtr++] = mouthX;
  }

  // 3) High-resolution smooth multi-row oral interior mesh bridging INNER_UPPER_LIP <-> INNER_LOWER_LIP
  const M_COLS = INNER_UPPER_LIP.length;
  const M_ROWS = 11;
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
      mouthVerts[mPtr + 7] = 1.0;
      mouthVerts[mPtr + 18] = t;
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

  // Unified whole-portrait Delaunay mesh blending landmarks cleanly into outer portrait grid
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
    // Keep face mesh 100% watertight so stretched mouth corners never expose background pinholes
    continuous.push(a, b, c);
    if (a < numPts && b < numPts && c < numPts) {
      // Leave eye-aperture interior clear in wireframe mode so iris/pupil detail shines through
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

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type);
  if (!s) throw new Error("No se pudo crear el shader WebGL2");
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s) || "Error de compilación de shader";
    gl.deleteShader(s);
    throw new Error(log);
  }
  return s;
}

export function AnimatedAvatar({
  src,
  name,
  active,
  listening = false,
  emotion = "neutral",
  engine = "mediapipe-mesh",
  liveAvatarUrl = "http://localhost:7860",
  localOnly = false,
  fit = "contain",
  userFace,
  mirrorMode = false,
  showMesh = false,
  blurredBackground = false,
  animationStyle = "auto",
  className = "",
}: {
  src: string | null;
  name: string;
  active: boolean;
  emotion?: AvatarEmotion | undefined;
  listening?: boolean | undefined;
  gesture?: AvatarGesture | undefined;
  engine?: AvatarEngine | undefined;
  liveAvatarUrl?: string | undefined;
  localOnly?: boolean | undefined;
  fit?: "contain" | "cover" | undefined;
  userFace?: UserFaceObservation | undefined;
  mirrorMode?: boolean | undefined;
  showMesh?: boolean;
  blurredBackground?: boolean | undefined;
  animationStyle?: "auto" | "human" | "anime";
  className?: string | undefined;
}) {
  const effectiveSrc = !src || src.includes("/__l5e/") ? DEFAULT_FALLBACK_PHOTO : src;
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [face, setFace] = useState<FacePts | null>(null);
  const [rigReady, setRigReady] = useState(false);
  const [webglFailed, setWebglFailed] = useState(false);

  const listeningRef = useRef(listening);
  listeningRef.current = listening;
  const activeRef = useRef(active);
  activeRef.current = active;
  const emotionRef = useRef<AvatarEmotion>(emotion);
  emotionRef.current = emotion;
  const localOnlyRef = useRef(localOnly);
  localOnlyRef.current = localOnly;
  const fitRef = useRef<"contain" | "cover">(fit);
  fitRef.current = fit;
  const userFaceRef = useRef<UserFaceObservation | undefined>(userFace);
  userFaceRef.current = userFace;
  const mirrorModeRef = useRef(mirrorMode);
  mirrorModeRef.current = mirrorMode;
  const meshRef = useRef(showMesh);
  meshRef.current = showMesh;
  const blurredBgRef = useRef(!!blurredBackground);
  blurredBgRef.current = !!blurredBackground;
  const styleRef = useRef(animationStyle);
  styleRef.current = animationStyle;

  const resolvedStyleHint = "human" as const;

  useEffect(() => {
    let cancel = false;
    setWebglFailed(false);
    setImg(null);
    setFace(null);
    setRigReady(false);
    const i = new Image();
    i.crossOrigin = "anonymous";
    i.onload = async () => {
      if (cancel) return;
      setImg(i);
      const pts = await detectFace(i, effectiveSrc.slice(-48) + ":" + effectiveSrc.length, {
        localOnly: localOnlyRef.current,
        styleHint: resolvedStyleHint,
      });
      if (!cancel) setFace(pts);
    };
    i.onerror = () => {
      if (cancel || effectiveSrc === DEFAULT_FALLBACK_PHOTO) return;
      i.src = DEFAULT_FALLBACK_PHOTO;
    };
    i.src = effectiveSrc;
    return () => {
      cancel = true;
    };
  }, [effectiveSrc, resolvedStyleHint]);

  useEffect(() => {
    if (!img || !face || webglFailed) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      setWebglFailed(true);
      return;
    }

    let program: WebGLProgram;
    try {
      const vs = compileShader(gl, gl.VERTEX_SHADER, VERT_SRC);
      const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAG_SRC);
      const created = gl.createProgram();
      if (!created) throw new Error("No se pudo crear el retrato");
      program = created;
      gl.attachShader(program, vs);
      gl.attachShader(program, fs);
      gl.linkProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) || "Link failed");
      }
    } catch (e) {
      console.warn("WebGL2 shader error:", e);
      setWebglFailed(true);
      return;
    }

    const meshes = buildAvatarMeshes(face);
    const { fgMaskCanvas, bgCanvas, bgBlurCanvas } = buildPortraitLayers(img, face);
    const eyeAtlasCanvas = buildEyeCutoutAtlas(img, face);
    const imgW = img.naturalWidth || img.width || 512;
    const imgH = img.naturalHeight || img.height || 512;
    const imgAspect = imgW / imgH;

    // Texture 0: Source portrait image
    const tex = gl.createTexture();
    if (!tex) throw new Error("No se pudo preparar el retrato");
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Texture 1: Photorealistic dental arch & oral cavity texture
    const teethTex = gl.createTexture();
    if (!teethTex) throw new Error("No se pudo preparar el retrato");
    gl.bindTexture(gl.TEXTURE_2D, teethTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([38, 18, 20, 255]),
    );

    // Texture 2: Inpainted reconstructed background (sharp)
    const bgTex = gl.createTexture();
    if (!bgTex) throw new Error("No se pudo preparar el fondo");
    gl.bindTexture(gl.TEXTURE_2D, bgTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bgCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Texture 3: Soft foreground person cutout mask
    const fgMaskTex = gl.createTexture();
    if (!fgMaskTex) throw new Error("No se pudo preparar la máscara");
    gl.bindTexture(gl.TEXTURE_2D, fgMaskTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, fgMaskCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Texture 4: Smooth pre-filtered Gaussian optical bokeh background
    const bgBlurTex = gl.createTexture();
    if (!bgBlurTex) throw new Error("No se pudo preparar el bokeh");
    gl.bindTexture(gl.TEXTURE_2D, bgBlurTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bgBlurCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Texture 5: Cropped & sclera-extended real eye interior atlas
    const eyeAtlasTex = gl.createTexture();
    if (!eyeAtlasTex) throw new Error("No se pudo preparar la mirada");
    gl.bindTexture(gl.TEXTURE_2D, eyeAtlasTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, eyeAtlasCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    let disposed = false;
    const teethImg = new Image();
    teethImg.onload = () => {
      if (disposed) return;
      gl.bindTexture(gl.TEXTURE_2D, teethTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, teethImg);
    };
    teethImg.src = teethTextureUrl;

    const setupVao = (verts: Float32Array, indices: Uint16Array) => {
      const vao = gl.createVertexArray();
      if (!vao) throw new Error("No se pudo preparar el retrato");
      gl.bindVertexArray(vao);
      const vbo = gl.createBuffer();
      if (!vbo) throw new Error("No se pudo preparar el retrato");
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
      const ibo = gl.createBuffer();
      if (!ibo) throw new Error("No se pudo preparar el retrato");
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

      const strideBytes = 20 * 4;
      const attrs: Array<[string, number]> = [
        ["aPosUv", 0],
        ["aHeadRig", 16],
        ["aFaceRig", 32],
        ["aBrowRig", 48],
        ["aHandRig", 64],
      ];
      for (const [attrName, offset] of attrs) {
        const loc = gl.getAttribLocation(program, attrName);
        if (loc >= 0) {
          gl.enableVertexAttribArray(loc);
          gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, strideBytes, offset);
        }
      }
      gl.bindVertexArray(null);
      return { vao, vbo, ibo, count: indices.length };
    };

    const bgDraw = setupVao(meshes.bgVerts, meshes.bgIdx);
    const faceDraw = setupVao(meshes.faceVerts, meshes.faceIdx);
    const mouthDraw = setupVao(meshes.mouthVerts, meshes.mouthIdx);
    const wireDraw = setupVao(meshes.faceVerts, meshes.wireIdx);

    const loc = (u: string) => gl.getUniformLocation(program, u);
    const uLoc = {
      uStylized: loc("uStylized"),
      uWireframe: loc("uWireframe"),
      uPhotoTex: loc("uPhotoTex"),
      uTeethTex: loc("uTeethTex"),
      uBgTex: loc("uBgTex"),
      uFgMaskTex: loc("uFgMaskTex"),
      uBgBlurTex: loc("uBgBlurTex"),
      uEyeAtlasTex: loc("uEyeAtlasTex"),
      uBgTexel: loc("uBgTexel"),
      uBlurredBg: loc("uBlurredBg"),
      uPass: loc("uPass"),
      uViewSize: loc("uViewSize"),
      uImgSize: loc("uImgSize"),
      uCoverOffset: loc("uCoverOffset"),
      uCoverSize: loc("uCoverSize"),
      uNeckPivot: loc("uNeckPivot"),
      uJawPivot: loc("uJawPivot"),
      uMouthCenter: loc("uMouthCenter"),
      uFaceCenter: loc("uFaceCenter"),
      uFaceAxis: loc("uFaceAxis"),
      uMouthAxis: loc("uMouthAxis"),
      uFaceW: loc("uFaceW"),
      uFaceH: loc("uFaceH"),
      uMouthW: loc("uMouthW"),
      uAspect: loc("uAspect"),
      uYaw: loc("uYaw"),
      uTime: loc("uTime"),
      uOpen: loc("uOpen"),
      uWide: loc("uWide"),
      uRound: loc("uRound"),
      uLipPress: loc("uLipPress"),
      uLabiodental: loc("uLabiodental"),
      uUpperLipLift: loc("uUpperLipLift"),
      uLowerLipDrop: loc("uLowerLipDrop"),
      uLipBite: loc("uLipBite"),
      uSmile: loc("uSmile"),
      uSmileR: loc("uSmileR"),
      uSmileL: loc("uSmileL"),
      uMouthSkewX: loc("uMouthSkewX"),
      uCheekLift: loc("uCheekLift"),
      uInnerBrowY: loc("uInnerBrowY"),
      uInnerBrowPinch: loc("uInnerBrowPinch"),
      uOuterBrowY: loc("uOuterBrowY"),
      uBrowAsym: loc("uBrowAsym"),
      uSquint: loc("uSquint"),
      uEyeWide: loc("uEyeWide"),
      uEyeRoll: loc("uEyeRoll"),
      uLidDroop: loc("uLidDroop"),
      uNoseFlare: loc("uNoseFlare"),
      uTongueOut: loc("uTongueOut"),
      uTearAmount: loc("uTearAmount"),
      uEyeWater: loc("uEyeWater"),
      uEyeRedness: loc("uEyeRedness"),
      uPupilDilate: loc("uPupilDilate"),
      uSweatAmount: loc("uSweatAmount"),
      uBlushAmount: loc("uBlushAmount"),
      uHeadRot: loc("uHeadRot"),
      uHeadRoll: loc("uHeadRoll"),
      uNeckRot: loc("uNeckRot"),
      uNeckRoll: loc("uNeckRoll"),
      uTorsoYaw: loc("uTorsoYaw"),
      uForwardLean: loc("uForwardLean"),
      uHeadTranslate: loc("uHeadTranslate"),
      uTorsoBreath: loc("uTorsoBreath"),
      uTorsoSwayX: loc("uTorsoSwayX"),
      uShoulderTilt: loc("uShoulderTilt"),
      uShoulderShrug: loc("uShoulderShrug"),
      uHand0Delta: loc("uHand0Delta"),
      uHand1Delta: loc("uHand1Delta"),
      uHairDelta: loc("uHairDelta"),
      uSwallow: loc("uSwallow"),
      uAdamAppleCoord: loc("uAdamAppleCoord"),
      uCheekCenters: loc("uCheekCenters"),
      uTearDucts: loc("uTearDucts"),
      uIsBasePass: loc("uIsBasePass"),
      uREyeCenter: loc("uREyeCenter"),
      uREyePupil: loc("uREyePupil"),
      uREyeAxis: loc("uREyeAxis"),
      uREyeScale: loc("uREyeScale"),
      uREyeUpperA: loc("uREyeUpperA"),
      uREyeUpperB: loc("uREyeUpperB"),
      uREyeLowerA: loc("uREyeLowerA"),
      uREyeLowerB: loc("uREyeLowerB"),
      uRIrisRad: loc("uRIrisRad"),
      uLEyeCenter: loc("uLEyeCenter"),
      uLEyePupil: loc("uLEyePupil"),
      uLEyeAxis: loc("uLEyeAxis"),
      uLEyeScale: loc("uLEyeScale"),
      uLEyeUpperA: loc("uLEyeUpperA"),
      uLEyeUpperB: loc("uLEyeUpperB"),
      uLEyeLowerA: loc("uLEyeLowerA"),
      uLEyeLowerB: loc("uLEyeLowerB"),
      uLIrisRad: loc("uLIrisRad"),
      uGaze: loc("uGaze"),
      uBlink: loc("uBlink"),
      uBlinkR: loc("uBlinkR"),
      uBlinkL: loc("uBlinkL"),
      uHasVisibleTeeth: loc("uHasVisibleTeeth"),
      uRestMouthTopY: loc("uRestMouthTopY"),
      uRestMouthBotY: loc("uRestMouthBotY"),
      uLipColor: loc("uLipColor"),
      uEnamelColor: loc("uEnamelColor"),
      uEyelidColor: loc("uEyelidColor"),
      uLowerEyelidColor: loc("uLowerEyelidColor"),
      uRScleraNeg: loc("uRScleraNeg"),
      uRScleraPos: loc("uRScleraPos"),
      uLScleraNeg: loc("uLScleraNeg"),
      uLScleraPos: loc("uLScleraPos"),
    };

    let viewW = 640;
    let viewH = 480;
    const syncSize = () => {
      const rect = canvas.getBoundingClientRect();
      const rawW = rect.width >= 16 ? rect.width : canvas.parentElement?.clientWidth || 640;
      const rawH = rect.height >= 16 ? rect.height : canvas.parentElement?.clientHeight || 480;
      const budget = portraitRenderBudget(rawW, rawH, window.devicePixelRatio || 1);
      viewW = budget.width;
      viewH = budget.height;
      if (canvas.width !== viewW || canvas.height !== viewH) {
        canvas.width = viewW;
        canvas.height = viewH;
      }
    };
    syncSize();
    const ro = new ResizeObserver(syncSize);
    ro.observe(canvas);

    let contextLost = false;
    const onLost = (e: Event) => {
      e.preventDefault();
      contextLost = true;
    };
    const onRestored = () => {
      contextLost = false;
    };
    canvas.addEventListener("webglcontextlost", onLost);
    canvas.addEventListener("webglcontextrestored", onRestored);

    const reducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    const exprState = createExpressionEngine(performance.now());
    setRigReady(true);

    let raf = 0;
    let lastFrame = 0;
    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (contextLost || document.hidden || now - lastFrame < 1000 / 30) return;
      lastFrame = now;

      const isActive = activeRef.current;
      const visemes = isActive ? audioVisemes() : { level: 0, open: 0, wide: 0, round: 0 };

      const frame = stepExpressionEngine(exprState, now, {
        active: isActive,
        listening: listeningRef.current,
        emotion: emotionRef.current,
        visemes,
        hasVisibleTeeth: face.hasVisibleTeeth,
        restMouthRatio: face.restMouthRatio,
        userFace: userFaceRef.current,
        mirrorMode: mirrorModeRef.current,
        reducedMotion,
        animeStyle: false,
      });

      gl.viewport(0, 0, viewW, viewH);
      gl.clearColor(0.039, 0.039, 0.039, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      gl.useProgram(program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(uLoc.uPhotoTex, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, teethTex);
      gl.uniform1i(uLoc.uTeethTex, 1);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, bgTex);
      gl.uniform1i(uLoc.uBgTex, 2);

      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, fgMaskTex);
      gl.uniform1i(uLoc.uFgMaskTex, 3);

      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, bgBlurTex);
      gl.uniform1i(uLoc.uBgBlurTex, 4);

      gl.activeTexture(gl.TEXTURE5);
      gl.bindTexture(gl.TEXTURE_2D, eyeAtlasTex);
      gl.uniform1i(uLoc.uEyeAtlasTex, 5);

      gl.uniform2f(uLoc.uBgTexel, 1 / bgCanvas.width, 1 / bgCanvas.height);
      gl.uniform1f(uLoc.uBlurredBg, blurredBgRef.current ? 1.0 : 0.0);

      const isContain = fitRef.current === "contain";
      const fitScale = isContain
        ? Math.min(viewW / imgW, viewH / imgH)
        : Math.max(viewW / imgW, viewH / imgH);
      const drawnW = imgW * fitScale;
      const drawnH = imgH * fitScale;
      const offX = isContain
        ? (viewW - drawnW) * 0.5
        : Math.max(viewW - drawnW, Math.min(0, 0.5 * viewW - face.center[0] * drawnW));
      const offY = isContain
        ? (viewH - drawnH) * 0.5
        : Math.max(viewH - drawnH, Math.min(0, 0.43 * viewH - face.center[1] * drawnH));

      gl.uniform2f(uLoc.uViewSize, viewW, viewH);
      gl.uniform2f(uLoc.uImgSize, imgW, imgH);
      gl.uniform2f(uLoc.uCoverOffset, offX, offY);
      gl.uniform2f(uLoc.uCoverSize, drawnW, drawnH);

      gl.uniform2f(uLoc.uNeckPivot, meshes.neckPivot[0], meshes.neckPivot[1]);
      gl.uniform2f(uLoc.uJawPivot, meshes.jawPivot[0], meshes.jawPivot[1]);
      gl.uniform2f(uLoc.uMouthCenter, meshes.mouthCenter[0], meshes.mouthCenter[1]);
      gl.uniform2f(uLoc.uFaceCenter, face.center[0], face.center[1]);
      gl.uniform4f(
        uLoc.uFaceAxis,
        face.faceAxis.ux,
        face.faceAxis.uy,
        face.faceAxis.vx,
        face.faceAxis.vy,
      );
      gl.uniform4f(
        uLoc.uMouthAxis,
        face.mouthAxis.ux,
        face.mouthAxis.uy,
        face.mouthAxis.vx,
        face.mouthAxis.vy,
      );
      gl.uniform1f(uLoc.uFaceW, face.faceW);
      gl.uniform1f(uLoc.uFaceH, face.faceH);
      gl.uniform1f(uLoc.uMouthW, face.mouthW);
      gl.uniform1f(uLoc.uAspect, imgAspect);
      gl.uniform1f(uLoc.uYaw, face.yaw + frame.headYaw * 0.55);

      gl.uniform1f(uLoc.uStylized, 0);
      gl.uniform1f(uLoc.uWireframe, 0);
      gl.uniform1f(uLoc.uTime, now * 0.001);
      gl.uniform1f(uLoc.uOpen, frame.open);
      gl.uniform1f(uLoc.uWide, frame.wide);
      gl.uniform1f(uLoc.uRound, frame.round);
      gl.uniform1f(uLoc.uLipPress, frame.lipPress);
      gl.uniform1f(uLoc.uLabiodental, frame.labiodental);
      gl.uniform1f(uLoc.uUpperLipLift, frame.upperLipLift);
      gl.uniform1f(uLoc.uLowerLipDrop, frame.lowerLipDrop);
      gl.uniform1f(uLoc.uLipBite, frame.lipBite);
      gl.uniform1f(uLoc.uSmile, frame.smile);
      gl.uniform1f(uLoc.uSmileR, frame.smileR);
      gl.uniform1f(uLoc.uSmileL, frame.smileL);
      gl.uniform1f(uLoc.uMouthSkewX, frame.mouthSkewX);
      gl.uniform1f(uLoc.uCheekLift, frame.cheekLift);
      gl.uniform1f(uLoc.uInnerBrowY, frame.innerBrowY);
      gl.uniform1f(uLoc.uInnerBrowPinch, frame.innerBrowPinch);
      gl.uniform1f(uLoc.uOuterBrowY, frame.outerBrowY);
      gl.uniform1f(uLoc.uBrowAsym, frame.browAsym);
      gl.uniform1f(uLoc.uSquint, frame.eyeSquint);
      gl.uniform1f(uLoc.uEyeWide, frame.eyeWide);
      gl.uniform1f(uLoc.uEyeRoll, frame.eyeRoll);
      gl.uniform1f(uLoc.uLidDroop, frame.lidDroop);
      gl.uniform1f(uLoc.uNoseFlare, frame.noseFlare);
      gl.uniform1f(uLoc.uTongueOut, frame.tongueOut);
      gl.uniform1f(uLoc.uTearAmount, frame.tearAmount);
      gl.uniform1f(uLoc.uEyeWater, frame.eyeWater);
      gl.uniform1f(uLoc.uEyeRedness, frame.eyeRedness);
      gl.uniform1f(uLoc.uPupilDilate, frame.pupilDilate);
      gl.uniform1f(uLoc.uSweatAmount, frame.sweatAmount);
      gl.uniform1f(uLoc.uBlushAmount, frame.blushAmount);

      gl.uniform2f(uLoc.uHeadRot, frame.headYaw * 0.72, frame.headPitch * 0.52);
      gl.uniform1f(uLoc.uHeadRoll, frame.headRoll * 0.4);
      gl.uniform2f(uLoc.uNeckRot, frame.neckYaw * 0.64, frame.neckPitch * 0.48);
      gl.uniform1f(uLoc.uNeckRoll, frame.neckRoll * 0.42);
      gl.uniform1f(uLoc.uTorsoYaw, frame.torsoYaw);
      gl.uniform1f(uLoc.uForwardLean, frame.forwardLean);
      gl.uniform2f(uLoc.uHeadTranslate, frame.headTx * 0.85, frame.headTy * 0.85);
      gl.uniform1f(uLoc.uTorsoBreath, frame.torsoBreath);
      gl.uniform1f(uLoc.uTorsoSwayX, frame.torsoSwayX);
      gl.uniform1f(uLoc.uShoulderTilt, frame.shoulderTilt);
      gl.uniform1f(uLoc.uShoulderShrug, frame.shoulderShrug);
      gl.uniform2f(uLoc.uHand0Delta, frame.handShiftX, frame.handShiftY);
      gl.uniform2f(uLoc.uHand1Delta, -frame.handShiftX, frame.handShiftY);
      gl.uniform2f(uLoc.uHairDelta, frame.hairShiftX, frame.hairShiftY);
      gl.uniform1f(uLoc.uSwallow, frame.swallowAmount);
      const toFaceCoord = (pt: [number, number]): [number, number] => {
        const dx = (pt[0] - face.center[0]) * imgAspect;
        const dy = pt[1] - face.center[1];
        return [
          (dx * face.faceAxis.ux + dy * face.faceAxis.uy) / Math.max(0.05, face.faceW * 0.5),
          (dx * face.faceAxis.vx + dy * face.faceAxis.vy) / Math.max(0.05, face.faceH * 0.5),
        ];
      };
      const [aaX, aaY] = toFaceCoord(face.adamApple);
      gl.uniform2f(uLoc.uAdamAppleCoord, aaX, aaY);
      const [rChkX, rChkY] = toFaceCoord(face.rightCheekPos);
      const [lChkX, lChkY] = toFaceCoord(face.leftCheekPos);
      gl.uniform4f(uLoc.uCheekCenters, rChkX, rChkY, lChkX, lChkY);
      const [rDuctX, rDuctY] = toFaceCoord(face.rightInnerCanthus);
      const [lDuctX, lDuctY] = toFaceCoord(face.leftInnerCanthus);
      gl.uniform4f(uLoc.uTearDucts, rDuctX, rDuctY, lDuctX, lDuctY);

      gl.uniform2f(uLoc.uREyeCenter, face.rightEyeCenter[0], face.rightEyeCenter[1]);
      gl.uniform2f(uLoc.uREyePupil, face.rightPupil[0], face.rightPupil[1]);
      gl.uniform4f(
        uLoc.uREyeAxis,
        face.rightEyeAxis.ux,
        face.rightEyeAxis.uy,
        face.rightEyeAxis.vx,
        face.rightEyeAxis.vy,
      );
      gl.uniform4f(
        uLoc.uREyeScale,
        face.rightEyeCurve.halfW,
        face.rightEyeCurve.scaleY,
        face.rightEyeCurve.outerReach ?? 1.26,
        face.rightEyeCurve.outerWingY ?? 0,
      );
      gl.uniform4f(
        uLoc.uREyeUpperA,
        face.rightEyeCurve.upperArch[0],
        face.rightEyeCurve.upperArch[1],
        face.rightEyeCurve.upperArch[2],
        face.rightEyeCurve.upperArch[3],
      );
      gl.uniform4f(
        uLoc.uREyeUpperB,
        face.rightEyeCurve.upperArch[4],
        face.rightEyeCurve.upperArch[5],
        face.rightEyeCurve.upperArch[6],
        face.rightEyeCurve.upperArch[7],
      );
      gl.uniform4f(
        uLoc.uREyeLowerA,
        face.rightEyeCurve.lowerArch[0],
        face.rightEyeCurve.lowerArch[1],
        face.rightEyeCurve.lowerArch[2],
        face.rightEyeCurve.lowerArch[3],
      );
      gl.uniform4f(
        uLoc.uREyeLowerB,
        face.rightEyeCurve.lowerArch[4],
        face.rightEyeCurve.lowerArch[5],
        face.rightEyeCurve.lowerArch[6],
        face.rightEyeCurve.lowerArch[7],
      );
      gl.uniform1f(uLoc.uRIrisRad, face.rightIrisRadius);

      gl.uniform2f(uLoc.uLEyeCenter, face.leftEyeCenter[0], face.leftEyeCenter[1]);
      gl.uniform2f(uLoc.uLEyePupil, face.leftPupil[0], face.leftPupil[1]);
      gl.uniform4f(
        uLoc.uLEyeAxis,
        face.leftEyeAxis.ux,
        face.leftEyeAxis.uy,
        face.leftEyeAxis.vx,
        face.leftEyeAxis.vy,
      );
      gl.uniform4f(
        uLoc.uLEyeScale,
        face.leftEyeCurve.halfW,
        face.leftEyeCurve.scaleY,
        face.leftEyeCurve.outerReach ?? 1.26,
        face.leftEyeCurve.outerWingY ?? 0,
      );
      gl.uniform4f(
        uLoc.uLEyeUpperA,
        face.leftEyeCurve.upperArch[0],
        face.leftEyeCurve.upperArch[1],
        face.leftEyeCurve.upperArch[2],
        face.leftEyeCurve.upperArch[3],
      );
      gl.uniform4f(
        uLoc.uLEyeUpperB,
        face.leftEyeCurve.upperArch[4],
        face.leftEyeCurve.upperArch[5],
        face.leftEyeCurve.upperArch[6],
        face.leftEyeCurve.upperArch[7],
      );
      gl.uniform4f(
        uLoc.uLEyeLowerA,
        face.leftEyeCurve.lowerArch[0],
        face.leftEyeCurve.lowerArch[1],
        face.leftEyeCurve.lowerArch[2],
        face.leftEyeCurve.lowerArch[3],
      );
      gl.uniform4f(
        uLoc.uLEyeLowerB,
        face.leftEyeCurve.lowerArch[4],
        face.leftEyeCurve.lowerArch[5],
        face.leftEyeCurve.lowerArch[6],
        face.leftEyeCurve.lowerArch[7],
      );
      gl.uniform1f(uLoc.uLIrisRad, face.leftIrisRadius);

      gl.uniform2f(uLoc.uGaze, frame.gazeX, frame.gazeY);
      gl.uniform1f(uLoc.uBlink, frame.blink);
      gl.uniform1f(uLoc.uBlinkR, frame.blinkR);
      gl.uniform1f(uLoc.uBlinkL, frame.blinkL);
      gl.uniform1f(uLoc.uHasVisibleTeeth, face.hasVisibleTeeth ? 1 : 0);
      gl.uniform1f(uLoc.uRestMouthTopY, face.restMouthTopY);
      gl.uniform1f(uLoc.uRestMouthBotY, face.restMouthBotY);
      gl.uniform3f(
        uLoc.uLipColor,
        face.colors.lip[0] / 255,
        face.colors.lip[1] / 255,
        face.colors.lip[2] / 255,
      );
      gl.uniform3f(
        uLoc.uEnamelColor,
        face.colors.enamel[0] / 255,
        face.colors.enamel[1] / 255,
        face.colors.enamel[2] / 255,
      );
      gl.uniform3f(
        uLoc.uEyelidColor,
        face.colors.eyelid[0] / 255,
        face.colors.eyelid[1] / 255,
        face.colors.eyelid[2] / 255,
      );
      gl.uniform3f(
        uLoc.uLowerEyelidColor,
        face.colors.lowerEyelid[0] / 255,
        face.colors.lowerEyelid[1] / 255,
        face.colors.lowerEyelid[2] / 255,
      );
      gl.uniform3f(
        uLoc.uRScleraNeg,
        face.colors.scleraRightNeg[0] / 255,
        face.colors.scleraRightNeg[1] / 255,
        face.colors.scleraRightNeg[2] / 255,
      );
      gl.uniform3f(
        uLoc.uRScleraPos,
        face.colors.scleraRightPos[0] / 255,
        face.colors.scleraRightPos[1] / 255,
        face.colors.scleraRightPos[2] / 255,
      );
      gl.uniform3f(
        uLoc.uLScleraNeg,
        face.colors.scleraLeftNeg[0] / 255,
        face.colors.scleraLeftNeg[1] / 255,
        face.colors.scleraLeftNeg[2] / 255,
      );
      gl.uniform3f(
        uLoc.uLScleraPos,
        face.colors.scleraLeftPos[0] / 255,
        face.colors.scleraLeftPos[1] / 255,
        face.colors.scleraLeftPos[2] / 255,
      );

      // Pass -1: Inpainted background (clean or bokeh-blurred)
      gl.uniform1f(uLoc.uIsBasePass, 0.0);
      gl.uniform1f(uLoc.uPass, -1.0);
      gl.bindVertexArray(bgDraw.vao);
      gl.drawElements(gl.TRIANGLES, bgDraw.count, gl.UNSIGNED_SHORT, 0);

      // Pass 1: Unified cutout portrait liquify mesh
      gl.uniform1f(uLoc.uPass, 1.0);
      gl.bindVertexArray(faceDraw.vao);
      gl.drawElements(gl.TRIANGLES, faceDraw.count, gl.UNSIGNED_SHORT, 0);

      // Pass 2: Oral interior & dental arch
      if (mouthDraw.count > 0) {
        gl.uniform1f(uLoc.uPass, 2.0);
        gl.bindVertexArray(mouthDraw.vao);
        gl.drawElements(gl.TRIANGLES, mouthDraw.count, gl.UNSIGNED_SHORT, 0);
      }

      // Pass 1 (wireframe): Thin translucent near-white lines using the exact same vertex shader
      if (meshRef.current) {
        gl.uniform1f(uLoc.uWireframe, 1);
        gl.uniform1f(uLoc.uPass, 1.0);
        gl.bindVertexArray(wireDraw.vao);
        gl.drawElements(gl.LINES, wireDraw.count, gl.UNSIGNED_SHORT, 0);
      }
      containerRef.current?.setAttribute("data-scene-ready", "true");
      containerRef.current?.setAttribute("data-avatar-style", "human");
      gl.bindVertexArray(null);
    };

    raf = requestAnimationFrame(draw);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
      for (const d of [bgDraw, faceDraw, mouthDraw, wireDraw]) {
        gl.deleteVertexArray(d.vao);
        gl.deleteBuffer(d.vbo);
        gl.deleteBuffer(d.ibo);
      }
      gl.deleteTexture(tex);
      gl.deleteTexture(teethTex);
      gl.deleteTexture(bgTex);
      gl.deleteTexture(fgMaskTex);
      gl.deleteTexture(bgBlurTex);
      gl.deleteTexture(eyeAtlasTex);
      gl.deleteProgram(program);
    };
  }, [img, face, webglFailed]);

  return (
    <div ref={containerRef} className={`relative overflow-hidden select-none bg-card ${className}`}>
      <div
        className="absolute inset-0 bg-cover bg-center filter blur-2xl scale-110 opacity-70 pointer-events-none"
        style={{ backgroundImage: `url(${effectiveSrc})` }}
      />
      <img
        src={effectiveSrc}
        alt={name}
        className={`pointer-events-none absolute inset-0 h-full w-full ${rigReady && !webglFailed ? "invisible" : "visible"} ${
          fit === "contain" ? "object-contain object-center" : "object-cover object-[50%_28%]"
        }`}
      />
      {!webglFailed && (
        <canvas
          ref={canvasRef}
          className={`relative block h-full w-full ${rigReady ? "opacity-100" : "opacity-0"}`}
        />
      )}
      {engine === "liveavatar" && (
        <LiveAvatarStreamPlayer
          serverUrl={liveAvatarUrl}
          active={active}
          name={name}
          fallbackPhoto={effectiveSrc}
        />
      )}
    </div>
  );
}
