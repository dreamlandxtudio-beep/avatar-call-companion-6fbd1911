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
uniform float uPucker;
uniform float uLipPress;
uniform float uLabiodental;
uniform float uUpperLipLift;
uniform float uLowerLipDrop;
uniform float uLipBite;
uniform float uSmile;
uniform float uSmileR;
uniform float uSmileL;
uniform float uSmileTeeth;
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
uniform float uSwallowPhase;
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
out float vHeadTurn;

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
    vHeadTurn = 0.0;
    return;
  }

  float aspect = max(0.25, uAspect);
  float headW = aHeadRig.x;
  float torsoW = aHeadRig.y;
  float jawW = aHeadRig.z;
  float neckW = uPass < 1.5 ? aHandRig.z : 0.0;
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
  float orbitLift = aBrowRig.z * (uSquint * 0.016 - uEyeWide * 0.006) * uFaceH;
  p = addAxisDelta(p, uFaceAxis, vec2(browPinch, -(browLift - furrowDrop + orbitLift)), aspect);

  // 2) Subtle nostril breath only (keep nose bridge & contour completely anchored)
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(
      side * aFaceRig.w * max(0.0, uNoseFlare) * 0.0032 * uFaceW,
      -aFaceRig.w * max(0.0, uNoseFlare) * 0.0028 * uFaceH
    ),
    aspect
  );

  // 3) Gentle zygomatic cheek lift preserving rounded oval cheekbones (stronger on true smiles/laughs)
  float sideSmile = mix(uSmileR, uSmileL, smoothstep(-0.35, 0.35, side));
  float laughCheek = uCheekLift * 0.018 + max(0.0, sideSmile) * 0.014;
  float cheekSpread = side * aFaceRig.z * (max(0.0, sideSmile) * 0.006 + uCheekLift * 0.004) * uFaceW;
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(cheekSpread, -aFaceRig.z * laughCheek * uFaceH),
    aspect
  );

  // 4) Balanced mouth opening, pucker contraction ("besito"), closed/parted smile modes & parabolic jaw drop (zero neck pull)
  vec2 mouthRel = vec2((p.x - uMouthCenter.x) * aspect, p.y - uMouthCenter.y);
  float mSign = clamp(dot(mouthRel, uMouthAxis.xy) / max(0.01, uMouthW * 0.46), -1.0, 1.0);
  float localSmile = mix(uSmileR, uSmileL, smoothstep(-0.35, 0.35, mSign));
  float lipSealZone = 1.0 - smoothstep(uMouthW * 0.32, uMouthW * 0.86, length(mouthRel));
  float cornerTaper = 1.0 - 0.16 * pow(abs(mSign), 2.2);
  float centerArch = 1.0 - smoothstep(0.18, 0.92, abs(mSign));
  float closedTeethPart = uSmileTeeth * clamp((max(uSmileR, uSmileL) - 0.30) * 1.55, 0.0, 1.0) * (1.0 - smoothstep(0.06, 0.20, uOpen));

  float kissPlumpUpper = aFaceRig.x * centerArch * uPucker * 0.007 * uMouthW * (1.0 - uOpen);
  float kissPlumpLower = centerArch * (1.0 - aFaceRig.x) * jawW * uPucker * 0.006 * uFaceH * (1.0 - uOpen);
  float puckerInward = -mSign * (aFaceRig.y * 0.80 + aFaceRig.x * 0.36 + (1.0 - aFaceRig.x) * jawW * 0.26) * (uPucker * 0.080 + uRound * 0.022) * uMouthW;

  float upperLift = aFaceRig.x * cornerTaper * (
    uOpen * 0.032 + max(0.0, localSmile) * 0.032 + uUpperLipLift * 0.030 + closedTeethPart * 0.011 * centerArch + uRound * 0.006 - uLipPress * 0.009
  ) * uMouthW + kissPlumpUpper;
  float cornerWide = mSign * aFaceRig.y * (uWide * 0.022 + localSmile * 0.056 - uRound * 0.024 - uLipPress * 0.010) * uMouthW + puckerInward;
  float cornerLift = abs(mSign) * aFaceRig.y * (localSmile * 0.062 + uWide * 0.008 - uRound * 0.008 - uPucker * 0.004 + mSign * uMouthSkewX * 0.012) * uMouthW;
  float cornerJawFollow = aFaceRig.y * (1.0 - aFaceRig.x) * (uOpen * 0.010 + uLowerLipDrop * 0.008) * uFaceH;
  float mouthLatSkew = uMouthSkewX * 0.110 * uMouthW * clamp(aFaceRig.y * 1.05 + aFaceRig.x * 0.85 + jawW * 0.45, 0.0, 1.2);

  float wideBoost = smoothstep(0.42, 1.0, uOpen) * 0.028;
  float jawBoneDrop = (uOpen * 0.076 + wideBoost) * uFaceH;
  float jawOpenDrop = (uOpen * 0.074 + wideBoost + uLowerLipDrop * 0.024) * uFaceH;
  float lowerSmilePart = closedTeethPart * centerArch * (1.0 - aFaceRig.x) * 0.013 * uFaceH * jawW;
  float jawPressSeal = -jawW * lipSealZone * (uLipPress * 0.009 + uLabiodental * 0.011 + uLipBite * 0.015) * uFaceH;
  float jawDrop = jawOpenDrop * jawW + lowerSmilePart + jawPressSeal + kissPlumpLower;
  float jawInwardTuck = -mSign * jawW * (uOpen * 0.005 + uRound * 0.004 + uPucker * 0.008 + uLabiodental * 0.003 + uLipBite * 0.004) * uFaceW;
  p = addAxisDelta(
    p,
    uMouthAxis,
    vec2(
      cornerWide + jawInwardTuck + mouthLatSkew,
      jawDrop + cornerJawFollow - upperLift - cornerLift
    ),
    aspect
  );

  // Cranial mouth coordinates & decoupled mandible bone drop so teeth stay anchored cleanly inside the mouth
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

  // 5) Cohesive 3D head turn: entire cranium, top hair crown, lateral ear-side hair, ears, brows, nose, mouth & chin rotate as one unit
  vHeadTurn = 0.0;
  if (headW > 0.001) {
    vec2 skullPivot = mix(uNeckPivot, uFaceCenter, 0.44);
    vec2 rel = vec2((p.x - skullPivot.x) * aspect, p.y - skullPivot.y);
    float activeRoll = uHeadRoll * headW;
    float cRoll = cos(activeRoll);
    float sRoll = sin(activeRoll);
    vec2 r = vec2(rel.x * cRoll - rel.y * sRoll, rel.x * sRoll + rel.y * cRoll);

    float hx = clamp(vFaceCoord.x, -1.65, 1.65);
    float hy = clamp(vFaceCoord.y, -1.65, 1.25);
    float faceR = length(vec2(hx * 0.88, hy * 0.78));
    float innerMask = 1.0 - smoothstep(0.32, 0.98, faceR);
    float hairDomeMask = smoothstep(0.10, -1.12, hy) * (1.0 - smoothstep(1.02, 1.65, abs(hx)));
    float templeHairMask = smoothstep(0.42, 1.18, abs(hx)) * smoothstep(0.58, -0.60, hy) * (1.0 - smoothstep(1.26, 1.65, abs(hx)));
    float featureCohesion = max(innerMask, max(hairDomeMask * 0.96, templeHairMask * 0.92));

    float noseBump = exp(-(hx * hx * 9.0 + (hy - 0.06) * (hy - 0.06) * 5.2)) * 0.042 * uFaceW;
    float browRidgeBump = exp(-(hx * hx * 3.0 + (hy + 0.34) * (hy + 0.34) * 16.0)) * 0.019 * uFaceW;
    float mouthMalarBump = exp(-(hx * hx * 4.5 + (hy - 0.35) * (hy - 0.35) * 9.5)) * 0.022 * uFaceW;
    float chinBump = exp(-(hx * hx * 6.5 + (hy - 0.80) * (hy - 0.80) * 10.5)) * 0.022 * uFaceW;
    float crownBump = smoothstep(0.05, -1.22, hy) * (1.0 - smoothstep(0.82, 1.65, abs(hx))) * 0.056 * uFaceW;
    float r2 = (hx * 0.25) * (hx * 0.25) + (hy * 0.21) * (hy * 0.21);
    float lateralCyl = max(0.0, 1.0 - hx * hx * 0.20);
    float skullZ = (sqrt(max(0.18, 1.0 - min(0.82, r2))) * 0.22 + lateralCyl * 0.060) * uFaceW + noseBump + browRidgeBump + mouthMalarBump + chinBump + crownBump;
    float pivotZ = uFaceW * 0.16;
    float dz = skullZ - pivotZ;

    float yawRad = clamp(uHeadRot.x * 0.44, -0.46, 0.46) * headW;
    float pitchRad = clamp(uHeadRot.y * 0.20, -0.20, 0.20) * headW;

    float cosY = cos(yawRad);
    float sinY = sin(yawRad);
    vHeadTurn = sinY * headW;
    // Cohesive skull+top-hair shift from jaw/ears up through forehead and crown hair so upper hair clearly moves toward the turn
    float carriageHeight = smoothstep(1.06, -0.28, vFaceCoord.y);
    float topHairFollow = smoothstep(-0.12, -1.18, vFaceCoord.y) * (1.0 - smoothstep(1.08, 1.65, abs(hx)));
    float neckCarriageShift = sinY * (0.088 * carriageHeight + 0.026 * topHairFollow) * uFaceW * headW;
    float foreshorten = mix(1.0, cosY, 0.16 * headW);
    float rotX = r.x * foreshorten + dz * sinY * mix(0.24, 0.44, featureCohesion) + neckCarriageShift;
    r.x = rotX;
    r.y = r.y - dz * sin(pitchRad) * mix(0.12, 0.28, max(innerMask, hairDomeMask * 0.65));

    // Coherent 3D ear + surrounding lateral hair compression/visibility
    float earZone = smoothstep(0.66, 1.04, abs(vFaceCoord.x)) *
      smoothstep(-0.76, -0.30, vFaceCoord.y) *
      (1.0 - smoothstep(0.32, 0.68, vFaceCoord.y)) * headW;
    float sideHairZone = smoothstep(0.56, 0.98, abs(vFaceCoord.x)) *
      smoothstep(-1.22, -0.32, vFaceCoord.y) *
      (1.0 - smoothstep(0.42, 0.86, vFaceCoord.y)) *
      (1.0 - earZone * 0.55) * headW;
    float latTurnNearness = -sign(vFaceCoord.x) * sinY;
    float nearAmt = clamp(latTurnNearness * 2.6, 0.0, 1.0);
    float farAmt = clamp(-latTurnNearness * 2.6, 0.0, 1.0);

    float earShift = sign(vFaceCoord.x) * (nearAmt * 0.056 - farAmt * 0.052) * uFaceW * earZone;
    float sideHairShift = sign(vFaceCoord.x) * (nearAmt * 0.032 - farAmt * 0.058) * uFaceW * sideHairZone;
    float earScale = 1.0 + (nearAmt * 0.16 - farAmt * 0.28) * earZone;
    float hairLatScale = 1.0 + (nearAmt * 0.06 - farAmt * 0.18) * sideHairZone;
    vec2 earAnchor = vec2(sign(vFaceCoord.x) * uFaceW * 0.42, (uFaceCenter.y - uFaceH * 0.08) - skullPivot.y);
    r = mix(r, earAnchor + (r - earAnchor) * earScale, earZone);
    vec2 hairAnchor = vec2(sign(vFaceCoord.x) * uFaceW * 0.46, r.y);
    r.x = mix(r.x, hairAnchor.x + (r.x - hairAnchor.x) * hairLatScale, sideHairZone);

    p.x = skullPivot.x + (r.x + earShift + sideHairShift) / aspect + uHeadTranslate.x * headW;
    p.y = skullPivot.y + r.y + uHeadTranslate.y * headW;
  }

  // 6) Strictly decoupled cervical neck column (pivots at lowerNeck, zero trapezius twist) + swallow motion
  if (uPass < 1.5 && neckW > 0.001) {
    float nEff = neckW * (1.0 - headW * 0.25);
    vec2 nRel = vec2((p.x - uNeckPivot.x) * aspect, p.y - uNeckPivot.y);
    float nr = uNeckRoll * nEff;
    float cN = cos(nr);
    float sN = sin(nr);
    vec2 nRot = vec2(nRel.x * cN - nRel.y * sN, nRel.x * sN + nRel.y * cN);
    nRot.x += clamp(uNeckRot.x, -0.65, 0.65) * 0.14 * uFaceW * nEff;
    nRot.y += clamp(uNeckRot.y, -0.65, 0.65) * 0.08 * uFaceH * nEff;

    float swPhase = clamp(uSwallowPhase, 0.0, 1.0);
    float swWave = swPhase > 0.005 ? sin(swPhase * 3.14159265) : 0.0;
    float hyoidCenterY = uAdamAppleCoord.y - swWave * 0.22;
    float hyoidY = exp(-pow((vFaceCoord.y - hyoidCenterY) / 0.22, 2.0));
    float hyoidX = 1.0 - smoothstep(0.12, 0.38, abs(vFaceCoord.x - uAdamAppleCoord.x));
    float hyoidBand = hyoidY * hyoidX * neckW;
    float swAmt = clamp(uSwallow, 0.0, 1.6);
    float swallowLift = swAmt * hyoidBand * 0.022 * uFaceH;
    float swallowSwell = swAmt * swWave * hyoidBand * sign(vFaceCoord.x - uAdamAppleCoord.x + 1e-4) * 0.005 * uFaceW;

    p.x = uNeckPivot.x + (nRot.x + swallowSwell) / aspect;
    p.y = uNeckPivot.y + nRot.y - swallowLift;
  }

  // 7) Cohesive trapezius + shoulders moving independently below neckWallEndY without pinching clavicles
  if (uPass < 1.5) {
    float yokeW = clamp(torsoW * (1.0 - headW) * (1.0 - neckW), 0.0, 1.0);
    if (yokeW > 0.001) {
      float relShX = (p.x - uNeckPivot.x) * aspect;
      float shoulderOuter = smoothstep(uFaceW * 0.28, uFaceW * 0.62, abs(relShX));
      float sideSign = relShX < 0.0 ? -1.0 : 1.0;
      float torsoYawShiftX = clamp(uTorsoYaw, -0.5, 0.5) * 0.012 * uFaceW * yokeW;
      float shrugLift = clamp(uShoulderShrug, -0.2, 1.0) * shoulderOuter * 0.018 * uFaceH * yokeW;
      float tiltDelta = clamp(uShoulderTilt, -0.8, 0.8) * sideSign * shoulderOuter * 0.015 * uFaceH * yokeW;
      p.x += (uTorsoSwayX * 0.007 * uFaceW * yokeW + torsoYawShiftX) / aspect;
      p.y += -shrugLift + tiltDelta - uTorsoBreath * 0.005 * uFaceH * yokeW;
    }
  }

  // Camera approach perspective: subtly enlarge upper head/forehead & gently taper lower face/jaw
  if (headW > 0.001) {
    float leanVal = clamp(uForwardLean, -0.45, 0.95);
    float approachAmt = max(0.0, leanVal);
    float retreatAmt = max(0.0, -leanVal);
    float vY = clamp(vFaceCoord.y, -1.35, 1.25);
    float upperHeadMask = smoothstep(0.12, -0.95, vY);
    float lowerFaceMask = smoothstep(0.12, 0.95, vY);
    float zoomX = 1.0 + (approachAmt * (0.018 + 0.040 * upperHeadMask - 0.044 * lowerFaceMask) - retreatAmt * 0.018) * headW;
    float zoomY = 1.0 + (approachAmt * (0.016 + 0.036 * upperHeadMask - 0.028 * lowerFaceMask) - retreatAmt * 0.018) * headW;
    vec2 leanAnchor = mix(uMouthCenter, uFaceCenter, 0.62);
    vec2 relLean = vec2((p.x - leanAnchor.x) * aspect * zoomX, (p.y - leanAnchor.y) * zoomY);
    p.x = leanAnchor.x + relLean.x / aspect;
    p.y = leanAnchor.y + relLean.y + (approachAmt * (0.004 - 0.010 * lowerFaceMask) - retreatAmt * 0.003) * headW * uFaceH;
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
uniform float uPucker;
uniform float uLipPress;
uniform float uUpperLipLift;
uniform float uLowerLipDrop;
uniform float uLipBite;
uniform float uSmile;
uniform float uSmileR;
uniform float uSmileL;
uniform float uSmileTeeth;
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
uniform float uSwallowPhase;
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
uniform vec2 uEyeBrowFloor;

in vec2 vUv;
in float vEdgeFade;
in float vMouthLipT;
in float vMouthX;
in vec2 vMouthPlane;
in float vJawDropMouthW;
in float vJawBoneDropMouthW;
in vec2 vRigidMouth;
in vec2 vFaceCoord;
in float vHeadTurn;

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

vec3 sampleSoftTeeth(vec2 uv) {
  vec2 d = vec2(0.0022, 0.0038);
  vec3 c = texture(uTeethTex, uv).rgb * 0.44;
  c += texture(uTeethTex, clamp(uv + vec2( d.x, 0.0), 0.002, 0.998)).rgb * 0.14;
  c += texture(uTeethTex, clamp(uv + vec2(-d.x, 0.0), 0.002, 0.998)).rgb * 0.14;
  c += texture(uTeethTex, clamp(uv + vec2(0.0,  d.y), 0.002, 0.998)).rgb * 0.14;
  c += texture(uTeethTex, clamp(uv + vec2(0.0, -d.y), 0.002, 0.998)).rgb * 0.14;
  return c;
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

// Photorealistic static-photo eye & eyelid animation:
// - 100% original photo around the eye when open (ZERO cutout halo, ZERO runtime luminance/color swapping)
// - Horizontal-only gaze (left <-> right) keeping clean surrounding sclera with zero duplicate cornea
// - Eye-widening smoothly stretches upper lid + clean upper eye aperture seamlessly without seams or extra corneas
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
  float browFloorEy,
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

  bool isOuterSide = (ex * outerSign) > 0.0;
  float maxReach = isOuterSide ? outerReach : 1.06;
  float exNorm = abs(ex) / maxReach;
  float cornerEnv = 1.0 - smoothstep(0.84, 1.0, exNorm);
  if (cornerEnv < 0.002) return baseRgb;

  // Slightly raised tarsal resting baseline so default lid isn't glued down onto dark eyelashes
  float effectiveTopBase = rawTopY - 0.05;
  float effectiveBotBase = rawBotY;

  float effectiveBlink = max(eyeBlink, clamp(uLidDroop, 0.0, 0.85) * 0.68);
  float blinkClose = clamp(effectiveBlink - uEyeWide * 0.06, 0.0, 1.0);
  float squintClose = clamp(uSquint * 0.22, 0.0, 0.32);
  float lidAction = max(blinkClose, squintClose);

  float paraCurve = 1.0 - min(1.0, exNorm * exNorm);
  float wideLift = clamp(uEyeWide, 0.0, 1.0) * 0.28 * (1.0 - blinkClose) * (1.0 - clamp(uSquint, 0.0, 1.0) * 0.70) * (0.38 + 0.62 * paraCurve);
  float meetArch = 0.056 * paraCurve * blinkClose;
  float outerCloseBias = smoothstep(0.28, outerReach * 0.86, ex * outerSign);
  float meet = mix(effectiveTopBase, effectiveBotBase, mix(0.86, 1.0, outerCloseBias)) + meetArch;
  float topY = mix(effectiveTopBase, meet, blinkClose) - wideLift;
  float squintBot = mix(effectiveBotBase, mix(effectiveBotBase, effectiveTopBase, 0.22), squintClose);
  float botY = mix(squintBot + wideLift * 0.12 * (1.0 - blinkClose), meet + 0.006, blinkClose);

  float feather = max(0.020, fwidth(ey) * 1.25);
  float rawAperture = cornerEnv * smoothstep(effectiveTopBase - feather, effectiveTopBase + feather, ey) *
    (1.0 - smoothstep(effectiveBotBase - feather, effectiveBotBase + feather, ey));
  float aperture = cornerEnv * smoothstep(topY - feather, topY + feather, ey) *
    (1.0 - smoothstep(botY - feather, botY + feather, ey));

  // Raised tarsal skin ceiling well above dark lash line (bounded safely below rest browFloorEy)
  float archEnv = 1.0 - 0.05 * min(1.0, exNorm * exNorm);
  float safeBrowCeil = min(rawTopY - 0.36, browFloorEy + 0.02);
  float upperCrease = max(safeBrowCeil, rawTopY - 0.82 * archEnv);
  float lowerCrease = effectiveBotBase + 0.64 * archEnv;

  float lidTopFeather = smoothstep(upperCrease - 0.04, upperCrease + 0.14, ey);
  float lidBotFeather = 1.0 - smoothstep(lowerCrease - 0.14, lowerCrease + 0.04, ey);
  float lidZoneMask = cornerEnv * lidTopFeather * lidBotFeather;
  float lidMovingMask = max(
    smoothstep(0.003, 0.028, lidAction),
    smoothstep(0.006, 0.055, wideLift)
  );

  // When open and resting, ONLY pixels inside the open eye aperture can ever change; surrounding eyelids stay 100% baseRgb
  float effectMask = max(max(rawAperture, aperture), lidZoneMask * lidMovingMask);
  if (effectMask < 0.002) return baseRgb;

  // Stretch strictly clean tarsal skin well above the dark lash margin so moving/widening lids stay smooth
  float sourceY = ey;
  float upperEdgeBlend = 0.0;
  if (lidMovingMask > 0.002) {
    if (ey <= topY) {
      float tUp = clamp((ey - upperCrease) / max(0.04, topY - upperCrease), 0.0, 1.0);
      float skinSplit = clamp(tUp / 0.90, 0.0, 1.0);
      upperEdgeBlend = smoothstep(0.86, 1.0, tUp) * smoothstep(0.02, 0.25, lidAction);
      float targetUpperSkin = mix(rawTopY - 0.30, rawTopY - 0.18, upperEdgeBlend);
      sourceY = mix(upperCrease, targetUpperSkin, skinSplit);
    } else if (ey >= botY) {
      float tDn = clamp((ey - botY) / max(0.04, lowerCrease - botY), 0.0, 1.0);
      sourceY = mix(effectiveBotBase + 0.06, lowerCrease, tDn);
    } else {
      float tAp = clamp((ey - topY) / max(0.02, botY - topY), 0.0, 1.0);
      sourceY = mix(rawTopY - 0.24, effectiveBotBase + 0.06, tAp);
    }
  }
  vec2 sourceDelta = eyeAxis.xy * (ex * halfW) + eyeAxis.zw * (sourceY * scaleY);
  vec2 sourceUv = eyeCenter + vec2(sourceDelta.x / aspect, sourceDelta.y);
  vec3 deformedLid = texture(uPhotoTex, clamp(sourceUv, 0.001, 0.999)).rgb;
  if (upperEdgeBlend > 0.01) {
    deformedLid = mix(deformedLid, uEyelidColor * 0.90, upperEdgeBlend * 0.28);
  }

  // Horizontal-only gaze (left <-> right, shiftEy = 0) + seamless upper stretch when eye opens wider
  vec2 dPupil = vec2((eyePupil.x - eyeCenter.x) * aspect, eyePupil.y - eyeCenter.y);
  float pupilEx = dot(dPupil, eyeAxis.xy) / halfW;
  float pupilEy = dot(dPupil, eyeAxis.zw) / scaleY;
  float baseIrisR = max(0.002, irisRadNorm);

  float maxYawClamp = mix(0.46, 0.40, clamp(abs(vHeadTurn) * 2.4, 0.0, 1.0));
  float rawShiftEx = clamp(uGaze.x * 0.42, -maxYawClamp, maxYawClamp);
  float shiftEx = clamp(pupilEx + rawShiftEx, -0.58, 0.58) - pupilEx;
  float shiftEy = 0.0;
  float shiftMag = abs(shiftEx);
  float moveActive = smoothstep(0.004, 0.024, shiftMag);

  float tileXOff = outerSign < 0.0 ? 0.0 : 0.5;
  float innerCornerAnchor = smoothstep(0.68, 0.92, -ex * outerSign);
  float clearR = baseIrisR * 1.24;

  vec3 eyeCol = baseRgb;
  float wideAct = smoothstep(0.006, 0.055, wideLift) * aperture * (1.0 - smoothstep(0.55, 0.88, blinkClose));
  if (wideAct > 0.002) {
    // Seamlessly remap open upper & lower aperture coordinates strictly inside clean below-lash photo bounds (zero extra cornea or seam)
    float safeAnchorTop = min(pupilEy - 0.14, effectiveTopBase + 0.13);
    float safeAnchorBot = max(pupilEy + 0.14, effectiveBotBase - 0.04);
    float tOpen = clamp((ey - topY) / max(0.04, botY - topY), 0.0, 1.0);
    float mappedEy = mix(safeAnchorTop, safeAnchorBot, tOpen);
    vec2 stretchDelta = eyeAxis.xy * (ex * halfW) + eyeAxis.zw * (mappedEy * scaleY);
    vec2 stretchUv = eyeCenter + vec2(stretchDelta.x / aspect, stretchDelta.y);
    vec3 stretchedEye = texture(uPhotoTex, clamp(stretchUv, 0.001, 0.999)).rgb;
    float upperLidShade = mix(0.92, 1.0, smoothstep(topY, topY + 0.18, ey));
    eyeCol = mix(eyeCol, stretchedEye * upperLidShade, wideAct);
  }

  if (moveActive > 0.002) {
    vec2 X = vec2(ex * halfW, ey * scaleY);
    vec2 C0 = vec2(pupilEx * halfW, pupilEy * scaleY);
    vec2 shiftVec = vec2(shiftEx * halfW, shiftEy * scaleY);
    vec2 C1 = C0 + shiftVec;
    float dOld = length(X - C0);
    float dNew = length(X - C1);

    // Horizontal gaze preserves photo sclera outside the vacated circle and fills ONLY vacated resting circle from true neighbor sclera
    vec2 dirFromC1 = dNew > 1e-5 ? (X - C1) / dNew : vec2(1.0, 0.0);
    vec2 pRestLimbus = C0 + dirFromC1 * clearR;
    float span = max(baseIrisR * 1.25, halfW * 0.58);
    float t1 = clamp((dNew - baseIrisR * 0.94) / span, 0.0, 1.0);
    float t2 = clamp((dOld - clearR) / span, 0.0, 1.0);
    float s = smoothstep(0.0, 1.0, min(t1, t2));

    if (s < 0.999 && dNew >= baseIrisR * 0.84) {
      vec2 pSrc = mix(pRestLimbus, X, s);
      float dRest = length(pSrc - C0);
      float remapW = moveActive * (1.0 - s) * (1.0 - innerCornerAnchor * 0.94);

      float pSrcEx = clamp(pSrc.x / halfW, -0.94, 0.94);
      float lidTopAtSrc = evalLidCurveOuter(pSrcEx, upperA, upperB, outerSign, outerReach, outerWingY, -0.05);
      float lidBotAtSrc = evalLidCurveOuter(pSrcEx, lowerA, lowerB, outerSign, outerReach, outerWingY, 0.05);
      float safeMinEy = lidTopAtSrc + 0.14;
      float safeMaxEy = max(safeMinEy + 0.02, lidBotAtSrc - 0.11);
      float pSrcEy = clamp(pSrc.y / scaleY, safeMinEy, safeMaxEy);
      vec2 scleraDelta = eyeAxis.xy * (pSrcEx * halfW) + eyeAxis.zw * (pSrcEy * scaleY);
      vec3 directSclera = texture(uPhotoTex, clamp(eyeCenter + vec2(scleraDelta.x / aspect, scleraDelta.y), 0.001, 0.999)).rgb;

      float patchU = tileXOff + clamp((clamp(pSrcEx, -1.85, 1.85) / 1.9) * 0.5 + 0.5, 0.005, 0.995) * 0.5;
      float patchV = 0.5 + clamp((clamp(pSrcEy, -2.85, 2.85) / 2.9) * 0.5 + 0.5, 0.005, 0.995) * 0.5;
      vec3 atlasSclera = texture(uEyeAtlasTex, vec2(patchU, patchV)).rgb;

      float bridgeWeight = 1.0 - smoothstep(clearR * 0.78, clearR * 1.08, dRest);
      vec3 chosenSclera = mix(directSclera, atlasSclera, bridgeWeight * 0.78);
      eyeCol = mix(eyeCol, chosenSclera, remapW);
    }

    // Extract and composite crisp photographic cornea at horizontal gaze target
    vec2 dirShift = length(shiftVec) > 1e-5 ? normalize(shiftVec) : vec2(1.0, 0.0);
    vec2 relM = X - C1;
    float axialProj = dot(relM, dirShift);
    float gazeStretch = 1.0 + clamp(shiftMag, 0.0, 1.2) * 0.06;
    vec2 relStretched = relM - dirShift * (axialProj * (1.0 - 1.0 / gazeStretch));
    float dStretched = length(relStretched);
    float dilateFactor = clamp(uPupilDilate, 0.0, 1.0) * 0.22;
    float radialFrac = clamp(dStretched / baseIrisR, 0.0, 1.4);
    float radialScale = radialFrac < 0.44 ? mix(1.0 - dilateFactor, 1.0, smoothstep(0.22, 0.44, radialFrac)) : 1.0;
    vec2 sampleOffset = relStretched * radialScale;
    float sampleEx = pupilEx + sampleOffset.x / halfW;
    float sampleEy = pupilEy + sampleOffset.y / scaleY;

    float corneaU = tileXOff + clamp((sampleEx / 1.9) * 0.5 + 0.5, 0.005, 0.995) * 0.5;
    float corneaV = clamp((sampleEy / 2.9) * 0.5 + 0.5, 0.005, 0.995) * 0.5;
    vec4 corneaSamp = texture(uEyeAtlasTex, vec2(corneaU, corneaV));

    float lidCastAO = mix(0.90, 1.0, smoothstep(topY, topY + 0.22, ey));
    vec3 litCornea = clamp(corneaSamp.rgb * lidCastAO, 0.0, 1.0);
    float corneaAlpha = corneaSamp.a * moveActive * (1.0 - innerCornerAnchor * 0.94);
    eyeCol = mix(eyeCol, litCornea, corneaAlpha);
  }

  float distMoved = length(vec2((ex - shiftEx - pupilEx) * halfW, (ey - shiftEy - pupilEy) * scaleY));
  float eraseR = baseIrisR * 1.14;

  float redAmt = clamp(uEyeRedness, 0.0, 1.0);
  if (redAmt > 0.02) {
    float irisArea = 1.0 - smoothstep(baseIrisR * 0.92, eraseR * 1.04, distMoved);
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
  vec3 result = mix(deformedLid, eyeCol, openInside);

  return mix(baseRgb, result, effectMask);
}

// Subtle dynamic facial creases (zero dark nose-contour outlines when angry or furrowing brows)
vec3 applyExpressionWrinkles(vec3 col, vec2 fc, vec2 rm) {
  float absX = abs(fc.x);
  float skinMask = (1.0 - smoothstep(0.84, 1.08, absX)) * (1.0 - smoothstep(0.88, 1.10, abs(fc.y)));
  if (skinMask < 0.01) return col;

  float luma = dot(col, vec3(0.299, 0.587, 0.114));
  float skinToneMask = smoothstep(0.14, 0.30, luma);
  if (skinToneMask < 0.01) return col;

  // 1) Nasolabial smile folds (strictly smile/cheek-lift driven, never outlining nose on anger)
  float sideSmile = fc.x < 0.0 ? max(0.0, uSmileR) : max(0.0, uSmileL);
  float foldWeight = clamp(sideSmile * 0.62 + uCheekLift * 0.46, 0.0, 1.0);
  float nasoCurveX = 0.26 + smoothstep(0.04, 0.54, fc.y) * 0.34;
  float nasoVert = smoothstep(0.02, 0.18, fc.y) * (1.0 - smoothstep(0.46, 0.64, fc.y));
  float nasoDist = abs(absX - nasoCurveX);
  float nasoShadow = exp(-pow(nasoDist / 0.048, 2.0)) * nasoVert * foldWeight;

  // 2) Subtle crow's-feet outer-eye smile crinkles
  float squintAct = smoothstep(0.16, 0.85, max(0.0, uSquint + uCheekLift * 0.36 + sideSmile * 0.28));
  float crowZone = smoothstep(0.52, 0.70, absX) * (1.0 - smoothstep(0.90, 1.04, absX)) *
    exp(-pow((fc.y + 0.16) / 0.14, 2.0));
  float crowRays = 0.5 + 0.5 * cos((fc.y + 0.16) * 44.0 - (absX - 0.64) * 16.0);
  float crowShadow = crowZone * crowRays * squintAct * 0.55;

  float underEyeCrease = exp(-pow((absX - 0.36) / 0.18, 2.0)) *
    exp(-pow((fc.y + 0.05) / 0.042, 2.0)) * (sideSmile * 0.35 + uCheekLift * 0.40);

  // 3) Soft central glabella furrow between inner brows only (no nose-bridge or side-nose outlines)
  float furrowAct = smoothstep(0.08, 0.78, uInnerBrowPinch);
  float glabellaZone = exp(-pow((absX - 0.062) / 0.028, 2.0)) *
    smoothstep(-0.56, -0.44, fc.y) * (1.0 - smoothstep(-0.28, -0.18, fc.y));
  float glabellaShadow = glabellaZone * furrowAct * 0.85;

  // 4) Soft forehead expression lines on high brow raise
  float browHigh = smoothstep(0.24, 0.88, (uInnerBrowY + uOuterBrowY) * 0.5);
  float foreheadZone = (1.0 - smoothstep(0.28, 0.66, absX)) *
    smoothstep(-0.88, -0.74, fc.y) * (1.0 - smoothstep(-0.52, -0.42, fc.y));
  float foreheadBands = 0.5 + 0.5 * cos((fc.y + 0.66) * 42.0);
  float foreheadShadow = foreheadZone * foreheadBands * browHigh * 0.52;

  // 5) Subtle mouth corner dimple shading
  float restSlitY = (uRestMouthTopY + uRestMouthBotY) * 0.5;
  float cornerDimple = exp(-pow((abs(rm.x) - 0.96) / 0.12, 2.0)) *
    exp(-pow((rm.y - restSlitY) / 0.075, 2.0)) * (0.16 + uSmile * 0.28 + uLipPress * 0.24);

  float totalCrease = clamp(
    (nasoShadow * 0.075 + crowShadow * 0.075 + underEyeCrease * 0.06 + glabellaShadow * 0.11 +
     foreheadShadow * 0.10 + cornerDimple * 0.08) * skinMask * skinToneMask,
    0.0,
    0.22
  );
  vec3 shaded = col * mix(vec3(1.0), vec3(0.68, 0.56, 0.52), totalCrease);
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
  vec3 neutralWhite = mix(rawTex, vec3(lum * 1.015, lum * 1.005, lum * 0.985), 0.68);
  vec3 portraitLight = mix(vec3(1.0, 0.99, 0.97), clamp(uEnamelColor * 1.32, 0.68, 1.04), 0.34);
  float enamelWeight = smoothstep(0.24, 0.72, lum);
  vec3 scaleCol = neutralWhite * portraitLight * mix(0.86, brighten, enamelWeight);
  return clamp(scaleCol, vec3(0.0), vec3(0.96));
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
    // Watertight photorealistic oral cavity: closed/parted smile modes, portrait-matched dental enamel & wide jaw gap
    float closedBite = uSmileTeeth * smoothstep(0.32, 0.76, uSmile) * (1.0 - smoothstep(0.06, 0.18, uOpen)) * (1.0 - smoothstep(0.18, 0.45, uPucker + uLipPress));
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
    float cornerCurve = 0.03 * pow(absX, 1.8);
    float rimAA = max(0.004, fwidth(vMouthLipT) * 0.75);
    float topRim = smoothstep(cornerCurve, cornerCurve + rimAA, vMouthLipT);
    // Crisp inner lower-lip cutoff + labial sulcus stop so teeth/gums NEVER mix blurrily into the lower lip
    float botCut = 0.992 - cornerCurve * 0.25;
    float botRim = smoothstep(botCut, botCut - rimAA, vMouthLipT);
    float sideAlmondFade = 1.0 - smoothstep(0.92, 0.998, absX);
    float lipRimFeather = topRim * botRim * sideAlmondFade;

    // Skull-space spatial mouth coordinates: natural upward-curving outward smile arch for both upper and lower dental arches + visible gums
    float x = vMouthPlane.x;
    float y = vMouthPlane.y;
    float jawY = y - vJawBoneDropMouthW * 0.84;
    float ax = clamp(x - uYaw * 0.070, -0.56, 0.56);
    float absAx = abs(ax);
    float archCurve = min(1.0, absAx / 0.54);
    float toothTexU = clamp(0.5 + ax * 0.86, 0.025, 0.975);
    float uCurve = pow(abs(toothTexU - 0.5) / 0.475, 1.75);

    // 1) Deep very-dark-red flesh interior cavity
    vec2 cavUv = vec2(clamp(0.5 + ax * 0.52, 0.20, 0.80), mix(0.36, 0.50, clamp(vMouthLipT, 0.0, 1.0)));
    vec3 rawCav = sampleSoftTeeth(cavUv);
    float cavDepth = 1.0 - 0.56 * (1.0 - abs(vMouthLipT - 0.44) * 1.25) * (1.0 - absX * 0.65);
    vec3 deepRedFlesh = vec3(0.048, 0.005, 0.007) + vec3(rawCav.r * 0.088, rawCav.g * 0.012, rawCav.b * 0.014);
    vec3 oralCol = clamp(deepRedFlesh * clamp(cavDepth, 0.28, 1.0), vec3(0.018, 0.002, 0.003), vec3(0.145, 0.022, 0.028));

    // 2) Upper dental arch curving outward & higher at corners, showing natural soft-focus enamel + pink scalloped upper gums
    float pitchPersp = -uHeadRot.y * 0.007;
    float midRestY = (uRestMouthTopY + uRestMouthBotY) * 0.5;
    float upperSmileArch = 0.084 * ax * ax;
    float upperEdgeOpen = uRestMouthTopY + 0.046 + clamp(uUpperLipLift, 0.0, 1.0) * 0.014 - upperSmileArch + pitchPersp;
    float upperEdgeSmile = midRestY + 0.008 - 0.092 * ax * ax + pitchPersp;
    float upperEdgeY = mix(upperEdgeOpen, upperEdgeSmile, closedBite);
    float upperCrownH = mix(0.084, 0.062, archCurve);
    float gumScallop = sin(toothTexU * 26.0) * 0.004;
    float upperGumY = upperEdgeY - upperCrownH + gumScallop;
    float upperFrac = clamp((y - upperGumY) / max(0.020, upperCrownH), 0.0, 1.0);
    float upperVTop = mix(0.128, 0.195, uCurve);
    float upperVBot = mix(0.292, 0.328, uCurve);
    float upperTexV = (y < upperGumY)
      ? mix(0.055, upperVTop, clamp(1.0 - (upperGumY - y) / 0.048, 0.0, 1.0))
      : mix(upperVTop, upperVBot, upperFrac);

    vec3 rawUpper = sampleSoftTeeth(vec2(toothTexU, upperTexV));
    vec3 whiteUpper = whitenToothEnamel(rawUpper, 1.11);
    float archDepthShade = 1.0 - 0.34 * pow(archCurve, 1.55);
    float upperLipShadow = mix(0.78, 1.0, smoothstep(-0.028, 0.022, y - upperGumY));
    float wetEnamelSheen = exp(-pow((upperFrac - 0.56) / 0.16, 2.0)) * (1.0 - archCurve * 0.85) * 0.032;
    float upperGumMask = y < upperGumY + 0.003 ? smoothstep(upperGumY + 0.005, upperGumY - 0.014, y) : 0.0;
    vec3 naturalUpperGum = clamp(
      mix(rawUpper, vec3(rawUpper.r * 1.04, rawUpper.g * 0.84, rawUpper.b * 0.84), 0.30),
      vec3(0.32, 0.14, 0.16),
      vec3(0.84, 0.48, 0.50)
    );
    vec3 upperToothCol = clamp((mix(whiteUpper, naturalUpperGum, upperGumMask) + vec3(wetEnamelSheen) * (1.0 - upperGumMask)) * archDepthShade * upperLipShadow, 0.0, 1.0);

    // 3) Lower dental arch curving cleanly upward toward corners + natural pink lower gums well above the lower lip border
    float lowerTexU = clamp(0.5 + ax * 0.84, 0.03, 0.97);
    float lowerUCurve = pow(abs(lowerTexU - 0.5) / 0.47, 1.75);
    float lowerSmileArch = 0.072 * ax * ax;
    float lowerRestTipY = uRestMouthBotY - 0.034 - lowerSmileArch;
    float lowerSmileTipY = upperEdgeY + 0.012;
    float lowerTipLocalY = mix(lowerRestTipY, lowerSmileTipY, closedBite);
    float lowerCrownH = mix(0.072, 0.056, archCurve);
    float lowerGumLocalY = lowerTipLocalY + lowerCrownH - gumScallop * 0.85;
    float lowerFrac = clamp((jawY - lowerTipLocalY) / max(0.018, lowerCrownH), 0.0, 1.0);
    float lowerVTip = mix(0.752, 0.665, lowerUCurve);
    float lowerVGum = mix(0.896, 0.845, lowerUCurve);
    float lowerTexV = (jawY > lowerGumLocalY)
      ? mix(lowerVGum, 0.955, clamp((jawY - lowerGumLocalY) / 0.044, 0.0, 1.0))
      : mix(lowerVTip, lowerVGum, lowerFrac);

    vec3 rawLower = sampleSoftTeeth(vec2(lowerTexU, lowerTexV));
    vec3 whiteLower = whitenToothEnamel(rawLower, 1.07);
    float lowerLipShadow = mix(1.0, 0.82, smoothstep(0.0, 0.032, jawY - lowerGumLocalY));
    float lowerGumMask = jawY > lowerGumLocalY - 0.003 ? smoothstep(lowerGumLocalY - 0.005, lowerGumLocalY + 0.014, jawY) : 0.0;
    vec3 naturalLowerGum = clamp(
      mix(rawLower, vec3(rawLower.r * 1.02, rawLower.g * 0.82, rawLower.b * 0.82), 0.30),
      vec3(0.30, 0.13, 0.15),
      vec3(0.80, 0.46, 0.48)
    );
    vec3 lowerToothCol = clamp(mix(whiteLower, naturalLowerGum, lowerGumMask) * archDepthShade * lowerLipShadow, 0.0, 1.0);

    // 4) Perspective-tracked real photographic tongue seated behind lower teeth
    float tongueAdv = clamp(uTongueOut, 0.0, 1.0);
    float speechTongueWave = activeOpen > 0.05 ? sin(uTime * 3.8 + uRound * 2.0) * 0.005 : 0.0;
    float openTongueRise = smoothstep(0.10, 0.72, activeOpen) * 0.062;
    float tongueLift = clamp(uRound, 0.0, 1.0) * 0.008 + tongueAdv * 0.036 + speechTongueWave;
    float tonguePerspX = x - uYaw * 0.15;
    float tongueY = jawY + uHeadRot.y * 0.010;
    float tongueTopY = lowerTipLocalY - 0.022 - openTongueRise + 0.11 * tonguePerspX * tonguePerspX - tongueLift;
    float tongueBotY = lowerGumLocalY + 0.052;
    float tongueLatMask = 1.0 - smoothstep(0.26 + tongueAdv * 0.06, 0.46 + tongueAdv * 0.08, abs(tonguePerspX));
    float tongueVertMask =
      smoothstep(tongueTopY - 0.012, tongueTopY + 0.010, tongueY) *
      (1.0 - smoothstep(tongueBotY - 0.010, tongueBotY + 0.012, tongueY));
    float tongueShow = max(smoothstep(0.08, 0.24, activeOpen), smoothstep(0.03, 0.16, tongueAdv));
    float tongueMask = tongueLatMask * tongueVertMask * tongueShow * (1.0 - closedBite * 0.92);
    float tNormY = clamp((tongueY - tongueTopY) / max(0.020, tongueBotY - tongueTopY), 0.0, 1.0);
    vec2 tongueUv = vec2(
      clamp(0.5 + tonguePerspX * 0.56, 0.24, 0.76),
      mix(0.535, 0.725, tNormY)
    );
    vec3 rawTongue = sampleSoftTeeth(tongueUv);
    float wideOpenLight = 1.0 + smoothstep(0.28, 0.85, activeOpen) * 0.22;
    float tongueDepthShade = mix(0.54, 1.04, smoothstep(0.0, 0.48, tNormY)) * (1.0 - 0.20 * smoothstep(0.20, 0.46, abs(tonguePerspX))) * wideOpenLight;
    vec3 tongueCol = clamp(
      mix(rawTongue, vec3(rawTongue.r * 1.12, rawTongue.g * 0.84, rawTongue.b * 0.84), 0.35) * tongueDepthShade,
      vec3(0.06, 0.014, 0.018),
      vec3(0.86, 0.50, 0.52)
    );

    float yFeather = max(0.009, fwidth(y) * 2.1);
    float upperZone =
      (1.0 - smoothstep(upperEdgeY - yFeather, upperEdgeY + yFeather, y)) *
      (1.0 - smoothstep(0.47, 0.56, absAx));

    float lowerShowGate = max(
      max(smoothstep(0.035, 0.14, activeOpen), smoothstep(0.030, 0.10, uLowerLipDrop)),
      closedBite * 0.94
    );
    // Clean dark oral gap separating upper & lower teeth AND clean vestibule stop before the lower lip edge
    float clearBelowUpper = smoothstep(upperEdgeY + 0.005, upperEdgeY + 0.020, y);
    float lowerLipClear = 1.0 - smoothstep(0.82, 0.94, vMouthLipT);
    float lowerZone =
      smoothstep(lowerTipLocalY - yFeather, lowerTipLocalY + yFeather, jawY) *
      clearBelowUpper *
      lowerLipClear *
      lowerShowGate *
      (1.0 - smoothstep(0.45, 0.54, absAx));

    oralCol = mix(oralCol, tongueCol, tongueMask * (1.0 - tongueAdv * 0.12));
    oralCol = mix(oralCol, upperToothCol, upperZone * 0.98);
    oralCol = mix(oralCol, lowerToothCol, lowerZone * 0.97 * (1.0 - tongueAdv * 0.75) * (1.0 - clamp(uLipBite, 0.0, 1.0) * 0.85));
    if (tongueAdv > 0.06) {
      oralCol = mix(oralCol, tongueCol, tongueMask * smoothstep(0.06, 0.38, tongueAdv) * 0.94);
    }

    // Soft occlusal shadow line when teeth meet on a closed smile so upper & lower rows remain distinctly separated
    if (closedBite > 0.08) {
      float biteGap = exp(-pow((y - (upperEdgeY + 0.005)) / 0.0038, 2.0));
      oralCol = mix(oralCol, vec3(0.025, 0.004, 0.006), biteGap * closedBite * 0.82 * (1.0 - smoothstep(0.44, 0.54, absAx)));
    }

    if (uLipBite > 0.05) {
      float biteBand = smoothstep(0.54, 0.92, vMouthLipT) * (1.0 - smoothstep(0.28, 0.56, absX));
      vec3 tuckedLipCol = mix(uLipColor * 0.84, upperToothCol, smoothstep(0.54, 0.70, vMouthLipT) * 0.45);
      oralCol = mix(oralCol, tuckedLipCol, biteBand * clamp(uLipBite, 0.0, 1.0) * 0.82);
    }

    // Crisp inner-lip mucosal contact shadow along top and bottom borders (zero blurry bleeding into lower lip)
    float topAO = mix(0.74, 1.0, smoothstep(0.0, 0.08, vMouthLipT));
    float botAO = mix(0.56, 1.0, smoothstep(botCut, botCut - 0.11, vMouthLipT));
    float sideAO = 1.0 - 0.42 * smoothstep(0.72, 0.98, absX);
    float cavityLight = mix(0.86, 1.0, max(smoothstep(0.02, 0.22, activeOpen), closedBite));
    oralCol *= clamp(topAO * botAO * sideAO * cavityLight, 0.22, 1.0);

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
    uEyeBrowFloor.x,
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
    uEyeBrowFloor.y,
    uBlinkL,
    1.0,
    uAspect
  );

  col = applyPhysicalEffects(col, vFaceCoord);

  // Subtle directional face volume modeling strictly inside face skin (NEVER darkens ears or background!)
  if (abs(vHeadTurn) > 0.01) {
    float faceVolumeZone = (1.0 - smoothstep(0.68, 0.96, abs(vFaceCoord.x))) *
      (1.0 - smoothstep(0.76, 1.02, abs(vFaceCoord.y)));
    float turnShade = clamp(vFaceCoord.x * vHeadTurn * 0.09, -0.032, 0.032) * faceVolumeZone * fgAlpha;
    col = clamp(col * (1.0 - turnShade), 0.0, 1.0);
  }

  // Traveling larynx-centered throat swallow shadow wave when swallowing saliva
  if (uSwallow > 0.01) {
    float swWave = uSwallowPhase > 0.005 ? sin(clamp(uSwallowPhase, 0.0, 1.0) * 3.14159265) : clamp(uSwallow, 0.0, 1.0);
    float applePosY = uAdamAppleCoord.y - swWave * 0.24;
    float swBand = exp(-pow((vFaceCoord.y - applePosY) / 0.14, 2.0)) *
      (1.0 - smoothstep(0.09, 0.32, abs(vFaceCoord.x - uAdamAppleCoord.x)));
    col *= 1.0 - swBand * clamp(uSwallow, 0.0, 1.6) * 0.125;
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
        const nearTopBorder = ny < crownTopY + face.faceH * 0.16;
        segArr[py * W + px] = nearTopBorder ? seg : Math.max(seg, geomEnvelope * 0.55);
        const segTight = nearTopBorder ? smoothstep(0.2, 0.48, seg) : smoothstep(0.26, 0.52, seg);
        fg = nearTopBorder
          ? Math.max(faceCore, segTight)
          : Math.max(faceCore, segTight, bodyMask * 0.75);
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
 * Builds a 2x2 Eye Atlas (`384 x 384`) with two distinct RGBA layers per eye:
 * - Top row (`y: 0..192`): Clean extracted circular cornea+iris+pupil (`alpha = 0` outside iris),
 *   reconstructing any lid-clipped upper/lower iris pixels radially/horizontally (never flipping bottom catchlights).
 * - Bottom row (`y: 192..384`): Real photo-matched surrounding-sclera inpaint filling ONLY the vacated cornea region
 *   via inverse-distance weighting from the actual exposed sclera pixels around that photo's cornea.
 */
function buildEyeCutoutAtlas(img: HTMLImageElement, face: FacePts): HTMLCanvasElement {
  const TILE_W = 192;
  const TILE_H = 192;
  const EX_SPAN = 1.9;
  const EY_SPAN = 2.9;

  const atlas = document.createElement("canvas");
  atlas.width = TILE_W * 2;
  atlas.height = TILE_H * 2;
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

    const toNormPhoto = (ex: number, ey: number): [number, number] => {
      const dx = eyeAxis.ux * (ex * halfW) + eyeAxis.vx * (ey * scaleY);
      const dy = eyeAxis.uy * (ex * halfW) + eyeAxis.vy * (ey * scaleY);
      return [eyeCenter[0] + dx / aspect, eyeCenter[1] + dy];
    };

    // 1) Collect actual exposed surrounding sclera perimeter samples without discarding shaded/warm real tones
    type ScleraPerimeterPt = {
      ex: number;
      ey: number;
      r: number;
      g: number;
      b: number;
      weight: number;
    };
    const perimSamples: ScleraPerimeterPt[] = [];

    for (let ey = -0.78; ey <= 0.78; ey += 0.045) {
      for (let ex = -0.88; ex <= 0.88; ex += 0.04) {
        const sEx = ex * outerSign;
        if (sEx < -0.68 || sEx > 0.82) continue;
        const topY = evalLidCurveJS(ex, curve.upperArch);
        const botY = evalLidCurveJS(ex, curve.lowerArch);
        const safeTop = topY + Math.max(0.12, (botY - topY) * 0.14);
        const safeBot = botY - Math.max(0.09, (botY - topY) * 0.12);
        if (ey < safeTop || ey > safeBot) continue;

        const [nx, ny] = toNormPhoto(ex, ey);
        const distP = Math.hypot((nx - eyePupil[0]) * aspect, ny - eyePupil[1]);
        if (distP < irisRad * 1.24 || distP > irisRad * 2.15) continue;

        const [r, g, b] = sampleBilinear(nx, ny);
        const lum = r * 0.299 + g * 0.587 + b * 0.114;
        if (lum < 52) continue;
        const redness = r - (g + b) * 0.5;
        if (redness > 26) continue;
        // Prioritize pixels closest to the clean lateral limbus boundary so evacuated circle preserves local sclera gradient identically
        const nearLimbusBoost = distP < irisRad * 1.54 ? 1.8 : 1.0;

        perimSamples.push({
          ex,
          ey,
          r,
          g,
          b,
          weight: nearLimbusBoost,
        });
      }
    }

    const fbMid: [number, number, number] = [
      (fallbackNeg[0] + fallbackPos[0]) * 0.5,
      (fallbackNeg[1] + fallbackPos[1]) * 0.5,
      (fallbackNeg[2] + fallbackPos[2]) * 0.5,
    ];

    const corneaImg = aCtx.createImageData(TILE_W, TILE_H);
    const scleraPatchImg = aCtx.createImageData(TILE_W, TILE_H);

    for (let ly = 0; ly < TILE_H; ly++) {
      const ey = ((ly + 0.5) / TILE_H - 0.5) * 2 * EY_SPAN;
      for (let lx = 0; lx < TILE_W; lx++) {
        const ex = ((lx + 0.5) / TILE_W - 0.5) * 2 * EX_SPAN;
        const o4 = (ly * TILE_W + lx) * 4;

        // A) Bottom-row tile: tight local sclera interpolation preserving exact photo sclera tone & shading gradient
        let pr = fbMid[0];
        let pg = fbMid[1];
        let pb = fbMid[2];
        if (perimSamples.length >= 3) {
          let sr = 0;
          let sg = 0;
          let sb = 0;
          let swSum = 0;
          for (let i = 0; i < perimSamples.length; i++) {
            const sp = perimSamples[i]!;
            const dx = (ex - sp.ex) * 1.05;
            const dy = (ey - sp.ey) * 0.82;
            const d2 = dx * dx + dy * dy + 0.018;
            const w = sp.weight / (d2 * d2 * d2);
            sr += sp.r * w;
            sg += sp.g * w;
            sb += sp.b * w;
            swSum += w;
          }
          if (swSum > 0) {
            pr = sr / swSum;
            pg = sg / swSum;
            pb = sb / swSum;
          }
        } else {
          const gradT = smoothstep(-0.6, 0.6, ex - pupilEx);
          pr = fallbackNeg[0] * (1 - gradT) + fallbackPos[0] * gradT;
          pg = fallbackNeg[1] * (1 - gradT) + fallbackPos[1] * gradT;
          pb = fallbackNeg[2] * (1 - gradT) + fallbackPos[2] * gradT;
        }
        scleraPatchImg.data[o4] = Math.round(Math.max(0, Math.min(255, pr)));
        scleraPatchImg.data[o4 + 1] = Math.round(Math.max(0, Math.min(255, pg)));
        scleraPatchImg.data[o4 + 2] = Math.round(Math.max(0, Math.min(255, pb)));
        scleraPatchImg.data[o4 + 3] = 255;

        // B) Top-row tile: strictly cropped circular cornea+iris+pupil (zero outer sclera rim captured)
        const [nx, ny] = toNormPhoto(ex, ey);
        const distP = Math.hypot((nx - eyePupil[0]) * aspect, ny - eyePupil[1]);
        const corneaAlpha = 1 - smoothstep(irisRad * 0.88, irisRad * 1.01, distP);
        if (corneaAlpha > 0.002) {
          const topY = evalLidCurveJS(ex, curve.upperArch);
          const botY = evalLidCurveJS(ex, curve.lowerArch);
          const safeTop = topY + Math.max(0.04, (botY - topY) * 0.05);
          const safeBot = botY - Math.max(0.03, (botY - topY) * 0.04);

          let ir: number;
          let ig: number;
          let ib: number;

          if (ey >= safeTop && ey <= safeBot) {
            const [ix, iy] = toNormPhoto(ex, ey);
            [ir, ig, ib] = sampleBilinear(ix, iy);
          } else {
            // Reconstruct ONLY lid-clipped edge using crisp mirrored texture from visible iris sectors (never blurry flat rings)
            const mirEx = 2 * pupilEx - ex;
            const mirTop = evalLidCurveJS(mirEx, curve.upperArch) + 0.06;
            const mirBot = evalLidCurveJS(mirEx, curve.lowerArch) - 0.04;
            const dFromCenterEx = ex - pupilEx;
            const dFromCenterEy = ey - pupilEy;
            const rotEy = pupilEy + Math.abs(dFromCenterEy) * (ey < pupilEy ? 0.55 : -0.55);
            const [rx, ry] = toNormPhoto(
              pupilEx + dFromCenterEx,
              Math.min(mirBot, Math.max(mirTop, rotEy)),
            );
            const [rr, rg, rb] = sampleBilinear(rx, ry);
            if (ey >= mirTop && ey <= mirBot) {
              const [mx, my] = toNormPhoto(mirEx, ey);
              const [mr, mg, mb] = sampleBilinear(mx, my);
              ir = rr * 0.55 + mr * 0.45;
              ig = rg * 0.55 + mg * 0.45;
              ib = rb * 0.55 + mb * 0.45;
            } else {
              ir = rr;
              ig = rg;
              ib = rb;
            }
          }

          corneaImg.data[o4] = Math.round(Math.max(0, Math.min(255, ir)));
          corneaImg.data[o4 + 1] = Math.round(Math.max(0, Math.min(255, ig)));
          corneaImg.data[o4 + 2] = Math.round(Math.max(0, Math.min(255, ib)));
          corneaImg.data[o4 + 3] = Math.round(corneaAlpha * 255);
        } else {
          corneaImg.data[o4] = 0;
          corneaImg.data[o4 + 1] = 0;
          corneaImg.data[o4 + 2] = 0;
          corneaImg.data[o4 + 3] = 0;
        }
      }
    }
    aCtx.putImageData(corneaImg, offsetX, 0);
    aCtx.putImageData(scleraPatchImg, offsetX, TILE_H);
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
  const neckFadeStartY = chin[1] + face.faceH * 0.015;
  // Strictly partition headW (face+cranium), neckW (central cervical cylinder strictly above trapezius), and torsoW (trapezius+shoulders)
  const neckWallEndY = face.neckInflectionY ?? face.lowerNeck[1];
  const neckFadeEndY = Math.min(chin[1] + face.faceH * 0.095, neckWallEndY - face.faceH * 0.04);
  const trapSpanHalf = Math.abs(face.leftTrapezius[0] - face.rightTrapezius[0]) * aspect * 0.45;
  const neckColHalfW = Math.min(
    face.faceW * 0.34,
    Math.max(face.faceW * 0.25, trapSpanHalf * 0.78),
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
      [-10, 0],
      [10, 0],
      [0, -10],
      [0, 10],
      [-7, -7],
      [7, 7],
      [-7, 7],
      [7, -7],
      [-4, 0],
      [4, 0],
      [0, -4],
      [0, 4],
    ] as const) {
      const mx = Math.max(0, Math.min(sw - 1, sx + dx));
      const my = Math.max(0, Math.min(sh - 1, sy + dy));
      best = Math.max(best, mask[my * sw + mx] ?? 0);
    }
    return best;
  };
  const hasSegMask = !!(face.segMask && face.segWidth && face.segHeight);
  const computeHeadWeight = (nx: number, ny: number): number => {
    const rx = ((nx - craniumX) * aspect) / Math.max(0.08, radX);
    const ry = (ny - craniumY) / Math.max(0.08, radY);
    const oval = hasSegMask
      ? ny <= chin[1]
        ? 1 - smoothstep(1.22, 1.72, Math.hypot(rx, ry))
        : 1 - smoothstep(1.02, 1.38, Math.hypot(rx, ry))
      : ny <= chin[1]
        ? 1 - smoothstep(1.28, 1.96, Math.hypot(rx, ry))
        : 1 - smoothstep(1.1, 1.72, Math.hypot(rx, ry));
    const latOffset = Math.abs((nx - face.center[0]) * aspect);
    const hairYMax =
      latOffset > face.faceW * 0.24 ? chin[1] + face.faceH * 0.26 : chin[1] + face.faceH * 0.05;
    const upperHairSeg = ny <= hairYMax && sampleHairSegMax(nx, ny) > 0.06 ? 1 : 0;
    const radial = Math.max(oval, upperHairSeg);
    const neckFade = ny > neckFadeStartY ? 1 - smoothstep(neckFadeStartY, neckFadeEndY, ny) : 1;
    const lateralCut =
      ny > chin[1] + face.faceH * 0.05 && latOffset > face.faceW * 0.58
        ? 1 - smoothstep(face.faceW * 0.58, face.faceW * 0.78, latOffset)
        : 1;
    return radial * neckFade * lateralCut;
  };
  const computeNeckWeight = (nx: number, ny: number): number => {
    const vertStart = smoothstep(chin[1] - face.faceH * 0.02, chin[1] + face.faceH * 0.065, ny);
    const vertEnd =
      1 - smoothstep(neckWallEndY - face.faceH * 0.075, neckWallEndY - face.faceH * 0.025, ny);
    const lat = Math.abs((nx - face.neckBase[0]) * aspect);
    const latMask = 1 - smoothstep(neckColHalfW * 0.8, neckColHalfW * 1.12, lat);
    return vertStart * vertEnd * latMask;
  };

  // Classic elliptical mandible grip centered on real chin/jaw, with soft sub-chin taper so speaking opens jaw cleanly without pulling neck
  const lipBotY = point(17)[1];
  const jawCenterY = (lipBotY + chin[1]) * 0.5;
  const jawRx = Math.max(0.03, face.faceW * 0.42);
  const jawRy = Math.max(face.faceH * 0.19, (chin[1] - mouthCenter[1]) * 1.18);
  const adamLimitY = Math.min(
    neckWallEndY - face.faceH * 0.02,
    face.adamApple[1] - face.faceH * 0.02,
  );
  const computeMandibleWeight = (nx: number, ny: number, mpY?: number): number => {
    const eR = Math.hypot(((nx - mouthCenter[0]) * aspect) / jawRx, (ny - jawCenterY) / jawRy);
    const ovalBell = 1 - smoothstep(0.26, 1.08, eR);
    const vertUp =
      mpY !== undefined
        ? smoothstep(-face.mouthH * 0.1, face.mouthH * 0.35, mpY)
        : smoothstep(mouthCenter[1] - 0.01 * face.faceH, lipBotY, ny);
    const chinStopY = Math.min(adamLimitY, chin[1] + face.faceH * 0.105);
    const subChinFade =
      ny <= chin[1] + face.faceH * 0.022
        ? 1
        : 1 - smoothstep(chin[1] + face.faceH * 0.022, chinStopY, ny);
    return ovalBell * vertUp * subChinFade;
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
    const borderY = smoothstep(0.0, 0.015, ny) * smoothstep(1.0, 0.985, ny);
    for (let ix = 0; ix <= GX; ix++) {
      const nx = ix / GX;
      const borderX = smoothstep(0.0, 0.015, nx) * smoothstep(1.0, 0.985, nx);
      const borderMask = borderX * borderY;

      const headW = computeHeadWeight(nx, ny) * borderMask;
      const neckW = computeNeckWeight(nx, ny) * borderMask;
      const underChinJawW = computeMandibleWeight(nx, ny) * borderMask;
      // Cohesive trapezius & shoulder plateau starting strictly below the cervical neck wall
      const torsoW =
        smoothstep(neckWallEndY - 0.04 * face.faceH, neckWallEndY + 0.02 * face.faceH, ny) *
        (1 - neckW) *
        borderMask;

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
      baseVerts[ptr++] = neckW;
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
        ? (1 -
            smoothstep(
              0.0,
              Math.max(face.mouthH * 1.15, face.faceH * 0.065),
              Math.max(0, yOuterUp - mp.y),
            )) *
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
    let cornerW = (1 - smoothstep(0.0, face.mouthW * 0.85, dCorners)) * localScale;
    if (
      innerLipSet.has(i) ||
      outerUpperSet.has(i) ||
      outerLowerSet.has(i) ||
      inUpperVermilion ||
      inLowerVermilion
    ) {
      cornerW = Math.max(
        cornerW,
        (1 - smoothstep(0.0, face.mouthW * 0.85, dMouth)) * 0.9 * localScale,
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
    const dWing = Math.min(
      dist2(nx, ny, rNoseWing[0], rNoseWing[1]),
      dist2(nx, ny, lNoseWing[0], lNoseWing[1]),
    );
    const noseW = (1 - smoothstep(0.0, face.faceW * 0.1, dWing)) * localScale;

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
      smoothstep(0.0, 0.015, nx) *
      smoothstep(1.0, 0.985, nx) *
      smoothstep(0.0, 0.015, ny) *
      smoothstep(1.0, 0.985, ny);
    const headW =
      (ny <= chin[1] ? Math.max(0.96, computeHeadWeight(nx, ny)) : computeHeadWeight(nx, ny)) *
      borderMask;

    const neckW = computeNeckWeight(nx, ny) * borderMask;

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
    faceVerts[fPtr++] = neckW;
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
  thinking = false,
  userSpeaking = false,
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
  swallowSignal,
  topicComplexity = 0,
  className = "",
}: {
  src: string | null;
  name: string;
  active: boolean;
  emotion?: AvatarEmotion | undefined;
  listening?: boolean | undefined;
  thinking?: boolean | undefined;
  userSpeaking?: boolean | undefined;
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
  swallowSignal?: { seq: number; strength?: number } | undefined;
  topicComplexity?: number | undefined;
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
  const thinkingRef = useRef(thinking);
  thinkingRef.current = thinking;
  const userSpeakingRef = useRef(userSpeaking);
  userSpeakingRef.current = userSpeaking;
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
  const swallowSignalRef = useRef(swallowSignal);
  swallowSignalRef.current = swallowSignal;
  const topicComplexityRef = useRef(topicComplexity);
  topicComplexityRef.current = topicComplexity;

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
      uPucker: loc("uPucker"),
      uLipPress: loc("uLipPress"),
      uLabiodental: loc("uLabiodental"),
      uUpperLipLift: loc("uUpperLipLift"),
      uLowerLipDrop: loc("uLowerLipDrop"),
      uLipBite: loc("uLipBite"),
      uSmile: loc("uSmile"),
      uSmileR: loc("uSmileR"),
      uSmileL: loc("uSmileL"),
      uSmileTeeth: loc("uSmileTeeth"),
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
      uSwallowPhase: loc("uSwallowPhase"),
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
      uEyeBrowFloor: loc("uEyeBrowFloor"),
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
        thinking: thinkingRef.current,
        userSpeaking: userSpeakingRef.current,
        emotion: emotionRef.current,
        visemes,
        hasVisibleTeeth: face.hasVisibleTeeth,
        restMouthRatio: face.restMouthRatio,
        userFace: userFaceRef.current,
        mirrorMode: mirrorModeRef.current,
        reducedMotion,
        animeStyle: false,
        swallowSignal: swallowSignalRef.current,
        topicComplexity: topicComplexityRef.current,
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
      gl.uniform1f(uLoc.uPucker, frame.pucker);
      gl.uniform1f(uLoc.uLipPress, frame.lipPress);
      gl.uniform1f(uLoc.uLabiodental, frame.labiodental);
      gl.uniform1f(uLoc.uUpperLipLift, frame.upperLipLift);
      gl.uniform1f(uLoc.uLowerLipDrop, frame.lowerLipDrop);
      gl.uniform1f(uLoc.uLipBite, frame.lipBite);
      gl.uniform1f(uLoc.uSmile, frame.smile);
      gl.uniform1f(uLoc.uSmileR, frame.smileR);
      gl.uniform1f(uLoc.uSmileL, frame.smileL);
      gl.uniform1f(uLoc.uSmileTeeth, frame.smileTeeth);
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
      gl.uniform1f(uLoc.uSwallowPhase, frame.swallowPhase);
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
      gl.uniform2f(
        uLoc.uEyeBrowFloor,
        face.rightEyeCurve.browFloorEy ?? -0.78,
        face.leftEyeCurve.browFloorEy ?? -0.78,
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
