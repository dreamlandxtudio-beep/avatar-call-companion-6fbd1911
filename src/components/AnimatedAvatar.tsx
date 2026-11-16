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
uniform float uForwardLean;
uniform vec2 uHeadTranslate;
uniform float uTorsoBreath;
uniform float uTorsoSwayX;
uniform float uShoulderTilt;
uniform vec2 uHand0Delta;
uniform vec2 uHand1Delta;
uniform vec2 uHairDelta;
uniform float uIsBasePass;
uniform float uPass;

out vec2 vUv;
out float vEdgeFade;
out float vMouthLipT;
out float vMouthX;
out float vMouthPlaneY;
out float vJawDropMouthW;
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
    vMouthPlaneY = 0.0;
    vJawDropMouthW = 0.0;
    vRigidMouth = vec2(0.0);
    vFaceCoord = vec2(0.0);
    return;
  }

  float aspect = max(0.25, uAspect);
  float headW = aHeadRig.x;
  float torsoW = aHeadRig.y;
  float jawW = aHeadRig.z;
  float side = aBrowRig.w;

  // Compute rest-frame normalized face coordinates for consistent dynamic expression wrinkles
  vec2 faceRel = vec2((aPosUv.x - uFaceCenter.x) * aspect, aPosUv.y - uFaceCenter.y);
  vFaceCoord = vec2(
    dot(faceRel, uFaceAxis.xy) / max(0.05, uFaceW * 0.5),
    dot(faceRel, uFaceAxis.zw) / max(0.05, uFaceH * 0.5)
  );

  // 1) Subtle, cohesive brow ribbon lift & orbital squint (zero skin stretching)
  float asym = side * uBrowAsym;
  float browLift = clamp((aBrowRig.x * (uInnerBrowY + asym * 0.3) + aBrowRig.y * (uOuterBrowY + asym)) * 0.020, -0.024, 0.026) * uFaceH;
  float furrowDrop = aBrowRig.x * max(0.0, uInnerBrowPinch) * 0.010 * uFaceH;
  float browPinch = -side * aBrowRig.x * uInnerBrowPinch * 0.018 * uFaceW;
  float orbitLift = aBrowRig.z * (uSquint * 0.011 - uEyeWide * 0.007) * uFaceH;
  p = addAxisDelta(p, uFaceAxis, vec2(browPinch, -(browLift - furrowDrop + orbitLift)), aspect);

  // 2) Subtle nostril flare
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(
      side * aFaceRig.w * uNoseFlare * 0.008 * uFaceW,
      -aFaceRig.w * max(0.0, uNoseFlare) * 0.007 * uFaceH
    ),
    aspect
  );

  // 3) Gentle zygomatic cheek lift preserving rounded oval cheekbones
  float sideSmile = mix(uSmileR, uSmileL, smoothstep(-0.35, 0.35, side));
  float laughCheek = uCheekLift * 0.015 + max(0.0, sideSmile) * 0.009;
  float cheekSpread = side * aFaceRig.z * (max(0.0, sideSmile) * 0.005 + uCheekLift * 0.004) * uFaceW;
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(cheekSpread, -aFaceRig.z * laughCheek * uFaceH),
    aspect
  );

  // 4) Smooth rounded chin & mandible hinge + subtle phoneme lip shaping
  vec2 mouthRel = vec2((p.x - uMouthCenter.x) * aspect, p.y - uMouthCenter.y);
  float mSign = clamp(dot(mouthRel, uMouthAxis.xy) / max(0.01, uMouthW * 0.46), -1.0, 1.0);
  float localSmile = mix(uSmileR, uSmileL, smoothstep(-0.35, 0.35, mSign));
  float lipSealZone = 1.0 - smoothstep(uMouthW * 0.32, uMouthW * 0.86, length(mouthRel));

  float upperLipLift = aFaceRig.x * (uOpen * 0.008 + max(0.0, localSmile) * 0.009 + uRound * 0.003 - uLipPress * 0.007) * uMouthW;
  float cornerWide = mSign * aFaceRig.y * (uWide * 0.016 + localSmile * 0.018 - uRound * 0.024 - uLipPress * 0.008) * uMouthW;
  float cornerLift = abs(mSign) * aFaceRig.y * (localSmile * 0.020 + uWide * 0.004 - uRound * 0.005) * uMouthW;
  float cornerJawFollow = aFaceRig.y * (1.0 - aFaceRig.x) * uOpen * 0.010 * uFaceH;
  float mouthLatSkew = uMouthSkewX * 0.009 * uMouthW * (aFaceRig.y * 0.85 + aFaceRig.x * 0.35);

  float jawOpenDrop = uOpen * 0.044 * uFaceH;
  float jawPressSeal = -jawW * lipSealZone * uLipPress * 0.007 * uFaceH;
  float jawDrop = jawOpenDrop * jawW + jawPressSeal;
  float jawInwardTuck = -mSign * jawW * (uRound * 0.004) * uFaceW;
  p = addAxisDelta(
    p,
    uMouthAxis,
    vec2(
      cornerWide + jawInwardTuck + mouthLatSkew,
      jawDrop + cornerJawFollow - upperLipLift - cornerLift
    ),
    aspect
  );

  // Rigid cranial mouth coordinates & exact synchronized jaw drop in uMouthW units
  vec2 postMouthRel = vec2((p.x - uMouthCenter.x) * aspect, p.y - uMouthCenter.y);
  vMouthPlaneY = dot(postMouthRel, uMouthAxis.zw) / max(0.01, uMouthW);
  vJawDropMouthW = jawOpenDrop / max(0.01, uMouthW);
  vec2 rigidMouthRel = vec2((aPosUv.x - uMouthCenter.x) * aspect, aPosUv.y - uMouthCenter.y);
  vRigidMouth = vec2(
    dot(rigidMouthRel, uMouthAxis.xy) / max(0.01, uMouthW),
    dot(rigidMouthRel, uMouthAxis.zw) / max(0.01, uMouthW)
  );

  // 5) Subtle oval-preserving 2.5D cranial rotation (keeps ears, cheeks & hair rounded and human)
  if (headW > 0.001) {
    vec2 skullPivot = mix(uNeckPivot, uFaceCenter, 0.38);
    vec2 rel = vec2((p.x - skullPivot.x) * aspect, p.y - skullPivot.y);
    float cRoll = cos(uHeadRoll * headW);
    float sRoll = sin(uHeadRoll * headW);
    vec2 r = vec2(rel.x * cRoll - rel.y * sRoll, rel.x * sRoll + rel.y * cRoll);

    float hx = clamp(vFaceCoord.x, -1.7, 1.7);
    float hy = clamp(vFaceCoord.y, -1.75, 1.55);
    float rawSkull = sqrt(max(0.08, 1.0 - min(0.92, (hx * 0.48) * (hx * 0.48) + (hy * 0.36) * (hy * 0.36))));
    float depthRelief = clamp(aHandRig.z, 0.0, 1.0);
    float skullZ = (rawSkull * 0.85 + depthRelief * 0.18) * uFaceW * 0.34;

    float yawRad = clamp(uHeadRot.x * 0.28, -0.16, 0.16) * headW;
    float pitchRad = clamp(uHeadRot.y * 0.20, -0.14, 0.14) * headW;

    float cosY = cos(yawRad);
    float sinY = sin(yawRad);
    float rotX = r.x * mix(1.0, cosY, 0.28) + skullZ * sinY * 0.72;
    float rotZ = -r.x * sinY * 0.72 + skullZ * cosY;

    float perspZ = 1.0 + clamp(rotZ / max(0.08, uFaceW) * 0.032, -0.025, 0.03);
    r.x = rotX * perspZ;
    r.y = r.y * perspZ - pitchRad * 0.09 * (rawSkull * 0.82 + 0.18) * uFaceH;

    p.x = skullPivot.x + r.x / aspect + uHeadTranslate.x * headW;
    p.y = skullPivot.y + r.y + uHeadTranslate.y * headW;
  }

  // 6) Calm torso breathing + micro attentive bust lean
  float shoulderSide = clamp((p.x - uNeckPivot.x) * aspect / max(0.05, uFaceW * 0.65), -1.0, 1.0);
  p.x += (uTorsoSwayX * 0.005 * uFaceW * torsoW) / aspect;
  p.y -= (uTorsoBreath * 0.006 + shoulderSide * uShoulderTilt * 0.006) * uFaceH * torsoW;

  float bustWeight = max(headW, torsoW * 0.72);
  if (bustWeight > 0.001) {
    vec2 bustOrigin = mix(uNeckPivot, uFaceCenter, 0.38);
    float leanVal = clamp(uForwardLean, -0.5, 0.55);
    float bustZoom = 1.0 + leanVal * bustWeight * 0.024;
    vec2 relBust = vec2((p.x - bustOrigin.x) * aspect, p.y - bustOrigin.y) * bustZoom;
    p.x = bustOrigin.x + relBust.x / aspect;
    p.y = bustOrigin.y + relBust.y + leanVal * bustWeight * 0.005 * uFaceH;
  }

  // 7) Gentle organic hair follow-through in upper hair crown zone
  if (p.y < uFaceCenter.y && headW > 0.05) {
    float hairZone = headW * smoothstep(uFaceCenter.y, uFaceCenter.y - uFaceH * 0.45, p.y);
    p += uHairDelta * hairZone;
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
uniform vec2 uBgTexel;
uniform float uBlurredBg;
uniform float uPass; // -1.0 = reconstructed background, 1.0 = cutout portrait mesh, 2.0 = oral interior

uniform vec2 uREyeCenter;
uniform vec2 uREyePupil;
uniform vec4 uREyeAxis;
uniform vec2 uREyeScale;
uniform vec4 uREyeUpperA;
uniform vec4 uREyeUpperB;
uniform vec4 uREyeLowerA;
uniform vec4 uREyeLowerB;
uniform float uRIrisRad;

uniform vec2 uLEyeCenter;
uniform vec2 uLEyePupil;
uniform vec4 uLEyeAxis;
uniform vec2 uLEyeScale;
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

uniform float uOpen;
uniform float uWide;
uniform float uRound;
uniform float uLipPress;
uniform float uSmile;
uniform float uSmileR;
uniform float uSmileL;
uniform float uCheekLift;
uniform float uInnerBrowY;
uniform float uInnerBrowPinch;
uniform float uOuterBrowY;
uniform float uNoseFlare;
uniform float uTongueOut;
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
in float vMouthPlaneY;
in float vJawDropMouthW;
in vec2 vRigidMouth;
in vec2 vFaceCoord;

out vec4 fragColor;

float evalLidCurve8(float t, vec4 a, vec4 b) {
  float x = clamp(t, -1.0, 1.0);
  // Smooth 8-knot almond eye contour pinned cleanly to 0 at left (-1) & right (+1) eye commissures
  float y = 0.0;
  if (x <= -0.82) {
    y = mix(0.0, a.x, smoothstep(-1.0, -0.82, x));
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
    y = mix(b.w, 0.0, smoothstep(0.82, 1.0, x));
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

// Photorealistic human eye: commissure-anchored almond lids (zero black smear), side-consistent sclera, obsidian pupil & subtle iris light
vec3 renderHumanEye(
  vec3 baseRgb,
  vec2 eyeCenter,
  vec2 eyePupil,
  vec4 eyeAxis,
  vec2 eyeScale,
  vec4 upperA,
  vec4 upperB,
  vec4 lowerA,
  vec4 lowerB,
  float irisRadNorm,
  vec3 scleraNegColor,
  vec3 scleraPosColor,
  float eyeBlink,
  float aspect
) {
  vec2 dUv = vec2((vUv.x - eyeCenter.x) * aspect, vUv.y - eyeCenter.y);
  float halfW = max(0.005, eyeScale.x);
  float scaleY = max(0.004, eyeScale.y);

  float ex = dot(dUv, eyeAxis.xy) / halfW;
  float ey = dot(dUv, eyeAxis.zw) / scaleY;

  float rawTopY = evalLidCurve8(ex, upperA, upperB);
  float rawBotY = evalLidCurve8(ex, lowerA, lowerB);

  float cornerEnv = 1.0 - smoothstep(0.86, 1.02, abs(ex));
  if (cornerEnv < 0.002) return baseRgb;

  float blinkClose = clamp(eyeBlink - uEyeWide * 0.05, 0.0, 1.0);
  float squintClose = clamp(uSquint * 0.24, 0.0, 0.36);

  // Anatomical blink: upper lid sweeps down over the spherical globe to meet a gently rising lower lid (~16%)
  float meetArch = 0.04 * (1.0 - min(1.0, ex * ex * 0.85)) * blinkClose;
  float meet = mix(rawTopY, rawBotY, 0.82) + meetArch;
  float topY = mix(rawTopY, meet, blinkClose);
  float squintBot = mix(rawBotY, mix(rawBotY, rawTopY, 0.22), squintClose);
  float botY = mix(squintBot, meet + 0.010, blinkClose);

  float feather = max(0.024, fwidth(ey) * 1.25);
  float rawAperture = cornerEnv * smoothstep(rawTopY - feather, rawTopY + feather, ey) *
    (1.0 - smoothstep(rawBotY - feather, rawBotY + feather, ey));
  if (rawAperture < 0.002) return baseRgb;

  float aperture = cornerEnv * smoothstep(topY - feather, topY + feather, ey) *
    (1.0 - smoothstep(botY - feather, botY + feather, ey));

  // Authentic photo-textured eyelid lowering:
  // The person's real upper eyelid skin from the palpebral crease stretches downward naturally,
  // preserving 100% of the authentic skin texture, pores, fold lines, and natural coloring.
  float upperCrease = rawTopY - 1.22 * (1.0 - 0.15 * min(1.0, ex * ex));
  float lowerCrease = rawBotY + 0.65 * (1.0 - 0.15 * min(1.0, ex * ex));

  float sourceY;
  if (ey <= topY) {
    float tUp = clamp((ey - upperCrease) / max(0.03, topY - upperCrease), 0.0, 1.0);
    float tWarp = pow(tUp, 1.10);
    sourceY = mix(upperCrease, rawTopY - 0.030, tWarp);
  } else if (ey >= botY) {
    float tDn = clamp((ey - botY) / max(0.03, lowerCrease - botY), 0.0, 1.0);
    sourceY = mix(rawBotY + 0.025, lowerCrease, tDn);
  } else {
    float tAp = clamp((ey - topY) / max(0.02, botY - topY), 0.0, 1.0);
    sourceY = mix(rawTopY - 0.030, rawBotY + 0.025, tAp);
  }

  vec2 sourceDelta = eyeAxis.xy * (ex * halfW) + eyeAxis.zw * (sourceY * scaleY);
  vec2 sourceUv = eyeCenter + vec2(sourceDelta.x / aspect, sourceDelta.y);
  vec3 photoSkin = texture(uPhotoTex, clamp(sourceUv, 0.001, 0.999)).rgb;

  // Delicate natural eyelash contact line right where upper and lower lids meet when closed
  float crispLashEdge = exp(-pow((ey - topY) / 0.048, 2.0)) * (0.04 + 0.28 * blinkClose) *
    (1.0 - smoothstep(0.84, 1.00, abs(ex)));

  // Subtle spherical dome curvature highlight across the closed eyelid
  float domeHighlight = exp(-pow(ex / 0.58, 2.0)) *
    exp(-pow((ey - mix(rawTopY, topY, 0.52)) / 0.48, 2.0)) *
    blinkClose * 0.035;

  vec3 sculptedLid = clamp(photoSkin * (1.0 - crispLashEdge) + vec3(domeHighlight), 0.0, 1.0);

  // Iris & cornea gaze shift within open eye region; preserve left & right sclera tone consistency
  vec2 gazeA = eyeAxis.xy * (uGaze.x * halfW * 0.38) - eyeAxis.zw * (uGaze.y * scaleY * 0.24);
  vec2 shift = vec2(gazeA.x / aspect, gazeA.y);
  float irisR = max(0.002, irisRadNorm);
  vec2 oldD = vec2((vUv.x - eyePupil.x) * aspect, vUv.y - eyePupil.y);
  vec2 newD = vec2((vUv.x - eyePupil.x - shift.x) * aspect, vUv.y - eyePupil.y - shift.y);

  float lenNew = length(newD);
  float newMask = 1.0 - smoothstep(irisR * 0.90, irisR * 1.08, lenNew);
  float oldDist = length(oldD);
  float oldMask = (1.0 - smoothstep(irisR * 0.85, irisR * 1.25, oldDist)) * (1.0 - newMask);

  vec2 dPupil = vec2((eyePupil.x - eyeCenter.x) * aspect, eyePupil.y - eyeCenter.y);
  float pupilEx = dot(dPupil, eyeAxis.xy) / halfW;
  float irisEx = irisR / halfW;
  float rowY = clamp(ey, rawTopY + 0.16, rawBotY - 0.16);
  float negEx = clamp(pupilEx - irisEx * 1.22, -0.84, pupilEx - irisEx * 1.10);
  float posEx = clamp(pupilEx + irisEx * 1.22, pupilEx + irisEx * 1.10, 0.84);

  vec2 negDelta = eyeAxis.xy * (negEx * halfW) + eyeAxis.zw * (rowY * scaleY);
  vec2 posDelta = eyeAxis.xy * (posEx * halfW) + eyeAxis.zw * (rowY * scaleY);
  vec3 rowNeg = texture(uPhotoTex, clamp(eyeCenter + vec2(negDelta.x / aspect, negDelta.y), 0.001, 0.999)).rgb;
  vec3 rowPos = texture(uPhotoTex, clamp(eyeCenter + vec2(posDelta.x / aspect, posDelta.y), 0.001, 0.999)).rgb;

  float negBaseL = dot(scleraNegColor, vec3(0.299, 0.587, 0.114));
  float posBaseL = dot(scleraPosColor, vec3(0.299, 0.587, 0.114));
  float negRowL = dot(rowNeg, vec3(0.299, 0.587, 0.114));
  float posRowL = dot(rowPos, vec3(0.299, 0.587, 0.114));

  float negOk = smoothstep(negBaseL * 0.40, negBaseL * 0.75, negRowL);
  float posOk = smoothstep(posBaseL * 0.40, posBaseL * 0.75, posRowL);

  vec3 leftSclera = mix(scleraNegColor, rowNeg, clamp(negOk * 0.85, 0.35, 0.94));
  vec3 rightSclera = mix(scleraPosColor, rowPos, clamp(posOk * 0.85, 0.35, 0.94));
  float sideSplit = smoothstep(-0.25, 0.25, ex - pupilEx);
  float topScleraAO = mix(0.88, 1.0, smoothstep(rawTopY, rawTopY + 0.32, ey));
  vec3 exactSclera = mix(leftSclera, rightSclera, sideSplit) * topScleraAO;

  // Seamless radial blend with original eye background
  float fadeEdge = smoothstep(irisR * 0.75, irisR * 1.25, oldDist);
  vec3 smoothSclera = mix(exactSclera, baseRgb, fadeEdge * 0.50);
  float travel = smoothstep(0.0, irisR * 0.03, length(gazeA));
  vec3 eyeCol = mix(baseRgb, smoothSclera, oldMask * travel);

  vec3 irisSample = texture(uPhotoTex, clamp(vUv - shift, 0.001, 0.999)).rgb;
  float irisNorm = lenNew / irisR;
  float irisLuma = dot(irisSample, vec3(0.299, 0.587, 0.114));
  float glintProtect = smoothstep(0.56, 0.78, irisLuma);
  float pupilMask = (1.0 - smoothstep(0.22, 0.34, irisNorm)) * (1.0 - glintProtect);
  irisSample = mix(irisSample, irisSample * 0.42 + vec3(0.008), pupilMask * 0.62);
  float irisRing = smoothstep(0.35, 0.48, irisNorm) * (1.0 - smoothstep(0.76, 0.90, irisNorm)) * (1.0 - glintProtect);
  float sideCatch = smoothstep(0.08, 0.85, abs(uGaze.x)) * 0.85 + smoothstep(-0.25, 0.75, uGaze.x) * 0.25;
  float irisBoost = clamp(sideCatch, 0.0, 1.0) * irisRing;
  irisSample = mix(vec3(irisLuma), irisSample, 1.0 + 0.10 * irisBoost) * (1.0 + 0.06 * irisBoost);

  float upperLidShadow = mix(0.88, 1.0, smoothstep(topY, topY + 0.26, ey));
  eyeCol = mix(eyeCol, clamp(irisSample, 0.0, 1.0) * upperLidShadow, newMask);

  float openInside = clamp(aperture / max(0.004, rawAperture), 0.0, 1.0) *
    (1.0 - smoothstep(0.86, 0.98, blinkClose));
  vec3 result = mix(sculptedLid, eyeCol, openInside);
  return mix(baseRgb, result, rawAperture);
}

// Dedicated Anime / 2D eye model: uniform arched blink, untouched open eye background, dual left/right sclera
vec3 renderAnimeEye(
  vec3 baseRgb,
  vec2 eyeCenter,
  vec2 eyePupil,
  vec4 eyeAxis,
  vec2 eyeScale,
  vec4 upperA,
  vec4 upperB,
  vec4 lowerA,
  vec4 lowerB,
  float irisRadNorm,
  vec3 scleraNegColor,
  vec3 scleraPosColor,
  vec3 eyelidSkinColor,
  float eyeBlink,
  float aspect
) {
  vec2 dUv = vec2((vUv.x - eyeCenter.x) * aspect, vUv.y - eyeCenter.y);
  float halfW = max(0.006, eyeScale.x * 1.08);
  float scaleY = max(0.006, eyeScale.y * 1.24);

  float ex = dot(dUv, eyeAxis.xy) / halfW;
  float ey = dot(dUv, eyeAxis.zw) / scaleY;

  float rawTopY = evalLidCurve8(ex, upperA, upperB) * 1.06;
  float rawBotY = evalLidCurve8(ex, lowerA, lowerB) * 1.06;

  float midY = (rawTopY + rawBotY) * 0.5 - 0.04;
  float archShape = sqrt(max(0.20, 1.0 - 0.46 * min(1.25, ex * ex)));
  float spanY = max(0.88, (rawBotY - rawTopY) * 0.66 + 0.62) * archShape;
  float ovalR = length(vec2(ex / 1.15, (ey - midY) / spanY));
  float ovalMask = 1.0 - smoothstep(0.70, 1.06, ovalR);
  if (ovalMask < 0.002) return baseRgb;

  float blinkClose = clamp(eyeBlink - uEyeWide * 0.08, 0.0, 1.0);
  float squintClose = clamp(uSquint * 0.30, 0.0, 0.44);
  float lidAction = max(blinkClose, squintClose);

  float cornerArch = 0.08 * (1.0 - ex * ex) * blinkClose;
  float meet = mix(rawTopY, rawBotY, 0.84) + cornerArch;
  float topY = mix(rawTopY, meet, blinkClose);
  float squintBot = mix(rawBotY, mix(rawBotY, rawTopY, 0.30), squintClose);
  float botY = mix(squintBot, meet + 0.015, blinkClose);

  float feather = max(0.032, fwidth(ey) * 1.4);
  float rawAperture = smoothstep(rawTopY - feather, rawTopY + feather, ey) *
    (1.0 - smoothstep(rawBotY - feather, rawBotY + feather, ey));
  float aperture = smoothstep(topY - feather, topY + feather, ey) *
    (1.0 - smoothstep(botY - feather, botY + feather, ey));

  float outsideAction = smoothstep(0.018, 0.22, lidAction);
  float effectMask = max(rawAperture, outsideAction) * ovalMask;
  if (effectMask < 0.002) return baseRgb;

  float archEnvelope = 1.0 - 0.14 * min(1.0, ex * ex);
  float upperCrease = rawTopY - 1.50 * archEnvelope;
  float lowerCrease = rawBotY + 0.80 * archEnvelope;
  float sourceY;
  if (ey <= topY) {
    float tUp = clamp((ey - upperCrease) / max(0.04, topY - upperCrease), 0.0, 1.0);
    sourceY = mix(upperCrease, rawTopY - 0.012, tUp);
  } else if (ey >= botY) {
    float tDn = clamp((ey - botY) / max(0.04, lowerCrease - botY), 0.0, 1.0);
    sourceY = mix(rawBotY + 0.012, lowerCrease, tDn);
  } else {
    float tAp = clamp((ey - topY) / max(0.025, botY - topY), 0.0, 1.0);
    sourceY = mix(rawTopY, rawBotY, tAp);
  }

  vec2 sourceDelta = eyeAxis.xy * (ex * halfW) + eyeAxis.zw * (sourceY * scaleY);
  vec2 sourceUv = eyeCenter + vec2(sourceDelta.x / aspect, sourceDelta.y);
  vec3 photoLid = mix(baseRgb, texture(uPhotoTex, clamp(sourceUv, 0.001, 0.999)).rgb, outsideAction);
  vec3 texturedLid = mix(photoLid, mix(photoLid, eyelidSkinColor, 0.55), blinkClose * (1.0 - aperture));

  float lashLine = exp(-pow((ey - topY) / 0.11, 2.0)) * blinkClose * (1.0 - smoothstep(0.86, 1.12, abs(ex)));
  texturedLid = mix(texturedLid, vec3(0.08, 0.06, 0.09), lashLine * 0.82);

  vec2 gazeA = eyeAxis.xy * (uGaze.x * halfW * 0.34) - eyeAxis.zw * (uGaze.y * scaleY * 0.28);
  vec2 shift = vec2(gazeA.x / aspect, gazeA.y);
  float irisR = max(0.003, irisRadNorm * 1.06);
  vec2 oldD = vec2((vUv.x - eyePupil.x) * aspect, vUv.y - eyePupil.y);
  vec2 newD = vec2((vUv.x - eyePupil.x - shift.x) * aspect, vUv.y - eyePupil.y - shift.y);
  float newMask = 1.0 - smoothstep(irisR * 0.88, irisR * 1.12, length(newD));
  float oldMask = (1.0 - smoothstep(irisR * 0.94, irisR * 1.22, length(oldD))) * (1.0 - newMask);

  float sideMix = smoothstep(-0.36, 0.36, ex);
  vec3 sideSclera = mix(scleraNegColor, scleraPosColor, sideMix);
  vec3 cleanSclera = sideSclera * mix(0.90, 1.0, smoothstep(topY, mix(topY, botY, 0.38), ey));
  float travel = smoothstep(0.0, irisR * 0.04, length(gazeA));
  vec3 eyeCol = mix(baseRgb, cleanSclera, oldMask * travel);
  vec3 irisSample = texture(uPhotoTex, clamp(vUv - shift, 0.001, 0.999)).rgb;
  eyeCol = mix(eyeCol, irisSample, newMask);

  vec3 result = mix(texturedLid, eyeCol, aperture * (1.0 - smoothstep(0.84, 0.98, blinkClose)));
  return mix(baseRgb, result, effectMask);
}

// Dynamic facial creases, glabella "11" wrinkles, surprise forehead lines & mouth corner/inter-lip crease shading
vec3 applyExpressionWrinkles(vec3 col, vec2 fc, vec2 rm) {
  float absX = abs(fc.x);
  float skinMask = (1.0 - smoothstep(0.84, 1.08, absX)) * (1.0 - smoothstep(0.88, 1.10, abs(fc.y)));
  if (skinMask < 0.01) return col;

  float luma = dot(col, vec3(0.299, 0.587, 0.114));
  float skinToneMask = smoothstep(0.14, 0.32, luma);
  if (skinToneMask < 0.01) return col;

  // 1) Nasolabial folds (zygomatic smile & speech cheek lift per side)
  float sideSmile = fc.x < 0.0 ? max(0.0, uSmileR) : max(0.0, uSmileL);
  float foldWeight = clamp(sideSmile * 0.62 + uCheekLift * 0.46, 0.0, 1.1);
  float nasoCurveX = 0.24 + smoothstep(0.02, 0.56, fc.y) * 0.36;
  float nasoVert = smoothstep(0.00, 0.14, fc.y) * (1.0 - smoothstep(0.46, 0.64, fc.y));
  float nasoDist = abs(absX - nasoCurveX);
  float nasoShadow = exp(-pow(nasoDist / 0.052, 2.0)) * nasoVert * foldWeight;

  // 2) Crow's-feet outer-eye smile crinkles
  float squintAct = smoothstep(0.18, 0.85, uSquint + uCheekLift * 0.35);
  float crowZone = smoothstep(0.54, 0.72, absX) * (1.0 - smoothstep(0.88, 1.02, absX)) *
    exp(-pow((fc.y + 0.16) / 0.14, 2.0));
  float crowRays = 0.5 + 0.5 * cos((fc.y + 0.16) * 44.0 - (absX - 0.66) * 18.0);
  float crowShadow = crowZone * crowRays * squintAct * 0.56;

  // 3) Crisp glabella "11" frown creases between brows + brow-bridge scrunch
  float furrowAct = smoothstep(0.08, 0.78, uInnerBrowPinch);
  float glabellaZone = exp(-pow((absX - 0.072) / 0.032, 2.0)) *
    smoothstep(-0.56, -0.44, fc.y) * (1.0 - smoothstep(-0.24, -0.14, fc.y));
  float glabellaShadow = glabellaZone * furrowAct;

  float noseScrunchAct = smoothstep(0.14, 0.82, max(0.0, uNoseFlare));
  float bunnyZone = exp(-pow((absX - 0.12) / 0.045, 2.0)) *
    exp(-pow((fc.y + 0.02) / 0.08, 2.0));
  float bunnyShadow = bunnyZone * noseScrunchAct * 0.65;

  // 4) Expressive forehead lines when brows lift in surprise / curiosity
  float browHigh = smoothstep(0.28, 0.88, (uInnerBrowY + uOuterBrowY) * 0.5);
  float foreheadZone = (1.0 - smoothstep(0.28, 0.68, absX)) *
    smoothstep(-0.88, -0.74, fc.y) * (1.0 - smoothstep(-0.50, -0.40, fc.y));
  float foreheadBands = 0.5 + 0.5 * cos((fc.y + 0.66) * 40.0);
  float foreheadShadow = foreheadZone * foreheadBands * browHigh * 0.48;

  // 5) Whisper-subtle mouth corner dimple shading
  float restSlitY = (uRestMouthTopY + uRestMouthBotY) * 0.5;
  float cornerDimple = exp(-pow((abs(rm.x) - 0.96) / 0.12, 2.0)) *
    exp(-pow((rm.y - restSlitY) / 0.075, 2.0)) * (0.14 + uSmile * 0.25 + uLipPress * 0.25);

  float totalCrease = clamp(
    (nasoShadow * 0.07 + crowShadow * 0.06 + glabellaShadow * 0.12 + bunnyShadow * 0.06 +
     foreheadShadow * 0.06 + cornerDimple * 0.07) * skinMask * skinToneMask,
    0.0,
    0.11
  );
  vec3 shaded = col * mix(vec3(1.0), vec3(0.82, 0.74, 0.71), totalCrease);
  return clamp(shaded, 0.0, 1.0);
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
    // Watertight multi-row oral interior between inner upper and inner lower lips
    float openMask = uHasVisibleTeeth > 0.5
      ? smoothstep(0.035, 0.12, uOpen)
      : smoothstep(0.012, 0.055, uOpen);
    if (openMask <= 0.002) {
      discard;
    }

    float absX = abs(vMouthX);
    float lipRimFeather =
      smoothstep(0.0, 0.06, vMouthLipT) *
      smoothstep(1.0, 0.95, vMouthLipT) *
      (1.0 - smoothstep(0.80, 0.96, absX));

    float rigidX = vRigidMouth.x;
    float absRigidX = abs(rigidX);
    float mouthY = vMouthPlaneY;
    float lowerPlaneY = mouthY - vJawDropMouthW;

    if (uStylized > 0.5) {
      vec3 cavityCol = mix(vec3(0.24, 0.06, 0.10), uLipColor * 0.34, 0.38);
      float animeUpperLift = smoothstep(0.06, 0.65, uOpen) * 0.006 + clamp(uSmile, 0.0, 1.0) * 0.005;
      float animeUpperTop = uRestMouthTopY - 0.036 - animeUpperLift;
      float animeUpperBot = uRestMouthTopY + 0.038 - animeUpperLift;
      float upperTeethBar =
        smoothstep(animeUpperTop - 0.008, animeUpperTop + 0.008, mouthY) *
        (1.0 - smoothstep(animeUpperBot - 0.010, animeUpperBot + 0.006, mouthY)) *
        (1.0 - smoothstep(0.38, 0.52, absRigidX));
      float animeLowerTop = uRestMouthBotY - 0.044;
      float animeLowerBot = uRestMouthBotY + 0.018;
      float lowerTeethBar =
        smoothstep(animeLowerTop - 0.006, animeLowerTop + 0.008, lowerPlaneY) *
        (1.0 - smoothstep(animeLowerBot - 0.008, animeLowerBot + 0.006, lowerPlaneY)) *
        smoothstep(0.08, 0.22, uOpen) *
        (1.0 - smoothstep(0.38, 0.52, absRigidX));
      float tongueShift = clamp(uTongueOut, 0.0, 1.0) * 0.018;
      float tongueArc = smoothstep(0.42 - tongueShift * 4.0, 0.58, vMouthLipT) *
        (1.0 - smoothstep(0.36, 0.66, absX));
      vec3 animeOral = mix(cavityCol, vec3(0.78, 0.36, 0.44), tongueArc * 0.88);
      animeOral = mix(animeOral, vec3(0.95, 0.94, 0.95), upperTeethBar * 0.90);
      animeOral = mix(animeOral, vec3(0.90, 0.89, 0.91), lowerTeethBar * 0.85);
      float cornerDark = 1.0 - 0.62 * smoothstep(0.54, 0.92, absX);
      fragColor = vec4(animeOral * cornerDark, openMask * lipRimFeather * fgAlpha);
      return;
    }

    // Full, continuous dental arch (archZoom = 0.44 keeps incisors and canines in solid enamel without dark gaps)
    const float archZoom = 0.44;
    float tx = clamp(0.5 + (rigidX - uYaw * 0.06) * archZoom, 0.12, 0.88);
    float biteCurve = 0.016 * rigidX * rigidX;
    float archCurveT = clamp(abs(tx - 0.5) * 2.2, 0.0, 1.0);

    float cavY = clamp(0.37 + vMouthLipT * 0.14, 0.34, 0.53);
    vec3 cavTex = sampleCrispTeeth(vec2(tx, cavY));
    vec3 deepRedMucosa = vec3(0.068, 0.012, 0.016) + cavTex * vec3(0.14, 0.026, 0.032);
    float throatDepth = 1.0 - 0.56 * (1.0 - abs(vMouthLipT - 0.46) * 1.4) * (1.0 - absX * 0.7);
    vec3 oralCol = deepRedMucosa * clamp(throatDepth, 0.24, 1.0);

    // Subtle, well-proportioned upper & lower dental arches
    float upperMicroLift = smoothstep(0.06, 0.65, uOpen) * 0.005 + clamp(uSmile, 0.0, 1.0) * 0.004;
    float upperToothH = max(0.076, mix(0.092, 0.078, archCurveT));
    float upperEdgeY = (uRestMouthTopY + 0.028) - biteCurve * 0.35 - upperMicroLift * 0.4;
    float upperGumY = upperEdgeY - upperToothH;
    float upperFrac = clamp((mouthY - upperGumY) / upperToothH, 0.0, 1.0);
    float upperTexTop = mix(0.125, 0.155, archCurveT);
    float upperTexBot = mix(0.298, 0.312, archCurveT);
    vec3 rawUpperTeeth = sampleCrispTeeth(vec2(tx, mix(upperTexTop, upperTexBot, upperFrac)));

    float lowerToothH = max(0.066, mix(0.076, 0.066, archCurveT));
    float lowerTipY = uRestMouthBotY - 0.036 + biteCurve * 0.20;
    float lowerGumY = lowerTipY + lowerToothH;
    float lowerFrac = clamp((lowerPlaneY - lowerTipY) / lowerToothH, 0.0, 1.0);
    float lowerTexTop = mix(0.758, 0.738, archCurveT);
    float lowerTexBot = mix(0.898, 0.865, archCurveT);
    vec3 rawLowerTeeth = sampleCrispTeeth(vec2(tx, mix(lowerTexTop, lowerTexBot, lowerFrac)));

    // Wet sculpted tongue dome sitting behind lower teeth
    float tongueAdv = clamp(uTongueOut, 0.0, 1.0);
    float tongueLift = clamp(uRound, 0.0, 1.0) * 0.006 + tongueAdv * 0.018;
    float tongueTopY = lowerTipY - 0.018 + 0.12 * rigidX * rigidX - tongueLift;
    float tongueBotY = lowerGumY + 0.006;
    float tongueLatMask = 1.0 - smoothstep(0.24 + tongueAdv * 0.03, 0.44 + tongueAdv * 0.04, absRigidX);
    float tongueVertMask =
      smoothstep(tongueTopY - 0.014, tongueTopY + 0.012, lowerPlaneY) *
      (1.0 - smoothstep(tongueBotY - 0.012, tongueBotY + 0.010, lowerPlaneY));
    float tongueShow = max(smoothstep(0.14, 0.34, uOpen), smoothstep(0.05, 0.25, tongueAdv));
    float tongueMask = tongueLatMask * tongueVertMask * tongueShow;
    float tNormY = clamp((lowerPlaneY - tongueTopY) / max(0.024, tongueBotY - tongueTopY), 0.0, 1.0);
    float tongueTexY = mix(0.55, 0.72, tNormY);
    vec3 tongueSample = sampleCrispTeeth(vec2(tx, tongueTexY));
    float tongueGroove = 1.0 - 0.14 * exp(-pow(rigidX / 0.09, 2.0));
    float tongueSheen = exp(-pow((tNormY - 0.34) / 0.22, 2.0)) * (1.0 - smoothstep(0.06, 0.32, absRigidX)) * 0.07;
    float tongueShade = mix(0.58, 0.98, smoothstep(0.0, 0.42, tNormY)) * tongueGroove;
    vec3 tongueCol = (tongueSample * vec3(0.70, 0.34, 0.36) + vec3(0.08, 0.018, 0.024)) * tongueShade + vec3(tongueSheen);

    vec3 enamelTarget = clamp(mix(uEnamelColor, vec3(0.89, 0.86, 0.82), 0.42), vec3(0.70), vec3(0.92));
    vec3 texScale = enamelTarget / vec3(0.86, 0.84, 0.80);
    float archRecessShade = 1.0 - 0.34 * archCurveT;
    float upperLipCast = mix(0.48, 1.0, smoothstep(0.0, 0.28, upperFrac));
    float upperIncisalSheen = exp(-pow((upperFrac - 0.68) / 0.20, 2.0)) * (1.0 - archCurveT) * 0.04;
    vec3 gumTint = mix(vec3(0.50, 0.21, 0.23), uLipColor * 0.70, 0.32);
    vec3 upperTeethSample = clamp(rawUpperTeeth * texScale * archRecessShade * upperLipCast + vec3(upperIncisalSheen), 0.0, 1.0);
    upperTeethSample = mix(gumTint * upperLipCast, upperTeethSample, smoothstep(0.0, 0.14, upperFrac));

    float lowerLipCast = mix(1.0, 0.72, smoothstep(0.74, 1.0, lowerFrac));
    float lowerIncisalSheen = exp(-pow((lowerFrac - 0.24) / 0.18, 2.0)) * (1.0 - archCurveT) * 0.035;
    vec3 lowerTeethSample = clamp(rawLowerTeeth * texScale * 0.93 * archRecessShade * lowerLipCast + vec3(lowerIncisalSheen), 0.0, 1.0);
    lowerTeethSample = mix(lowerTeethSample, gumTint * 0.82, smoothstep(0.86, 1.0, lowerFrac));

    float upperZone =
      smoothstep(upperGumY - 0.006, upperGumY + 0.008, mouthY) *
      (1.0 - smoothstep(upperEdgeY - 0.006, upperEdgeY + 0.004, mouthY)) *
      (1.0 - smoothstep(0.58, 0.84, absRigidX));
    float lowerZone =
      smoothstep(lowerTipY - 0.005, lowerTipY + 0.006, lowerPlaneY) *
      (1.0 - smoothstep(lowerGumY - 0.006, lowerGumY + 0.006, lowerPlaneY)) *
      smoothstep(upperEdgeY - 0.002, upperEdgeY + 0.012, mouthY) *
      smoothstep(0.06, 0.18, uOpen) *
      (1.0 - smoothstep(0.54, 0.80, absRigidX));

    oralCol = mix(oralCol, tongueCol, tongueMask * (1.0 - tongueAdv * 0.35));
    oralCol = mix(oralCol, upperTeethSample, upperZone * 0.95);
    oralCol = mix(oralCol, lowerTeethSample, lowerZone * 0.94 * (1.0 - tongueAdv * 0.65));
    if (tongueAdv > 0.08) {
      oralCol = mix(oralCol, tongueCol, tongueMask * smoothstep(0.08, 0.55, tongueAdv) * 0.82);
    }

    // Wet inner-vermilion border blend + anatomical ambient occlusion
    float innerRimTint = (1.0 - smoothstep(0.03, 0.15, vMouthLipT)) + smoothstep(0.84, 0.97, vMouthLipT);
    oralCol = mix(oralCol, uLipColor * 0.58, clamp(innerRimTint, 0.0, 1.0) * 0.42);
    float topAO = mix(0.46, 1.0, smoothstep(0.0, 0.14, vMouthLipT));
    float botAO = mix(0.74, 1.0, smoothstep(1.0, 0.86, vMouthLipT));
    float sideAO = 1.0 - 0.82 * smoothstep(0.48, 0.90, absX);
    float cavityLight = mix(0.68, 1.0, smoothstep(0.02, 0.22, uOpen));
    vec3 warmShadowTint = max(vec3(0.036, 0.008, 0.010), uLipColor * vec3(0.14, 0.05, 0.06));
    oralCol = mix(warmShadowTint, oralCol, topAO * botAO * sideAO * cavityLight);

    fragColor = vec4(oralCol, openMask * lipRimFeather * fgAlpha);
    return;
  }

  // Diagnostic face lines: thin, soft, nearly white so eye detail remains crisp underneath
  if (uWireframe > 0.5) {
    fragColor = vec4(0.97, 0.98, 1.0, 0.30);
    return;
  }

  if (uStylized > 0.5) {
    col = renderAnimeEye(
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
      uEyelidColor,
      uBlinkR,
      uAspect
    );
    col = renderAnimeEye(
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
      uEyelidColor,
      uBlinkL,
      uAspect
    );
  } else {
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
      uAspect
    );
  }

  // Tight anti-aliased edge transition (~1px) so skin, hair & shoulders cut flush ("ras") against blurred background
  if (fgAlpha > 0.01 && fgAlpha < 0.38) {
    vec3 bgApprox = uBlurredBg > 0.5 ? sampleBlurredBg(uvSafe) : texture(uBgTex, uvSafe).rgb;
    float fringe = 1.0 - smoothstep(0.02, 0.38, fgAlpha);
    col = mix(col, bgApprox * 0.25 + col * 0.75, fringe * 0.22);
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

      let fg: number;
      if (hasSegMask) {
        const seg = sampleSeg(nx, ny);
        // Tight flush (~1.5px) alpha threshold right at foreground boundary ("ras" against skin/hair/clothes)
        const segTight = smoothstep(0.42, 0.58, seg);
        fg = Math.max(faceCore, segTight);
      } else {
        const headDist = Math.hypot(
          ((nx - headCenterX) * aspect) / Math.max(0.08, headRx),
          (ny - headCenterY) / Math.max(0.08, headRy),
        );
        const headHalo = 1 - smoothstep(0.94, 1.04, headDist);
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

  // Dilate hole mask tightly around foreground silhouette so no sharp subject halo bleeds into blurred background
  const holeMask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let inHole = false;
      for (let dy = -5; dy <= 5 && !inHole; dy++) {
        const sy = y + dy;
        if (sy < 0 || sy >= H) continue;
        for (let dx = -5; dx <= 5; dx++) {
          const sx = x + dx;
          if (sx < 0 || sx >= W) continue;
          if (dx * dx + dy * dy <= 26 && alphaArr[sy * W + sx]! >= 0.15) {
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

  // Targeted hair-edge band smoothing: only smooth pixels inside or hugging the dilated hair/head edge
  const filled = levels[0]!.rgba;
  const edgeBand = new Uint8Array(W * H);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const idx = y * W + x;
      if (holeMask[idx]) {
        edgeBand[idx] = 1;
      } else {
        for (let dy = -2; dy <= 2 && !edgeBand[idx]; dy++) {
          const sy = y + dy;
          if (sy < 0 || sy >= H) continue;
          for (let dx = -2; dx <= 2; dx++) {
            const sx = x + dx;
            if (sx >= 0 && sx < W && holeMask[sy * W + sx]) {
              edgeBand[idx] = 1;
              break;
            }
          }
        }
      }
    }
  }

  for (let pass = 0; pass < 6; pass++) {
    const copy = new Float32Array(filled);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const idx = y * W + x;
        if (!edgeBand[idx]) continue;
        let sr = 0;
        let sg = 0;
        let sb = 0;
        let sw = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nIdx = (y + dy) * W + (x + dx);
            const w = holeMask[nIdx] ? 1.0 : 1.35;
            const o = nIdx * 4;
            sr += copy[o]! * w;
            sg += copy[o + 1]! * w;
            sb += copy[o + 2]! * w;
            sw += w;
          }
        }
        const o = idx * 4;
        const blend = holeMask[idx] ? 1.0 : 0.45;
        filled[o] = copy[o]! * (1 - blend) + (sr / sw) * blend;
        filled[o + 1] = copy[o + 1]! * (1 - blend) + (sg / sw) * blend;
        filled[o + 2] = copy[o + 2]! * (1 - blend) + (sb / sw) * blend;
      }
    }
  }

  for (let i = 0; i < W * H; i++) {
    const o = i * 4;
    if (edgeBand[i]) {
      bgImgData.data[o] = Math.round(Math.max(0, Math.min(255, filled[o]!)));
      bgImgData.data[o + 1] = Math.round(Math.max(0, Math.min(255, filled[o + 1]!)));
      bgImgData.data[o + 2] = Math.round(Math.max(0, Math.min(255, filled[o + 2]!)));
    }
    bgImgData.data[o + 3] = 255;
  }
  bgCtx.putImageData(bgImgData, 0, 0);

  // Pre-render smooth optical Gaussian blur onto bgBlurCanvas (edge-padded to eliminate border darkening)
  if (blurCtx) {
    const pad = 28;
    const tmp = document.createElement("canvas");
    tmp.width = W + pad * 2;
    tmp.height = H + pad * 2;
    const tCtx = tmp.getContext("2d");
    if (tCtx) {
      tCtx.drawImage(bgCanvas, 0, 0, W, H, pad, pad, W, H);
      // Replicate edge strips into padding
      tCtx.drawImage(bgCanvas, 0, 0, W, 1, pad, 0, W, pad);
      tCtx.drawImage(bgCanvas, 0, H - 1, W, 1, pad, H + pad, W, pad);
      tCtx.drawImage(bgCanvas, 0, 0, 1, H, 0, pad, pad, H);
      tCtx.drawImage(bgCanvas, W - 1, 0, 1, H, W + pad, pad, pad, H);
      tCtx.drawImage(bgCanvas, 0, 0, 1, 1, 0, 0, pad, pad);
      tCtx.drawImage(bgCanvas, W - 1, 0, 1, 1, W + pad, 0, pad, pad);
      tCtx.drawImage(bgCanvas, 0, H - 1, 1, 1, 0, H + pad, pad, pad);
      tCtx.drawImage(bgCanvas, W - 1, H - 1, 1, 1, W + pad, H + pad, pad, pad);
      blurCtx.filter = "blur(18px)";
      blurCtx.drawImage(tmp, pad, pad, W, H, 0, 0, W, H);
      blurCtx.filter = "none";
    } else {
      blurCtx.drawImage(bgCanvas, 0, 0);
    }
  }

  return { fgMaskCanvas, bgCanvas, bgBlurCanvas };
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

  // Expansive solid cranial plateau covering entire hair, ears, beard & temples uniformly, blending smoothly down the neck
  const craniumX = (face.center[0] * 2 + face.rightEar[0] + face.leftEar[0]) * 0.25;
  const craniumY = (face.hairTop[1] + chin[1]) * 0.46;
  const radX = Math.max(
    face.faceW * 1.06,
    Math.abs(face.hairRight[0] - face.hairLeft[0]) * aspect * 0.78,
  );
  const radY = Math.max(face.faceH * 0.98, (chin[1] - face.hairTop[1]) * 0.68);
  const neckFadeStartY = chin[1] + face.faceH * 0.05;
  const neckFadeEndY = Math.min(
    0.98,
    Math.max(face.neckBase[1] + face.faceH * 0.16, chin[1] + face.faceH * 0.46),
  );

  // Continuous rounded oval cranial envelope shared seamlessly across all vertices (zero segmentation tears or square edges)
  const computeHeadWeight = (nx: number, ny: number): number => {
    const rx = ((nx - craniumX) * aspect) / Math.max(0.08, radX);
    const ry = (ny - craniumY) / Math.max(0.08, radY);
    const radial = 1 - smoothstep(1.15, 1.85, Math.hypot(rx, ry));
    const neckFade = ny > neckFadeStartY ? 1 - smoothstep(neckFadeStartY, neckFadeEndY, ny) : 1;
    return radial * neckFade;
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
    const borderY = smoothstep(0.0, 0.06, ny) * smoothstep(1.0, 0.94, ny);
    for (let ix = 0; ix <= GX; ix++) {
      const nx = ix / GX;
      const borderX = smoothstep(0.0, 0.06, nx) * smoothstep(1.0, 0.94, nx);
      const borderMask = borderX * borderY;

      const headW = computeHeadWeight(nx, ny) * borderMask;
      const underChinJawW = computeMandibleWeight(nx, ny) * borderMask;

      const torsoW =
        smoothstep(chin[1] + 0.06 * face.faceH, neckFadeEndY + 0.06 * face.faceH, ny) *
        (1 - smoothstep(0.45 * face.faceW * 3.1, 0.96, Math.abs((nx - neckPivot[0]) * aspect))) *
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
      baseVerts[ptr++] = 0.08 * headW;
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
  let minZ = 1e9;
  let maxZ = -1e9;
  for (let i = 0; i < numPts; i++) {
    const z = point(i)[2] ?? 0;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const zSpan = Math.max(0.005, maxZ - minZ);

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
    const lipCenterArch = Math.max(0, Math.cos(mouthX * Math.PI * 0.48));
    const yOuterUp = evalLipCurve(outerUpCurve, mouthX);
    const yInnerUp = evalLipCurve(innerUpCurve, mouthX);
    const yInnerLo = evalLipCurve(innerLoCurve, mouthX);
    const yOuterLo = evalLipCurve(outerLoCurve, mouthX);
    const ySlit = mouthMidSlitY(mouthX);

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
      jawW = 0.28 * localScale;
      upperLipW = 0.16 * localScale;
    } else if (isExplicitUpper || inUpperVermilion) {
      jawW = 0;
      upperLipW = (0.38 + 0.62 * lipCenterArch) * localScale;
    } else if (isExplicitLower || inLowerVermilion) {
      // Cohesive plateau across entire lower lip vermilion matching 3D chin mandible drop
      jawW = Math.max(0.36, 0.36 + 0.64 * lipCenterArch);
      upperLipW = 0;
    } else if (mp.y <= yInnerUp) {
      jawW = 0;
    }

    const dMouth = dist2(nx, ny, mouthCenter[0], mouthCenter[1]);
    const dCorners = Math.min(
      dist2(nx, ny, rCorner[0], rCorner[1]),
      dist2(nx, ny, lCorner[0], lCorner[1]),
    );
    let cornerW = (1 - smoothstep(0.0, face.mouthW * 0.58, dCorners)) * localScale;
    if (
      innerLipSet.has(i) ||
      outerUpperSet.has(i) ||
      outerLowerSet.has(i) ||
      inUpperVermilion ||
      inLowerVermilion
    ) {
      cornerW = Math.max(
        cornerW,
        (1 - smoothstep(0.0, face.mouthW * 0.66, dMouth)) * 0.88 * localScale,
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

    const dWing = Math.min(
      dist2(nx, ny, rNoseWing[0], rNoseWing[1]),
      dist2(nx, ny, lNoseWing[0], lNoseWing[1]),
    );
    const noseW = (1 - smoothstep(0.0, face.faceW * 0.15, dWing)) * localScale;

    // Cohesive ribbon eyebrow weight: uniform plateau across upper & lower brow rows so brows keep true thickness
    let innerBrowW = 0;
    let outerBrowW = 0;
    const isRBrow = rightBrowSet.has(i);
    const isLBrow = leftBrowSet.has(i);
    const box = relFaceX < 0 ? rBrowBox : lBrowBox;
    const xRatio = Math.min(1, Math.abs(relFaceX) / Math.max(0.02, face.faceW * 0.36));

    if (isRBrow || isLBrow) {
      // Uniform 1.0 plateau vertically across the whole brow hair band; split horizontally inner vs outer
      innerBrowW = (1 - smoothstep(0.18, 0.58, xRatio)) * 0.95;
      outerBrowW = smoothstep(0.16, 0.58, xRatio) * 0.95;
    } else if (
      ny < box.eyeTopY - face.faceH * 0.01 &&
      ny > box.minY - face.faceH * 0.12 &&
      nx > box.minX - face.faceW * 0.08 &&
      nx < box.maxX + face.faceW * 0.08
    ) {
      let browBand = 0;
      if (ny >= box.minY && ny <= box.maxY) {
        browBand = 1.0;
      } else if (ny < box.minY) {
        browBand = 1 - smoothstep(0.0, face.faceH * 0.11, box.minY - ny);
      } else {
        // Taper across the upper-lid brow bone space below the brow so lifting reveals more brow bone!
        browBand = 1 - smoothstep(0.0, Math.max(0.012, box.eyeTopY - box.maxY), ny - box.maxY);
      }
      innerBrowW = browBand * (1 - smoothstep(0.18, 0.58, xRatio)) * 0.88 * localScale;
      outerBrowW = browBand * smoothstep(0.16, 0.58, xRatio) * 0.88 * localScale;
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

    const rawDepth = Math.max(0, Math.min(1, (maxZ - (pt[2] ?? 0)) / zSpan));
    const radialDepth =
      1 - smoothstep(0.0, face.faceW * 0.65, dist2(nx, ny, noseTip[0], noseTip[1]));
    const depthZ = isOval ? 0.25 : Math.min(1, rawDepth * 0.65 + radialDepth * 0.35);

    const borderMask =
      smoothstep(0.0, 0.06, nx) *
      smoothstep(1.0, 0.94, nx) *
      smoothstep(0.0, 0.06, ny) *
      smoothstep(1.0, 0.94, ny);
    const headW =
      (ny <= chin[1] ? Math.max(0.92, computeHeadWeight(nx, ny)) : computeHeadWeight(nx, ny)) *
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
    faceVerts[fPtr++] = depthZ;
    faceVerts[fPtr++] = mouthX;
  }

  // 3) Explicit watertight multi-row oral interior mesh bridging INNER_UPPER_LIP <-> INNER_LOWER_LIP
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
  }, [effectiveSrc]);

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
      uOpen: loc("uOpen"),
      uWide: loc("uWide"),
      uRound: loc("uRound"),
      uLipPress: loc("uLipPress"),
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
      uNoseFlare: loc("uNoseFlare"),
      uTongueOut: loc("uTongueOut"),
      uHeadRot: loc("uHeadRot"),
      uHeadRoll: loc("uHeadRoll"),
      uForwardLean: loc("uForwardLean"),
      uHeadTranslate: loc("uHeadTranslate"),
      uTorsoBreath: loc("uTorsoBreath"),
      uTorsoSwayX: loc("uTorsoSwayX"),
      uShoulderTilt: loc("uShoulderTilt"),
      uHand0Delta: loc("uHand0Delta"),
      uHand1Delta: loc("uHand1Delta"),
      uHairDelta: loc("uHairDelta"),
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

      const curProfile = getProfile();
      const chosenStyle =
        styleRef.current === "auto" ? (curProfile.animationStyle ?? "auto") : styleRef.current;
      const anime = Boolean(
        chosenStyle === "anime" ||
        (chosenStyle === "auto" &&
          (face.stylized || findCharacterById(curProfile.characterId)?.tags.includes("Anime"))),
      );

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
        animeStyle: anime,
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
      gl.uniform1f(uLoc.uYaw, face.yaw);

      gl.uniform1f(uLoc.uStylized, anime ? 1 : 0);
      gl.uniform1f(uLoc.uWireframe, 0);
      gl.uniform1f(uLoc.uOpen, frame.open * (anime ? 1.08 : 1));
      gl.uniform1f(uLoc.uWide, frame.wide);
      gl.uniform1f(uLoc.uRound, frame.round);
      gl.uniform1f(uLoc.uLipPress, frame.lipPress);
      const smileAmp = anime ? 1.05 : 1;
      gl.uniform1f(uLoc.uSmile, frame.smile * smileAmp);
      gl.uniform1f(uLoc.uSmileR, frame.smileR * smileAmp);
      gl.uniform1f(uLoc.uSmileL, frame.smileL * smileAmp);
      gl.uniform1f(uLoc.uMouthSkewX, frame.mouthSkewX);
      gl.uniform1f(uLoc.uCheekLift, frame.cheekLift);
      gl.uniform1f(uLoc.uInnerBrowY, frame.innerBrowY);
      gl.uniform1f(uLoc.uInnerBrowPinch, frame.innerBrowPinch);
      gl.uniform1f(uLoc.uOuterBrowY, frame.outerBrowY);
      gl.uniform1f(uLoc.uBrowAsym, frame.browAsym);
      gl.uniform1f(uLoc.uSquint, frame.eyeSquint);
      gl.uniform1f(uLoc.uEyeWide, frame.eyeWide);
      gl.uniform1f(uLoc.uNoseFlare, frame.noseFlare);
      gl.uniform1f(uLoc.uTongueOut, frame.tongueOut);

      gl.uniform2f(uLoc.uHeadRot, frame.headYaw * 0.56, frame.headPitch * 0.28);
      gl.uniform1f(uLoc.uHeadRoll, frame.headRoll * 0.52);
      gl.uniform1f(uLoc.uForwardLean, frame.forwardLean);
      gl.uniform2f(uLoc.uHeadTranslate, frame.headTx * 0.85, frame.headTy * 0.85);
      gl.uniform1f(uLoc.uTorsoBreath, frame.torsoBreath);
      gl.uniform1f(uLoc.uTorsoSwayX, frame.torsoSwayX);
      gl.uniform1f(uLoc.uShoulderTilt, frame.shoulderTilt);
      gl.uniform2f(uLoc.uHand0Delta, frame.handShiftX, frame.handShiftY);
      gl.uniform2f(uLoc.uHand1Delta, -frame.handShiftX, frame.handShiftY);
      gl.uniform2f(uLoc.uHairDelta, frame.hairShiftX, frame.hairShiftY);

      gl.uniform2f(uLoc.uREyeCenter, face.rightEyeCenter[0], face.rightEyeCenter[1]);
      gl.uniform2f(uLoc.uREyePupil, face.rightPupil[0], face.rightPupil[1]);
      gl.uniform4f(
        uLoc.uREyeAxis,
        face.rightEyeAxis.ux,
        face.rightEyeAxis.uy,
        face.rightEyeAxis.vx,
        face.rightEyeAxis.vy,
      );
      gl.uniform2f(uLoc.uREyeScale, face.rightEyeCurve.halfW, face.rightEyeCurve.scaleY);
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
      gl.uniform2f(uLoc.uLEyeScale, face.leftEyeCurve.halfW, face.leftEyeCurve.scaleY);
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
      containerRef.current?.setAttribute("data-avatar-style", anime ? "anime" : "human");
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
