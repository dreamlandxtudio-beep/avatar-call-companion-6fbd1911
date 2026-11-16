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
import teethTextureUrl from "@/assets/images/natural_teeth_tongue_1794843941119.jpg";

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
uniform float uSwallow;
uniform float uGenderMale;

uniform vec2 uHeadRot;      // x: yaw-parallax, y: pitch-parallax
uniform float uHeadRoll;
uniform float uForwardLean;
uniform vec2 uHeadTranslate;
uniform float uTorsoBreath;
uniform float uTorsoSwayX;
uniform float uShoulderTilt;
uniform float uShoulderShrug;
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

  // Compute rest-frame normalized face coordinates for consistent dynamic expression effects
  vec2 faceRel = vec2((aPosUv.x - uFaceCenter.x) * aspect, aPosUv.y - uFaceCenter.y);
  vFaceCoord = vec2(
    dot(faceRel, uFaceAxis.xy) / max(0.05, uFaceW * 0.5),
    dot(faceRel, uFaceAxis.zw) / max(0.05, uFaceH * 0.5)
  );

  // 1) Expressive brow shaping (supports symmetric & independent left/right brow lift)
  float asym = side * uBrowAsym;
  float browLift = clamp((aBrowRig.x * (uInnerBrowY + asym * 0.72) + aBrowRig.y * (uOuterBrowY + asym * 1.18)) * 0.026, -0.028, 0.036) * uFaceH;
  float furrowDrop = aBrowRig.x * max(0.0, uInnerBrowPinch) * 0.011 * uFaceH;
  float browPinch = -side * aBrowRig.x * uInnerBrowPinch * 0.028 * uFaceW;
  float orbitLift = aBrowRig.z * (uSquint * 0.010 - uEyeWide * 0.006) * uFaceH;
  p = addAxisDelta(p, uFaceAxis, vec2(browPinch, -(browLift - furrowDrop + orbitLift)), aspect);

  // 2) Realistic nasal wing flare & upward AU9 nose wrinkle/scrunch (+ slight central upper-lip levator lift)
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(
      side * aFaceRig.w * uNoseFlare * 0.008 * uFaceW,
      -(aFaceRig.w * 0.019 + aFaceRig.x * 0.007) * max(0.0, uNoseFlare) * uFaceH
    ),
    aspect
  );

  // 3) Rounded zygomatic cheekbone lift + expressive Duchenne smile per side
  float sideSmile = mix(uSmileR, uSmileL, smoothstep(-0.35, 0.35, side));
  float laughCheek = uCheekLift * 0.013 + max(0.0, sideSmile) * 0.009;
  float cheekSpread = side * aFaceRig.z * (max(0.0, sideSmile) * 0.004 + uCheekLift * 0.003) * uFaceW;
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(cheekSpread, -aFaceRig.z * laughCheek * uFaceH),
    aspect
  );

  // 4) Smooth mouth opening, rounded un-pinched corners & expressive smile arc
  vec2 mouthRel = vec2((p.x - uMouthCenter.x) * aspect, p.y - uMouthCenter.y);
  float mSign = clamp(dot(mouthRel, uMouthAxis.xy) / max(0.01, uMouthW * 0.46), -1.0, 1.0);
  float localSmile = mix(uSmileR, uSmileL, smoothstep(-0.35, 0.35, mSign));
  float lipSealZone = 1.0 - smoothstep(uMouthW * 0.32, uMouthW * 0.86, length(mouthRel));

  float upperLipLift = aFaceRig.x * (uOpen * 0.020 + max(0.0, localSmile) * 0.021 + uRound * 0.006 - uLipPress * 0.009) * uMouthW;
  float cornerWide = mSign * aFaceRig.y * (uWide * 0.016 + localSmile * 0.024 - uRound * 0.028 - uLipPress * 0.009) * uMouthW;
  float cornerLift = abs(mSign) * aFaceRig.y * (localSmile * 0.029 + uWide * 0.004 - uRound * 0.006) * uMouthW;
  float cornerJawFollow = aFaceRig.y * (1.0 - aFaceRig.x) * uOpen * 0.004 * uFaceH;
  float mouthLatSkew = uMouthSkewX * 0.016 * uMouthW * (aFaceRig.y * 0.85 + aFaceRig.x * 0.35);

  float jawOpenDrop = uOpen * 0.074 * uFaceH;
  float jawPressSeal = -jawW * lipSealZone * (uLipPress * 0.008 + uLabiodental * 0.011) * uFaceH;
  float jawDrop = jawOpenDrop * jawW + jawPressSeal;
  float jawInwardTuck = -mSign * jawW * (uRound * 0.005 + uLabiodental * 0.003) * uFaceW;
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

  // 5) Unified 3D head-neck turn with realistic 3/4 perspective depth
  if (headW > 0.001) {
    vec2 skullPivot = mix(uNeckPivot, uFaceCenter, 0.32);
    vec2 rel = vec2((p.x - skullPivot.x) * aspect, p.y - skullPivot.y);
    float cRoll = cos(uHeadRoll * headW);
    float sRoll = sin(uHeadRoll * headW);
    vec2 r = vec2(rel.x * cRoll - rel.y * sRoll, rel.x * sRoll + rel.y * cRoll);

    float hx = clamp(vFaceCoord.x, -1.25, 1.25);
    float hy = clamp(vFaceCoord.y, -1.25, 1.25);
    float r2 = (hx * 0.34) * (hx * 0.34) + (hy * 0.28) * (hy * 0.28);
    float skullZ = sqrt(max(0.14, 1.0 - min(0.86, r2))) * uFaceW * 0.38;
    float pivotZ = uFaceW * 0.14;
    float dz = skullZ - pivotZ;

    float yawRad = clamp(uHeadRot.x * 0.38, -0.28, 0.28) * headW;
    float pitchRad = clamp(uHeadRot.y * 0.26, -0.18, 0.18) * headW;

    float cosY = cos(yawRad);
    float sinY = sin(yawRad);
    float rotX = r.x * mix(1.0, cosY, 0.42) + dz * sinY * 0.90;
    float rotZ = -r.x * sinY * 0.90 + dz * cosY;

    float perspZ = 1.0 + clamp(rotZ / max(0.08, uFaceW) * 0.038, -0.034, 0.038);
    r.x = rotX * perspZ;
    r.y = r.y * perspZ - dz * sin(pitchRad) * 0.72;

    p.x = skullPivot.x + r.x / aspect + uHeadTranslate.x * headW;
    p.y = skullPivot.y + r.y + uHeadTranslate.y * headW;
  }

  // 6) Cohesive anatomical shoulders: keep original shoulder contours locked symmetrically (subtle breath lift only)
  if (uPass < 1.5) {
    float shoulderRidgeW = aHandRig.z;
    float shoulderWeight = max(torsoW, shoulderRidgeW * 0.65);
    float shrugShift = clamp(uShoulderShrug, -0.6, 0.8) * (0.008 * torsoW + 0.009 * shoulderRidgeW) * uFaceH;
    p.y += -uTorsoBreath * 0.0048 * uFaceH * shoulderWeight - shrugShift;

    // Gentle anterior neck hyoid/larynx tissue lift during automatic swallowing reflex
    if (uSwallow > 0.005) {
      float swWave = uSwallow < 0.44
        ? smoothstep(0.0, 0.44, uSwallow)
        : 1.0 - smoothstep(0.44, 1.0, uSwallow);
      float throatMask = smoothstep(0.95, 1.15, vFaceCoord.y) * (1.0 - smoothstep(1.52, 1.85, vFaceCoord.y)) *
        (1.0 - smoothstep(0.10, 0.34, abs(vFaceCoord.x)));
      p.y -= swWave * throatMask * mix(0.006, 0.011, uGenderMale) * uFaceH;
    }
  }

  // Prominent uniform camera approach across whole cranium while torso/shoulders stay anchored
  if (headW > 0.001) {
    float leanVal = clamp(uForwardLean, -0.4, 0.75);
    float craniumZoom = 1.0 + leanVal * headW * 0.095;
    vec2 leanAnchor = mix(uNeckPivot, uFaceCenter, 0.55);
    vec2 relLean = vec2((p.x - leanAnchor.x) * aspect, p.y - leanAnchor.y) * craniumZoom;
    p.x = leanAnchor.x + relLean.x / aspect;
    p.y = leanAnchor.y + relLean.y + leanVal * headW * 0.008 * uFaceH;
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

uniform float uTime;
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
uniform float uLidDroop;
uniform float uTearAmount;
uniform float uEyeWater;
uniform float uEyeRedness;
uniform float uPupilDilate;
uniform float uSweatAmount;
uniform float uBlushAmount;
uniform float uSwallow;
uniform float uGenderMale;
uniform vec2 uFaceCenter;
uniform vec2 uNeckPivot;
uniform vec2 uHeadRot;
uniform vec4 uFaceAxis;
uniform float uFaceW;
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

vec3 sampleSoftTeeth(vec2 uv) {
  vec2 d = vec2(0.011, 0.014);
  vec3 c = texture(uTeethTex, uv).rgb * 0.36;
  c += texture(uTeethTex, clamp(uv + vec2( d.x,  0.0), 0.001, 0.999)).rgb * 0.16;
  c += texture(uTeethTex, clamp(uv + vec2(-d.x,  0.0), 0.001, 0.999)).rgb * 0.16;
  c += texture(uTeethTex, clamp(uv + vec2( 0.0,  d.y), 0.001, 0.999)).rgb * 0.16;
  c += texture(uTeethTex, clamp(uv + vec2( 0.0, -d.y), 0.001, 0.999)).rgb * 0.16;
  return c;
}

vec2 projectToFaceCoord(vec2 uvPt) {
  float asp = max(0.25, uAspect);
  vec2 rel = vec2((uvPt.x - uFaceCenter.x) * asp, uvPt.y - uFaceCenter.y);
  return vec2(
    dot(rel, uFaceAxis.xy) / max(0.05, uFaceW * 0.5),
    dot(rel, uFaceAxis.zw) / max(0.05, uFaceH * 0.5)
  );
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

// Photorealistic human eye: 100% photo-faithful sclera, clean outer crow's-feet lid sweep when blinking, zero open-eye wing artifacts
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
  float outerSign,
  float aspect
) {
  vec2 dUv = vec2((vUv.x - eyeCenter.x) * aspect, vUv.y - eyeCenter.y);
  float halfW = max(0.005, eyeScale.x);
  float scaleY = max(0.004, eyeScale.y);

  float ex = dot(dUv, eyeAxis.xy) / halfW;
  float ey = dot(dUv, eyeAxis.zw) / scaleY;

  float rawTopY = evalLidCurve8(ex, upperA, upperB);
  float rawBotY = evalLidCurve8(ex, lowerA, lowerB);

  // Slit stays strictly within natural eye commissures; closing lid sweeps cleanly over outer lash tail & crow's feet
  bool isOuterSide = (ex * outerSign) > 0.0;
  float maxReach = isOuterSide ? 1.46 : 1.08;
  float exNorm = abs(ex) / maxReach;
  float cornerEnv = 1.0 - smoothstep(0.84, 1.0, exNorm);
  if (cornerEnv < 0.002) return baseRgb;
  float slitEnv = 1.0 - smoothstep(0.92, 1.04, abs(ex));

  float effectiveBlink = max(eyeBlink, clamp(uLidDroop, 0.0, 0.85) * 0.68);
  float blinkClose = clamp(effectiveBlink - uEyeWide * 0.05, 0.0, 1.0);
  float squintClose = clamp(uSquint * 0.24, 0.0, 0.36);
  float lidAction = max(blinkClose, squintClose);

  // Outer wing drop only activates as eyelid actually closes, leaving open eyes 100% authentic
  float outerWing = smoothstep(0.62, 1.02, ex * outerSign) * (1.0 - smoothstep(1.24, 1.46, ex * outerSign));
  float outerMeetDrop = 0.26 * blinkClose * outerWing;

  float meetArch = 0.04 * (1.0 - min(1.0, ex * ex * 0.72)) * blinkClose;
  float meet = mix(rawTopY, rawBotY, 0.91) + meetArch + outerMeetDrop;
  float topY = mix(rawTopY, meet, blinkClose);
  float squintBot = mix(rawBotY, mix(rawBotY, rawTopY, 0.22), squintClose);
  float botY = mix(squintBot, meet + 0.014, blinkClose);

  float feather = max(0.024, fwidth(ey) * 1.25);
  float rawAperture = slitEnv * smoothstep(rawTopY - feather, rawTopY + feather, ey) *
    (1.0 - smoothstep(rawBotY - feather, rawBotY + feather, ey));
  float aperture = slitEnv * smoothstep(topY - feather, topY + feather, ey) *
    (1.0 - smoothstep(botY - feather, botY + feather, ey));

  // Upper tarsal fold below brows tracks down smoothly with real skin texture (no inner double-eyelid outline)
  float archEnv = 1.0 - 0.14 * min(1.0, exNorm * exNorm);
  float upperCrease = rawTopY - 1.12 * archEnv;
  float lowerCrease = rawBotY + 0.68 * archEnv;

  float lidTopFeather = smoothstep(upperCrease - 0.08, upperCrease + 0.24, ey);
  float lidBotFeather = 1.0 - smoothstep(lowerCrease - 0.18, lowerCrease + 0.08, ey);
  float lidZoneMask = cornerEnv * lidTopFeather * lidBotFeather;
  float lidMovingMask = smoothstep(0.004, 0.045, lidAction);
  float effectMask = max(rawAperture, lidZoneMask * lidMovingMask);
  if (effectMask < 0.002) return baseRgb;

  float powerWarp = mix(1.0, 1.06, blinkClose);
  float sourceY;
  if (ey <= topY) {
    float tUp = clamp((ey - upperCrease) / max(0.03, topY - upperCrease), 0.0, 1.0);
    sourceY = mix(upperCrease, rawTopY - 0.024 * lidMovingMask, pow(tUp, powerWarp));
  } else if (ey >= botY) {
    float tDn = clamp((ey - botY) / max(0.03, lowerCrease - botY), 0.0, 1.0);
    sourceY = mix(rawBotY + 0.018 * lidMovingMask, lowerCrease, tDn);
  } else {
    float tAp = clamp((ey - topY) / max(0.02, botY - topY), 0.0, 1.0);
    sourceY = mix(rawTopY - 0.024 * lidMovingMask, rawBotY + 0.018 * lidMovingMask, tAp);
  }

  vec2 sourceDelta = eyeAxis.xy * (ex * halfW) + eyeAxis.zw * (sourceY * scaleY);
  vec2 sourceUv = eyeCenter + vec2(sourceDelta.x / aspect, sourceDelta.y);
  vec3 photoSkin = texture(uPhotoTex, clamp(sourceUv, 0.001, 0.999)).rgb;

  // Delicate natural lash contact crease extending cleanly across the outer lash tail
  float crispLashEdge = exp(-pow((ey - topY) / 0.048, 2.0)) * (0.14 * blinkClose) * cornerEnv;
  float domeHighlight = exp(-pow(ex / 0.68, 2.0)) *
    exp(-pow((ey - mix(rawTopY, topY, 0.52)) / 0.50, 2.0)) *
    blinkClose * 0.024;
  vec3 sculptedLid = clamp(photoSkin * (1.0 - crispLashEdge) + vec3(domeHighlight), 0.0, 1.0);

  // Synchronized ocular travel across both eyes
  float sharedHalfW = max(0.005, min(uREyeScale.x, uLEyeScale.x));
  float sharedScaleY = max(0.004, min(uREyeScale.y, uLEyeScale.y));
  vec2 gazeA = eyeAxis.xy * (uGaze.x * sharedHalfW * 0.28) - eyeAxis.zw * (uGaze.y * sharedScaleY * 0.22);
  vec2 shift = vec2(gazeA.x / aspect, gazeA.y);
  float irisR = max(0.002, irisRadNorm);
  vec2 oldD = vec2((vUv.x - eyePupil.x) * aspect, vUv.y - eyePupil.y);
  vec2 newD = vec2((vUv.x - eyePupil.x - shift.x) * aspect, vUv.y - eyePupil.y - shift.y);

  float lenNew = length(newD);
  float newMask = 1.0 - smoothstep(irisR * 0.78, irisR * 0.96, lenNew);
  float oldDist = length(oldD);
  float oldMask = (1.0 - smoothstep(irisR * 0.84, irisR * 1.38, oldDist)) * (1.0 - newMask);

  // Keep original sclera 100% photo-authentic everywhere untouched; fill vacated iris space from real sclera of this same eye
  vec2 dPupil = vec2((eyePupil.x - eyeCenter.x) * aspect, eyePupil.y - eyeCenter.y);
  float pupilEx = dot(dPupil, eyeAxis.xy) / halfW;
  float sideSplit = smoothstep(-0.78, 0.78, ex - pupilEx);
  float travel = smoothstep(0.0, irisR * 0.04, length(gazeA));
  float scleraClearEx = max(irisR / halfW * 1.18, 0.58);
  float exSampleNeg = clamp(pupilEx - scleraClearEx, -0.84, -0.34);
  float exSamplePos = clamp(pupilEx + scleraClearEx, 0.34, 0.84);
  float exSample = mix(exSampleNeg, exSamplePos, sideSplit);
  float sclTop = evalLidCurve8(exSample, upperA, upperB);
  float sclBot = evalLidCurve8(exSample, lowerA, lowerB);
  float eyMid = mix(sclTop, sclBot, 0.50);
  float eySample = clamp(mix(eyMid, ey, 0.45), sclTop + 0.14, sclBot - 0.14);
  vec2 sclDelta0 = eyeAxis.xy * (exSample * halfW) + eyeAxis.zw * (eySample * scaleY);
  vec2 sclDelta1 = eyeAxis.xy * (clamp(exSample + 0.08, -0.86, 0.86) * halfW) + eyeAxis.zw * (eyMid * scaleY);
  vec3 photoSclera =
    texture(uPhotoTex, clamp(eyeCenter + vec2(sclDelta0.x / aspect, sclDelta0.y), 0.001, 0.999)).rgb * 0.65 +
    texture(uPhotoTex, clamp(eyeCenter + vec2(sclDelta1.x / aspect, sclDelta1.y), 0.001, 0.999)).rgb * 0.35;
  float topScleraAO = mix(0.92, 1.0, smoothstep(rawTopY, rawTopY + 0.24, ey));
  vec3 exactSclera = mix(scleraNegColor, scleraPosColor, sideSplit) * topScleraAO;
  vec3 filledSclera = mix(photoSclera * topScleraAO, exactSclera, 0.26);

  float keepOrig = 1.0 - oldMask * travel;
  vec3 eyeCol = mix(filledSclera, baseRgb, keepOrig);

  // Sample strictly within original cornea disk around eyePupil so moved iris never grabs lashes/eyelids/inner corner
  vec2 localIris = newD / irisR;
  vec2 irisSampleUv = eyePupil + vec2(localIris.x / aspect, localIris.y) * (irisR * 0.90);
  vec3 irisSample = texture(uPhotoTex, clamp(irisSampleUv, 0.001, 0.999)).rgb;
  float irisNorm = lenNew / irisR;
  float irisLuma = dot(irisSample, vec3(0.299, 0.587, 0.114));
  float glintProtect = smoothstep(0.56, 0.78, irisLuma);
  float pupilScale = clamp(1.0 + uPupilDilate * 0.42, 0.80, 1.45);
  float pupilMask = (1.0 - smoothstep(0.22 * pupilScale, 0.34 * pupilScale, irisNorm)) * (1.0 - glintProtect);
  irisSample = mix(irisSample, irisSample * 0.40 + vec3(0.008), pupilMask * 0.64);
  float irisRing = smoothstep(0.35, 0.48, irisNorm) * (1.0 - smoothstep(0.76, 0.90, irisNorm)) * (1.0 - glintProtect);
  float sideCatch = smoothstep(0.08, 0.85, abs(uGaze.x)) * 0.85 + smoothstep(-0.25, 0.75, uGaze.x) * 0.25;
  float irisBoost = clamp(sideCatch, 0.0, 1.0) * irisRing;
  irisSample = mix(vec3(irisLuma), irisSample, 1.0 + 0.10 * irisBoost) * (1.0 + 0.06 * irisBoost);

  float upperLidShadow = mix(0.90, 1.0, smoothstep(topY, topY + 0.24, ey));
  eyeCol = mix(eyeCol, clamp(irisSample, 0.0, 1.0) * upperLidShadow, newMask);

  // Subtle realistic scleral redness/irritation on exposed sclera (crying, drunk, high, pain)
  float redAmt = clamp(uEyeRedness, 0.0, 1.0);
  if (redAmt > 0.01) {
    float scleralArea = (1.0 - newMask) * aperture;
    float innerRedBias = 0.65 + 0.35 * smoothstep(0.22, 0.88, abs(ex));
    vec3 irritated = eyeCol * mix(vec3(1.0), vec3(1.04, 0.72, 0.70), redAmt * 0.58 * innerRedBias) +
      vec3(0.055, 0.0, 0.0) * redAmt * innerRedBias;
    eyeCol = mix(eyeCol, clamp(irritated, 0.0, 1.0), scleralArea * 0.82);
  }

  // Watery tear film shimmer pooling delicately along lower lid margin & bright wet corneal catchlight
  float waterAmt = clamp(uEyeWater, 0.0, 1.0);
  if (waterAmt > 0.01) {
    float rimPool = exp(-pow((ey - (botY - 0.14)) / 0.10, 2.0)) *
      (1.0 - smoothstep(0.68, 0.96, abs(ex))) * aperture;
    eyeCol += vec3(0.24, 0.27, 0.31) * rimPool * waterAmt * 0.75;
    float wetCornea = exp(-pow(length((newD / irisR) - vec2(-0.22, -0.24)) / 0.14, 2.0)) * newMask;
    eyeCol += vec3(0.26, 0.28, 0.30) * wetCornea * waterAmt * 0.65;
  }

  float openInside = aperture * (1.0 - smoothstep(0.86, 0.98, blinkClose));
  vec3 result = mix(sculptedLid, eyeCol, openInside);
  return mix(baseRgb, result, effectMask);
}

// Dedicated Anime / 2D eye model: preserves stylized sclera colors (white, amber, red, etc.) & clean outer wing blink
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
  float outerSign,
  float aspect
) {
  vec2 dUv = vec2((vUv.x - eyeCenter.x) * aspect, vUv.y - eyeCenter.y);
  float halfW = max(0.006, eyeScale.x * 1.08);
  float scaleY = max(0.006, eyeScale.y * 1.24);

  float ex = dot(dUv, eyeAxis.xy) / halfW;
  float ey = dot(dUv, eyeAxis.zw) / scaleY;

  float rawTopY = evalLidCurve8(ex, upperA, upperB) * 1.06;
  float rawBotY = evalLidCurve8(ex, lowerA, lowerB) * 1.06;

  bool isOuterSide = (ex * outerSign) > 0.0;
  float maxReach = isOuterSide ? 1.48 : 1.12;
  float slitMask = 1.0 - smoothstep(0.94, 1.06, abs(ex));
  float midY = (rawTopY + rawBotY) * 0.5 - 0.04;
  float archShape = sqrt(max(0.20, 1.0 - 0.40 * min(1.30, (ex / maxReach) * (ex / maxReach))));
  float spanY = max(0.88, (rawBotY - rawTopY) * 0.66 + 0.64) * archShape;
  float ovalR = length(vec2(ex / maxReach, (ey - midY) / spanY));
  float ovalMask = 1.0 - smoothstep(0.76, 1.08, ovalR);
  if (ovalMask < 0.002) return baseRgb;

  float effectiveBlink = max(eyeBlink, clamp(uLidDroop, 0.0, 0.85) * 0.72);
  float blinkClose = clamp(effectiveBlink - uEyeWide * 0.08, 0.0, 1.0);
  float squintClose = clamp(uSquint * 0.30, 0.0, 0.44);
  float lidAction = max(blinkClose, squintClose);

  float outerWing = smoothstep(0.62, 1.02, ex * outerSign) * (1.0 - smoothstep(1.24, 1.48, ex * outerSign));
  float cornerArch = 0.08 * (1.0 - min(1.0, ex * ex * 0.72)) * blinkClose + 0.26 * blinkClose * outerWing;
  float meet = mix(rawTopY, rawBotY, 0.92) + cornerArch;
  float topY = mix(rawTopY, meet, blinkClose);
  float squintBot = mix(rawBotY, mix(rawBotY, rawTopY, 0.30), squintClose);
  float botY = mix(squintBot, meet + 0.015, blinkClose);

  float feather = max(0.032, fwidth(ey) * 1.4);
  float rawAperture = slitMask * smoothstep(rawTopY - feather, rawTopY + feather, ey) *
    (1.0 - smoothstep(rawBotY - feather, rawBotY + feather, ey));
  float aperture = slitMask * smoothstep(topY - feather, topY + feather, ey) *
    (1.0 - smoothstep(botY - feather, botY + feather, ey));

  float outsideAction = smoothstep(0.018, 0.22, lidAction);
  float effectMask = max(rawAperture, outsideAction) * ovalMask;
  if (effectMask < 0.002) return baseRgb;

  float archEnvelope = 1.0 - 0.14 * min(1.0, ex * ex * 0.65);
  float upperCrease = rawTopY - 1.56 * archEnvelope;
  float lowerCrease = rawBotY + 0.82 * archEnvelope;
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
  vec3 texturedLid = mix(photoLid, mix(photoLid, eyelidSkinColor, 0.50), blinkClose * (1.0 - aperture));

  float lashLine = exp(-pow((ey - topY) / 0.11, 2.0)) * blinkClose * (1.0 - smoothstep(1.05, 1.38, abs(ex)));
  texturedLid = mix(texturedLid, vec3(0.08, 0.06, 0.09), lashLine * 0.82);

  float sharedHalfW = max(0.006, min(uREyeScale.x, uLEyeScale.x) * 1.08);
  float sharedScaleY = max(0.006, min(uREyeScale.y, uLEyeScale.y) * 1.24);
  vec2 gazeA = eyeAxis.xy * (uGaze.x * sharedHalfW * 0.28) - eyeAxis.zw * (uGaze.y * sharedScaleY * 0.24);
  vec2 shift = vec2(gazeA.x / aspect, gazeA.y);
  float irisR = max(0.003, irisRadNorm * 1.02);
  vec2 oldD = vec2((vUv.x - eyePupil.x) * aspect, vUv.y - eyePupil.y);
  vec2 newD = vec2((vUv.x - eyePupil.x - shift.x) * aspect, vUv.y - eyePupil.y - shift.y);
  float lenNew = length(newD);
  float newMask = 1.0 - smoothstep(irisR * 0.80, irisR * 0.98, lenNew);
  float oldMask = (1.0 - smoothstep(irisR * 0.88, irisR * 1.38, length(oldD))) * (1.0 - newMask);

  vec2 dPupil = vec2((eyePupil.x - eyeCenter.x) * aspect, eyePupil.y - eyeCenter.y);
  float pupilEx = dot(dPupil, eyeAxis.xy) / halfW;
  float sideMix = smoothstep(-0.75, 0.75, ex - pupilEx);
  float travel = smoothstep(0.0, irisR * 0.04, length(gazeA));
  float sclEx = clamp(mix(pupilEx - 0.62, pupilEx + 0.62, sideMix), -0.84, 0.84);
  float sclTop = evalLidCurve8(sclEx, upperA, upperB) * 1.06;
  float sclBot = evalLidCurve8(sclEx, lowerA, lowerB) * 1.06;
  float sclEy = clamp(mix(mix(sclTop, sclBot, 0.50), ey, 0.45), sclTop + 0.14, sclBot - 0.14);
  vec2 sclDelta = eyeAxis.xy * (sclEx * halfW) + eyeAxis.zw * (sclEy * scaleY);
  vec3 photoSclera = texture(uPhotoTex, clamp(eyeCenter + vec2(sclDelta.x / aspect, sclDelta.y), 0.001, 0.999)).rgb;
  vec3 sideSclera = mix(scleraNegColor, scleraPosColor, sideMix);
  vec3 cleanSclera = mix(photoSclera, sideSclera, 0.28) * mix(0.94, 1.0, smoothstep(rawTopY, mix(rawTopY, rawBotY, 0.38), ey));
  vec3 eyeCol = mix(cleanSclera, baseRgb, 1.0 - oldMask * travel);
  vec2 localIris = newD / irisR;
  vec2 irisSampleUv = eyePupil + vec2(localIris.x / aspect, localIris.y) * (irisR * 0.92);
  vec3 irisSample = texture(uPhotoTex, clamp(irisSampleUv, 0.001, 0.999)).rgb;
  eyeCol = mix(eyeCol, irisSample, newMask);

  float waterAmt = clamp(uEyeWater, 0.0, 1.0);
  if (waterAmt > 0.01) {
    float rimPool = exp(-pow((ey - (botY - 0.14)) / 0.12, 2.0)) * aperture;
    eyeCol += vec3(0.28, 0.30, 0.34) * rimPool * waterAmt * 0.72;
  }

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

  // 3) Crisp glabella "11" frown creases between brows + brow-bridge / bunny-line AU9 scrunch
  float furrowAct = smoothstep(0.06, 0.72, uInnerBrowPinch);
  float glabellaZone = exp(-pow((absX - 0.072) / 0.032, 2.0)) *
    smoothstep(-0.58, -0.44, fc.y) * (1.0 - smoothstep(-0.22, -0.12, fc.y));
  float glabellaShadow = glabellaZone * furrowAct;

  float noseScrunchAct = smoothstep(0.10, 0.78, max(0.0, uNoseFlare));
  float bunnyLine = exp(-pow(((absX - 0.13) * 0.80 + (fc.y + 0.02) * 0.60) / 0.036, 2.0)) *
    smoothstep(-0.16, -0.05, fc.y) * (1.0 - smoothstep(0.12, 0.24, fc.y));
  float bunnyZone = exp(-pow((absX - 0.11) / 0.048, 2.0)) *
    exp(-pow((fc.y + 0.03) / 0.09, 2.0));
  float bunnyShadow = (bunnyZone * 0.65 + bunnyLine * 0.85) * noseScrunchAct;

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
    (nasoShadow * 0.07 + crowShadow * 0.06 + glabellaShadow * 0.18 + bunnyShadow * 0.13 +
     foreheadShadow * 0.06 + cornerDimple * 0.07) * skinMask * skinToneMask,
    0.0,
    0.16
  );
  vec3 shaded = col * mix(vec3(1.0), vec3(0.80, 0.72, 0.69), totalCrease);
  return clamp(shaded, 0.0, 1.0);
}

// Helper: evaluates 3D refractive glassmorphism liquid droplet in face coordinates (teardrop or sweat bead)
vec3 blendGlassDroplet(
  vec3 col,
  vec2 fc,
  vec2 dropCenter,
  vec2 dropSize,
  float alpha,
  bool verticalTrail
) {
  if (alpha < 0.01) return col;
  vec2 rel = (fc - dropCenter) / max(vec2(0.004), dropSize);
  // Subtle wet skin track above a sliding teardrop
  if (verticalTrail && rel.y < 0.0 && rel.y > -4.2) {
    float trackW = exp(-pow(rel.x / 0.58, 2.0)) * smoothstep(-4.2, -0.4, rel.y);
    float trailSpec = exp(-pow((rel.x + 0.12) / 0.28, 2.0)) * trackW;
    col += vec3(0.11, 0.12, 0.14) * trailSpec * alpha * 0.55;
  }

  // Slight organic teardrop taper at top, rounder bottom dome
  float taper = 1.0 - clamp(-rel.y * 0.28, -0.18, 0.34);
  vec2 shaped = vec2(rel.x / taper, rel.y);
  float r = length(shaped);
  if (r > 1.55) return col;

  // Soft realistic skin contact shadow directly beneath the fluid dome
  vec2 shRel = (fc - (dropCenter + vec2(dropSize.x * 0.18, dropSize.y * 0.34))) / max(vec2(0.004), dropSize * 1.15);
  float shadowDist = length(shRel);
  float shadowMask = (1.0 - smoothstep(0.35, 1.18, shadowDist)) * smoothstep(0.65, 1.15, r);
  col = mix(col, col * vec3(0.76, 0.74, 0.76), shadowMask * alpha * 0.45);

  float dropMask = (1.0 - smoothstep(0.84, 1.02, r)) * alpha;
  if (dropMask < 0.004) return col;

  float nz = sqrt(max(0.02, 1.0 - min(0.98, r * r)));
  vec2 normal = shaped / max(0.15, sqrt(dot(shaped, shaped) + nz * nz));
  float aspect = max(0.25, uAspect);
  vec2 refFace = -normal * dropSize * 0.52;
  vec2 refUv = vec2(
    (uFaceAxis.x * refFace.x * uFaceW * 0.5 + uFaceAxis.z * refFace.y * uFaceH * 0.5) / aspect,
    uFaceAxis.y * refFace.x * uFaceW * 0.5 + uFaceAxis.w * refFace.y * uFaceH * 0.5
  );
  vec3 refracted = texture(uPhotoTex, clamp(vUv + refUv, 0.001, 0.999)).rgb;

  // Subtle glassmorphic cool tint + Fresnel rim + crisp key-light studio dot
  float fresnel = pow(1.0 - nz, 2.2);
  vec3 glassBody = mix(refracted * 1.05, vec3(0.88, 0.94, 0.99), 0.16 + fresnel * 0.32);
  float mainSpec = exp(-pow(length(shaped - vec2(-0.34, -0.36)) / 0.24, 2.0));
  float rimBounce = exp(-pow(length(shaped - vec2(0.28, 0.48)) / 0.36, 2.0)) * 0.42;
  glassBody += vec3(0.94, 0.97, 1.0) * (mainSpec * 0.88 + rimBounce * 0.35);

  return mix(col, clamp(glassBody, 0.0, 1.0), dropMask);
}

// Physical overlays: cheeks/nose blush (or pale), clammy nervous sweat sheen & beads, and 3D glassmorphism teardrops
vec3 applyPhysicalEffects(vec3 col, vec2 fc) {
  float faceBound = (1.0 - smoothstep(0.92, 1.18, abs(fc.x))) * (1.0 - smoothstep(0.92, 1.18, abs(fc.y)));
  if (faceBound < 0.01) return col;

  // 1) Natural upper-cheekbone blush (never below nose!) or gentle lifelike fear/sick chill
  if (abs(uBlushAmount) > 0.01) {
    float cheekL = exp(-(pow((fc.x - 0.43) / 0.24, 2.0) + pow((fc.y - 0.09) / 0.15, 2.0)));
    float cheekR = exp(-(pow((fc.x + 0.43) / 0.24, 2.0) + pow((fc.y - 0.09) / 0.15, 2.0)));
    float blushZone = clamp((cheekL + cheekR) * faceBound, 0.0, 1.0);
    if (uBlushAmount > 0.0) {
      vec3 flushCol = mix(col, col * vec3(1.06, 0.68, 0.69) + vec3(0.14, 0.014, 0.018), 0.86);
      col = mix(col, clamp(flushCol, 0.0, 1.0), blushZone * clamp(uBlushAmount, 0.0, 1.0) * 0.72);
    } else {
      float paleAmt = clamp(-uBlushAmount, 0.0, 1.0);
      float lum = dot(col, vec3(0.299, 0.587, 0.114));
      vec3 paleCol = clamp(vec3(lum * 0.96 + 0.025, lum * 0.95 + 0.022, lum * 0.96 + 0.025), 0.0, 1.0);
      col = mix(col, mix(col, paleCol, 0.46), faceBound * paleAmt * 0.52);
    }
  }

  // 2) Nervous / sweating sheen & droplets STRICTLY on upper forehead & outer temples (never between/inside eyes!)
  if (uSweatAmount > 0.01) {
    float sw = clamp(uSweatAmount, 0.0, 1.0);
    float eyeBoxClear = smoothstep(0.46, 0.64, length(vec2(fc.x / 0.58, (fc.y + 0.24) / 0.26)));
    float foreheadBand = smoothstep(-0.46, -0.58, fc.y) * (1.0 - smoothstep(-0.88, -0.98, fc.y));
    float templeBand = smoothstep(0.56, 0.70, abs(fc.x)) * (1.0 - smoothstep(-0.18, -0.02, fc.y));
    float browSheen = exp(-(pow(fc.x / 0.52, 2.0) + pow((fc.y + 0.64) / 0.16, 2.0))) * foreheadBand;
    float templeSheen = exp(-(pow((abs(fc.x) - 0.66) / 0.14, 2.0) + pow((fc.y + 0.42) / 0.16, 2.0))) * templeBand;
    float gloss = clamp((browSheen + templeSheen * 0.85) * faceBound * eyeBoxClear, 0.0, 1.0);
    col = clamp(col + vec3(0.15, 0.16, 0.18) * gloss * sw, 0.0, 1.0);

    // 3 gentle animated sweat beads strictly along upper forehead & lateral temples
    for (int sIdx = 0; sIdx < 3; sIdx++) {
      float fi = float(sIdx);
      float prog = fract(uTime * (0.14 + fi * 0.04) + fi * 0.37);
      float fade = smoothstep(0.0, 0.14, prog) * (1.0 - smoothstep(0.76, 1.0, prog));
      float sx = sIdx == 0 ? 0.68 : (sIdx == 1 ? -0.66 : 0.24);
      float startY = sIdx == 2 ? -0.78 : -0.60;
      float sy = startY + prog * (sIdx == 2 ? 0.24 : 0.32);
      col = blendGlassDroplet(
        col,
        fc,
        vec2(sx, sy),
        vec2(0.028, 0.038),
        sw * fade * 0.90 * eyeBoxClear,
        true
      );
    }
  }

  // 3) Crying 3D glassmorphism teardrops anchored directly to lower lash margins / outer eye corners
  if (uTearAmount > 0.01) {
    float tr = clamp(uTearAmount, 0.0, 1.0);
    vec2 rEyeFc = projectToFaceCoord(uREyeCenter);
    vec2 lEyeFc = projectToFaceCoord(uLEyeCenter);
    float rHalfW = uREyeScale.x / max(0.05, uFaceW * 0.5);
    float rScaleY = uREyeScale.y / max(0.05, uFaceH * 0.5);
    float lHalfW = uLEyeScale.x / max(0.05, uFaceW * 0.5);
    float lScaleY = uLEyeScale.y / max(0.05, uFaceH * 0.5);
    float rRimOffsetY = ((uREyeLowerA.w + uREyeLowerB.x) * 0.5) * rScaleY;
    float lRimOffsetY = ((uLEyeLowerA.w + uLEyeLowerB.x) * 0.5) * lScaleY;

    for (int side = 0; side < 2; side++) {
      vec2 eyeFc = side == 0 ? rEyeFc : lEyeFc;
      float eHalfW = side == 0 ? rHalfW : lHalfW;
      float eRimY = side == 0 ? rRimOffsetY : lRimOffsetY;
      float outerDir = side == 0 ? -1.0 : 1.0;

      for (int d = 0; d < 2; d++) {
        float fd = float(d);
        float speed = 0.20 + fd * 0.06;
        float phase = float(side) * 0.31 + fd * 0.53;
        float prog = fract(uTime * speed + phase);
        float fade = smoothstep(0.0, 0.12, prog) * (1.0 - smoothstep(0.76, 1.0, prog));
        // Emit right along the outer-mid lower lash rim & drip gently down over cheekbone (never nose!)
        float startX = eyeFc.x + outerDir * eHalfW * (d == 0 ? 0.26 : 0.48);
        float curveX = startX + outerDir * (0.045 * sin(prog * 3.14159));
        float dropY = (eyeFc.y + eRimY + 0.025) + prog * 0.48;
        vec2 dropSize = d == 0 ? vec2(0.036, 0.048) : vec2(0.028, 0.038);
        col = blendGlassDroplet(col, fc, vec2(curveX, dropY), dropSize, tr * fade * 0.96, true);
      }
    }
  }

  return col;
}

// Cervical anatomy: subtle SCM sternocleidomastoid tendons on head turns/upward gaze & larynx swallow reflex ("Nuez de Adán")
vec3 applyNeckAnatomy(vec3 col, vec2 fc) {
  if (uSwallow < 0.005 && abs(uHeadRot.x) < 0.14 && uHeadRot.y < 0.03) return col;
  vec2 neckFc = projectToFaceCoord(uNeckPivot);
  float neckTop = 1.02;
  float neckBot = max(neckTop + 0.36, neckFc.y + 0.10);
  if (fc.y < neckTop - 0.08 || fc.y > neckBot + 0.08) return col;

  float tNeck = clamp((fc.y - neckTop) / max(0.24, neckBot - neckTop), 0.0, 1.0);
  float neckCenterX = mix(0.0, neckFc.x, tNeck);
  float nx = fc.x - neckCenterX;

  float neckMask =
    smoothstep(0.0, 0.16, tNeck) *
    (1.0 - smoothstep(0.84, 1.04, tNeck)) *
    (1.0 - smoothstep(0.46, 0.72, abs(nx)));
  if (neckMask < 0.01) return col;

  // Skin-tone gate so shirt collars or clothing never receive neck skin shading
  float luma = dot(col, vec3(0.299, 0.587, 0.114));
  float warmSkin = clamp((col.r - col.b) * 2.4 + 0.35, 0.0, 1.0);
  float skinMask = smoothstep(0.18, 0.36, luma) * smoothstep(0.15, 0.45, warmSkin);
  if (skinMask < 0.01) return col;

  // 1) Bilateral sternocleidomastoid (SCM) V-cords tensing on upward gaze, turns, or swallow
  float upTense = smoothstep(0.03, 0.38, uHeadRot.y) * 0.55;
  float rTurn = smoothstep(0.12, 0.52, -uHeadRot.x);
  float lTurn = smoothstep(0.12, 0.52, uHeadRot.x);
  float scmLineX = mix(0.56, 0.14, tNeck);
  float rCord = exp(-pow((nx + scmLineX) / 0.055, 2.0)) * (upTense + lTurn * 0.75 + uSwallow * 0.25);
  float lCord = exp(-pow((nx - scmLineX) / 0.055, 2.0)) * (upTense + rTurn * 0.75 + uSwallow * 0.25);
  float scmRidge = clamp(rCord + lCord, 0.0, 1.0);
  float scmHollow = exp(-pow((abs(nx) - (scmLineX - 0.08)) / 0.055, 2.0)) * clamp(upTense + (rTurn + lTurn) * 0.5, 0.0, 1.0);
  float sternalNotch = exp(-pow(nx / 0.09, 2.0)) * exp(-pow((tNeck - 0.90) / 0.11, 2.0)) * clamp(upTense + (rTurn + lTurn) * 0.4 + uSwallow * 0.35, 0.0, 1.0);

  // 2) Swallowing wave: thyroid cartilage / Adam's apple ascends crisply and settles cleanly
  float swShade = 0.0;
  float swHighlight = 0.0;
  if (uSwallow > 0.005) {
    float swLift = uSwallow < 0.42
      ? smoothstep(0.0, 0.42, uSwallow)
      : 1.0 - smoothstep(0.42, 1.0, uSwallow);
    float swPulse = sin(clamp(uSwallow, 0.0, 1.0) * 3.14159);
    float appleY = 0.46 - swLift * 0.23;
    float promW = mix(0.11, 0.085, uGenderMale);
    float promH = mix(0.11, 0.078, uGenderMale);
    float promAmp = mix(0.45, 1.0, uGenderMale) * swPulse;
    swHighlight = exp(-pow(nx / promW, 2.0)) * exp(-pow((tNeck - (appleY - 0.03)) / promH, 2.0)) * promAmp;
    swShade = exp(-pow(nx / (promW * 1.15), 2.0)) * exp(-pow((tNeck - (appleY + 0.06)) / promH, 2.0)) * promAmp;
  }

  float mask = neckMask * skinMask;
  col *= 1.0 - mask * (scmHollow * 0.055 + sternalNotch * 0.075 + swShade * 0.11);
  col += mask * (scmRidge * 0.032 + swHighlight * 0.055);
  return clamp(col, 0.0, 1.0);
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
      smoothstep(0.0, 0.028, vMouthLipT) *
      smoothstep(1.0, 0.965, vMouthLipT) *
      (1.0 - smoothstep(0.90, 0.995, absX));

    float rigidX = vRigidMouth.x;
    float absRigidX = abs(rigidX);
    float mouthY = vMouthPlaneY;
    float lowerPlaneY = mouthY - vJawDropMouthW;

    if (uStylized > 0.5) {
      vec3 cavityCol = mix(vec3(0.24, 0.06, 0.10), uLipColor * 0.34, 0.38);
      float animeUpperLift = smoothstep(0.06, 0.65, uOpen) * 0.005 + clamp(uSmile, 0.0, 1.0) * 0.004;
      float animeUpperTop = uRestMouthTopY - 0.028 - animeUpperLift;
      float animeUpperBot = uRestMouthTopY + 0.022 - animeUpperLift;
      float upperTeethBar =
        smoothstep(animeUpperTop - 0.007, animeUpperTop + 0.007, mouthY) *
        (1.0 - smoothstep(animeUpperBot - 0.008, animeUpperBot + 0.005, mouthY)) *
        (1.0 - smoothstep(0.38, 0.48, absRigidX));
      float animeLowerTop = uRestMouthBotY - 0.026;
      float animeLowerBot = uRestMouthBotY + 0.014;
      float lowerTeethBar =
        smoothstep(animeLowerTop - 0.005, animeLowerTop + 0.007, lowerPlaneY) *
        (1.0 - smoothstep(animeLowerBot - 0.007, animeLowerBot + 0.005, lowerPlaneY)) *
        smoothstep(0.08, 0.22, uOpen) *
        (1.0 - smoothstep(0.36, 0.46, absRigidX));
      float tongueShift = clamp(uTongueOut, 0.0, 1.0) * 0.018;
      float tongueArc = smoothstep(0.42 - tongueShift * 4.0, 0.58, vMouthLipT) *
        (1.0 - smoothstep(0.24, 0.40, absX));
      vec3 animeOral = mix(cavityCol, vec3(0.76, 0.34, 0.42), tongueArc * 0.85);
      animeOral = mix(animeOral, vec3(0.95, 0.94, 0.95), upperTeethBar * 0.90);
      animeOral = mix(animeOral, vec3(0.90, 0.89, 0.91), lowerTeethBar * 0.85);
      float cornerDark = 1.0 - 0.78 * smoothstep(0.46, 0.90, absX);
      fragColor = vec4(animeOral * cornerDark, openMask * lipRimFeather * fgAlpha);
      return;
    }

    // Soft-focus photographic dental arch matching portrait lens sharpness + dark buccal corridors cleanly into corners
    float ax = clamp(rigidX - uYaw * 0.065, -0.58, 0.58);
    float absAx = abs(ax);
    float archCurve = min(1.0, absAx / 0.50);
    float biteCurve = 0.032 * ax * ax;

    vec3 deepRedMucosa = vec3(0.055, 0.011, 0.015) + uLipColor * vec3(0.045, 0.010, 0.012);
    float throatDepth = 1.0 - 0.60 * (1.0 - abs(vMouthLipT - 0.46) * 1.35) * (1.0 - absX * 0.65);
    vec3 oralCol = deepRedMucosa * clamp(throatDepth, 0.24, 1.0);

    float texU = clamp(0.5 + ax * 0.92, 0.06, 0.94);

    // Upper teeth row with softly dimmed upper gumline tucked beneath upper lip shadow
    float upperMicroLift = smoothstep(0.06, 0.65, uOpen) * 0.004 + clamp(uSmile, 0.0, 1.0) * 0.003;
    float upperToothH = mix(0.064, 0.050, archCurve);
    float upperEdgeY = (uRestMouthTopY + 0.019) - biteCurve * 0.35 - upperMicroLift * 0.18;
    float upperGumY = upperEdgeY - upperToothH;
    float upperFrac = clamp((mouthY - upperGumY) / max(0.016, upperToothH), 0.0, 1.0);

    // Lower teeth row clearly visible in the lower dental arch during open speech/smiles
    float lowerToothH = mix(0.050, 0.038, archCurve);
    float lowerTipY = uRestMouthBotY - 0.015 + biteCurve * 0.25;
    float lowerGumY = lowerTipY + lowerToothH;
    float lowerFrac = clamp((lowerPlaneY - lowerTipY) / max(0.014, lowerToothH), 0.0, 1.0);

    vec3 photoUpperTex = sampleSoftTeeth(vec2(texU, mix(0.16, 0.345, upperFrac)));
    vec3 photoLowerTex = sampleSoftTeeth(vec2(texU, mix(0.755, 0.920, lowerFrac)));
    float enamelRef = max(0.66, dot(uEnamelColor, vec3(0.299, 0.587, 0.114)));
    float uLuma = dot(photoUpperTex, vec3(0.299, 0.587, 0.114));
    photoUpperTex = mix(vec3(uLuma), photoUpperTex, 0.62);
    float lLuma = dot(photoLowerTex, vec3(0.299, 0.587, 0.114));
    photoLowerTex = mix(vec3(lLuma), photoLowerTex, 0.62);

    float uArchDepth = 1.0 - 0.54 * pow(archCurve, 1.3);
    float upperGumShadow = smoothstep(0.04, 0.32, upperFrac);
    float uLipShadow = mix(0.56, 1.0, smoothstep(0.18, 0.62, upperFrac));
    vec3 upperToothCol = clamp(photoUpperTex * (enamelRef / 0.88) * uArchDepth * uLipShadow, 0.0, 1.0);
    upperToothCol = mix(deepRedMucosa * 0.72, upperToothCol, mix(0.45, 1.0, upperGumShadow));

    float lowerGumShadow = 1.0 - smoothstep(0.72, 0.96, lowerFrac);
    float lLipShadow = mix(1.0, 0.66, smoothstep(0.72, 1.0, lowerFrac));
    vec3 lowerToothCol = clamp(photoLowerTex * (enamelRef / 0.90) * 0.88 * uArchDepth * lLipShadow, 0.0, 1.0);
    lowerToothCol = mix(deepRedMucosa * 0.72, lowerToothCol, mix(0.50, 1.0, lowerGumShadow));

    // Anatomical dorsal tongue strictly confined to central dental arch (never bleeds pink into mouth corners!)
    float tongueAdv = clamp(uTongueOut, 0.0, 1.0);
    float openTongueRise = smoothstep(0.14, 0.64, uOpen) * 0.034;
    float tongueLift = clamp(uRound, 0.0, 1.0) * 0.008 + tongueAdv * 0.034;
    float tongueTopY = lowerTipY - 0.014 - openTongueRise + 0.14 * rigidX * rigidX - tongueLift;
    float tongueBotY = lowerGumY + 0.016 + tongueAdv * 0.024;
    float tongueLatMask = 1.0 - smoothstep(0.20 + tongueAdv * 0.04, 0.34 + tongueAdv * 0.04, absRigidX);
    float tongueVertMask =
      smoothstep(tongueTopY - 0.014, tongueTopY + 0.012, lowerPlaneY) *
      (1.0 - smoothstep(tongueBotY - 0.012, tongueBotY + 0.012, lowerPlaneY));
    float tongueShow = max(smoothstep(0.12, 0.28, uOpen), smoothstep(0.04, 0.18, tongueAdv));
    float tongueMask = tongueLatMask * tongueVertMask * tongueShow;
    float tNormY = clamp((lowerPlaneY - tongueTopY) / max(0.022, tongueBotY - tongueTopY), 0.0, 1.0);
    vec3 photoTongueTex = sampleSoftTeeth(vec2(clamp(0.5 + rigidX * 0.85, 0.22, 0.78), mix(0.56, 0.76, tNormY)));
    float tongueGroove = 1.0 - 0.12 * exp(-pow(rigidX / 0.07, 2.0));
    float tongueSheen = exp(-pow((tNormY - 0.34) / 0.22, 2.0)) * (1.0 - smoothstep(0.06, 0.28, absRigidX)) * 0.06;
    float tongueShade = mix(0.74, 0.96, smoothstep(0.0, 0.44, tNormY)) * tongueGroove;
    vec3 tongueCol = clamp(photoTongueTex * tongueShade + vec3(tongueSheen), 0.0, 1.0);

    float yFeather = max(0.012, fwidth(mouthY) * 2.5);
    float upperZone =
      smoothstep(upperGumY - yFeather, upperGumY + yFeather, mouthY) *
      (1.0 - smoothstep(upperEdgeY - yFeather, upperEdgeY + yFeather, mouthY)) *
      (1.0 - smoothstep(0.42, 0.56, absAx));
    float lowerZone =
      smoothstep(lowerTipY - yFeather, lowerTipY + yFeather, lowerPlaneY) *
      (1.0 - smoothstep(lowerGumY - yFeather, lowerGumY + yFeather, lowerPlaneY)) *
      smoothstep(upperEdgeY + 0.002, upperEdgeY + 0.014, mouthY) *
      smoothstep(0.06, 0.20, uOpen) *
      (1.0 - smoothstep(0.38, 0.52, absAx));

    oralCol = mix(oralCol, tongueCol, tongueMask * (1.0 - tongueAdv * 0.15));
    oralCol = mix(oralCol, upperToothCol, upperZone * 0.90);
    oralCol = mix(oralCol, lowerToothCol, lowerZone * 0.88 * (1.0 - tongueAdv * 0.78));
    if (tongueAdv > 0.08) {
      oralCol = mix(oralCol, tongueCol, tongueMask * smoothstep(0.08, 0.36, tongueAdv) * 0.92);
    }

    // Soft inner-vermilion border blend strictly outside mouth corners + deep dark buccal shadow inside corners
    float cornerSuppress = 1.0 - smoothstep(0.54, 0.82, absX);
    float innerRimTint = ((1.0 - smoothstep(0.02, 0.14, vMouthLipT)) + smoothstep(0.84, 0.98, vMouthLipT)) * cornerSuppress;
    oralCol = mix(oralCol, uLipColor * 0.52, clamp(innerRimTint, 0.0, 1.0) * 0.36);
    float topAO = mix(0.64, 1.0, smoothstep(0.0, 0.12, vMouthLipT));
    float botAO = mix(0.68, 1.0, smoothstep(1.0, 0.82, vMouthLipT));
    float sideAO = 1.0 - 0.86 * smoothstep(0.42, 0.84, absX);
    float cavityLight = mix(0.74, 1.0, smoothstep(0.02, 0.22, uOpen));
    vec3 warmShadowTint = vec3(0.032, 0.007, 0.009);
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
      -1.0,
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
      1.0,
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
  }

  col = applyPhysicalEffects(col, vFaceCoord);
  col = applyNeckAnatomy(col, vFaceCoord);

  // Inward foreground color decontamination: pull pure interior hair/skin RGB into semi-transparent edges so zero dark halo appears over sharp backgrounds
  if (fgAlpha > 0.02 && fgAlpha < 0.72) {
    vec2 dirIn = uFaceCenter - uvSafe;
    float dLen = length(dirIn);
    if (dLen > 0.001) {
      vec2 stepDir = dirIn / dLen;
      vec2 uvIn1 = clamp(uvSafe + stepDir * uBgTexel * 1.8, 0.001, 0.999);
      vec2 uvIn2 = clamp(uvSafe + stepDir * uBgTexel * 3.6, 0.001, 0.999);
      float a1 = texture(uFgMaskTex, uvIn1).r;
      float a2 = texture(uFgMaskTex, uvIn2).r;
      vec2 pureUv = a2 > a1 ? uvIn2 : uvIn1;
      float pureAlpha = max(a1, a2);
      if (pureAlpha > fgAlpha + 0.06) {
        vec3 pureInteriorRgb = texture(uPhotoTex, pureUv).rgb;
        float fringe = 1.0 - smoothstep(0.02, 0.72, fgAlpha);
        col = mix(col, pureInteriorRgb, fringe * 0.85);
      }
    }
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
        // Smooth matte transition across fine hair strands so non-blurred backgrounds blend cleanly without clipped edges
        const segSmooth = smoothstep(0.16, 0.54, seg);
        fg = Math.max(faceCore, segSmooth);
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

  // Erase 4px cleanly past low-alpha hair & neck contour so sharp unblurred backgrounds never leave a stale shadow ring
  const holeMask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let inHole = false;
      for (let dy = -4; dy <= 4 && !inHole; dy++) {
        const sy = y + dy;
        if (sy < 0 || sy >= H) continue;
        for (let dx = -4; dx <= 4; dx++) {
          const sx = x + dx;
          if (sx < 0 || sx >= W) continue;
          if (dx * dx + dy * dy <= 18 && alphaArr[sy * W + sx]! >= 0.08) {
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
  const neckFadeEndY = Math.min(
    0.96,
    Math.max(face.neckBase[1] + face.faceH * 0.04, chin[1] + face.faceH * 0.42),
  );

  // Full segmented hair & cranium silhouette locks solidly above chin; below chin restricts strictly to cervical neck column so shoulders never warp
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
    const upperHairSeg = ny <= chin[1] + face.faceH * 0.04 && sampleHairSeg(nx, ny) > 0.35 ? 1 : 0;
    const belowChinT = ny > chin[1] ? smoothstep(chin[1], face.neckBase[1], ny) : 0;
    const neckAxisX = face.center[0] + belowChinT * (face.neckBase[0] - face.center[0]);
    const neckLatFade =
      1 - smoothstep(face.faceW * 0.4, face.faceW * 0.66, Math.abs((nx - neckAxisX) * aspect));
    const lateralMask = 1 - belowChinT * (1 - neckLatFade);
    const radial = Math.max(oval, upperHairSeg) * lateralMask;
    const neckFade = ny > neckFadeStartY ? 1 - smoothstep(neckFadeStartY, neckFadeEndY, ny) : 1;
    return radial * neckFade;
  };

  // Bilateral shoulder/clavicle ridge weights using detected neckBase -> rightShoulder / leftShoulder
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
      // Smooth continuous blend between upper lip and lower mandible at mouth corners so wide openings stay rounded
      jawW = 0.36 * localScale;
      upperLipW = 0.24 * localScale;
    } else if (isExplicitUpper || inUpperVermilion) {
      const cornerTaper = smoothstep(0.76, 1.0, Math.abs(mouthX));
      jawW = 0.36 * localScale * cornerTaper * 0.45;
      upperLipW = (0.24 + 0.76 * lipCenterArch) * localScale;
    } else if (isExplicitLower || inLowerVermilion) {
      // Cohesive plateau across entire lower lip vermilion tapering cleanly into rounded outer corners
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

    // Include dorsal nasal bridge, nasal wings, and upper-lip levators so AU9 visibly lifts & wrinkles the nose
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

    // Cohesive ribbon eyebrow weight: uniform plateau across upper & lower brow rows so brows keep true thickness
    let innerBrowW = 0;
    let outerBrowW = 0;
    const isRBrow = rightBrowSet.has(i);
    const isLBrow = leftBrowSet.has(i);
    const box = relFaceX < 0 ? rBrowBox : lBrowBox;
    const xRatio = Math.min(1, Math.abs(relFaceX) / Math.max(0.02, face.faceW * 0.36));

    if (isRBrow || isLBrow) {
      // Uniform 1.0 plateau vertically across the whole brow hair band; split horizontally inner vs outer
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
        // Taper across the upper-lid brow bone space below the brow so lifting reveals more brow bone cleanly
        browBand = 1 - smoothstep(0.0, Math.max(0.012, box.eyeTopY - box.maxY), ny - box.maxY);
      }
      innerBrowW = browBand * (1 - smoothstep(0.18, 0.58, xRatio)) * 0.9 * localScale;
      outerBrowW = browBand * smoothstep(0.16, 0.58, xRatio) * 0.9 * localScale;
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
      smoothstep(0.0, 0.018, nx) *
      smoothstep(1.0, 0.982, nx) *
      smoothstep(0.0, 0.018, ny) *
      smoothstep(1.0, 0.982, ny);
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
      uTime: loc("uTime"),
      uOpen: loc("uOpen"),
      uWide: loc("uWide"),
      uRound: loc("uRound"),
      uLipPress: loc("uLipPress"),
      uLabiodental: loc("uLabiodental"),
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
      uLidDroop: loc("uLidDroop"),
      uNoseFlare: loc("uNoseFlare"),
      uTongueOut: loc("uTongueOut"),
      uTearAmount: loc("uTearAmount"),
      uEyeWater: loc("uEyeWater"),
      uEyeRedness: loc("uEyeRedness"),
      uPupilDilate: loc("uPupilDilate"),
      uSweatAmount: loc("uSweatAmount"),
      uBlushAmount: loc("uBlushAmount"),
      uSwallow: loc("uSwallow"),
      uGenderMale: loc("uGenderMale"),
      uHeadRot: loc("uHeadRot"),
      uHeadRoll: loc("uHeadRoll"),
      uForwardLean: loc("uForwardLean"),
      uHeadTranslate: loc("uHeadTranslate"),
      uTorsoBreath: loc("uTorsoBreath"),
      uTorsoSwayX: loc("uTorsoSwayX"),
      uShoulderTilt: loc("uShoulderTilt"),
      uShoulderShrug: loc("uShoulderShrug"),
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
      gl.uniform1f(uLoc.uYaw, face.yaw + frame.headYaw * 0.55);

      gl.uniform1f(uLoc.uStylized, anime ? 1 : 0);
      gl.uniform1f(uLoc.uWireframe, 0);
      gl.uniform1f(uLoc.uTime, now * 0.001);
      gl.uniform1f(uLoc.uOpen, frame.open * (anime ? 1.08 : 1));
      gl.uniform1f(uLoc.uWide, frame.wide);
      gl.uniform1f(uLoc.uRound, frame.round);
      gl.uniform1f(uLoc.uLipPress, frame.lipPress);
      gl.uniform1f(uLoc.uLabiodental, frame.labiodental);
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
      gl.uniform1f(uLoc.uLidDroop, frame.lidDroop);
      gl.uniform1f(uLoc.uNoseFlare, frame.noseFlare);
      gl.uniform1f(uLoc.uTongueOut, frame.tongueOut);
      gl.uniform1f(uLoc.uTearAmount, frame.tearAmount);
      gl.uniform1f(uLoc.uEyeWater, frame.eyeWater);
      gl.uniform1f(uLoc.uEyeRedness, frame.eyeRedness);
      gl.uniform1f(uLoc.uPupilDilate, frame.pupilDilate);
      gl.uniform1f(uLoc.uSweatAmount, frame.sweatAmount);
      gl.uniform1f(uLoc.uBlushAmount, frame.blushAmount);
      gl.uniform1f(uLoc.uSwallow, frame.swallow);
      gl.uniform1f(uLoc.uGenderMale, curProfile.gender === "male" ? 1 : 0);

      gl.uniform2f(uLoc.uHeadRot, frame.headYaw * 0.72, frame.headPitch * 0.52);
      gl.uniform1f(uLoc.uHeadRoll, frame.headRoll * 0.4);
      gl.uniform1f(uLoc.uForwardLean, frame.forwardLean);
      gl.uniform2f(uLoc.uHeadTranslate, frame.headTx * 0.85, frame.headTy * 0.85);
      gl.uniform1f(uLoc.uTorsoBreath, frame.torsoBreath);
      gl.uniform1f(uLoc.uTorsoSwayX, frame.torsoSwayX);
      gl.uniform1f(uLoc.uShoulderTilt, frame.shoulderTilt);
      gl.uniform1f(uLoc.uShoulderShrug, frame.shoulderShrug);
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
