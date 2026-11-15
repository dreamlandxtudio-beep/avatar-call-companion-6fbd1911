import { useEffect, useRef, useState } from "react";
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
import teethTextureUrlAsset from "@/assets/images/teeth_mouth_cavity_1794754794898.jpg.asset.json";
const teethTextureUrl = teethTextureUrlAsset.url;

export type { AvatarEmotion };
export type AvatarGesture = "idle" | "talk";

// Vertex layout (20 floats per vertex):
// aPosUv:    vec4(nx, ny, uvx, uvy)        [photo normalized 0..1]
// aHeadRig:  vec4(headW, torsoW, jawW, edgeFade)
// aFaceRig:  vec4(upperLipW, cornerW, cheekW, noseW)
// aBrowRig:  vec4(innerBrowW, outerBrowW, eyeOrbitW, sideSign)
// aHandRig:  vec4(hand0W, hand1W, mouthLipT, mouthX)

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
uniform vec4 uFaceAxis;   // xy = right unit vector, zw = down unit vector
uniform vec4 uMouthAxis;  // xy = mouth right unit vector, zw = mouth down unit vector
uniform float uFaceW;
uniform float uFaceH;
uniform float uMouthW;
uniform float uAspect;

uniform float uOpen;
uniform float uWide;
uniform float uRound;
uniform float uSmile;
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
uniform vec2 uHeadTranslate;
uniform float uTorsoBreath;
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
  float aspect = max(0.25, uAspect);
  float headW = aHeadRig.x;
  float torsoW = aHeadRig.y;
  float jawW = aHeadRig.z;
  float side = aBrowRig.w;

  // 1) Eyebrows & outer orbital skin oriented along true facial tilt axis
  float asym = side * uBrowAsym;
  float browLift = (aBrowRig.x * (uInnerBrowY + asym * 0.45) + aBrowRig.y * (uOuterBrowY + asym)) * 0.052 * uFaceH;
  float browPinch = -side * aBrowRig.x * uInnerBrowPinch * 0.036 * uFaceW;
  float orbitLift = aBrowRig.z * (uEyeWide * 0.014 - uSquint * 0.015) * uFaceH;
  p = addAxisDelta(p, uFaceAxis, vec2(browPinch, -(browLift + orbitLift)), aspect);

  // 2) Subtle nostril flare along facial axis
  p = addAxisDelta(p, uFaceAxis, vec2(side * aFaceRig.w * uNoseFlare * 0.016 * uFaceW, 0.0), aspect);

  // 3) Cheek & zygomatic smile lift along facial axis
  p = addAxisDelta(
    p,
    uFaceAxis,
    vec2(0.0, -aFaceRig.z * (uCheekLift * 0.028 + max(0.0, uSmile) * 0.014) * uFaceH),
    aspect
  );

  // 4) Localized lips & jaw articulation (strictly tapered to zero at chin fold so lower neck never moves)
  vec2 mouthRel = vec2((p.x - uMouthCenter.x) * aspect, p.y - uMouthCenter.y);
  float mSign = clamp(dot(mouthRel, uMouthAxis.xy) / max(0.01, uMouthW * 0.44), -1.0, 1.0);

  float upperLipLift = aFaceRig.x * (uOpen * 0.024 + max(0.0, uSmile) * 0.016) * uMouthW;
  float cornerWide = mSign * aFaceRig.y * (uWide * 0.044 + uSmile * 0.044 - uRound * 0.050) * uMouthW;
  float cornerLift = abs(mSign) * aFaceRig.y * (uSmile * 0.052 + uWide * 0.012) * uMouthW;
  float jawDrop = uOpen * 0.068 * uFaceH * jawW;
  float jawPinch = -mSign * uRound * 0.012 * uFaceW * jawW;
  p = addAxisDelta(
    p,
    uMouthAxis,
    vec2(cornerWide + jawPinch, jawDrop - upperLipLift - cornerLift),
    aspect
  );

  // Skull-space vertical coordinate inside mouth opening (fixed uMouthW units)
  vec2 postMouthRel = vec2((p.x - uMouthCenter.x) * aspect, p.y - uMouthCenter.y);
  vMouthPlaneY = dot(postMouthRel, uMouthAxis.zw) / max(0.01, uMouthW);

  // 5) Genuine 3D Z-depth head perspective turn (foreshortening + nose/cheek parallax, zero seam tearing)
  if (headW > 0.001) {
    vec2 rel = vec2((p.x - uNeckPivot.x) * aspect, p.y - uNeckPivot.y);
    float c = cos(uHeadRoll * headW);
    float s = sin(uHeadRoll * headW);
    vec2 r = vec2(rel.x * c - rel.y * s, rel.x * s + rel.y * c);

    float depthFactor = uPass < 0.5
      ? (0.24 * headW)
      : (uPass < 1.5 ? mix(0.24, 1.08, clamp(aHandRig.z, 0.0, 1.0)) : 0.64);
    float yawA = uHeadRot.x * 0.28 * headW;
    float pitchA = uHeadRot.y * 0.22 * headW;
    float faceNormX = clamp(r.x / max(0.05, uFaceW * 0.56), -1.35, 1.35);
    float faceNormY = clamp((p.y - uMouthCenter.y + 0.18 * uFaceH) / max(0.05, uFaceH * 0.58), -1.35, 1.35);
    float perspX = max(0.76, 1.0 + sin(yawA) * faceNormX * 0.24);
    float perspY = max(0.80, 1.0 - sin(pitchA) * faceNormY * 0.16);

    r.x = (r.x / perspX) * (1.0 - abs(yawA) * 0.10) + yawA * 0.15 * uFaceW * depthFactor;
    r.y = (r.y / perspY) * (1.0 - abs(pitchA) * 0.08) - pitchA * 0.13 * uFaceH * depthFactor;

    p.x = uNeckPivot.x + r.x / aspect + uHeadTranslate.x * headW;
    p.y = uNeckPivot.y + r.y + uHeadTranslate.y * headW;
  }

  // 6) Hair locked to head (zero extra warp behind hair so background stays pristine)
  if (uIsBasePass > 0.5) {
    p += uHairDelta * 0.0;
  }

  // 7) Subtle chest/shoulder breathing
  p.y -= uTorsoBreath * 0.008 * uFaceH * torsoW;

  // 8) Subtle hand micro-gestures when hands are visible
  p += uHand0Delta * aHandRig.x;
  p += uHand1Delta * aHandRig.y;

  // Fit normalized [0..1] image coordinate into canvas NDC
  if (uIsBasePass > 0.5) p = aPosUv.xy;
  vec2 px = uCoverOffset + p * uCoverSize;
  vec2 ndc = vec2((px.x / max(1.0, uViewSize.x)) * 2.0 - 1.0, 1.0 - (px.y / max(1.0, uViewSize.y)) * 2.0);
  gl_Position = vec4(ndc, 0.0, 1.0);
}
`;

const FRAG_SRC = `#version 300 es
precision highp float;

uniform sampler2D uPhotoTex;
uniform sampler2D uTeethTex;
uniform float uPass; // 0.0 = base stitch mesh, 1.0 = face mesh, 2.0 = oral interior

uniform vec2 uREyeCenter;
uniform vec2 uREyePupil;
uniform vec4 uREyeAxis;
uniform vec2 uREyeScale; // x: halfW (aspect space), y: max vertical aperture
uniform vec4 uREyeUpper; // normalized lid contour at t = -0.6, -0.2, +0.2, +0.6
uniform vec4 uREyeLower;
uniform float uRIrisRad;

uniform vec2 uLEyeCenter;
uniform vec2 uLEyePupil;
uniform vec4 uLEyeAxis;
uniform vec2 uLEyeScale;
uniform vec4 uLEyeUpper;
uniform vec4 uLEyeLower;
uniform float uLIrisRad;

uniform vec2 uGaze; // +x right, +y up
uniform float uBlink;
uniform float uSquint;
uniform float uEyeWide;
uniform float uAspect;
uniform float uYaw;
uniform float uFaceH;
uniform float uMouthW;

uniform float uOpen;
uniform float uWide;
uniform float uRound;
uniform float uSmile;
uniform float uHasVisibleTeeth;
uniform vec3 uLipColor;
uniform vec3 uEnamelColor;
uniform vec3 uEyelidColor;
uniform vec3 uRScleraColor;
uniform vec3 uLScleraColor;

in vec2 vUv;
in float vEdgeFade;
in float vMouthLipT;
in float vMouthX;
in float vMouthPlaneY;

out vec4 fragColor;

float evalLidCurve(float t, vec4 c) {
  float x = clamp(t, -1.0, 1.0);
  if (x < -0.6) return mix(0.0, c.x, smoothstep(-1.0, -0.6, x));
  if (x < -0.2) return mix(c.x, c.y, smoothstep(-0.6, -0.2, x));
  if (x < 0.2)  return mix(c.y, c.z, smoothstep(-0.2, 0.2, x));
  if (x < 0.6)  return mix(c.z, c.w, smoothstep(0.2, 0.6, x));
  return mix(c.w, 0.0, smoothstep(0.6, 1.0, x));
}

// Crisp portrait-matched teeth sampler (micro-AA only, zero blurry box filter)
vec3 sampleCrispTeeth(vec2 uv) {
  vec2 d = vec2(0.0011, 0.0015);
  vec3 c = texture(uTeethTex, uv).rgb * 0.64;
  c += texture(uTeethTex, uv + vec2(-d.x, 0.0)).rgb * 0.09;
  c += texture(uTeethTex, uv + vec2( d.x, 0.0)).rgb * 0.09;
  c += texture(uTeethTex, uv + vec2(0.0, -d.y)).rgb * 0.09;
  c += texture(uTeethTex, uv + vec2(0.0,  d.y)).rgb * 0.09;
  return c;
}

// Anatomical eye-axis rendering: unmistakable corneal/pupil motion + clean top-to-bottom closing eyelid
vec3 renderEye(
  vec3 baseRgb,
  vec2 eyeCenter,
  vec2 eyePupil,
  vec4 eyeAxis,
  vec2 eyeScale,
  vec4 upperCurve,
  vec4 lowerCurve,
  float irisRadNorm,
  vec3 scleraCleanColor,
  float aspect
) {
  vec2 dUv = vec2((vUv.x - eyeCenter.x) * aspect, vUv.y - eyeCenter.y);
  float halfW = max(0.005, eyeScale.x);
  float scaleY = max(0.004, eyeScale.y);

  // Local rotated eye coordinates: ex in [-1..+1] corner-to-corner, ey in [-1..+1] vertical
  float ex = dot(dUv, eyeAxis.xy) / halfW;
  float ey = dot(dUv, eyeAxis.zw) / scaleY;

  float socket = 1.0 - smoothstep(0.94, 1.28, length(vec2(ex * 0.88, ey * 0.56)));
  if (socket < 0.002) return baseRgb;

  // Exact anatomical upper & lower lid contours (0 at both eye corners)
  float rawTopY = evalLidCurve(ex, upperCurve);
  float rawBotY = evalLidCurve(ex, lowerCurve);
  float lidSpread = max(0.08, 1.0 - ex * ex);

  float restTopY = rawTopY - (uEyeWide * 0.28 - uSquint * 0.20) * lidSpread;
  float restBotY = rawBotY + (uEyeWide * 0.12 - uSquint * 0.36) * lidSpread;

  // Sweep descending upper eyelid cleanly down to lower palpebral margin so blinks close 100% visibly
  float meetY = rawBotY * 0.82;
  float topY = mix(restTopY, meetY + 0.08, uBlink);
  float botY = mix(restBotY, meetY - 0.08, uBlink * 0.88);

  float openTop = smoothstep(topY - 0.05, topY + 0.09, ey);
  float openBot = 1.0 - smoothstep(botY - 0.09, botY + 0.05, ey);
  float lidOpen = openTop * openBot * (1.0 - smoothstep(0.76, 0.94, uBlink));

  // Rigid 3D spherical cornea/iris shift inside anatomical palpebral aperture (clearly visible travel)
  vec2 gazeDeltaAspect =
    eyeAxis.xy * (uGaze.x * halfW * 0.62) -
    eyeAxis.zw * (uGaze.y * max(scaleY, halfW * 0.44) * 0.46);
  vec2 gazeShift = vec2(gazeDeltaAspect.x / aspect, gazeDeltaAspect.y);

  float irisR = max(0.006, irisRadNorm);
  float yawForeshorten = max(0.58, 1.0 - abs(uYaw) * 0.42);

  // 1) Clean sclera layer: reveal bright real sclera where original iris moved away from
  vec2 dOrigPupil = vec2((vUv.x - eyePupil.x) * aspect, vUv.y - eyePupil.y);
  float distOrig = length(vec2(dot(dOrigPupil, eyeAxis.xy) / yawForeshorten, dot(dOrigPupil, eyeAxis.zw)));
  float oldIrisReveal = 1.0 - smoothstep(irisR * 0.68, irisR * 1.24, distOrig);

  float sY = clamp(ey, rawTopY * 0.55, rawBotY * 0.55);
  vec2 scLeftAspect = eyeAxis.xy * (-0.68 * halfW) + eyeAxis.zw * (sY * scaleY);
  vec2 scRightAspect = eyeAxis.xy * (0.68 * halfW) + eyeAxis.zw * (sY * scaleY);
  vec3 scLeft = texture(uPhotoTex, clamp(eyeCenter + vec2(scLeftAspect.x / aspect, scLeftAspect.y), 0.001, 0.999)).rgb;
  vec3 scRight = texture(uPhotoTex, clamp(eyeCenter + vec2(scRightAspect.x / aspect, scRightAspect.y), 0.001, 0.999)).rgb;
  vec3 scPhoto = mix(scLeft, scRight, smoothstep(-0.35, 0.35, ex));
  vec3 scleraClean = mix(scPhoto, max(scPhoto, scleraCleanColor * 0.96), 0.58);
  vec3 scleraCol = mix(baseRgb, scleraClean, oldIrisReveal);

  // 2) Crisp rigid shifted cornea/iris layer (zero smearing!)
  vec2 newPupil = eyePupil + gazeShift;
  vec2 dNewPupil = vec2((vUv.x - newPupil.x) * aspect, vUv.y - newPupil.y);
  float distNew = length(vec2(dot(dNewPupil, eyeAxis.xy) / yawForeshorten, dot(dNewPupil, eyeAxis.zw)));
  float irisMask = 1.0 - smoothstep(irisR * 0.80, irisR * 1.08, distNew);

  vec2 irisSrcUv = clamp(vUv - gazeShift, 0.001, 0.999);
  vec3 irisCol = texture(uPhotoTex, irisSrcUv).rgb;
  float limbalRing = exp(-pow((distNew - irisR * 0.93) / max(0.0014, irisR * 0.14), 2.0)) * 0.16 * irisMask;

  vec3 eyeCol = mix(scleraCol, irisCol, irisMask);

  // Natural upper-eyelid ambient shadow + sclera corner depth + crisp limbal border
  float lidDepthShadow = mix(0.72, 1.0, smoothstep(topY - 0.04, topY + 0.52, ey));
  float cornerAO = 1.0 - 0.14 * smoothstep(0.62, 1.0, abs(ex));
  eyeCol *= lidDepthShadow * cornerAO * (1.0 - limbalRing);

  // Dual wet corneal highlight sliding smoothly with cornea
  vec2 specOffset1 = eyeAxis.xy * (-irisR * 0.22) + eyeAxis.zw * (-irisR * 0.24);
  vec2 specCenter1 = newPupil + vec2(specOffset1.x / aspect, specOffset1.y) + gazeShift * 0.12;
  float specDist1 = length(vec2((vUv.x - specCenter1.x) * aspect, vUv.y - specCenter1.y)) / max(0.0016, irisR * 0.18);
  float spec1 = (1.0 - smoothstep(0.18, 1.0, specDist1)) * 0.24 * lidOpen * irisMask;

  vec2 specOffset2 = eyeAxis.xy * (irisR * 0.18) + eyeAxis.zw * (irisR * 0.16);
  vec2 specCenter2 = newPupil + vec2(specOffset2.x / aspect, specOffset2.y) + gazeShift * 0.08;
  float specDist2 = length(vec2((vUv.x - specCenter2.x) * aspect, vUv.y - specCenter2.y)) / max(0.0014, irisR * 0.12);
  float spec2 = (1.0 - smoothstep(0.22, 1.0, specDist2)) * 0.11 * lidOpen * irisMask;
  eyeCol += vec3(spec1 + spec2);

  // True descending upper eyelid skin: sample clean upper-lid crease + skin tone so dark liner never smears
  float lidPadTop = mix(0.16, 0.26, uBlink);
  float lidPadBot = mix(0.10, 0.16, uBlink);
  float lidOpenRegion =
    smoothstep(rawTopY - lidPadTop, rawTopY + 0.03, ey) *
    (1.0 - smoothstep(rawBotY - 0.03, rawBotY + lidPadBot, ey));
  float sampleUpperEy = clamp(rawTopY - 0.82 + (ey - rawTopY) * 0.10, rawTopY - 1.05, rawTopY - 0.55);
  vec2 lidSampleAspect = eyeAxis.xy * (ex * halfW) + eyeAxis.zw * (sampleUpperEy * scaleY);
  vec2 lidSkinUv = clamp(eyeCenter + vec2(lidSampleAspect.x / aspect, lidSampleAspect.y), 0.001, 0.999);
  vec3 rawUpperSkin = texture(uPhotoTex, lidSkinUv).rgb;
  float upperLuma = dot(rawUpperSkin, vec3(0.299, 0.587, 0.114));
  float skinMix = mix(0.55, 0.86, 1.0 - smoothstep(0.18, 0.58, upperLuma));
  vec3 closedUpperSkin = mix(rawUpperSkin, uEyelidColor, skinMix);

  // Realistic 3D cylindrical eyelid dome highlight & fold shading
  float domeCurve = 0.90 + 0.13 * sin(clamp((ey - rawTopY) / max(0.2, topY - rawTopY), 0.0, 1.0) * 3.14159);
  closedUpperSkin *= domeCurve;

  float sampleLowerEy = clamp(rawBotY + 0.48, rawBotY + 0.20, rawBotY + 0.72);
  vec2 botSampleAspect = eyeAxis.xy * (ex * halfW) + eyeAxis.zw * (sampleLowerEy * scaleY);
  vec2 botSkinUv = clamp(eyeCenter + vec2(botSampleAspect.x / aspect, botSampleAspect.y), 0.001, 0.999);
  vec3 closedLowerSkin = mix(texture(uPhotoTex, botSkinUv).rgb, uEyelidColor * 0.96, 0.45);

  float isUpperHalf = 1.0 - smoothstep(topY - 0.04, topY + 0.08, ey);
  vec3 lidSkin = mix(closedLowerSkin, closedUpperSkin, isUpperHalf);

  // Soft anatomical crease + crisp curved lashline right along descending topY
  float cornerTaper = 1.0 - smoothstep(0.76, 1.0, abs(ex));
  float creaseShadow = exp(-pow((ey - (topY - 0.42)) / 0.18, 2.0)) * smoothstep(0.06, 0.65, uBlink) * 0.20 * cornerTaper;
  float lashLine = exp(-pow((ey - topY) / 0.065, 2.0)) * smoothstep(0.02, 0.28, uBlink) * 0.58 * cornerTaper;
  lidSkin *= (1.0 - lashLine - creaseShadow);

  vec3 finalEye = mix(lidSkin, eyeCol, lidOpen);
  return mix(baseRgb, finalEye, socket * lidOpenRegion);
}

void main() {
  vec4 photoCol = texture(uPhotoTex, clamp(vUv, 0.001, 0.999));
  vec3 col = photoCol.rgb;

  if (uPass > 1.5) {
    // Watertight multi-row oral interior between inner upper and inner lower lips
    float openMask = smoothstep(0.012, 0.065, uOpen);
    if (openMask <= 0.002) {
      discard;
    }

    float absX = abs(vMouthX);
    float lipRimFeather =
      smoothstep(0.0, 0.085, vMouthLipT) *
      smoothstep(1.0, 0.915, vMouthLipT) *
      (1.0 - smoothstep(0.88, 1.0, absX));

    // Full proportional dental arch width (0.42..0.46) fitting central, lateral, canine, and rear premolar teeth
    float archZoom = mix(0.46, 0.41, clamp(uWide, 0.0, 1.0)) * (1.0 + clamp(uRound, 0.0, 1.0) * 0.10);
    float tx = clamp(0.5 + (vMouthX - uYaw * 0.14) * archZoom, 0.04, 0.96);
    float biteCurve = 0.011 * vMouthX * vMouthX;

    // 1) Deep dark reddish oral cavity interior (stationary mucosa depth)
    float cavY = clamp(0.35 + vMouthLipT * 0.16, 0.33, 0.52);
    vec3 cavTex = sampleCrispTeeth(vec2(tx, cavY));
    vec3 deepRedMucosa = vec3(0.085, 0.016, 0.022) + cavTex * vec3(0.16, 0.032, 0.038);
    float throatDepth = 1.0 - 0.54 * (1.0 - abs(vMouthLipT - 0.45) * 1.4) * (1.0 - absX * 0.7);
    vec3 oralCol = deepRedMucosa * clamp(throatDepth, 0.28, 1.0);

    // Coordinate relative to moving lower jaw (mandible) in fixed uMouthW units (matched cleanly to 0.068 jawDrop)
    float jawDropMouthW = (uOpen * 0.068 * uFaceH) / max(0.01, uMouthW);
    float lowerPlaneY = vMouthPlaneY - jawDropMouthW;

    // 2) Soft natural resting tongue anchored on lower jaw
    float tongueLift = clamp(uRound, 0.0, 1.0) * 0.008 - clamp(uOpen, 0.0, 1.0) * 0.005;
    float tongueTopY = -0.056 + 0.042 * vMouthX * vMouthX + tongueLift;
    float tongueBotY = 0.046 - 0.014 * vMouthX * vMouthX;
    float tongueLatMask = 1.0 - smoothstep(0.42, 0.70, absX);
    float tongueVertMask =
      smoothstep(tongueTopY - 0.022, tongueTopY + 0.018, lowerPlaneY) *
      (1.0 - smoothstep(tongueBotY - 0.016, tongueBotY + 0.012, lowerPlaneY));
    float tongueMask = tongueLatMask * tongueVertMask * smoothstep(0.10, 0.30, uOpen);
    float tNormY = clamp((lowerPlaneY - tongueTopY) / max(0.035, tongueBotY - tongueTopY), 0.0, 1.0);
    float tongueTexY = mix(0.56, 0.70, tNormY);
    vec3 tongueSample = sampleCrispTeeth(vec2(tx, tongueTexY));
    float tongueGroove = 1.0 - 0.12 * exp(-pow(vMouthX / 0.15, 2.0));
    float tongueShade = mix(0.54, 0.96, smoothstep(0.0, 0.45, tNormY)) * tongueGroove;
    vec3 tongueCol = (tongueSample * vec3(0.62, 0.32, 0.34) + vec3(0.075, 0.018, 0.022)) * tongueShade;
    oralCol = mix(oralCol, tongueCol, tongueMask * 0.88);

    // 3) SKULL-LOCKED UPPER TEETH: crisp, clean natural white enamel matching portrait sharpness
    float archCurveT = pow(clamp(abs(tx - 0.5) * 2.0, 0.0, 1.0), 1.75);
    float upperGumY = -0.042 - clamp(uSmile, -0.1, 0.5) * 0.009 - biteCurve;
    float upperToothH = mix(0.098, 0.084, archCurveT);
    float upperFrac = clamp((vMouthPlaneY - upperGumY) / upperToothH, 0.0, 1.0);
    float upperTexTop = 0.165 + 0.095 * archCurveT;
    float upperTexBot = 0.294 + 0.096 * archCurveT;
    vec3 rawUpperTeeth = sampleCrispTeeth(vec2(tx, mix(upperTexTop, upperTexBot, upperFrac)));

    // 4) MANDIBLE-LOCKED LOWER TEETH: crisp, clean natural enamel
    float lowerEdgeY = -0.030 + biteCurve * 0.75;
    float lowerToothH = mix(0.080, 0.070, archCurveT);
    float lowerFrac = clamp((lowerPlaneY - lowerEdgeY) / lowerToothH, 0.0, 1.0);
    float lowerTexTop = 0.762 - 0.150 * archCurveT;
    float lowerTexBot = 0.858 - 0.140 * archCurveT;
    vec3 rawLowerTeeth = sampleCrispTeeth(vec2(tx, mix(lowerTexTop, lowerTexBot, lowerFrac)));

    // Equalize warm color cast while keeping crisp interdental edge definition
    float uLuma = dot(rawUpperTeeth, vec3(0.299, 0.587, 0.114));
    vec3 cleanUpper = mix(rawUpperTeeth, vec3(clamp(uLuma * 1.05 + 0.04, 0.54, 0.95)), 0.55);
    cleanUpper = mix(vec3(0.92, 0.92, 0.915), cleanUpper, 0.78);

    float lLuma = dot(rawLowerTeeth, vec3(0.299, 0.587, 0.114));
    vec3 cleanLower = mix(rawLowerTeeth, vec3(clamp(lLuma * 1.03 + 0.03, 0.50, 0.91)), 0.58);
    cleanLower = mix(vec3(0.88, 0.88, 0.875), cleanLower, 0.76);

    vec3 enamelTarget = clamp(mix(uEnamelColor, vec3(0.92, 0.92, 0.915), 0.40), vec3(0.68), vec3(0.95));
    vec3 upperTeethSample = cleanUpper * (enamelTarget / vec3(0.89, 0.89, 0.885));
    vec3 lowerTeethSample = cleanLower * (enamelTarget / vec3(0.89, 0.89, 0.885)) * 0.84;

    if (uHasVisibleTeeth > 0.5) {
      float keepPhoto = (1.0 - smoothstep(0.16, 0.42, uOpen)) * (1.0 - smoothstep(0.32, 0.52, vMouthLipT));
      upperTeethSample = mix(upperTeethSample, photoCol.rgb, keepPhoto * 0.82);
    }

    float upperZone =
      smoothstep(upperGumY - 0.008, upperGumY + 0.012, vMouthPlaneY) *
      (1.0 - smoothstep(upperGumY + upperToothH - 0.016, upperGumY + upperToothH + 0.006, vMouthPlaneY)) *
      (1.0 - smoothstep(0.84, 0.99, absX));
    float lowerZone =
      smoothstep(lowerEdgeY - 0.008, lowerEdgeY + 0.014, lowerPlaneY) *
      (1.0 - smoothstep(lowerEdgeY + lowerToothH - 0.014, lowerEdgeY + lowerToothH + 0.010, lowerPlaneY)) *
      smoothstep(0.14, 0.36, uOpen) *
      (1.0 - smoothstep(0.80, 0.98, absX));

    oralCol = mix(oralCol, upperTeethSample, upperZone * 0.94);
    oralCol = mix(oralCol, lowerTeethSample, lowerZone * 0.86);

    // Physical upper-lip overhang shadow, lower-lip crease shadow, and buccal corner AO
    float topAO = mix(0.42, 1.0, smoothstep(0.0, 0.22, vMouthLipT));
    float botAO = mix(0.50, 1.0, smoothstep(1.0, 0.80, vMouthLipT));
    float sideAO = 1.0 - 0.78 * smoothstep(0.45, 0.96, absX);
    float cavityLight = mix(0.58, 1.0, smoothstep(0.02, 0.26, uOpen));
    vec3 warmShadowTint = max(vec3(0.042, 0.009, 0.012), uLipColor * vec3(0.16, 0.06, 0.07));
    oralCol = mix(warmShadowTint, oralCol, topAO * botAO * sideAO * cavityLight);

    fragColor = vec4(oralCol, openMask * lipRimFeather);
    return;
  }

  if (uPass > 0.5) {
    col = renderEye(
      col,
      uREyeCenter,
      uREyePupil,
      uREyeAxis,
      uREyeScale,
      uREyeUpper,
      uREyeLower,
      uRIrisRad,
      uRScleraColor,
      uAspect
    );
    col = renderEye(
      col,
      uLEyeCenter,
      uLEyePupil,
      uLEyeAxis,
      uLEyeScale,
      uLEyeUpper,
      uLEyeLower,
      uLIrisRad,
      uLScleraColor,
      uAspect
    );
    fragColor = vec4(col, vEdgeFade);
    return;
  }

  // Pass 0: full crisp background + body + neck + hair base mesh
  fragColor = vec4(col, 1.0);
}
`;

type BuiltMeshes = {
  baseVerts: Float32Array;
  baseIdx: Uint16Array;
  faceVerts: Float32Array;
  faceIdx: Uint16Array;
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

function buildAvatarMeshes(face: FacePts): BuiltMeshes {
  const aspect = face.aspect;
  const pts = face.landmarks;
  const faceCenter = face.center;
  const chin = pts[152]!;
  const forehead = pts[10]!;
  const noseTip = pts[1]!;
  const mouthCenter: [number, number] = [
    (pts[61]![0] + pts[291]![0]) * 0.5,
    (pts[13]![1] + pts[14]![1]) * 0.5,
  ];
  const neckPivot: [number, number] = [face.neckBase[0], face.neckBase[1]];
  const jawPivot: [number, number] = [
    (pts[234]![0] + pts[454]![0]) * 0.5,
    (pts[234]![1] + pts[454]![1]) * 0.5,
  ];

  // 1) Base Stitch Grid (44 x 44 across full [0..1] x [0..1] photo)
  const GX = 44;
  const GY = 44;
  const STRIDE = 20;
  const baseVerts = new Float32Array((GX + 1) * (GY + 1) * STRIDE);

  const faceRad = Math.max(face.faceW, face.faceH) * 0.66;
  const neckBottomY = Math.min(0.95, chin[1] + face.faceH * 0.16);
  const hand0 = face.hands[0];
  const hand1 = face.hands[1];

  const sampleSeg = (nx: number, ny: number) => {
    const mask = face.segMask;
    const sw = face.segWidth;
    const sh = face.segHeight;
    if (!mask || !sw || !sh) return 1.0;
    const fx = Math.max(0, Math.min(sw - 1, nx * (sw - 1)));
    const fy = Math.max(0, Math.min(sh - 1, ny * (sh - 1)));
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(sw - 1, x0 + 1);
    const y1 = Math.min(sh - 1, y0 + 1);
    const tx = fx - x0;
    const ty = fy - y0;
    const v00 = mask[y0 * sw + x0] ?? 1;
    const v10 = mask[y0 * sw + x1] ?? 1;
    const v01 = mask[y1 * sw + x0] ?? 1;
    const v11 = mask[y1 * sw + x1] ?? 1;
    const top = v00 * (1 - tx) + v10 * tx;
    const bot = v01 * (1 - tx) + v11 * tx;
    return top * (1 - ty) + bot * ty;
  };

  const hasSegMask = !!(face.segMask && face.segWidth && face.segHeight);
  let ptr = 0;
  for (let iy = 0; iy <= GY; iy++) {
    const ny = iy / GY;
    const borderY = smoothstep(0.0, 0.08, ny) * smoothstep(1.0, 0.92, ny);
    for (let ix = 0; ix <= GX; ix++) {
      const nx = ix / GX;
      const borderX = smoothstep(0.0, 0.08, nx) * smoothstep(1.0, 0.92, nx);
      const borderMask = borderX * borderY;

      const dHead = Math.hypot(
        (nx - faceCenter[0]) * aspect,
        (ny - (faceCenter[1] - 0.04 * face.faceH)) * 0.92,
      );
      const fg = sampleSeg(nx, ny);
      const fgLock = smoothstep(0.72, 0.95, fg);
      let headW = hasSegMask
        ? (1 - smoothstep(faceRad * 0.72, faceRad * 1.18, dHead)) *
          (1 - smoothstep(faceRad * 0.65, faceRad * 0.9, dHead) * (1 - fgLock)) *
          borderMask
        : (1 - smoothstep(faceRad * 0.62, faceRad * 0.96, dHead)) * borderMask;
      if (ny > chin[1] + 0.01 * face.faceH) {
        headW *= 1 - smoothstep(chin[1] + 0.01 * face.faceH, neckBottomY, ny);
      }

      const torsoW =
        smoothstep(chin[1] + 0.09 * face.faceH, neckBottomY + 0.16 * face.faceH, ny) *
        (1 - smoothstep(0.45 * face.faceW * 3.0, 0.95, Math.abs((nx - neckPivot[0]) * aspect))) *
        (hasSegMask ? fgLock : 1) *
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

      const jawBaseLat =
        1 -
        smoothstep(face.mouthW * 0.48, face.faceW * 0.74, Math.abs((nx - mouthCenter[0]) * aspect));
      const jawUpperMask = smoothstep(mouthCenter[1] - 0.01 * face.faceH, pts[17]![1], ny);
      const jawLowerMask =
        ny <= chin[1]
          ? 1 - smoothstep(pts[17]![1], chin[1], ny) * 0.72
          : 0.28 * (1 - smoothstep(chin[1], chin[1] + 0.025 * face.faceH, ny));
      const baseJawW = jawBaseLat * jawUpperMask * jawLowerMask * borderMask;

      baseVerts[ptr++] = nx;
      baseVerts[ptr++] = ny;
      baseVerts[ptr++] = nx;
      baseVerts[ptr++] = ny;
      baseVerts[ptr++] = headW;
      baseVerts[ptr++] = torsoW;
      baseVerts[ptr++] = baseJawW;
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
      baseVerts[ptr++] = 0;
      baseVerts[ptr++] = 0;
    }
  }

  const baseIdxList: number[] = [];
  for (let iy = 0; iy < GY; iy++) {
    for (let ix = 0; ix < GX; ix++) {
      const a = iy * (GX + 1) + ix;
      const b = a + 1;
      const c = a + (GX + 1);
      const d = c + 1;
      baseIdxList.push(a, c, b, b, c, d);
    }
  }
  const baseIdx = new Uint16Array(baseIdxList);

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

  const innerBrowSet = new Set([107, 66, 55, 65, 336, 296, 285, 295]);
  const outerBrowSet = new Set([105, 63, 70, 46, 53, 52, 334, 293, 300, 276, 283, 282]);
  const rNoseWing = pts[49] ?? pts[102] ?? noseTip;
  const lNoseWing = pts[279] ?? pts[331] ?? noseTip;
  const rCheekPt = pts[50] ?? pts[118] ?? pts[205]!;
  const lCheekPt = pts[280] ?? pts[347] ?? pts[425]!;
  const rCorner = pts[61]!;
  const lCorner = pts[291]!;
  const innerMouthHalfW = Math.max(
    0.02,
    Math.hypot((pts[308]![0] - pts[78]![0]) * aspect, pts[308]![1] - pts[78]![1]) * 0.5,
  );
  const innerMouthMidX = (pts[78]![0] + pts[308]![0]) * 0.5;
  const innerMouthMidY = (pts[78]![1] + pts[308]![1]) * 0.5;

  const dist2 = (ax: number, ay: number, bx: number, by: number) =>
    Math.hypot((ax - bx) * aspect, ay - by);

  const cornerLipSet = new Set([61, 78, 291, 308]);
  let minZ = 1e9;
  let maxZ = -1e9;
  for (let i = 0; i < numPts; i++) {
    const z = pts[i]![2] ?? 0;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }
  const zSpan = Math.max(0.005, maxZ - minZ);

  let fPtr = 0;
  for (let i = 0; i < numPts; i++) {
    const pt = pts[i]!;
    const nx = pt[0];
    const ny = pt[1];
    const isOval = ovalSet.has(i);
    const isEyeAp = eyeLoopSet.has(i);
    const edgeFade = isOval ? 0.0 : 1.0;

    const localScale = isOval ? 0.0 : 1.0;
    const relFaceX =
      (nx - noseTip[0]) * aspect * face.faceAxis.ux + (ny - noseTip[1]) * face.faceAxis.uy;
    const side = Math.max(-1, Math.min(1, relFaceX / (face.faceW * 0.22)));

    const mouthX = Math.max(
      -1,
      Math.min(
        1,
        ((nx - innerMouthMidX) * aspect * face.mouthAxis.ux +
          (ny - innerMouthMidY) * face.mouthAxis.uy) /
          innerMouthHalfW,
      ),
    );
    const lipCenterArch = Math.max(0, Math.cos(mouthX * Math.PI * 0.48));

    const jawLat =
      1 -
      smoothstep(face.mouthW * 0.48, face.faceW * 0.74, Math.abs((nx - mouthCenter[0]) * aspect));
    const jawUp = smoothstep(mouthCenter[1] - 0.01 * face.faceH, pts[17]![1], ny);
    const jawDn =
      ny <= chin[1]
        ? 1 - smoothstep(pts[17]![1], chin[1], ny) * 0.72
        : 0.28 * (1 - smoothstep(chin[1], chin[1] + 0.025 * face.faceH, ny));
    let jawW = jawLat * jawUp * jawDn;

    const dMouth = dist2(nx, ny, mouthCenter[0], mouthCenter[1]);
    const upperLipDist = dist2(nx, ny, pts[13]![0], pts[13]![1]);
    let upperLipW =
      (1 - smoothstep(0.0, face.mouthW * 0.42, upperLipDist)) *
      (ny <= mouthCenter[1] + 0.015 * face.faceH ? 1 : 0) *
      localScale;

    if (cornerLipSet.has(i)) {
      jawW = 0.24 * localScale;
      upperLipW = 0.16 * localScale;
    } else if (innerUpperSet.has(i) || outerUpperSet.has(i)) {
      jawW = 0.04 * (1 - lipCenterArch) * localScale;
      upperLipW = (0.34 + 0.62 * lipCenterArch) * localScale;
    } else if (innerLowerSet.has(i) || outerLowerSet.has(i)) {
      jawW = (0.32 + 0.68 * lipCenterArch) * localScale;
      upperLipW = 0;
    }

    const dCorners = Math.min(
      dist2(nx, ny, rCorner[0], rCorner[1]),
      dist2(nx, ny, lCorner[0], lCorner[1]),
    );
    let cornerW = (1 - smoothstep(0.0, face.mouthW * 0.55, dCorners)) * localScale;
    if (innerLipSet.has(i) || outerUpperSet.has(i) || outerLowerSet.has(i)) {
      cornerW = Math.max(
        cornerW,
        (1 - smoothstep(0.0, face.mouthW * 0.62, dMouth)) * 0.85 * localScale,
      );
    }

    const dCheek = Math.min(
      dist2(nx, ny, rCheekPt[0], rCheekPt[1]),
      dist2(nx, ny, lCheekPt[0], lCheekPt[1]),
    );
    const cheekW =
      (1 - smoothstep(0.0, face.faceW * 0.28, dCheek)) * localScale * (isEyeAp ? 0 : 1);

    const dWing = Math.min(
      dist2(nx, ny, rNoseWing[0], rNoseWing[1]),
      dist2(nx, ny, lNoseWing[0], lNoseWing[1]),
    );
    const noseW = (1 - smoothstep(0.0, face.faceW * 0.14, dWing)) * localScale;

    const browY = (pts[105]![1] + pts[334]![1]) * 0.5;
    let innerBrowW = 0;
    let outerBrowW = 0;
    if (innerBrowSet.has(i)) innerBrowW = 1.0 * localScale;
    else if (outerBrowSet.has(i)) outerBrowW = 1.0 * localScale;
    else if (ny < Math.min(pts[159]![1], pts[386]![1]) - 0.015 * face.faceH && ny > forehead[1]) {
      const browBand = 1 - smoothstep(0.0, face.faceH * 0.14, Math.abs(ny - browY));
      const xRatio = Math.abs(relFaceX) / (face.faceW * 0.38);
      innerBrowW = browBand * (1 - smoothstep(0.18, 0.55, xRatio)) * 0.72 * localScale;
      outerBrowW = browBand * smoothstep(0.28, 0.75, xRatio) * 0.72 * localScale;
    }

    let eyeOrbitW = 0;
    if (!isEyeAp && !isOval) {
      const dR = dist2(nx, ny, face.rightPupil[0], face.rightPupil[1]);
      const dL = dist2(nx, ny, face.leftPupil[0], face.leftPupil[1]);
      const rSpan = face.rightEyeRadius[0] * aspect;
      const lSpan = face.leftEyeRadius[0] * aspect;
      eyeOrbitW = Math.max(
        (1 - smoothstep(rSpan * 0.9, rSpan * 2.1, dR)) * smoothstep(rSpan * 0.5, rSpan * 0.95, dR),
        (1 - smoothstep(lSpan * 0.9, lSpan * 2.1, dL)) * smoothstep(lSpan * 0.5, lSpan * 0.95, dL),
      );
    }

    // Normalized facial 3D depth (0 at face oval boundary, ~1 at nose bridge/tip) for true perspective turns
    const rawDepth = Math.max(0, Math.min(1, (maxZ - (pt[2] ?? 0)) / zSpan));
    const radialDepth =
      1 - smoothstep(0.0, face.faceW * 0.54, dist2(nx, ny, noseTip[0], noseTip[1]));
    const depthZ = isOval ? 0.0 : Math.min(1, rawDepth * 0.7 + radialDepth * 0.35);

    const headW = 1.0;

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

  const faceIdxArr: number[] = [];
  const tris = face.faceTris;
  for (let i = 0; i < tris.length; i += 3) {
    const a = tris[i]!;
    const b = tris[i + 1]!;
    const c = tris[i + 2]!;
    if (a >= numPts || b >= numPts || c >= numPts) continue;
    if (innerLipSet.has(a) && innerLipSet.has(b) && innerLipSet.has(c)) continue;
    faceIdxArr.push(a, b, c);
  }
  // MediaPipe tessellation leaves holes inside both eye contours. Fill those apertures
  // so the fragment shader can actually move the irises instead of exposing the static base photo.
  for (const loop of [RIGHT_EYE_LOOP, LEFT_EYE_LOOP]) {
    const anchor = loop[0];
    if (anchor === undefined) continue;
    for (let i = 1; i < loop.length - 1; i++) {
      const b = loop[i];
      const c = loop[i + 1];
      if (b !== undefined && c !== undefined) faceIdxArr.push(anchor, b, c);
    }
  }

  // 3) Explicit watertight multi-row oral interior mesh bridging INNER_UPPER_LIP <-> INNER_LOWER_LIP
  const M_COLS = INNER_UPPER_LIP.length; // 11 paired columns from corner 78 to corner 308
  const M_ROWS = 7;
  const mouthVerts = new Float32Array(M_COLS * M_ROWS * STRIDE);
  let mPtr = 0;
  for (let r = 0; r < M_ROWS; r++) {
    const t = r / (M_ROWS - 1);
    for (let k = 0; k < M_COLS; k++) {
      const upIdx = INNER_UPPER_LIP[k]!;
      const loIdx = INNER_LOWER_LIP[k]!;
      const upOff = upIdx * STRIDE;
      const loOff = loIdx * STRIDE;
      for (let f = 0; f < STRIDE; f++) {
        mouthVerts[mPtr + f] = faceVerts[upOff + f]! * (1 - t) + faceVerts[loOff + f]! * t;
      }
      // Keep full alpha & exact parametric mouth coordinates inside the oral patch
      mouthVerts[mPtr + 7] = 1.0; // edgeFade
      mouthVerts[mPtr + 18] = t; // mouthLipT (0 = upper inner lip, 1 = lower inner lip)
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

  return {
    baseVerts,
    baseIdx,
    faceVerts,
    faceIdx: new Uint16Array(faceIdxArr),
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
  className?: string | undefined;
}) {
  const effectiveSrc = src || "/__l5e/assets-v1/ea7b23b3-bee4-43b1-ae6c-9228330211ec/default.jpg";
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

  useEffect(() => {
    let cancel = false;
    setWebglFailed(false);
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
      if (cancel || effectiveSrc === "/__l5e/assets-v1/ea7b23b3-bee4-43b1-ae6c-9228330211ec/default.jpg") return;
      i.src = "/__l5e/assets-v1/ea7b23b3-bee4-43b1-ae6c-9228330211ec/default.jpg";
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
      program = gl.createProgram()!;
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
    const imgW = img.naturalWidth || img.width || 512;
    const imgH = img.naturalHeight || img.height || 512;
    const imgAspect = imgW / imgH;

    // Texture 0: Source portrait image
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Texture 1: Photorealistic dental arch & oral cavity texture
    const teethTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, teethTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    // Fallback 1x1 dark oral cavity until image decodes
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
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      const vbo = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
      const ibo = gl.createBuffer()!;
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

    const baseDraw = setupVao(meshes.baseVerts, meshes.baseIdx);
    const faceDraw = setupVao(meshes.faceVerts, meshes.faceIdx);
    const mouthDraw = setupVao(meshes.mouthVerts, meshes.mouthIdx);

    const loc = (u: string) => gl.getUniformLocation(program, u);
    const uLoc = {
      uPhotoTex: loc("uPhotoTex"),
      uTeethTex: loc("uTeethTex"),
      uPass: loc("uPass"),
      uViewSize: loc("uViewSize"),
      uImgSize: loc("uImgSize"),
      uCoverOffset: loc("uCoverOffset"),
      uCoverSize: loc("uCoverSize"),
      uNeckPivot: loc("uNeckPivot"),
      uJawPivot: loc("uJawPivot"),
      uMouthCenter: loc("uMouthCenter"),
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
      uSmile: loc("uSmile"),
      uCheekLift: loc("uCheekLift"),
      uInnerBrowY: loc("uInnerBrowY"),
      uInnerBrowPinch: loc("uInnerBrowPinch"),
      uOuterBrowY: loc("uOuterBrowY"),
      uBrowAsym: loc("uBrowAsym"),
      uSquint: loc("uSquint"),
      uEyeWide: loc("uEyeWide"),
      uNoseFlare: loc("uNoseFlare"),
      uHeadRot: loc("uHeadRot"),
      uHeadRoll: loc("uHeadRoll"),
      uHeadTranslate: loc("uHeadTranslate"),
      uTorsoBreath: loc("uTorsoBreath"),
      uHand0Delta: loc("uHand0Delta"),
      uHand1Delta: loc("uHand1Delta"),
      uHairDelta: loc("uHairDelta"),
      uIsBasePass: loc("uIsBasePass"),
      uREyeCenter: loc("uREyeCenter"),
      uREyePupil: loc("uREyePupil"),
      uREyeAxis: loc("uREyeAxis"),
      uREyeScale: loc("uREyeScale"),
      uREyeUpper: loc("uREyeUpper"),
      uREyeLower: loc("uREyeLower"),
      uRIrisRad: loc("uRIrisRad"),
      uLEyeCenter: loc("uLEyeCenter"),
      uLEyePupil: loc("uLEyePupil"),
      uLEyeAxis: loc("uLEyeAxis"),
      uLEyeScale: loc("uLEyeScale"),
      uLEyeUpper: loc("uLEyeUpper"),
      uLEyeLower: loc("uLEyeLower"),
      uLIrisRad: loc("uLIrisRad"),
      uGaze: loc("uGaze"),
      uBlink: loc("uBlink"),
      uHasVisibleTeeth: loc("uHasVisibleTeeth"),
      uLipColor: loc("uLipColor"),
      uEnamelColor: loc("uEnamelColor"),
      uEyelidColor: loc("uEyelidColor"),
      uRScleraColor: loc("uRScleraColor"),
      uLScleraColor: loc("uLScleraColor"),
    };

    let viewW = 640;
    let viewH = 480;
    const syncSize = () => {
      const rect = canvas.getBoundingClientRect();
      const budget = portraitRenderBudget(rect.width, rect.height, window.devicePixelRatio || 1);
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

      gl.uniform1f(uLoc.uOpen, frame.open);
      gl.uniform1f(uLoc.uWide, frame.wide);
      gl.uniform1f(uLoc.uRound, frame.round);
      gl.uniform1f(uLoc.uSmile, frame.smile);
      gl.uniform1f(uLoc.uCheekLift, frame.cheekLift);
      gl.uniform1f(uLoc.uInnerBrowY, frame.innerBrowY);
      gl.uniform1f(uLoc.uInnerBrowPinch, frame.innerBrowPinch);
      gl.uniform1f(uLoc.uOuterBrowY, frame.outerBrowY);
      gl.uniform1f(uLoc.uBrowAsym, frame.browAsym);
      gl.uniform1f(uLoc.uSquint, frame.eyeSquint);
      gl.uniform1f(uLoc.uEyeWide, frame.eyeWide);
      gl.uniform1f(uLoc.uNoseFlare, frame.noseFlare);

      gl.uniform2f(uLoc.uHeadRot, frame.headYaw * 0.2, frame.headPitch * 0.18);
      gl.uniform1f(uLoc.uHeadRoll, frame.headRoll * 0.5);
      gl.uniform2f(uLoc.uHeadTranslate, 0, 0);
      gl.uniform1f(uLoc.uTorsoBreath, 0);
      gl.uniform2f(uLoc.uHand0Delta, 0, 0);
      gl.uniform2f(uLoc.uHand1Delta, 0, 0);
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
        uLoc.uREyeUpper,
        face.rightEyeCurve.upper[0],
        face.rightEyeCurve.upper[1],
        face.rightEyeCurve.upper[2],
        face.rightEyeCurve.upper[3],
      );
      gl.uniform4f(
        uLoc.uREyeLower,
        face.rightEyeCurve.lower[0],
        face.rightEyeCurve.lower[1],
        face.rightEyeCurve.lower[2],
        face.rightEyeCurve.lower[3],
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
        uLoc.uLEyeUpper,
        face.leftEyeCurve.upper[0],
        face.leftEyeCurve.upper[1],
        face.leftEyeCurve.upper[2],
        face.leftEyeCurve.upper[3],
      );
      gl.uniform4f(
        uLoc.uLEyeLower,
        face.leftEyeCurve.lower[0],
        face.leftEyeCurve.lower[1],
        face.leftEyeCurve.lower[2],
        face.leftEyeCurve.lower[3],
      );
      gl.uniform1f(uLoc.uLIrisRad, face.leftIrisRadius);

      gl.uniform2f(uLoc.uGaze, frame.gazeX, frame.gazeY);
      gl.uniform1f(uLoc.uBlink, frame.blink);
      gl.uniform1f(uLoc.uHasVisibleTeeth, face.hasVisibleTeeth ? 1 : 0);
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
        (face.colors.skin[0] * 0.72 + face.colors.eyelid[0] * 0.28) / 255,
        (face.colors.skin[1] * 0.72 + face.colors.eyelid[1] * 0.28) / 255,
        (face.colors.skin[2] * 0.72 + face.colors.eyelid[2] * 0.28) / 255,
      );
      gl.uniform3f(
        uLoc.uRScleraColor,
        face.colors.scleraRight[0] / 255,
        face.colors.scleraRight[1] / 255,
        face.colors.scleraRight[2] / 255,
      );
      gl.uniform3f(
        uLoc.uLScleraColor,
        face.colors.scleraLeft[0] / 255,
        face.colors.scleraLeft[1] / 255,
        face.colors.scleraLeft[2] / 255,
      );

      gl.uniform1f(uLoc.uIsBasePass, 1.0);
      gl.uniform1f(uLoc.uPass, 0.0);
      gl.bindVertexArray(baseDraw.vao);
      gl.drawElements(gl.TRIANGLES, baseDraw.count, gl.UNSIGNED_SHORT, 0);

      gl.uniform1f(uLoc.uIsBasePass, 0.0);
      gl.uniform1f(uLoc.uPass, 1.0);
      gl.bindVertexArray(faceDraw.vao);
      gl.drawElements(gl.TRIANGLES, faceDraw.count, gl.UNSIGNED_SHORT, 0);

      if (mouthDraw.count > 0) {
        gl.uniform1f(uLoc.uIsBasePass, 0.0);
        gl.uniform1f(uLoc.uPass, 2.0);
        gl.bindVertexArray(mouthDraw.vao);
        gl.drawElements(gl.TRIANGLES, mouthDraw.count, gl.UNSIGNED_SHORT, 0);
      }

      gl.bindVertexArray(null);
    };

    raf = requestAnimationFrame(draw);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
      for (const d of [baseDraw, faceDraw, mouthDraw]) {
        gl.deleteVertexArray(d.vao);
        gl.deleteBuffer(d.vbo);
        gl.deleteBuffer(d.ibo);
      }
      gl.deleteTexture(tex);
      gl.deleteTexture(teethTex);
      gl.deleteProgram(program);
    };
  }, [img, face, webglFailed]);

  return (
    <div ref={containerRef} className={`relative overflow-hidden select-none ${className}`}>
      <img
        src={effectiveSrc}
        alt={name}
        className={`pointer-events-none absolute inset-0 h-full w-full ${
          fit === "contain" ? "object-contain object-center" : "object-cover object-[50%_28%]"
        }`}
      />
      {!webglFailed && (
        <canvas
          ref={canvasRef}
          className={`relative block h-full w-full transition-opacity duration-300 ${
            rigReady ? "opacity-100" : "opacity-0"
          }`}
        />
      )}
      {engine === "liveavatar" && (
        <LiveAvatarStreamPlayer serverUrl={liveAvatarUrl} active={active} name={name} fallbackPhoto={effectiveSrc} />
      )}
    </div>
  );
}
