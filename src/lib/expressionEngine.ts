// 100% local client-side facial expression, ocular saccade, blink, 3D head perspective & lip-sync kinematics engine.
import type { AvatarEmotion } from "./ai";
import { blinkClosure, mouthMotion } from "./avatar-motion";
import type { UserFaceObservation } from "./userCamera";
import type { AudioVisemes } from "./voice";

export type ExpressionFrame = {
  blink: number;
  open: number;
  wide: number;
  round: number;
  smile: number;
  cheekLift: number;
  innerBrowY: number;
  innerBrowPinch: number;
  outerBrowY: number;
  browAsym: number;
  eyeSquint: number;
  eyeWide: number;
  noseFlare: number;
  gazeX: number;
  gazeY: number;
  headYaw: number;
  headPitch: number;
  headRoll: number;
  headTx: number;
  headTy: number;
  torsoBreath: number;
  handShiftX: number;
  handShiftY: number;
  hairShiftX: number;
  hairShiftY: number;
};

export type ExpressionEngineState = {
  lastNow: number;
  nextBlinkAt: number;
  blinkProgress: number;
  doubleBlinkQueued: boolean;
  nextSaccadeAt: number;
  targetGazeX: number;
  targetGazeY: number;
  gazeX: number;
  gazeY: number;
  nextNodAt: number;
  nodImpulse: number;
  nodVelocity: number;
  emotionBlend: number;
  open: number;
  wide: number;
  round: number;
  smile: number;
  cheekLift: number;
  innerBrowY: number;
  innerBrowPinch: number;
  outerBrowY: number;
  eyeSquint: number;
  eyeWide: number;
  noseFlare: number;
  headYaw: number;
  headPitch: number;
  headRoll: number;
  hairShiftX: number;
  hairShiftY: number;
  prevVisemeOpen: number;
  browBeat: number;
};

export function createExpressionEngine(now = performance.now()): ExpressionEngineState {
  return {
    lastNow: now,
    nextBlinkAt: now + 380,
    blinkProgress: 1,
    doubleBlinkQueued: false,
    nextSaccadeAt: now + 220,
    targetGazeX: 0.42,
    targetGazeY: 0.08,
    gazeX: 0,
    gazeY: 0,
    nextNodAt: now + 1600,
    nodImpulse: 0,
    nodVelocity: 0,
    emotionBlend: 0.5,
    open: 0,
    wide: 0,
    round: 0,
    smile: 0,
    cheekLift: 0,
    innerBrowY: 0,
    innerBrowPinch: 0,
    outerBrowY: 0,
    eyeSquint: 0,
    eyeWide: 0,
    noseFlare: 0,
    headYaw: 0,
    headPitch: 0,
    headRoll: 0,
    hairShiftX: 0,
    hairShiftY: 0,
    prevVisemeOpen: 0,
    browBeat: 0,
  };
}

export function stepExpressionEngine(
  st: ExpressionEngineState,
  now: number,
  opts: {
    active: boolean;
    listening: boolean;
    emotion: AvatarEmotion;
    visemes: AudioVisemes;
    hasVisibleTeeth: boolean;
    restMouthRatio: number;
    userFace?: UserFaceObservation | undefined;
    reducedMotion?: boolean | undefined;
  },
): ExpressionFrame {
  const dt = Math.min(0.05, Math.max(0.004, (now - st.lastNow) * 0.001));
  st.lastNow = now;
  const t = now * 0.001;
  const mScale = opts.reducedMotion ? 0.15 : 1.0;

  const { active, listening, emotion, visemes, userFace } = opts;
  const hasUserFace = !!userFace?.detected;

  // 1) Distinct, unmistakable human blink cadence (double-blink ~22% chance)
  if (now >= st.nextBlinkAt) {
    st.blinkProgress = 0;
    if (!st.doubleBlinkQueued && Math.random() < 0.22) {
      st.doubleBlinkQueued = true;
      st.nextBlinkAt = now + 320;
    } else {
      st.doubleBlinkQueued = false;
      const interval = active ? 1850 + Math.random() * 1750 : 1650 + Math.random() * 1950;
      st.nextBlinkAt = now + interval;
    }
  }
  let blink = 0;
  if (st.blinkProgress < 1) {
    // 400ms full eyelid sweep + unmistakable closed dwell
    st.blinkProgress = Math.min(1, st.blinkProgress + dt / 0.4);
    blink = blinkClosure(st.blinkProgress);
  }

  // 2) Authentic, crisp extraocular saccades & corneal scanning (never chases mouse cursor)
  if (now >= st.nextSaccadeAt) {
    if (active) {
      const thinkGlance = Math.random() < 0.36;
      st.targetGazeX = thinkGlance
        ? (Math.random() < 0.5 ? -1 : 1) * (0.68 + Math.random() * 0.34)
        : (Math.random() - 0.5) * 1.12;
      st.targetGazeY = thinkGlance ? 0.18 + Math.random() * 0.28 : (Math.random() - 0.5) * 0.56;
      st.nextSaccadeAt =
        now + (thinkGlance ? 440 + Math.random() * 420 : 540 + Math.random() * 820);
    } else if (listening) {
      st.targetGazeX = (Math.random() - 0.5) * 1.04;
      st.targetGazeY = (Math.random() - 0.5) * 0.52;
      st.nextSaccadeAt = now + 620 + Math.random() * 940;
    } else {
      st.targetGazeX = (Math.random() - 0.5) * 1.15;
      st.targetGazeY = (Math.random() - 0.5) * 0.58;
      st.nextSaccadeAt = now + 760 + Math.random() * 1100;
    }
  }

  const microDriftX = Math.sin(t * 1.45) * 0.12 * mScale;
  const microDriftY = Math.cos(t * 1.05) * 0.07 * mScale;

  // Subconscious gaze mirroring when user turns/looks toward a side
  const mirrorGazeX = hasUserFace ? userFace.headYaw * 0.36 + userFace.gazeX * 0.18 : 0;
  const mirrorGazeY = hasUserFace ? userFace.headPitch * 0.22 + userFace.gazeY * 0.14 : 0;

  const targetGx = Math.max(-1.05, Math.min(1.05, st.targetGazeX + microDriftX + mirrorGazeX));
  const targetGy = Math.max(-0.68, Math.min(0.68, st.targetGazeY + microDriftY + mirrorGazeY));

  const gazeAlpha = 1 - Math.exp(-dt * 28);
  st.gazeX += (targetGx - st.gazeX) * gazeAlpha;
  st.gazeY += (targetGy - st.gazeY) * gazeAlpha;

  // 3) Fluid expression carryover: expressive when speaking, decaying smoothly afterward (never resets abruptly)
  const targetBlend = active ? 1.0 : listening ? 0.52 : 0.44;
  const blendRate = active ? 6.2 : 0.45;
  st.emotionBlend += (targetBlend - st.emotionBlend) * (1 - Math.exp(-dt * blendRate));
  const eb = st.emotionBlend;

  const ambientWarmth = Math.max(0, Math.sin(t * 0.45)) * 0.14;
  const ambientBrow = Math.sin(t * 0.78) * 0.1 + Math.cos(t * 0.34) * 0.06;

  let emOpen = 0;
  let emWide = 0;
  let emRound = 0;
  let emSmile = ambientWarmth;
  let emCheek = ambientWarmth * 0.85;
  let emInnerBrowY = ambientBrow;
  let emInnerBrowPinch = 0;
  let emOuterBrowY = ambientBrow * 1.15;
  let emSquint = 0.05;
  let emEyeWide = 0;
  let emNoseFlare = 0;
  let emHeadTiltPitch = 0;

  if (emotion === "happy") {
    emSmile = (opts.hasVisibleTeeth ? 0.45 : 0.65) * eb + ambientWarmth * 0.4;
    emCheek = 0.86 * eb;
    emSquint = 0.44 * eb;
    emInnerBrowY = 0.32 * eb + ambientBrow * 0.4;
    emOuterBrowY = 0.38 * eb + ambientBrow * 0.4;
    emNoseFlare = 0.22 * eb;
  } else if (emotion === "laugh") {
    const chuckle = active ? 0.5 + 0.5 * Math.sin(t * 14.5) : 0;
    emSmile = (opts.hasVisibleTeeth ? 0.64 : 0.86) * eb;
    emWide = 0.42 * eb;
    emCheek = 1.0 * eb;
    emSquint = 0.64 * eb;
    emInnerBrowY = 0.42 * eb;
    emOuterBrowY = 0.52 * eb;
    emNoseFlare = 0.52 * eb;
    emOpen = active ? 0.24 + chuckle * 0.32 : 0;
  } else if (emotion === "angry") {
    emInnerBrowY = -1.05 * eb;
    emInnerBrowPinch = 1.05 * eb;
    emOuterBrowY = -0.34 * eb;
    emSquint = 0.55 * eb;
    emSmile = -0.62 * eb;
    emNoseFlare = 0.64 * eb;
    emHeadTiltPitch = -0.18 * eb;
  } else if (emotion === "surprised") {
    emInnerBrowY = 1.04 * eb;
    emOuterBrowY = 1.08 * eb;
    emEyeWide = 0.68 * eb;
    emRound = active ? 0.46 : 0.2 * eb;
    emNoseFlare = 0.36 * eb;
    emOpen = active ? 0.16 : 0.14 * eb;
    emHeadTiltPitch = 0.16 * eb;
  } else if (emotion === "fear") {
    emInnerBrowY = 0.84 * eb;
    emInnerBrowPinch = 0.58 * eb;
    emOuterBrowY = 0.38 * eb;
    emEyeWide = 0.6 * eb;
    emSmile = -0.36 * eb;
    emNoseFlare = 0.38 * eb;
  } else if (emotion === "sad") {
    emInnerBrowY = 0.88 * eb;
    emOuterBrowY = -0.54 * eb;
    emSmile = -0.56 * eb;
    emSquint = 0.28 * eb;
    emHeadTiltPitch = -0.22 * eb;
  }

  // Subconscious empathetic facial mirroring from user's camera expression
  if (hasUserFace) {
    const mirrorWeight = active ? 0.24 : 0.42;
    if (emotion !== "angry" && emotion !== "sad") {
      emSmile += userFace.smile * 0.38 * mirrorWeight;
      emCheek += userFace.smile * 0.34 * mirrorWeight;
      emSquint += userFace.smile * 0.18 * mirrorWeight;
    }
    if (emotion === "neutral" || emotion === "sad" || emotion === "angry") {
      emInnerBrowPinch += userFace.browFurrow * 0.32 * mirrorWeight;
      emInnerBrowY -= userFace.browFurrow * 0.22 * mirrorWeight;
    }
    if (emotion === "neutral" || emotion === "surprised") {
      emInnerBrowY += userFace.browRaise * 0.28 * mirrorWeight;
      emOuterBrowY += userFace.browRaise * 0.26 * mirrorWeight;
      if (userFace.emotion === "surprised") {
        emEyeWide += 0.22 * mirrorWeight;
      }
    }
  }

  // 4) Speech prosody & coarticulation
  const speechOpen = active ? Math.min(1, visemes.open * 0.94) : 0;
  const openDelta = Math.max(0, speechOpen - st.prevVisemeOpen);
  st.prevVisemeOpen = speechOpen;
  st.browBeat = Math.max(st.browBeat * Math.exp(-dt * 4.5), openDelta * 2.4);

  const gatedOpen = Math.max(speechOpen, emOpen);
  const motion = mouthMotion(
    gatedOpen < 0.025 ? 0 : gatedOpen,
    opts.hasVisibleTeeth,
    opts.restMouthRatio,
  );
  const targetOpen = motion.opening;
  const targetWide = active ? Math.max(visemes.wide, emWide) : emWide;
  const targetRound = active ? Math.max(visemes.round, emRound) : emRound;

  const openRate = targetOpen > st.open ? 13.5 : 8.5;
  const mouthAlpha = 1 - Math.exp(-dt * openRate);
  st.open += (targetOpen - st.open) * mouthAlpha;
  st.wide += (targetWide - st.wide) * (1 - Math.exp(-dt * 10));
  st.round += (targetRound - st.round) * (1 - Math.exp(-dt * 10));

  const speechBrowBoost = active ? Math.min(0.46, st.open * 0.24 + st.browBeat * 0.36) : 0;
  const exprAlpha = 1 - Math.exp(-dt * 7.8);
  st.innerBrowY +=
    ((emotion === "angry" ? emInnerBrowY : emInnerBrowY + speechBrowBoost) - st.innerBrowY) *
    exprAlpha;
  st.innerBrowPinch += (emInnerBrowPinch - st.innerBrowPinch) * exprAlpha;
  st.outerBrowY += (emOuterBrowY + speechBrowBoost - st.outerBrowY) * exprAlpha;
  st.eyeSquint += (emSquint - st.eyeSquint) * exprAlpha;
  st.eyeWide += (emEyeWide - st.eyeWide) * exprAlpha;
  st.cheekLift += (emCheek - st.cheekLift) * (1 - Math.exp(-dt * 6.8));
  st.smile += (emSmile - st.smile) * (1 - Math.exp(-dt * 6.8));
  st.noseFlare +=
    (Math.max(emNoseFlare, active ? visemes.level * 0.26 : 0) - st.noseFlare) *
    (1 - Math.exp(-dt * 9));

  // 5) Natural human 3D head perspective turns while speaking + subtle subconscious user head-turn mirroring
  if (listening && now >= st.nextNodAt) {
    st.nodImpulse = 0.72 + Math.random() * 0.45;
    st.nextNodAt = now + 1750 + Math.random() * 2100;
  } else if (!listening) {
    st.nodImpulse *= Math.exp(-dt * 2.5);
  }
  const nodWave = listening ? Math.sin(t * 4.6) * st.nodImpulse * 0.55 : 0;
  st.nodImpulse *= Math.exp(-dt * 2.2);

  const breath = Math.sin(t * 1.45) * 0.5 + 0.5;
  const gazeHeadLeadX = st.gazeX * 0.26;
  const gazeHeadLeadY = st.gazeY * 0.18;
  const speechNod = active ? Math.sin(t * 3.2) * st.browBeat * 0.38 : 0;

  // Subconscious side-turn mirroring: if user turns head left/right, avatar subtly follows toward that side
  const mirrorYaw = hasUserFace ? userFace.headYaw * 0.34 : 0;
  const mirrorPitch = hasUserFace ? userFace.headPitch * 0.24 : 0;

  const targetYaw =
    (Math.sin(t * 0.68) * (active ? 0.62 : 0.42) +
      Math.cos(t * 0.29) * (active ? 0.28 : 0.18) +
      (active ? Math.sin(t * 1.75) * visemes.level * 0.28 : 0) +
      gazeHeadLeadX +
      mirrorYaw) *
    mScale;

  const targetPitch =
    (Math.sin(t * 0.88) * 0.34 +
      Math.cos(t * 0.38) * 0.15 +
      (active ? -st.open * 0.12 - speechNod : nodWave * 1.35 + 0.06) +
      emHeadTiltPitch +
      gazeHeadLeadY +
      mirrorPitch) *
    mScale;

  const targetRoll =
    (Math.sin(t * 0.78) * 0.048 +
      Math.cos(t * 0.34) * 0.022 +
      (listening ? 0.022 : 0) +
      (emotion === "surprised" ? 0.025 * eb : 0) +
      (hasUserFace ? userFace.headYaw * 0.025 : 0)) *
    mScale;

  const headAlpha = 1 - Math.exp(-dt * 3.8);
  st.headYaw += (Math.max(-1.05, Math.min(1.05, targetYaw)) - st.headYaw) * headAlpha;
  st.headPitch += (Math.max(-0.78, Math.min(0.78, targetPitch)) - st.headPitch) * headAlpha;
  st.headRoll += (targetRoll - st.headRoll) * headAlpha;

  // Keep hair displacement virtually zero so hair never looks weird or warps background
  st.hairShiftX = 0;
  st.hairShiftY = 0;

  const browAsym = (active ? Math.sin(t * 2.15) * 0.24 : Math.sin(t * 0.72) * 0.1) * mScale;

  return {
    blink,
    open: st.open,
    wide: st.wide,
    round: st.round,
    smile: st.smile,
    cheekLift: st.cheekLift,
    innerBrowY: st.innerBrowY,
    innerBrowPinch: st.innerBrowPinch,
    outerBrowY: st.outerBrowY,
    browAsym,
    eyeSquint: st.eyeSquint,
    eyeWide: st.eyeWide,
    noseFlare: st.noseFlare,
    gazeX: st.gazeX,
    gazeY: st.gazeY,
    headYaw: st.headYaw,
    headPitch: st.headPitch,
    headRoll: st.headRoll,
    headTx: Math.sin(t * 0.5) * 0.002 * mScale,
    headTy: ((breath - 0.5) * 0.0024 + nodWave * 0.0048) * mScale,
    torsoBreath: (breath - 0.5) * mScale,
    handShiftX: Math.sin(t * 1.18) * 0.002 * mScale,
    handShiftY: ((breath - 0.5) * 0.0025 + (active ? visemes.level * 0.004 : 0)) * mScale,
    hairShiftX: 0,
    hairShiftY: 0,
  };
}
