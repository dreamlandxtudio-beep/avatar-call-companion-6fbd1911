// 100% local client-side facial expression, ocular saccade, eye-contact, wink, blink, 3D head/neck/shoulder & lip-sync kinematics engine.
import type { AvatarEmotion } from "./ai";
import {
  createObservedReaction,
  stepObservedReaction,
  type ObservedReaction,
} from "./observed-reaction";
import { blinkClosure, mouthMotion } from "./avatar-motion";
import type { UserFaceObservation } from "./userCamera";
import type { AudioVisemes } from "./voice";

export type ExpressionFrame = {
  blink: number;
  blinkR: number;
  blinkL: number;
  open: number;
  wide: number;
  round: number;
  smile: number;
  smileR: number;
  smileL: number;
  mouthSkewX: number;
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
  torsoSwayX: number;
  shoulderTilt: number;
  handShiftX: number;
  handShiftY: number;
  hairShiftX: number;
  hairShiftY: number;
};

export type ExpressionEngineState = {
  reaction: ObservedReaction;
  lastNow: number;
  nextBlinkAt: number;
  blinkProgress: number;
  doubleBlinkQueued: boolean;
  nextWinkAt: number;
  winkStartAt: number;
  winkUntil: number;
  winkSide: 1 | -1; // 1 = right eye, -1 = left eye
  nextSaccadeAt: number;
  targetGazeX: number;
  targetGazeY: number;
  gazeX: number;
  gazeY: number;
  nextNodAt: number;
  nodImpulse: number;
  nodVelocity: number;
  nextBrowAccentAt: number;
  browAccentUntil: number;
  browAccentSide: number;
  nextFurrowAt: number;
  furrowUntil: number;
  nextSmirkAt: number;
  smirkUntil: number;
  smirkSide: 1 | -1;
  wasActive: boolean;
  wasListening: boolean;
  thinkUntil: number;
  thinkDir: 1 | -1;
  preSpeechUntil: number;
  emotionBlend: number;
  open: number;
  wide: number;
  round: number;
  smile: number;
  smileR: number;
  smileL: number;
  mouthSkewX: number;
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
  torsoSwayX: number;
  shoulderTilt: number;
  hairShiftX: number;
  hairShiftY: number;
  prevVisemeOpen: number;
  browBeat: number;
};

export function createExpressionEngine(now = performance.now()): ExpressionEngineState {
  return {
    reaction: createObservedReaction(),
    lastNow: now,
    nextBlinkAt: now + 480,
    blinkProgress: 1,
    doubleBlinkQueued: false,
    nextWinkAt: now + 11500,
    winkStartAt: 0,
    winkUntil: 0,
    winkSide: 1,
    nextSaccadeAt: now + 260,
    targetGazeX: 0.0,
    targetGazeY: 0.0,
    gazeX: 0,
    gazeY: 0,
    nextNodAt: now + 1600,
    nodImpulse: 0,
    nodVelocity: 0,
    nextBrowAccentAt: now + 7200,
    browAccentUntil: 0,
    browAccentSide: 1,
    nextFurrowAt: now + 9500,
    furrowUntil: 0,
    nextSmirkAt: now + 6200,
    smirkUntil: 0,
    smirkSide: 1,
    wasActive: false,
    wasListening: false,
    thinkUntil: 0,
    thinkDir: 1,
    preSpeechUntil: 0,
    emotionBlend: 0.5,
    open: 0,
    wide: 0,
    round: 0,
    smile: 0,
    smileR: 0,
    smileL: 0,
    mouthSkewX: 0,
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
    torsoSwayX: 0,
    shoulderTilt: 0,
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
    mirrorMode?: boolean | undefined;
    animeStyle?: boolean | undefined;
  },
): ExpressionFrame {
  const dt = Math.min(0.05, Math.max(0.004, (now - st.lastNow) * 0.001));
  st.lastNow = now;
  const t = now * 0.001;
  const mScale = opts.reducedMotion ? 0.15 : 1.0;

  const { active, listening, emotion, visemes, userFace } = opts;
  const isAnime = !!opts.animeStyle;
  const mouthAmp = isAnime ? 1.24 : 1.0;
  const wideAmp = isAnime ? 1.22 : 1.0;
  const browAmp = isAnime ? 1.18 : 1.0;
  const hasUserFace = !!userFace?.detected && now - userFace.updatedAt < 1000;
  const directMirror = !!opts.mirrorMode && hasUserFace;
  const observedSmile = directMirror ? 0 : stepObservedReaction(st.reaction, now, userFace);

  // State transitions: thoughtful brief pause/glance right as speaking starts or between listening -> speaking
  if (active && !st.wasActive) {
    st.thinkUntil = now + 520;
    st.thinkDir = Math.random() < 0.5 ? -1 : 1;
    st.preSpeechUntil = now + 420;
  } else if (!active && st.wasActive && !listening) {
    st.thinkUntil = now + 380;
  }
  if (listening && !st.wasListening) {
    // Immediate attentive eye contact when the user speaks
    st.targetGazeX = (Math.random() - 0.5) * 0.08;
    st.targetGazeY = (Math.random() - 0.5) * 0.05;
    st.nextSaccadeAt = now + 950;
  }
  st.wasActive = active;
  st.wasListening = listening;

  const isThinkingPause =
    !directMirror &&
    ((active && now < st.thinkUntil) || (active && visemes.level < 0.075 && emotion !== "laugh"));

  // 1) Distinct, unmistakable human blink cadence (double-blink ~22% chance)
  if (now >= st.nextBlinkAt) {
    st.blinkProgress = 0;
    if (!st.doubleBlinkQueued && Math.random() < 0.22) {
      st.doubleBlinkQueued = true;
      st.nextBlinkAt = now + 320;
    } else {
      st.doubleBlinkQueued = false;
      const interval = active ? 2100 + Math.random() * 1900 : 1850 + Math.random() * 2100;
      st.nextBlinkAt = now + interval;
    }
  }
  let baseBlink = 0;
  if (st.blinkProgress < 1) {
    // 400ms full eyelid sweep + clean closed dwell
    st.blinkProgress = Math.min(1, st.blinkProgress + dt / 0.4);
    baseBlink = blinkClosure(st.blinkProgress);
  }

  // Occasional playful wink ONLY in warm/playful contexts and ALWAYS while looking directly at the user
  if (
    !directMirror &&
    !opts.reducedMotion &&
    now >= st.nextWinkAt &&
    now >= st.winkUntil &&
    emotion !== "sad" &&
    emotion !== "angry" &&
    emotion !== "fear"
  ) {
    // Start 220ms gaze-lock lead-in so eyes center directly onto user BEFORE winking
    st.winkStartAt = now + 220;
    st.winkUntil = st.winkStartAt + 420;
    st.winkSide = Math.random() < 0.5 ? 1 : -1;
    st.nextWinkAt = now + 13500 + Math.random() * 11000;
  }
  const winkPreparing = !directMirror && now < st.winkUntil && now >= st.winkStartAt - 220;
  let winkStrength = 0;
  if (!directMirror && now >= st.winkStartAt && now < st.winkUntil) {
    const wu = (now - st.winkStartAt) / 420;
    winkStrength = Math.sin(Math.min(1, Math.max(0, wu)) * Math.PI) * 0.96;
  }

  // 2) Eye contact & extraocular saccades:
  //    - Looking at the user during listening or winking
  //    - Natural left/right thoughtful glances when preparing a thought or pondering
  if (winkPreparing) {
    st.targetGazeX = 0;
    st.targetGazeY = 0;
  } else if (now >= st.nextSaccadeAt) {
    if (listening) {
      // Attentive eye contact while user talks (~90% directly at user, occasional tiny natural micro-fixation)
      const briefShift = Math.random() < 0.12;
      st.targetGazeX = briefShift
        ? (Math.random() < 0.5 ? -1 : 1) * (0.14 + Math.random() * 0.12)
        : (Math.random() - 0.5) * 0.09;
      st.targetGazeY = (Math.random() - 0.5) * 0.06;
      st.nextSaccadeAt = now + (briefShift ? 420 : 1200 + Math.random() * 1100);
    } else if (active) {
      const thinkGlance = isThinkingPause || Math.random() < 0.42;
      st.targetGazeX = thinkGlance
        ? st.thinkDir * (0.48 + Math.random() * 0.42)
        : (Math.random() - 0.5) * 0.18;
      st.targetGazeY = thinkGlance ? 0.08 + Math.random() * 0.2 : (Math.random() - 0.5) * 0.1;
      if (Math.random() < 0.45) st.thinkDir = st.thinkDir === 1 ? -1 : 1;
      st.nextSaccadeAt =
        now + (thinkGlance ? 560 + Math.random() * 440 : 950 + Math.random() * 1050);
    } else {
      const idleGlance = Math.random() < 0.28;
      st.targetGazeX = idleGlance
        ? (Math.random() < 0.5 ? -1 : 1) * (0.36 + Math.random() * 0.34)
        : (Math.random() - 0.5) * 0.14;
      st.targetGazeY = (Math.random() - 0.5) * 0.12;
      st.nextSaccadeAt = now + 900 + Math.random() * 1150;
    }
  }

  const microDriftX = (listening || winkPreparing ? 0.015 : 0.032) * Math.sin(t * 1.6) * mScale;
  const microDriftY = (listening || winkPreparing ? 0.01 : 0.02) * Math.cos(t * 1.15) * mScale;

  const mirrorGazeX = directMirror ? userFace.gazeX : 0;
  const mirrorGazeY = directMirror ? userFace.gazeY : 0;

  const targetGx = Math.max(
    -1.05,
    Math.min(
      1.05,
      (directMirror ? 0 : winkPreparing ? 0 : st.targetGazeX + microDriftX) + mirrorGazeX,
    ),
  );
  const targetGy = Math.max(
    -0.68,
    Math.min(
      0.68,
      (directMirror ? 0 : winkPreparing ? 0 : st.targetGazeY + microDriftY) + mirrorGazeY,
    ),
  );

  let blinkR = baseBlink;
  let blinkL = baseBlink;
  if (directMirror) {
    blinkR = userFace.blinkRight ?? userFace.blink ?? 0;
    blinkL = userFace.blinkLeft ?? userFace.blink ?? 0;
  } else if (winkStrength > 0.02) {
    if (st.winkSide > 0) {
      blinkR = Math.max(baseBlink, winkStrength);
      blinkL = Math.max(baseBlink, winkStrength * 0.08);
    } else {
      blinkL = Math.max(baseBlink, winkStrength);
      blinkR = Math.max(baseBlink, winkStrength * 0.08);
    }
  }
  const blink = (blinkR + blinkL) * 0.5;

  const gazeAlpha = 1 - Math.exp(-dt * (winkPreparing ? 34 : 26));
  st.gazeX += (targetGx - st.gazeX) * gazeAlpha;
  st.gazeY += (targetGy - st.gazeY) * gazeAlpha;

  // 3) Fluid expression carryover + natural micro-gestures (frown, smirk, laugh, pre-speech parting)
  const targetBlend = active ? 1.0 : listening ? 0.62 : 0.48;
  const blendRate = active ? 6.5 : 0.7;
  st.emotionBlend += (targetBlend - st.emotionBlend) * (1 - Math.exp(-dt * blendRate));
  const eb = st.emotionBlend;

  const ambientWarmth = Math.max(0, Math.sin(t * 0.45)) * 0.18;
  const ambientBrow = Math.sin(t * 0.78) * 0.1 + Math.cos(t * 0.34) * 0.06;

  // Occasional subtle thoughtful brow furrow (conciso fruncir el ceño al pensar o concentrarse)
  if (!directMirror && now >= st.nextFurrowAt) {
    st.furrowUntil = now + 950;
    st.nextFurrowAt = now + 9500 + Math.random() * 8500;
  }
  let gestureFurrow = 0;
  if (!directMirror && now < st.furrowUntil && emotion !== "happy" && emotion !== "laugh") {
    const fu = 1 - (st.furrowUntil - now) / 950;
    gestureFurrow = Math.sin(Math.min(1, Math.max(0, fu)) * Math.PI) * 0.36 * mScale;
  }
  if (isThinkingPause && emotion === "neutral") {
    gestureFurrow = Math.max(gestureFurrow, 0.22 * mScale);
  }

  // Occasional asymmetric corner smirk / lateral mouth pull (mover una comisura sin la otra)
  if (!directMirror && now >= st.nextSmirkAt) {
    st.smirkUntil = now + 1350;
    st.smirkSide = Math.random() < 0.5 ? 1 : -1;
    st.nextSmirkAt = now + 6800 + Math.random() * 7200;
  }
  let smirkAmount = 0;
  if (
    !directMirror &&
    now < st.smirkUntil &&
    emotion !== "sad" &&
    emotion !== "angry" &&
    emotion !== "fear"
  ) {
    const su = 1 - (st.smirkUntil - now) / 1350;
    smirkAmount = Math.sin(Math.min(1, Math.max(0, su)) * Math.PI) * 0.34 * mScale;
  }

  const speechWarmth =
    active && emotion !== "sad" && emotion !== "angry" && emotion !== "fear"
      ? (0.18 + Math.max(0, Math.sin(t * 1.35)) * 0.26) * (0.45 + visemes.level * 0.55)
      : 0;

  // Gentle parted lips while thinking/preparing a response or pondering quietly
  const preSpeechOpen =
    !directMirror &&
    emotion !== "angry" &&
    ((active && (now < st.preSpeechUntil || visemes.level < 0.08)) ||
      (listening && Math.sin(t * 0.9) > 0.72))
      ? 0.11 * mScale
      : 0;

  let emOpen = preSpeechOpen;
  let emWide = speechWarmth * 0.35;
  let emRound = 0;
  let emSmile = ambientWarmth + speechWarmth * 0.95;
  let emCheek = ambientWarmth * 0.85 + speechWarmth * 0.9;
  let emInnerBrowY = ambientBrow - gestureFurrow * 0.45;
  let emInnerBrowPinch = gestureFurrow;
  let emOuterBrowY = ambientBrow * 1.15 - gestureFurrow * 0.18;
  let emSquint = 0.05 + speechWarmth * 0.44 + winkStrength * 0.28;
  let emEyeWide = 0;
  let emNoseFlare = 0;
  let emHeadTiltPitch = 0;
  let emMouthSkewX = smirkAmount * st.smirkSide * 0.75;
  let smirkBias = smirkAmount * st.smirkSide;

  if (emotion === "happy") {
    emSmile = (opts.hasVisibleTeeth ? 0.6 : 0.78) * eb + ambientWarmth * 0.4 + speechWarmth * 0.4;
    emWide = 0.28 * eb;
    emCheek = 0.98 * eb;
    emSquint = 0.54 * eb;
    emInnerBrowY = 0.34 * eb + ambientBrow * 0.4;
    emOuterBrowY = 0.36 * eb + ambientBrow * 0.4;
    emNoseFlare = 0.22 * eb;
  } else if (emotion === "laugh") {
    const chuckle = active ? 0.5 + 0.5 * Math.sin(t * 14.5) : 0.5 + 0.5 * Math.sin(t * 7.2);
    emSmile = (opts.hasVisibleTeeth ? 0.82 : 1.0) * eb;
    emWide = 0.62 * eb;
    emCheek = 1.24 * eb;
    emSquint = 0.86 * eb;
    emInnerBrowY = 0.52 * eb;
    emOuterBrowY = 0.58 * eb;
    emNoseFlare = 0.58 * eb;
    emOpen = Math.max(emOpen, active ? 0.36 + chuckle * 0.36 : 0.32 * eb + chuckle * 0.14);
  } else if (emotion === "angry") {
    emInnerBrowY = -1.05 * eb;
    emInnerBrowPinch = 1.05 * eb;
    emOuterBrowY = -0.34 * eb;
    emSquint = 0.58 * eb;
    emSmile = -0.62 * eb;
    emNoseFlare = 0.64 * eb;
    emHeadTiltPitch = -0.18 * eb;
    emMouthSkewX = 0;
    smirkBias = 0;
  } else if (emotion === "surprised") {
    emInnerBrowY = 1.04 * eb;
    emOuterBrowY = 1.04 * eb;
    emEyeWide = 0.68 * eb;
    emRound = active ? 0.46 : 0.2 * eb;
    emNoseFlare = 0.36 * eb;
    emOpen = Math.max(emOpen, active ? 0.18 : 0.16 * eb);
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
    emMouthSkewX = 0;
    smirkBias = 0;
  }

  // Empathetic subtle response from user's camera expression
  emSmile += observedSmile;
  emCheek += observedSmile * 0.85;
  emSquint += observedSmile * 0.42;
  emOpen = Math.max(emOpen, observedSmile * 0.18);

  let emSmileR =
    emSmile + Math.max(0, smirkBias) * 0.72 + (st.winkSide > 0 ? winkStrength * 0.24 : 0);
  let emSmileL =
    emSmile + Math.max(0, -smirkBias) * 0.72 + (st.winkSide < 0 ? winkStrength * 0.24 : 0);

  if (directMirror) {
    const mirrorWeight = 1;
    emSmile = userFace.smile * 0.9 - (userFace.mouthFrown ?? 0) * 0.65;
    emSmileR = (userFace.smileRight ?? userFace.smile) * 0.95 - (userFace.mouthFrown ?? 0) * 0.65;
    emSmileL = (userFace.smileLeft ?? userFace.smile) * 0.95 - (userFace.mouthFrown ?? 0) * 0.65;
    emMouthSkewX = userFace.mouthSkewX ?? 0;
    emCheek = userFace.smile * 0.72 * mirrorWeight;
    emSquint = (userFace.eyeSquint ?? 0) * mirrorWeight + userFace.smile * 0.35;
    emOpen = Math.max(emOpen, userFace.mouthOpen * mirrorWeight);
    emInnerBrowY = (userFace.browInnerUp ?? userFace.browRaise) * 0.9 - userFace.browFurrow * 0.7;
    emInnerBrowPinch = userFace.browFurrow;
    emOuterBrowY = userFace.browRaise * 0.8 - userFace.browFurrow * 0.3;
    emNoseFlare = userFace.noseSneer ?? 0;
    emEyeWide = userFace.eyeWide ?? 0;
  }

  // 4) Speech prosody & coarticulation
  const speechOpen = active ? Math.min(1, visemes.open * 0.94) : 0;
  const openDelta = Math.max(0, speechOpen - st.prevVisemeOpen);
  st.prevVisemeOpen = speechOpen;
  st.browBeat = Math.max(st.browBeat * Math.exp(-dt * 4.5), openDelta * 2.4);

  const gatedOpen = Math.max(speechOpen, emOpen);
  const motion = mouthMotion(
    gatedOpen < 0.025 ? 0 : Math.min(1, gatedOpen * mouthAmp),
    opts.hasVisibleTeeth,
    opts.restMouthRatio,
  );
  const targetOpen = Math.min(1, motion.opening * (isAnime ? 1.16 : 1.0));
  const targetWide = Math.min(1, (active ? Math.max(visemes.wide, emWide) : emWide) * wideAmp);
  const targetRound = active ? Math.max(visemes.round, emRound) : emRound;

  const openRate = targetOpen > st.open ? 13.5 : 8.5;
  const mouthAlpha = 1 - Math.exp(-dt * openRate);
  st.open += (targetOpen - st.open) * mouthAlpha;
  st.wide += (targetWide - st.wide) * (1 - Math.exp(-dt * 10));
  st.round += (targetRound - st.round) * (1 - Math.exp(-dt * 10));

  const speechBrowBoost = active ? Math.min(0.46, st.open * 0.24 + st.browBeat * 0.36) : 0;
  const exprAlpha = 1 - Math.exp(-dt * 7.8);
  st.innerBrowY +=
    ((emotion === "angry" ? emInnerBrowY : emInnerBrowY + speechBrowBoost) * browAmp -
      st.innerBrowY) *
    exprAlpha;
  st.innerBrowPinch += (emInnerBrowPinch * browAmp - st.innerBrowPinch) * exprAlpha;
  st.outerBrowY += ((emOuterBrowY + speechBrowBoost) * browAmp - st.outerBrowY) * exprAlpha;
  st.eyeSquint += (emSquint - st.eyeSquint) * exprAlpha;
  st.eyeWide += (Math.min(1, emEyeWide + (isAnime && active ? 0.1 : 0)) - st.eyeWide) * exprAlpha;
  st.cheekLift += (emCheek - st.cheekLift) * (1 - Math.exp(-dt * 6.8));
  const smileAlpha = 1 - Math.exp(-dt * 7.2);
  const sAmp = isAnime ? 1.12 : 1.0;
  st.smileR += (emSmileR * sAmp - st.smileR) * smileAlpha;
  st.smileL += (emSmileL * sAmp - st.smileL) * smileAlpha;
  st.smile = (st.smileR + st.smileL) * 0.5;
  st.mouthSkewX += (emMouthSkewX - st.mouthSkewX) * smileAlpha;
  st.noseFlare +=
    (Math.max(emNoseFlare, active ? visemes.level * 0.26 : 0) - st.noseFlare) *
    (1 - Math.exp(-dt * 9));

  // 5) Organic 3D head, neck & shoulder motion (richer yet always believable)
  if (listening && now >= st.nextNodAt) {
    st.nodImpulse = 0.85 + Math.random() * 0.45;
    st.nextNodAt = now + 1500 + Math.random() * 1850;
  } else if (!listening) {
    st.nodImpulse *= Math.exp(-dt * 2.5);
  }
  const nodWave = listening ? Math.sin(t * 4.6) * st.nodImpulse * 0.62 : 0;
  st.nodImpulse *= Math.exp(-dt * 2.2);

  const breath = Math.sin(t * 1.45) * 0.5 + 0.5;
  const gazeHeadLeadX = winkPreparing ? 0 : st.gazeX * 0.34;
  const gazeHeadLeadY = winkPreparing ? 0 : st.gazeY * 0.24;
  const speechNod = active ? Math.sin(t * 3.2) * st.browBeat * 0.44 : 0;
  const chuckleBounce = emotion === "laugh" ? Math.sin(t * 11.5) * 0.18 : 0;

  const mirrorYaw = directMirror ? userFace.headYaw : 0;
  const mirrorPitch = directMirror ? userFace.headPitch : 0;

  const targetYaw = directMirror
    ? mirrorYaw * mScale
    : ((winkPreparing ? 0.15 : 1) *
        (Math.sin(t * 0.62) * (active ? 0.64 : 0.46) + Math.cos(t * 0.27) * (active ? 0.3 : 0.2)) +
        (active ? Math.sin(t * 1.75) * visemes.level * 0.3 : 0) +
        gazeHeadLeadX +
        mirrorYaw) *
      mScale;

  const targetPitch = directMirror
    ? mirrorPitch * mScale
    : (Math.sin(t * 0.84) * 0.36 +
        Math.cos(t * 0.36) * 0.18 +
        (active ? -st.open * 0.12 - speechNod : nodWave * 1.38 + 0.05) +
        chuckleBounce +
        emHeadTiltPitch +
        gazeHeadLeadY +
        mirrorPitch) *
      mScale;

  const targetRoll = directMirror
    ? (userFace?.headRoll ?? 0) * mScale
    : (Math.sin(t * 0.72) * 0.056 +
        Math.cos(t * 0.31) * 0.026 +
        (listening ? 0.028 : 0) +
        (winkStrength > 0.1 ? st.winkSide * 0.032 : 0) +
        (emotion === "surprised" ? 0.028 * eb : 0)) *
      mScale;

  const headAlpha = 1 - Math.exp(-dt * 4.2);
  st.headYaw += (Math.max(-0.98, Math.min(0.98, targetYaw)) - st.headYaw) * headAlpha;
  st.headPitch += (Math.max(-0.76, Math.min(0.76, targetPitch)) - st.headPitch) * headAlpha;
  st.headRoll += (targetRoll - st.headRoll) * headAlpha;

  const targetTorsoSwayX = directMirror
    ? mirrorYaw * 0.35 * mScale
    : (Math.sin(t * 0.42) * 0.48 + st.headYaw * 0.28) * mScale;
  const targetShoulderTilt = directMirror
    ? (userFace?.headRoll ?? 0) * 0.6 * mScale
    : (Math.sin(t * 0.58) * 0.45 + st.headRoll * 2.2) * mScale;
  const torsoAlpha = 1 - Math.exp(-dt * 3.0);
  st.torsoSwayX += (targetTorsoSwayX - st.torsoSwayX) * torsoAlpha;
  st.shoulderTilt += (targetShoulderTilt - st.shoulderTilt) * torsoAlpha;

  // Keep hair displacement zero; hair follows smoothly through the head liquify envelope
  st.hairShiftX = 0;
  st.hairShiftY = 0;

  // Eyebrows normally rise and lower symmetrically together; rare brief micro-accent every ~7-14s
  if (now >= st.nextBrowAccentAt) {
    st.browAccentUntil = now + 1150;
    st.browAccentSide = Math.random() < 0.5 ? 1 : -1;
    st.nextBrowAccentAt = now + 7200 + Math.random() * 6800;
  }
  let browAsym = 0;
  if (now < st.browAccentUntil) {
    const u = Math.max(0, Math.min(1, 1 - (st.browAccentUntil - now) / 1150));
    browAsym = Math.sin(u * Math.PI) * 0.15 * st.browAccentSide * mScale;
  }

  return {
    blink,
    blinkR,
    blinkL,
    open: st.open,
    wide: st.wide,
    round: st.round,
    smile: st.smile,
    smileR: st.smileR,
    smileL: st.smileL,
    mouthSkewX: st.mouthSkewX,
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
    headTx: Math.sin(t * 0.5) * 0.0028 * mScale,
    headTy: ((breath - 0.5) * 0.0032 + nodWave * 0.0056) * mScale,
    torsoBreath: (breath - 0.5) * mScale,
    torsoSwayX: st.torsoSwayX,
    shoulderTilt: st.shoulderTilt,
    handShiftX: Math.sin(t * 1.18) * 0.0024 * mScale,
    handShiftY: ((breath - 0.5) * 0.003 + (active ? visemes.level * 0.0045 : 0)) * mScale,
    hairShiftX: 0,
    hairShiftY: 0,
  };
}
