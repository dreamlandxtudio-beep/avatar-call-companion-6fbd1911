// 100% local client-side facial expression, ocular saccade, eye-contact, wink, blink, 3D head/neck/shoulder & lip-sync kinematics engine.
import type { AvatarEmotion } from "./ai";
import {
  createObservedEmpathy,
  createObservedReaction,
  stepObservedEmpathy,
  stepObservedReaction,
  type ObservedEmpathyState,
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
  lipPress: number;
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
  forwardLean: number;
  tongueOut: number;
};

export type ExpressionPose =
  | "neutral"
  | "thinkSide"
  | "curiousTilt"
  | "attentiveWarm"
  | "browInterest"
  | "thoughtfulFurrow"
  | "gentleSmirk"
  | "softPonder";

export type ExpressionEngineState = {
  reaction: ObservedReaction;
  empathy: ObservedEmpathyState;
  lastNow: number;
  nextBlinkAt: number;
  blinkProgress: number;
  doubleBlinkQueued: boolean;
  nextWinkAt: number;
  winkStartAt: number;
  winkUntil: number;
  winkSide: 1 | -1; // 1 = right eye, -1 = left eye
  nextSaccadeAt: number;
  listeningGlanceActive: boolean;
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
  poseMode: ExpressionPose;
  poseUntil: number;
  nextPoseAt: number;
  poseDir: 1 | -1;
  wasActive: boolean;
  wasListening: boolean;
  thinkUntil: number;
  thinkDir: 1 | -1;
  preSpeechUntil: number;
  emotionBlend: number;
  open: number;
  wide: number;
  round: number;
  lipPress: number;
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
  forwardLean: number;
  tongueOut: number;
  nextTongueAt: number;
  tongueUntil: number;
  prevVisemeOpen: number;
  browBeat: number;
};

function pickExpressionPose(preferThink: boolean): ExpressionPose {
  const r = Math.random();
  if (preferThink) {
    if (r < 0.52) return "thinkSide";
    if (r < 0.68) return "softPonder";
    if (r < 0.8) return "thoughtfulFurrow";
    if (r < 0.9) return "curiousTilt";
    return "browInterest";
  }
  if (r < 0.44) return "thinkSide";
  if (r < 0.56) return "softPonder";
  if (r < 0.68) return "curiousTilt";
  if (r < 0.78) return "attentiveWarm";
  if (r < 0.87) return "browInterest";
  if (r < 0.94) return "thoughtfulFurrow";
  return "gentleSmirk";
}

export function createExpressionEngine(now = performance.now()): ExpressionEngineState {
  return {
    reaction: createObservedReaction(),
    empathy: createObservedEmpathy(now),
    lastNow: now,
    nextBlinkAt: now + 620,
    blinkProgress: 1,
    doubleBlinkQueued: false,
    nextWinkAt: now + 48000 + Math.random() * 24000,
    winkStartAt: 0,
    winkUntil: 0,
    winkSide: 1,
    nextSaccadeAt: now + 380,
    listeningGlanceActive: false,
    targetGazeX: 0.0,
    targetGazeY: 0.0,
    gazeX: 0,
    gazeY: 0,
    nextNodAt: now + 1600,
    nodImpulse: 0,
    nodVelocity: 0,
    nextBrowAccentAt: now + 6500,
    browAccentUntil: 0,
    browAccentSide: 1,
    nextFurrowAt: now + 8200,
    furrowUntil: 0,
    nextSmirkAt: now + 9800,
    smirkUntil: 0,
    smirkSide: 1,
    poseMode: "thinkSide",
    poseUntil: now + 2600,
    nextPoseAt: now + 4800,
    poseDir: 1,
    wasActive: false,
    wasListening: false,
    thinkUntil: now + 1400,
    thinkDir: 1,
    preSpeechUntil: 0,
    emotionBlend: 0.5,
    open: 0,
    wide: 0,
    round: 0,
    lipPress: 0,
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
    forwardLean: 0,
    tongueOut: 0,
    nextTongueAt: now + 28000 + Math.random() * 22000,
    tongueUntil: 0,
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
  const mouthAmp = isAnime ? 1.22 : 1.0;
  const wideAmp = isAnime ? 1.2 : 1.0;
  const browAmp = isAnime ? 1.12 : 1.0;
  const hasUserFace = !!userFace?.detected && now - userFace.updatedAt < 1000;
  const directMirror = !!opts.mirrorMode && hasUserFace;
  const observedSmile = directMirror ? 0 : stepObservedReaction(st.reaction, now, userFace);
  const empathy = directMirror
    ? {
        smile: 0,
        browFurrow: 0,
        browRaise: 0,
        frown: 0,
        eyeWide: 0,
        noseScrunch: 0,
        headYaw: 0,
        headPitch: 0,
        headRoll: 0,
      }
    : stepObservedEmpathy(st.empathy, now, userFace);

  // State transitions: thoughtful brief pause/glance when speaking starts; immediate direct eye contact on listening
  if (active && !st.wasActive) {
    st.thinkUntil = now + 720;
    st.thinkDir = Math.random() < 0.5 ? -1 : 1;
    st.preSpeechUntil = now + 420;
    st.poseMode = "thinkSide";
    st.poseDir = st.thinkDir;
    st.poseUntil = now + 1100;
    st.nextPoseAt = now + 2600 + Math.random() * 1800;
  } else if (!active && st.wasActive && !listening) {
    st.thinkUntil = now + 950;
    st.poseMode = "thinkSide";
    st.poseDir = Math.random() < 0.5 ? -1 : 1;
    st.poseUntil = now + 2200;
  }
  if (listening && !st.wasListening) {
    // Always re-anchor eye contact onto user immediately when listening begins
    st.listeningGlanceActive = false;
    st.targetGazeX = 0;
    st.targetGazeY = 0;
    st.nextSaccadeAt = now + 2400 + Math.random() * 2000;
  }
  st.wasActive = active;
  st.wasListening = listening;

  // Cycle expression repertoire naturally during idle & active conversation
  if (!directMirror && !listening && now >= st.nextPoseAt) {
    st.poseMode = pickExpressionPose(active && visemes.level < 0.1);
    st.poseDir = Math.random() < 0.5 ? -1 : 1;
    const dur =
      st.poseMode === "thinkSide" ? 2100 + Math.random() * 1400 : 1500 + Math.random() * 1200;
    st.poseUntil = now + dur;
    st.nextPoseAt = now + dur + 1600 + Math.random() * 2000;
    if (st.poseMode === "thinkSide" || st.poseMode === "softPonder") {
      st.thinkDir = st.poseDir;
    }
  }
  const activePose: ExpressionPose =
    !directMirror && !listening && now < st.poseUntil ? st.poseMode : "neutral";
  const isSideThinking =
    !directMirror &&
    !listening &&
    (activePose === "thinkSide" ||
      (active && now < st.thinkUntil) ||
      (active && visemes.level < 0.065 && emotion !== "laugh" && activePose === "softPonder"));

  // 1) Distinct natural blink cadence (double-blink ~22% chance)
  if (now >= st.nextBlinkAt) {
    st.blinkProgress = 0;
    if (!st.doubleBlinkQueued && Math.random() < 0.22) {
      st.doubleBlinkQueued = true;
      st.nextBlinkAt = now + 320;
    } else {
      st.doubleBlinkQueued = false;
      const interval = active ? 2200 + Math.random() * 2000 : 2100 + Math.random() * 2200;
      st.nextBlinkAt = now + interval;
    }
  }
  let baseBlink = 0;
  if (st.blinkProgress < 1) {
    st.blinkProgress = Math.min(1, st.blinkProgress + dt / 0.4);
    baseBlink = blinkClosure(st.blinkProgress);
  }

  // Rare wink (~every 62-110s, suppressed while listening or negative emotions)
  if (
    !directMirror &&
    !opts.reducedMotion &&
    !listening &&
    now >= st.nextWinkAt &&
    now >= st.winkUntil &&
    emotion !== "sad" &&
    emotion !== "angry" &&
    emotion !== "fear"
  ) {
    st.winkStartAt = now + 220;
    st.winkUntil = st.winkStartAt + 420;
    st.winkSide = Math.random() < 0.5 ? 1 : -1;
    st.nextWinkAt = now + 62000 + Math.random() * 48000;
  } else if (listening && now >= st.nextWinkAt) {
    st.nextWinkAt = now + 12000 + Math.random() * 10000;
  }
  const winkPreparing = !directMirror && now < st.winkUntil && now >= st.winkStartAt - 220;
  let winkStrength = 0;
  if (!directMirror && now >= st.winkStartAt && now < st.winkUntil) {
    const wu = (now - st.winkStartAt) / 420;
    winkStrength = Math.sin(Math.min(1, Math.max(0, wu)) * Math.PI) * 0.96;
  }

  // 2) Eye contact & ocular saccades:
  //    - Listening: lock firmly on the user's eyes; at most an occasional brief glance (~380ms) then straight back to the eyes
  //    - Thinking / Idle: frequent deliberate side gaze paired with turning the head while thinking
  if (winkPreparing) {
    st.targetGazeX = 0;
    st.targetGazeY = 0;
    st.listeningGlanceActive = false;
  } else if (now >= st.nextSaccadeAt) {
    if (listening) {
      if (st.listeningGlanceActive) {
        // Always snap back directly to the user's eyes after any brief side glance
        st.listeningGlanceActive = false;
        st.targetGazeX = (Math.random() - 0.5) * 0.03;
        st.targetGazeY = (Math.random() - 0.5) * 0.02;
        st.nextSaccadeAt = now + 3800 + Math.random() * 3400;
      } else if (Math.random() < 0.16) {
        // Rare brief glance at something, always followed by immediate return to eye contact
        st.listeningGlanceActive = true;
        st.targetGazeX = (Math.random() < 0.5 ? -1 : 1) * (0.28 + Math.random() * 0.14);
        st.targetGazeY = (Math.random() - 0.5) * 0.14;
        st.nextSaccadeAt = now + 340 + Math.random() * 120;
      } else {
        st.targetGazeX = (Math.random() - 0.5) * 0.04;
        st.targetGazeY = (Math.random() - 0.5) * 0.03;
        st.nextSaccadeAt = now + 2200 + Math.random() * 1800;
      }
    } else if (isSideThinking || activePose === "softPonder") {
      st.targetGazeX = st.poseDir * (0.62 + Math.random() * 0.26);
      st.targetGazeY =
        activePose === "softPonder" ? -0.14 - Math.random() * 0.1 : 0.06 + Math.random() * 0.14;
      st.nextSaccadeAt = now + 850 + Math.random() * 650;
    } else if (active) {
      const thinkGlance = Math.random() < 0.42;
      st.targetGazeX = thinkGlance
        ? st.thinkDir * (0.48 + Math.random() * 0.36)
        : (Math.random() - 0.5) * 0.12;
      st.targetGazeY = thinkGlance ? 0.05 + Math.random() * 0.14 : (Math.random() - 0.5) * 0.06;
      if (Math.random() < 0.45) st.thinkDir = st.thinkDir === 1 ? -1 : 1;
      st.nextSaccadeAt =
        now + (thinkGlance ? 700 + Math.random() * 520 : 1200 + Math.random() * 1100);
    } else {
      const idleGlance = Math.random() < 0.48;
      st.targetGazeX = idleGlance
        ? st.poseDir * (0.46 + Math.random() * 0.34)
        : (Math.random() - 0.5) * 0.1;
      st.targetGazeY = (Math.random() - 0.5) * 0.1;
      st.nextSaccadeAt = now + 950 + Math.random() * 1050;
    }
  }

  const microDriftX = (listening || winkPreparing ? 0.008 : 0.025) * Math.sin(t * 1.55) * mScale;
  const microDriftY = (listening || winkPreparing ? 0.006 : 0.016) * Math.cos(t * 1.1) * mScale;

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

  const gazeAlpha = 1 - Math.exp(-dt * (winkPreparing || listening ? 30 : 24));
  st.gazeX += (targetGx - st.gazeX) * gazeAlpha;
  st.gazeY += (targetGy - st.gazeY) * gazeAlpha;

  // 3) Fluid expression carryover + expressive pose repertoire
  const targetBlend = active ? 1.0 : listening ? 0.64 : 0.52;
  const blendRate = active ? 6.5 : 0.8;
  st.emotionBlend += (targetBlend - st.emotionBlend) * (1 - Math.exp(-dt * blendRate));
  const eb = st.emotionBlend;

  const ambientWarmth = Math.max(0, Math.sin(t * 0.45)) * 0.16;
  const ambientBrow = Math.sin(t * 0.78) * 0.1 + Math.cos(t * 0.34) * 0.06;

  // Thoughtful brow furrow
  if (!directMirror && now >= st.nextFurrowAt) {
    st.furrowUntil = now + 1100;
    st.nextFurrowAt = now + 8200 + Math.random() * 7500;
  }
  let gestureFurrow = 0;
  if (
    !directMirror &&
    (now < st.furrowUntil || activePose === "thoughtfulFurrow") &&
    emotion !== "happy" &&
    emotion !== "laugh"
  ) {
    const fu = now < st.furrowUntil ? 1 - (st.furrowUntil - now) / 1100 : 0.5;
    gestureFurrow = Math.sin(Math.min(1, Math.max(0, fu)) * Math.PI) * 0.26 * mScale;
  }
  if (isSideThinking && emotion === "neutral") {
    gestureFurrow = Math.max(gestureFurrow, 0.16 * mScale);
  }

  // Asymmetric corner smirk
  if (!directMirror && now >= st.nextSmirkAt) {
    st.smirkUntil = now + 1350;
    st.smirkSide = Math.random() < 0.5 ? 1 : -1;
    st.nextSmirkAt = now + 7500 + Math.random() * 7500;
  }
  let smirkAmount = 0;
  if (
    !directMirror &&
    (now < st.smirkUntil || activePose === "gentleSmirk") &&
    emotion !== "sad" &&
    emotion !== "angry" &&
    emotion !== "fear"
  ) {
    const su = now < st.smirkUntil ? 1 - (st.smirkUntil - now) / 1350 : 0.5;
    smirkAmount = Math.sin(Math.min(1, Math.max(0, su)) * Math.PI) * 0.2 * mScale;
  }

  const poseWarm = activePose === "attentiveWarm" ? 0.2 * mScale : 0;
  const poseBrowLift =
    activePose === "browInterest"
      ? 0.18 * mScale
      : activePose === "curiousTilt"
        ? 0.12 * mScale
        : 0;
  const poseSquint =
    isSideThinking || activePose === "thoughtfulFurrow"
      ? 0.12 * mScale
      : activePose === "attentiveWarm"
        ? 0.14 * mScale
        : 0;

  const speechWarmth =
    active && emotion !== "sad" && emotion !== "angry" && emotion !== "fear"
      ? (0.12 + Math.max(0, Math.sin(t * 1.35)) * 0.16) * (0.45 + visemes.level * 0.55)
      : 0;

  const preSpeechOpen =
    !directMirror &&
    emotion !== "angry" &&
    active &&
    (now < st.preSpeechUntil || visemes.level < 0.08)
      ? 0.045 * mScale
      : 0;

  let emOpen = preSpeechOpen;
  let emWide = speechWarmth * 0.24;
  let emRound = activePose === "softPonder" ? 0.08 * mScale : 0;
  let emSmile = ambientWarmth * 0.65 + speechWarmth * 0.75 + poseWarm;
  let emCheek = ambientWarmth * 0.55 + speechWarmth * 0.65 + poseWarm * 0.75;
  let emInnerBrowY = ambientBrow * 0.75 + poseBrowLift - gestureFurrow * 0.35;
  let emInnerBrowPinch = gestureFurrow;
  let emOuterBrowY = ambientBrow * 0.85 + poseBrowLift * 0.85 - gestureFurrow * 0.12;
  let emSquint = speechWarmth * 0.32 + poseSquint + winkStrength * 0.22;
  let emEyeWide = activePose === "browInterest" ? 0.12 * mScale : 0;
  let emNoseFlare = 0;
  let emHeadTiltPitch = isSideThinking
    ? 0.05 * mScale
    : activePose === "softPonder"
      ? 0.08 * mScale
      : 0;
  let emMouthSkewX = smirkAmount * st.smirkSide * 0.55;
  let smirkBias = smirkAmount * st.smirkSide;

  if (emotion === "happy") {
    emSmile =
      (opts.hasVisibleTeeth ? 0.42 : 0.54) * eb + ambientWarmth * 0.22 + speechWarmth * 0.25;
    emWide = 0.18 * eb;
    emCheek = 0.56 * eb;
    emSquint = 0.34 * eb;
    emInnerBrowY = 0.18 * eb + ambientBrow * 0.25;
    emOuterBrowY = 0.2 * eb + ambientBrow * 0.25;
    emNoseFlare = 0.14 * eb;
  } else if (emotion === "laugh") {
    const chuckle = active ? 0.5 + 0.5 * Math.sin(t * 14.5) : 0.5 + 0.5 * Math.sin(t * 7.2);
    emSmile = (opts.hasVisibleTeeth ? 0.54 : 0.68) * eb;
    emWide = 0.34 * eb;
    emCheek = 0.72 * eb;
    emSquint = 0.54 * eb;
    emInnerBrowY = 0.26 * eb;
    emOuterBrowY = 0.3 * eb;
    emNoseFlare = 0.3 * eb;
    emOpen = Math.max(emOpen, active ? 0.26 + chuckle * 0.24 : 0.22 * eb + chuckle * 0.1);
  } else if (emotion === "angry") {
    emInnerBrowY = -0.68 * eb;
    emInnerBrowPinch = 0.76 * eb;
    emOuterBrowY = -0.22 * eb;
    emSquint = 0.42 * eb;
    emSmile = -0.38 * eb;
    emCheek = 0.1 * eb;
    emNoseFlare = 0.56 * eb;
    emHeadTiltPitch = -0.14 * eb;
    emMouthSkewX = 0;
    smirkBias = 0;
  } else if (emotion === "surprised") {
    emInnerBrowY = 0.64 * eb;
    emOuterBrowY = 0.66 * eb;
    emEyeWide = 0.54 * eb;
    emCheek = 0.1 * eb;
    emRound = active ? 0.32 : 0.18 * eb;
    emNoseFlare = 0.18 * eb;
    emOpen = Math.max(emOpen, active ? 0.2 : 0.16 * eb);
    emHeadTiltPitch = 0.1 * eb;
  } else if (emotion === "fear") {
    emInnerBrowY = 0.52 * eb;
    emInnerBrowPinch = 0.56 * eb;
    emOuterBrowY = 0.18 * eb;
    emEyeWide = 0.52 * eb;
    emSmile = -0.3 * eb;
    emNoseFlare = 0.28 * eb;
    emOpen = Math.max(emOpen, active ? 0.18 : 0.14 * eb);
    emHeadTiltPitch = 0.08 * eb;
  } else if (emotion === "sad") {
    emInnerBrowY = 0.46 * eb;
    emInnerBrowPinch = 0.28 * eb;
    emOuterBrowY = -0.34 * eb;
    emSmile = -0.38 * eb;
    emSquint = 0.24 * eb;
    emHeadTiltPitch = -0.14 * eb;
    emMouthSkewX = 0;
    smirkBias = 0;
  }

  // Subconscious empathetic response from user's camera expression & mood carryover
  const empathyWeight = emotion === "neutral" || emotion === "happy" ? 1.0 : 0.45;
  emSmile += observedSmile + (empathy.smile - empathy.frown * 0.95) * empathyWeight;
  emCheek += observedSmile * 0.85 + empathy.smile * 0.65 * empathyWeight;
  emSquint += observedSmile * 0.42 + empathy.smile * 0.28 * empathyWeight;
  emInnerBrowPinch += empathy.browFurrow * 0.75 * empathyWeight;
  emInnerBrowY += (empathy.browRaise * 0.65 - empathy.browFurrow * 0.45) * empathyWeight;
  emOuterBrowY += (empathy.browRaise * 0.55 - empathy.browFurrow * 0.22) * empathyWeight;
  emEyeWide += empathy.eyeWide * 0.55 * empathyWeight;
  emNoseFlare +=
    (empathy.noseScrunch * 0.72 + (activePose === "thoughtfulFurrow" ? 0.14 : 0)) * empathyWeight;
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

  // 4) Phoneme-differentiated speech prosody & coarticulation (P/B/M seal vs A open vs E/I spread vs O/U pucker)
  const visPress = active ? Math.min(1, visemes.press ?? 0) : 0;
  const speechOpen = active ? Math.min(1, visemes.open * 0.98 * (1 - visPress * 0.65)) : 0;
  const openDelta = Math.max(0, speechOpen - st.prevVisemeOpen);
  st.prevVisemeOpen = speechOpen;
  st.browBeat = Math.max(st.browBeat * Math.exp(-dt * 4.5), openDelta * 2.6);

  const gatedOpen = Math.max(speechOpen, visPress > 0.45 ? 0 : emOpen);
  const motion = mouthMotion(
    gatedOpen < 0.025 ? 0 : Math.min(1, gatedOpen * mouthAmp),
    opts.hasVisibleTeeth,
    opts.restMouthRatio,
  );
  const targetOpen = Math.min(1, motion.opening * (isAnime ? 1.16 : 1.0));
  const targetWide = Math.min(1, (active ? Math.max(visemes.wide, emWide) : emWide) * wideAmp);
  const targetRound = Math.min(
    1,
    active ? Math.max(visemes.round, visemes.pucker ?? 0, emRound) : emRound,
  );
  const targetPress = active
    ? Math.max(visPress, emotion === "angry" ? 0.25 : 0)
    : emotion === "angry"
      ? 0.22
      : 0;

  const openRate = targetOpen > st.open ? 17.0 : 12.5;
  const mouthAlpha = 1 - Math.exp(-dt * openRate);
  st.open += (targetOpen - st.open) * mouthAlpha;
  st.wide += (targetWide - st.wide) * (1 - Math.exp(-dt * 12));
  st.round += (targetRound - st.round) * (1 - Math.exp(-dt * 12));
  st.lipPress += (targetPress - st.lipPress) * (1 - Math.exp(-dt * 16));

  const speechBrowBoost = active ? Math.min(0.22, st.open * 0.12 + st.browBeat * 0.18) : 0;
  const exprAlpha = 1 - Math.exp(-dt * 7.8);
  st.innerBrowY +=
    ((emotion === "angry" ? emInnerBrowY : emInnerBrowY + speechBrowBoost) * browAmp -
      st.innerBrowY) *
    exprAlpha;
  st.innerBrowPinch += (emInnerBrowPinch * browAmp - st.innerBrowPinch) * exprAlpha;
  st.outerBrowY += ((emOuterBrowY + speechBrowBoost) * browAmp - st.outerBrowY) * exprAlpha;
  st.eyeSquint += (emSquint - st.eyeSquint) * exprAlpha;
  st.eyeWide += (Math.min(1, emEyeWide + (isAnime && active ? 0.08 : 0)) - st.eyeWide) * exprAlpha;
  st.cheekLift +=
    (emCheek + (active ? st.open * 0.08 + st.wide * 0.12 : 0) - st.cheekLift) *
    (1 - Math.exp(-dt * 6.8));
  const smileAlpha = 1 - Math.exp(-dt * 7.2);
  const sAmp = isAnime ? 1.08 : 1.0;
  st.smileR += (emSmileR * sAmp - st.smileR) * smileAlpha;
  st.smileL += (emSmileL * sAmp - st.smileL) * smileAlpha;
  st.smile = (st.smileR + st.smileL) * 0.5;
  st.mouthSkewX += (emMouthSkewX - st.mouthSkewX) * smileAlpha;
  st.noseFlare +=
    (Math.max(emNoseFlare, active ? visemes.level * 0.16 : 0) - st.noseFlare) *
    (1 - Math.exp(-dt * 9));

  // 5) Subtle, natural 3D head, neck & shoulder motion preserving human oval contours
  if (listening && now >= st.nextNodAt) {
    st.nodImpulse = 0.65 + Math.random() * 0.35;
    st.nextNodAt = now + 1800 + Math.random() * 2000;
  } else if (!listening) {
    st.nodImpulse *= Math.exp(-dt * 2.5);
  }
  const nodWave = listening ? Math.sin(t * 4.4) * st.nodImpulse * 0.42 : 0;
  st.nodImpulse *= Math.exp(-dt * 2.2);

  const breath = Math.sin(t * 1.45) * 0.5 + 0.5;
  const gazeHeadLeadX = winkPreparing ? 0 : listening ? st.gazeX * 0.12 : st.gazeX * 0.24;
  const gazeHeadLeadY = winkPreparing ? 0 : listening ? st.gazeY * 0.1 : st.gazeY * 0.16;
  const speechNod = active ? Math.sin(t * 3.1) * st.browBeat * 0.3 : 0;
  const chuckleBounce = emotion === "laugh" ? Math.sin(t * 11.5) * 0.12 : 0;

  const mirrorYaw = directMirror ? userFace.headYaw : 0;
  const mirrorPitch = directMirror ? userFace.headPitch : 0;

  const sideTurnBias = isSideThinking
    ? st.poseDir * 0.34
    : activePose === "softPonder"
      ? st.poseDir * 0.22
      : activePose === "curiousTilt"
        ? st.poseDir * 0.16
        : 0;
  const baseYawWave = listening
    ? Math.sin(t * 0.4) * 0.08 + Math.cos(t * 0.2) * 0.04
    : Math.sin(t * 0.48) * (active ? 0.22 : 0.17) + Math.cos(t * 0.22) * (active ? 0.11 : 0.08);

  const targetYaw = directMirror
    ? mirrorYaw * mScale
    : ((winkPreparing ? 0.15 : 1) * baseYawWave +
        sideTurnBias +
        (active ? Math.sin(t * 1.45) * visemes.level * 0.1 : 0) +
        gazeHeadLeadX * 0.72 +
        empathy.headYaw * 0.6) *
      mScale;

  const targetPitch = directMirror
    ? mirrorPitch * mScale
    : (Math.sin(t * 0.74) * (listening ? 0.11 : 0.15) +
        Math.cos(t * 0.34) * 0.08 +
        (active ? -st.open * 0.05 - speechNod * 0.52 : nodWave * 0.88 + 0.02) +
        chuckleBounce * 0.6 +
        emHeadTiltPitch +
        gazeHeadLeadY * 0.68 +
        empathy.headPitch * 0.6) *
      mScale;

  const poseRollBias =
    activePose === "curiousTilt" ? st.poseDir * 0.048 : isSideThinking ? st.poseDir * 0.024 : 0;
  const targetRoll = directMirror
    ? (userFace?.headRoll ?? 0) * mScale
    : (Math.sin(t * 0.62) * 0.026 +
        Math.cos(t * 0.28) * 0.014 +
        poseRollBias +
        empathy.headRoll * 0.55 +
        (listening ? 0.014 : 0) +
        (winkStrength > 0.1 ? st.winkSide * 0.018 : 0) +
        (emotion === "surprised" ? 0.014 * eb : 0)) *
      mScale;

  const headAlpha = 1 - Math.exp(-dt * 3.8);
  st.headYaw += (Math.max(-0.56, Math.min(0.56, targetYaw)) - st.headYaw) * headAlpha;
  st.headPitch += (Math.max(-0.46, Math.min(0.46, targetPitch)) - st.headPitch) * headAlpha;
  st.headRoll += (targetRoll - st.headRoll) * headAlpha;

  // Ultra-subtle breathing & attentive bust lean (avoids any noticeable camera zooming)
  const targetLean = directMirror
    ? Math.max(-0.35, Math.min(0.45, -mirrorPitch * 0.45)) * mScale
    : (Math.sin(t * 0.42) * 0.18 +
        Math.cos(t * 0.2) * 0.08 +
        (listening ? 0.24 : 0) +
        (activePose === "attentiveWarm" || activePose === "curiousTilt" ? 0.18 : 0) +
        (activePose === "softPonder" || isSideThinking ? -0.14 : 0) +
        (emotion === "fear" ? -0.28 * eb : emotion === "angry" ? 0.26 * eb : 0) +
        -st.headPitch * 0.14) *
      mScale;
  st.forwardLean += (Math.max(-0.45, Math.min(0.52, targetLean)) - st.forwardLean) * headAlpha;

  // Rare subtle tongue articulation / phoneme tongue tip lift
  if (
    !directMirror &&
    !opts.reducedMotion &&
    !listening &&
    emotion !== "angry" &&
    emotion !== "sad" &&
    emotion !== "fear" &&
    now >= st.nextTongueAt &&
    now >= st.tongueUntil
  ) {
    st.tongueUntil = now + 460;
    st.nextTongueAt = now + 26000 + Math.random() * 28000;
  }
  let targetTongue = 0;
  if (!directMirror && now < st.tongueUntil) {
    const tu = 1 - (st.tongueUntil - now) / 460;
    targetTongue =
      Math.sin(Math.min(1, Math.max(0, tu)) * Math.PI) * (emotion === "laugh" ? 0.65 : 0.42);
  } else if (
    active &&
    ((visemes.tongue ?? 0) > 0.1 || (visemes.level > 0.18 && visemes.round > 0.18))
  ) {
    targetTongue = Math.min(0.28, Math.max((visemes.tongue ?? 0) * 0.42, visemes.round * 0.22));
  }
  st.tongueOut += (targetTongue - st.tongueOut) * (1 - Math.exp(-dt * 12));

  const targetTorsoSwayX = directMirror
    ? mirrorYaw * 0.28 * mScale
    : (Math.sin(t * 0.4) * 0.3 + st.headYaw * 0.22) * mScale;
  const targetShoulderTilt = directMirror
    ? (userFace?.headRoll ?? 0) * 0.48 * mScale
    : (Math.sin(t * 0.54) * 0.28 + st.headRoll * 1.65) * mScale;
  const torsoAlpha = 1 - Math.exp(-dt * 2.8);
  st.torsoSwayX += (targetTorsoSwayX - st.torsoSwayX) * torsoAlpha;
  st.shoulderTilt += (targetShoulderTilt - st.shoulderTilt) * torsoAlpha;

  // Subtle organic hair follow-through inertia (gentle lag responding to head rotation and breathing)
  const targetHairX =
    (-st.headYaw * 0.003 - st.headRoll * 0.002 + Math.sin(t * 1.2) * 0.0008) * mScale;
  const targetHairY = ((breath - 0.5) * 0.0012 - st.headPitch * 0.0018) * mScale;
  const hairAlpha = 1 - Math.exp(-dt * 3.5);
  st.hairShiftX += (targetHairX - st.hairShiftX) * hairAlpha;
  st.hairShiftY += (targetHairY - st.hairShiftY) * hairAlpha;

  // Eyebrows normally rise and lower symmetrically together; rare brief micro-accent every ~6.5-13s
  if (now >= st.nextBrowAccentAt) {
    st.browAccentUntil = now + 1150;
    st.browAccentSide = Math.random() < 0.5 ? 1 : -1;
    st.nextBrowAccentAt = now + 6500 + Math.random() * 6500;
  }
  let browAsym = 0;
  if (now < st.browAccentUntil || activePose === "curiousTilt") {
    const u =
      now < st.browAccentUntil
        ? Math.max(0, Math.min(1, 1 - (st.browAccentUntil - now) / 1150))
        : 0.5;
    browAsym = Math.sin(u * Math.PI) * 0.16 * st.browAccentSide * mScale;
  }

  return {
    blink,
    blinkR,
    blinkL,
    open: Math.max(st.open, st.tongueOut * 0.28),
    wide: st.wide,
    round: st.round,
    lipPress: st.lipPress,
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
    hairShiftX: st.hairShiftX,
    hairShiftY: st.hairShiftY,
    forwardLean: st.forwardLean,
    tongueOut: st.tongueOut,
  };
}
