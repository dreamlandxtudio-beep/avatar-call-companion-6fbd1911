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
  lidDroop: number;
  open: number;
  wide: number;
  round: number;
  lipPress: number;
  labiodental: number;
  smile: number;
  smileR: number;
  smileL: number;
  mouthSkewX: number;
  upperLipLift: number;
  lowerLipDrop: number;
  lipBite: number;
  cheekLift: number;
  innerBrowY: number;
  innerBrowPinch: number;
  outerBrowY: number;
  browAsym: number;
  eyeSquint: number;
  eyeWide: number;
  eyeRoll: number;
  noseFlare: number;
  gazeX: number;
  gazeY: number;
  headYaw: number;
  headPitch: number;
  headRoll: number;
  neckYaw: number;
  neckPitch: number;
  neckRoll: number;
  torsoYaw: number;
  headTx: number;
  headTy: number;
  torsoBreath: number;
  torsoSwayX: number;
  shoulderTilt: number;
  shoulderShrug: number;
  handShiftX: number;
  handShiftY: number;
  hairShiftX: number;
  hairShiftY: number;
  forwardLean: number;
  tongueOut: number;
  tearAmount: number;
  eyeWater: number;
  eyeRedness: number;
  pupilDilate: number;
  sweatAmount: number;
  blushAmount: number;
  swallowAmount: number;
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
  lidDroop: number;
  open: number;
  wide: number;
  round: number;
  lipPress: number;
  labiodental: number;
  smile: number;
  smileR: number;
  smileL: number;
  mouthSkewX: number;
  upperLipLift: number;
  lowerLipDrop: number;
  lipBite: number;
  cheekLift: number;
  innerBrowY: number;
  innerBrowPinch: number;
  outerBrowY: number;
  eyeSquint: number;
  eyeWide: number;
  eyeRoll: number;
  noseFlare: number;
  headYaw: number;
  headPitch: number;
  headRoll: number;
  neckYaw: number;
  neckPitch: number;
  neckRoll: number;
  torsoYaw: number;
  torsoSwayX: number;
  shoulderTilt: number;
  shoulderShrug: number;
  hairShiftX: number;
  hairShiftY: number;
  forwardLean: number;
  tongueOut: number;
  tearAmount: number;
  eyeWater: number;
  eyeRedness: number;
  pupilDilate: number;
  sweatAmount: number;
  blushAmount: number;
  nextTongueAt: number;
  tongueUntil: number;
  swallowStartAt: number;
  swallowUntil: number;
  nextSwallowEligibleAt: number;
  swallowAmount: number;
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
    lidDroop: 0,
    open: 0,
    wide: 0,
    round: 0,
    lipPress: 0,
    labiodental: 0,
    smile: 0,
    smileR: 0,
    smileL: 0,
    mouthSkewX: 0,
    upperLipLift: 0,
    lowerLipDrop: 0,
    lipBite: 0,
    cheekLift: 0,
    innerBrowY: 0,
    innerBrowPinch: 0,
    outerBrowY: 0,
    eyeSquint: 0,
    eyeWide: 0,
    eyeRoll: 0,
    noseFlare: 0,
    headYaw: 0,
    headPitch: 0,
    headRoll: 0,
    neckYaw: 0,
    neckPitch: 0,
    neckRoll: 0,
    torsoYaw: 0,
    torsoSwayX: 0,
    shoulderTilt: 0,
    shoulderShrug: 0,
    hairShiftX: 0,
    hairShiftY: 0,
    forwardLean: 0,
    tongueOut: 0,
    tearAmount: 0,
    eyeWater: 0,
    eyeRedness: 0,
    pupilDilate: 0,
    sweatAmount: 0,
    blushAmount: 0,
    nextTongueAt: now + 28000 + Math.random() * 22000,
    tongueUntil: 0,
    swallowStartAt: 0,
    swallowUntil: 0,
    nextSwallowEligibleAt: now + 9000,
    swallowAmount: 0,
    prevVisemeOpen: 0,
    browBeat: 0,
  };
}

const SUPPRESS_POSITIVE_EMOTIONS = new Set<AvatarEmotion>([
  "sad",
  "crying",
  "heartbroken",
  "melancholic",
  "angry",
  "furious",
  "disgusted",
  "fear",
  "nervous",
  "sweating",
  "sick",
  "pain",
  "eyeroll",
]);

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
  const mouthAmp = 0.88;
  const wideAmp = 0.94;
  const browAmp = 1.0;
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

  const isSolemnOrTense = SUPPRESS_POSITIVE_EMOTIONS.has(emotion);
  const isAlteredSlow = emotion === "drunk" || emotion === "high" || emotion === "sleepy";

  // State transitions: thoughtful brief pause/glance when speaking starts; immediate direct eye contact on listening
  if (active && !st.wasActive) {
    st.thinkUntil = now + 720;
    st.thinkDir = Math.random() < 0.5 ? -1 : 1;
    st.preSpeechUntil = now + 420;
    st.poseMode = "thinkSide";
    st.poseDir = st.thinkDir;
    st.poseUntil = now + 1100;
    st.nextPoseAt = now + 2600 + Math.random() * 1800;
    st.swallowStartAt = 0;
    st.swallowUntil = 0;
  } else if (!active && st.wasActive) {
    if (!listening) {
      st.thinkUntil = now + 950;
      st.poseMode = "thinkSide";
      st.poseDir = Math.random() < 0.5 ? -1 : 1;
      st.poseUntil = now + 2200;
    }
    if (!directMirror && now >= st.nextSwallowEligibleAt && Math.random() < 0.42) {
      st.swallowStartAt = now + 850 + Math.random() * 1250;
      st.swallowUntil = st.swallowStartAt + 720;
      st.nextSwallowEligibleAt = now + 13000 + Math.random() * 9000;
    }
  }
  if (listening && !st.wasListening) {
    st.listeningGlanceActive = false;
    st.targetGazeX = 0;
    st.targetGazeY = 0;
    st.nextSaccadeAt = now + 2400 + Math.random() * 2000;
  }
  st.wasActive = active;
  st.wasListening = listening;

  // Cycle subtle pose accents naturally during idle & speech
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

  // 1) Distinct natural blink cadence (slower heavy blinks when sleepy/drunk/high; flutter blinks when crying/nervous)
  if (now >= st.nextBlinkAt) {
    st.blinkProgress = 0;
    const flutterChance = emotion === "nervous" || emotion === "crying" ? 0.34 : 0.22;
    if (!st.doubleBlinkQueued && Math.random() < flutterChance) {
      st.doubleBlinkQueued = true;
      st.nextBlinkAt = now + (emotion === "nervous" ? 240 : 320);
    } else {
      st.doubleBlinkQueued = false;
      const interval =
        emotion === "nervous"
          ? 1200 + Math.random() * 1100
          : isAlteredSlow
            ? 1900 + Math.random() * 1700
            : active
              ? 2200 + Math.random() * 2000
              : 2100 + Math.random() * 2200;
      st.nextBlinkAt = now + interval;
    }
  }
  let baseBlink = 0;
  if (st.blinkProgress < 1) {
    const blinkDuration = emotion === "sleepy" ? 0.68 : isAlteredSlow ? 0.56 : 0.4;
    st.blinkProgress = Math.min(1, st.blinkProgress + dt / blinkDuration);
    baseBlink = blinkClosure(st.blinkProgress);
  }

  // Rare wink (~every 62-110s, suppressed while listening or solemn/tense emotions)
  if (
    !directMirror &&
    !opts.reducedMotion &&
    !listening &&
    now >= st.nextWinkAt &&
    now >= st.winkUntil &&
    !isSolemnOrTense &&
    !isAlteredSlow
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

  // 2) Eye contact & ocular saccades (adapted for nervous darting, sleepy/drunk drift, sad downward gaze)
  if (winkPreparing) {
    st.targetGazeX = 0;
    st.targetGazeY = 0;
    st.listeningGlanceActive = false;
  } else if (now >= st.nextSaccadeAt) {
    if (listening) {
      if (st.listeningGlanceActive) {
        st.listeningGlanceActive = false;
        st.targetGazeX = (Math.random() - 0.5) * 0.03;
        st.targetGazeY = (Math.random() - 0.5) * 0.02;
        st.nextSaccadeAt = now + 3800 + Math.random() * 3400;
      } else if (Math.random() < 0.16) {
        st.listeningGlanceActive = true;
        st.targetGazeX = (Math.random() < 0.5 ? -1 : 1) * (0.28 + Math.random() * 0.14);
        st.targetGazeY = (Math.random() - 0.5) * 0.14;
        st.nextSaccadeAt = now + 340 + Math.random() * 120;
      } else {
        st.targetGazeX = (Math.random() - 0.5) * 0.04;
        st.targetGazeY = (Math.random() - 0.5) * 0.03;
        st.nextSaccadeAt = now + 2200 + Math.random() * 1800;
      }
    } else if (emotion === "eyeroll") {
      st.targetGazeX = Math.sin(t * 2.2) * 0.22;
      st.targetGazeY = 0.88;
      st.nextSaccadeAt = now + 520 + Math.random() * 480;
    } else if (emotion === "sarcastic") {
      st.targetGazeX = (Math.random() < 0.5 ? -1 : 1) * 0.18;
      st.targetGazeY = 0.36;
      st.nextSaccadeAt = now + 780 + Math.random() * 620;
    } else if (emotion === "intense") {
      st.targetGazeX = (Math.random() - 0.5) * 0.08;
      st.targetGazeY = 0.28 + Math.random() * 0.08;
      st.nextSaccadeAt = now + 1400 + Math.random() * 1100;
    } else if (emotion === "nervous" || emotion === "sweating") {
      st.targetGazeX = (Math.random() - 0.5) * 0.85;
      st.targetGazeY = (Math.random() - 0.5) * 0.28 - 0.06;
      st.nextSaccadeAt = now + 280 + Math.random() * 380;
    } else if (emotion === "blushing") {
      st.targetGazeX = (Math.random() < 0.5 ? -1 : 1) * (0.35 + Math.random() * 0.25);
      st.targetGazeY = -0.18 - Math.random() * 0.12;
      st.nextSaccadeAt = now + 900 + Math.random() * 700;
    } else if (
      emotion === "sad" ||
      emotion === "crying" ||
      emotion === "heartbroken" ||
      emotion === "melancholic"
    ) {
      st.targetGazeX = (Math.random() - 0.5) * 0.36;
      st.targetGazeY = -0.22 - Math.random() * 0.16;
      st.nextSaccadeAt = now + 1100 + Math.random() * 900;
    } else if (emotion === "drunk" || emotion === "high") {
      st.targetGazeX = Math.sin(t * 0.9) * 0.38 + (Math.random() - 0.5) * 0.22;
      st.targetGazeY = -0.08 + (Math.random() - 0.5) * 0.18;
      st.nextSaccadeAt = now + 950 + Math.random() * 850;
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
      st.targetGazeY = (Math.random() - 0.5) * 0.16;
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
    -0.82,
    Math.min(
      0.95,
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
  } else if (emotion === "drunk") {
    // Slightly lazy asymmetric eyelid lag when intoxicated
    blinkR = Math.max(blinkR, 0.08 + 0.06 * Math.sin(t * 1.4));
    blinkL = Math.max(blinkL, 0.04 + 0.06 * Math.cos(t * 1.1));
  }
  const blink = (blinkR + blinkL) * 0.5;

  const gazeAlpha = 1 - Math.exp(-dt * (winkPreparing || listening ? 30 : isAlteredSlow ? 10 : 24));
  st.gazeX += (targetGx - st.gazeX) * gazeAlpha;
  st.gazeY += (targetGy - st.gazeY) * gazeAlpha;

  // 3) Responsive expression transitions & vivid emotional range
  const targetBlend = active ? 1.0 : emotion !== "neutral" ? 0.94 : listening ? 0.64 : 0.52;
  const blendRate = active || emotion !== "neutral" ? 12.2 : 3.2;
  st.emotionBlend += (targetBlend - st.emotionBlend) * (1 - Math.exp(-dt * blendRate));
  const eb = st.emotionBlend;

  const ambientWarmth = isSolemnOrTense ? 0 : Math.max(0, Math.sin(t * 0.45)) * 0.16;
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
    emotion !== "laugh" &&
    emotion !== "euphoric"
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
  if (!directMirror && (now < st.smirkUntil || activePose === "gentleSmirk") && !isSolemnOrTense) {
    const su = now < st.smirkUntil ? 1 - (st.smirkUntil - now) / 1350 : 0.5;
    smirkAmount = Math.sin(Math.min(1, Math.max(0, su)) * Math.PI) * 0.2 * mScale;
  }

  const poseWarm = activePose === "attentiveWarm" && !isSolemnOrTense ? 0.2 * mScale : 0;
  const poseBrowLift =
    activePose === "browInterest"
      ? 0.22 * mScale
      : activePose === "curiousTilt"
        ? 0.16 * mScale
        : 0;
  const poseSquint =
    isSideThinking || activePose === "thoughtfulFurrow"
      ? 0.12 * mScale
      : activePose === "attentiveWarm"
        ? 0.14 * mScale
        : 0;

  const speechWarmth =
    active && !isSolemnOrTense
      ? (0.12 + Math.max(0, Math.sin(t * 1.35)) * 0.16) * (0.45 + visemes.level * 0.55)
      : 0;

  const preSpeechOpen =
    !directMirror &&
    emotion !== "angry" &&
    emotion !== "furious" &&
    active &&
    (now < st.preSpeechUntil || visemes.level < 0.08)
      ? 0.045 * mScale
      : 0;

  let emOpen = preSpeechOpen;
  let emWide = speechWarmth * 0.24;
  let emRound = activePose === "softPonder" ? 0.08 * mScale : 0;
  let emPress = 0;
  let emSmile = ambientWarmth * 0.65 + speechWarmth * 0.75 + poseWarm;
  let emCheek = ambientWarmth * 0.55 + speechWarmth * 0.65 + poseWarm * 0.75;
  let emInnerBrowY = ambientBrow * 0.85 + poseBrowLift - gestureFurrow * 0.38;
  let emInnerBrowPinch = gestureFurrow;
  let emOuterBrowY = ambientBrow * 0.92 + poseBrowLift * 0.85 - gestureFurrow * 0.14;
  let emExplicitBrowAsym = 0;
  let emSquint = speechWarmth * 0.32 + poseSquint + winkStrength * 0.22;
  let emEyeWide = activePose === "browInterest" ? 0.12 * mScale : 0;
  let emLidDroop = 0;
  let emNoseFlare = 0;
  let emHeadTiltPitch = isSideThinking
    ? 0.05 * mScale
    : activePose === "softPonder"
      ? 0.08 * mScale
      : 0;
  let emMouthSkewX = smirkAmount * st.smirkSide * 0.55;
  let emUpperLipLift = 0;
  let emLowerLipDrop = 0;
  let emLipBite = 0;
  let emEyeRoll = 0;
  let smirkBias = smirkAmount * st.smirkSide;
  let emShoulderShrug = 0;
  let emTearAmount = 0;
  let emEyeWater = 0;
  let emEyeRedness = 0;
  let emPupilDilate = 0;
  let emSweatAmount = 0;
  let emBlushAmount = 0;

  // Comprehensive expression & physical-state mapping
  switch (emotion) {
    case "happy": {
      emSmile =
        (opts.hasVisibleTeeth ? 0.68 : 0.78) * eb + ambientWarmth * 0.2 + speechWarmth * 0.25;
      emWide = 0.28 * eb;
      emUpperLipLift = 0.24 * eb;
      emCheek = 0.72 * eb;
      emSquint = 0.42 * eb;
      emInnerBrowY = 0.22 * eb + ambientBrow * 0.25;
      emOuterBrowY = 0.24 * eb + ambientBrow * 0.25;
      emNoseFlare = 0.14 * eb;
      emBlushAmount = 0.26 * eb;
      break;
    }
    case "euphoric": {
      emSmile = (opts.hasVisibleTeeth ? 0.82 : 0.92) * eb;
      emWide = 0.46 * eb;
      emUpperLipLift = 0.38 * eb;
      emLowerLipDrop = 0.22 * eb;
      emCheek = 0.86 * eb;
      emSquint = 0.38 * eb;
      emEyeWide = 0.28 * eb;
      emInnerBrowY = 0.52 * eb;
      emOuterBrowY = 0.56 * eb;
      emOpen = Math.max(emOpen, active ? 0.24 : 0.28 * eb);
      emNoseFlare = 0.26 * eb;
      emHeadTiltPitch = 0.14 * eb;
      emPupilDilate = 0.3 * eb;
      emBlushAmount = 0.4 * eb;
      emShoulderShrug = 0.22 * eb;
      break;
    }
    case "laugh": {
      // Smooth breathy laughter cadence (organic burst curve instead of robotic jitter)
      const breathWave = 0.5 + 0.5 * Math.sin(t * 6.4 + Math.sin(t * 1.9) * 0.65);
      emSmile = (opts.hasVisibleTeeth ? 0.84 : 0.92) * eb;
      emWide = 0.42 * eb;
      emUpperLipLift = 0.38 * eb;
      emLowerLipDrop = 0.2 * eb;
      emCheek = 0.86 * eb;
      emSquint = 0.64 * eb;
      emInnerBrowY = 0.34 * eb;
      emOuterBrowY = 0.38 * eb;
      emNoseFlare = 0.34 * eb;
      emEyeWater = 0.28 * eb;
      emBlushAmount = 0.36 * eb;
      emShoulderShrug = (0.14 + breathWave * 0.22) * eb;
      emOpen = Math.max(
        emOpen,
        active ? 0.26 + visemes.open * 0.24 : (0.32 + breathWave * 0.2) * eb,
      );
      break;
    }
    case "flirty": {
      emSmile = 0.36 * eb;
      smirkBias = 0.42 * eb;
      emMouthSkewX = 0.22 * eb;
      emCheek = 0.44 * eb;
      emSquint = 0.28 * eb;
      emLidDroop = 0.24 * eb;
      emInnerBrowY = 0.18 * eb;
      emOuterBrowY = 0.24 * eb;
      emExplicitBrowAsym = 0.28 * eb;
      emPupilDilate = 0.24 * eb;
      emBlushAmount = 0.52 * eb;
      emHeadTiltPitch = -0.06 * eb;
      break;
    }
    case "smirk": {
      emSmile = 0.16 * eb;
      smirkBias = 0.78 * eb;
      emMouthSkewX = 0.48 * eb;
      emUpperLipLift = 0.14 * eb;
      emCheek = 0.36 * eb;
      emSquint = 0.24 * eb;
      emInnerBrowY = 0.12 * eb;
      emOuterBrowY = 0.18 * eb;
      emExplicitBrowAsym = 0.34 * eb;
      emHeadTiltPitch = -0.05 * eb;
      break;
    }
    case "compassionate": {
      emSmile = 0.32 * eb;
      emCheek = 0.38 * eb;
      emSquint = 0.22 * eb;
      emInnerBrowY = 0.44 * eb;
      emInnerBrowPinch = 0.24 * eb;
      emOuterBrowY = -0.08 * eb;
      emEyeWater = 0.34 * eb;
      emPupilDilate = 0.26 * eb;
      emBlushAmount = 0.22 * eb;
      emHeadTiltPitch = -0.1 * eb;
      break;
    }
    case "intense": {
      emSmile = 0.1 * eb;
      smirkBias = 0.18 * eb;
      emPress = 0.18 * eb;
      emInnerBrowY = -0.14 * eb;
      emInnerBrowPinch = 0.32 * eb;
      emOuterBrowY = 0.1 * eb;
      emSquint = 0.24 * eb;
      emLidDroop = 0.18 * eb;
      emPupilDilate = 0.52 * eb;
      emNoseFlare = 0.18 * eb;
      emHeadTiltPitch = -0.32 * eb;
      break;
    }
    case "bitingLip": {
      emLipBite = 0.92 * eb;
      emUpperLipLift = 0.18 * eb;
      emPress = 0.24 * eb;
      emSquint = 0.22 * eb;
      emLidDroop = 0.22 * eb;
      emInnerBrowY = 0.16 * eb;
      emExplicitBrowAsym = 0.18 * eb;
      emPupilDilate = 0.36 * eb;
      emBlushAmount = 0.56 * eb;
      emHeadTiltPitch = -0.1 * eb;
      break;
    }
    case "sarcastic": {
      emSmile = 0.08 * eb;
      smirkBias = 0.58 * eb;
      emMouthSkewX = 0.42 * eb;
      emInnerBrowY = 0.1 * eb;
      emOuterBrowY = 0.32 * eb;
      emExplicitBrowAsym = 0.84 * eb;
      emSquint = 0.28 * eb;
      emLidDroop = 0.24 * eb;
      emEyeRoll = 0.3 * eb;
      break;
    }
    case "eyeroll": {
      emEyeRoll = 1.0 * eb;
      emEyeWide = 0.28 * eb;
      emLidDroop = 0.08 * eb;
      emInnerBrowY = 0.22 * eb;
      emInnerBrowPinch = 0.2 * eb;
      emExplicitBrowAsym = 0.36 * eb;
      emSmile = -0.18 * eb;
      emPress = 0.34 * eb;
      emMouthSkewX = -0.2 * eb;
      emHeadTiltPitch = 0.14 * eb;
      break;
    }
    case "proud": {
      emSmile = 0.4 * eb;
      emWide = 0.16 * eb;
      emCheek = 0.38 * eb;
      emSquint = 0.18 * eb;
      emInnerBrowY = 0.14 * eb;
      emOuterBrowY = 0.22 * eb;
      emHeadTiltPitch = 0.16 * eb;
      emShoulderShrug = -0.15 * eb;
      break;
    }
    case "blushing": {
      emSmile = 0.36 * eb;
      emPress = 0.18 * eb;
      emCheek = 0.4 * eb;
      emSquint = 0.24 * eb;
      emInnerBrowY = 0.22 * eb;
      emOuterBrowY = 0.14 * eb;
      emBlushAmount = 0.96 * eb;
      emHeadTiltPitch = -0.1 * eb;
      break;
    }
    case "sad": {
      emInnerBrowY = 0.54 * eb;
      emInnerBrowPinch = 0.38 * eb;
      emOuterBrowY = -0.34 * eb;
      emSmile = -0.46 * eb;
      emSquint = 0.24 * eb;
      emLidDroop = 0.18 * eb;
      emEyeWater = 0.44 * eb;
      emEyeRedness = 0.18 * eb;
      emHeadTiltPitch = -0.18 * eb;
      emShoulderShrug = -0.35 * eb;
      emMouthSkewX = 0;
      smirkBias = 0;
      break;
    }
    case "crying": {
      const sobQuiver = Math.sin(t * 9.5) * 0.08 * eb;
      emInnerBrowY = 0.64 * eb;
      emInnerBrowPinch = 0.58 * eb;
      emOuterBrowY = -0.38 * eb;
      emSmile = -0.6 * eb;
      emLowerLipDrop = 0.18 * eb;
      emSquint = 0.48 * eb;
      emLidDroop = 0.16 * eb;
      emOpen = Math.max(emOpen, 0.14 * eb + Math.max(0, sobQuiver));
      emNoseFlare = 0.24 * eb;
      emHeadTiltPitch = -0.2 * eb;
      emShoulderShrug = -0.24 * eb + Math.sin(t * 3.8) * 0.12 * eb;
      emTearAmount = 0.88 * eb;
      emEyeWater = 0.92 * eb;
      emEyeRedness = 0.52 * eb;
      emBlushAmount = 0.34 * eb;
      emMouthSkewX = 0;
      smirkBias = 0;
      break;
    }
    case "heartbroken": {
      const sob = Math.sin(t * 5.2) * 0.14 * eb;
      emInnerBrowY = 0.72 * eb;
      emInnerBrowPinch = 0.72 * eb;
      emOuterBrowY = -0.44 * eb;
      emSmile = -0.7 * eb;
      emLowerLipDrop = 0.24 * eb;
      emSquint = 0.58 * eb;
      emOpen = Math.max(emOpen, 0.26 * eb + Math.max(0, sob));
      emNoseFlare = 0.36 * eb;
      emHeadTiltPitch = -0.24 * eb;
      emShoulderShrug = -0.32 * eb + Math.sin(t * 4.4) * 0.16 * eb;
      emTearAmount = 1.0 * eb;
      emEyeWater = 1.0 * eb;
      emEyeRedness = 0.65 * eb;
      emBlushAmount = 0.42 * eb;
      emMouthSkewX = 0;
      smirkBias = 0;
      break;
    }
    case "melancholic": {
      emInnerBrowY = 0.34 * eb;
      emInnerBrowPinch = 0.18 * eb;
      emOuterBrowY = -0.22 * eb;
      emSmile = -0.24 * eb;
      emLidDroop = 0.34 * eb;
      emEyeWater = 0.3 * eb;
      emHeadTiltPitch = -0.14 * eb;
      emShoulderShrug = -0.38 * eb;
      emMouthSkewX = 0;
      smirkBias = 0;
      break;
    }
    case "angry": {
      // Strong readable brow glare + clenched tense jaw/nostril flare without grotesque distortion
      emInnerBrowY = -0.46 * eb;
      emInnerBrowPinch = 1.0 * eb;
      emOuterBrowY = 0.26 * eb;
      emSquint = 0.46 * eb;
      emSmile = -0.52 * eb;
      emWide = 0.12 * eb;
      emPress = 0.38 * eb;
      emUpperLipLift = 0.14 * eb;
      emCheek = 0.12 * eb;
      emNoseFlare = 0.68 * eb;
      emHeadTiltPitch = -0.16 * eb;
      emShoulderShrug = 0.18 * eb;
      emMouthSkewX = 0;
      smirkBias = 0;
      break;
    }
    case "furious": {
      emInnerBrowY = -0.56 * eb;
      emInnerBrowPinch = 1.05 * eb;
      emOuterBrowY = 0.34 * eb;
      emSquint = 0.5 * eb;
      emEyeWide = 0.2 * eb;
      emSmile = -0.66 * eb;
      emWide = 0.34 * eb;
      emUpperLipLift = 0.46 * eb;
      emLowerLipDrop = 0.28 * eb;
      emPress = active ? 0.12 : 0.22 * eb;
      emOpen = Math.max(emOpen, active ? 0.3 : 0.28 * eb);
      emNoseFlare = 0.92 * eb;
      emHeadTiltPitch = -0.2 * eb;
      emShoulderShrug = 0.34 * eb;
      emEyeRedness = 0.32 * eb;
      emSweatAmount = 0.28 * eb;
      emBlushAmount = 0.36 * eb;
      emMouthSkewX = 0;
      smirkBias = 0;
      break;
    }
    case "disgusted": {
      emInnerBrowY = -0.36 * eb;
      emInnerBrowPinch = 0.94 * eb;
      emOuterBrowY = -0.14 * eb;
      emSquint = 0.64 * eb;
      emCheek = 0.38 * eb;
      emSmile = -0.48 * eb;
      emUpperLipLift = 0.78 * eb;
      emLowerLipDrop = 0.16 * eb;
      emMouthSkewX = -0.36 * eb;
      emNoseFlare = 1.0 * eb;
      emHeadTiltPitch = 0.1 * eb;
      emShoulderShrug = 0.2 * eb;
      break;
    }
    case "skeptical": {
      emInnerBrowY = 0.08 * eb;
      emInnerBrowPinch = 0.28 * eb;
      emOuterBrowY = 0.32 * eb;
      emExplicitBrowAsym = 0.8 * eb;
      emSquint = 0.28 * eb;
      emPress = 0.3 * eb;
      emMouthSkewX = 0.38 * eb;
      smirkBias = 0.28 * eb;
      break;
    }
    case "surprised": {
      emInnerBrowY = 0.76 * eb;
      emOuterBrowY = 0.78 * eb;
      emEyeWide = 0.68 * eb;
      emCheek = 0.1 * eb;
      emRound = active ? 0.4 : 0.38 * eb;
      emLowerLipDrop = 0.24 * eb;
      emNoseFlare = 0.22 * eb;
      emOpen = Math.max(emOpen, active ? 0.32 : 0.36 * eb);
      emHeadTiltPitch = 0.12 * eb;
      emShoulderShrug = 0.26 * eb;
      emPupilDilate = 0.28 * eb;
      break;
    }
    case "fear": {
      const tremor = Math.sin(t * 16) * 0.05 * eb;
      emInnerBrowY = 0.66 * eb;
      emInnerBrowPinch = 0.66 * eb;
      emOuterBrowY = 0.22 * eb;
      emEyeWide = 0.72 * eb;
      emSmile = -0.4 * eb;
      emWide = 0.24 * eb;
      emLowerLipDrop = 0.18 * eb;
      emNoseFlare = 0.42 * eb;
      emOpen = Math.max(emOpen, active ? 0.22 : 0.22 * eb + Math.max(0, tremor));
      emHeadTiltPitch = 0.1 * eb;
      emShoulderShrug = 0.62 * eb;
      emPupilDilate = 0.44 * eb;
      emSweatAmount = 0.54 * eb;
      emBlushAmount = -0.62 * eb;
      break;
    }
    case "nervous": {
      emInnerBrowY = 0.44 * eb;
      emInnerBrowPinch = 0.5 * eb;
      emOuterBrowY = 0.08 * eb;
      emEyeWide = 0.28 * eb;
      emPress = 0.56 * eb;
      emLipBite = 0.24 * eb;
      emSmile = -0.16 * eb;
      emMouthSkewX = Math.sin(t * 2.4) * 0.24 * eb;
      emNoseFlare = 0.28 * eb;
      emShoulderShrug = 0.45 * eb;
      emSweatAmount = 0.8 * eb;
      emPupilDilate = 0.24 * eb;
      break;
    }
    case "sweating": {
      emInnerBrowY = 0.28 * eb;
      emInnerBrowPinch = 0.36 * eb;
      emOuterBrowY = 0.12 * eb;
      emEyeWide = 0.18 * eb;
      emOpen = Math.max(emOpen, 0.16 * eb + Math.sin(t * 3.8) * 0.05 * eb);
      emNoseFlare = 0.48 * eb;
      emSweatAmount = 1.0 * eb;
      emBlushAmount = 0.24 * eb;
      emShoulderShrug = 0.22 * eb;
      break;
    }
    case "pain": {
      emInnerBrowY = 0.34 * eb;
      emInnerBrowPinch = 0.96 * eb;
      emOuterBrowY = -0.22 * eb;
      emSquint = 0.86 * eb;
      emLidDroop = 0.24 * eb;
      emSmile = -0.52 * eb;
      emWide = 0.56 * eb;
      emUpperLipLift = 0.48 * eb;
      emLowerLipDrop = 0.38 * eb;
      emOpen = Math.max(emOpen, 0.36 * eb);
      emNoseFlare = 0.68 * eb;
      emShoulderShrug = 0.68 * eb;
      emEyeWater = 0.58 * eb;
      emEyeRedness = 0.38 * eb;
      emSweatAmount = 0.6 * eb;
      break;
    }
    case "drunk": {
      emInnerBrowY = 0.18 * eb;
      emOuterBrowY = -0.14 * eb;
      emExplicitBrowAsym = 0.32 * eb;
      emLidDroop = 0.48 * eb;
      emSquint = 0.22 * eb;
      emSmile = 0.42 * eb;
      emMouthSkewX = 0.3 * eb;
      emOpen = Math.max(emOpen, 0.1 * eb);
      emHeadTiltPitch = -0.1 * eb;
      emEyeWater = 0.36 * eb;
      emEyeRedness = 0.45 * eb;
      emBlushAmount = 0.68 * eb;
      emShoulderShrug = -0.28 * eb;
      break;
    }
    case "high": {
      emInnerBrowY = 0.14 * eb;
      emOuterBrowY = 0.06 * eb;
      emLidDroop = 0.62 * eb;
      emSquint = 0.32 * eb;
      emSmile = 0.44 * eb;
      emCheek = 0.44 * eb;
      emOpen = Math.max(emOpen, 0.08 * eb);
      emEyeWater = 0.4 * eb;
      emEyeRedness = 0.58 * eb;
      emPupilDilate = 0.36 * eb;
      emShoulderShrug = -0.34 * eb;
      break;
    }
    case "sleepy": {
      const yawnCycle = Math.max(0, Math.sin(t * 0.75) - 0.55) / 0.45;
      emInnerBrowY = 0.16 * eb;
      emOuterBrowY = -0.22 * eb;
      emLidDroop = 0.68 * eb;
      emOpen = Math.max(emOpen, active ? 0 : yawnCycle * 0.48 * eb);
      emRound = Math.max(emRound, yawnCycle * 0.4 * eb);
      emHeadTiltPitch = -0.16 * eb;
      emShoulderShrug = -0.42 * eb;
      emEyeWater = 0.25 * eb;
      break;
    }
    case "sick": {
      emInnerBrowY = 0.36 * eb;
      emInnerBrowPinch = 0.42 * eb;
      emOuterBrowY = -0.28 * eb;
      emLidDroop = 0.36 * eb;
      emSmile = -0.36 * eb;
      emNoseFlare = 0.22 * eb;
      emHeadTiltPitch = -0.14 * eb;
      emShoulderShrug = -0.32 * eb;
      emSweatAmount = 0.56 * eb;
      emBlushAmount = -0.72 * eb;
      emEyeWater = 0.34 * eb;
      emEyeRedness = 0.22 * eb;
      break;
    }
    default:
      break;
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

  let emSmileR =
    emSmile + Math.max(0, smirkBias) * 0.72 + (st.winkSide > 0 ? winkStrength * 0.24 : 0);
  let emSmileL =
    emSmile + Math.max(0, -smirkBias) * 0.72 + (st.winkSide < 0 ? winkStrength * 0.24 : 0);

  let mirrorBrowAsym = 0;
  if (directMirror) {
    const mirrorWeight = 1;
    emSmile = userFace.smile * 1.02 - (userFace.mouthFrown ?? 0) * 0.68;
    emSmileR = (userFace.smileRight ?? userFace.smile) * 1.05 - (userFace.mouthFrown ?? 0) * 0.68;
    emSmileL = (userFace.smileLeft ?? userFace.smile) * 1.05 - (userFace.mouthFrown ?? 0) * 0.68;
    emMouthSkewX = userFace.mouthSkewX ?? 0;
    emUpperLipLift = Math.max(emUpperLipLift, userFace.upperLipLift ?? 0);
    emLowerLipDrop = Math.max(emLowerLipDrop, userFace.lowerLipDrop ?? 0);
    emLipBite = Math.max(emLipBite, (userFace.mouthShrug ?? 0) * 0.9);
    emWide = userFace.mouthWide ?? Math.max(0, userFace.smile * 0.35);
    emRound = userFace.mouthRound ?? 0;
    emPress = userFace.mouthPress ?? 0;
    emCheek = userFace.smile * 0.78 * mirrorWeight;
    emSquint = (userFace.eyeSquint ?? 0) * mirrorWeight + userFace.smile * 0.4;
    emOpen = Math.max(emOpen, userFace.mouthOpen * mirrorWeight);
    emInnerBrowY = (userFace.browInnerUp ?? userFace.browRaise) * 0.95 - userFace.browFurrow * 0.55;
    emInnerBrowPinch = userFace.browFurrow;
    emOuterBrowY = userFace.browRaise * 0.9 - userFace.browFurrow * 0.08;
    const bL = userFace.browRaiseLeft ?? userFace.browRaise;
    const bR = userFace.browRaiseRight ?? userFace.browRaise;
    mirrorBrowAsym = Math.max(-0.9, Math.min(0.9, (bL - bR) * 1.15));
    emNoseFlare = userFace.noseSneer ?? 0;
    emEyeWide = userFace.eyeWide ?? 0;
    if (userFace.gazeY > 0.45) {
      emEyeRoll = Math.min(1, (userFace.gazeY - 0.45) * 1.6);
    }
  }

  // 4) Phoneme-differentiated speech prosody & coarticulation (conversational by default, occasional wider emphasis)
  const visPress = active ? Math.min(1, visemes.press ?? 0) : 0;
  const visLabio = active ? Math.min(1, visemes.labiodental ?? 0) : 0;
  const visStress = active ? Math.min(1, visemes.stress ?? 0) : 0;
  const emphaticPeak =
    active && visemes.open > 0.56 && visStress > 0.24
      ? ((visemes.open - 0.56) / 0.44) * Math.min(1, (visStress - 0.2) * 1.45) * 0.28
      : 0;
  const speechOpen = active
    ? Math.min(
        1.06,
        (visemes.open * 0.88 + emphaticPeak) * (1 - visPress * 0.85) * (1 - visLabio * 0.45),
      )
    : 0;
  const openDelta = Math.max(0, speechOpen - st.prevVisemeOpen);
  st.prevVisemeOpen = speechOpen;
  st.browBeat = Math.max(st.browBeat * Math.exp(-dt * 5.2), openDelta * 2.2 + visStress * 0.45);

  // Trigger occasional natural swallows during brief speech breath pauses as well as after speech
  if (
    active &&
    !directMirror &&
    st.swallowStartAt === 0 &&
    visemes.level < 0.14 &&
    st.open < 0.08 &&
    now >= st.nextSwallowEligibleAt &&
    Math.random() < 0.38
  ) {
    st.swallowStartAt = now + 60;
    st.swallowUntil = st.swallowStartAt + 680;
    st.nextSwallowEligibleAt = now + 11500 + Math.random() * 9500;
  }

  let swallowWave = 0;
  let swallowLipContact = 0;
  let swallowChinDip = 0;
  if (!directMirror && st.swallowStartAt > 0 && now >= st.swallowStartAt && now < st.swallowUntil) {
    const swU = (now - st.swallowStartAt) / Math.max(1, st.swallowUntil - st.swallowStartAt);
    const speechFade = active ? Math.max(0.25, 1 - Math.max(0, visemes.level - 0.12) * 2.5) : 1;
    swallowWave = Math.sin(Math.min(1, Math.max(0, swU)) * Math.PI) * speechFade;
    swallowLipContact = active
      ? 0
      : Math.sin(Math.min(1, swU / 0.55) * Math.PI) * (swU < 0.55 ? 0.26 : 0);
    swallowChinDip = -swallowWave * 0.036;
  } else if (now >= st.swallowUntil) {
    st.swallowStartAt = 0;
    st.swallowUntil = 0;
  }
  st.swallowAmount += (swallowWave * mScale - st.swallowAmount) * (1 - Math.exp(-dt * 18));

  const gatedOpen = Math.max(speechOpen, visPress > 0.4 ? 0 : emOpen);
  const motion = mouthMotion(
    gatedOpen < 0.004 ? 0 : Math.min(1, gatedOpen * (directMirror ? 1.02 : mouthAmp)),
    opts.hasVisibleTeeth,
    opts.restMouthRatio,
  );
  const targetOpen = Math.min(1.04, motion.opening * 0.98 + emphaticPeak * 0.24);
  const targetWide = Math.min(1, (active ? Math.max(visemes.wide, emWide) : emWide) * wideAmp);
  const targetRound = Math.min(
    1,
    active ? Math.max(visemes.round, visemes.pucker ?? 0, emRound) : emRound,
  );
  const targetPress = active
    ? Math.max(visPress, emPress, emotion === "angry" ? 0.22 : 0)
    : Math.max(emPress, swallowLipContact, emotion === "angry" || emotion === "furious" ? 0.24 : 0);

  const openRate = targetOpen > st.open ? 22.0 : 17.0;
  const mouthAlpha = 1 - Math.exp(-dt * openRate);
  st.open += (targetOpen - st.open) * mouthAlpha;
  st.wide += (targetWide - st.wide) * (1 - Math.exp(-dt * 16));
  st.round += (targetRound - st.round) * (1 - Math.exp(-dt * 16));
  st.lipPress += (targetPress - st.lipPress) * (1 - Math.exp(-dt * 20));
  st.labiodental += (visLabio - st.labiodental) * (1 - Math.exp(-dt * 18));
  const smilePartBoost = Math.max(0, emSmile - 0.32) * (1 - Math.min(1, targetOpen * 5));
  st.upperLipLift +=
    (Math.min(1, emUpperLipLift + smilePartBoost * 0.36 + (active ? st.wide * 0.14 : 0)) -
      st.upperLipLift) *
    (1 - Math.exp(-dt * 18));
  st.lowerLipDrop +=
    (Math.min(1, emLowerLipDrop + smilePartBoost * 0.32 + (active ? st.open * 0.12 : 0)) -
      st.lowerLipDrop) *
    (1 - Math.exp(-dt * 18));
  st.lipBite += (emLipBite - st.lipBite) * (1 - Math.exp(-dt * 16));

  const speechBrowBoost = active ? Math.min(0.32, st.open * 0.14 + st.browBeat * 0.26) : 0;
  const exprAlpha = 1 - Math.exp(-dt * (directMirror ? 15.0 : 13.5));
  const angryLock = emotion === "angry" || emotion === "furious";
  st.innerBrowY +=
    ((angryLock ? emInnerBrowY : emInnerBrowY + speechBrowBoost) * browAmp - st.innerBrowY) *
    exprAlpha;
  st.innerBrowPinch += (emInnerBrowPinch * browAmp - st.innerBrowPinch) * exprAlpha;
  st.outerBrowY +=
    ((angryLock ? emOuterBrowY : emOuterBrowY + speechBrowBoost) * browAmp - st.outerBrowY) *
    exprAlpha;
  st.eyeSquint += (emSquint - st.eyeSquint) * exprAlpha;
  st.eyeWide += (Math.min(1, emEyeWide) - st.eyeWide) * exprAlpha;
  st.eyeRoll += (emEyeRoll - st.eyeRoll) * exprAlpha;
  st.lidDroop += (emLidDroop - st.lidDroop) * exprAlpha;
  st.cheekLift +=
    (emCheek + (active ? st.open * 0.08 + st.wide * 0.12 : 0) - st.cheekLift) *
    (1 - Math.exp(-dt * 11.5));
  const smileAlpha = 1 - Math.exp(-dt * 12.5);
  st.smileR += (emSmileR - st.smileR) * smileAlpha;
  st.smileL += (emSmileL - st.smileL) * smileAlpha;
  st.smile = (st.smileR + st.smileL) * 0.5;
  st.mouthSkewX += (emMouthSkewX - st.mouthSkewX) * smileAlpha;
  st.noseFlare +=
    (Math.max(emNoseFlare, active ? visemes.level * 0.18 : 0, Math.max(0, st.open - 0.24) * 0.36) -
      st.noseFlare) *
    (1 - Math.exp(-dt * 12));

  // Brisk physical shader tone/sweat/tear transitions
  const physAlpha = 1 - Math.exp(-dt * 8.5);
  st.tearAmount += (emTearAmount * mScale - st.tearAmount) * physAlpha;
  st.eyeWater += (emEyeWater * mScale - st.eyeWater) * physAlpha;
  st.eyeRedness += (emEyeRedness * 0.42 * mScale - st.eyeRedness) * physAlpha;
  st.pupilDilate += (emPupilDilate * mScale - st.pupilDilate) * physAlpha;
  st.sweatAmount += (emSweatAmount * mScale - st.sweatAmount) * physAlpha;
  st.blushAmount += (emBlushAmount * mScale - st.blushAmount) * physAlpha;

  // 5) Decoupled 3D cranium, cervical neck & anatomical torso/shoulder kinematics
  if (listening && now >= st.nextNodAt) {
    st.nodImpulse = 0.65 + Math.random() * 0.35;
    st.nextNodAt = now + 1800 + Math.random() * 2000;
  } else if (!listening) {
    st.nodImpulse *= Math.exp(-dt * 2.5);
  }
  const nodWave = listening ? Math.sin(t * 4.4) * st.nodImpulse * 0.42 : 0;
  st.nodImpulse *= Math.exp(-dt * 2.2);

  const breathFreq =
    emotion === "nervous" || emotion === "sweating" || emotion === "fear"
      ? 2.4
      : isAlteredSlow
        ? 1.05
        : 1.45;
  const breath = Math.sin(t * breathFreq) * 0.5 + 0.5;
  const gazeHeadLeadX = winkPreparing ? 0 : listening ? st.gazeX * 0.12 : st.gazeX * 0.24;
  const suppressUpwardHeadLead = emotion === "intense" || emotion === "eyeroll";
  const gazeHeadLeadY =
    winkPreparing || suppressUpwardHeadLead ? 0 : listening ? st.gazeY * 0.1 : st.gazeY * 0.16;
  const speechNod = active ? Math.sin(t * 3.1) * st.browBeat * 0.3 : 0;
  const chuckleBounce = emotion === "laugh" ? Math.sin(t * 6.4) * 0.1 : 0;
  const drunkSwayYaw =
    emotion === "drunk"
      ? Math.sin(t * 0.95) * 0.24
      : emotion === "high"
        ? Math.sin(t * 0.7) * 0.14
        : 0;

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
        drunkSwayYaw +
        (active ? Math.sin(t * 1.45) * visemes.level * 0.1 : 0) +
        gazeHeadLeadX * 0.72 +
        empathy.headYaw * 0.85) *
      mScale;

  const idlePitchDriftScale = emotion === "intense" ? 0.35 : 1;
  const targetPitch = directMirror
    ? mirrorPitch * mScale
    : ((Math.sin(t * 0.74) * (listening ? 0.11 : 0.15) + Math.cos(t * 0.34) * 0.08 + 0.02) *
        idlePitchDriftScale +
        (active ? -st.open * 0.05 - speechNod * 0.52 : nodWave * 0.88) +
        chuckleBounce * 0.55 +
        emHeadTiltPitch +
        swallowChinDip +
        gazeHeadLeadY * 0.68 +
        empathy.headPitch * 0.8) *
      mScale;

  const poseRollBias =
    activePose === "curiousTilt" ||
    emotion === "flirty" ||
    emotion === "smirk" ||
    emotion === "compassionate"
      ? st.poseDir * 0.045
      : isSideThinking
        ? st.poseDir * 0.02
        : 0;
  const drunkRoll =
    emotion === "drunk"
      ? Math.sin(t * 1.15) * 0.065
      : emotion === "high"
        ? Math.sin(t * 0.75) * 0.04
        : 0;
  const targetRoll = directMirror
    ? (userFace?.headRoll ?? 0) * 0.75 * mScale
    : (Math.sin(t * 0.62) * 0.024 +
        Math.cos(t * 0.28) * 0.012 +
        poseRollBias +
        drunkRoll +
        empathy.headRoll * 0.45 +
        (listening ? 0.012 : 0) +
        (winkStrength > 0.1 ? st.winkSide * 0.016 : 0) +
        (emotion === "surprised" ? 0.014 * eb : 0)) *
      mScale;

  const headAlpha = 1 - Math.exp(-dt * (directMirror ? 7.8 : isAlteredSlow ? 2.4 : 4.2));
  const yawLimit = directMirror ? 0.68 : 0.56;
  const pitchLimit = directMirror ? 0.52 : 0.44;
  st.headYaw += (Math.max(-yawLimit, Math.min(yawLimit, targetYaw)) - st.headYaw) * headAlpha;
  st.headPitch +=
    (Math.max(-pitchLimit, Math.min(pitchLimit, targetPitch)) - st.headPitch) * headAlpha;
  st.headRoll += (targetRoll - st.headRoll) * headAlpha;

  // Keep cervical neck & shoulders quiet during normal head turns; follow softly only on strong turns or mirror gestures
  const deepYaw = Math.max(0, (Math.abs(st.headYaw) - (directMirror ? 0.22 : 0.34)) / 0.48);
  const neckFollowRatio = directMirror
    ? 0.06 + Math.min(0.28, deepYaw * 0.3)
    : 0.04 + Math.min(0.18, deepYaw * 0.2);
  const neckIndepYaw = directMirror
    ? 0
    : (Math.sin(t * 0.18 + 1.2) * 0.018 +
        (activePose === "attentiveWarm" ? st.poseDir * 0.014 : 0)) *
      mScale;
  const neckIndepPitch = directMirror
    ? 0
    : (Math.cos(t * 0.24 + 0.5) * 0.016 + swallowWave * 0.018) * mScale;
  const neckAlpha = 1 - Math.exp(-dt * 2.8);
  st.neckYaw += (st.headYaw * neckFollowRatio + neckIndepYaw - st.neckYaw) * neckAlpha;
  st.neckPitch +=
    (st.headPitch * (directMirror ? 0.24 : 0.12) + neckIndepPitch - st.neckPitch) * neckAlpha;
  st.neckRoll += (st.headRoll * (directMirror ? 0.26 : 0.14) - st.neckRoll) * neckAlpha;

  // Attentive forward lean / camera approach perspective
  const targetLean = directMirror
    ? Math.max(-0.32, Math.min(0.6, -mirrorPitch * 0.58)) * mScale
    : (Math.sin(t * 0.38) * 0.24 +
        Math.cos(t * 0.19) * 0.1 +
        (listening ? 0.24 : 0) +
        (activePose === "attentiveWarm" || activePose === "curiousTilt" ? 0.2 : 0) +
        (activePose === "softPonder" || isSideThinking ? -0.14 : 0) +
        (emotion === "fear"
          ? -0.32 * eb
          : emotion === "angry" || emotion === "furious" || emotion === "intense"
            ? 0.34 * eb
            : emotion === "flirty" || emotion === "compassionate"
              ? 0.22 * eb
              : 0) +
        -st.headPitch * 0.12) *
      mScale;
  st.forwardLean += (Math.max(-0.4, Math.min(0.65, targetLean)) - st.forwardLean) * headAlpha;

  // Rare subtle tongue articulation / phoneme tongue tip lift
  if (
    !directMirror &&
    !opts.reducedMotion &&
    !listening &&
    !isSolemnOrTense &&
    now >= st.nextTongueAt &&
    now >= st.tongueUntil
  ) {
    st.tongueUntil = now + 460;
    st.nextTongueAt = now + 26000 + Math.random() * 28000;
  }
  let targetTongue = directMirror ? Math.max(0, userFace?.tongueOut ?? 0) * 0.96 : 0;
  if (!directMirror && now < st.tongueUntil) {
    const tu = 1 - (st.tongueUntil - now) / 460;
    targetTongue =
      Math.sin(Math.min(1, Math.max(0, tu)) * Math.PI) * (emotion === "laugh" ? 0.55 : 0.38);
  } else if (
    !directMirror &&
    active &&
    ((visemes.tongue ?? 0) > 0.08 || (visemes.level > 0.16 && visemes.round > 0.18))
  ) {
    targetTongue = Math.min(0.46, Math.max((visemes.tongue ?? 0) * 0.56, visemes.round * 0.2));
  }
  st.tongueOut += (targetTongue - st.tongueOut) * (1 - Math.exp(-dt * 15));

  // Minimal torso & shoulder sway in calls (reserve stronger coupled tilts for direct mirror or occasional big leans)
  const targetTorsoYaw = directMirror
    ? mirrorYaw * 0.16 * mScale
    : (Math.sin(t * 0.22) * 0.04 + st.neckYaw * 0.22) * mScale;
  const targetTorsoSwayX = directMirror
    ? mirrorYaw * 0.1 * mScale
    : (Math.sin(t * 0.32) * 0.05 +
        st.neckYaw * 0.14 +
        (emotion === "drunk" ? Math.sin(t * 0.9) * 0.18 : 0)) *
      mScale;
  const targetShoulderTilt = directMirror
    ? (userFace?.headRoll ?? 0) * 0.28 * mScale
    : (Math.sin(t * 0.44) * 0.08 + st.neckRoll * 0.65) * mScale;
  const torsoAlpha = 1 - Math.exp(-dt * 3.2);
  st.torsoYaw += (targetTorsoYaw - st.torsoYaw) * torsoAlpha;
  st.torsoSwayX += (targetTorsoSwayX - st.torsoSwayX) * torsoAlpha;
  st.shoulderTilt += (targetShoulderTilt - st.shoulderTilt) * torsoAlpha;
  st.shoulderShrug +=
    ((directMirror ? emShoulderShrug : emShoulderShrug * 0.55) * mScale - st.shoulderShrug) *
    torsoAlpha;

  // Keep hair strictly unified with the cranium (zero independent shear or detachment)
  st.hairShiftX = 0;
  st.hairShiftY = 0;

  // Eyebrows normally move symmetrically; raise single brow directly in mirror mode or on intentional asymmetric cues
  if (!directMirror && now >= st.nextBrowAccentAt) {
    st.browAccentUntil = now + 1150;
    st.browAccentSide = Math.random() < 0.5 ? 1 : -1;
    st.nextBrowAccentAt = now + 6500 + Math.random() * 6500;
  }
  let browAsym = directMirror
    ? mirrorBrowAsym * mScale
    : emExplicitBrowAsym * st.browAccentSide * mScale;
  if (
    !directMirror &&
    !isSolemnOrTense &&
    (now < st.browAccentUntil || activePose === "curiousTilt")
  ) {
    const u =
      now < st.browAccentUntil
        ? Math.max(0, Math.min(1, 1 - (st.browAccentUntil - now) / 1150))
        : 0.5;
    browAsym += Math.sin(u * Math.PI) * 0.16 * st.browAccentSide * mScale;
  }

  return {
    blink,
    blinkR,
    blinkL,
    lidDroop: st.lidDroop,
    open: Math.max(st.open, st.tongueOut * 0.36),
    wide: st.wide,
    round: st.round,
    lipPress: st.lipPress,
    labiodental: st.labiodental,
    smile: st.smile,
    smileR: st.smileR,
    smileL: st.smileL,
    mouthSkewX: st.mouthSkewX,
    upperLipLift: st.upperLipLift,
    lowerLipDrop: st.lowerLipDrop,
    lipBite: st.lipBite,
    cheekLift: st.cheekLift,
    innerBrowY: st.innerBrowY,
    innerBrowPinch: st.innerBrowPinch,
    outerBrowY: st.outerBrowY,
    browAsym,
    eyeSquint: st.eyeSquint,
    eyeWide: st.eyeWide,
    eyeRoll: st.eyeRoll,
    noseFlare: st.noseFlare,
    gazeX: st.gazeX,
    gazeY: st.gazeY,
    headYaw: st.headYaw,
    headPitch: st.headPitch,
    headRoll: st.headRoll,
    neckYaw: st.neckYaw,
    neckPitch: st.neckPitch,
    neckRoll: st.neckRoll,
    torsoYaw: st.torsoYaw,
    headTx: st.torsoSwayX * 0.0042,
    headTy: ((breath - 0.5) * 0.003 + nodWave * 0.005) * mScale,
    torsoBreath:
      (breath - 0.5) * (emotion === "sweating" || emotion === "nervous" ? 1.45 : 1.0) * mScale,
    torsoSwayX: st.torsoSwayX,
    shoulderTilt: st.shoulderTilt,
    shoulderShrug: st.shoulderShrug,
    handShiftX: Math.sin(t * 1.18) * 0.0024 * mScale,
    handShiftY: ((breath - 0.5) * 0.003 + (active ? visemes.level * 0.0045 : 0)) * mScale,
    hairShiftX: st.hairShiftX,
    hairShiftY: st.hairShiftY,
    forwardLean: st.forwardLean,
    tongueOut: st.tongueOut,
    tearAmount: st.tearAmount,
    eyeWater: st.eyeWater,
    eyeRedness: st.eyeRedness,
    pupilDilate: st.pupilDilate,
    sweatAmount: st.sweatAmount,
    blushAmount: st.blushAmount,
    swallowAmount: st.swallowAmount,
  };
}
