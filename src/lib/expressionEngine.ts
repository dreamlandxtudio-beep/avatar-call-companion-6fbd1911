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
  pucker: number;
  lipPress: number;
  labiodental: number;
  smile: number;
  smileR: number;
  smileL: number;
  smileTeeth: number;
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
  swallowPhase: number;
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
  wasThinking: boolean;
  thinkUntil: number;
  thinkDir: 1 | -1;
  preSpeechUntil: number;
  emotionBlend: number;
  lidDroop: number;
  open: number;
  wide: number;
  round: number;
  pucker: number;
  lipPress: number;
  labiodental: number;
  smile: number;
  smileR: number;
  smileL: number;
  smileTeeth: number;
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
  swallowPhase: number;
  swallowStrength: number;
  speechTurnsSinceSwallow: number;
  swallowTargetTurns: number;
  lastSwallowSeq: number;
  manualSwallowActive: boolean;
  prevVisemeOpen: number;
  browBeat: number;
};

function pickExpressionPose(preferThink: boolean, isSpeaking = false): ExpressionPose {
  const r = Math.random();
  if (preferThink) {
    if (r < 0.56) return "thinkSide";
    if (r < 0.72) return "softPonder";
    if (r < 0.84) return "thoughtfulFurrow";
    if (r < 0.92) return "curiousTilt";
    return "browInterest";
  }
  if (isSpeaking) {
    // While actively answering, stay mostly engaged toward the user and glance aside only rarely
    if (r < 0.34) return "attentiveWarm";
    if (r < 0.58) return "browInterest";
    if (r < 0.76) return "curiousTilt";
    if (r < 0.88) return "neutral";
    if (r < 0.94) return "gentleSmirk";
    return "thinkSide";
  }
  if (r < 0.24) return "attentiveWarm";
  if (r < 0.42) return "thinkSide";
  if (r < 0.54) return "softPonder";
  if (r < 0.68) return "curiousTilt";
  if (r < 0.82) return "browInterest";
  if (r < 0.92) return "thoughtfulFurrow";
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
    wasThinking: false,
    thinkUntil: now + 1400,
    thinkDir: 1,
    preSpeechUntil: 0,
    emotionBlend: 0.5,
    lidDroop: 0,
    open: 0,
    wide: 0,
    round: 0,
    pucker: 0,
    lipPress: 0,
    labiodental: 0,
    smile: 0,
    smileR: 0,
    smileL: 0,
    smileTeeth: 0,
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
    swallowPhase: 0,
    swallowStrength: 1,
    speechTurnsSinceSwallow: 0,
    swallowTargetTurns: Math.random() < 0.55 ? 2 : 3,
    lastSwallowSeq: 0,
    manualSwallowActive: false,
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
    thinking?: boolean | undefined;
    userSpeaking?: boolean | undefined;
    emotion: AvatarEmotion;
    visemes: AudioVisemes;
    hasVisibleTeeth: boolean;
    restMouthRatio: number;
    userFace?: UserFaceObservation | undefined;
    reducedMotion?: boolean | undefined;
    mirrorMode?: boolean | undefined;
    animeStyle?: boolean | undefined;
    swallowSignal?: { seq: number; strength?: number | undefined } | undefined;
    topicComplexity?: number | undefined;
  },
): ExpressionFrame {
  const dt = Math.min(0.05, Math.max(0.004, (now - st.lastNow) * 0.001));
  st.lastNow = now;
  const t = now * 0.001;
  const mScale = opts.reducedMotion ? 0.15 : 1.0;

  const { active, listening, emotion, visemes, userFace } = opts;
  const thinking = !!opts.thinking;
  const userSpeaking = !!opts.userSpeaking;
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
        browAsym: 0,
        frown: 0,
        eyeWide: 0,
        pucker: 0,
        mouthSkewX: 0,
        noseScrunch: 0,
        headYaw: 0,
        headPitch: 0,
        headRoll: 0,
      }
    : stepObservedEmpathy(st.empathy, now, userFace, { active, listening });

  const isSolemnOrTense = SUPPRESS_POSITIVE_EMOTIONS.has(emotion);
  const isAlteredSlow = emotion === "drunk" || emotion === "high" || emotion === "sleepy";

  const complexTopicEmotions =
    emotion === "nervous" ||
    emotion === "fear" ||
    emotion === "sweating" ||
    emotion === "sad" ||
    emotion === "crying" ||
    emotion === "heartbroken" ||
    emotion === "angry" ||
    emotion === "furious" ||
    emotion === "intense" ||
    emotion === "compassionate" ||
    emotion === "bitingLip" ||
    emotion === "pain" ||
    emotion === "disgusted" ||
    emotion === "skeptical";
  const topicComplexity = Math.max(
    Math.min(1, Math.max(0, opts.topicComplexity ?? 0)),
    complexTopicEmotions ? 0.68 : 0,
  );

  // Manual swallow trigger from Mirror UI works immediately in any mode
  if (opts.swallowSignal && opts.swallowSignal.seq !== st.lastSwallowSeq) {
    st.lastSwallowSeq = opts.swallowSignal.seq;
    const reqStrength = Math.max(0.55, Math.min(1.5, opts.swallowSignal.strength ?? 1.0));
    st.swallowStrength = reqStrength;
    st.manualSwallowActive = true;
    st.swallowStartAt = now;
    st.swallowUntil = now + (reqStrength > 1.15 ? 900 : 720);
  }

  // State transitions:
  // - Listening/userSpeaking: immediate steady direct eye contact with user
  // - Thinking (right after user finishes speaking, before AI responds): thoughtful upward-side glance
  // - Active speaking: brief carry-over from thinking, then return promptly to eye contact
  if (thinking && !st.wasThinking && !directMirror) {
    st.thinkDir = Math.random() < 0.5 ? -1 : 1;
    st.thinkUntil = now + 1800;
    st.poseMode = "thinkSide";
    st.poseDir = st.thinkDir;
    st.poseUntil = now + 1800;
    st.nextPoseAt = now + 2200;
    st.nextSaccadeAt = Math.min(st.nextSaccadeAt, now + 90);
  }
  if (active && !st.wasActive) {
    st.thinkUntil = st.wasThinking ? now + 520 : now + 420;
    st.thinkDir = Math.random() < 0.5 ? -1 : 1;
    st.preSpeechUntil = now + 380;
    st.poseMode = st.wasThinking ? "thinkSide" : "attentiveWarm";
    st.poseDir = st.thinkDir;
    st.poseUntil = now + (st.wasThinking ? 640 : 1200);
    st.nextPoseAt = now + 2600 + Math.random() * 1800;
    st.nextSaccadeAt = now + (st.wasThinking ? 540 : 180);
    if (!st.manualSwallowActive) {
      st.swallowStartAt = 0;
      st.swallowUntil = 0;
    }
  } else if (!active && st.wasActive) {
    if (!listening && !thinking) {
      st.thinkUntil = now + 750;
      st.poseMode = "attentiveWarm";
      st.poseDir = Math.random() < 0.5 ? -1 : 1;
      st.poseUntil = now + 1800;
    }
    if (!directMirror && !st.manualSwallowActive) {
      st.speechTurnsSinceSwallow += 1;
      if (st.speechTurnsSinceSwallow >= st.swallowTargetTurns && now >= st.nextSwallowEligibleAt) {
        const isComplex = topicComplexity >= 0.42;
        st.speechTurnsSinceSwallow = 0;
        st.swallowTargetTurns = Math.random() < 0.55 ? 2 : 3;
        st.swallowStrength = isComplex
          ? 1.26 + Math.min(0.24, topicComplexity * 0.24)
          : 0.85 + Math.random() * 0.18;
        st.swallowStartAt = now + (isComplex ? 520 : 820) + Math.random() * 680;
        st.swallowUntil = st.swallowStartAt + (isComplex ? 880 : 700);
        st.nextSwallowEligibleAt = now + 9000 + Math.random() * 6000;
      }
    }
  }
  if ((listening || userSpeaking) && (!st.wasListening || userSpeaking)) {
    st.listeningGlanceActive = false;
    st.targetGazeX = 0;
    st.targetGazeY = 0;
    st.nextSaccadeAt = now + 2800 + Math.random() * 2200;
  }
  st.wasActive = active;
  st.wasListening = listening;
  st.wasThinking = thinking;

  // Cycle subtle pose accents naturally during idle & speech
  if (!directMirror && !listening && !thinking && now >= st.nextPoseAt) {
    st.poseMode = pickExpressionPose(false, active);
    st.poseDir = Math.random() < 0.5 ? -1 : 1;
    const dur =
      st.poseMode === "thinkSide" ? 1100 + Math.random() * 650 : 1600 + Math.random() * 1400;
    st.poseUntil = now + dur;
    st.nextPoseAt = now + dur + 1800 + Math.random() * 2200;
    if (st.poseMode === "thinkSide" || st.poseMode === "softPonder") {
      st.thinkDir = st.poseDir;
    }
  }
  const activePose: ExpressionPose =
    !directMirror && !listening && (thinking || now < st.poseUntil)
      ? thinking
        ? "thinkSide"
        : st.poseMode
      : "neutral";
  const isSideThinking =
    !directMirror &&
    !listening &&
    (thinking ||
      activePose === "thinkSide" ||
      (active && now < st.thinkUntil) ||
      (active && visemes.level < 0.06 && emotion !== "laugh" && activePose === "softPonder"));

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

  // 2) Eye contact & ocular saccades (steady eye contact while listening/answering; thoughtful look when thinking)
  if (winkPreparing || (userSpeaking && !directMirror)) {
    st.targetGazeX = 0;
    st.targetGazeY = 0;
    st.listeningGlanceActive = false;
  } else if (now >= st.nextSaccadeAt) {
    if (thinking) {
      st.targetGazeX = st.thinkDir * (0.44 + Math.random() * 0.18);
      st.targetGazeY = 0.14 + Math.random() * 0.12;
      st.nextSaccadeAt = now + 680 + Math.random() * 480;
    } else if (listening) {
      // Steady attentive eye contact while listening to the user (only ultra-rare micro-blink glance when silent)
      if (st.listeningGlanceActive) {
        st.listeningGlanceActive = false;
        st.targetGazeX = (Math.random() - 0.5) * 0.02;
        st.targetGazeY = (Math.random() - 0.5) * 0.02;
        st.nextSaccadeAt = now + 4600 + Math.random() * 3400;
      } else if (Math.random() < 0.06) {
        st.listeningGlanceActive = true;
        st.targetGazeX = (Math.random() < 0.5 ? -1 : 1) * (0.2 + Math.random() * 0.1);
        st.targetGazeY = (Math.random() - 0.5) * 0.08;
        st.nextSaccadeAt = now + 260 + Math.random() * 110;
      } else {
        st.targetGazeX = (Math.random() - 0.5) * 0.03;
        st.targetGazeY = (Math.random() - 0.5) * 0.02;
        st.nextSaccadeAt = now + 2600 + Math.random() * 2000;
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
      st.targetGazeX = st.poseDir * (0.52 + Math.random() * 0.22);
      st.targetGazeY =
        activePose === "softPonder" ? -0.1 - Math.random() * 0.08 : 0.08 + Math.random() * 0.14;
      st.nextSaccadeAt = now + 680 + Math.random() * 480;
    } else if (active) {
      // During speech, maintain clear eye contact ~88% of the time; glance aside only briefly and rarely
      const rareThinkGlance = Math.random() < 0.12;
      st.targetGazeX = rareThinkGlance
        ? st.thinkDir * (0.36 + Math.random() * 0.22)
        : (Math.random() - 0.5) * 0.06;
      st.targetGazeY = rareThinkGlance ? 0.08 + Math.random() * 0.12 : (Math.random() - 0.5) * 0.04;
      if (Math.random() < 0.45) st.thinkDir = st.thinkDir === 1 ? -1 : 1;
      st.nextSaccadeAt =
        now + (rareThinkGlance ? 480 + Math.random() * 280 : 1800 + Math.random() * 1400);
    } else {
      const idleGlance = Math.random() < 0.26;
      st.targetGazeX = idleGlance
        ? st.poseDir * (0.38 + Math.random() * 0.26)
        : (Math.random() - 0.5) * 0.08;
      st.targetGazeY = (Math.random() - 0.5) * 0.12;
      st.nextSaccadeAt =
        now + (idleGlance ? 750 + Math.random() * 550 : 1500 + Math.random() * 1200);
    }
  }

  const microDriftX =
    (listening || userSpeaking || winkPreparing ? 0.008 : 0.022) * Math.sin(t * 1.55) * mScale;
  const microDriftY =
    (listening || userSpeaking || winkPreparing ? 0.006 : 0.014) * Math.cos(t * 1.1) * mScale;

  const mirrorGazeX = directMirror ? userFace.gazeX : 0;
  const mirrorGazeY = directMirror ? userFace.gazeY : 0;

  // Vestibulo-ocular eye-contact compensation (VOR): when head turns/tilts while maintaining screen eye contact,
  // counter-steer pupils opposite to headYaw/headPitch so eyes stay locked on the viewer rather than turning away with the face.
  const lookingAwayExplicit =
    isSideThinking ||
    thinking ||
    activePose === "softPonder" ||
    emotion === "eyeroll" ||
    emotion === "blushing" ||
    Math.abs(st.targetGazeX) > 0.22;
  const autoVorStrength =
    directMirror || lookingAwayExplicit
      ? 0
      : listening || userSpeaking
        ? 0.62
        : active
          ? 0.58
          : 0.5;
  const mirrorCenterLock =
    directMirror && Math.abs(mirrorGazeX) < 0.28 && Math.abs(st.headYaw) > 0.08
      ? (1 - Math.abs(mirrorGazeX) / 0.28) * 0.54
      : 0;
  const vorGazeX = -(st.headYaw * (autoVorStrength + mirrorCenterLock)) * 0.86;
  const vorGazeY = -(st.headPitch * (autoVorStrength * 0.32 + mirrorCenterLock * 0.36));

  const targetGx = Math.max(
    -1.05,
    Math.min(
      1.05,
      (directMirror ? 0 : winkPreparing || userSpeaking ? 0 : st.targetGazeX + microDriftX) +
        mirrorGazeX +
        vorGazeX,
    ),
  );
  const targetGy = Math.max(
    -0.82,
    Math.min(
      0.95,
      (directMirror ? 0 : winkPreparing || userSpeaking ? 0 : st.targetGazeY + microDriftY) +
        mirrorGazeY +
        vorGazeY,
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

  const gazeAlpha =
    1 -
    Math.exp(-dt * (directMirror ? 36 : winkPreparing || listening ? 30 : isAlteredSlow ? 10 : 24));
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
    gestureFurrow = Math.max(gestureFurrow, 0.2 * mScale);
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

  const poseWarm = activePose === "attentiveWarm" && !isSolemnOrTense ? 0.22 * mScale : 0;
  const poseBrowLift =
    activePose === "browInterest"
      ? 0.24 * mScale
      : activePose === "curiousTilt"
        ? 0.18 * mScale
        : 0;
  const poseSquint =
    isSideThinking || activePose === "thoughtfulFurrow"
      ? 0.13 * mScale
      : activePose === "attentiveWarm"
        ? 0.15 * mScale
        : 0;

  const speechWarmth =
    active && !isSolemnOrTense
      ? (0.14 + Math.max(0, Math.sin(t * 1.55)) * 0.18) * (0.48 + visemes.level * 0.52)
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
  let emWide = speechWarmth * 0.26;
  let emRound = activePose === "softPonder" ? 0.08 * mScale : isSideThinking ? 0.05 * mScale : 0;
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
        (opts.hasVisibleTeeth ? 0.78 : 0.88) * eb + ambientWarmth * 0.22 + speechWarmth * 0.28;
      emWide = 0.42 * eb;
      emUpperLipLift = 0.34 * eb;
      emLowerLipDrop = 0.14 * eb;
      emCheek = 0.84 * eb;
      emSquint = 0.52 * eb;
      emInnerBrowY = 0.24 * eb + ambientBrow * 0.25;
      emOuterBrowY = 0.26 * eb + ambientBrow * 0.25;
      emNoseFlare = 0.16 * eb;
      emBlushAmount = 0.3 * eb;
      break;
    }
    case "euphoric": {
      emSmile = (opts.hasVisibleTeeth ? 0.9 : 0.98) * eb;
      emWide = 0.54 * eb;
      emUpperLipLift = 0.44 * eb;
      emLowerLipDrop = 0.28 * eb;
      emCheek = 0.94 * eb;
      emSquint = 0.48 * eb;
      emEyeWide = 0.28 * eb;
      emInnerBrowY = 0.56 * eb;
      emOuterBrowY = 0.6 * eb;
      emOpen = Math.max(emOpen, active ? 0.32 + visemes.open * 0.28 : 0.36 * eb);
      emNoseFlare = 0.28 * eb;
      emHeadTiltPitch = 0.14 * eb;
      emPupilDilate = 0.3 * eb;
      emBlushAmount = 0.42 * eb;
      emShoulderShrug = 0.24 * eb;
      break;
    }
    case "laugh": {
      // Smooth breathy laughter cadence (explicit joyful smile + wide mouth + cheek lift)
      const breathWave = 0.5 + 0.5 * Math.sin(t * 6.4 + Math.sin(t * 1.9) * 0.65);
      emSmile = (opts.hasVisibleTeeth ? 0.94 : 1.0) * eb;
      emWide = 0.56 * eb;
      emUpperLipLift = 0.46 * eb;
      emLowerLipDrop = 0.34 * eb;
      emCheek = 0.96 * eb;
      emSquint = 0.72 * eb;
      emInnerBrowY = 0.38 * eb;
      emOuterBrowY = 0.42 * eb;
      emNoseFlare = 0.34 * eb;
      emEyeWater = 0.28 * eb;
      emBlushAmount = 0.4 * eb;
      emShoulderShrug = (0.16 + breathWave * 0.24) * eb;
      emOpen = Math.max(
        emOpen,
        active ? 0.36 + visemes.open * 0.36 : (0.48 + breathWave * 0.3) * eb,
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
      emOpen = Math.max(emOpen, active ? 0.38 : 0.48 * eb);
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
      emOpen = Math.max(emOpen, active ? 0 : yawnCycle * 0.68 * eb);
      emRound = Math.max(emRound, yawnCycle * 0.46 * eb);
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
  emExplicitBrowAsym += empathy.browAsym * 0.55 * empathyWeight;
  emEyeWide += empathy.eyeWide * 0.68 * empathyWeight;
  emMouthSkewX += empathy.mouthSkewX * 0.45 * empathyWeight;
  const emPucker = empathy.pucker * 0.52 * empathyWeight;
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
    emEyeWide = Math.min(1, (userFace.eyeWide ?? 0) * 1.12);
    if (userFace.gazeY > 0.45) {
      emEyeRoll = Math.min(1, (userFace.gazeY - 0.45) * 1.6);
    }
  }

  // 4) Phoneme-differentiated speech prosody & coarticulation (conversational by default, visibly wider on emphatic stress & wide emotions)
  const visPress = active ? Math.min(1, visemes.press ?? 0) : 0;
  const visLabio = active ? Math.min(1, visemes.labiodental ?? 0) : 0;
  const visStress = active ? Math.min(1, visemes.stress ?? 0) : 0;
  const emphaticPeak =
    active && visemes.open > 0.52 && visStress > 0.2
      ? ((visemes.open - 0.52) / 0.48) * Math.min(1, (visStress - 0.18) * 1.48) * 0.36
      : 0;
  const speechOpen = active
    ? Math.min(
        1.1,
        (visemes.open * 0.88 + emphaticPeak) * (1 - visPress * 0.85) * (1 - visLabio * 0.45),
      )
    : 0;
  const openDelta = Math.max(0, speechOpen - st.prevVisemeOpen);
  st.prevVisemeOpen = speechOpen;
  st.browBeat = Math.max(st.browBeat * Math.exp(-dt * 5.2), openDelta * 2.2 + visStress * 0.45);

  // Trigger stronger thoughtful swallows during complicated-topic speech pauses too
  if (
    active &&
    !directMirror &&
    !st.manualSwallowActive &&
    st.swallowStartAt === 0 &&
    visemes.level < 0.14 &&
    st.open < 0.08 &&
    topicComplexity >= 0.42 &&
    now >= st.nextSwallowEligibleAt &&
    Math.random() < 0.42
  ) {
    st.speechTurnsSinceSwallow = 0;
    st.swallowStrength = 1.24 + Math.min(0.24, topicComplexity * 0.24);
    st.swallowStartAt = now + 50;
    st.swallowUntil = st.swallowStartAt + 840;
    st.nextSwallowEligibleAt = now + 11500 + Math.random() * 8000;
  }

  let swallowWave = 0;
  let swallowPhase = 0;
  let swallowLipContact = 0;
  let swallowChinDip = 0;
  if (
    (st.manualSwallowActive || !directMirror) &&
    st.swallowStartAt > 0 &&
    now >= st.swallowStartAt &&
    now < st.swallowUntil
  ) {
    const swU = Math.min(
      1,
      Math.max(0, (now - st.swallowStartAt) / Math.max(1, st.swallowUntil - st.swallowStartAt)),
    );
    const speechFade =
      st.manualSwallowActive || !active
        ? 1
        : Math.max(0.32, 1 - Math.max(0, visemes.level - 0.14) * 2.2);
    swallowWave = Math.sin(swU * Math.PI) * st.swallowStrength * speechFade;
    swallowPhase = swU;
    swallowLipContact =
      active && !st.manualSwallowActive
        ? 0
        : Math.sin(Math.min(1, swU / 0.55) * Math.PI) *
          (swU < 0.55 ? 0.24 * st.swallowStrength : 0);
    swallowChinDip = -swallowWave * 0.04;
  } else if (now >= st.swallowUntil) {
    st.swallowStartAt = 0;
    st.swallowUntil = 0;
    st.manualSwallowActive = false;
  }
  st.swallowAmount += (swallowWave * mScale - st.swallowAmount) * (1 - Math.exp(-dt * 18));
  st.swallowPhase += (swallowPhase - st.swallowPhase) * (1 - Math.exp(-dt * 24));

  const gatedOpen = Math.max(speechOpen, visPress > 0.4 ? 0 : emOpen);
  const isWideEmotion =
    emotion === "laugh" ||
    emotion === "surprised" ||
    emotion === "euphoric" ||
    emotion === "sleepy";
  const openDrive = directMirror
    ? gatedOpen
    : gatedOpen *
      (gatedOpen > 0.72 || emphaticPeak > 0.08 || (isWideEmotion && emOpen > 0.52)
        ? 0.96
        : mouthAmp);
  const motion = mouthMotion(
    gatedOpen < 0.004 ? 0 : Math.min(1, openDrive),
    opts.hasVisibleTeeth,
    opts.restMouthRatio,
  );
  const targetOpen = Math.min(
    0.98,
    motion.opening * (directMirror ? 1.02 : 0.94) + emphaticPeak * 0.16,
  );
  const targetWide = Math.min(1, (active ? Math.max(visemes.wide, emWide) : emWide) * wideAmp);
  const targetRound = Math.min(
    1,
    active ? Math.max(visemes.round, visemes.pucker ?? 0, emRound) : emRound,
  );
  const targetPucker = directMirror
    ? Math.min(1, userFace.mouthPucker ?? (userFace.mouthRound ?? 0) * 0.72)
    : active
      ? Math.min(1, Math.max(visemes.pucker ?? visemes.round * 0.58, emPucker * 0.45))
      : Math.min(1, Math.max(activePose === "softPonder" ? 0.18 * mScale : 0, emPucker));
  const targetPress = active
    ? Math.max(visPress, emPress, emotion === "angry" ? 0.22 : 0)
    : Math.max(emPress, swallowLipContact, emotion === "angry" || emotion === "furious" ? 0.24 : 0);

  const openRate = directMirror
    ? targetOpen > st.open
      ? 32.0
      : 26.0
    : targetOpen > st.open
      ? 26.0
      : 20.0;
  const mouthAlpha = 1 - Math.exp(-dt * openRate);
  st.open += (targetOpen - st.open) * mouthAlpha;
  st.wide += (targetWide - st.wide) * (1 - Math.exp(-dt * (directMirror ? 26 : 18)));
  st.round += (targetRound - st.round) * (1 - Math.exp(-dt * (directMirror ? 26 : 18)));
  st.pucker += (targetPucker - st.pucker) * (1 - Math.exp(-dt * (directMirror ? 28 : 19)));
  st.lipPress += (targetPress - st.lipPress) * (1 - Math.exp(-dt * (directMirror ? 28 : 22)));
  st.labiodental += (visLabio - st.labiodental) * (1 - Math.exp(-dt * 20));
  // Support closed-lip warm smiles vs parted-teeth smiles faithfully
  const mirrorTeethParted = directMirror
    ? (userFace.mouthOpen > 0.022 ||
        (userFace.upperLipLift ?? 0) > 0.2 ||
        (userFace.lowerLipDrop ?? 0) > 0.22) &&
      (userFace.mouthPress ?? 0) < 0.25 &&
      targetPucker < 0.28
    : emotion === "laugh" ||
      emotion === "euphoric" ||
      (emotion === "happy" && (active || opts.hasVisibleTeeth || Math.sin(t * 0.55) > 0.15));
  const targetSmileTeeth = mirrorTeethParted ? 1 : 0;
  st.smileTeeth +=
    (targetSmileTeeth - st.smileTeeth) * (1 - Math.exp(-dt * (directMirror ? 22 : 16)));
  const smilePartBoost =
    Math.max(0, emSmile - 0.34) * st.smileTeeth * (1 - Math.min(1, targetOpen * 5));
  st.upperLipLift +=
    (Math.min(1, emUpperLipLift + smilePartBoost * 0.36 + (active ? st.wide * 0.14 : 0)) -
      st.upperLipLift) *
    (1 - Math.exp(-dt * (directMirror ? 26 : 20)));
  st.lowerLipDrop +=
    (Math.min(1, emLowerLipDrop + smilePartBoost * 0.32 + (active ? st.open * 0.12 : 0)) -
      st.lowerLipDrop) *
    (1 - Math.exp(-dt * (directMirror ? 26 : 20)));
  st.lipBite += (emLipBite - st.lipBite) * (1 - Math.exp(-dt * (directMirror ? 24 : 18)));

  const speechBrowBoost = active ? Math.min(0.38, st.open * 0.18 + st.browBeat * 0.34) : 0;
  const exprAlpha = 1 - Math.exp(-dt * (directMirror ? 24.0 : 14.5));
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
    (1 - Math.exp(-dt * (directMirror ? 20.0 : 12.5)));
  const smileAlpha = 1 - Math.exp(-dt * (directMirror ? 22.0 : 14.0));
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
  const speechNod = active
    ? Math.sin(t * 3.8) * st.browBeat * 0.42 + Math.sin(t * 1.9) * visemes.level * 0.08
    : 0;
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
    : Math.sin(t * (active ? 0.62 : 0.48)) * (active ? 0.25 : 0.17) +
      Math.cos(t * (active ? 0.28 : 0.22)) * (active ? 0.12 : 0.08);

  const targetYaw = directMirror
    ? mirrorYaw * mScale
    : ((winkPreparing ? 0.15 : 1) * baseYawWave +
        sideTurnBias +
        drunkSwayYaw +
        (active ? Math.sin(t * 1.75) * visemes.level * 0.14 : 0) +
        gazeHeadLeadX * 0.72 +
        empathy.headYaw * 0.85) *
      mScale;

  const idlePitchDriftScale = emotion === "intense" ? 0.35 : 1;
  const targetPitch = directMirror
    ? mirrorPitch * mScale
    : ((Math.sin(t * 0.74) * (listening ? 0.11 : 0.15) + Math.cos(t * 0.34) * 0.08 + 0.02) *
        idlePitchDriftScale +
        (active ? -st.open * 0.055 - speechNod * 0.56 : nodWave * 0.88) +
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
        ? st.poseDir * 0.032
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

  const headAlpha =
    1 - Math.exp(-dt * (directMirror ? 20.0 : isAlteredSlow ? 2.6 : active ? 5.6 : 4.6));
  const yawLimit = directMirror ? 0.68 : 0.56;
  const pitchLimit = directMirror ? 0.52 : 0.44;
  st.headYaw += (Math.max(-yawLimit, Math.min(yawLimit, targetYaw)) - st.headYaw) * headAlpha;
  st.headPitch +=
    (Math.max(-pitchLimit, Math.min(pitchLimit, targetPitch)) - st.headPitch) * headAlpha;
  st.headRoll += (targetRoll - st.headRoll) * headAlpha;

  // Cervical neck follows head naturally on turns, plus subtle independent neck posture & breath
  const deepYaw = Math.max(0, (Math.abs(st.headYaw) - (directMirror ? 0.16 : 0.22)) / 0.46);
  const neckFollowRatio = directMirror
    ? 0.14 + Math.min(0.3, deepYaw * 0.32)
    : 0.14 + Math.min(0.26, deepYaw * 0.26);
  const neckIndepYaw = directMirror
    ? 0
    : (Math.sin(t * 0.21 + 1.2) * 0.038 +
        (activePose === "attentiveWarm" ? st.poseDir * 0.024 : 0)) *
      mScale;
  const neckIndepPitch = directMirror
    ? 0
    : (Math.cos(t * 0.26 + 0.5) * 0.026 + swallowWave * 0.024) * mScale;
  const neckIndepRoll = directMirror ? 0 : Math.sin(t * 0.19 + 2.1) * 0.022 * mScale;
  const neckAlpha = 1 - Math.exp(-dt * (directMirror ? 13.0 : 3.6));
  st.neckYaw += (st.headYaw * neckFollowRatio + neckIndepYaw - st.neckYaw) * neckAlpha;
  st.neckPitch +=
    (st.headPitch * (directMirror ? 0.28 : 0.18) + neckIndepPitch - st.neckPitch) * neckAlpha;
  st.neckRoll +=
    (st.headRoll * (directMirror ? 0.3 : 0.2) + neckIndepRoll - st.neckRoll) * neckAlpha;

  // Attentive forward lean / camera approach perspective
  const mirrorProx = directMirror ? (userFace?.faceProximity ?? 0) : 0;
  const targetLean = directMirror
    ? Math.max(-0.42, Math.min(0.95, mirrorProx * 0.92 - mirrorPitch * 0.34)) * mScale
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
  st.forwardLean += (Math.max(-0.45, Math.min(0.95, targetLean)) - st.forwardLean) * headAlpha;

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
  st.tongueOut += (targetTongue - st.tongueOut) * (1 - Math.exp(-dt * (directMirror ? 22 : 15)));

  // Cohesive shoulders & torso: subtle independent breath/tilt/sway + gentle coupling to deep neck/head shifts
  const breathShrug = (breath - 0.5) * 0.14;
  const targetTorsoYaw = directMirror
    ? mirrorYaw * 0.22 * mScale
    : (Math.sin(t * 0.24) * 0.08 + st.neckYaw * 0.26) * mScale;
  const targetTorsoSwayX = directMirror
    ? mirrorYaw * 0.16 * mScale
    : (Math.sin(t * 0.31) * 0.12 +
        st.neckYaw * 0.18 +
        (emotion === "drunk" ? Math.sin(t * 0.9) * 0.22 : 0)) *
      mScale;
  const targetShoulderTilt = directMirror
    ? (userFace?.headRoll ?? 0) * 0.42 * mScale
    : (Math.sin(t * 0.41 + 0.8) * 0.18 + st.neckRoll * 0.72) * mScale;
  const targetShoulderShrug = directMirror
    ? (emShoulderShrug + breathShrug + Math.max(0, mirrorProx) * 0.12) * mScale
    : (emShoulderShrug * 0.72 + breathShrug + swallowWave * 0.08) * mScale;
  const torsoAlpha = 1 - Math.exp(-dt * (directMirror ? 11.5 : 4.4));
  st.torsoYaw += (targetTorsoYaw - st.torsoYaw) * torsoAlpha;
  st.torsoSwayX += (targetTorsoSwayX - st.torsoSwayX) * torsoAlpha;
  st.shoulderTilt += (targetShoulderTilt - st.shoulderTilt) * torsoAlpha;
  st.shoulderShrug += (targetShoulderShrug - st.shoulderShrug) * torsoAlpha;

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
    pucker: st.pucker,
    lipPress: st.lipPress,
    labiodental: st.labiodental,
    smile: st.smile,
    smileR: st.smileR,
    smileL: st.smileL,
    smileTeeth: st.smileTeeth,
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
    swallowPhase: st.swallowPhase,
  };
}
