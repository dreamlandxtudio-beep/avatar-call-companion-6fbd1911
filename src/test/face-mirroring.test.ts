import { describe, expect, it, vi } from "vitest";
import { expressionFromSignals } from "@/lib/face-signals";
import { createExpressionEngine, stepExpressionEngine } from "@/lib/expressionEngine";
import { dynamicSpeechTempo, voiceProsody } from "@/lib/voice-prosody";
import { deriveSpeechEmotionTimeline, extractEmotion } from "@/lib/ai";
import { defaultVoiceForGender } from "@/lib/store";
import { textToVisemeProtos } from "@/lib/phonemes";
import { mouthMotion } from "@/lib/avatar-motion";
import {
  buildVisemeTimeline,
  defaultGeminiVoiceForGender,
  sampleVisemeTimeline,
} from "@/lib/voice";
import type { UserFaceObservation } from "@/lib/userCamera";

const face: UserFaceObservation = {
  detected: true,
  emotion: "neutral",
  smile: 0,
  browFurrow: 0,
  browRaise: 0,
  mouthOpen: 0.8,
  headYaw: 0.4,
  headPitch: 0,
  gazeX: -0.8,
  gazeY: 0,
  blink: 1,
  labelEs: "",
  contextPrompt: "",
  updatedAt: 100,
};
const options = {
  active: false,
  listening: false,
  emotion: "neutral" as const,
  visemes: { level: 0, open: 0, wide: 0, round: 0 },
  hasVisibleTeeth: false,
  restMouthRatio: 0,
  mirrorMode: true,
};
describe("Observable expression mirroring", () => {
  it("maps lowered mouth corners with raised inner brows to a sad-style pose before a frown", () => {
    expect(expressionFromSignals(0, 0, 0.4, 0.4, 0.4, 0).emotion).toBe("sad");
    expect(expressionFromSignals(0, 0, 0.4, 0, 0, 0).label).toBe("Ceño fruncido");
  });
  it("moves eyes opposite to head direction independently and mirrors open mouth and eyelids", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const state = createExpressionEngine(0);
    let frame = stepExpressionEngine(state, 100, { ...options, userFace: face });
    for (let i = 1; i < 10; i++)
      frame = stepExpressionEngine(state, 100 + i * 30, { ...options, userFace: face });
    expect(frame.headYaw).toBeGreaterThan(0);
    expect(frame.gazeX).toBeLessThan(-0.5);
    expect(frame.open).toBeGreaterThan(0.4);
    expect(frame.blink).toBe(1);
    vi.restoreAllMocks();
  });
  it("does not mirror an old camera observation", () => {
    const frame = stepExpressionEngine(createExpressionEngine(2000), 2100, {
      ...options,
      userFace: face,
    });
    expect(frame.open).toBe(0);
  });
  it("uses slower sad prosody and brighter happy prosody without fabricated voice cracks", () => {
    expect(voiceProsody("sad")).toEqual({ rate: 0.9, pitch: 0.96 });
    expect(voiceProsody("happy")).toEqual({ rate: 1.04, pitch: 1.06 });
  });
  it("distinguishes moderate smiles as happy and requires wide laughter signals for laugh", () => {
    expect(expressionFromSignals(0.56, 0.2, 0.05, 0, 0, 0).emotion).toBe("happy");
    expect(expressionFromSignals(0.4, 0.28, 0, 0, 0, 0).emotion).toBe("happy");
    expect(expressionFromSignals(0.68, 0.4, 0.05, 0, 0, 0).emotion).toBe("laugh");
    expect(expressionFromSignals(0.78, 0.28, 0.05, 0, 0, 0).emotion).toBe("laugh");
  });
  it("looks directly at the user before winking and mirrors asymmetric blinks and smiles", () => {
    const state = createExpressionEngine(0);
    state.targetGazeX = 0.7;
    state.targetGazeY = 0.3;
    state.gazeX = 0.7;
    state.gazeY = 0.3;
    state.nextWinkAt = 100;
    let frame = stepExpressionEngine(state, 100, { ...options, mirrorMode: false });
    for (let t = 140; t <= 420; t += 35) {
      frame = stepExpressionEngine(state, t, { ...options, mirrorMode: false });
    }
    expect(Math.abs(frame.gazeX)).toBeLessThan(0.15);
    expect(Math.abs(frame.blinkR - frame.blinkL)).toBeGreaterThan(0.4);

    const asymObs: UserFaceObservation = {
      ...face,
      blinkLeft: 0.95,
      blinkRight: 0.05,
      smileLeft: 0.85,
      smileRight: 0.1,
      mouthSkewX: -0.4,
      updatedAt: 500,
    };
    const mState = createExpressionEngine(480);
    let mFrame = stepExpressionEngine(mState, 500, { ...options, userFace: asymObs });
    for (let t = 530; t <= 800; t += 35) {
      mFrame = stepExpressionEngine(mState, t, {
        ...options,
        userFace: { ...asymObs, updatedAt: t },
      });
    }
    expect(mFrame.blinkL).toBeGreaterThan(0.9);
    expect(mFrame.blinkR).toBeLessThan(0.1);
    expect(mFrame.smileL).toBeGreaterThan(mFrame.smileR + 0.3);
  });

  it("maintains eye contact while listening and turns head to the side while thinking", () => {
    const state = createExpressionEngine(0);
    // Rare wink: nextWinkAt should start at least 45 seconds in the future
    expect(state.nextWinkAt).toBeGreaterThanOrEqual(45000);

    // Side thinking pose turns head + looks toward the chosen side
    state.poseMode = "thinkSide";
    state.poseDir = 1;
    state.poseUntil = 3000;
    let thinkFrame = stepExpressionEngine(state, 500, {
      ...options,
      mirrorMode: false,
      listening: false,
    });
    for (let t = 550; t <= 1400; t += 50) {
      thinkFrame = stepExpressionEngine(state, t, {
        ...options,
        mirrorMode: false,
        listening: false,
      });
    }
    expect(thinkFrame.headYaw).toBeGreaterThan(0.2);
    expect(thinkFrame.gazeX).toBeGreaterThan(0.3);

    // Switching to listening snaps gaze straight back onto the user's eyes
    let listenFrame = stepExpressionEngine(state, 1450, {
      ...options,
      mirrorMode: false,
      listening: true,
    });
    for (let t = 1500; t <= 2000; t += 50) {
      listenFrame = stepExpressionEngine(state, t, {
        ...options,
        mirrorMode: false,
        listening: true,
      });
    }
    expect(Math.abs(listenFrame.gazeX)).toBeLessThan(0.12);
    expect(Math.abs(listenFrame.gazeY)).toBeLessThan(0.1);
  });

  it("engages forward camera lean, brow pinch, and nose scrunch on intense emotions", () => {
    const state = createExpressionEngine(0);
    let angryFrame = stepExpressionEngine(state, 50, {
      ...options,
      mirrorMode: false,
      emotion: "angry",
    });
    for (let t = 100; t <= 900; t += 50) {
      angryFrame = stepExpressionEngine(state, t, {
        ...options,
        mirrorMode: false,
        emotion: "angry",
      });
    }
    expect(angryFrame.innerBrowPinch).toBeGreaterThan(0.3);
    expect(angryFrame.noseFlare).toBeGreaterThan(0.25);
    expect(angryFrame.forwardLean).toBeGreaterThan(0.1);

    let listenFrame = angryFrame;
    for (let t = 950; t <= 1700; t += 50) {
      listenFrame = stepExpressionEngine(state, t, {
        ...options,
        mirrorMode: false,
        listening: true,
        emotion: "neutral",
      });
    }
    expect(listenFrame.forwardLean).toBeGreaterThan(0.1);
  });

  it("assigns distinct natural male and female voices without cross-gender fallback", () => {
    expect(defaultGeminiVoiceForGender("male")).toBe("Charon");
    expect(defaultGeminiVoiceForGender("female")).toBe("Kore");
    expect(defaultVoiceForGender("male", { engine: "gemini", id: "Kore" })).toEqual({
      engine: "gemini",
      id: "Charon",
    });
    expect(defaultVoiceForGender("female", { engine: "gemini", id: "Charon" })).toEqual({
      engine: "gemini",
      id: "Kore",
    });
    expect(defaultVoiceForGender("male", { engine: "kokoro", id: "af_heart" })).toEqual({
      engine: "kokoro",
      id: "am_michael",
    });
    expect(defaultVoiceForGender("female", { engine: "gemini", id: "Gacrux" })).toEqual({
      engine: "gemini",
      id: "Gacrux",
    });
  });

  it("extracts inline mid-sentence emotion tags cleanly and protects tragic context from laughing", () => {
    const inline = extractEmotion(
      "[triste] Siento mucho lo que pasó, [llorando] estoy aquí contigo.",
    );
    expect(inline.emotion).toBe("sad");
    expect(inline.text).toBe("Siento mucho lo que pasó, estoy aquí contigo.");
    expect(inline.timeline).toHaveLength(2);
    expect(inline.timeline?.[1]?.emotion).toBe("crying");

    const grief = extractEmotion(
      "[feliz] Qué triste noticia, siento muchísimo que murió tu familiar.",
    );
    expect(["sad", "crying"]).toContain(grief.emotion);

    const griefTimeline = deriveSpeechEmotionTimeline(
      "Lo siento muchísimo, estoy aquí acompañándote en este momento tan duro.",
      "sad",
    );
    expect(griefTimeline.some((b) => b.emotion === "laugh" || b.emotion === "happy")).toBe(false);
  });

  it("shapes realistic angry brows, crying teardrops, nervous sweat, sleepy droop, and locked hair", () => {
    const runEmotion = (emotion: Parameters<typeof stepExpressionEngine>[2]["emotion"]) => {
      const state = createExpressionEngine(0);
      let frame = stepExpressionEngine(state, 50, { ...options, mirrorMode: false, emotion });
      for (let t = 100; t <= 1200; t += 50) {
        frame = stepExpressionEngine(state, t, { ...options, mirrorMode: false, emotion });
      }
      return frame;
    };

    const angry = runEmotion("angry");
    expect(angry.innerBrowPinch).toBeGreaterThan(0.5);
    expect(angry.innerBrowY).toBeLessThan(0);
    expect(angry.outerBrowY).toBeGreaterThan(0.05);
    expect(angry.hairShiftX).toBe(0);
    expect(angry.hairShiftY).toBe(0);

    const crying = runEmotion("crying");
    expect(crying.tearAmount).toBeGreaterThan(0.5);
    expect(crying.eyeWater).toBeGreaterThan(0.5);
    expect(crying.eyeRedness).toBeGreaterThan(0.2);

    const sweating = runEmotion("sweating");
    expect(sweating.sweatAmount).toBeGreaterThan(0.6);

    const sleepy = runEmotion("sleepy");
    expect(sleepy.lidDroop).toBeGreaterThan(0.4);

    // Rapid pale chill on fear and sick, warm blush on blushing
    const fear = runEmotion("fear");
    expect(fear.blushAmount).toBeLessThan(-0.4);
    const blushing = runEmotion("blushing");
    expect(blushing.blushAmount).toBeGreaterThan(0.8);
  });

  it("converts Spanish and English words into accurate lip-reading viseme phonemes and aligns to speech window", () => {
    // Spanish "que" should skip silent u and produce 2 phoneme cues (/k/, /e/) rather than rounding into /u/
    const que = textToVisemeProtos("que");
    expect(que.protos).toHaveLength(2);
    expect(que.protos.every((p) => p.pucker < 0.2)).toBe(true);

    // Spanish silent h in "hola" should be skipped -> starts on /o/
    const hola = textToVisemeProtos("hola");
    expect(hola.protos[0]?.round).toBeGreaterThan(0.7);

    // Bilabial seal on "mama", labiodental on "fuego", interdental tongue on "think"
    const mama = textToVisemeProtos("mamá");
    expect(mama.protos[0]?.press).toBeGreaterThan(0.9);
    expect(mama.protos[0]?.open).toBe(0);
    expect(mama.protos[3]?.stress).toBeGreaterThan(0.8);

    const fuego = textToVisemeProtos("fuego");
    expect(fuego.protos[0]?.labiodental).toBeGreaterThan(0.8);

    const think = textToVisemeProtos("think");
    expect(think.protos[0]?.tongue).toBeGreaterThan(0.6);

    // Speech window trims leading silence cleanly
    const tl = buildVisemeTimeline("hola", 2.0, 1, { startSec: 0.4, endSec: 1.6 });
    const atSilence = sampleVisemeTimeline(tl, 0.1);
    const atSpeech = sampleVisemeTimeline(tl, 0.48);
    expect(atSilence?.open).toBe(0);
    expect(atSpeech?.round).toBeGreaterThan(0.5);
  });

  it("mirrors asymmetric single-eyebrow raises independently while staying symmetric on equal raises", () => {
    const stepMirror = (obs: Partial<UserFaceObservation>) => {
      const state = createExpressionEngine(0);
      let frame = stepExpressionEngine(state, 50, {
        ...options,
        userFace: { ...face, ...obs, mouthOpen: 0, headYaw: 0, gazeX: 0, updatedAt: 50 },
      });
      for (let t = 100; t <= 900; t += 50) {
        frame = stepExpressionEngine(state, t, {
          ...options,
          userFace: { ...face, ...obs, mouthOpen: 0, headYaw: 0, gazeX: 0, updatedAt: t },
        });
      }
      return frame;
    };

    const asymLeft = stepMirror({ browRaiseLeft: 0.9, browRaiseRight: 0.05, browRaise: 0.48 });
    expect(Math.abs(asymLeft.browAsym)).toBeGreaterThan(0.25);

    const symmetric = stepMirror({ browRaiseLeft: 0.7, browRaiseRight: 0.7, browRaise: 0.7 });
    expect(Math.abs(symmetric.browAsym)).toBeLessThan(0.08);
    expect(symmetric.outerBrowY).toBeGreaterThan(0.15);
  });

  it("drives new expressions (smirk, sarcastic, eyeroll, intense, bitingLip, compassionate) and decoupled neck/torso yaw", () => {
    const runEmotion = (emotion: Parameters<typeof stepExpressionEngine>[2]["emotion"]) => {
      const state = createExpressionEngine(0);
      let frame = stepExpressionEngine(state, 50, { ...options, mirrorMode: false, emotion });
      for (let t = 100; t <= 1200; t += 50) {
        frame = stepExpressionEngine(state, t, { ...options, mirrorMode: false, emotion });
      }
      return frame;
    };

    const smirk = runEmotion("smirk");
    expect(Math.abs(smirk.mouthSkewX)).toBeGreaterThan(0.15);
    expect(Math.abs(smirk.smileR - smirk.smileL)).toBeGreaterThan(0.15);

    const sarcastic = runEmotion("sarcastic");
    expect(Math.abs(sarcastic.browAsym)).toBeGreaterThan(0.2);
    expect(sarcastic.eyeRoll).toBeGreaterThan(0.15);
    expect(Math.abs(sarcastic.mouthSkewX)).toBeGreaterThan(0.1);

    const eyeroll = runEmotion("eyeroll");
    expect(eyeroll.gazeY).toBeGreaterThan(0.4);
    expect(eyeroll.eyeRoll).toBeGreaterThan(0.5);

    const intense = runEmotion("intense");
    expect(intense.headPitch).toBeLessThan(-0.15);
    expect(intense.gazeY).toBeGreaterThan(0.2);
    expect(intense.pupilDilate).toBeGreaterThan(0.3);

    const bitingLip = runEmotion("bitingLip");
    expect(bitingLip.lipBite).toBeGreaterThan(0.5);

    const compassionate = runEmotion("compassionate");
    expect(compassionate.innerBrowY).toBeGreaterThan(0.15);
    expect(compassionate.smile).toBeGreaterThan(0.15);

    // Decoupled head/neck/torso yaw when mirroring a clear head turn
    const turnState = createExpressionEngine(0);
    let turnFrame = stepExpressionEngine(turnState, 50, {
      ...options,
      userFace: { ...face, headYaw: 0.75, mouthOpen: 0, gazeX: 0, updatedAt: 50 },
    });
    for (let t = 100; t <= 1000; t += 50) {
      turnFrame = stepExpressionEngine(turnState, t, {
        ...options,
        userFace: { ...face, headYaw: 0.75, mouthOpen: 0, gazeX: 0, updatedAt: t },
      });
    }
    expect(turnFrame.headYaw).toBeGreaterThan(0.45);
    expect(turnFrame.neckYaw).toBeGreaterThan(0.1);
    expect(turnFrame.torsoYaw).toBeGreaterThan(0.05);
    expect(turnFrame.headYaw).toBeGreaterThan(turnFrame.neckYaw);
    expect(turnFrame.neckYaw).toBeGreaterThan(turnFrame.torsoYaw);
  });

  it("never opens mouth from observed user smile alone when idle in neutral", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const state = createExpressionEngine(0);
    let frame = stepExpressionEngine(state, 50, {
      ...options,
      mirrorMode: false,
      emotion: "neutral",
      userFace: { ...face, smile: 0.9, mouthOpen: 0, updatedAt: 50 },
    });
    for (let t = 100; t <= 900; t += 50) {
      frame = stepExpressionEngine(state, t, {
        ...options,
        mirrorMode: false,
        emotion: "neutral",
        userFace: { ...face, smile: 0.9, mouthOpen: 0, updatedAt: t },
      });
    }
    expect(frame.open).toBe(0);
    vi.restoreAllMocks();
  });

  it("isolates subtle head turns from neck, swallows occasionally after speech, and mirrors tongueOut", () => {
    // 1) Small head turn keeps cervical neckYaw minimal compared to headYaw
    const isoState = createExpressionEngine(0);
    let isoFrame = stepExpressionEngine(isoState, 50, {
      ...options,
      userFace: { ...face, headYaw: 0.15, mouthOpen: 0, gazeX: 0, updatedAt: 50 },
    });
    for (let t = 100; t <= 900; t += 50) {
      isoFrame = stepExpressionEngine(isoState, t, {
        ...options,
        userFace: { ...face, headYaw: 0.15, mouthOpen: 0, gazeX: 0, updatedAt: t },
      });
    }
    expect(isoFrame.headYaw).toBeGreaterThan(0.11);
    expect(isoFrame.neckYaw).toBeLessThan(isoFrame.headYaw * 0.35);

    // 2) Automatic swallow waits 2-3 speech turns (never every single turn) and is stronger on complex topics
    vi.spyOn(Math, "random").mockReturnValue(0.2);
    const swState = createExpressionEngine(0);
    swState.swallowTargetTurns = 2;
    // Turn 1: should NOT trigger swallow yet
    stepExpressionEngine(swState, 10000, {
      ...options,
      mirrorMode: false,
      active: true,
      visemes: { level: 0.5, open: 0.4, wide: 0.1, round: 0.1 },
    });
    stepExpressionEngine(swState, 10100, { ...options, mirrorMode: false, active: false });
    let turn1Max = 0;
    for (let t = 10160; t <= 12200; t += 60) {
      turn1Max = Math.max(
        turn1Max,
        stepExpressionEngine(swState, t, { ...options, mirrorMode: false, active: false })
          .swallowAmount,
      );
    }
    expect(turn1Max).toBeLessThan(0.05);

    // Turn 2: triggers natural swallow (~520-1500ms after response completes)
    stepExpressionEngine(swState, 12300, {
      ...options,
      mirrorMode: false,
      active: true,
      visemes: { level: 0.5, open: 0.4, wide: 0.1, round: 0.1 },
    });
    stepExpressionEngine(swState, 12400, {
      ...options,
      mirrorMode: false,
      active: false,
      topicComplexity: 0.85,
    });
    let maxSwallow = 0;
    let maxSwallowPhase = 0;
    for (let t = 12460; t <= 14600; t += 60) {
      const f = stepExpressionEngine(swState, t, {
        ...options,
        mirrorMode: false,
        active: false,
        topicComplexity: 0.85,
      });
      maxSwallow = Math.max(maxSwallow, f.swallowAmount);
      maxSwallowPhase = Math.max(maxSwallowPhase, f.swallowPhase);
    }
    expect(maxSwallow).toBeGreaterThan(0.95);
    expect(maxSwallowPhase).toBeGreaterThan(0.5);

    // Manual swallowSignal in Mirror mode executes immediately
    const mirrorSwState = createExpressionEngine(0);
    let manualMax = 0;
    for (let t = 50; t <= 900; t += 60) {
      const f = stepExpressionEngine(mirrorSwState, t, {
        ...options,
        mirrorMode: true,
        swallowSignal: { seq: 1, strength: 1.4 },
      });
      manualMax = Math.max(manualMax, f.swallowAmount);
    }
    expect(manualMax).toBeGreaterThan(1.0);
    vi.restoreAllMocks();

    // 3) Mirror tongueOut drives tongue articulation and modest lip parting without full jaw drop
    const tState = createExpressionEngine(0);
    let tFrame = stepExpressionEngine(tState, 50, {
      ...options,
      userFace: { ...face, mouthOpen: 0, tongueOut: 0.85, headYaw: 0, gazeX: 0, updatedAt: 50 },
    });
    for (let t = 100; t <= 800; t += 50) {
      tFrame = stepExpressionEngine(tState, t, {
        ...options,
        userFace: { ...face, mouthOpen: 0, tongueOut: 0.85, headYaw: 0, gazeX: 0, updatedAt: t },
      });
    }
    expect(tFrame.tongueOut).toBeGreaterThan(0.5);
    expect(tFrame.open).toBeGreaterThan(0.18);
    expect(tFrame.open).toBeLessThan(0.45);
  });

  it("drives disgusted nose flare/upper-lip lift and mirrors lateral smirks and wide jaw opening", () => {
    const dState = createExpressionEngine(0);
    let dFrame = stepExpressionEngine(dState, 50, {
      ...options,
      mirrorMode: false,
      emotion: "disgusted",
    });
    for (let t = 100; t <= 1000; t += 50) {
      dFrame = stepExpressionEngine(dState, t, {
        ...options,
        mirrorMode: false,
        emotion: "disgusted",
      });
    }
    expect(dFrame.noseFlare).toBeGreaterThan(0.7);
    expect(dFrame.upperLipLift).toBeGreaterThan(0.5);

    const mState = createExpressionEngine(0);
    let mFrame = stepExpressionEngine(mState, 50, {
      ...options,
      userFace: {
        ...face,
        mouthOpen: 0.9,
        mouthSkewX: 0.65,
        headYaw: 0,
        gazeX: 0,
        updatedAt: 50,
      },
    });
    for (let t = 100; t <= 800; t += 50) {
      mFrame = stepExpressionEngine(mState, t, {
        ...options,
        userFace: {
          ...face,
          mouthOpen: 0.9,
          mouthSkewX: 0.65,
          headYaw: 0,
          gazeX: 0,
          updatedAt: t,
        },
      });
    }
    expect(Math.abs(mFrame.mouthSkewX)).toBeGreaterThan(0.25);
    expect(mFrame.open).toBeGreaterThan(0.75);
  });

  it("opens wider on stressed emphatic speech peaks than neutral syllables and triggers brief pause swallows", () => {
    const baseState = createExpressionEngine(0);
    let modOpen = 0;
    for (let t = 50; t <= 450; t += 40) {
      modOpen = stepExpressionEngine(baseState, t, {
        ...options,
        mirrorMode: false,
        active: true,
        visemes: { level: 0.45, open: 0.5, wide: 0.1, round: 0.1, stress: 0.1 },
      }).open;
    }

    const emphState = createExpressionEngine(0);
    let emphOpen = 0;
    for (let t = 50; t <= 450; t += 40) {
      emphOpen = stepExpressionEngine(emphState, t, {
        ...options,
        mirrorMode: false,
        active: true,
        visemes: { level: 0.85, open: 0.88, wide: 0.1, round: 0.1, stress: 0.72 },
      }).open;
    }
    expect(modOpen).toBeLessThan(0.55);
    expect(emphOpen).toBeGreaterThan(modOpen + 0.25);

    vi.spyOn(Math, "random").mockReturnValue(0.2);
    const pauseSwallow = createExpressionEngine(0);
    pauseSwallow.nextSwallowEligibleAt = 5000;
    pauseSwallow.open = 0.02;
    let maxPauseSwallow = 0;
    for (let t = 5100; t <= 6200; t += 60) {
      const f = stepExpressionEngine(pauseSwallow, t, {
        ...options,
        mirrorMode: false,
        active: true,
        topicComplexity: 0.75,
        visemes: { level: 0.04, open: 0.02, wide: 0, round: 0 },
      });
      maxPauseSwallow = Math.max(maxPauseSwallow, f.swallowAmount);
    }
    expect(maxPauseSwallow).toBeGreaterThan(0.6);
    vi.restoreAllMocks();
  });

  it("mirrors camera proximity into forwardLean and achieves selective wide mouth opening cleanly", () => {
    const proxState = createExpressionEngine(0);
    let closeFrame = stepExpressionEngine(proxState, 50, {
      ...options,
      userFace: {
        ...face,
        faceProximity: 0.85,
        mouthOpen: 1.0,
        headPitch: 0,
        headYaw: 0,
        gazeX: 0,
        updatedAt: 50,
      },
    });
    for (let t = 100; t <= 1000; t += 50) {
      closeFrame = stepExpressionEngine(proxState, t, {
        ...options,
        userFace: {
          ...face,
          faceProximity: 0.85,
          mouthOpen: 1.0,
          headPitch: 0,
          headYaw: 0,
          gazeX: 0,
          updatedAt: t,
        },
      });
    }
    expect(closeFrame.forwardLean).toBeGreaterThan(0.5);
    expect(closeFrame.open).toBeGreaterThan(0.92);

    const laughState = createExpressionEngine(0);
    let maxLaughOpen = 0;
    for (let t = 50; t <= 1200; t += 50) {
      const lf = stepExpressionEngine(laughState, t, {
        ...options,
        mirrorMode: false,
        emotion: "laugh",
      });
      maxLaughOpen = Math.max(maxLaughOpen, lf.open);
    }
    expect(maxLaughOpen).toBeGreaterThan(0.5);

    // mouthMotion preserves restrained natural mid speech while reaching full 1.0 on peaks
    const modViseme = mouthMotion(0.5, false, 0.02);
    const peakViseme = mouthMotion(1.0, false, 0.02);
    expect(modViseme.opening).toBeLessThan(0.5);
    expect(peakViseme.opening).toBeCloseTo(1.0, 2);
  });

  it("compensates eye gaze opposite to head turn for screen-locked eye contact and drives thinking vs userSpeaking gaze", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    // 1) Mirror mode: turning head right while looking near screen center counter-steers pupils left (screen-locked eye contact)
    const vorState = createExpressionEngine(0);
    let vorFrame = stepExpressionEngine(vorState, 50, {
      ...options,
      userFace: {
        ...face,
        headYaw: 0.48,
        headPitch: 0,
        gazeX: 0,
        gazeY: 0,
        mouthOpen: 0,
        updatedAt: 50,
      },
    });
    for (let t = 100; t <= 900; t += 50) {
      vorFrame = stepExpressionEngine(vorState, t, {
        ...options,
        userFace: {
          ...face,
          headYaw: 0.48,
          headPitch: 0,
          gazeX: 0,
          gazeY: 0,
          mouthOpen: 0,
          updatedAt: t,
        },
      });
    }
    expect(vorFrame.headYaw).toBeGreaterThan(0.3);
    expect(vorFrame.gazeX).toBeLessThan(-0.14);

    // 2) Conversational states: userSpeaking locks gaze straight on user; thinking glances aside briefly
    const convState = createExpressionEngine(0);
    convState.targetGazeX = 0.5;
    convState.gazeX = 0.5;
    let listenUserFrame = stepExpressionEngine(convState, 50, {
      ...options,
      mirrorMode: false,
      listening: true,
      userSpeaking: true,
    });
    for (let t = 100; t <= 600; t += 50) {
      listenUserFrame = stepExpressionEngine(convState, t, {
        ...options,
        mirrorMode: false,
        listening: true,
        userSpeaking: true,
      });
    }
    expect(Math.abs(listenUserFrame.gazeX)).toBeLessThan(0.15);

    let thinkAfterFrame = stepExpressionEngine(convState, 650, {
      ...options,
      mirrorMode: false,
      listening: false,
      thinking: true,
    });
    for (let t = 700; t <= 1250; t += 50) {
      thinkAfterFrame = stepExpressionEngine(convState, t, {
        ...options,
        mirrorMode: false,
        listening: false,
        thinking: true,
      });
    }
    expect(Math.abs(thinkAfterFrame.gazeX)).toBeGreaterThan(0.25);
    expect(thinkAfterFrame.gazeY).toBeGreaterThan(0.05);
    vi.restoreAllMocks();
  });

  it("mirrors lip pucker contraction and distinguishes closed-lip smiles from teeth-parted smiles", () => {
    const stepObs = (obs: Partial<UserFaceObservation>) => {
      const s = createExpressionEngine(0);
      let f = stepExpressionEngine(s, 50, {
        ...options,
        userFace: { ...face, headYaw: 0, gazeX: 0, updatedAt: 50, ...obs },
      });
      for (let t = 100; t <= 850; t += 50) {
        f = stepExpressionEngine(s, t, {
          ...options,
          userFace: { ...face, headYaw: 0, gazeX: 0, updatedAt: t, ...obs },
        });
      }
      return f;
    };

    const kiss = stepObs({ mouthPucker: 0.88, mouthOpen: 0, smile: 0 });
    expect(kiss.pucker).toBeGreaterThan(0.65);
    expect(kiss.smileTeeth).toBeLessThan(0.2);

    const closedSmile = stepObs({ smile: 0.85, mouthOpen: 0, mouthPress: 0.1, mouthPucker: 0 });
    expect(closedSmile.smile).toBeGreaterThan(0.6);
    expect(closedSmile.smileTeeth).toBeLessThan(0.2);

    const teethSmile = stepObs({ smile: 0.85, mouthOpen: 0.12, mouthPress: 0, mouthPucker: 0 });
    expect(teethSmile.smile).toBeGreaterThan(0.6);
    expect(teethSmile.smileTeeth).toBeGreaterThan(0.75);
  });

  it("mirrors wide-eyed openness directly and echoes remembered eyeWide/pucker softly while speaking", () => {
    const mState = createExpressionEngine(0);
    let mFrame = stepExpressionEngine(mState, 50, {
      ...options,
      userFace: { ...face, eyeWide: 0.85, mouthOpen: 0, headYaw: 0, gazeX: 0, updatedAt: 50 },
    });
    for (let t = 100; t <= 850; t += 50) {
      mFrame = stepExpressionEngine(mState, t, {
        ...options,
        userFace: { ...face, eyeWide: 0.85, mouthOpen: 0, headYaw: 0, gazeX: 0, updatedAt: t },
      });
    }
    expect(mFrame.eyeWide).toBeGreaterThan(0.75);

    // Non-mirror conversational mode: sustain eyeWide + pucker, pause to neutral, then speak
    const cState = createExpressionEngine(0);
    for (let t = 50; t <= 850; t += 100) {
      stepExpressionEngine(cState, t, {
        ...options,
        mirrorMode: false,
        listening: true,
        userFace: {
          ...face,
          eyeWide: 0.85,
          mouthPucker: 0.82,
          mouthOpen: 0,
          headYaw: 0,
          gazeX: 0,
          updatedAt: t,
        },
      });
    }
    for (let t = 950; t <= 4200; t += 100) {
      stepExpressionEngine(cState, t, {
        ...options,
        mirrorMode: false,
        listening: false,
      });
    }
    let talkFrame = cState as unknown as ReturnType<typeof stepExpressionEngine>;
    for (let t = 4250; t <= 5150; t += 50) {
      talkFrame = stepExpressionEngine(cState, t, {
        ...options,
        mirrorMode: false,
        active: true,
        visemes: { level: 0.4, open: 0.35, wide: 0.05, round: 0.05 },
      });
    }
    expect(talkFrame.eyeWide).toBeGreaterThan(0.08);
  });

  it("varies speech tempo faster for excited/urgent speech and slower for reflective speech", () => {
    const fast = dynamicSpeechTempo("¡Corre, mira rápido! ¡Qué increíble!!", "euphoric");
    const slow = dynamicSpeechTempo("Lo siento mucho… respira despacio, todo irá bien…", "sad");
    expect(fast.tempo).toBe("fast");
    expect(fast.rate).toBeGreaterThan(1.12);
    expect(slow.tempo).toBe("slow");
    expect(slow.rate).toBeLessThan(0.89);
    expect(fast.rate - slow.rate).toBeGreaterThan(0.28);
  });

  it("converges fast in directMirror (~150ms) and drives lively active speech brow/head motion", () => {
    const fastState = createExpressionEngine(0);
    let mFrame = stepExpressionEngine(fastState, 50, {
      ...options,
      userFace: { ...face, headYaw: 0.45, mouthOpen: 0.85, gazeX: 0, updatedAt: 50 },
    });
    for (let t = 83; t <= 160; t += 33) {
      mFrame = stepExpressionEngine(fastState, t, {
        ...options,
        userFace: { ...face, headYaw: 0.45, mouthOpen: 0.85, gazeX: 0, updatedAt: t },
      });
    }
    expect(mFrame.headYaw).toBeGreaterThan(0.38);
    expect(mFrame.open).toBeGreaterThan(0.65);

    const talkState = createExpressionEngine(0);
    let maxBrow = 0;
    for (let t = 50; t <= 900; t += 50) {
      const f = stepExpressionEngine(talkState, t, {
        ...options,
        mirrorMode: false,
        active: true,
        visemes: { level: 0.72, open: 0.65, wide: 0.25, round: 0.1, stress: 0.8 },
      });
      maxBrow = Math.max(maxBrow, f.innerBrowY);
    }
    expect(maxBrow).toBeGreaterThan(0.18);
  });
});
