import { describe, expect, it, vi } from "vitest";
import { expressionFromSignals } from "@/lib/face-signals";
import { createExpressionEngine, stepExpressionEngine } from "@/lib/expressionEngine";
import { voiceProsody } from "@/lib/voice-prosody";
import { defaultVoiceForGender } from "@/lib/store";
import { defaultGeminiVoiceForGender } from "@/lib/voice";
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
  it("detects both wide laughter and smaller chuckling signals", () => {
    expect(expressionFromSignals(0.56, 0.2, 0.05, 0, 0, 0).emotion).toBe("laugh");
    expect(expressionFromSignals(0.4, 0.28, 0, 0, 0, 0).emotion).toBe("laugh");
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
});
