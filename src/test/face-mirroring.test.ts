import { describe, expect, it, vi } from "vitest";
import { expressionFromSignals } from "@/lib/face-signals";
import { createExpressionEngine, stepExpressionEngine } from "@/lib/expressionEngine";
import { voiceProsody } from "@/lib/voice-prosody";
import type { UserFaceObservation } from "@/lib/userCamera";

const face: UserFaceObservation = { detected: true, emotion: "neutral", smile: 0, browFurrow: 0, browRaise: 0, mouthOpen: 0.8, headYaw: 0.4, headPitch: 0, gazeX: -0.8, gazeY: 0, blink: 1, labelEs: "", contextPrompt: "", updatedAt: 100 };
const options = { active: false, listening: false, emotion: "neutral" as const, visemes: { level: 0, open: 0, wide: 0, round: 0 }, hasVisibleTeeth: false, restMouthRatio: 0, mirrorMode: true };
describe("Observable expression mirroring", () => {
  it("maps lowered mouth corners with raised inner brows to a sad-style pose before a frown", () => {
    expect(expressionFromSignals(0, 0, 0.4, 0.4, 0.4, 0).emotion).toBe("sad");
    expect(expressionFromSignals(0, 0, 0.4, 0, 0, 0).label).toBe("Ceño fruncido");
  });
  it("moves eyes opposite to head direction independently and mirrors open mouth and eyelids", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);
    const state = createExpressionEngine(0);
    let frame = stepExpressionEngine(state, 100, { ...options, userFace: face });
    for (let i = 1; i < 10; i++) frame = stepExpressionEngine(state, 100 + i * 30, { ...options, userFace: face });
    expect(frame.headYaw).toBeGreaterThan(0); expect(frame.gazeX).toBeLessThan(-0.5);
    expect(frame.open).toBeGreaterThan(0.4); expect(frame.blink).toBe(1);
    vi.restoreAllMocks();
  });
  it("does not mirror an old camera observation", () => {
    const frame = stepExpressionEngine(createExpressionEngine(2000), 2100, { ...options, userFace: face });
    expect(frame.open).toBe(0);
  });
  it("uses slower sad prosody and brighter happy prosody without fabricated voice cracks", () => {
    expect(voiceProsody("sad")).toEqual({ rate: 0.9, pitch: 0.96 });
    expect(voiceProsody("happy")).toEqual({ rate: 1.04, pitch: 1.06 });
  });
});