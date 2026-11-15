import { describe, expect, it } from "vitest";
import { createObservedReaction, stepObservedReaction } from "@/lib/observed-reaction";
import type { UserFaceObservation } from "@/lib/userCamera";
const observation = (at: number, smile = 0.9): UserFaceObservation => ({
  detected: true,
  updatedAt: at,
  smile,
  eyeSquint: 0.4,
  mouthOpen: 0.4,
  emotion: "happy",
  headYaw: 0,
  headPitch: 0,
  gazeX: 0,
  gazeY: 0,
  browRaise: 0,
  browFurrow: 0,
  labelEs: "",
  contextPrompt: "",
});
describe("Delayed observed responses", () => {
  it("waits nearly a second and then reacts after looking first", () => {
    const state = createObservedReaction();
    for (let at = 0; at <= 900; at += 100)
      expect(stepObservedReaction(state, at, observation(at))).toBe(0);
    expect(stepObservedReaction(state, 1100, observation(1100))).toBe(0);
    expect(stepObservedReaction(state, 1200, observation(1200))).toBeGreaterThan(0);
  });
  it("never reacts to a single old frame or a lost camera", () => {
    const state = createObservedReaction();
    stepObservedReaction(state, 0, observation(0));
    expect(stepObservedReaction(state, 1000, observation(0))).toBe(0);
    expect(stepObservedReaction(state, 1200)).toBe(0);
  });
  it("does not copy merely opening the mouth", () => {
    const state = createObservedReaction();
    for (let at = 0; at <= 2000; at += 100)
      expect(stepObservedReaction(state, at, observation(at, 0))).toBe(0);
  });
  it("does not retrigger continuously during an eight-second cooldown", () => {
    const state = createObservedReaction();
    for (let at = 0; at <= 4000; at += 100) stepObservedReaction(state, at, observation(at));
    expect(stepObservedReaction(state, 4100, observation(4100))).toBe(0);
    expect(state.cooldownUntil).toBe(8900);
  });
});
