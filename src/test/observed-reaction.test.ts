import { describe, expect, it } from "vitest";
import {
  createObservedEmpathy,
  createObservedReaction,
  stepObservedEmpathy,
  stepObservedReaction,
} from "@/lib/observed-reaction";
import type { UserFaceObservation } from "@/lib/userCamera";
const observation = (
  at: number,
  smile = 0.9,
  extra: Partial<UserFaceObservation> = {},
): UserFaceObservation => ({
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
  ...extra,
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

describe("Subconscious camera empathy carryover", () => {
  it("does not echo a stale frame or brief glitch", () => {
    const state = createObservedEmpathy(0);
    stepObservedEmpathy(
      state,
      100,
      observation(100, 0, { browFurrow: 0.85, noseSneer: 0.6, emotion: "angry" }),
    );
    const stale = stepObservedEmpathy(
      state,
      1200,
      observation(100, 0, { browFurrow: 0.85, noseSneer: 0.6, emotion: "angry" }),
    );
    expect(stale.browFurrow).toBeLessThan(0.01);
    expect(stale.noseScrunch).toBeLessThan(0.01);
  });

  it("subconsciously carries over sustained fresh brows, nose scrunch, and head lean after a delay", () => {
    const state = createObservedEmpathy(0);
    const makeAngry = (t: number) =>
      observation(t, 0, {
        emotion: "angry",
        eyeSquint: 0.2,
        mouthOpen: 0.1,
        browFurrow: 0.85,
        noseSneer: 0.72,
        headYaw: 0.35,
      });

    // Build 600ms of sustained fresh expression: should still stay quiet at t=600
    let frame = state.current;
    for (let t = 0; t <= 600; t += 100) {
      frame = stepObservedEmpathy(state, t, makeAngry(t));
    }
    expect(frame.browFurrow).toBe(0);

    // After the ~360ms subconscious latency, carryover ramps in gently and lingers into later ticks
    for (let t = 650; t <= 1450; t += 50) {
      frame = stepObservedEmpathy(state, t, makeAngry(t));
    }
    expect(frame.browFurrow).toBeGreaterThan(0.15);
    expect(frame.noseScrunch).toBeGreaterThan(0.1);
    expect(frame.headYaw).toBeGreaterThan(0.03);
  });
});
