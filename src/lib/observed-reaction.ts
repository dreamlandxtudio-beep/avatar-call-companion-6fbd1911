import type { UserFaceObservation } from "./userCamera";

export const REACTION_HOLD_MS = 850;
export const OBSERVATION_MAX_AGE_MS = 350;
export type ObservedReaction = { since: number | null; lastSample: number; pendingAt: number; until: number; cooldownUntil: number; strength: number };
export function createObservedReaction(): ObservedReaction {
  return { since: null, lastSample: -1, pendingAt: 0, until: 0, cooldownUntil: 0, strength: 0 };
}
/** Only sustained, freshly seen smiling/squinting can trigger an occasional response. */
export function stepObservedReaction(s: ObservedReaction, now: number, face?: UserFaceObservation) {
  const fresh = !!face?.detected && now >= face.updatedAt && now - face.updatedAt <= OBSERVATION_MAX_AGE_MS;
  if (!fresh) {
    s.since = null; s.pendingAt = 0; s.until = 0;
    return 0;
  }
  if (face.updatedAt !== s.lastSample) {
    if (face.updatedAt - s.lastSample > OBSERVATION_MAX_AGE_MS) s.since = null;
    s.lastSample = face.updatedAt;
    const smiling = face.smile >= 0.5 && ((face.eyeSquint ?? 0) >= 0.12 || face.mouthOpen >= 0.15);
    if (!smiling) s.since = null;
    else if (s.since === null) s.since = face.updatedAt;
    if (smiling && s.since !== null && face.updatedAt - s.since >= REACTION_HOLD_MS && now >= s.cooldownUntil && !s.pendingAt) {
      s.pendingAt = now + 280; s.strength = Math.min(0.6, face.smile * 0.65); s.cooldownUntil = now + 8000;
    }
  }
  if (s.pendingAt && now >= s.pendingAt) { s.pendingAt = 0; s.until = now + 1700; }
  return now < s.until ? s.strength * Math.min(1, (s.until - now) / 450) : 0;
}