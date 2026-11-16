import type { UserFaceObservation } from "./userCamera";

export const REACTION_HOLD_MS = 850;
export const OBSERVATION_MAX_AGE_MS = 350;
export type ObservedReaction = {
  since: number | null;
  lastSample: number;
  pendingAt: number;
  until: number;
  cooldownUntil: number;
  strength: number;
};
export function createObservedReaction(): ObservedReaction {
  return { since: null, lastSample: -1, pendingAt: 0, until: 0, cooldownUntil: 0, strength: 0 };
}
/** Only sustained, freshly seen smiling/squinting can trigger an occasional response. */
export function stepObservedReaction(s: ObservedReaction, now: number, face?: UserFaceObservation) {
  const fresh =
    !!face?.detected && now >= face.updatedAt && now - face.updatedAt <= OBSERVATION_MAX_AGE_MS;
  if (!fresh) {
    s.since = null;
    s.pendingAt = 0;
    s.until = 0;
    return 0;
  }
  if (face.updatedAt !== s.lastSample) {
    if (face.updatedAt - s.lastSample > OBSERVATION_MAX_AGE_MS) s.since = null;
    s.lastSample = face.updatedAt;
    const smiling =
      face.smile >= 0.66 &&
      face.browFurrow < 0.2 &&
      (face.eyeSquint ?? 0) >= 0.22 &&
      face.mouthOpen >= 0.22;
    if (!smiling) s.since = null;
    else if (s.since === null) s.since = face.updatedAt;
    if (
      smiling &&
      s.since !== null &&
      face.updatedAt - s.since >= REACTION_HOLD_MS &&
      now >= s.cooldownUntil &&
      !s.pendingAt
    ) {
      s.pendingAt = now + 280;
      s.strength = Math.min(0.6, face.smile * 0.65);
      s.cooldownUntil = now + 8000;
    }
  }
  if (s.pendingAt && now >= s.pendingAt) {
    s.pendingAt = 0;
    s.until = now + 1700;
  }
  return now < s.until ? s.strength * Math.min(1, (s.until - now) / 450) : 0;
}

export type ObservedEmpathyFrame = {
  smile: number;
  browFurrow: number;
  browRaise: number;
  frown: number;
  eyeWide: number;
  noseScrunch: number;
  headYaw: number;
  headPitch: number;
  headRoll: number;
};

export type ObservedEmpathyState = {
  lastNow: number;
  lastSample: number;
  sustainedMs: number;
  pendingTriggerAt: number;
  holdUntil: number;
  cooldownUntil: number;
  target: ObservedEmpathyFrame;
  current: ObservedEmpathyFrame;
};

const ZERO_EMPATHY: ObservedEmpathyFrame = {
  smile: 0,
  browFurrow: 0,
  browRaise: 0,
  frown: 0,
  eyeWide: 0,
  noseScrunch: 0,
  headYaw: 0,
  headPitch: 0,
  headRoll: 0,
};

export function createObservedEmpathy(now = 0): ObservedEmpathyState {
  return {
    lastNow: now,
    lastSample: -1,
    sustainedMs: 0,
    pendingTriggerAt: 0,
    holdUntil: 0,
    cooldownUntil: 0,
    target: { ...ZERO_EMPATHY },
    current: { ...ZERO_EMPATHY },
  };
}

/**
 * Subconsciously absorbs sustained fresh user camera behavior (posture, brows, smile, frown, nose scrunch)
 * after a natural delay so the avatar subtly echoes the user's vibe during and right after interactions,
 * without robotic 1:1 copying or treating facial movements as certain emotional proof.
 */
export function stepObservedEmpathy(
  st: ObservedEmpathyState,
  now: number,
  face?: UserFaceObservation,
): ObservedEmpathyFrame {
  const dt = Math.min(0.06, Math.max(0.004, (now - st.lastNow) * 0.001));
  st.lastNow = now;

  const fresh =
    !!face?.detected && now >= face.updatedAt && now - face.updatedAt <= OBSERVATION_MAX_AGE_MS;

  if (fresh && face.updatedAt !== st.lastSample) {
    const gap = st.lastSample >= 0 ? face.updatedAt - st.lastSample : 0;
    st.lastSample = face.updatedAt;
    if (gap > OBSERVATION_MAX_AGE_MS) {
      st.sustainedMs = 0;
    } else {
      const smileSig = Math.max(0, face.smile - 0.34);
      const furrowSig = Math.max(0, face.browFurrow - 0.18);
      const raiseSig = Math.max(0, face.browRaise - 0.28);
      const frownSig = Math.max(0, (face.mouthFrown ?? 0) - 0.24);
      const wideSig = Math.max(0, (face.eyeWide ?? 0) - 0.26);
      const noseSig = Math.max(0, (face.noseSneer ?? 0) - 0.2);
      const poseMag = Math.max(
        Math.abs(face.headYaw) - 0.14,
        Math.abs(face.headPitch) - 0.14,
        Math.abs(face.headRoll ?? 0) - 0.06,
      );
      const salient = Math.max(smileSig, furrowSig, raiseSig, frownSig, wideSig, noseSig, poseMag);

      if (salient > 0.05) {
        st.sustainedMs = Math.min(2000, st.sustainedMs + Math.max(30, gap));
      } else {
        st.sustainedMs = Math.max(0, st.sustainedMs - Math.max(30, gap) * 1.4);
      }

      if (st.sustainedMs >= 600 && now >= st.cooldownUntil && !st.pendingTriggerAt) {
        // Delayed subconscious mirroring snapshot (~360ms reaction latency, ~2.8s carryover)
        st.pendingTriggerAt = now + 360;
        st.cooldownUntil = now + 2600;
        st.target = {
          smile: Math.min(0.32, Math.max(0, face.smile * 0.34)),
          browFurrow: Math.min(0.46, Math.max(0, face.browFurrow * 0.52)),
          browRaise: Math.min(0.34, Math.max(0, face.browRaise * 0.36)),
          frown: Math.min(0.3, Math.max(0, (face.mouthFrown ?? 0) * 0.38)),
          eyeWide: Math.min(0.28, Math.max(0, (face.eyeWide ?? 0) * 0.34)),
          noseScrunch: Math.min(0.42, Math.max(0, (face.noseSneer ?? 0) * 0.52)),
          headYaw: Math.max(-0.28, Math.min(0.28, face.headYaw * 0.34)),
          headPitch: Math.max(-0.24, Math.min(0.24, face.headPitch * 0.32)),
          headRoll: Math.max(-0.06, Math.min(0.06, (face.headRoll ?? 0) * 0.22)),
        };
      }
    }
  } else if (!fresh) {
    st.sustainedMs = 0;
  }

  if (st.pendingTriggerAt && now >= st.pendingTriggerAt) {
    st.pendingTriggerAt = 0;
    st.holdUntil = now + 2900;
  }

  const activeWeight = now < st.holdUntil ? Math.min(1, (st.holdUntil - now) / 900) : 0;
  const followRate = activeWeight > 0 ? 1 - Math.exp(-dt * 2.4) : 1 - Math.exp(-dt * 1.1);

  (Object.keys(ZERO_EMPATHY) as Array<keyof ObservedEmpathyFrame>).forEach((k) => {
    const goal = st.target[k] * activeWeight;
    st.current[k] += (goal - st.current[k]) * followRate;
  });

  return st.current;
}
