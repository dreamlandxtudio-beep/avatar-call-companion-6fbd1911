export type SpeechTempoTier = "fast" | "brisk" | "normal" | "slow";

export type SpeechTempoProfile = {
  rate: number;
  pitch: number;
  tempo: SpeechTempoTier;
};

export function voiceProsody(emotion?: string) {
  if (emotion === "sad" || emotion === "triste") return { rate: 0.9, pitch: 0.96 };
  if (emotion === "crying" || emotion === "heartbroken" || emotion === "melancholic")
    return { rate: 0.87, pitch: 0.95 };
  if (emotion === "happy" || emotion === "laugh" || emotion === "feliz" || emotion === "riendo")
    return { rate: 1.04, pitch: 1.06 };
  if (emotion === "euphoric") return { rate: 1.07, pitch: 1.08 };
  if (emotion === "surprised" || emotion === "fear") return { rate: 1.02, pitch: 1.08 };
  if (emotion === "nervous" || emotion === "sweating") return { rate: 1.05, pitch: 1.04 };
  if (emotion === "furious" || emotion === "angry") return { rate: 1.01, pitch: 0.96 };
  if (emotion === "drunk" || emotion === "high" || emotion === "sleepy")
    return { rate: 0.88, pitch: 0.94 };
  if (emotion === "sick" || emotion === "pain") return { rate: 0.91, pitch: 0.95 };
  return { rate: 0.97, pitch: 1 };
}

export function dynamicSpeechTempo(text: string, emotion?: string): SpeechTempoProfile {
  const base = voiceProsody(emotion);
  const trimmed = text.trim();
  if (!trimmed) {
    const tier: SpeechTempoTier =
      base.rate >= 1.12
        ? "fast"
        : base.rate >= 1.03
          ? "brisk"
          : base.rate <= 0.89
            ? "slow"
            : "normal";
    return { ...base, tempo: tier };
  }

  let delta = 0;
  const exclamations = (trimmed.match(/[!¡]/g) ?? []).length;
  const ellipses = (trimmed.match(/\.\.\.|…/g) ?? []).length;
  const commas = (trimmed.match(/[,;—–]/g) ?? []).length;
  const words = trimmed.split(/\s+/).filter(Boolean).length;

  if (exclamations >= 1) delta += Math.min(0.14, exclamations * 0.065);
  if (/\?!|!\?/.test(trimmed)) delta += 0.06;
  if (/\b(?:jaja|haha|jeje|hehe)\w*/i.test(trimmed)) delta += 0.09;
  if (
    /\b(?:rápido|rapido|corre|prisa|vamos|mira|escucha|ahora mismo|en serio|increíble|increible|genial|hurry|quick|fast|wait|wow|omg)\b/i.test(
      trimmed,
    )
  ) {
    delta += 0.09;
  }

  if (ellipses >= 1) delta -= Math.min(0.14, ellipses * 0.075);
  if (
    /\b(?:despacio|calma|tranquil[oa]?|respira|lo siento|perdón|perdon|difícil|dificil|silencio|triste|solo|sola|slow|breathe|sorry|quiet|sigh|hmm)\b/i.test(
      trimmed,
    )
  ) {
    delta -= 0.08;
  }

  if (words <= 6 && exclamations >= 1) delta += 0.05;
  else if (words >= 20 && commas >= 2 && exclamations === 0) delta -= 0.03;

  const rate = Math.round(Math.max(0.8, Math.min(1.3, base.rate + delta)) * 100) / 100;
  const tempo: SpeechTempoTier =
    rate >= 1.12 ? "fast" : rate >= 1.03 ? "brisk" : rate <= 0.9 ? "slow" : "normal";

  return { rate, pitch: base.pitch, tempo };
}
