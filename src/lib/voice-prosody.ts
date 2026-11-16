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
