export function voiceProsody(emotion?: string) {
  if (emotion === "sad" || emotion === "triste") return { rate: 0.9, pitch: 0.96 };
  if (emotion === "happy" || emotion === "laugh" || emotion === "feliz" || emotion === "riendo")
    return { rate: 1.04, pitch: 1.06 };
  if (emotion === "surprised") return { rate: 1.02, pitch: 1.08 };
  return { rate: 0.97, pitch: 1 };
}
