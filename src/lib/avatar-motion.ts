/** Photo warping cannot reconstruct occluded geometry. Keep motion within the source's evidence. */
export function mouthMotion(open: number, hasTeeth: boolean, restRatio: number) {
  const speech = Math.max(0, Math.min(1, open));
  return {
    opening: speech * (hasTeeth ? 0.68 : 0.78),
    closure: Math.max(0, Math.min(0.35, restRatio)) * (1 - Math.min(1, speech * 4)),
    useOriginalTeeth: hasTeeth,
    syntheticTeeth: true,
  };
}

export function blinkClosure(progress: number) {
  if (progress <= 0 || progress >= 1) return 0;
  // Crisp downward sweep, clear complete closure dwell, smooth natural opening.
  if (progress < 0.28) return Math.sin(((progress / 0.28) * Math.PI) / 2);
  if (progress < 0.62) return 1;
  return Math.cos((((progress - 0.62) / 0.38) * Math.PI) / 2);
}
