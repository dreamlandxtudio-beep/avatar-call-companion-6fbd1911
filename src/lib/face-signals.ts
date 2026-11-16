/** Observable facial actions, not a diagnosis of the person's feelings. */
export function expressionFromSignals(
  smile: number,
  open: number,
  furrow: number,
  frown: number,
  innerUp: number,
  wide: number,
  noseSneer = 0,
  cheekSquint = 0,
) {
  if (
    furrow < 0.22 &&
    ((smile >= 0.62 && open >= 0.26) ||
      (smile >= 0.5 && open >= 0.34 && (cheekSquint >= 0.2 || smile >= 0.58)))
  ) {
    return { emotion: "laugh" as const, label: "Risa · sonrisa amplia" };
  }
  if (noseSneer >= 0.32 && smile < 0.45) {
    return { emotion: "disgusted" as const, label: "Desagrado · nariz fruncida" };
  }
  if (frown >= 0.22 && innerUp >= 0.2)
    return { emotion: "sad" as const, label: "Comisuras bajas · cejas interiores elevadas" };
  if (furrow >= 0.22) {
    return { emotion: "angry" as const, label: "Ceño fruncido" };
  }
  if (smile >= 0.3) return { emotion: "happy" as const, label: "Sonrisa" };
  if (innerUp >= 0.42 && wide >= 0.22 && open >= 0.16)
    return { emotion: "surprised" as const, label: "Cejas elevadas · ojos abiertos" };
  if (frown >= 0.26) return { emotion: "sad" as const, label: "Comisuras bajas" };
  return { emotion: "neutral" as const, label: "Expresión en reposo" };
}
