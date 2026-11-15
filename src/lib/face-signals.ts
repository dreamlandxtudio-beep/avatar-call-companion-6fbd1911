/** Observable facial actions, not a diagnosis of the person's feelings. */
export function expressionFromSignals(
  smile: number,
  open: number,
  furrow: number,
  frown: number,
  innerUp: number,
  wide: number,
) {
  if ((smile > 0.36 && open > 0.24) || (smile > 0.52 && open > 0.14 && furrow < 0.2))
    return { emotion: "laugh" as const, label: "Risa · sonrisa amplia" };
  if (smile > 0.28) return { emotion: "happy" as const, label: "Sonrisa" };
  if (frown > 0.22 && innerUp > 0.2)
    return { emotion: "sad" as const, label: "Comisuras bajas · cejas interiores elevadas" };
  if (innerUp > 0.42 && wide > 0.22 && open > 0.16)
    return { emotion: "surprised" as const, label: "Cejas elevadas · ojos abiertos" };
  if (furrow > 0.28) return { emotion: "angry" as const, label: "Ceño fruncido" };
  if (frown > 0.26) return { emotion: "sad" as const, label: "Comisuras bajas" };
  return { emotion: "neutral" as const, label: "Expresión en reposo" };
}
