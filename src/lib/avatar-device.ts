/** Keep portrait rendering bounded even on high-DPI phones and integrated graphics. */
export function portraitRenderBudget(width: number, height: number, dpr: number) {
  const scale = Math.min(Math.max(1, dpr), 1.5, 1280 / Math.max(1, width, height));
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

export const DEFAULT_DEVICE_VOICE = { engine: "system" as const, id: "" };
