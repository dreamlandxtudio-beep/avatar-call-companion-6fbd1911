export function providerError(error: unknown): { status: number; message: string } {
  let message = error instanceof Error ? error.message : String(error);
  let status = 500;
  try {
    const parsed = JSON.parse(message) as { error?: { code?: number; message?: string } };
    status = parsed.error?.code ?? status;
    message = parsed.error?.message ?? message;
  } catch {
    /* plain provider message */
  }
  if (status === 429 && /quota|limit/i.test(message)) {
    return {
      status,
      message:
        "Google ha agotado la cuota de voz. Revisa el límite y la facturación de Google AI Studio.",
    };
  }
  return { status: status >= 400 && status <= 599 ? status : 500, message };
}
