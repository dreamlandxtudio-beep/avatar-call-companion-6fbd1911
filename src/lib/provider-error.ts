export function providerError(error: unknown): { status: number; message: string } {
  let message = error instanceof Error ? error.message : String(error);
  let status = 500;
  try {
    const parsed = JSON.parse(message) as {
      error?: { code?: number; status?: string; message?: string };
    };
    status = parsed.error?.code ?? status;
    message = parsed.error?.message ?? message;
  } catch {
    /* plain provider message */
  }
  if (status === 429 && /quota|limit|exhausted/i.test(message)) {
    return {
      status,
      message:
        "Google ha agotado la cuota de voz. Revisa el límite y la facturación de Google AI Studio.",
    };
  }
  if (
    /API_KEY_INVALID|API key not valid|PERMISSION_DENIED|UNAUTHENTICATED|no está configurada/i.test(
      message,
    ) ||
    status === 401 ||
    status === 403
  ) {
    return {
      status: status === 403 ? 403 : 400,
      message:
        "Falta una clave válida de Google Gemini. Verifica GEMINI_API_KEY en Settings > Secrets o activa el modo local.",
    };
  }
  return { status: status >= 400 && status <= 599 ? status : 500, message };
}
