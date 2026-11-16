// Bilingual Spanish + English grapheme-to-phoneme/viseme engine for lip-reading-grade speech articulation.

export type PhonemeProto = {
  open: number;
  wide: number;
  round: number;
  press: number;
  pucker: number;
  tongue: number;
  labiodental: number;
  stress: number;
  weight: number;
  charIdx: number;
};

const VOWEL_RE = /[aáàäeéèëiíìïoóòöuúùüy]/i;
const ACCENTED_VOWEL_RE = /[áéíóú]/i;
const LETTER_RE = /[a-záéíóúüñ]/i;

function makeProto(charIdx: number, partial: Partial<Omit<PhonemeProto, "charIdx">>): PhonemeProto {
  return {
    open: 0,
    wide: 0,
    round: 0,
    press: 0,
    pucker: 0,
    tongue: 0,
    labiodental: 0,
    stress: 0,
    weight: 1,
    ...partial,
    charIdx,
  };
}

function detectEnglishBias(text: string): boolean {
  const enMarkers = (
    text.match(
      /\b(the|and|you|that|with|this|what|have|for|are|was|not|hello|please|thank|love|feel)\b/gi,
    ) ?? []
  ).length;
  const esMarkers = (
    text.match(
      /[áéíóúüñ¿¡]|\b(que|como|cómo|hola|para|pero|estoy|eres|muy|bien|todo|nada|gracias|por|con)\b/gi,
    ) ?? []
  ).length;
  return enMarkers > esMarkers;
}

export function textToVisemeProtos(text: string): {
  protos: PhonemeProto[];
  wordStartChars: number[];
} {
  const chars = Array.from(text);
  const lower = chars.map((c) => c.toLowerCase());
  const protos: PhonemeProto[] = [];
  const wordStartChars: number[] = [];
  const englishBias = detectEnglishBias(text);

  let inWord = false;
  let i = 0;

  while (i < lower.length) {
    const ch = lower[i]!;
    const prev = i > 0 ? lower[i - 1]! : " ";
    const next = i + 1 < lower.length ? lower[i + 1]! : " ";
    const next2 = i + 2 < lower.length ? lower[i + 2]! : " ";
    const isLetter = LETTER_RE.test(ch);

    if (isLetter && !inWord) {
      wordStartChars.push(i);
      inWord = true;
    } else if (!isLetter) {
      inWord = false;
    }

    // 1) Punctuation and whitespace boundaries
    if (/[,;:]/.test(ch)) {
      protos.push(makeProto(i, { open: 0.02, press: 0.14, weight: 1.5 }));
      i++;
      continue;
    }
    if (/[.!?…]/.test(ch)) {
      protos.push(makeProto(i, { open: 0.0, press: 0.22, weight: 2.3 }));
      i++;
      continue;
    }
    if (/\s/.test(ch)) {
      protos.push(makeProto(i, { open: 0.07, wide: 0.04, round: 0.03, press: 0.05, weight: 0.32 }));
      i++;
      continue;
    }
    if (!isLetter) {
      i++;
      continue;
    }

    // 2) Spanish/English multi-character digraphs & silent letters
    // "qu" before e/i -> crisp /k/ + skip silent u
    if (ch === "q" && next === "u" && /[eéèiíì]/i.test(next2)) {
      protos.push(
        makeProto(i, { open: 0.28, wide: 0.14, round: 0.04, tongue: 0.14, weight: 0.65 }),
      );
      i += 2;
      continue;
    }
    // "gu" before e/i -> /g/ + skip silent u
    if (ch === "g" && next === "u" && /[eéèiíì]/i.test(next2)) {
      protos.push(
        makeProto(i, { open: 0.24, wide: 0.12, round: 0.06, tongue: 0.16, weight: 0.65 }),
      );
      i += 2;
      continue;
    }
    // "gü" -> /g/ + /w/
    if (ch === "g" && next === "ü") {
      protos.push(makeProto(i, { open: 0.22, round: 0.92, pucker: 0.85, weight: 0.95 }));
      i += 2;
      continue;
    }
    // "ch" -> postalveolar /tʃ/ (gentle lip rounding + teeth)
    if (ch === "c" && next === "h") {
      protos.push(
        makeProto(i, {
          open: 0.18,
          wide: 0.12,
          round: 0.42,
          pucker: 0.28,
          tongue: 0.24,
          weight: 0.85,
        }),
      );
      i += 2;
      continue;
    }
    // "sh" -> /ʃ/ lip protrusion
    if (ch === "s" && next === "h") {
      protos.push(
        makeProto(i, { open: 0.14, round: 0.56, pucker: 0.48, tongue: 0.16, weight: 0.88 }),
      );
      i += 2;
      continue;
    }
    // "th" -> interdental /θ/ /ð/ (visible tongue tip)
    if (ch === "t" && next === "h") {
      protos.push(makeProto(i, { open: 0.22, wide: 0.24, tongue: 0.72, weight: 0.82 }));
      i += 2;
      continue;
    }
    // "ph" -> labiodental /f/
    if (ch === "p" && next === "h") {
      protos.push(
        makeProto(i, { open: 0.08, wide: 0.14, labiodental: 0.92, press: 0.18, weight: 0.78 }),
      );
      i += 2;
      continue;
    }
    // "wh" -> /w/ tight pucker
    if (ch === "w" && next === "h") {
      protos.push(makeProto(i, { open: 0.2, round: 0.95, pucker: 0.9, weight: 0.95 }));
      i += 2;
      continue;
    }
    // "ll" -> palatal /ʝ/ vs double l
    if (ch === "l" && next === "l") {
      protos.push(makeProto(i, { open: 0.28, wide: 0.58, tongue: 0.34, weight: 0.86 }));
      i += 2;
      continue;
    }
    // "rr" -> trilled alveolar /r/
    if (ch === "r" && next === "r") {
      protos.push(
        makeProto(i, { open: 0.26, wide: 0.14, round: 0.16, tongue: 0.62, weight: 0.92 }),
      );
      i += 2;
      continue;
    }
    // English vowel teams: ee / ea / oo / ou / ow
    if (ch === "e" && (next === "e" || next === "a")) {
      protos.push(
        makeProto(i, { open: 0.3, wide: 0.88, tongue: 0.22, stress: 0.65, weight: 1.25 }),
      );
      i += 2;
      continue;
    }
    if (ch === "o" && next === "o") {
      protos.push(
        makeProto(i, { open: 0.24, round: 0.96, pucker: 0.92, stress: 0.55, weight: 1.25 }),
      );
      i += 2;
      continue;
    }
    if (ch === "o" && (next === "u" || next === "w")) {
      protos.push(
        makeProto(i, { open: 0.64, round: 0.42, pucker: 0.22, stress: 0.6, weight: 0.85 }),
      );
      protos.push(makeProto(i + 1, { open: 0.24, round: 0.94, pucker: 0.88, weight: 0.8 }));
      i += 2;
      continue;
    }
    // Silent Spanish "h"
    if (ch === "h" && !englishBias) {
      i++;
      continue;
    }
    // Silent English trailing "e" after vowel+consonant (e.g. "home", "make")
    if (
      englishBias &&
      ch === "e" &&
      !LETTER_RE.test(next) &&
      i >= 2 &&
      !VOWEL_RE.test(prev) &&
      VOWEL_RE.test(lower[i - 2]!)
    ) {
      i++;
      continue;
    }

    // 3) Single vowels
    if (/[aáàä]/i.test(ch)) {
      const stressed = ACCENTED_VOWEL_RE.test(ch) ? 0.95 : 0.45;
      protos.push(
        makeProto(i, {
          open: stressed > 0.7 ? 0.92 : 0.84,
          wide: 0.22,
          round: 0.04,
          tongue: 0.06,
          stress: stressed,
          weight: stressed > 0.7 ? 1.48 : 1.32,
        }),
      );
      i++;
      continue;
    }
    if (/[eéèë]/i.test(ch)) {
      const stressed = ACCENTED_VOWEL_RE.test(ch) ? 0.9 : 0.4;
      protos.push(
        makeProto(i, {
          open: 0.52,
          wide: 0.72,
          tongue: 0.14,
          stress: stressed,
          weight: stressed > 0.7 ? 1.3 : 1.15,
        }),
      );
      i++;
      continue;
    }
    if (/[iíìïy]/i.test(ch)) {
      const stressed = ACCENTED_VOWEL_RE.test(ch) ? 0.85 : 0.35;
      protos.push(
        makeProto(i, {
          open: 0.26,
          wide: 0.88,
          tongue: 0.22,
          stress: stressed,
          weight: stressed > 0.7 ? 1.18 : 1.04,
        }),
      );
      i++;
      continue;
    }
    if (/[oóòö]/i.test(ch)) {
      const stressed = ACCENTED_VOWEL_RE.test(ch) ? 0.92 : 0.45;
      protos.push(
        makeProto(i, {
          open: 0.58,
          round: 0.84,
          pucker: 0.62,
          tongue: 0.04,
          stress: stressed,
          weight: stressed > 0.7 ? 1.4 : 1.24,
        }),
      );
      i++;
      continue;
    }
    if (/[uúùüw]/i.test(ch)) {
      const stressed = ACCENTED_VOWEL_RE.test(ch) ? 0.88 : 0.38;
      protos.push(
        makeProto(i, {
          open: 0.22,
          round: 0.96,
          pucker: 0.92,
          tongue: 0.04,
          stress: stressed,
          weight: stressed > 0.7 ? 1.28 : 1.14,
        }),
      );
      i++;
      continue;
    }

    // 4) Single consonants collapsed across immediate doublets (e.g. "mm", "pp", "tt")
    if (/[pbm]/i.test(ch)) {
      protos.push(makeProto(i, { open: 0.0, press: 0.98, pucker: 0.1, weight: 0.76 }));
    } else if (/[fv]/i.test(ch)) {
      protos.push(
        makeProto(i, { open: 0.09, wide: 0.14, labiodental: 0.92, press: 0.22, weight: 0.76 }),
      );
    } else if (/[szx]/i.test(ch) || (ch === "c" && /[eéèiíì]/i.test(next))) {
      protos.push(
        makeProto(i, {
          open: 0.16,
          wide: 0.46,
          round: 0.06,
          press: 0.04,
          tongue: 0.18,
          weight: 0.76,
        }),
      );
    } else if (/[tdlnrñ]/i.test(ch)) {
      const strongTongue = /[ln]/i.test(ch) ? 0.68 : /[td]/i.test(ch) ? 0.54 : 0.44;
      protos.push(
        makeProto(i, { open: 0.28, wide: 0.22, round: 0.06, tongue: strongTongue, weight: 0.72 }),
      );
    } else if (/[ckgjh]/i.test(ch)) {
      protos.push(makeProto(i, { open: 0.34, wide: 0.16, round: 0.1, tongue: 0.18, weight: 0.7 }));
    }

    // Collapse double identical consonant so "mm" or "ss" doesn't stall timing
    if (next === ch && !VOWEL_RE.test(ch)) {
      i += 2;
    } else {
      i++;
    }
  }

  return { protos, wordStartChars };
}
