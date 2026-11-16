import { resolveVisionFileset } from "./vision-runtime";
// Client-side user webcam capture & MediaPipe real-time facial expression + gaze/head-turn tracking.
import type { AvatarEmotion } from "./ai";
import { expressionFromSignals } from "./face-signals";

export type UserFaceObservation = {
  detected: boolean;
  emotion: AvatarEmotion;
  smile: number;
  browFurrow: number;
  browRaise: number;
  mouthOpen: number;
  mouthFrown?: number;
  mouthRound?: number;
  mouthWide?: number;
  mouthPress?: number;
  browInnerUp?: number;
  browLeft?: number;
  browRight?: number;
  blink?: number;
  blinkLeft?: number;
  blinkRight?: number;
  smileLeft?: number;
  smileRight?: number;
  cheekSquintLeft?: number;
  cheekSquintRight?: number;
  mouthSkewX?: number;
  tongueOut?: number;
  eyeWide?: number;
  eyeSquint?: number;
  noseSneer?: number;
  headRoll?: number;
  headYaw: number; // -1 (looking left on screen) to +1 (looking right on screen)
  headPitch: number; // -1 (looking down) to +1 (looking up)
  gazeX: number; // -1 (eyes looking left) to +1 (eyes looking right)
  gazeY: number; // -1 (eyes looking down) to +1 (eyes looking up)
  labelEs: string;
  contextPrompt: string;
  updatedAt: number;
};

export type CameraStartResult =
  | { stream: MediaStream; error?: undefined }
  | { stream: null; error: "not-allowed" | "not-found" | "in-use" | "error" };

export function canUseCamera(): boolean {
  return typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;
}

function classifyMediaError(err: unknown): "not-allowed" | "not-found" | "in-use" | "error" {
  const name = err instanceof Error ? err.name : "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
    return "not-allowed";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "in-use";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "not-found";
  }
  return "error";
}

export async function startUserCamera(): Promise<CameraStartResult> {
  if (!canUseCamera()) {
    return { stream: null, error: "not-found" };
  }
  const attempts: MediaStreamConstraints[] = [
    {
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 24, max: 30 },
      },
      audio: false,
    },
    { video: { facingMode: "user" }, audio: false },
    { video: true, audio: false },
  ];
  let lastErr: unknown;
  for (const constraints of attempts) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      return { stream };
    } catch (err) {
      lastErr = err;
      const classified = classifyMediaError(err);
      if (classified === "not-allowed" || classified === "not-found") {
        return { stream: null, error: classified };
      }
    }
  }
  return { stream: null, error: classifyMediaError(lastErr) };
}

export function stopUserCamera(stream: MediaStream | null | undefined) {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try {
      track.stop();
    } catch {
      /* ignore */
    }
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let videoLandmarker: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let videoLandmarkerPromise: Promise<any> | null = null;
let lastVideoTimestampMs = -1;

export async function ensureUserFaceTracker(): Promise<boolean> {
  if (videoLandmarker) return true;
  if (!videoLandmarkerPromise) {
    videoLandmarkerPromise = (async () => {
      const { FaceLandmarker } = await import("@mediapipe/tasks-vision");
      const init = async () => {
        const fs = await resolveVisionFileset();
        return FaceLandmarker.createFromOptions(fs, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          outputFaceBlendshapes: true,
          numFaces: 1,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
        });
      };
      return await init();
    })();
  }
  try {
    videoLandmarker = await videoLandmarkerPromise;
    return !!videoLandmarker;
  } catch {
    videoLandmarkerPromise = null;
    videoLandmarker = null;
    return false;
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

let mouthCropCanvas: HTMLCanvasElement | null = null;
let mouthCropCtx: CanvasRenderingContext2D | null = null;

/**
 * Estimates tongue protrusion from the central lower oral opening when the jaw opens.
 * MediaPipe's 52 ARKit blendshapes omit `tongueOut`, so we inspect pinkness/brightness over the inner lower lip.
 */
function detectTongueOutFromVideo(
  video: HTMLVideoElement,
  pts: Array<{ x: number; y: number }>,
  mouthOpen: number,
): number {
  if (mouthOpen < 0.24 || typeof document === "undefined") return 0;
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (vw < 16 || vh < 16) return 0;

  const pLeft = pts[78];
  const pRight = pts[308];
  const pTop = pts[13];
  const pBot = pts[14];
  const pOuterBot = pts[17];
  if (!pLeft || !pRight || !pTop || !pBot || !pOuterBot) return 0;

  const x0 = Math.max(0, Math.min(pLeft.x, pRight.x));
  const x1 = Math.min(1, Math.max(pLeft.x, pRight.x));
  const y0 = Math.max(0, pTop.y);
  const y1 = Math.min(1, pOuterBot.y + Math.max(0.01, pOuterBot.y - pBot.y) * 0.45);
  const w = x1 - x0;
  const h = y1 - y0;
  if (w < 0.02 || h < 0.015) return 0;

  if (!mouthCropCanvas) {
    mouthCropCanvas = document.createElement("canvas");
    mouthCropCanvas.width = 24;
    mouthCropCanvas.height = 18;
    mouthCropCtx = mouthCropCanvas.getContext("2d", { willReadFrequently: true });
  }
  if (!mouthCropCtx) return 0;

  mouthCropCtx.drawImage(video, x0 * vw, y0 * vh, w * vw, h * vh, 0, 0, 24, 18);
  const data = mouthCropCtx.getImageData(0, 0, 24, 18).data;

  // Compare central lower cavity/lip rim (protruding tongue is bright warm pink-red compared to dark cavity shadow)
  let pinkSum = 0;
  let count = 0;
  for (let py = 6; py < 15; py++) {
    for (let px = 7; px < 17; px++) {
      const o = (py * 24 + px) * 4;
      const r = data[o]!;
      const g = data[o + 1]!;
      const b = data[o + 2]!;
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const redness = r - (g + b) * 0.5;
      // Distinct moist tongue pink over shadowed oral cavity without triggering on white teeth
      if (luma > 68 && luma < 225 && redness > 24 && r > g + 16 && r > b + 14) {
        pinkSum += Math.min(1, (redness - 20) / 36);
      }
      count++;
    }
  }
  const frac = count > 0 ? pinkSum / count : 0;
  return clamp((frac - 0.18) * 2.4, 0, 1);
}

export function analyzeUserVideoFrame(
  video: HTMLVideoElement,
  timestampMs: number,
): UserFaceObservation | null {
  if (!videoLandmarker || video.readyState < 2 || video.videoWidth < 8 || video.videoHeight < 8) {
    return null;
  }
  const monoTs = Math.max(lastVideoTimestampMs + 1, Math.round(timestampMs));
  lastVideoTimestampMs = monoTs;

  try {
    const res = videoLandmarker.detectForVideo(video, monoTs);
    const pts = res?.faceLandmarks?.[0] as Array<{ x: number; y: number; z?: number }> | undefined;
    if (!pts || pts.length < 468) {
      return {
        detected: false,
        emotion: "neutral",
        smile: 0,
        browFurrow: 0,
        browRaise: 0,
        mouthOpen: 0,
        headYaw: 0,
        headPitch: 0,
        gazeX: 0,
        gazeY: 0,
        labelEs: "Sin rostro detectado",
        contextPrompt: "",
        updatedAt: timestampMs,
      };
    }

    const bsMap = new Map<string, number>();
    const cats = res?.faceBlendshapes?.[0]?.categories as
      Array<{ categoryName: string; score: number }> | undefined;
    if (cats) {
      for (const c of cats) {
        bsMap.set(c.categoryName, c.score);
      }
    }
    const bs = (name: string) => bsMap.get(name) ?? 0;

    const rawSmile = clamp((bs("mouthSmileLeft") + bs("mouthSmileRight")) * 0.5, 0, 1);
    const smile = clamp(Math.max(0, rawSmile - 0.04) * 1.05, 0, 1);
    const browFurrow = clamp((bs("browDownLeft") + bs("browDownRight")) * 0.68, 0, 1);
    const browRaise = clamp(
      Math.max(bs("browInnerUp"), (bs("browOuterUpLeft") + bs("browOuterUpRight")) * 0.55),
      0,
      1,
    );
    const mouthFrown = clamp((bs("mouthFrownLeft") + bs("mouthFrownRight")) * 0.58, 0, 1);
    const mouthOpen = clamp(bs("jawOpen") * 1.24, 0, 1);
    const eyeWide = clamp((bs("eyeWideLeft") + bs("eyeWideRight")) * 0.5, 0, 1);
    const noseSneer = clamp((bs("noseSneerLeft") + bs("noseSneerRight")) * 0.85, 0, 1);
    const cheekSquintLeft = clamp(bs("cheekSquintLeft") * 1.15, 0, 1);
    const cheekSquintRight = clamp(bs("cheekSquintRight") * 1.15, 0, 1);
    const avgCheekSquint = (cheekSquintLeft + cheekSquintRight) * 0.5;

    // Compute head yaw & pitch from facial symmetry in mirrored selfie coordinate space
    const nose = pts[1]!;
    const rCheek = pts[234]!;
    const lCheek = pts[454]!;
    const forehead = pts[10]!;
    const chin = pts[152]!;
    const rEye = pts[33]!;
    const lEye = pts[263]!;

    const dR = Math.max(0.01, Math.abs(nose.x - rCheek.x));
    const dL = Math.max(0.01, Math.abs(lCheek.x - nose.x));
    // Positive = user turns toward their right (right side of mirrored selfie view)
    const headYaw = clamp(((dL - dR) / (dL + dR)) * 2.1, -1, 1);

    const eyeMidY = (rEye.y + lEye.y) * 0.5;
    const faceSpanY = Math.max(0.05, chin.y - forehead.y);
    const noseRelY = (nose.y - eyeMidY) / faceSpanY;
    // Neutral noseRelY is ~0.27; lower noseRelY means looking up (+), higher means looking down (-)
    const headPitch = clamp((0.27 - noseRelY) * 4.5, -1, 1);

    // Gaze estimation from MediaPipe eyeLook blendshapes (inverted for mirrored selfie coords)
    const lookRightRaw = (bs("eyeLookInLeft") + bs("eyeLookOutRight")) * 0.5;
    const lookLeftRaw = (bs("eyeLookOutLeft") + bs("eyeLookInRight")) * 0.5;
    const lookUp = (bs("eyeLookUpLeft") + bs("eyeLookUpRight")) * 0.5;
    const lookDown = (bs("eyeLookDownLeft") + bs("eyeLookDownRight")) * 0.5;
    const gazeX = clamp((lookLeftRaw - lookRightRaw) * 1.4, -1, 1);
    const gazeY = clamp((lookUp - lookDown) * 1.4, -1, 1);

    const signal = expressionFromSignals(
      smile,
      mouthOpen,
      browFurrow,
      mouthFrown,
      bs("browInnerUp"),
      eyeWide,
      noseSneer,
      avgCheekSquint,
    );
    const emotion = signal.emotion;
    const moodEs = signal.label;

    let dirEs = "mirando al frente";
    if (headYaw < -0.26 || gazeX < -0.35) {
      dirEs = "mirando hacia la izquierda";
    } else if (headYaw > 0.26 || gazeX > 0.35) {
      dirEs = "mirando hacia la derecha";
    } else if (headPitch > 0.28) {
      dirEs = "mirando hacia arriba";
    } else if (headPitch < -0.28) {
      dirEs = "mirando hacia abajo";
    }

    const labelEs = `${moodEs} · ${dirEs}`;
    const contextPrompt = `Se observan rasgos faciales compatibles con ${moodEs.toLowerCase()}, ${dirEs}. No establecen su emoción real: pregunta si es relevante.`;

    const smileLeft = clamp(Math.max(0, bs("mouthSmileLeft") - 0.04) * 1.08, 0, 1);
    const smileRight = clamp(Math.max(0, bs("mouthSmileRight") - 0.04) * 1.08, 0, 1);
    const blinkLeft = bs("eyeBlinkLeft");
    const blinkRight = bs("eyeBlinkRight");
    const browLeft = clamp(
      bs("browOuterUpLeft") * 0.95 + bs("browInnerUp") * 0.65 - bs("browDownLeft") * 1.15,
      -0.95,
      1,
    );
    const browRight = clamp(
      bs("browOuterUpRight") * 0.95 + bs("browInnerUp") * 0.65 - bs("browDownRight") * 1.15,
      -0.95,
      1,
    );
    const tongueOut = detectTongueOutFromVideo(video, pts, mouthOpen);
    const mouthSkewX = clamp(
      bs("mouthRight") - bs("mouthLeft") + (smileRight - smileLeft) * 0.45,
      -1,
      1,
    );
    const mouthRound = clamp(Math.max(bs("mouthPucker"), bs("mouthFunnel") * 1.15), 0, 1);
    const mouthWide = clamp(
      (bs("mouthStretchLeft") + bs("mouthStretchRight")) * 0.55 + smile * 0.36,
      0,
      1,
    );
    const mouthPress = clamp(
      (bs("mouthPressLeft") + bs("mouthPressRight")) * 0.55 + bs("mouthClose") * 0.35,
      0,
      1,
    );

    return {
      detected: true,
      emotion,
      smile,
      smileLeft,
      smileRight,
      cheekSquintLeft,
      cheekSquintRight,
      mouthSkewX,
      tongueOut,
      mouthRound,
      mouthWide,
      mouthPress,
      browFurrow,
      browRaise,
      browLeft,
      browRight,
      mouthOpen,
      mouthFrown,
      browInnerUp: bs("browInnerUp"),
      blink: (blinkLeft + blinkRight) * 0.5,
      blinkLeft,
      blinkRight,
      eyeWide,
      eyeSquint: (bs("eyeSquintLeft") + bs("eyeSquintRight")) * 0.5,
      noseSneer,
      // Mirror-matched roll sign for -scale-x-100 selfie preview (rEye is on screen-right, lEye on screen-left)
      headRoll: clamp(Math.atan2(rEye.y - lEye.y, lEye.x - rEye.x), -0.45, 0.45),
      headYaw,
      headPitch,
      gazeX,
      gazeY,
      labelEs,
      contextPrompt,
      updatedAt: timestampMs,
    };
  } catch {
    return null;
  }
}
