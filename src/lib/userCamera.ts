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
  browRaiseLeft?: number;
  browRaiseRight?: number;
  mouthOpen: number;
  tongueOut?: number;
  upperLipLift?: number;
  lowerLipDrop?: number;
  mouthShrug?: number;
  mouthFrown?: number;
  mouthRound?: number;
  mouthPucker?: number;
  mouthWide?: number;
  mouthPress?: number;
  browInnerUp?: number;
  blink?: number;
  blinkLeft?: number;
  blinkRight?: number;
  smileLeft?: number;
  smileRight?: number;
  mouthSkewX?: number;
  eyeWide?: number;
  eyeSquint?: number;
  noseSneer?: number;
  headRoll?: number;
  headYaw: number; // -1 (looking left on screen) to +1 (looking right on screen)
  headPitch: number; // -1 (looking down) to +1 (looking up)
  faceProximity?: number; // -0.48 (farther) to +1.0 (closer to camera)
  gazeX: number; // -1 (eyes looking left) to +1 (eyes looking right)
  gazeY: number; // -1 (eyes looking down) to +1 (eyes looking up)
  labelEs: string;
  contextPrompt: string;
  updatedAt: number;
  meshPoints?: Array<[number, number]>;
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
let baselineFaceScale: number | null = null;

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

export function analyzeUserVideoFrame(
  video: HTMLVideoElement,
  timestampMs: number,
  options?: { includeMesh?: boolean },
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
    const cheekSquint = clamp((bs("cheekSquintLeft") + bs("cheekSquintRight")) * 0.5, 0, 1);
    const mouthDimple = clamp((bs("mouthDimpleLeft") + bs("mouthDimpleRight")) * 0.5, 0, 1);
    const smile = clamp(rawSmile * 1.24 + mouthDimple * 0.14 + cheekSquint * 0.08, 0, 1);
    const browFurrow = clamp((bs("browDownLeft") + bs("browDownRight")) * 0.5, 0, 1);
    const browRaiseLeft = clamp(
      Math.max(bs("browOuterUpLeft") * 1.15, bs("browInnerUp") * 0.72) - bs("browDownLeft") * 0.45,
      0,
      1,
    );
    const browRaiseRight = clamp(
      Math.max(bs("browOuterUpRight") * 1.15, bs("browInnerUp") * 0.72) -
        bs("browDownRight") * 0.45,
      0,
      1,
    );
    const browRaise = clamp(
      Math.max(bs("browInnerUp"), (browRaiseLeft + browRaiseRight) * 0.5),
      0,
      1,
    );
    const mouthFrown = clamp((bs("mouthFrownLeft") + bs("mouthFrownRight")) * 0.5, 0, 1);
    const upperLipLift = clamp((bs("mouthUpperUpLeft") + bs("mouthUpperUpRight")) * 0.88, 0, 1);
    const lowerLipDrop = clamp((bs("mouthLowerDownLeft") + bs("mouthLowerDownRight")) * 0.88, 0, 1);
    const mouthShrug = clamp(Math.max(bs("mouthRollLower"), bs("mouthShrugLower") * 0.88), 0, 1);
    const tongueOut = clamp(bs("tongueOut") * 1.45, 0, 1);

    // Proportional mouth-open curve: immediate subtle sensitivity with full wide-open peak reach
    const lipSpanX = Math.max(0.02, Math.hypot(pts[291]!.x - pts[61]!.x, pts[291]!.y - pts[61]!.y));
    const lipGapY = Math.max(0, pts[14]!.y - pts[13]!.y);
    const geomOpen = clamp((lipGapY / lipSpanX - 0.006) * 1.92, 0, 1);
    const rawOpen = Math.max(bs("jawOpen") * 1.14, geomOpen * 1.02);
    const mouthOpen = rawOpen < 0.004 ? 0 : clamp(Math.pow(rawOpen, 0.88) * 1.04, 0, 1);
    const eyeWide = clamp((bs("eyeWideLeft") + bs("eyeWideRight")) * 0.68, 0, 1);

    // Compute head yaw, pitch & camera proximity from facial landmarks in mirrored selfie space
    const nose = pts[1]!;
    const rCheek = pts[234]!;
    const lCheek = pts[454]!;
    const forehead = pts[10]!;
    const chin = pts[152]!;
    const rEye = pts[33]!;
    const lEye = pts[263]!;

    const eyeSpan = Math.hypot(lEye.x - rEye.x, lEye.y - rEye.y);
    const vertFaceSpan = Math.hypot(chin.x - forehead.x, chin.y - forehead.y);
    const apparentScale = eyeSpan * 1.45 + vertFaceSpan * 0.55;
    if (baselineFaceScale === null) {
      baselineFaceScale = clamp(apparentScale, 0.3, 0.52);
    } else {
      baselineFaceScale = baselineFaceScale * 0.997 + clamp(apparentScale, 0.3, 0.48) * 0.003;
    }
    const rawProx = clamp((apparentScale - 0.39) / 0.22, -0.48, 1);
    const relProx = clamp((apparentScale / Math.max(0.24, baselineFaceScale) - 1) * 1.85, -0.48, 1);
    const faceProximity = clamp(rawProx * 0.45 + relProx * 0.55, -0.48, 1);

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
    const gazeX = clamp((lookLeftRaw - lookRightRaw) * 1.48, -1, 1);
    const gazeY = clamp((lookUp - lookDown) * 1.95, -1, 1);

    const signal = expressionFromSignals(
      smile,
      mouthOpen,
      browFurrow,
      mouthFrown,
      bs("browInnerUp"),
      eyeWide,
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

    // Damp perspective-induced mouth shift & smile asymmetry during profile head turns (prevent fake sideways smirk)
    const yawProfileDamp = Math.max(0, 1 - Math.abs(headYaw) * 1.85);
    const rawSmileLeft = clamp(bs("mouthSmileLeft") * 1.42, 0, 1);
    const rawSmileRight = clamp(bs("mouthSmileRight") * 1.42, 0, 1);
    const strongIntentionalSmirk = Math.abs(rawSmileRight - rawSmileLeft) > 0.34;
    const profileSymBlend = !strongIntentionalSmirk ? 1 - yawProfileDamp : 0;
    const smileLeft =
      rawSmileLeft * (1 - profileSymBlend * 0.72) + smile * (profileSymBlend * 0.72);
    const smileRight =
      rawSmileRight * (1 - profileSymBlend * 0.72) + smile * (profileSymBlend * 0.72);
    const blinkLeft = bs("eyeBlinkLeft");
    const blinkRight = bs("eyeBlinkRight");
    const mouthMidX = (pts[61]!.x + pts[291]!.x) * 0.5;
    const eyeSpanX = Math.max(0.04, Math.abs(lEye.x - rEye.x));
    const cornerTilt = (pts[61]!.y - pts[291]!.y) / lipSpanX - (rEye.y - lEye.y) / eyeSpanX;
    const geomMouthShift = clamp(
      ((mouthMidX - nose.x) / lipSpanX) * 2.5 + cornerTilt * 1.45,
      -0.95,
      0.95,
    );
    const mouthSkewX = clamp(
      (bs("mouthRight") - bs("mouthLeft")) * 2.15 +
        (smileRight - smileLeft) * 1.15 * yawProfileDamp +
        geomMouthShift * 0.55 * yawProfileDamp,
      -1,
      1,
    );
    const mouthPucker = clamp(bs("mouthPucker") * 1.24, 0, 1);
    const mouthRound = clamp(Math.max(bs("mouthPucker"), bs("mouthFunnel") * 1.18), 0, 1);
    const mouthWide = clamp(
      (bs("mouthStretchLeft") + bs("mouthStretchRight")) * 0.68 + smile * 0.38 + mouthDimple * 0.16,
      0,
      1,
    );
    const mouthPress = clamp(
      (bs("mouthPressLeft") + bs("mouthPressRight")) * 0.55 + bs("mouthClose") * 0.35,
      0,
      1,
    );
    const meshPoints = options?.includeMesh
      ? pts.slice(0, 468).map((pt): [number, number] => [pt.x, pt.y])
      : undefined;

    return {
      detected: true,
      emotion,
      smile,
      smileLeft,
      smileRight,
      mouthSkewX,
      mouthRound,
      mouthPucker,
      mouthWide,
      mouthPress,
      upperLipLift,
      lowerLipDrop,
      mouthShrug,
      browFurrow,
      browRaise,
      browRaiseLeft,
      browRaiseRight,
      mouthOpen,
      tongueOut,
      mouthFrown,
      browInnerUp: bs("browInnerUp"),
      blink: (blinkLeft + blinkRight) * 0.5,
      blinkLeft,
      blinkRight,
      eyeWide,
      eyeSquint: clamp(
        (bs("eyeSquintLeft") + bs("eyeSquintRight")) * 0.5 + cheekSquint * 0.25,
        0,
        1,
      ),
      noseSneer: clamp((bs("noseSneerLeft") + bs("noseSneerRight")) * 0.68, 0, 1),
      // Mirror-matched roll sign for -scale-x-100 selfie preview (rEye is on screen-right, lEye on screen-left)
      headRoll: clamp(Math.atan2(rEye.y - lEye.y, lEye.x - rEye.x), -0.45, 0.45),
      headYaw,
      headPitch,
      faceProximity,
      gazeX,
      gazeY,
      labelEs,
      contextPrompt,
      updatedAt: timestampMs,
      ...(meshPoints ? { meshPoints } : {}),
    };
  } catch {
    return null;
  }
}
