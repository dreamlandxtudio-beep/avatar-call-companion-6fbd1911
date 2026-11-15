import { visionFileset } from "./vision-runtime";
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
  browInnerUp?: number;
  blink?: number;
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
  | { stream: null; error: "not-allowed" | "not-found" | "error" };

export function canUseCamera(): boolean {
  return typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;
}

export async function startUserCamera(): Promise<CameraStartResult> {
  if (!canUseCamera()) {
    return { stream: null, error: "not-found" };
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 24, max: 30 },
      },
      audio: false,
    });
    return { stream };
  } catch (err) {
    const name = err instanceof Error ? err.name : "";
    if (
      name === "NotAllowedError" ||
      name === "PermissionDeniedError" ||
      name === "SecurityError"
    ) {
      return { stream: null, error: "not-allowed" };
    }
    if (
      name === "NotFoundError" ||
      name === "DevicesNotFoundError" ||
      name === "OverconstrainedError"
    ) {
      return { stream: null, error: "not-found" };
    }
    return { stream: null, error: "error" };
  }
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
        const fs = await Promise.resolve(visionFileset);
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

    const smile = clamp((bs("mouthSmileLeft") + bs("mouthSmileRight")) * 0.5, 0, 1);
    const browFurrow = clamp((bs("browDownLeft") + bs("browDownRight")) * 0.5, 0, 1);
    const browRaise = clamp(
      Math.max(bs("browInnerUp"), (bs("browOuterUpLeft") + bs("browOuterUpRight")) * 0.5),
      0,
      1,
    );
    const mouthFrown = clamp((bs("mouthFrownLeft") + bs("mouthFrownRight")) * 0.5, 0, 1);
    const mouthOpen = clamp(bs("jawOpen"), 0, 1);
    const eyeWide = clamp((bs("eyeWideLeft") + bs("eyeWideRight")) * 0.5, 0, 1);

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

    const signal = expressionFromSignals(smile, mouthOpen, browFurrow, mouthFrown, bs("browInnerUp"), eyeWide);
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

    return {
      detected: true,
      emotion,
      smile,
      browFurrow,
      browRaise,
      mouthOpen,
      mouthFrown,
      browInnerUp: bs("browInnerUp"),
      blink: (bs("eyeBlinkLeft") + bs("eyeBlinkRight")) * 0.5,
      eyeWide,
      eyeSquint: (bs("eyeSquintLeft") + bs("eyeSquintRight")) * 0.5,
      noseSneer: (bs("noseSneerLeft") + bs("noseSneerRight")) * 0.5,
      headRoll: clamp(Math.atan2(lEye.y - rEye.y, lEye.x - rEye.x), -0.45, 0.45),
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
