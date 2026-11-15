export const visionFileset = {
  wasmLoaderPath: "/vision/vision_wasm_internal.js",
  wasmBinaryPath: "/vision/vision_wasm_internal.wasm",
};

export async function resolveVisionFileset() {
  try {
    const { FilesetResolver } = await import("@mediapipe/tasks-vision");
    return await FilesetResolver.forVisionTasks("/vision");
  } catch {
    return visionFileset;
  }
}
