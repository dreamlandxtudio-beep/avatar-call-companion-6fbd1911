// Dedicated Web Worker for local open-source ONNX LLMs so the UI and avatar never freeze.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let gen: any = null;
let loadedRepo = "";

function stripThink(text: string): string {
  return text.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trimStart();
}

async function hasWebGPU(): Promise<boolean> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (self.navigator as any)?.gpu;
    if (!gpu) return false;
    const adapter = await gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as
    | { type: "load"; id: string; repo: string; cpuDtype: "q4" | "q8" }
    | {
        type: "reply";
        reqId: number;
        id: string;
        repo: string;
        cpuDtype: "q4" | "q8";
        messages: Array<{ role: string; content: string }>;
      };

  try {
    if (msg.type === "load" || (msg.type === "reply" && (!gen || loadedRepo !== msg.repo))) {
      if (gen && loadedRepo !== msg.repo) {
        try {
          await gen.dispose?.();
        } catch {
          /* ignore */
        }
        gen = null;
        loadedRepo = "";
      }

      if (!gen) {
        const { pipeline, env } = await import("@huggingface/transformers");
        env.allowLocalModels = false;
        env.useBrowserCache = true;
        // Single-threaded WASM avoids SharedArrayBuffer / COOP-COEP hangs inside iframes.
        if (env.backends?.onnx?.wasm) {
          env.backends.onnx.wasm.numThreads = 1;
        }

        const useGPU = await hasWebGPU();
        const createWith = async (gpu: boolean) => {
          const files: Record<string, { loaded: number; total: number }> = {};
          return pipeline("text-generation", msg.repo, {
            device: gpu ? "webgpu" : "wasm",
            dtype: gpu ? "q4f16" : msg.cpuDtype,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            progress_callback: (p: any) => {
              if (p.status === "progress") {
                const totalBytes = typeof p.total === "number" && p.total > 0 ? p.total : 0;
                const loadedBytes = typeof p.loaded === "number" ? p.loaded : 0;
                if (totalBytes > 0 && p.file) {
                  files[p.file] = { loaded: loadedBytes, total: totalBytes };
                  const vals = Object.values(files);
                  const sumLoaded = vals.reduce((a, b) => a + b.loaded, 0);
                  const sumTotal = vals.reduce((a, b) => a + b.total, 0);
                  const ratio = sumTotal > 0 ? Math.min(1, sumLoaded / sumTotal) : 0;
                  self.postMessage({
                    type: "progress",
                    id: msg.id,
                    phase: ratio >= 0.995 ? "initializing" : "downloading",
                    progress: ratio,
                    loadedMB: Math.round(sumLoaded / 1024 / 1024),
                    totalMB: Math.round(sumTotal / 1024 / 1024),
                  });
                } else if (typeof p.progress === "number") {
                  const ratio = Math.min(1, Math.max(0, p.progress / 100));
                  self.postMessage({
                    type: "progress",
                    id: msg.id,
                    phase: ratio >= 0.995 ? "initializing" : "downloading",
                    progress: ratio,
                  });
                }
              } else if (p.status === "done" && p.file?.endsWith(".onnx")) {
                self.postMessage({
                  type: "progress",
                  id: msg.id,
                  phase: "initializing",
                  progress: 1,
                });
              }
            },
          });
        };

        let device: "GPU" | "CPU" = useGPU ? "GPU" : "CPU";
        try {
          gen = await createWith(useGPU);
        } catch (err) {
          if (!useGPU) throw err;
          device = "CPU";
          self.postMessage({
            type: "progress",
            id: msg.id,
            phase: "downloading",
            progress: 0,
            loadedMB: 0,
            totalMB: 0,
          });
          gen = await createWith(false);
        }
        loadedRepo = msg.repo;
        self.postMessage({ type: "ready", id: msg.id, device });
      }
    }

    if (msg.type === "reply") {
      const { TextStreamer } = await import("@huggingface/transformers");
      let acc = "";
      const streamer = new TextStreamer(gen.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (t: string) => {
          acc += t;
          const clean = stripThink(acc);
          if (clean) {
            self.postMessage({ type: "token", reqId: msg.reqId, text: clean });
          }
        },
      });
      const out = await gen(msg.messages, {
        max_new_tokens: 110,
        do_sample: true,
        temperature: 0.7,
        top_p: 0.9,
        repetition_penalty: 1.15,
        tokenizer_encode_kwargs: { enable_thinking: false },
        streamer,
      });
      const raw = out?.[0]?.generated_text?.at?.(-1)?.content ?? acc;
      const text = stripThink(String(raw)).trim() || "…";
      self.postMessage({ type: "result", reqId: msg.reqId, text });
    }
  } catch (e) {
    const error = String((e as Error)?.message || e);
    if (msg.type === "reply") {
      self.postMessage({ type: "reply_error", reqId: msg.reqId, error });
    } else {
      self.postMessage({ type: "load_error", id: msg.id, error });
    }
  }
};
