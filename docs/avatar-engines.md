# Portrait engines and asset sourcing

## Realistic portraits
- LivePortrait: https://github.com/KwaiVGI/LivePortrait — neural stitching and expression retargeting, Python/PyTorch deployment. Code license alone does not clear all third-party face detector weights for commercial use; inspect each component before deployment.
- LiveAvatar: https://github.com/Alibaba-Quark/LiveAvatar — GPU-hosted audio-driven generation. Intel Iris Xe cannot execute its CUDA pipeline; a URL to Gradio is not a live session API.

## Anime portraits
- Talking Head Anime 3: https://github.com/pkhungurn/talking-head-anime-3-demo — dedicated anime eye, mouth and pose controls, designed around prepared front-facing character illustrations; not arbitrary photos. The demo code is MIT but the pretrained model terms restrict commercial use; do not ship its weights in a paid product without clearance.
- AniPortrait: https://github.com/Zejun-Yang/AniPortrait — despite its name, primarily photorealistic audio/video-driven synthesis, not a dedicated anime browser engine.

These are researched candidates, not installed or connected inference engines. Current animation is local portrait deformation; it cannot reconstruct hidden geometry, synthesize actual tears or infer a person's internal feelings. Camera mirror exposes observable face actions only.

## Public avatar sourcing
Publicly accessible portraits are not automatically reusable or suitable for romantic impersonation. Existing synthetic realistic/anime defaults remain in use. Future catalogue imports need explicit image rights, adult characters and permission for the intended use. Wikimedia Commons offers file-specific licenses and Unsplash has its own license, but neither automatically grants a depicted person's endorsement or romantic impersonation rights.

## Device downloads
Transformers.js Browser Cache retains model weights across reloads, and persistent storage is requested. This is best-effort: a user can clear site data and a browser can decline protection. Preparing weights in RAM is still necessary on each new page session. Small local models may be slower and less capable than hosted models on integrated graphics.