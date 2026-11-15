import { describe, expect, it } from "vitest";
import { DEFAULT_DEVICE_VOICE, portraitRenderBudget } from "@/lib/avatar-device";
import { defaultVoiceForGender } from "@/lib/store";

describe("Device-first avatar defaults", () => {
  it("defaults to system speech rather than a shared online voice quota", () => {
    expect(DEFAULT_DEVICE_VOICE).toEqual({ engine: "system", id: "" });
    expect(defaultVoiceForGender("female")).toEqual(DEFAULT_DEVICE_VOICE);
    expect(defaultVoiceForGender("male")).toEqual(DEFAULT_DEVICE_VOICE);
  });
  it("limits high-DPI portrait output to 1280 pixels", () => {
    expect(portraitRenderBudget(1000, 2000, 3)).toEqual({ width: 640, height: 1280 });
  });
});
