import { useState } from "react";

export const DEFAULT_AVATAR_URL = "/avatars/default.jpg";

export function AvatarImg({
  src,
  name,
  className = "",
}: {
  src: string | null;
  name: string;
  className?: string;
}) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const effectiveSrc =
    !src || src.includes("/__l5e/") || failedSrc === src ? DEFAULT_AVATAR_URL : src;

  return (
    <img
      src={effectiveSrc}
      alt={name}
      onError={() => {
        if (effectiveSrc !== DEFAULT_AVATAR_URL) {
          setFailedSrc(src);
        }
      }}
      className={`rounded-full object-cover ${className}`}
    />
  );
}
