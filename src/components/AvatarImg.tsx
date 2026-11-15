export function AvatarImg({
  src,
  name,
  className = "",
}: {
  src: string | null;
  name: string;
  className?: string;
}) {
  const effectiveSrc = src || "/__l5e/assets-v1/ea7b23b3-bee4-43b1-ae6c-9228330211ec/default.jpg";
  return <img src={effectiveSrc} alt={name} className={`rounded-full object-cover ${className}`} />;
}
