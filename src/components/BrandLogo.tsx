export function BrandLogo({ compact = false }: { compact?: boolean | undefined }) {
  return (
    <span className="inline-flex items-center gap-2.5 font-display select-none">
      <svg
        width="30"
        height="30"
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        className="shrink-0"
      >
        <defs>
          <linearGradient
            id="yume-mono"
            x1="4"
            y1="4"
            x2="28"
            y2="28"
            gradientUnits="userSpaceOnUse"
          >
            <stop offset="0%" stopColor="#fafafa" />
            <stop offset="55%" stopColor="#d4d4d4" />
            <stop offset="100%" stopColor="#737373" />
          </linearGradient>
        </defs>
        {/* Crisp rounded aperture chat badge */}
        <path
          d="M16 3.2C8.93 3.2 3.2 8.48 3.2 15C3.2 18.28 4.66 21.24 7.02 23.38L5.35 28.25L10.82 25.92C12.44 26.48 14.18 26.8 16 26.8C23.07 26.8 28.8 21.52 28.8 15C28.8 8.48 23.07 3.2 16 3.2Z"
          fill="#121212"
          stroke="url(#yume-mono)"
          strokeWidth="2.1"
          strokeLinejoin="round"
        />
        {/* Sculpted crescent moon inside bubble */}
        <path
          d="M15.8 9.2C12.35 9.48 9.7 12.36 9.85 15.85C10.01 19.35 12.95 22.05 16.45 21.9C18.35 21.82 20.02 20.92 21.08 19.55C18.2 19.95 15.58 17.85 15.25 14.95C15.02 12.88 15.95 10.95 17.52 9.78C16.96 9.42 16.38 9.15 15.8 9.2Z"
          fill="url(#yume-mono)"
        />
        {/* Crisp 4-point star */}
        <path
          d="M20.6 9.6L21.35 11.75L23.5 12.5L21.35 13.25L20.6 15.4L19.85 13.25L17.7 12.5L19.85 11.75L20.6 9.6Z"
          fill="#ffffff"
        />
      </svg>
      {!compact && (
        <span className="text-[22px] font-semibold tracking-tight text-foreground">yume</span>
      )}
    </span>
  );
}
