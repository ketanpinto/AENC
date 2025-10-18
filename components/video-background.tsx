export function VideoBackground() {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      {/* Video container */}
      <div className="absolute inset-0">
        <video autoPlay loop muted playsInline className="h-full w-full object-cover">
          <source src="/hackbg.mp4" type="video/mp4" />
          {/* Fallback gradient background if video doesn't load */}
        </video>

        {/* Dark overlay to ensure text readability */}
        <div className="absolute inset-0 bg-background/70 backdrop-blur-[2px]" />
      </div>

      {/* Grid overlay */}
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(to right, oklch(0.98 0 0 / 0.05) 1px, transparent 1px),
            linear-gradient(to bottom, oklch(0.98 0 0 / 0.05) 1px, transparent 1px)
          `,
          backgroundSize: "80px 80px",
        }}
      />
    </div>
  )
}
