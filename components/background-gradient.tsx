export function BackgroundGradient() {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      {/* Main gradient orbs */}
      <div className="absolute -left-1/4 top-0 h-[600px] w-[600px] rounded-full bg-primary/30 blur-[120px]" />
      <div className="absolute -right-1/4 top-1/4 h-[500px] w-[500px] rounded-full bg-accent/20 blur-[100px]" />
      <div className="absolute bottom-0 left-1/3 h-[400px] w-[400px] rounded-full bg-primary/20 blur-[90px]" />

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
