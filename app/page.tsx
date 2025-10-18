import { DashboardHeader } from "@/components/dashboard-header"
import { FeatureCards } from "@/components/feature-cards"
import { VideoBackground } from "@/components/video-background"

export default function Home() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      <VideoBackground />

      <div className="relative z-10">
        <DashboardHeader />

        <div className="container mx-auto px-4 py-24 md:py-32">
          <div className="mx-auto max-w-7xl">
            {/* Hero Section */}
            <div className="mb-20 text-center">
              <div className="mb-6 inline-block rounded-full border border-primary/20 bg-primary/10 px-5 py-2 text-sm font-semibold text-primary backdrop-blur-xl">
                AI-Powered Intelligence
              </div>
              <h1 className="mb-6 text-balance text-5xl font-bold tracking-tight text-foreground md:text-7xl">
                Adaptive Enterprise
                <br />
                <span className="bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
                  Nerve Center
                </span>
              </h1>
              <p className="mx-auto mb-10 max-w-2xl text-pretty text-lg leading-relaxed text-muted-foreground md:text-xl">
                The central brain for your business. Harness AI-driven insights to predict, analyze, and optimize every
                aspect of your enterprise operations.
              </p>
              <div className="flex flex-wrap items-center justify-center gap-4">
                <button className="glass-effect rounded-2xl border border-glass-border px-8 py-4 font-semibold text-foreground shadow-xl backdrop-blur-xl transition-all hover:scale-105 hover:shadow-2xl">
                  Get Started
                </button>
                <button className="rounded-2xl border border-glass-border bg-transparent px-8 py-4 font-semibold text-foreground backdrop-blur-xl transition-all hover:bg-glass-bg hover:scale-105">
                  Watch Demo
                </button>
              </div>
            </div>

            {/* Feature Cards */}
            <FeatureCards />
          </div>
        </div>
      </div>
    </main>
  )
}
