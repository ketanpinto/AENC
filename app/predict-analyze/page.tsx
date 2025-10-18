import { DashboardHeader } from "@/components/dashboard-header"
import { VideoBackground } from "@/components/video-background"
import { TrendingUp, ArrowLeft } from "lucide-react"
import Link from "next/link"

export default function PredictAnalyzePage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      <VideoBackground />
      <DashboardHeader />

      <div className="relative z-10 mx-auto max-w-5xl px-6 pt-32">
        {/* Back Button */}
        <Link
          href="/"
          className="glass-effect mb-8 inline-flex items-center gap-2 rounded-full border border-glass-border px-4 py-2 text-sm font-medium text-foreground backdrop-blur-xl transition-all hover:bg-glass-bg"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </Link>

        {/* Header Section */}
        <div className="glass-effect mb-12 rounded-3xl border border-glass-border p-12 backdrop-blur-2xl">
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 via-cyan-500 to-teal-500 shadow-lg">
            <TrendingUp className="h-8 w-8 text-white" />
          </div>

          <h1 className="mb-4 bg-gradient-to-r from-blue-500 via-cyan-500 to-teal-500 bg-clip-text text-5xl font-bold text-transparent">
            Predict and Analyze
          </h1>

          <p className="text-pretty text-xl leading-relaxed text-muted-foreground">
            Leverage advanced AI algorithms to forecast trends, identify patterns, and make data-driven decisions with
            unprecedented accuracy.
          </p>
        </div>

        {/* Intro Section */}
        <div className="glass-effect rounded-3xl border border-glass-border p-12 backdrop-blur-2xl">
          <h2 className="mb-6 text-3xl font-bold text-foreground">Introduction</h2>
          <div className="space-y-4 text-pretty leading-relaxed text-muted-foreground">
            <p>
              Our Predict and Analyze module harnesses the power of cutting-edge artificial intelligence to transform
              your raw data into actionable insights. By analyzing historical patterns and real-time data streams, our
              system delivers predictions with industry-leading accuracy.
            </p>
            <p>
              Whether you're forecasting market trends, predicting customer behavior, or optimizing operational
              efficiency, our AI-driven platform provides the intelligence you need to stay ahead of the competition.
            </p>
            <p>
              With processing speeds measured in seconds and accuracy rates exceeding 98%, you can make confident
              decisions backed by data science and machine learning expertise.
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
