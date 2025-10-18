import { DashboardHeader } from "@/components/dashboard-header"
import { VideoBackground } from "@/components/video-background"
import { FileText, ArrowLeft } from "lucide-react"
import Link from "next/link"

export default function ContractAnalysisPage() {
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
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-purple-500 via-pink-500 to-rose-500 shadow-lg">
            <FileText className="h-8 w-8 text-white" />
          </div>

          <h1 className="mb-4 bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500 bg-clip-text text-5xl font-bold text-transparent">
            Contract Analysis
          </h1>

          <p className="text-pretty text-xl leading-relaxed text-muted-foreground">
            Automatically extract key terms, identify risks, and ensure compliance across thousands of contracts in
            seconds.
          </p>
        </div>

        {/* Intro Section */}
        <div className="glass-effect rounded-3xl border border-glass-border p-12 backdrop-blur-2xl">
          <h2 className="mb-6 text-3xl font-bold text-foreground">Introduction</h2>
          <div className="space-y-4 text-pretty leading-relaxed text-muted-foreground">
            <p>
              Transform your contract management process with AI-powered analysis that reads, understands, and evaluates
              legal documents at scale. Our Contract Analysis module uses natural language processing to extract
              critical information and identify potential risks before they become problems.
            </p>
            <p>
              Say goodbye to manual contract review. Our system automatically identifies key clauses, payment terms,
              obligations, and compliance requirements across your entire contract portfolio. With 99.2% risk detection
              accuracy, you can trust our AI to catch what human reviewers might miss.
            </p>
            <p>
              Reduce contract review time by 87% while improving accuracy and consistency. Our platform ensures your
              organization maintains compliance, minimizes legal exposure, and accelerates deal cycles.
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
