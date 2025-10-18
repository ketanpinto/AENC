import { DashboardHeader } from "@/components/dashboard-header"
import { VideoBackground } from "@/components/video-background"
import { Package, ArrowLeft } from "lucide-react"
import Link from "next/link"

export default function SupplyChainPage() {
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
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 via-teal-500 to-cyan-500 shadow-lg">
            <Package className="h-8 w-8 text-white" />
          </div>

          <h1 className="mb-4 bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 bg-clip-text text-5xl font-bold text-transparent">
            Supply Chain Intelligence
          </h1>

          <p className="text-pretty text-xl leading-relaxed text-muted-foreground">
            Optimize logistics, predict disruptions, and maintain seamless operations with real-time supply chain
            intelligence.
          </p>
        </div>

        {/* Intro Section */}
        <div className="glass-effect rounded-3xl border border-glass-border p-12 backdrop-blur-2xl">
          <h2 className="mb-6 text-3xl font-bold text-foreground">Introduction</h2>
          <div className="space-y-4 text-pretty leading-relaxed text-muted-foreground">
            <p>
              Navigate the complexities of modern supply chains with AI-powered intelligence that sees around corners.
              Our Supply Chain module monitors thousands of data points across your entire logistics network, predicting
              disruptions before they impact your operations.
            </p>
            <p>
              From supplier performance to inventory optimization, our platform provides real-time visibility and
              predictive insights that keep your supply chain running smoothly. Identify bottlenecks, optimize routes,
              and reduce costs while improving delivery times and customer satisfaction.
            </p>
            <p>
              Companies using our Supply Chain Intelligence have achieved an average 34% cost reduction and 56%
              efficiency gain. Our AI continuously learns from your operations, delivering increasingly accurate
              predictions and recommendations over time.
            </p>
          </div>
        </div>
      </div>
    </main>
  )
}
