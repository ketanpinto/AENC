import { TrendingUp, FileText, Package, ArrowUpRight, Sparkles } from "lucide-react"
import Link from "next/link"

const features = [
  {
    icon: TrendingUp,
    title: "Predict and Analyze",
    description:
      "Leverage advanced AI algorithms to forecast trends, identify patterns, and make data-driven decisions with unprecedented accuracy.",
    stats: [
      { label: "Accuracy", value: "98.5%" },
      { label: "Processing Speed", value: "2.3s" },
    ],
    gradient: "from-blue-500 via-cyan-500 to-teal-500",
    glowColor: "shadow-blue-500/50",
    link: "/predict-analyze",
  },
  {
    icon: FileText,
    title: "Contract Analysis",
    description:
      "Automatically extract key terms, identify risks, and ensure compliance across thousands of contracts in seconds.",
    stats: [
      { label: "Time Saved", value: "87%" },
      { label: "Risk Detection", value: "99.2%" },
    ],
    gradient: "from-purple-500 via-pink-500 to-rose-500",
    glowColor: "shadow-purple-500/50",
    link: "/contract-analysis",
  },
  {
    icon: Package,
    title: "Supply Chain",
    description:
      "Optimize logistics, predict disruptions, and maintain seamless operations with real-time supply chain intelligence.",
    stats: [
      { label: "Cost Reduction", value: "34%" },
      { label: "Efficiency Gain", value: "56%" },
    ],
    gradient: "from-emerald-500 via-teal-500 to-cyan-500",
    glowColor: "shadow-emerald-500/50",
    link: "/supply-chain",
  },
]

export function FeatureCards() {
  return (
    <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
      {features.map((feature, index) => (
        <div
          key={index}
          className={`glass-effect group relative overflow-hidden rounded-3xl border border-glass-border p-8 shadow-2xl backdrop-blur-2xl transition-all duration-500 hover:scale-[1.02] hover:shadow-3xl hover:${feature.glowColor}`}
        >
          {/* Animated Gradient Background */}
          <div
            className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 transition-opacity duration-500 group-hover:opacity-20`}
          />

          {/* Animated Border Gradient */}
          <div
            className={`absolute inset-0 rounded-3xl bg-gradient-to-br ${feature.gradient} opacity-0 blur-xl transition-opacity duration-500 group-hover:opacity-30`}
          />

          {/* Sparkle Effect */}
          <div className="absolute right-4 top-4 opacity-0 transition-opacity duration-500 group-hover:opacity-100">
            <Sparkles className="h-5 w-5 text-primary animate-pulse" />
          </div>

          <div className="relative z-10">
            {/* Icon with Gradient */}
            <div
              className={`mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br ${feature.gradient} shadow-lg transition-transform duration-500 group-hover:scale-110 group-hover:rotate-3`}
            >
              <feature.icon className="h-7 w-7 text-white" />
            </div>

            {/* Title */}
            <h3 className="mb-4 text-2xl font-bold text-foreground transition-colors duration-300 group-hover:text-primary">
              {feature.title}
            </h3>

            {/* Description */}
            <p className="mb-8 text-pretty leading-relaxed text-muted-foreground">{feature.description}</p>

            {/* Stats with Enhanced Styling */}
            <div className="mb-6 grid grid-cols-2 gap-6 rounded-2xl border border-glass-border bg-glass-bg/50 p-4 backdrop-blur-sm">
              {feature.stats.map((stat, statIndex) => (
                <div key={statIndex} className="text-center">
                  <div
                    className={`mb-1 bg-gradient-to-r ${feature.gradient} bg-clip-text text-3xl font-bold text-transparent`}
                  >
                    {stat.value}
                  </div>
                  <div className="text-xs font-medium text-muted-foreground">{stat.label}</div>
                </div>
              ))}
            </div>

            {/* Enhanced Learn More Button */}
            <Link
              href={feature.link}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-glass-bg/50 py-3 text-sm font-semibold text-foreground backdrop-blur-sm transition-all duration-300 hover:bg-glass-bg group-hover:gap-3"
            >
              Explore Feature
              <ArrowUpRight className="h-4 w-4 transition-transform duration-300 group-hover:translate-x-1 group-hover:-translate-y-1" />
            </Link>
          </div>
        </div>
      ))}
    </div>
  )
}
