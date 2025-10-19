"use client";

import React, { useState, useEffect } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { VideoBackground } from "@/components/video-background";
import { FileText, ArrowLeft, UploadCloud } from "lucide-react";
import Link from "next/link";

type AnalysisResponse = {
  risk_score: number | null;
  risk_level: string | null;
  summary: string | null;
  features: Record<string, any> | null;
  chart_url: string | null;
  file_saved_as?: string;
};

export default function ContractAnalysisPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Change this if your backend is hosted elsewhere
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5000";

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!file) {
      setError("Please select a PDF file first.");
      return;
    }

    const fd = new FormData();
    fd.append("file", file);

    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/analyze-contract`, {
        method: "POST",
        body: fd,
      });

      if (!resp.ok) {
        const txt = await resp.text();
        setError(`Server error: ${resp.status} ${txt}`);
        setLoading(false);
        return;
      }

      const json = await resp.json();
      setResult({
        risk_score: json.risk_score ?? null,
        risk_level: json.risk_level ?? null,
        summary: json.summary ?? null,
        features: json.features ?? null,
        chart_url: json.chart_url ? `${API_BASE}${json.chart_url}` : null,
        file_saved_as: json.file_saved_as,
      });
    } catch (err: any) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

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
        <div className="glass-effect mb-8 rounded-3xl border border-glass-border p-12 backdrop-blur-2xl">
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-purple-500 via-pink-500 to-rose-500 shadow-lg">
            <FileText className="h-8 w-8 text-white" />
          </div>

          <h1 className="mb-4 bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500 bg-clip-text text-5xl font-bold text-transparent">
            Contract Analysis
          </h1>

          <p className="text-pretty text-xl leading-relaxed text-muted-foreground">
            Upload a contract (PDF). The backend will extract clauses, compute features, predict risk score/level, and return a visual chart.
          </p>
        </div>

        {/* Upload Card */}
        <div className="glass-effect rounded-3xl border border-glass-border p-8 backdrop-blur-2xl">
          <form onSubmit={handleUpload} className="space-y-6">
            <div className="flex items-center gap-4">
              <label className="flex cursor-pointer items-center gap-3 rounded-md border border-dashed border-glass-border px-4 py-3 text-sm hover:bg-glass-bg">
                <UploadCloud className="h-5 w-5" />
                <span>{file ? file.name : "Choose a PDF file..."}</span>
                <input
                  type="file"
                  accept="application/pdf"
                  className="hidden"
                  onChange={(e) => {
                    setFile(e.target.files?.[0] ?? null);
                  }}
                />
              </label>

              <button
                type="submit"
                disabled={loading}
                className="rounded-md bg-indigo-600 px-4 py-2 text-white hover:bg-indigo-500 disabled:opacity-60"
              >
                {loading ? "Analyzing..." : "Analyze Contract"}
              </button>
            </div>

            {error && <div className="text-sm text-red-400">Error: {error}</div>}

            {result && (
              <div className="mt-8 space-y-8">
                {/* Risk Score & Level - Top Section */}
                <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                  {/* Risk Score */}
                  <div className="glass-effect rounded-2xl border border-glass-border p-6 backdrop-blur-xl">
                    <div className="mb-4 flex items-center justify-between">
                      <h4 className="text-lg font-semibold text-muted-foreground">Risk Score</h4>
                      <div className="rounded-full bg-gradient-to-r from-purple-500/20 to-pink-500/20 px-3 py-1 text-xs font-medium text-purple-300">
                        OUT OF 100
                      </div>
                    </div>
                    <div className="flex items-end justify-between">
                      <AnimatedCounter 
                        value={result.risk_score ?? 0} 
                        className="text-5xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent" 
                      />
                      <div className="mb-2 h-2 w-24 rounded-full bg-gray-700">
                        <div 
                          className="h-full rounded-full bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 transition-all duration-1000 ease-out"
                          style={{ width: `${Math.min((result.risk_score ?? 0), 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Risk Level */}
                  <div className="glass-effect rounded-2xl border border-glass-border p-6 backdrop-blur-xl">
                    <div className="mb-4 flex items-center justify-between">
                      <h4 className="text-lg font-semibold text-muted-foreground">Risk Level</h4>
                      <div className="rounded-full bg-gradient-to-r from-purple-500/20 to-pink-500/20 px-3 py-1 text-xs font-medium text-purple-300">
                        CLASSIFICATION
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <AnimatedRiskLevel 
                        level={result.risk_level ?? "Unknown"} 
                        className="text-4xl font-bold" 
                      />
                      <RiskLevelIndicator level={result.risk_level} />
                    </div>
                  </div>
                </div>

                {/* Summary - Bottom Section */}
                <div className="glass-effect rounded-2xl border border-glass-border p-6 backdrop-blur-xl">
                  <div className="mb-4 flex items-center justify-between">
                    <h4 className="text-lg font-semibold text-muted-foreground">Executive Summary</h4>
                    <div className="rounded-full bg-gradient-to-r from-purple-500/20 to-pink-500/20 px-3 py-1 text-xs font-medium text-purple-300">
                      AI ANALYSIS
                    </div>
                  </div>
                  <div className="text-lg leading-relaxed text-gray-200">
                    {result.summary ?? "No summary available"}
                  </div>
                </div>
              </div>
            )}

            {/* Chart */}
            {result?.chart_url && (
              <div className="mt-8">
                <h4 className="mb-3 text-lg font-semibold text-muted-foreground">Risk Analysis Chart</h4>
                <img
                  src={result.chart_url}
                  alt="Risk Chart"
                  className="max-h-80 w-full rounded-lg border border-glass-border object-contain"
                />
              </div>
            )}
          </form>
        </div>
      </div>
    </main>
  );
}

// Animated Counter Component
function AnimatedCounter({ value, className }: { value: number | null; className?: string }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    if (value === null) return;

    let start = 0;
    const end = value;
    const duration = 2000; // 2 seconds
    const increments = 60;
    const stepTime = duration / increments;
    const stepValue = end / increments;

    let current = 0;
    const timer = setInterval(() => {
      current += 1;
      const progress = current / increments;
      // Ease out function - fast start, slow end
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const currentValue = easeOut * end;
      setDisplayValue(parseFloat(currentValue.toFixed(2)));

      if (current >= increments) {
        setDisplayValue(parseFloat(end.toFixed(2)));
        clearInterval(timer);
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [value]);

  if (value === null) return <span className={className}>N/A</span>;

  return <span className={className}>{displayValue}</span>;
}

// Animated Risk Level Component
function AnimatedRiskLevel({ level, className }: { level: string; className?: string }) {
  const [displayLevel, setDisplayLevel] = useState("");

  useEffect(() => {
    let currentIndex = 0;
    const targetLevel = level;
    const placeholder = "•••••••";
    
    const timer = setInterval(() => {
      if (currentIndex <= targetLevel.length) {
        setDisplayLevel(targetLevel.slice(0, currentIndex) + placeholder.slice(currentIndex));
        currentIndex++;
      } else {
        setDisplayLevel(targetLevel);
        clearInterval(timer);
      }
    }, 100);

    return () => clearInterval(timer);
  }, [level]);

  return <span className={className}>{displayLevel}</span>;
}

// Risk Level Indicator Component
function RiskLevelIndicator({ level }: { level: string | null }) {
  if (!level) return null;

  const getColor = (lvl: string) => {
    const lower = lvl.toLowerCase();
    if (lower.includes('low')) return 'bg-green-500';
    if (lower.includes('medium')) return 'bg-yellow-500';
    if (lower.includes('high')) return 'bg-orange-500';
    if (lower.includes('critical') || lower.includes('very high')) return 'bg-red-500';
    return 'bg-gray-500';
  };

  const getPulse = (lvl: string) => {
    const lower = lvl.toLowerCase();
    if (lower.includes('critical') || lower.includes('high')) return 'animate-pulse';
    return '';
  };

  return (
    <div className="flex items-center gap-2">
      <div className={`h-3 w-3 rounded-full ${getColor(level)} ${getPulse(level)}`} />
    </div>
  );
}