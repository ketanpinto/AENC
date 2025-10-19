"use client";

import React, { useState, useEffect } from "react";
import { DashboardHeader } from "@/components/dashboard-header";
import { VideoBackground } from "@/components/video-background";
import { ArrowLeft, TrendingUp } from "lucide-react";
import Link from "next/link";

type SupplyPredictionResponse = {
  risk_score: number | null;
  risk_level: string | null;
  summary: string | null;
  chart_url: string | null;
};

export default function SupplyChainPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SupplyPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Manual input states
  const [orderId, setOrderId] = useState<number>(0);
  const [supplier, setSupplier] = useState<string>("");
  const [productSKU, setProductSKU] = useState<string>("");
  const [orderQuantity, setOrderQuantity] = useState<number>(0);
  const [leadTime, setLeadTime] = useState<number>(0);
  const [stockLevel, setStockLevel] = useState<number>(0);
  const [inventoryThreshold, setInventoryThreshold] = useState<number>(0);
  const [cost, setCost] = useState<number>(0);

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5000";

  async function handlePredict(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);

    setLoading(true);
    try {
      const resp = await fetch(`${API_BASE}/supplychain/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          order_id: orderId,
          supplier,
          product_sku: productSKU,
          order_quantity: orderQuantity,
          lead_time: leadTime,
          stock_level: stockLevel,
          inventory_threshold: inventoryThreshold,
          cost,
        }),
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
        chart_url: json.chart_url ? `${API_BASE}${json.chart_url}` : null,
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
        <Link
          href="/"
          className="glass-effect mb-8 inline-flex items-center gap-2 rounded-full border border-glass-border px-4 py-2 text-sm font-medium text-foreground backdrop-blur-xl transition-all hover:bg-glass-bg"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </Link>

        <div className="glass-effect mb-8 rounded-3xl border border-glass-border p-12 backdrop-blur-2xl">
          <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-green-500 via-lime-500 to-emerald-500 shadow-lg">
            <TrendingUp className="h-8 w-8 text-white" />
          </div>
          <h1 className="mb-4 bg-gradient-to-r from-green-500 via-lime-500 to-emerald-500 bg-clip-text text-5xl font-bold text-transparent">
            Supply Chain Forecast
          </h1>
          <p className="text-pretty text-xl leading-relaxed text-muted-foreground">
            Enter all supply chain order details manually. The system predicts risks, stockouts, and generates visual insights.
          </p>
        </div>

        <div className="glass-effect rounded-3xl border border-glass-border p-8 backdrop-blur-2xl">
          <form onSubmit={handlePredict} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              <InputField label="Order ID" value={orderId} onChange={setOrderId} type="number" />
              <InputField label="Supplier" value={supplier} onChange={setSupplier} />
              <InputField label="Product SKU" value={productSKU} onChange={setProductSKU} />
              <InputField label="Order Quantity" value={orderQuantity} onChange={setOrderQuantity} type="number" />
              <InputField label="Lead Time (days)" value={leadTime} onChange={setLeadTime} type="number" />
              <InputField label="Stock Level" value={stockLevel} onChange={setStockLevel} type="number" />
              <InputField label="Inventory Threshold" value={inventoryThreshold} onChange={setInventoryThreshold} type="number" />
              <InputField label="Cost" value={cost} onChange={setCost} type="number" />

            </div>

            <button
              type="submit"
              disabled={loading}
              className="rounded-md bg-green-600 px-6 py-3 text-white hover:bg-green-500 disabled:opacity-60"
            >
              {loading ? "Predicting..." : "Predict Supply Risk"}
            </button>

            {error && <div className="text-sm text-red-400 mt-2">{error}</div>}

            {result && (
              <div className="mt-8 space-y-8">
                <ResultCard title="Risk Score" value={result.risk_score ?? 0} max={100} />
                <RiskLevelCard level={result.risk_level ?? "Unknown"} />
                <SummaryCard summary={result.summary ?? "No summary available"} />
                {result.chart_url && (
                  <div className="mt-8">
                    <h4 className="mb-3 text-lg font-semibold text-muted-foreground">Supply Risk Chart</h4>
                    <img
                      src={result.chart_url}
                      alt="Supply Risk Chart"
                      className="max-h-80 w-full rounded-lg border border-glass-border object-contain"
                    />
                  </div>
                )}
              </div>
            )}
          </form>
        </div>
      </div>
    </main>
  );
}

// Reusable InputField
function InputField({
  label,
  value,
  onChange,
  type = "text",
}: {
  label: string;
  value: string | number;
  onChange: (v: any) => void;
  type?: string;
}) {
  return (
    <div>
      <label className="block mb-2 text-sm font-medium text-muted-foreground">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(type === "number" ? Number(e.target.value) : e.target.value)}
        className="w-full rounded-md border border-glass-border px-3 py-2 bg-transparent text-white placeholder:text-gray-400"
        placeholder={`Enter ${label}`}
        required
      />
    </div>
  );
}

// Result Components
function ResultCard({ title, value, max }: { title: string; value: number; max: number }) {
  return (
    <div className="glass-effect rounded-2xl border border-glass-border p-6 backdrop-blur-xl">
      <div className="mb-4 flex items-center justify-between">
        <h4 className="text-lg font-semibold text-muted-foreground">{title}</h4>
        <div className="rounded-full bg-gradient-to-r from-green-500/20 to-lime-500/20 px-3 py-1 text-xs font-medium text-green-300">
          OUT OF {max}
        </div>
      </div>
      <div className="flex items-end justify-between">
        <AnimatedCounter value={value} className="text-5xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent" />
        <div className="mb-2 h-2 w-24 rounded-full bg-gray-700">
          <div className="h-full rounded-full bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 transition-all duration-1000 ease-out" style={{ width: `${Math.min(value, max)}%` }} />
        </div>
      </div>
    </div>
  );
}

function RiskLevelCard({ level }: { level: string }) {
  return (
    <div className="glass-effect rounded-2xl border border-glass-border p-6 backdrop-blur-xl">
      <div className="mb-4 flex items-center justify-between">
        <h4 className="text-lg font-semibold text-muted-foreground">Risk Level</h4>
        <div className="rounded-full bg-gradient-to-r from-green-500/20 to-lime-500/20 px-3 py-1 text-xs font-medium text-green-300">
          CLASSIFICATION
        </div>
      </div>
      <div className="flex items-center justify-between">
        <AnimatedRiskLevel level={level} className="text-4xl font-bold" />
        <RiskLevelIndicator level={level} />
      </div>
    </div>
  );
}

function SummaryCard({ summary }: { summary: string }) {
  return (
    <div className="glass-effect rounded-2xl border border-glass-border p-6 backdrop-blur-xl">
      <div className="mb-4 flex items-center justify-between">
        <h4 className="text-lg font-semibold text-muted-foreground">Executive Summary</h4>
        <div className="rounded-full bg-gradient-to-r from-green-500/20 to-lime-500/20 px-3 py-1 text-xs font-medium text-green-300">
          AI ANALYSIS
        </div>
      </div>
      <div className="text-lg leading-relaxed text-gray-200">{summary}</div>
    </div>
  );
}

// Animated Counter
function AnimatedCounter({ value, className }: { value: number; className?: string }) {
  const [displayValue, setDisplayValue] = useState(0);
  useEffect(() => {
    let start = 0;
    const end = value;
    const increments = 60;
    const duration = 2000;
    const stepTime = duration / increments;

    const timer = setInterval(() => {
      start++;
      const progress = start / increments;
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const currentValue = easeOut * end;
      setDisplayValue(parseFloat(currentValue.toFixed(2)));
      if (start >= increments) clearInterval(timer);
    }, stepTime);

    return () => clearInterval(timer);
  }, [value]);

  return <span className={className}>{displayValue}</span>;
}

// Animated Risk Level
function AnimatedRiskLevel({ level, className }: { level: string; className?: string }) {
  const [displayLevel, setDisplayLevel] = useState("");
  useEffect(() => {
    let idx = 0;
    const placeholder = "•••••••";
    const timer = setInterval(() => {
      if (idx <= level.length) {
        setDisplayLevel(level.slice(0, idx) + placeholder.slice(idx));
        idx++;
      } else {
        setDisplayLevel(level);
        clearInterval(timer);
      }
    }, 100);
    return () => clearInterval(timer);
  }, [level]);

  return <span className={className}>{displayLevel}</span>;
}

// Risk Level Indicator
function RiskLevelIndicator({ level }: { level: string }) {
  const getColor = (lvl: string) => {
    const lower = lvl.toLowerCase();
    if (lower.includes("low")) return "bg-green-500";
    if (lower.includes("medium")) return "bg-yellow-500";
    if (lower.includes("high")) return "bg-orange-500";
    if (lower.includes("critical") || lower.includes("very high")) return "bg-red-500";
    return "bg-gray-500";
  };

  const getPulse = (lvl: string) => {
    const lower = lvl.toLowerCase();
    if (lower.includes("critical") || lower.includes("high")) return "animate-pulse";
    return "";
  };

  return (
    <div className="flex items-center gap-2">
      <div className={`h-3 w-3 rounded-full ${getColor(level)} ${getPulse(level)}`} />
    </div>
  );
}
