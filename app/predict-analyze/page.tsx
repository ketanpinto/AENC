"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { VideoBackground } from "@/components/video-background"
import { TrendingUp, ArrowLeft, Play, Zap, Upload } from "lucide-react"
import Link from "next/link"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Separator } from "@/components/ui/separator"

export default function PredictAnalyzePage() {
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [features, setFeatures] = useState<string[]>([])
  const [prediction, setPrediction] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)

  useEffect(() => {
    // Fetch the features from backend
    fetch('http://localhost:5000/features')
      .then(response => response.json())
      .then(data => {
        setFeatures(data.features);
      })
      .catch(err => {
        console.error('Failed to fetch features:', err);
        setError('Failed to load features. Refresh the page.');
      });
  }, []);

  const handleInputChange = (feature: string, value: string) => {
    setInputs({ ...inputs, [feature]: value })
  }

  const handlePredict = async () => {
    // Check if all inputs are filled
    const missing = features.filter(feature => !inputs[feature] || inputs[feature] === '')
    if (missing.length > 0) {
      setError(`Please fill in all required fields: ${missing.join(', ')}`)
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputs),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      if (data.status === 'success') {
        setPrediction(data.prediction)
      } else {
        setError(data.error)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setAnalysisLoading(true)
    setAnalysisResult(null)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://localhost:5000/analyze-budget', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      if (data.status === 'success') {
        setAnalysisResult(data)
      } else {
        setError(data.error)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setAnalysisLoading(false)
    }
  }

  return (
    <main className="relative min-h-screen overflow-hidden">
      <VideoBackground />
      <DashboardHeader />

      <div className="relative z-10 mx-auto max-w-6xl px-6 pt-32 pb-16">
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

        {/* Prediction Section */}
        <Card className="glass-effect mb-8 border border-glass-border backdrop-blur-2xl">
          <CardHeader>
            <CardTitle>Profit Prediction</CardTitle>
            <CardDescription>
              Enter the feature values below to predict profit using our AI model.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
              {features.map((feature) => (
                <div key={feature} className="space-y-2">
                  <Label htmlFor={feature}>{feature.replace('_', ' ')}</Label>
                  <Input
                    id={feature}
                    type="number"
                    step="any"
                    value={inputs[feature] || ''}
                    onChange={(e) => handleInputChange(feature, e.target.value)}
                    placeholder={`Enter ${feature.toLowerCase().replace('_', ' ')}`}
                  />
                </div>
              ))}
            </div>
            <Button onClick={handlePredict} disabled={loading} className="w-full md:w-auto">
              {loading ? 'Predicting...' : 'Predict Profit'}
            </Button>
            {prediction !== null && (
              <Alert className="mt-4">
                <TrendingUp className="h-4 w-4" />
                <AlertTitle>Prediction Result</AlertTitle>
                <AlertDescription>
                  Predicted Profit: {prediction.toFixed(2)}
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {error && (
          <Alert variant="destructive" className="mb-8">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Budget Analysis Section */}
        {/* <Card className="glass-effect mb-8 border border-glass-border backdrop-blur-2xl">
          <CardHeader>
            <CardTitle>Analyze Budget</CardTitle>
            <CardDescription>
              Upload a CSV file to analyze your budget data and get insights on revenue, cost, and top products by profit.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <Label htmlFor="csv-upload">Upload CSV File</Label>
                <Input
                  id="csv-upload"
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  disabled={analysisLoading}
                />
              </div>
              <Button
                onClick={() => document.getElementById('csv-upload')?.click()}
                disabled={analysisLoading}
                className="w-full md:w-auto"
              >
                <Upload className="h-4 w-4 mr-2" />
                {analysisLoading ? 'Analyzing...' : 'Analyze Budget'}
              </Button>
            </div>
          </CardContent>
        </Card> */}

        {/* Analysis Results */}
        {/* {analysisResult && (
          <div className="space-y-6 mb-8"> */}
            {/* Totals */}
            {/* <Card className="glass-effect border border-glass-border backdrop-blur-2xl">
              <CardHeader>
                <CardTitle>Totals</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-500">${analysisResult.totals.revenue.toFixed(2)}</div>
                    <div className="text-sm text-muted-foreground">Total Revenue</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-500">${analysisResult.totals.cost.toFixed(2)}</div>
                    <div className="text-sm text-muted-foreground">Total Cost</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-500">${analysisResult.totals.profit.toFixed(2)}</div>
                    <div className="text-sm text-muted-foreground">Total Profit</div>
                  </div>
                </div>
              </CardContent>
            </Card> */}

            {/* Average Metrics */}
            {/* <Card className="glass-effect border border-glass-border backdrop-blur-2xl">
              <CardHeader>
                <CardTitle>Average Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(analysisResult.averages).map(([key, value]: [string, any]) => (
                    <div key={key} className="text-center">
                      <div className="text-lg font-semibold">{value.toFixed(2)}</div>
                      <div className="text-sm text-muted-foreground">{key.replace('_', ' ')}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card> */}

            {/* Top 5 Products */}
            {/* <Card className="glass-effect border border-glass-border backdrop-blur-2xl">
              <CardHeader>
                <CardTitle>Top 5 Products by Profit</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {analysisResult.top5_products.map((product: any, index: number) => (
                    <div key={index} className="flex justify-between items-center p-4 bg-secondary/10 rounded-lg">
                      <div>
                        <div className="font-semibold">#{index + 1}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">Profit: ${product.Profit.toFixed(2)}</div>
                        <div className="text-sm text-muted-foreground">Revenue: ${product.Revenue.toFixed(2)}</div>
                        <div className="text-sm text-muted-foreground">Cost: ${product.Cost.toFixed(2)}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )} */}

        {/* Intro Section */}
        {/* <div className="glass-effect rounded-3xl border border-glass-border p-12 backdrop-blur-2xl">
          <h2 className="mb-6 text-3xl font-bold text-foreground">How It Works</h2>
          <div className="space-y-4 text-pretty leading-relaxed text-muted-foreground">
            <p>
              Our Predict and Analyze module harnesses the power of cutting-edge artificial intelligence to transform
              your raw data into actionable insights. By analyzing historical patterns and real-time data streams, our
              system delivers predictions with industry-leading accuracy.
            </p>
            <p>
              Enter feature values above to predict profit, or upload a CSV file to analyze your budget data. Our AI
              model will process the information and provide accurate predictions and analytics.
            </p>
            <p>
              Whether you're forecasting market trends, predicting customer behavior, or optimizing operational
              efficiency, our AI-driven platform provides the intelligence you need to stay ahead of the competition.
            </p>
          </div>
        </div> */}
      </div>
    </main>
  )
}
