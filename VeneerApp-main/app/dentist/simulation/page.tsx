"use client"

import { useState, useEffect } from "react"
import { VeneerShadeSelector } from "@/components/veneer-shade-selector"
import { SimulationLoading } from "@/components/simulation-loading"
import { Button } from "@/components/ui/button"
import { Download, FileText, Share2, ArrowLeft } from "lucide-react"
import { useRouter } from "next/navigation"
import { useToast } from "@/hooks/use-toast"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { BoundingBoxSelector } from "@/components/bounding-selector";

export default function DentistSimulationPage() {
  const [selectedShade, setSelectedShade] = useState("natural_white")
  const [numVariations, setNumVariations] = useState(4)
  const [isGenerating, setIsGenerating] = useState(false)
  const [simulatedImages, setSimulatedImages] = useState<string[]>([])
  const [patientName, setPatientName] = useState("")
  const [patientId, setPatientId] = useState("")
  const router = useRouter()
  const { toast } = useToast()

  const [boundingBox, setBoundingBox] = useState< {
    x: number,
    y: number,
    width: number,
    height: number
  } | null>(null);

  const [originalImage, setOriginalImage] = useState<string>("/placeholder.svg?height=600&width=800")

  useEffect(() => {
    const stored = localStorage.getItem('uploadedPatientImage')
    console.log('DEBUG Frontend: stored image from localStorage:', stored?.substring(0, 100))
    if (stored) {
      setOriginalImage(stored)
    }
  }, [])

  const handleGenerate = async () => {
    if (!originalImage || originalImage.startsWith('/placeholder')) {
      alert('Please upload or capture an image first!');
      return;
    }
    if (!originalImage.startsWith('data:image/')) {
      alert('Invalid image format. Please upload a valid image.');
      return;
    }
    setIsGenerating(true)
    try {
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 600000) // 10 min timeout

      const response = await fetch("/api/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: originalImage,
          shade: selectedShade,
          numOutputs: numVariations,
          boundingBox: boundingBox,
        }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Simulation failed")
      }

      setSimulatedImages(Array.isArray(data.output) ? data.output : [data.output])
      toast({
        title: "Success!",
        description: `Generated ${numVariations} variation(s) successfully.`,
      })
    } catch (error) {
      console.error("Error:", error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to generate simulation. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsGenerating(false)
    }
  }

  const handleDownloadAll = () => {
    simulatedImages.forEach((img, i) => {
      const link = document.createElement("a")
      link.href = img
      link.download = `patient-${patientId || "unnamed"}-variation-${i + 1}.jpg`
      link.click()
    })
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border/50 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-center px-4 py-4 sm:px-6 lg:px-8">
          <div className="text-xl font-bold">
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              VeneerVision AI
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        {!isGenerating && simulatedImages.length === 0 && (
          <div className="space-y-8">
            <div className="text-center">
              <h1 className="mb-3 text-4xl font-bold">Configure Professional Simulation</h1>
              <p className="text-muted-foreground">Set patient details and simulation parameters</p>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              {/* Patient Information */}
              <div className="glass rounded-2xl p-6">
                <h3 className="mb-4 text-lg font-semibold">Patient Information</h3>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="patientName">Patient Name</Label>
                    <Input
                      id="patientName"
                      placeholder="Enter patient name"
                      value={patientName}
                      onChange={(e) => setPatientName(e.target.value)}
                      className="bg-background/50"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="patientId">Patient ID</Label>
                    <Input
                      id="patientId"
                      placeholder="Enter patient ID"
                      value={patientId}
                      onChange={(e) => setPatientId(e.target.value)}
                      className="bg-background/50"
                    />
                  </div>
                </div>
              </div>

              {/* Simulation Settings */}
              <div className="glass rounded-2xl p-6">
                <h3 className="mb-4 text-lg font-semibold">Simulation Settings</h3>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="variations">Number of Variations</Label>
                    <div className="flex items-center gap-4">
                      <Input
                        id="variations"
                        type="number"
                        min="1"
                        max="4"
                        value={numVariations}
                        onChange={(e) => setNumVariations(Number(e.target.value))}
                        className="bg-background/50"
                      />
                      <span className="text-sm text-muted-foreground">(Max: 4)</span>
                    </div>
                  </div>

                  <div className="rounded-lg border border-border/50 bg-accent/5 p-4">
                    <p className="text-sm leading-relaxed text-muted-foreground">
                      Multiple variations allow you to show patients different options during consultation. Each
                      variation may have subtle differences in appearance.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Shade Selection */}
            <div className="glass rounded-2xl p-6">
              <VeneerShadeSelector onShadeSelect={setSelectedShade} selectedShade={selectedShade} />
            </div>

            {/* Mouth Region Selection */}
            {originalImage && !originalImage.startsWith('/placeholder') && (
              <div className="glass rounded-2xl p-6">
                <BoundingBoxSelector
                  imageUrl={originalImage}
                  onBoundingBoxChange={setBoundingBox}
                  initialBox={boundingBox}
                />
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-3">
              <Button
                variant="outline"
                className="flex-1 bg-transparent"
                onClick={() => router.push("/dentist")}
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back
              </Button>
              <Button className="flex-1" onClick={handleGenerate} size="lg">
                Generate {numVariations} Variation{numVariations > 1 ? "s" : ""}
              </Button>
            </div>
          </div>
        )}

        {isGenerating && <SimulationLoading />}

        {!isGenerating && simulatedImages.length > 0 && (
          <div className="space-y-8">
            <div className="text-center">
              <h2 className="mb-2 text-3xl font-bold">Professional Simulation Results</h2>
              <p className="text-muted-foreground">
                {patientName && `Patient: ${patientName}`}
                {patientName && patientId && " • "}
                {patientId && `ID: ${patientId}`}
              </p>
            </div>

            {/* Results Grid */}
            <div className="grid gap-6 md:grid-cols-2">
              {/* Original */}
              <div className="glass overflow-hidden rounded-2xl p-4">
                <div className="mb-3 text-center">
                  <span className="inline-block rounded-full bg-muted px-3 py-1 text-sm font-medium">
                    Original Photo
                  </span>
                </div>
                <div className="relative aspect-[4/3] overflow-hidden rounded-xl">
                  <img
                    src={originalImage || "/placeholder.svg"}
                    alt="Original smile"
                    className="h-full w-full object-cover"
                  />
                </div>
              </div>

              {/* Variations */}
              {simulatedImages.map((image, index) => (
                <div key={index} className="glass relative overflow-hidden rounded-2xl p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="inline-block rounded-full bg-accent/20 px-3 py-1 text-sm font-medium text-accent">
                      Variation {index + 1} - {selectedShade}
                    </span>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-8 w-8 p-0"
                      onClick={() => {
                        const link = document.createElement("a")
                        link.href = image
                        link.download = `variation-${index + 1}.jpg`
                        link.click()
                      }}
                      aria-label="Download variation"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="relative aspect-[4/3] overflow-hidden rounded-xl">
                    <img
                      src={image || "/placeholder.svg"}
                      alt={`Variation ${index + 1}`}
                      className="h-full w-full object-cover"
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Professional Actions */}
            <div className="glass rounded-xl p-6">
              <h3 className="mb-4 text-lg font-semibold">Professional Tools</h3>
              <div className="flex flex-wrap gap-3">
                <Button onClick={handleDownloadAll}>
                  <Download className="mr-2 h-4 w-4" />
                  Download All Results
                </Button>
                <Button variant="outline" className="bg-transparent">
                  <FileText className="mr-2 h-4 w-4" />
                  Generate Report
                </Button>
                <Button variant="outline" className="bg-transparent">
                  <Share2 className="mr-2 h-4 w-4" />
                  Share with Patient
                </Button>
                <Button
                  variant="outline"
                  className="bg-transparent"
                  onClick={() => router.push("/dentist")}
                >
                  Start New Simulation
                </Button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
