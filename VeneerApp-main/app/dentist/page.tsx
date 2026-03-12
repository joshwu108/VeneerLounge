"use client"

import { useState } from "react"
import { ImageUpload } from "@/components/image-upload"
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"
import { useRouter } from "next/navigation"

export default function DentistPage() {
  const [selectedImage, setSelectedImage] = useState<string>("")
  const [imageFile, setImageFile] = useState<File | null>(null)
  const router = useRouter()

  const handleImageSelect = (file: File, preview: string) => {
    setImageFile(file)
    setSelectedImage(preview)
  }

  const handleClearImage = () => {
    setImageFile(null)
    setSelectedImage("")
  }

  const handleContinue = () => {
    if (selectedImage) {
      localStorage.setItem("uploadedPatientImage", selectedImage);
    }
    router.push("/dentist/simulation")
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-center px-4 py-4 sm:px-6 lg:px-8">
          <div className="text-xl font-bold">
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              VeneerVision AI
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-4xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="space-y-8">
          <div className="text-center">
            <h1 className="mb-3 text-4xl font-bold">Professional Veneer Simulation</h1>
            <p className="text-muted-foreground">
              Upload patient photos and generate up to 4 AI-powered veneer variations
            </p>
          </div>

          <ImageUpload onImageSelect={handleImageSelect} selectedImage={selectedImage} onClear={handleClearImage} />

          <div className="flex justify-center">
            <Button className="min-w-[300px]" disabled={!selectedImage} onClick={handleContinue}>
              Continue to Simulation
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      </main>
    </div>
  )
}
