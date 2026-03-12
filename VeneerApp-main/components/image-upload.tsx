"use client"

import type React from "react"

import { useCallback, useState, useRef } from "react"
import { Upload, Camera, X, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ImageUploadProps {
  onImageSelect: (file: File, preview: string) => void
  selectedImage?: string
  onClear?: () => void
}

export function ImageUpload({ onImageSelect, selectedImage, onClear }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const compressImage = (file: File): Promise<string> => {
    return new Promise((resolve) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const img = new Image()
        img.onload = () => {
          const canvas = document.createElement('canvas')
          let width = img.width
          let height = img.height

          // Resize to max 800px on longest side
          const maxSize = 800
          if (width > height && width > maxSize) {
            height = (height * maxSize) / width
            width = maxSize
          } else if (height > maxSize) {
            width = (width * maxSize) / height
            height = maxSize
          }

          canvas.width = width
          canvas.height = height
          const ctx = canvas.getContext('2d')
          if (ctx) {
            ctx.drawImage(img, 0, 0, width, height)
            // Compress to 70% quality JPEG
            resolve(canvas.toDataURL('image/jpeg', 0.7))
          }
        }
        img.src = e.target?.result as string
      }
      reader.readAsDataURL(file)
    })
  }

  const handleFile = useCallback(
    async (file: File) => {
      if (file && file.type.startsWith("image/")) {
        const compressed = await compressImage(file)
        onImageSelect(file, compressed)
      }
    },
    [onImageSelect],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      handleFile(file)
    },
    [handleFile],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile],
  )

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      })
      streamRef.current = stream
      setShowCamera(true)

      // Wait for the video element to be rendered before setting srcObject
      setTimeout(() => {
        if (videoRef.current && streamRef.current) {
          videoRef.current.srcObject = streamRef.current
          videoRef.current.play().catch(err => {
            console.error("Error playing video:", err)
          })
        }
      }, 100)
    } catch (error) {
      console.error("Error accessing camera:", error)
      alert("Unable to access camera. Please check permissions and ensure you're using HTTPS or localhost.")
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setShowCamera(false)
  }

  const capturePhoto = () => {
    if (videoRef.current && videoRef.current.videoWidth > 0) {
      const canvas = document.createElement("canvas")
      let width = videoRef.current.videoWidth
      let height = videoRef.current.videoHeight

      console.log("Video dimensions:", width, height)

      if (width === 0 || height === 0) {
        alert("Camera not ready yet. Please wait a moment and try again.")
        return
      }

      // Resize to max 800px
      const maxSize = 800
      if (width > height && width > maxSize) {
        height = (height * maxSize) / width
        width = maxSize
      } else if (height > maxSize) {
        width = (width * maxSize) / height
        height = maxSize
      }

      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext("2d")
      if (ctx) {
        ctx.drawImage(videoRef.current, 0, 0, width, height)
        // Compress to 70% quality
        const preview = canvas.toDataURL("image/jpeg", 0.7)
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], "camera-capture.jpg", { type: "image/jpeg" })
            onImageSelect(file, preview)
            stopCamera()
          }
        }, "image/jpeg", 0.7)
      }
    } else {
      alert("Camera not ready yet. Please wait a moment and try again.")
    }
  }

  if (showCamera) {
    return (
      <div className="glass relative overflow-hidden rounded-2xl p-6">
        <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-black">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="h-full w-full object-cover"
          />
        </div>
        <div className="mt-4 flex gap-3">
          <Button onClick={capturePhoto} className="flex-1">
            <Camera className="mr-2 h-4 w-4" />
            Capture Photo
          </Button>
          <Button onClick={stopCamera} variant="outline" className="flex-1">
            Cancel
          </Button>
        </div>
      </div>
    )
  }

  if (selectedImage) {
    return (
      <div className="glass relative overflow-hidden rounded-2xl p-6">
        <div className="relative">
          <img src={selectedImage || "/placeholder.svg"} alt="Selected smile" className="w-full rounded-xl" />
          {onClear && (
            <Button
              onClick={onClear}
              variant="destructive"
              size="icon"
              className="absolute right-2 top-2 h-8 w-8 rounded-full"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={cn(
          "glass glass-shine relative cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed p-12 text-center transition-all",
          isDragging ? "border-primary bg-primary/10" : "border-border hover:border-primary/50",
        )}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className="hidden"
          aria-label="Upload image"
        />

        <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/20">
          <ImageIcon className="h-8 w-8 text-primary" />
        </div>

        <h3 className="mb-2 text-xl font-semibold">Upload Your Smile Photo</h3>
        <p className="mb-4 text-sm text-muted-foreground">
          Drag & drop or click to select an image
          <br />
          <span className="text-xs">Supports JPG, PNG (Max 10MB)</span>
        </p>

        <div className="flex items-center justify-center gap-2">
          <Upload className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Choose File</span>
        </div>
      </div>

      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-border" />
        </div>
        <div className="relative flex justify-center text-xs uppercase">
          <span className="bg-background px-2 text-muted-foreground">Or</span>
        </div>
      </div>

      <Button onClick={startCamera} variant="outline" className="w-full bg-transparent">
        <Camera className="mr-2 h-4 w-4" />
        Take Photo with Camera
      </Button>
    </div>
  )
}
