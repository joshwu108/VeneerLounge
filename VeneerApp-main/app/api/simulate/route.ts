import { type NextRequest, NextResponse } from "next/server"
import http from "node:http"

// Allow up to 10 minutes for veneer generation on CPU
export const maxDuration = 600

/**
 * Make a POST request using Node.js http module with configurable timeout.
 * Node.js native fetch has a hard 5-minute headersTimeout that can't be
 * overridden, which is too short for CPU-based SDXL inference (~5-6 min).
 */
function postJSON(url: string, body: object, timeoutMs: number): Promise<{ status: number; data: Record<string, unknown> }> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url)
    const payload = JSON.stringify(body)

    const req = http.request(
      {
        hostname: parsed.hostname,
        port: parsed.port,
        path: parsed.pathname,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
        timeout: timeoutMs,
      },
      (res) => {
        const chunks: Buffer[] = []
        res.on("data", (chunk: Buffer) => chunks.push(chunk))
        res.on("end", () => {
          const raw = Buffer.concat(chunks).toString("utf-8")
          try {
            const data = JSON.parse(raw)
            resolve({ status: res.statusCode ?? 500, data })
          } catch {
            reject(new Error(`Invalid JSON from backend: ${raw.slice(0, 200)}`))
          }
        })
      }
    )

    req.on("timeout", () => {
      req.destroy()
      reject(new Error("Backend request timed out"))
    })
    req.on("error", reject)
    req.write(payload)
    req.end()
  })
}

export async function POST(request: NextRequest) {
  try {
    const { image, boundingBox } = await request.json()
    if (!image) {
      return NextResponse.json({ error: "Image is required" }, { status: 400 })
    }

    const backendUrl = process.env.BACKEND_URL || "http://localhost:8000"

    const { status, data } = await postJSON(
      `${backendUrl}/api/veneer-preview`,
      {
        image: image,
        intensity: 0.75,
        preserve_geometry: true,
        bounding_box: boundingBox,
      },
      600_000 // 10 minutes
    )

    if (status !== 200) {
      throw new Error((data.error as string) || "Backend Veneer Generation failed")
    }

    return NextResponse.json({ output: data.output })
  } catch (error: unknown) {
    console.error("Simulation error:", error)
    const errorMessage = error instanceof Error ? error.message : "Failed to generate simulation"
    return NextResponse.json({ error: errorMessage }, { status: 500 })
  }
}
