import { type NextRequest, NextResponse } from "next/server"

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:5000"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    console.log("[v0] BACKEND_URL:", BACKEND_URL)
    console.log("[v0] Full URL:", `${BACKEND_URL}/api/analyze`)
    console.log("[v0] Request body:", body)

    const response = await fetch(`${BACKEND_URL}/api/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    console.log("[v0] Backend response status:", response.status)

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`)
    }

    const data = await response.json()
    console.log("[v0] Backend response data:", data)
    return NextResponse.json(data)
  } catch (error) {
    console.error("[v0] Error proxying to backend:", error)
    return NextResponse.json({ error: "Failed to analyze mood" }, { status: 500 })
  }
}
