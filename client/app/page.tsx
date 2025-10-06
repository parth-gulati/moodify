"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Sparkles, Music2, Info } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

// FIRST RUN: npx shadcn-ui@latest add tooltip

interface Song {
  name: string
  artist: string
  uri: string
  labels: number
  popularity?: number
  acousticness?: number
  danceability?: number
  energy?: number
  source?: string
  llm_description?: string  // Added LLM description field
}

interface AnalysisResult {
  detected_emotion: string
  confidence: number
  mood: string
  mood_label: number
  recommended_songs: Song[]
  total_recommendations: number
}

export default function LiftMyMoodPage() {
  const [mousePosition, setMousePosition] = useState({ x: 50, y: 50 })
  const [inputText, setInputText] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Track mouse movement for gradient effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        const x = ((e.clientX - rect.left) / rect.width) * 100
        const y = ((e.clientY - rect.top) / rect.height) * 100
        setMousePosition({ x, y })
      }
    }

    window.addEventListener("mousemove", handleMouseMove)
    return () => window.removeEventListener("mousemove", handleMouseMove)
  }, [])

  const analyzeMood = async () => {
    if (!inputText.trim()) return

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: inputText,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to analyze mood")
      }

      const data = await response.json()
      console.log("Received data:", data) // Debug log
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong")
      console.error("Error analyzing mood:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isLoading) {
      analyzeMood()
    }
  }

  const getMoodEmoji = (mood: string) => {
    const emojis: Record<string, string> = {
      happy: "😊",
      sad: "😢",
      energetic: "⚡",
      calm: "😌",
    }
    return emojis[mood] || "🎵"
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div
        ref={containerRef}
        className="relative min-h-screen w-full overflow-hidden"
        style={{
          background: `radial-gradient(circle at ${mousePosition.x}% ${mousePosition.y}%, 
            oklch(0.35 0.12 180) 0%, 
            oklch(0.25 0.08 180) 25%, 
            oklch(0.15 0.04 180) 50%, 
            oklch(0.08 0.02 180) 100%)`,
          transition: "background 0.3s ease-out",
        }}
      >
        {/* Ambient glow effects */}
        <div className="pointer-events-none absolute inset-0">
          <div
            className="absolute h-[500px] w-[500px] rounded-full opacity-20 blur-[100px]"
            style={{
              background: "oklch(0.55 0.18 180)",
              left: `${mousePosition.x}%`,
              top: `${mousePosition.y}%`,
              transform: "translate(-50%, -50%)",
              transition: "all 0.5s ease-out",
            }}
          />
        </div>

        <main className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4 py-12 sm:px-6 lg:px-8">
          <div className="w-full max-w-3xl space-y-8 text-center">
            {/* Main heading */}
            <div className="space-y-4 animate-fade-in-up">
              <h1
                className="font-bebas text-7xl tracking-wider text-white sm:text-8xl md:text-9xl"
                style={{ textShadow: "0 0 30px rgba(255, 255, 255, 0.4), 0 0 60px rgba(255, 255, 255, 0.2)" }}
              >
                Lift myMood
              </h1>
              <p className="text-balance text-lg text-white/80 sm:text-xl">
                Tell us how you feel, and we'll find the perfect soundtrack for your soul
              </p>
            </div>

            {/* Input section with glass morphism */}
            <div
              className="animate-fade-in-up space-y-4 rounded-2xl border border-border/50 p-6 backdrop-blur-xl sm:p-8"
              style={{
                background: "oklch(0.18 0.03 180 / 0.3)",
                boxShadow: "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
                animationDelay: "0.1s",
                animationFillMode: "backwards",
              }}
            >
              <div className="flex flex-col gap-3 sm:flex-row">
                <Input
                  type="text"
                  placeholder="How are you feeling today?"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  disabled={isLoading}
                  className="h-12 flex-1 border-border/50 bg-secondary/50 text-base text-white placeholder:text-white/50 backdrop-blur-sm transition-all focus:border-primary/50 focus:bg-secondary/70 sm:h-14 sm:text-lg"
                />
                <Button
                  onClick={analyzeMood}
                  disabled={isLoading || !inputText.trim()}
                  className="h-12 gap-2 bg-primary/90 px-6 text-base font-medium text-primary-foreground backdrop-blur-sm transition-all hover:bg-primary hover:shadow-lg hover:shadow-primary/20 sm:h-14 sm:px-8 sm:text-lg"
                >
                  {isLoading ? (
                    <>
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-5 w-5" />
                      Discover
                    </>
                  )}
                </Button>
              </div>

              {error && <div className="rounded-lg bg-destructive/10 p-4 text-sm text-destructive">{error}</div>}
            </div>

            {/* Results section */}
            {result && (
              <div
                className="animate-fade-in-up space-y-6"
                style={{
                  animationDelay: "0.2s",
                  animationFillMode: "backwards",
                }}
              >
                {/* Mood detection card */}
                <div
                  className="rounded-xl border border-border/50 p-6 backdrop-blur-xl"
                  style={{
                    background: "oklch(0.18 0.03 180 / 0.3)",
                    boxShadow: "0 4px 24px 0 rgba(0, 0, 0, 0.25)",
                  }}
                >
                  <div className="flex flex-col items-center gap-3">
                    <span className="text-5xl">{getMoodEmoji(result.mood)}</span>
                    <div className="flex flex-col items-center gap-1 sm:flex-row sm:gap-2">
                      <span className="text-sm text-white/70">Detected mood:</span>
                      <span className="text-sm font-semibold text-white/80 sm:text-sm uppercase">{result.mood}</span>
                      <span className="text-sm text-white/70">({Math.round(result.confidence * 100)}% confident)</span>
                    </div>
                    <span className="text-xs text-white/60">Emotion: {result.detected_emotion}</span>
                  </div>
                </div>

                {/* Song recommendations */}
                <div className="space-y-3">
                  <h2 className="flex items-center justify-center gap-2 text-xl font-semibold text-white sm:text-2xl">
                    <Music2 className="h-6 w-6 text-primary" />
                    Your Personalized Playlist
                  </h2>
                  <div className="grid gap-3 sm:grid-cols-2">
                    {result.recommended_songs.map((song, index) => (
                      <Tooltip key={song.uri || index}>
                        <TooltipTrigger asChild>
                          <div
                            className="animate-fade-in-up group rounded-lg border border-border/30 p-4 backdrop-blur-md transition-all hover:border-primary/50 hover:bg-secondary/30 cursor-pointer"
                            style={{
                              background: "oklch(0.18 0.03 180 / 0.2)",
                              animationDelay: `${0.3 + index * 0.05}s`,
                              animationFillMode: "backwards",
                            }}
                          >
                            <div className="flex items-start gap-3">
                              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/20 text-sm font-semibold text-primary">
                                {index + 1}
                              </div>
                              <div className="min-w-0 flex-1 text-left">
                                <div className="flex items-center gap-1.5">
                                  <p className="truncate font-medium text-white group-hover:text-primary transition-colors">
                                    {song.name}
                                  </p>
                                  {song.llm_description && (
                                    <Info className="h-3.5 w-3.5 text-white/40 shrink-0" />
                                  )}
                                </div>
                                <p className="truncate text-sm text-white/70">{song.artist}</p>
                              </div>
                              {song.uri && (
                                <a
                                  href={`https://open.spotify.com/track/${song.uri.split(":")[2]}`}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="shrink-0 rounded-full bg-green-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-green-700 transition-colors flex items-center gap-1"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  ▶ Play
                                </a>
                              )}
                            </div>
                          </div>
                        </TooltipTrigger>
                        {song.llm_description && (
                          <TooltipContent
                            side="top"
                            className="max-w-xs border-white/20 bg-black text-white"
                          >
                            <p className="text-sm">{song.llm_description}</p>
                          </TooltipContent>
                        )}
                      </Tooltip>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Empty state hint */}
            {!result && !isLoading && (
              <p
                className="animate-fade-in-up text-sm text-white/60"
                style={{ animationDelay: "0.3s", animationFillMode: "backwards" }}
              >
                Share your thoughts, feelings, or current state of mind...
              </p>
            )}
          </div>
        </main>
      </div>
    </TooltipProvider>
  )
}