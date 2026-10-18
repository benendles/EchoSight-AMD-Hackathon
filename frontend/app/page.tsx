"use client";

import {
  ChangeEvent,
  CSSProperties,
  DragEvent,
  SyntheticEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { apiFetch, getSession, resetSession } from "../lib/session";
import { VisionConnection } from "../lib/vision-connection";
import { VoiceToolResults } from "../lib/voice-tools";

const BACKEND_URL = (
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"
).replace(/\/$/, "");
const VISION_WS_URL = `${BACKEND_URL.replace(/^http/, "ws")}/ws/vision`;
const ASSEMBLYAI_WS_URL = "wss://agents.assemblyai.com/v1/ws";

type VoiceState =
  | "idle"
  | "connecting"
  | "listening"
  | "thinking"
  | "speaking"
  | "error";

type Detection = {
  label?: string;
  class?: string;
  confidence?: number;
  distance?: number | string;
  distance_meters?: number;
  direction?: string;
  box?: number[];
  [key: string]: unknown;
};

type Navigation = {
  action?: string;
  direction?: string;
  distance?: number | string;
  message?: string;
  [key: string]: unknown;
};

type VisionResult = {
  type?: string;
  answer?: string;
  description?: string;
  scene_description?: string;
  scene?: Detection[];
  detections?: Detection[];
  navigation?: Navigation | null;
  result?: string | VisionResult;
  [key: string]: unknown;
};

type ConversationMessage = {
  id: number;
  role: "user" | "assistant" | "system";
  text: string;
};

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;

  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }

  return btoa(binary);
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  return bytes.buffer;
}

function float32ToPCM16(input: Float32Array): ArrayBuffer {
  const output = new Int16Array(input.length);

  for (let i = 0; i < input.length; i++) {
    const sample = Math.max(-1, Math.min(1, input[i]));

    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }

  return output.buffer;
}

export default function Home() {
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");

  const [connected, setConnected] = useState(false);

  const [voiceConnected, setVoiceConnected] = useState(false);

  const [cameraActive, setCameraActive] = useState(false);

  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [dragging, setDragging] = useState(false);

  const [imageUrl, setImageUrl] = useState<string | null>(null);

  const [filename, setFilename] = useState("");

  const [imageDimensions, setImageDimensions] = useState({
    width: 0,
    height: 0,
  });

  const [lastMessage, setLastMessage] = useState("Ready to assist.");

  const [conversation, setConversation] = useState<ConversationMessage[]>([]);

  const [scene, setScene] = useState("");
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [cloudStatus, setCloudStatus] = useState("");

  const [navigation, setNavigation] = useState<Navigation | null>(null);

  const [sceneObjects, setSceneObjects] = useState<Detection[]>([]);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);

  const imageRef = useRef<HTMLImageElement | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const visionConnectionRef = useRef<VisionConnection | null>(null);
  const cameraRequestedRef = useRef(false);
  const cameraStartingRef = useRef(false);
  const latestObservationRef = useRef<string | null>(null);
  const navigationExpiryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const uploadControllerRef = useRef<AbortController | null>(null);
  const questionControllerRef = useRef<AbortController | null>(null);
  const hazardUntilRef = useRef(0);
  const lastHazardRef = useRef("");
  const lastHazardAtRef = useRef(0);

  const assemblyWsRef = useRef<WebSocket | null>(null);

  const cameraStreamRef = useRef<MediaStream | null>(null);

  const microphoneStreamRef = useRef<MediaStream | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);

  const microphoneSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const processorRef = useRef<ScriptProcessorNode | null>(null);

  const imageUrlRef = useRef<string | null>(null);

  const analysisIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null,
  );

  const playbackTimeRef = useRef(0);

  const playbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const messageIdRef = useRef(0);

  const mountedRef = useRef(true);

  const voiceStateRef = useRef<VoiceState>("idle");
  const voiceGenerationRef = useRef(0);
  const voiceStartingRef = useRef(false);
  const voiceToolsRef = useRef<VoiceToolResults | null>(null);

  const addMessage = useCallback(
    (role: ConversationMessage["role"], text: string) => {
      if (!text.trim()) return;

      setConversation((prev) => [
        ...prev.slice(-30),
        {
          id: ++messageIdRef.current,
          role,
          text: text.trim(),
        },
      ]);
    },
    [],
  );

  /* VISION RESULT NORMALIZATION */

  const applyVisionResult = useCallback((rawResult: VisionResult) => {
    let result: VisionResult = rawResult;

    // Normalize every response shape used by EchoSight.
    if (rawResult?.result !== undefined) {
      if (typeof rawResult.result === "string") {
        try {
          result = JSON.parse(rawResult.result) as VisionResult;
        } catch {
          // AssemblyAI/tool responses can contain plain text.
          result = {
            ...rawResult,
            answer: rawResult.result,
          };
        }
      } else if (rawResult.result && typeof rawResult.result === "object") {
        result = rawResult.result as VisionResult;
      }
    }

    if (result.type === "error") {
      setIsAnalyzing(false);
      setLastMessage(typeof result.message === "string" ? result.message : "Analysis failed. Please try again.");
      return;
    }
    if (result.stale) {
      setIsAnalyzing(false);
      setNavigation(result.navigation || { action: "CAUTION", message: "A more recent camera view is needed." });
      setLastMessage(typeof result.answer === "string" ? result.answer : "A more recent camera view is needed.");
      return;
    }

    const isCloud = Array.isArray(result.sources) || result.cloud !== undefined;
    const observationId = typeof result.observation_id === "string" ? result.observation_id : null;
    const superseded = isCloud && observationId !== latestObservationRef.current;
    if (isCloud) {
      const sources = Array.isArray(result.sources) ? result.sources : [];
      setCloudStatus([
        sources.includes("amd") ? "AMD Vision completed" : "AMD Vision unavailable or not requested",
        sources.includes("evolus") ? "Evolus answer completed" : "Using local scene description",
      ].join(" · "));
    } else if (observationId) {
      latestObservationRef.current = observationId;
    }

    const objects = Array.isArray(result.scene)
      ? result.scene
      : Array.isArray(result.detections)
        ? result.detections
        : [];

    const description =
      typeof result.description === "string" && result.description.trim()
        ? result.description.trim()
        : typeof result.scene_description === "string" &&
            result.scene_description.trim()
          ? result.scene_description.trim()
          : typeof result.answer === "string"
            ? result.answer.trim()
            : "";

    const nav =
      result.navigation && typeof result.navigation === "object"
        ? result.navigation
        : null;

    // A delayed conversation answer may refer to an older image. It must not
    // replace the current camera scene or its more recent local warning.
    if (!superseded) {
      setScene(isCloud && typeof result.answer === "string" ? result.answer : description);
      setSceneObjects(objects);
      setNavigation(nav);
      if (navigationExpiryRef.current) clearTimeout(navigationExpiryRef.current);
      if (result.source === "camera") {
        const lifetime = typeof result.navigation_valid_for_seconds === "number"
          ? result.navigation_valid_for_seconds * 1000 : 0;
        navigationExpiryRef.current = setTimeout(() => {
          setNavigation({ action: "CAUTION", message: "I need a more recent camera view to assess what's ahead." });
        }, lifetime);
      }
      if (nav?.action === "STOP" && typeof nav.message === "string") {
        hazardUntilRef.current = Date.now() + 5000;
        // Stop queued conversational audio before announcing the local warning.
        for (const source of audioSourcesRef.current) {
          try { source.stop(); } catch { /* Already ended. */ }
        }
        audioSourcesRef.current.clear();
        playbackTimeRef.current = audioContextRef.current?.currentTime || 0;
        if ("speechSynthesis" in window && (lastHazardRef.current !== nav.message || Date.now() - lastHazardAtRef.current > 10000)) {
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(new SpeechSynthesisUtterance(nav.message));
          lastHazardRef.current = nav.message;
          lastHazardAtRef.current = Date.now();
        }
      }
    }
    setIsAnalyzing(false);

    if (typeof result.answer === "string" && result.answer.trim()) {
      setLastMessage(result.answer.trim());
    } else if (description) {
      setLastMessage(description);
    } else if (nav?.message) {
      setLastMessage(String(nav.message));
    } else if (objects.length > 0) {
      setLastMessage(
        `Detected ${objects.length} object${
          objects.length === 1 ? "" : "s"
        } in the scene.`,
      );
    } else {
      setLastMessage("Scene analyzed, but no visual information was returned.");
    }
  }, []);

  /* ANALYZE CURRENT FRAME */

  const analyzeCurrentFrame = useCallback(
    async (
      command = "Describe what is visible around me and identify obstacles or useful directions.",
    ) => {
      setIsAnalyzing(true);
      questionControllerRef.current?.abort();
      const controller = new AbortController();
      questionControllerRef.current = controller;
      const deadline = setTimeout(() => controller.abort(), 35000);

      try {
        const response = await apiFetch(BACKEND_URL, "/api/voice/analyze", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            command,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Vision request failed: ${response.status}`);
        }

        const rawResult = (await response.json()) as VisionResult;

        // All response normalization happens in one place.
        if (mountedRef.current) applyVisionResult(rawResult);

        return rawResult;
      } catch (error) {
        console.error("[EchoSight] Vision analysis error:", error);

        setIsAnalyzing(false);

        setLastMessage("I could not analyze the current camera frame.");

        return {
          error: "Vision analysis failed.",
        };
      } finally {
        clearTimeout(deadline);
      }
    },
    [applyVisionResult],
  );

  /* Camera transport recovers from backend errors, closures and timeouts. */
  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!cameraRequestedRef.current || !video || !canvas || video.readyState < 2) return;
    const scale = Math.min(1, 960 / Math.max(video.videoWidth, 1));
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    visionConnectionRef.current?.sendFrame(canvas.toDataURL("image/jpeg", 0.72).split(",")[1]);
  }, []);

  const stopCamera = useCallback(() => {
    cameraRequestedRef.current = false;
    if (navigationExpiryRef.current) clearTimeout(navigationExpiryRef.current);
    navigationExpiryRef.current = null;
    if (analysisIntervalRef.current) clearInterval(analysisIntervalRef.current);
    analysisIntervalRef.current = null;
    visionConnectionRef.current?.stop();
    visionConnectionRef.current = null;
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setConnected(false);
    setNavigation({ action: "CAUTION", message: "Camera stopped. A live route cannot be assessed." });
  }, []);

  const startCamera = useCallback(async () => {
    if (cameraRequestedRef.current || cameraStartingRef.current) return;
    cameraRequestedRef.current = true;
    cameraStartingRef.current = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (!mountedRef.current || !cameraRequestedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      uploadControllerRef.current?.abort();
      cameraStreamRef.current = stream;
      latestObservationRef.current = null;
      setCameraActive(true);
      setCloudStatus("");
      setNavigation({ action: "CAUTION", message: "Waiting for a recent camera view." });
      setLastMessage("Camera active. Connecting to local vision...");
      const connection = new VisionConnection({
        url: VISION_WS_URL,
        session: () => getSession(BACKEND_URL),
        resetSession,
        onConnected: (active) => { if (mountedRef.current) setConnected(active); },
        onResult: (result) => { if (mountedRef.current) applyVisionResult(result as VisionResult); },
        onError: (message) => {
          if (!mountedRef.current) return;
          setIsAnalyzing(false);
          setLastMessage(message);
        },
      });
      visionConnectionRef.current = connection;
      connection.start();
      analysisIntervalRef.current = setInterval(sendFrame, 1500);
    } catch {
      cameraRequestedRef.current = false;
      setLastMessage("Camera permission was denied or the camera is unavailable.");
    } finally {
      cameraStartingRef.current = false;
    }
  }, [applyVisionResult, sendFrame]);

  // The video element is mounted only after cameraActive becomes true.
  useEffect(() => {
    if (!cameraActive || !videoRef.current || !cameraStreamRef.current) return;
    videoRef.current.srcObject = cameraStreamRef.current;
    void videoRef.current.play().catch(() => {
      setLastMessage("Could not play the camera feed. Stop and restart the camera.");
    });
  }, [cameraActive]);

  /* ASSEMBLYAI MESSAGE HANDLER */

  const handleAssemblyAIMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        voiceToolsRef.current?.receive(message);

        switch (message.type) {
          case "session.ready":
            voiceStateRef.current = "listening";
            setVoiceState("listening");

            setVoiceConnected(true);

            setLastMessage("Listening. Ask me what you see.");

            break;

          case "transcript.user.delta":
            if (message.text) {
              setLastMessage(message.text);
            }

            break;

          case "transcript.user":
            if (message.text) {
              addMessage("user", message.text);
            }

            break;

          case "reply.started":
            if (playbackTimerRef.current) {
              clearTimeout(playbackTimerRef.current);
              playbackTimerRef.current = null;
            }

            voiceStateRef.current = "speaking";
            setVoiceState("speaking");

            setLastMessage("EchoSight is speaking...");

            break;

          case "reply.audio": {
            if (Date.now() < hazardUntilRef.current) break;
            const encodedAudio =
              message.data || message.audio || message.audio_data;

            if (!encodedAudio) {
              console.warn("[EchoSight] Reply audio had no audio payload.");
              break;
            }

            const context = audioContextRef.current;

            if (!context) {
              console.warn(
                "[EchoSight] Reply audio arrived before AudioContext.",
              );
              break;
            }

            void context.resume();

            const pcm = new Int16Array(base64ToArrayBuffer(encodedAudio));

            const audioBuffer = context.createBuffer(1, pcm.length, 24000);

            const channel = audioBuffer.getChannelData(0);

            for (let i = 0; i < pcm.length; i++) {
              channel[i] = pcm[i] / 32768;
            }

            const source = context.createBufferSource();

            source.buffer = audioBuffer;

            source.connect(context.destination);

            audioSourcesRef.current.add(source);

            const now = context.currentTime;

            const startAt = Math.max(now + 0.005, playbackTimeRef.current);

            source.start(startAt);

            playbackTimeRef.current = startAt + audioBuffer.duration;

            source.onended = () => {
              audioSourcesRef.current.delete(source);
            };

            break;
          }

          case "transcript.agent":
            if (message.text) {
              addMessage("assistant", message.text);

              setLastMessage(message.text);
            }

            break;

          /*
           * AssemblyAI asks EchoSight
           * to analyze the current scene.
           */

          case "tool.call": {
            const callId = message.call_id;

            const name = message.name;

            const args = message.arguments || {};

            if (name !== "analyze_echo_sight_scene" || !callId) {
              break;
            }

            setVoiceState("thinking");

            voiceStateRef.current = "thinking";

            const command =
              typeof args.command === "string"
                ? args.command
                : "Describe the current scene.";

            voiceToolsRef.current?.register(callId);

            void analyzeCurrentFrame(command).then((result) => {
              const ws = assemblyWsRef.current;

              if (!ws || ws.readyState !== WebSocket.OPEN) {
                voiceStateRef.current = "error";
                setVoiceState("error");
                setLastMessage("Voice connection was lost during scene analysis.");
                return;
              }

              voiceToolsRef.current?.complete(callId, result);
            });

            break;
          }

          case "reply.done": {
            const status = message.status;

            if (status === "interrupted") {
              if (playbackTimerRef.current) {
                clearTimeout(playbackTimerRef.current);
                playbackTimerRef.current = null;
              }

              for (const source of audioSourcesRef.current) {
                try {
                  source.stop();
                } catch {
                  // A source may already have finished.
                }
              }

              audioSourcesRef.current.clear();

              playbackTimeRef.current =
                audioContextRef.current?.currentTime || 0;

              voiceStateRef.current = "listening";
              setVoiceState("listening");

              break;
            }

            const context = audioContextRef.current;
            const remainingPlaybackMs = context
              ? Math.max(
                  0,
                  (playbackTimeRef.current - context.currentTime) * 1000,
                )
              : 0;

            if (remainingPlaybackMs > 30) {
              voiceStateRef.current = "speaking";
              setVoiceState("speaking");
              playbackTimerRef.current = setTimeout(() => {
                playbackTimerRef.current = null;

                if (
                  mountedRef.current &&
                  voiceStateRef.current === "speaking"
                ) {
                  voiceStateRef.current = "listening";
                  setVoiceState("listening");
                }
              }, remainingPlaybackMs + 30);
            } else if (voiceStateRef.current !== "thinking") {
              voiceStateRef.current = "listening";
              setVoiceState("listening");
            }

            break;
          }

          case "session.error":
            console.error("[EchoSight] AssemblyAI session error:", message);

            setVoiceState("error");

            setLastMessage(message.message || "AssemblyAI session error.");

            break;

          default:
            break;
        }
      } catch (error) {
        console.error("[EchoSight] AssemblyAI message error:", error);
      }
    },
    [addMessage, analyzeCurrentFrame],
  );

  /* STOP VOICE */

  const stopVoice = useCallback(() => {
    voiceGenerationRef.current++;
    voiceStartingRef.current = false;
    voiceToolsRef.current?.clear();
    voiceToolsRef.current = null;
    questionControllerRef.current?.abort();
    if (playbackTimerRef.current) clearTimeout(playbackTimerRef.current);
    playbackTimerRef.current = null;
    audioSourcesRef.current.clear();
    processorRef.current?.disconnect();

    processorRef.current = null;

    microphoneSourceRef.current?.disconnect();

    microphoneSourceRef.current = null;

    microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());

    microphoneStreamRef.current = null;

    if (audioContextRef.current) {
      void audioContextRef.current.close();

      audioContextRef.current = null;
    }

    playbackTimeRef.current = 0;

    if (assemblyWsRef.current) {
      if (assemblyWsRef.current.readyState === WebSocket.OPEN) {
        assemblyWsRef.current.send(JSON.stringify({ type: "session.end" }));
      }
      assemblyWsRef.current.close();

      assemblyWsRef.current = null;
    }

    setVoiceConnected(false);

    setVoiceState("idle");
  }, []);

  /* START VOICE */

  const startVoice = useCallback(async () => {
    if (
      voiceStartingRef.current ||
      voiceState === "connecting" ||
      voiceState === "listening" ||
      voiceState === "thinking" ||
      voiceState === "speaking"
    ) {
      return;
    }

    voiceStartingRef.current = true;
    const generation = ++voiceGenerationRef.current;

    setVoiceState("connecting");

    setLastMessage("Connecting to EchoSight voice assistant...");

    try {
      // Create and resume playback in the initiating user gesture.
      const context = new AudioContext({ sampleRate: 24000 });
      audioContextRef.current = context;
      await context.resume();
      const tokenResponse = await apiFetch(BACKEND_URL, "/api/voice-token");

      if (!tokenResponse.ok) {
        throw new Error(`Token request failed: ${tokenResponse.status}`);
      }

      const tokenData = await tokenResponse.json();

      if (!mountedRef.current || generation !== voiceGenerationRef.current) return;

      if (!tokenData.token) {
        throw new Error("No temporary AssemblyAI token returned.");
      }

      const ws = new WebSocket(
        `${ASSEMBLYAI_WS_URL}?token=${encodeURIComponent(tokenData.token)}`,
      );

      assemblyWsRef.current = ws;
      voiceToolsRef.current = new VoiceToolResults((message) => {
        if (generation === voiceGenerationRef.current && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(message));
        }
      });

      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        if (generation !== voiceGenerationRef.current) { ws.close(); return; }
        ws.send(
          JSON.stringify({
            type: "session.update",

            session: {
              system_prompt:
                "You are EchoSight, an accessibility assistant for a blind or visually impaired person. Use analyze_echo_sight_scene for every visual or navigation question. Its result may contain a serialized result: speak the answer field faithfully in 1-2 sentences. Prioritize STOP and CAUTION messages. Never invent visual information, distances, features, or a safe crossing. Never replace uncertainty with a movement instruction.",

              greeting: "EchoSight is ready. Tell me what you need help with.",

              output: {
                voice: "ivy",
              },

              tools: [
                {
                  type: "function",

                  name: "analyze_echo_sight_scene",
                  execution_mode: "interactive",
                  timeout_seconds: 40,

                  description:
                    "Analyze the current EchoSight camera frame with object detection, depth estimation, spatial reasoning, and navigation. Use this whenever the user asks about what they see, objects or people around them, obstacles, directions, or whether it is safe to move.",

                  parameters: {
                    type: "object",

                    properties: {
                      command: {
                        type: "string",

                        description:
                          "The user's visual or navigation question, in natural language.",
                      },
                    },

                    required: ["command"],
                  },
                },
              ],

              input: {
                turn_detection: {
                  vad_threshold: 0.45,

                  min_silence: 700,

                  max_silence: 3000,

                  interrupt_response: true,
                },
              },
            },
          }),
        );

        console.log("[EchoSight] AssemblyAI session.update sent");
      };

      ws.onmessage = (event) => {
        if (generation === voiceGenerationRef.current) handleAssemblyAIMessage(event);
      };

      ws.onerror = (event) => {
        if (generation !== voiceGenerationRef.current) return;
        console.error("[EchoSight] AssemblyAI WebSocket error:", event);

        stopVoice();

        setVoiceState("error");

        setLastMessage("Voice connection error.");
      };

      ws.onclose = () => {
        if (generation !== voiceGenerationRef.current) return;
        stopVoice();

        if (mountedRef.current) {
          setVoiceState("idle");
        }
      };
    } catch (error) {
      if (generation !== voiceGenerationRef.current || !mountedRef.current) return;
      console.error("[EchoSight] Voice start error:", error);

      stopVoice();

      setVoiceState("error");

      setLastMessage("Could not connect to the voice assistant.");
    } finally {
      if (generation === voiceGenerationRef.current) voiceStartingRef.current = false;
    }
  }, [handleAssemblyAIMessage, stopVoice, voiceState]);

  useEffect(() => {
    voiceStateRef.current = voiceState;
  }, [voiceState]);

  /* MICROPHONE */

  useEffect(() => {
    if (!voiceConnected) {
      return;
    }

    let cancelled = false;

    const startMicrophone = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
          video: false,
        });

        if (cancelled || !mountedRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        microphoneStreamRef.current = stream;

        const context = audioContextRef.current;

        if (!context) {
          stream.getTracks().forEach((track) => track.stop());
          throw new Error("Audio playback context is unavailable.");
        }

        await context.resume();

        if (cancelled) {
          await context.close();
          return;
        }

        const source = context.createMediaStreamSource(stream);

        const processor = context.createScriptProcessor(2048, 1, 1);

        const silentGain = context.createGain();

        silentGain.gain.value = 0;

        microphoneSourceRef.current = source;
        processorRef.current = processor;

        processor.onaudioprocess = (event) => {
          const ws = assemblyWsRef.current;

          if (
            !ws ||
            ws.readyState !== WebSocket.OPEN ||
            voiceStateRef.current === "speaking" ||
            voiceStateRef.current === "thinking"
          ) {
            return;
          }

          const input = event.inputBuffer.getChannelData(0);

          const pcm = float32ToPCM16(input);

          ws.send(
            JSON.stringify({
              type: "input.audio",
              audio: arrayBufferToBase64(pcm),
            }),
          );
        };

        source.connect(processor);

        processor.connect(silentGain);

        silentGain.connect(context.destination);

        setLastMessage("Microphone active. Speak naturally.");
      } catch (error) {
        console.error("[EchoSight] Microphone error:", error);

        if (mountedRef.current) {
          setVoiceState("error");

          setLastMessage("Microphone permission was denied or unavailable.");
        }
      }
    };

    void startMicrophone();

    return () => {
      cancelled = true;

      processorRef.current?.disconnect();
      processorRef.current = null;

      microphoneSourceRef.current?.disconnect();
      microphoneSourceRef.current = null;

      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());

      microphoneStreamRef.current = null;

      const context = audioContextRef.current;

      audioContextRef.current = null;

      if (context) {
        void context.close();
      }
    };
  }, [voiceConnected]);

  /* CLEANUP */

  useEffect(() => {
    mountedRef.current = true;

    return () => {
      mountedRef.current = false;

      stopVoice();

      stopCamera();

      uploadControllerRef.current?.abort();
      questionControllerRef.current?.abort();
      if (navigationExpiryRef.current) clearTimeout(navigationExpiryRef.current);
      window.speechSynthesis?.cancel();

      if (imageUrlRef.current) {
        URL.revokeObjectURL(imageUrlRef.current);
      }
    };
  }, [stopCamera, stopVoice]);

  /* IMAGE UPLOAD - Important: Upload already returns the CV analysis. We do NOT run analyzeCurrentFrame again. */
  const handleImageFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        setLastMessage("Please select an image file.");

        return;
      }

      stopCamera();
      uploadControllerRef.current?.abort();
      const controller = new AbortController();
      uploadControllerRef.current = controller;
      setIsAnalyzing(true);
      setCloudStatus("");

      try {
        const url = URL.createObjectURL(file);

        if (imageUrlRef.current) {
          URL.revokeObjectURL(imageUrlRef.current);
        }

        imageUrlRef.current = url;

        setImageUrl(url);

        setFilename(file.name);

        const formData = new FormData();

        formData.append("file", file);

        const response = await apiFetch(BACKEND_URL, "/api/vision/upload", {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.status}`);
        }

        /*
         * Get the actual CV response.
         */

        const rawResult = (await response.json()) as VisionResult;

        console.log("[EchoSight] Upload vision result:", rawResult);

        if (rawResult.type === "error") {
          throw new Error(
            typeof rawResult.message === "string"
              ? rawResult.message
              : "The backend could not analyze this image.",
          );
        }

        // The upload endpoint already performs:
        // image → YOLO → depth → spatial reasoning → navigation
        // Therefore we consume its result directly and DO NOT
        // run /api/voice/analyze a second time.
        if (controller.signal.aborted || !mountedRef.current) return;
        applyVisionResult(rawResult);
        setLastMessage(`${file.name} loaded. Local analysis ready.`);
        setCloudStatus("Additional image analysis in progress...");
        // The local result is displayed before optional cloud enrichment starts.
        void apiFetch(BACKEND_URL, "/api/vision/enrich", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ observation_id: rawResult.observation_id }),
          signal: controller.signal,
        }).then(async (response) => {
          if (!response.ok) throw new Error("Cloud enrichment unavailable.");
          const enriched = await response.json();
          if (!controller.signal.aborted && mountedRef.current) applyVisionResult(enriched);
        }).catch(() => {
          if (!controller.signal.aborted && mountedRef.current) setCloudStatus("Additional analysis unavailable. Local results remain available.");
        });
      } catch (error) {
        if (controller.signal.aborted || !mountedRef.current) return;
        console.error("[EchoSight] Upload error:", error);

        setIsAnalyzing(false);

        setLastMessage("Could not analyze that image.");
      }
    },
    [applyVisionResult, stopCamera],
  );

  /* FILE INPUT */

  const handleFileInput = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (file) {
      void handleImageFile(file);
    }
  };

  /* DRAG AND DROP */

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();

    setDragging(false);

    const file = event.dataTransfer.files?.[0];

    if (file) {
      void handleImageFile(file);
    }
  };

  /* IMAGE LOAD */

  const handleImageLoad = (event: SyntheticEvent<HTMLImageElement>) => {
    const img = event.currentTarget;

    setImageDimensions({
      width: img.naturalWidth,
      height: img.naturalHeight,
    });
  };

  /* CLEAR IMAGE */

  const clearImage = () => {
    uploadControllerRef.current?.abort();
    questionControllerRef.current?.abort();
    if (navigationExpiryRef.current) clearTimeout(navigationExpiryRef.current);
    void apiFetch(BACKEND_URL, "/api/scene", { method: "DELETE" }).catch(() => {
      setLastMessage("Image cleared locally. The server observation expires with this session.");
    });
    latestObservationRef.current = null;
    setCloudStatus("");
    if (imageUrlRef.current) {
      URL.revokeObjectURL(imageUrlRef.current);
    }

    imageUrlRef.current = null;

    setImageUrl(null);

    setFilename("");

    setImageDimensions({
      width: 0,
      height: 0,
    });

    setScene("");

    setSceneObjects([]);

    setNavigation(null);

    setLastMessage("Image cleared.");
  };

  /* NAVIGATION TEXT */

  const navigationText =
    navigation?.message ||
    navigation?.action ||
    navigation?.direction ||
    (navigation?.distance !== undefined
      ? `Continue with caution. Distance: ${typeof navigation.distance === "number" ? `${navigation.distance} m` : navigation.distance}.`
      : "No navigation decision yet.");

  /* VOICE LABEL */

  const voiceLabel =
    voiceState === "listening"
      ? "Listening"
      : voiceState === "speaking"
        ? "Speaking"
        : voiceState === "thinking"
          ? "Thinking"
          : voiceState === "connecting"
            ? "Connecting"
            : voiceState === "error"
              ? "Error"
              : "Offline";

  const appStyle = {
    "--accent": "#49f08a",
  } as CSSProperties;

  /* UI */

  return (
    <main className="app-shell" style={appStyle}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">ES</div>

          <div>
            <h1>EchoSight</h1>

            <span>AI Vision Assistant</span>
          </div>
        </div>

        <div className="status-group">
          <div className={`status-pill ${cameraActive ? "active" : ""}`}>
            <span className="status-dot" />
            Camera {cameraActive ? "Active" : "Off"}
          </div>

          <div className={`status-pill ${voiceConnected ? "active" : ""}`}>
            <span className="status-dot" />
            Voice {voiceConnected ? "Connected" : "Offline"}
          </div>
        </div>
      </header>

      <section className="dashboard-grid">
        <div className="main-column">
          <section className="card camera-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">LIVE VISION</p>

                <h2>Camera View</h2>
              </div>

              <div className="heading-actions">
                <span className={`mini-status ${connected ? "green" : ""}`}>
                  <span className="status-dot" />

                  {connected ? "Vision online" : "Vision offline"}
                </span>

                {cameraActive ? (
                  <button className="btn btn-secondary" onClick={stopCamera}>
                    Stop camera
                  </button>
                ) : (
                  <button
                    className="btn btn-primary"
                    onClick={() => void startCamera()}
                  >
                    Start camera
                  </button>
                )}
              </div>
            </div>

            <div className="camera-stage">
              {cameraActive ? (
                <video
                  ref={videoRef}
                  className="camera-video"
                  muted
                  playsInline
                />
              ) : imageUrl ? (
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="Uploaded scene"
                  className="camera-image"
                  onLoad={handleImageLoad}
                />
              ) : (
                <div className="camera-empty">
                  <div className="camera-icon">◉</div>

                  <strong>No camera feed</strong>

                  <span>Start the camera or upload an image to begin.</span>
                </div>
              )}

              <div className="camera-overlay top-left">{cameraActive ? "LIVE" : "IMAGE"}</div>

              {isAnalyzing && (
                <div className="camera-overlay top-right">ANALYZING</div>
              )}

              <div className="camera-corner corner-tl" />
              <div className="camera-corner corner-tr" />
              <div className="camera-corner corner-bl" />
              <div className="camera-corner corner-br" />
            </div>

            <div className="camera-footer">
              <div className="status-message">
                <span className="green-dot" />

                {lastMessage}
              </div>

              <div className="image-meta">
                {filename || (cameraActive ? "Live camera" : "No image")}

                {imageDimensions.width > 0 &&
                  ` · ${imageDimensions.width}×${imageDimensions.height}`}
              </div>
            </div>
          </section>

          {/* ==================================================
              SCENE ANALYSIS
              ================================================== */}

          <div className="two-column-panels">
            <section className="card scene-panel">
              <div className="panel-heading compact">
                <div>
                  <p className="eyebrow">COMPUTER VISION</p>

                  <h2>Scene Analysis</h2>
                </div>

                <span className="live-badge">AI</span>
              </div>

              <p className="scene-description">
                {scene ||
                  (sceneObjects.length > 0
                    ? `Detected ${sceneObjects.length} object${
                        sceneObjects.length === 1 ? "" : "s"
                      } in the scene.`
                    : "Waiting for a camera frame to analyze.")}
              </p>

              {cloudStatus && <p className="cloud-status" role="status">{cloudStatus}</p>}

              <div className="detection-list">
                {sceneObjects.length > 0 ? (
                  sceneObjects.map((object, index) => {
                    const objectName = object.class || object.label || "Object";

                    const confidence =
                      typeof object.confidence === "number"
                        ? `${Math.round(object.confidence * 100)}%`
                        : "—";

                    const distanceMeters =
                      typeof object.distance_meters === "number"
                        ? `${object.distance_meters.toFixed(2)} m`
                        : object.distance !== undefined
                          ? String(object.distance)
                          : "—";

                    return (
                      <div className="info-row" key={`${objectName}-${index}`}>
                        <div>
                          <strong>{objectName}</strong>

                          <small>
                            {object.direction || "center"} · {distanceMeters}
                          </small>
                        </div>

                        <span>{confidence}</span>
                      </div>
                    );
                  })
                ) : scene ? (
                  <div className="info-row">
                    <span>Scene status</span>

                    <strong>Analyzed</strong>
                  </div>
                ) : (
                  <div className="empty-row">No detections yet.</div>
                )}
              </div>
            </section>

            {/* ==================================================
                NAVIGATION
                ================================================== */}

            <section className="card navigation-panel">
              <div className="panel-heading compact">
                <div>
                  <p className="eyebrow">SPATIAL ENGINE</p>

                  <h2>Navigation</h2>
                </div>

                <span className="direction-badge">NAV</span>
              </div>

              <div className="navigation-main" role="status" aria-live="assertive" aria-atomic="true">
                <div className="direction-arrow">→</div>

                <div>
                  <span className="nav-label">RECOMMENDATION</span>

                  <strong>{navigationText}</strong>
                </div>
              </div>

              {navigation?.distance !== undefined && (
                <div className="distance-row">
                  Distance
                  <strong>{typeof navigation.distance === "number" ? `${navigation.distance} m` : navigation.distance}</strong>
                </div>
              )}
            </section>
          </div>

          {/* ==================================================
              UPLOAD
              ================================================== */}

          <section
            className="card upload-panel"
            onDragOver={(e) => {
              e.preventDefault();

              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <div className="upload-copy">
              <p className="eyebrow">TEST IMAGE</p>

              <h2>Upload a scene</h2>

              <p>Drop an image here or select one from your computer.</p>
            </div>

            <div
              className={`drop-zone ${dragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <span className="upload-icon">↑</span>

              <span>Choose image</span>

              <small>JPG, PNG, WEBP</small>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={handleFileInput}
            />

            {imageUrl && (
              <button className="btn btn-secondary" onClick={clearImage}>
                Clear
              </button>
            )}
          </section>
        </div>

        {/* ======================================================
            SIDEBAR
            ====================================================== */}

        <aside className="sidebar">
          <section className="card voice-card">
            <div className="voice-header">
              <div>
                <p className="eyebrow">ASSEMBLYAI</p>

                <h2>Voice Assistant</h2>
              </div>

              <span className={`voice-state ${voiceState}`}>{voiceLabel}</span>
            </div>

            <button
              className={`voice-button ${voiceConnected ? "active" : ""}`}
              onClick={() => (voiceConnected || voiceState === "connecting" ? stopVoice() : void startVoice())}
            >
              <span className="voice-ring">
                <span className="voice-mic">●</span>
              </span>

              <strong>
                {voiceConnected ? "Stop listening" : voiceState === "connecting" ? "Cancel connection" : "Start voice"}
              </strong>

              <small>
                {voiceConnected ? "Tap to disconnect" : "Talk to EchoSight"}
              </small>
            </button>

            <div className="voice-tip">
              Ask: “What is in front of me?”, “Is it safe to move?”, or “Where
              should I go?”
            </div>
          </section>

          {/* ==================================================
              CONVERSATION
              ================================================== */}

          <section className="card conversation-panel">
            <div className="panel-heading compact">
              <div>
                <p className="eyebrow">LIVE TRANSCRIPT</p>

                <h2>Conversation</h2>
              </div>

              <span className="message-count">{conversation.length}</span>
            </div>

            <form className="question-form" onSubmit={async (event) => {
              event.preventDefault();
              const text = question.trim();
              if (!text || asking) return;
              setQuestion("");
              setAsking(true);
              addMessage("user", text);
              try {
                const raw = await analyzeCurrentFrame(text);
                const answer = typeof raw.result === "string" ? JSON.parse(raw.result).answer : raw.answer;
                addMessage("assistant", typeof answer === "string" ? answer : "I could not answer from the available observations.");
              } finally { setAsking(false); }
            }}>
              <label htmlFor="scene-question">Ask about this view</label>
              <input id="scene-question" value={question} onChange={(event) => setQuestion(event.target.value)}
                placeholder="What is on my left?" maxLength={2000} autoComplete="off" />
              <button className="btn btn-secondary" disabled={asking || !question.trim()} type="submit">
                {asking ? "Analyzing..." : "Ask"}
              </button>
            </form>

            <div className="conversation-list">
              {conversation.length === 0 ? (
                <div className="conversation-empty">
                  Ask a question below or start the voice assistant.
                </div>
              ) : (
                conversation.map((message) => (
                  <div key={message.id} className={`message ${message.role}`}>
                    <span className="message-role">
                      {message.role === "user" ? "YOU" : "ECHO"}
                    </span>

                    <p>{message.text}</p>
                  </div>
                ))
              )}
            </div>
          </section>

          {/* ==================================================
              QUICK STATUS
              ================================================== */}

          <section className="card quick-panel">
            <p className="eyebrow">QUICK STATUS</p>

            <div className="stat-grid">
              <div>
                <span>Camera</span>

                <strong>{cameraActive ? "ON" : "OFF"}</strong>
              </div>

              <div>
                <span>Vision WS</span>

                <strong>{connected ? "OK" : "—"}</strong>
              </div>

              <div>
                <span>Voice</span>

                <strong>{voiceConnected ? "OK" : "—"}</strong>
              </div>

              <div>
                <span>Scene</span>

                <strong>
                  {sceneObjects.length > 0 || scene ? "READY" : "—"}
                </strong>
              </div>
            </div>
          </section>
        </aside>
      </section>

      <canvas ref={canvasRef} className="hidden-canvas" />
    </main>
  );
}
