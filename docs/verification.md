# Implementation and verification record

Date: October 18, 2026. Local environment: macOS, Python 3.14.3, Node 25.8.2, Apple MPS. No changes were committed or deployed.

## Automated checks

| Check | Actual result |
| --- | --- |
| `../.venv/bin/python -m unittest discover -s backend/tests -q` | **76 passed, 0 failed** |
| `npm test` | **18 passed, 0 failed** |
| Python AST syntax checks | **21 files passed** |
| `npm run build` | **Passed**, including TypeScript and static page generation |
| `npm run lint` | **0 errors, 1 existing-style `<img>` warning** for the local upload preview |
| `git diff --check` | Passed |
| Scan changed files for actual configured API credential values | 0 matches; the real `.env` is ignored and untracked |

The tests cover AMD image/request construction, compression, response parsing, truncation, redirects, HTTP failures and slow responses without client timeouts; Evolus advisory catalog lookup, direct agent-chat fallback, separate agent/application IDs, response parsing, evidence validation and stateless conversations; local fallback, cancellation, feature/distance hallucination rejection and safety claims; obstacle classes and priority, missing/failed depth, empty scenes; session isolation, expiry, revision ordering, stale/fresh observations, clearing a session and source transitions; the existing serialized voice-tool contract; bounded inference and event-loop responsiveness; WebSocket authentication, malformed frames and recovery; browser transport reconnect, timeout, duplicate-connection prevention and cleanup; AssemblyAI tool results arriving before/after reply completion, ordinary and fc-prefixed reply IDs, interruptions and session teardown.

The suite uses simulated observations and mocked cloud responses explicitly as test fixtures. It does not require real credentials or load model weights. The test client emits an upstream Starlette warning recommending `httpx2`; tests still pass with the requested HTTPX dependency. Node's test runner emits a module-type detection warning for the standalone TypeScript test import. These warnings are not failures.

## Real local and provider checks

- **Local inference:** the repository's actual `bus.jpg` passed YOLO11 and Depth Anything V2, returning 5 detections with bus/person classes and `CAUTION: A bus is directly ahead.` First standalone analysis: 24.66 seconds including model loading; second analysis: 1.07 seconds. Hardware and cold caches affect these measurements. Meter estimates were absent because sample calibration is disabled.
- **Actual HTTP upload:** a session-authenticated upload returned HTTP 200 and five detections. `/health` returned HTTP 200 in about 0.089 seconds while real inference was running, confirming the API event loop remained responsive.
- **AMD live catalog:** configured model available. **Evolus live catalog:** selected application/agent route available. The agent lookup reports configured model display name **EchoSightCoordinator**, state `ok`, with zero configured sub-agents. This is configuration evidence, not proof that any other listed alias ran for a request.
- **Real cloud chain:** `python -m backend.scripts.smoke_cloud --live` sent a generated blue shape diagram to AMD and its actual observations to Evolus. A completed run returned `sources: [local, amd, evolus]`, with both provider statuses `success`, in **28,492.6 ms**. The response described the blue shape and acknowledged that it was a synthetic diagram, not a physical route.
- **Observed cloud failures:** earlier live runs hit invalid-grounding rejection and AMD request timeouts. The app displayed local results and fallback status. The application no longer imposes HTTP or overall cloud-enrichment deadlines, so slow or stalled provider calls can keep requests open indefinitely. A single fenced JSON response is accepted; Evolus plain-text answers are also accepted only when there are observations to attach as evidence and they pass the existing answer safety checks. Provider or infrastructure failures still return local fallback when an error is reported; fallback is not reported as a cloud success.
- **Real production browser:** Chrome loaded the built frontend, uploaded the actual bus photograph, displayed five detections and the local bus caution, showed actual cloud/fallback indicators, started a synthetic browser camera with a playing video and authenticated WebSocket, and stopped/cleaned up the camera. That check recorded no uncaught app exceptions. A typed crossing question returned explicit uncertainty.
- **AssemblyAI:** the existing temporary-token endpoint returned a token successfully (the token was not printed). The production browser reached a real speaking session. A generated recording of “Is it safe to cross?” was then injected as 24-kHz PCM into the browser WebSocket transport, explicitly bypassing Chrome's silent fake microphone. AssemblyAI transcribed it, emitted a real tool call, the browser invoked the authenticated backend voice route, sent the queued `tool.result`, and received the spoken agent response: “I can't reliably determine whether it's safe to cross from this camera view.” Network events confirmed both the backend request and outbound tool result. Camera shutdown and browser cleanup also passed with no uncaught app exceptions. This verifies the browser/AssemblyAI/backend roundtrip using synthetic audio; it does not certify a person's physical microphone/speaker or navigation conditions.
- **Live protocol correction:** current AssemblyAI documentation describes fc-prefixed reply IDs for tool turns; the actual connection returned ordinary reply IDs. The final client correlates `reply.started` and `reply.done` using their actual IDs and also supports the documented fc form. The recorded-question roundtrip initially exposed this mismatch and passed after the correction.

Some initial browser automation attempts needed harness corrections (selecting the actual page target, waiting for hydration and using boolean DOM checks). Those were automation errors, not passing app checks. The successful browser checks above were run after correcting them. A prerecorded-audio attempt through Chrome’s fake microphone produced silent PCM (confirmed by transport telemetry); it was not counted as a passing speech-recognition roundtrip.

## Changed/new files

| Area | Files and purpose |
| --- | --- |
| API and isolation | `backend/main.py`: retained endpoints/contracts, async token HTTP, bounded inference, session authentication, freshness, optional enrichment and safe response ordering. `backend/sessions.py`: independent expiring observations and revision checks. |
| AMD | `backend/services/amd_vision.py`: configured-model multimodal requests, bounded image preparation and validated responses. |
| Evolus | `backend/services/evolus.py`: documented agent-chat contract, advisory model-catalog lookup, exact application/agent routing and evidence parsing. |
| Cloud orchestration | `backend/services/cloud_http.py`: shared async transport, sanitized errors, pooled connections and timing. `backend/services/intelligence.py`: selective calls, local fallback, concurrency controls and answer policy. |
| Local vision | `backend/vision/detector.py`, `backend/vision/depth.py`: lazy model initialization. `backend/vision/scene.py`: handles absent/nonfinite depth. `backend/vision/metric_depth.py`: sample meter calibration requires explicit opt-in. |
| Navigation | `backend/navigation/engine.py`: expanded obstacle handling, urgent-obstacle priority and cautious wording. `backend/navigation/spatial.py`: invalid/flat depth remains unknown. |
| Browser | `frontend/app/page.tsx`: session-aware requests, camera lifecycle/recovery, local-first uploads, text questions, warning expiry, alert priority and voice cleanup. `frontend/lib/session.ts`: page-local bearer. `frontend/lib/vision-connection.ts`: authenticated, recoverable camera socket. `frontend/lib/voice-tools.ts`: verified reply-completion ordering for asynchronous tool results. |
| UI supporting files | `frontend/app/globals.css`: existing-style question controls/status. `frontend/app/layout.tsx`: EchoSight metadata. `frontend/package.json`: built-in transport test command. |
| Tests | `backend/tests/test_api.py`, `backend/tests/test_navigation.py`, `backend/tests/test_services.py`, `frontend/tests/vision-connection.test.mjs`, `frontend/tests/voice-tools.test.mjs`. |
| Demo/setup | `backend/scripts/smoke_cloud.py`: explicit optional real-provider test using a synthetic diagram. `backend/.env.example`: placeholders and limits. `backend/requirements.txt`: HTTPX, preserving the preexisting user addition. |
| Documentation | `README.md`, `docs/architecture.md`, `docs/verification.md`: setup, contracts/sources, implementation, limitations and demo plan. |

The user-created empty service files and existing HTTPX/environment-example edits were incorporated. The ignored local `backend/.env` changed **only** its agent ID after the user selected the existing Accessibility Assistant. No API key or unrelated setting was removed. No remote agent configuration was mutated.

## Remaining configuration and practical limits

The local credentials and selected agent route are configured and verified. Deployment still needs its own secrets, exact trusted frontend origin and build-time backend URL. The Evolus workspace owns the coordinator model connection; no claim is made that all four previously listed aliases were executed. Use one ASGI worker with the current memory store. Anonymous tokens are bearer capabilities, not account authentication. Public hosting needs HTTPS, authentication/rate limits and body-size limits appropriate to its audience.

Human camera/microphone checks in the intended environment remain necessary. An uncalibrated two-meter/six-meter demonstration must be labeled simulated; the app does not fabricate those measurements. Cloud latency varies and successful smoke tests do not guarantee future availability. Every demo should retain visible local warnings and an honest fallback path.

See the README for exact run commands and the 90-second demo sequence.
