# EchoSight — AMD AI Hackathon

EchoSight is an assistive-vision prototype for blind and low-vision people. It turns a live camera view or uploaded image into concise scene descriptions, local obstacle warnings, and voice-friendly answers to questions such as “What is ahead?” The person remains in control: cloud reasoning is requested only when useful, and safety decisions stay local.

> **Safety:** EchoSight is an experimental aid, not a certified mobility device. “No immediate obstacle detected” does not mean a route is clear or safe. The app does not certify crossings or tell someone that it is safe to move.

## What EchoSight does

1. The browser sends a camera frame or an uploaded image to the FastAPI service.
2. Local YOLO11 object detection and Depth Anything V2 relative-depth estimation produce scene observations.
3. A local navigation policy prioritizes potential obstacles and issues cautious `STOP`, `CAUTION`, or `CONTINUE` messages. Missing depth results in more cautious guidance.
4. When a person asks for a detailed description, optional AMD-hosted vision adds observations. An optional Evolus agent combines those observations with the local scene to answer in short, accessible language.
5. The web app presents the answer and supports voice interaction through the existing AssemblyAI integration. Local results remain available when a cloud service is disabled or unavailable.

Local obstacle handling does not wait for the AMD or Evolus services. User requests, image age, and local warnings determine whether cloud enrichment is appropriate; cloud answers cannot override local navigation warnings.

## Hackathon technologies and partner fit

| Technology or partner | Where it fits in EchoSight | How it helps the system |
| --- | --- | --- |
| **AMD Developer Cloud / AMD Radeon Token Factory** | Integrated as an optional, server-side vision API. EchoSight sends a normalized image and the user's question to the configured AMD model for detailed visual observations. | Uses AMD-hosted inference without requiring a GPU on the user's device. It can add detail beyond local object detections while the local warning path stays responsive. |
| **ROCm** | AMD's open-source GPU software platform is relevant to running supported AI workloads on AMD GPUs. The current app calls the AMD-hosted API; it does **not** directly install or invoke ROCm, and this project has not certified a local ROCm deployment. | Provides a path for teams to run or adapt compatible PyTorch workloads on AMD hardware when they move from hosted inference to direct GPU execution. |
| **AMD AI Developer Program and AI Academy** | Learning, tutorials, cloud access, and developer support for building and validating the AMD path. These are development resources, not runtime dependencies in the app. | Helps the team learn the AMD stack, benchmark workloads, and prepare for deployment without owning GPU hardware. |
| **Google DeepMind Gemma** | The AMD vision model identifier is configurable. A Gemma model can be selected when it is available and enabled for the chosen AMD endpoint; EchoSight does not hard-code Gemma or claim it is the active model. | An open model option for teams that want to evaluate a Google model on AMD-hosted GPU infrastructure. |
| **Evolus AI** | Integrated through its OpenAI-compatible agent-chat API. The app supplies fresh local observations and, when available, AMD observations; the configured agent returns a grounded, voice-friendly answer. | Adds a business-agent layer without replacing EchoSight's local vision and warning policy. The current integration uses agent chat, not Evolus workflows, extractors, knowledge libraries, or MCP connectors. |
| **Vibe Generation** | A project and process partner rather than a software dependency. EchoSight keeps the person in the loop and treats AI as an aid to human judgment. | Its human-plus-AI approach is relevant to designing accessible workflows that complement, rather than replace, a person's decisions. |
| **PwC Italy, IMMENT, and BAC Network** | Business, startup, and community ecosystem partners; they are not connected to the running application. | Their professional, incubation, and entrepreneurial perspectives can help assess practical value, responsible adoption, and the path from prototype to a sustainable product. |

The AMD and Evolus connections are optional and configured on the backend. Credentials must never be placed in frontend code or committed to the repository. The app's local detector, depth model, API, and web interface remain separate from those providers.

## Architecture

```text
Browser camera / upload
          |
          v
 FastAPI backend ----> YOLO11 + Depth Anything V2
          |                       |
          |                       v
          |              local scene + warnings ----> browser / voice
          |
          +---- user-requested detail ----> AMD vision (optional)
          |                                      |
          +--------------------------------------v
                                  Evolus agent (optional)
                                             |
                                             v
                                      accessible answer
```

The backend keeps observations isolated per browser session and applies freshness limits before answering questions. It does not continuously stream camera images to cloud models. See [the architecture and API contracts](docs/architecture.md) and [verification notes](docs/verification.md) for more detail.

## Run locally

Requirements: Python 3.11 or newer, Node.js 22.18 or newer, and npm.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp backend/.env.example backend/.env
```

Edit `backend/.env` and provide the credentials and exact model/workspace identifiers for the providers you want to enable. Keep the file private; it is ignored by Git. AMD and Evolus can be disabled when credentials are unavailable. Configure `CORS_ORIGINS` for the frontend origin if it differs from localhost.

Start the API from the repository root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000` and allow camera and microphone access. Localhost is a secure browser context for development; hosted camera/microphone access requires HTTPS. The first local analysis may take longer while model weights are loaded.

## Configuration

Backend defaults and all supported settings are listed in [`backend/.env.example`](backend/.env.example). In particular:

- AMD vision uses `AMD_TOKEN_FACTORY_API_KEY`, `AMD_TOKEN_FACTORY_BASE_URL`, and the exact `AMD_VISION_MODEL` identifier provided for your AMD account.
- Evolus agent chat uses `EVOLUS_API_KEY`, `EVOLUS_APPLICATION_ID`, and `EVOLUS_AGENT_ID`. The application and agent IDs are separate values.
- `AMD_VISION_ENABLED` and `EVOLUS_ENABLED` control optional cloud providers.
- `ECHOSIGHT_METRIC_DEPTH_ENABLED` must remain disabled unless the reference calibration has been measured for the actual camera. Relative depth is not a real-world distance measurement.

The frontend backend URL is configured with `NEXT_PUBLIC_BACKEND_URL` in `frontend/.env.local`; it defaults to the local API.

## Tests

From the repository root:

```bash
python -m unittest discover -s backend/tests -v
```

For frontend transport and voice-tool tests and checks:

```bash
cd frontend
npm test
npm run lint
npm run build
```

Backend provider tests use mocked HTTP transports and do not require live credentials. A real-provider smoke test is available with `python -m backend.scripts.smoke_cloud --live`; it sends a synthetic image to the configured providers and requires valid credentials and network access.
