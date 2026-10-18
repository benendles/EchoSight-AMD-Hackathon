"""EchoSight API: local vision first, optional cloud intelligence on user requests."""
from __future__ import annotations

import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
import io
import json
import logging
import math
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
import time

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import httpx
from PIL import Image, ImageOps, UnidentifiedImageError
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
# Support both `uvicorn backend.main:app` and the existing `uvicorn main:app`.
if str(BASE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BASE_DIR.parent))
load_dotenv(BASE_DIR / '.env')

from backend.navigation.engine import make_navigation_decision
from backend.sessions import Observation, SessionStore
from backend.services.cloud_http import seconds
from backend.services.intelligence import Intelligence, crossing_question, local_answer, navigation_question

metrics_logger = logging.getLogger('echosight')
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s %(name)s %(message)s'))
    metrics_logger.addHandler(handler)
metrics_logger.propagate = False
logger = logging.getLogger('echosight.api')
UPLOAD_DIR = BASE_DIR / 'uploads'
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {'JPEG', 'PNG', 'WEBP', 'BMP', 'GIF', 'TIFF'}
CORS_ORIGINS = [x.strip() for x in os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',') if x.strip()]
NAVIGATION_MAX_AGE = seconds('ECHOSIGHT_NAVIGATION_MAX_AGE_SECONDS', 5, 15)
DESCRIPTION_MAX_AGE = seconds('ECHOSIGHT_DESCRIPTION_MAX_AGE_SECONDS', 30, 300)
SESSION_TTL = seconds('ECHOSIGHT_SESSION_TTL_SECONDS', 1800, 86400)
store = SessionStore(ttl=SESSION_TTL)
_INFERENCE_LOCK = threading.Lock()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError('Choose a nonempty image smaller than 8 MiB.')
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in ALLOWED_IMAGE_FORMATS or source.width * source.height > 16_000_000:
                raise ValueError('Unsupported image or image larger than 16 megapixels.')
            source.seek(0)
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.load()
        return image
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise ValueError('Could not read this image. Please use a JPG, PNG, or WEBP image.') from None


def _prepare_image(image_bytes: bytes) -> bytes:
    image = load_image_from_bytes(image_bytes)
    image.thumbnail((1600, 1600))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=85)
    return output.getvalue()


def analyze_image(image_path: Path):
    """Preserved (scene, decision) contract. Only run in the inference worker.

    Model import/loading is lazy so startup, health checks and mock tests do not
    download weights or block the ASGI event loop.
    """
    from backend.vision.detector import detect
    from backend.vision.scene import analyze_scene
    import numpy as np

    with _INFERENCE_LOCK:
        detections = detect(str(image_path))
        depth = None
        try:
            from backend.vision.depth import estimate_depth
            depth = estimate_depth(str(image_path))
            finite = depth[np.isfinite(depth)]
            if not finite.size or np.ptp(finite) < 1e-6:
                depth = None
        except Exception as exc:
            logger.warning('depth_unavailable error_type=%s', type(exc).__name__)
        with Image.open(image_path) as image:
            scene = analyze_scene(detections, depth, image_size=image.size)
        decision = make_navigation_decision(scene, depth_available=depth is not None)
        decision['depth_available'] = depth is not None
        return _json_safe(scene), _json_safe(decision)


def analyze_image_bytes(image_bytes: bytes, filename: str = 'uploaded_image.jpg'):
    """Keep the existing tuple contract; client filenames never form paths."""
    normalized = _prepare_image(image_bytes)
    with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix='.jpg') as frame:
        frame.write(normalized)
        frame.flush()
        return analyze_image(Path(frame.name))


class InferenceRunner:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='echosight-vision')
        self.slots = asyncio.Semaphore(2)  # one running and one queued; no GPU overlap

    async def run(self, image_bytes: bytes):
        try:
            await asyncio.wait_for(self.slots.acquire(), timeout=seconds('ECHOSIGHT_INFERENCE_QUEUE_TIMEOUT_SECONDS', 5, 10))
        except TimeoutError:
            raise ValueError('Vision is busy. Please try the next frame.') from None
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self.pool, analyze_image_bytes, image_bytes)
        # A disconnected request must not release a slot while its native model
        # call is still running. Shield prevents cancellation of that worker.
        future.add_done_callback(lambda _: self.slots.release())
        try:
            return await asyncio.shield(future)
        finally:
            logger.info('local_inference elapsed_ms=%.1f', (time.monotonic() - started) * 1000)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


async def _cleanup_sessions():
    while True:
        await asyncio.sleep(60)
        store.cleanup()


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.inference = InferenceRunner()
    application.state.intelligence = Intelligence.from_environment()
    application.state.voice_http = httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5), follow_redirects=False)
    cleanup = asyncio.create_task(_cleanup_sessions())
    try:
        yield
    finally:
        cleanup.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup
        await application.state.intelligence.close()
        await application.state.voice_http.aclose()
        application.state.inference.close()
        store.clear()


app = FastAPI(title='EchoSight API', description='Local assistive vision with optional AMD and Evolus reasoning', version='1.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['Content-Type', 'X-EchoSight-Session'])


@app.middleware('http')
async def check_origin(request, call_next):
    origin = request.headers.get('origin')
    if origin and origin not in CORS_ORIGINS:
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': 'Origin is not allowed.'}, status_code=403)
    response = await call_next(request)
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


def require_session(x_echosight_session: str | None = Header(default=None)) -> str:
    if store.get(x_echosight_session) is None:
        raise HTTPException(401, 'Session missing or expired. Create a new session.')
    return x_echosight_session


@app.post('/api/session')
async def create_session():
    try:
        return {'session_id': store.create(), 'expires_in_seconds': SESSION_TTL}
    except ValueError as exc:
        raise HTTPException(503, str(exc)) from None


@app.delete('/api/scene')
async def clear_scene(session_id: str = Depends(require_session)):
    store.reserve(session_id)  # invalidate any inference still finishing
    store.get(session_id).observation = None
    return {'type': 'scene_cleared'}


def update_latest_scene(session_id: str, observation: Observation, revision: int) -> bool:
    return store.publish(session_id, revision, observation)


async def process_frame(session_id: str, image_bytes: bytes, source: str, *, captured_at: float | None = None) -> Observation:
    captured_at = time.monotonic() if captured_at is None else captured_at
    revision = store.reserve(session_id)
    # Decoding and compression also happen off the event loop.
    normalized = await asyncio.to_thread(_prepare_image, image_bytes)
    scene, decision = await app.state.inference.run(normalized)
    from backend.vision.scene import describe_scene
    observation = Observation(secrets.token_hex(12), normalized, scene, decision, describe_scene(scene),
                              captured_at, source, decision.get('depth_available', True))
    if not update_latest_scene(session_id, observation, revision):
        raise ValueError('A newer frame replaced this analysis. Please try again.')
    return observation


def cautious_navigation(message: str) -> dict:
    return {'action': 'CAUTION', 'priority': .5, 'message': message}


def current_navigation(observation: Observation) -> dict:
    if observation.source == 'camera' and observation.age > NAVIGATION_MAX_AGE:
        return cautious_navigation("I need a more recent camera view to assess what's ahead.")
    return observation.navigation


def description_max_age(observation: Observation) -> float:
    return seconds('ECHOSIGHT_UPLOAD_MAX_AGE_SECONDS', 600, 3600) if observation.source == 'upload' else DESCRIPTION_MAX_AGE


def vision_result(observation: Observation) -> dict:
    return {'type': 'vision_result', 'scene': observation.scene, 'navigation': current_navigation(observation),
            'description': observation.description, 'observation_id': observation.id,
            'source': observation.source, 'age_seconds': round(observation.age, 2),
            'navigation_valid_for_seconds': max(0, NAVIGATION_MAX_AGE - observation.age) if observation.source == 'camera' else 0,
            'depth_available': observation.depth_available}


def handle_voice_command(command: str, session_id: str | None = None):
    """Local-only compatibility helper for the optional desktop voice client."""
    session = store.get(session_id)
    observation = session.observation if session else None
    if not observation:
        return {'answer': "I don't have a current camera image to analyze yet.", 'scene': [],
                'navigation': None, 'description': 'No current camera frame is available.'}
    if observation.age > (NAVIGATION_MAX_AGE if navigation_question(command) else description_max_age(observation)):
        answer = "I can't reliably determine whether it's safe to cross from this camera view." if crossing_question(command) else "I need a more recent camera view to assess what's ahead."
        return {'answer': answer, 'scene': [],
                'navigation': cautious_navigation('Camera observations are stale.'), 'description': 'Previous observations have expired.', 'stale': True}
    answer = local_answer(command, observation)
    nav = current_navigation(observation)
    if nav.get('action') in {'STOP', 'CAUTION'} and not crossing_question(command):
        answer = nav['message'] + (' ' + answer if answer != nav['message'] else '')
    return {**vision_result(observation), 'answer': answer}


def _tool_result(result: dict) -> dict:
    return {'type': 'tool_result', 'result': json.dumps(_json_safe(result), allow_nan=False)}


@app.post('/api/voice/analyze')
async def voice_analyze(payload: dict = Body(...), session_id: str = Depends(require_session)):
    command = str(payload.get('command', '')).strip()[:2000]
    if not command:
        return {'type': 'error', 'message': 'No voice command was provided.'}
    result = handle_voice_command(command, session_id)
    session = store.get(session_id)
    observation = session.observation if session else None
    if not observation or result.get('stale') or navigation_question(command):
        return _tool_result(result)
    # Urgent warnings are returned immediately; they never wait for cloud calls.
    if current_navigation(observation).get('action') == 'STOP' and observation.source == 'camera':
        return _tool_result({**result, 'sources': ['local'], 'cloud': {'status': 'skipped_urgent_warning'}})
    return _tool_result(await enrich_observation(session_id, observation, command, result))


async def enrich_observation(session_id: str, observation: Observation, command: str, result: dict, *, upload=False) -> dict:
    session = store.get(session_id)
    if not session or session.cloud_busy or time.monotonic() - session.cloud_last_started < 2:
        return {**result, 'sources': ['local'], 'cloud': {'status': 'busy'}}
    session.cloud_busy = True
    session.cloud_last_started = time.monotonic()
    try:
        enriched = await app.state.intelligence.enrich(command, observation, upload=upload)
    finally:
        session.cloud_busy = False
    # Cloud completion must not overwrite a newer local observation. A camera
    # snapshot may also expire while the remote services are generating.
    active = store.get(session_id)
    if not active:
        return {'answer': 'This session has expired.', 'scene': [], 'navigation': None, 'stale': True}
    latest = active.observation
    if latest is None or (latest.id != observation.id and (observation.source == 'upload' or latest.source != observation.source)):
        return {**handle_voice_command(command, session_id), 'cloud': {'status': 'superseded'}}
    if observation.age > description_max_age(observation):
        return {**handle_voice_command(command, session_id), 'cloud': {'status': 'expired'}}
    nav = current_navigation(latest)
    answer = enriched['answer']
    if latest.id != observation.id:
        answer = 'From the earlier camera view: ' + answer
    if nav.get('action') in {'STOP', 'CAUTION'}:
        answer = nav['message'] + ' ' + answer
    return {**result, **enriched, 'answer': answer, 'navigation': nav,
            'age_seconds': round(observation.age, 2), 'navigation_valid_for_seconds': max(0, NAVIGATION_MAX_AGE - latest.age) if latest.source == 'camera' else 0}


@app.post('/api/vision/enrich')
async def enrich_image(payload: dict = Body(...), session_id: str = Depends(require_session)):
    observation = store.get(session_id).observation
    if not observation or payload.get('observation_id') != observation.id:
        return {'type': 'error', 'message': 'This image has been replaced. Analyze the latest view.'}
    if observation.age > description_max_age(observation):
        return _tool_result(handle_voice_command('Explain this image.', session_id))
    command = str(payload.get('command', 'Explain this image.'))[:2000]
    result = {**vision_result(observation), 'answer': local_answer(command, observation)}
    return _tool_result(await enrich_observation(session_id, observation, command, result, upload=True))


@app.get('/api/voice-token')
async def voice_token(session_id: str = Depends(require_session)):
    key = os.getenv('ASSEMBLYAI_API_KEY')
    if not key:
        return {'type': 'error', 'message': 'ASSEMBLYAI_API_KEY is not configured on the backend.'}
    try:
        # Existing AssemblyAI token contract, now using async I/O.
        response = await app.state.voice_http.get('https://agents.assemblyai.com/v1/token',
            params={'expires_in_seconds': '300'}, headers={'Authorization': f'Bearer {key}'})
        response.raise_for_status()
        token = response.json().get('token')
        if not isinstance(token, str) or not token:
            raise ValueError('missing token')
        return {'token': token}
    except (httpx.HTTPError, ValueError):
        logger.warning('voice_token_failed')
        return {'type': 'error', 'message': 'Could not create an AssemblyAI voice token.'}


@app.get('/')
async def root():
    return {'name': 'EchoSight', 'status': 'online', 'vision': 'YOLO + Depth Anything V2 + Spatial Reasoning',
            'voice': 'AssemblyAI Voice Agent', 'image_support': sorted(ALLOWED_IMAGE_FORMATS)}


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/api/vision/upload')
async def upload_image(file: UploadFile = File(...), session_id: str = Depends(require_session)):
    try:
        if file.content_type and not file.content_type.startswith('image/'):
            raise ValueError('Please upload an image file.')
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
        observation = await process_frame(session_id, image_bytes, 'upload')
        # Return local alerts first. The frontend can request optional enrichment.
        return {**vision_result(observation), 'filename': Path(file.filename or 'image').name[:200]}
    except ValueError as exc:
        return {'type': 'error', 'message': str(exc)}
    except Exception as exc:
        logger.warning('upload_inference_failed error_type=%s', type(exc).__name__)
        return {'type': 'error', 'message': 'Local vision analysis failed. Check the backend setup.'}
    finally:
        await file.close()


@app.get('/api/vision/test')
async def test_vision(session_id: str = Depends(require_session)):
    observation = store.get(session_id).observation
    if not observation:
        return {'type': 'error', 'message': 'No image has been submitted in this session.'}
    # A test can rerun inference, but it must not relabel an old capture as fresh.
    result = await process_frame(session_id, observation.frame, observation.source, captured_at=observation.captured_at)
    return vision_result(result)


@app.websocket('/ws/vision')
async def vision_websocket(websocket: WebSocket):
    origin = websocket.headers.get('origin')
    if origin and origin not in CORS_ORIGINS:
        await websocket.close(code=4403)
        return
    await websocket.accept()
    try:
        hello = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        session_id = hello.get('session_id') if isinstance(hello, dict) and hello.get('type') == 'session' else None
        if not isinstance(session_id, str) or store.get(session_id) is None:
            await websocket.close(code=4401)
            return
        await websocket.send_json({'type': 'session_ready'})
        while True:
            message = await websocket.receive_json()
            if store.get(session_id) is None:
                await websocket.close(code=4401)
                return
            if not isinstance(message, dict):
                await websocket.send_json({'type': 'error', 'message': 'Expected a frame message.'})
                continue
            if message.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
                continue
            if message.get('type') != 'frame':
                await websocket.send_json({'type': 'error', 'message': 'Unknown message type.'})
                continue
            try:
                encoded = message.get('image')
                if not isinstance(encoded, str) or len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 200:
                    raise ValueError('Camera image missing or too large.')
                image_bytes = base64.b64decode(encoded.split(',')[-1], validate=True)
                observation = await process_frame(session_id, image_bytes, 'camera')
                await websocket.send_json(vision_result(observation))
            except (ValueError, binascii.Error) as exc:
                await websocket.send_json({'type': 'error', 'message': str(exc)[:200]})
            except Exception as exc:
                logger.warning('websocket_inference_failed error_type=%s', type(exc).__name__)
                await websocket.send_json({'type': 'error', 'message': 'Vision analysis failed. Please try the next frame.'})
    except (WebSocketDisconnect, TimeoutError):
        pass
    except (ValueError, RuntimeError):
        with suppress(RuntimeError):
            await websocket.close(code=4400)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT', '8000')))
