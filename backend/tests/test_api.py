import asyncio
from dataclasses import replace
import io
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from backend import main
from backend.navigation.engine import make_navigation_decision
from backend.services.cloud_http import CloudError
from backend.services.intelligence import Intelligence, local_answer
from backend.sessions import Observation, SessionStore


def frame_bytes(color='blue'):
    output = io.BytesIO()
    Image.new('RGB', (640, 480), color).save(output, 'JPEG')
    return output.getvalue()


def observation(label='person', *, age=0, source='camera', action='CONTINUE'):
    scene = [{'class': label, 'confidence': .9, 'direction': 'center', 'position': 'directly ahead',
              'distance': 'medium', 'distance_meters': None, 'path_score': .7}]
    nav = {'action': action, 'message': f'{"Stop" if action == "STOP" else "Caution"}. A {label} is ahead.'}
    return Observation(label + '-id', frame_bytes(), scene, nav, f'A {label} is ahead.', time.monotonic() - age, source)


class SessionTests(unittest.TestCase):
    def test_secure_identifiers_and_isolation(self):
        store = SessionStore()
        a, b = store.create(), store.create()
        self.assertNotEqual(a, b)
        self.assertGreaterEqual(len(a), 40)
        store.publish(a, store.reserve(a), observation('chair'))
        self.assertIsNone(store.get(b).observation)
        self.assertIsNone(store.get('../../uploads/current_frame.jpg'))

    def test_expiration(self):
        store = SessionStore(ttl=.01)
        token = store.create()
        with patch('backend.sessions.time.monotonic', return_value=time.monotonic() + 1):
            self.assertIsNone(store.get(token))

    def test_cleanup_removes_expired_frames(self):
        store = SessionStore(ttl=.01)
        token = store.create()
        store.publish(token, store.reserve(token), observation())
        with patch('backend.sessions.time.monotonic', return_value=time.monotonic() + 1):
            store.cleanup()
        self.assertIsNone(store.get(token))

    def test_old_revision_cannot_overwrite_newer_frame(self):
        store = SessionStore()
        token = store.create()
        old, new = store.reserve(token), store.reserve(token)
        self.assertTrue(store.publish(token, new, observation('bus')))
        self.assertFalse(store.publish(token, old, observation('chair')))
        self.assertEqual(store.get(token).observation.scene[0]['class'], 'bus')


class DepthFallbackTests(unittest.TestCase):
    def test_depth_model_failure_keeps_local_detection(self):
        import sys
        import tempfile
        import types
        from pathlib import Path
        detector = types.ModuleType('backend.vision.detector')
        detector.detect = lambda _: [{'class': 'chair', 'confidence': .9, 'box': [150, 100, 500, 400], 'direction': 'center', 'vertical_position': 'center'}]
        depth = types.ModuleType('backend.vision.depth')
        def fail(_):
            raise RuntimeError('Simulated depth model outage')
        depth.estimate_depth = fail
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.jpg'
            path.write_bytes(frame_bytes())
            with patch.dict(sys.modules, {'backend.vision.detector': detector, 'backend.vision.depth': depth}):
                scene, decision = main.analyze_image(path)
        self.assertEqual(scene[0]['class'], 'chair')
        self.assertEqual(scene[0]['distance'], 'unknown')
        self.assertEqual(decision['action'], 'CAUTION')
        self.assertFalse(decision['depth_available'])


class APITests(unittest.TestCase):
    def setUp(self):
        main.store.clear()
        self.env = patch.dict(os.environ, {'AMD_VISION_ENABLED': 'false', 'EVOLUS_ENABLED': 'false'})
        self.env.start()
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        self.token = self.client.post('/api/session').json()['session_id']
        self.headers = {'X-EchoSight-Session': self.token}

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.env.stop()

    def put(self, obs, token=None):
        key = token or self.token
        main.store.publish(key, main.store.reserve(key), obs)

    def ask(self, command):
        response = self.client.post('/api/voice/analyze', headers=self.headers, json={'command': command})
        self.assertEqual(response.status_code, 200)
        envelope = response.json()
        self.assertEqual(envelope['type'], 'tool_result')
        self.assertIsInstance(envelope['result'], str)
        return json.loads(envelope['result'])

    def test_session_required(self):
        self.assertEqual(self.client.post('/api/voice/analyze', json={'command': 'Describe'}).status_code, 401)
        self.assertEqual(self.client.get('/api/vision/test').status_code, 401)

    def test_fresh_voice_tool_contract(self):
        self.put(observation())
        result = self.ask('What is ahead?')
        for key in ['answer', 'scene', 'navigation', 'description', 'observation_id']:
            self.assertIn(key, result)
        self.assertIn('person', result['answer'])

    def test_navigation_requires_recent_observation(self):
        self.put(observation(age=main.NAVIGATION_MAX_AGE + 1))
        result = self.ask('Can I walk forward?')
        self.assertTrue(result['stale'])
        self.assertEqual(result['scene'], [])
        self.assertEqual(result['navigation']['action'], 'CAUTION')

    def test_descriptive_question_allows_longer_window(self):
        self.put(observation(age=main.NAVIGATION_MAX_AGE + 1))
        result = self.ask('What is ahead?')
        self.assertNotIn('stale', result)
        self.assertIn('person', result['answer'])
        self.assertEqual(result['navigation']['action'], 'CAUTION')

    def test_expired_descriptive_scene(self):
        self.put(observation(age=main.DESCRIPTION_MAX_AGE + 1))
        self.assertTrue(self.ask('Describe the scene')['stale'])

    def test_uploaded_image_never_establishes_live_route(self):
        self.put(observation(source='upload'))
        result = self.ask('Is it safe to walk?')
        self.assertIn('uploaded image', result['answer'])

    def test_uncertain_crossing(self):
        self.put(observation(action='STOP'))
        result = self.ask('Is it safe to cross?')
        self.assertIn("can't reliably determine", result['answer'])
        self.assertEqual(result['navigation']['action'], 'STOP')

    def test_urgent_warning_does_not_wait_for_cloud(self):
        self.put(observation('chair', action='STOP'))
        cloud = AsyncMock(side_effect=AssertionError('Cloud must not run'))
        with patch.object(main.app.state.intelligence, 'enrich', cloud):
            result = self.ask('Describe everything')
        self.assertTrue(result['answer'].startswith('Stop'))
        cloud.assert_not_called()

    def test_cloud_response_preserves_warning(self):
        self.put(observation('chair', action='STOP', source='upload'))
        cloud = AsyncMock(return_value={'answer': 'A chair is visible.', 'sources': ['local', 'evolus'], 'cloud': {'evolus': 'success'}})
        with patch.object(main.app.state.intelligence, 'enrich', cloud):
            result = self.ask('Explain the image')
        self.assertTrue(result['answer'].startswith('Stop'))
        self.assertEqual(result['navigation']['action'], 'STOP')

    def test_two_sessions_do_not_share_observations(self):
        other = self.client.post('/api/session').json()['session_id']
        self.put(observation('chair'))
        self.put(observation('bus'), other)
        self.assertIn('chair', self.ask('What is ahead?')['answer'])
        response = self.client.post('/api/voice/analyze', headers={'X-EchoSight-Session': other}, json={'command': 'What is ahead?'})
        self.assertIn('bus', json.loads(response.json()['result'])['answer'])
        self.assertNotIn('chair', response.text)

    def test_empty_session_never_reads_legacy_current_frame(self):
        self.assertEqual(self.ask('What is ahead?')['scene'], [])

    def test_clear_removes_session_observation(self):
        self.put(observation())
        self.assertEqual(self.client.delete('/api/scene', headers=self.headers).status_code, 200)
        self.assertIsNone(main.store.get(self.token).observation)
        self.assertEqual(self.ask('What is ahead?')['scene'], [])

    def test_upload_local_contract_and_no_public_file(self):
        with patch('backend.main.analyze_image_bytes', return_value=([], make_navigation_decision([]))):
            response = self.client.post('/api/vision/upload', headers=self.headers, files={'file': ('../../anything.jpg', frame_bytes(), 'image/jpeg')})
        self.assertEqual(response.json()['type'], 'vision_result')
        self.assertEqual(response.json()['filename'], 'anything.jpg')
        self.assertEqual(main.store.get(self.token).observation.source, 'upload')
        self.assertEqual(self.client.get('/uploads/current_frame.jpg').status_code, 404)

    def test_oversized_and_invalid_uploads_rejected(self):
        for data in [b'broken', b'x' * (main.MAX_IMAGE_BYTES + 1)]:
            with self.subTest(size=len(data)):
                response = self.client.post('/api/vision/upload', headers=self.headers, files={'file': ('a.jpg', data, 'image/jpeg')})
                self.assertEqual(response.json()['type'], 'error')

    def test_reanalysis_does_not_refresh_capture_timestamp(self):
        old = observation(age=main.NAVIGATION_MAX_AGE + 1)
        self.put(old)
        with patch('backend.main.analyze_image_bytes', return_value=([], make_navigation_decision([]))):
            result = self.client.get('/api/vision/test', headers=self.headers).json()
        self.assertEqual(result['navigation']['action'], 'CAUTION')
        self.assertGreaterEqual(main.store.get(self.token).observation.age, old.age - .1)

    def test_untrusted_origin_rejected(self):
        self.assertEqual(self.client.post('/api/session', headers={'Origin': 'https://evil.test'}).status_code, 403)

    def test_websocket_error_then_valid_frame_recovers(self):
        import base64
        with patch('backend.main.analyze_image_bytes', return_value=([], make_navigation_decision([]))):
            with self.client.websocket_connect('/ws/vision') as ws:
                ws.send_json({'type': 'session', 'session_id': self.token})
                self.assertEqual(ws.receive_json()['type'], 'session_ready')
                ws.send_json({'type': 'frame', 'image': 'invalid!'})
                self.assertEqual(ws.receive_json()['type'], 'error')
                ws.send_json({'type': 'frame', 'image': base64.b64encode(frame_bytes()).decode()})
                self.assertEqual(ws.receive_json()['type'], 'vision_result')
                ws.send_json({'type': 'ping'})
                self.assertEqual(ws.receive_json()['type'], 'pong')

    def test_websocket_authentication_required(self):
        from starlette.websockets import WebSocketDisconnect
        with self.client.websocket_connect('/ws/vision') as ws:
            ws.send_json({'type': 'session', 'session_id': 'guessed-token'})
            with self.assertRaises(WebSocketDisconnect) as caught:
                ws.receive_json()
            self.assertEqual(caught.exception.code, 4401)

    def test_websocket_forbidden_origin(self):
        from starlette.websockets import WebSocketDisconnect
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect('/ws/vision', headers={'Origin': 'https://evil.test'}):
                self.fail('Forbidden origin connected')

    def test_cloud_cannot_overwrite_replaced_upload(self):
        self.put(observation('chair', source='upload'))
        async def delayed(*args, **kwargs):
            self.put(observation('bus', source='upload'))
            return {'answer': 'Old chair image', 'sources': ['evolus'], 'cloud': {}}
        with patch.object(main.app.state.intelligence, 'enrich', delayed):
            result = self.ask('Explain this image')
        self.assertNotIn('Old chair', result['answer'])
        self.assertIn('bus', result['answer'])

    def test_cloud_cannot_cross_camera_to_upload_transition(self):
        self.put(observation('person', source='camera'))
        async def delayed(*args, **kwargs):
            self.put(observation('bus', source='upload'))
            return {'answer': 'Old camera person', 'sources': ['evolus'], 'cloud': {}}
        with patch.object(main.app.state.intelligence, 'enrich', delayed):
            result = self.ask('Describe the person')
        self.assertNotIn('Old camera', result['answer'])
        self.assertIn('bus', result['answer'])


class IntelligenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_cloud_services_fail_local_survives(self):
        amd = AsyncMock()
        amd.analyze.side_effect = CloudError('timeout')
        evolus = AsyncMock()
        evolus.answer.side_effect = CloudError('http_503')
        result = await Intelligence(amd, evolus).enrich('Describe everything', observation())
        self.assertEqual(result['sources'], ['local'])
        self.assertIn('person', result['answer'])
        self.assertEqual(result['cloud'], {'amd': 'timeout', 'evolus': 'http_503'})

    async def test_successful_amd_then_evolus(self):
        amd, evolus = AsyncMock(), AsyncMock()
        amd.analyze.return_value = {'observations': ['A blue wall'], 'uncertainty': [], 'summary': 'A blue wall.'}
        evolus.answer.return_value = {'answer': 'A blue wall is behind a person.', 'evidence': ['local:0', 'amd:0']}
        result = await Intelligence(amd, evolus).enrich('Describe everything', observation())
        self.assertEqual(result['sources'], ['local', 'amd', 'evolus'])
        self.assertEqual(evolus.answer.call_args.args[3], amd.analyze.return_value)

    async def test_navigation_never_calls_cloud(self):
        amd, evolus = AsyncMock(), AsyncMock()
        await Intelligence(amd, evolus).enrich('Is it safe to cross?', observation())
        amd.analyze.assert_not_called()
        evolus.answer.assert_not_called()

    async def test_simple_count_avoids_amd(self):
        amd = AsyncMock()
        await Intelligence(amd=amd).enrich('How many people are in front of me?', observation())
        amd.analyze.assert_not_called()

    async def test_unsafe_agent_output_rejected(self):
        evolus = AsyncMock()
        evolus.answer.return_value = {'answer': 'It is safe to cross now.', 'evidence': ['local:0']}
        result = await Intelligence(evolus=evolus).enrich('Describe the person', observation())
        self.assertEqual(result['sources'], ['local'])
        self.assertEqual(result['cloud']['evolus'], 'unsafe_answer_rejected')

    async def test_unqualified_safety_claim_rejected(self):
        evolus = AsyncMock()
        evolus.answer.return_value = {'answer': 'The area ahead is safe.', 'evidence': ['local:0']}
        result = await Intelligence(evolus=evolus).enrich('Describe the person', observation())
        self.assertEqual(result['sources'], ['local'])

    async def test_invented_critical_feature_rejected(self):
        evolus = AsyncMock()
        evolus.answer.return_value = {'answer': 'A handrail is beside the person.', 'evidence': ['local:0']}
        result = await Intelligence(evolus=evolus).enrich('Describe the person', observation())
        self.assertEqual(result['sources'], ['local'])

    async def test_invented_metric_distance_rejected(self):
        evolus = AsyncMock()
        evolus.answer.return_value = {'answer': 'A person is two meters ahead.', 'evidence': ['local:0']}
        result = await Intelligence(evolus=evolus).enrich('Describe the person', observation())
        self.assertEqual(result['sources'], ['local'])

    async def test_mismatched_metric_value_rejected(self):
        evolus = AsyncMock()
        evolus.answer.return_value = {'answer': 'A person is ten meters ahead.', 'evidence': ['local:0']}
        obs = observation()
        obs.scene[0]['distance_meters'] = 2.1
        result = await Intelligence(evolus=evolus).enrich('Describe the person', obs)
        self.assertEqual(result['sources'], ['local'])

    async def test_missing_requested_object_is_explicit(self):
        self.assertIn("don't detect a chair", local_answer('Is there a chair near me?', observation()))

    async def test_cancellation_propagates(self):
        amd = AsyncMock()
        amd.analyze.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await Intelligence(amd=amd).enrich('Describe everything', observation())

    async def test_inference_does_not_block_event_loop(self):
        runner = main.InferenceRunner()
        try:
            with patch('backend.main.analyze_image_bytes', side_effect=lambda _: (time.sleep(.1), {})):
                task = asyncio.create_task(runner.run(frame_bytes()))
                started = time.monotonic()
                await asyncio.sleep(.01)
                self.assertLess(time.monotonic() - started, .07)
                await task
        finally:
            runner.close()

    async def test_bounded_inference_never_overlaps(self):
        import threading
        runner = main.InferenceRunner()
        active, maximum = 0, 0
        lock = threading.Lock()
        def infer(_):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(active, maximum)
            time.sleep(.03)
            with lock: active -= 1
            return [], {}
        try:
            with patch('backend.main.analyze_image_bytes', side_effect=infer):
                await asyncio.gather(*(runner.run(b'frame') for _ in range(3)))
            self.assertEqual(maximum, 1)
        finally:
            runner.close()
