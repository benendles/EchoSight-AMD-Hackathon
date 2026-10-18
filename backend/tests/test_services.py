import asyncio
import io
import json
import unittest

import httpx
from PIL import Image

from backend.services.amd_vision import AMDVisionClient
from backend.services.cloud_http import CloudError
from backend.services.evolus import EvolusClient


def image_bytes():
    output = io.BytesIO()
    Image.new('RGB', (640, 480), 'blue').save(output, 'PNG')
    return output.getvalue()


def completion(content, **extra):
    return {'model': 'configured-model', 'choices': [{'message': {'content': json.dumps(content)}, 'finish_reason': 'stop'}], **extra}


class AMDTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler, **kwargs):
        client = AMDVisionClient(base_url='https://amd.test/api/v1', api_key='test-secret',
                                 model='exact-configured-id', transport=httpx.MockTransport(handler), **kwargs)
        self.addAsyncCleanup(client.close)
        return client

    async def test_request_and_response(self):
        def handler(request):
            self.assertEqual(str(request.url), 'https://amd.test/api/v1/chat/completions')
            self.assertEqual(request.headers['authorization'], 'Bearer test-secret')
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'exact-configured-id')
            self.assertEqual(body['messages'][0]['role'], 'system')
            self.assertTrue(body['messages'][1]['content'][0]['image_url']['url'].startswith('data:image/jpeg;base64,'))
            self.assertEqual(body['response_format'], {'type': 'json_object'})
            return httpx.Response(200, json=completion({'summary': 'A blue surface.', 'observations': ['blue surface'], 'uncertainty': []}))
        result = await self.client(handler).analyze(image_bytes(), 'What is visible?')
        self.assertEqual(result['summary'], 'A blue surface.')
        self.assertFalse(result['verified_physical_safety'])

    async def test_invalid_image_never_sends(self):
        client = self.client(lambda r: self.fail('Invalid image sent'))
        with self.assertRaisesRegex(CloudError, 'invalid_image'):
            await client.analyze(b'not-image', 'describe')

    async def test_compression_and_extreme_aspect_ratio(self):
        client = self.client(lambda r: None)
        data = io.BytesIO()
        Image.new('RGB', (3000, 100)).save(data, 'PNG')
        payload = client.build_request(data.getvalue(), 'read')
        import base64
        encoded = payload['messages'][1]['content'][0]['image_url']['url'].split(',')[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            self.assertLessEqual(max(image.size), 1280)
            self.assertGreaterEqual(min(image.size), 384)

    async def test_timeout(self):
        def handler(request):
            raise httpx.ReadTimeout('test-secret must not escape')
        with self.assertRaisesRegex(CloudError, '^timeout$'):
            await self.client(handler).analyze(image_bytes(), 'describe')

    async def test_slow_response_has_no_client_timeout(self):
        async def handler(request):
            await asyncio.sleep(.2)
            return httpx.Response(200, json=completion({
                'summary': 'A blue surface.', 'observations': ['blue surface'], 'uncertainty': []
            }))
        client = self.client(handler)
        self.assertIsNone(client.client.timeout.read)
        result = await client.analyze(image_bytes(), 'describe')
        self.assertEqual(result['summary'], 'A blue surface.')

    async def test_http_error_sanitized(self):
        with self.assertRaisesRegex(CloudError, '^http_401$'):
            await self.client(lambda r: httpx.Response(401, text='test-secret')).analyze(image_bytes(), 'describe')

    async def test_missing_content_rejected(self):
        with self.assertRaises(CloudError):
            AMDVisionClient.parse_response({'choices': []})

    async def test_missing_structured_fields_rejected(self):
        with self.assertRaises(CloudError):
            AMDVisionClient.parse_response(completion({'summary': 'A scene'}))

    async def test_truncated_response_rejected(self):
        data = completion({'summary': 'A scene', 'observations': [], 'uncertainty': []})
        data['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(CloudError):
            AMDVisionClient.parse_response(data)

    async def test_redirect_not_followed(self):
        with self.assertRaisesRegex(CloudError, 'http_302'):
            await self.client(lambda r: httpx.Response(302, headers={'Location': 'https://attacker.test'})).analyze(image_bytes(), 'describe')


class EvolusTests(unittest.IsolatedAsyncioTestCase):
    scene = [{'class': 'person', 'confidence': .95, 'direction': 'center', 'distance_meters': 2.1}]

    def client(self, handler):
        client = EvolusClient(base_url='https://evolus.test', api_key='evolus-secret',
                              application_id='application-A', agent_id='agent-B', transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.close)
        return client

    async def test_request_construction_and_response(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            self.assertEqual(request.headers['authorization'], 'Bearer evolus-secret')
            self.assertNotIn('X-Conversation-Id', request.headers)
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': 'catalog-format-differs'}]})
            self.assertEqual(request.url.path, '/v1/chat/completions')
            payload = json.loads(request.content)
            self.assertEqual(payload['model'], 'application-A:agent-B')
            self.assertIsInstance(payload['messages'][0]['content'], str)
            self.assertIn('distance_meters', payload['messages'][0]['content'])
            self.assertNotIn('temperature', payload)
            return httpx.Response(200, json=completion({'answer': 'A person is ahead, approximately two meters away.', 'evidence': ['local:0']}), headers={'X-Conversation-Id': 'provider-conversation'})
        client = self.client(handler)
        result = await client.answer('What is ahead?', self.scene, {'action': 'CAUTION'})
        self.assertEqual(result['evidence'], ['local:0'])
        await client.answer('What is ahead?', self.scene, {'action': 'CAUTION'})
        self.assertEqual(calls, ['/v1/models', '/v1/chat/completions', '/v1/chat/completions'])

    async def test_chat_endpoint_is_authoritative_for_agent_availability(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': None})
            if request.url.path == '/v1/chat/completions':
                return httpx.Response(200, json=completion({
                    'answer': 'A person is ahead.', 'evidence': ['local:0']
                }))
            return httpx.Response(500)
        client = self.client(handler)
        result = await client.answer('What is ahead?', self.scene, {'action': 'CAUTION'})
        self.assertEqual(result['answer'], 'A person is ahead.')
        self.assertEqual(calls, ['/v1/models', '/v1/chat/completions'])

    async def test_unavailable_agent_reports_chat_endpoint_error(self):
        client = self.client(lambda r: httpx.Response(404))
        with self.assertRaisesRegex(CloudError, '^http_404$'):
            await client.answer('Describe', [], {})

    async def test_agent_and_application_are_distinct(self):
        client = self.client(lambda r: None)
        self.assertEqual(client.model, 'application-A:agent-B')
        self.assertNotEqual(client.model, 'agent-B:agent-B')

    async def test_bad_grounding_rejected(self):
        with self.assertRaisesRegex(CloudError, 'invalid_grounding'):
            EvolusClient.parse_response(completion({'answer': 'Stairs ahead.', 'evidence': ['local:99']}), self.scene, None)

    async def test_amd_grounding(self):
        result = EvolusClient.parse_response(completion({'answer': 'A doorway is visible.', 'evidence': ['amd:0']}), [], {'observations': ['doorway']})
        self.assertEqual(result['source'], 'evolus')

    async def test_empty_evidence_rejected(self):
        with self.assertRaises(CloudError):
            EvolusClient.parse_response(completion({'answer': 'A doorway.', 'evidence': []}), [], None)

    async def test_plain_text_answer_uses_supplied_observations_as_evidence(self):
        result = EvolusClient.parse_response(
            {'choices': [{'message': {'content': 'A person is visible.'}, 'finish_reason': 'stop'}]},
            self.scene, None,
        )
        self.assertEqual(result['answer'], 'A person is visible.')
        self.assertEqual(result['evidence'], ['local:0'])

    async def test_plain_text_answer_without_observations_is_rejected(self):
        with self.assertRaisesRegex(CloudError, 'invalid_grounding'):
            EvolusClient.parse_response(
                {'choices': [{'message': {'content': 'Unsupported prose'}, 'finish_reason': 'stop'}]},
                [], None,
            )

    async def test_single_json_fence_accepted(self):
        text = '```json\n' + json.dumps({'answer': 'A person is ahead.', 'evidence': ['local:0']}) + '\n```'
        result = EvolusClient.parse_response({'choices': [{'message': {'content': text}}]}, self.scene, None)
        self.assertEqual(result['evidence'], ['local:0'])

    async def test_credential_free_errors(self):
        with self.assertRaisesRegex(CloudError, '^http_429$'):
            await self.client(lambda r: httpx.Response(429, text='evolus-secret')).answer(
                'Describe', self.scene, {}
            )
