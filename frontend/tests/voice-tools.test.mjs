import test from 'node:test';
import assert from 'node:assert/strict';
import { VoiceToolResults } from '../lib/voice-tools.ts';
function setup() { const sent = []; return { sent, tools: new VoiceToolResults(value => sent.push(value)) }; }

test('fast analysis waits for matching reply.done', () => {
  const { sent, tools } = setup();
  tools.receive({ type: 'tool.call' }); tools.register('one');
  tools.complete('one', { type: 'tool_result', result: '{"answer":"Stop."}' });
  assert.equal(sent.length, 0);
  tools.receive({ type: 'reply.done', reply_id: 'fc-one', status: 'completed' });
  assert.equal(sent.length, 1);
  assert.equal(sent[0].call_id, 'one');
  assert.equal(JSON.parse(sent[0].result).type, 'tool_result');
});

test('slow analysis sends when reply.done already arrived', () => {
  const { sent, tools } = setup(); tools.register('one');
  tools.receive({ type: 'reply.done', reply_id: 'fc-one', status: 'completed' });
  tools.complete('one', { answer: 'A chair.' });
  assert.equal(sent.length, 1);
});

test('unrelated reply.done cannot release a result', () => {
  const { sent, tools } = setup(); tools.register('one'); tools.complete('one', {});
  tools.receive({ type: 'reply.done', reply_id: 'greeting', status: 'completed' });
  assert.equal(sent.length, 0);
});

test('interruption discards pending and late HTTP results', () => {
  const { sent, tools } = setup(); tools.register('one');
  tools.receive({ type: 'reply.done', reply_id: 'fc-one', status: 'interrupted' });
  tools.complete('one', { answer: 'Old answer' });
  tools.receive({ type: 'reply.done', reply_id: 'fc-one', status: 'completed' });
  assert.equal(sent.length, 0);
});

test('new speech discards an earlier tool call', () => {
  const { sent, tools } = setup(); tools.register('one');
  tools.receive({ type: 'input.speech.started' }); tools.complete('one', {});
  tools.receive({ type: 'reply.done', reply_id: 'fc-one' });
  assert.equal(sent.length, 0);
});

test('later server activity prevents an out-of-order tool result', () => {
  const { sent, tools } = setup(); tools.register('one');
  tools.receive({ type: 'reply.done', reply_id: 'fc-one' });
  tools.receive({ type: 'reply.started' }); tools.complete('one', {});
  assert.equal(sent.length, 0);
});

test('closing a session prevents late completion', () => {
  const { sent, tools } = setup(); tools.register('one'); tools.clear();
  tools.complete('one', {}); tools.receive({ type: 'reply.done', reply_id: 'fc-one' });
  assert.equal(sent.length, 0);
});

test('actual reply.started identifier releases tool turns without an fc prefix', () => {
  const { sent, tools } = setup();
  tools.receive({ type: 'reply.started', reply_id: 'ordinary-reply' });
  tools.receive({ type: 'tool.call' }); tools.register('one'); tools.complete('one', { answer: 'Uncertain.' });
  tools.receive({ type: 'reply.done', reply_id: 'ordinary-reply', status: 'completed' });
  assert.equal(sent.length, 1);
});

test('a different ordinary reply ID cannot release a pending tool result', () => {
  const { sent, tools } = setup();
  tools.receive({ type: 'reply.started', reply_id: 'current-reply' });
  tools.register('one'); tools.complete('one', {});
  tools.receive({ type: 'reply.done', reply_id: 'previous-reply', status: 'completed' });
  assert.equal(sent.length, 0);
});
