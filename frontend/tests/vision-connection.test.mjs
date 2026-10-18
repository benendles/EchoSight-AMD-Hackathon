import test from 'node:test';
import assert from 'node:assert/strict';
import { VisionConnection } from '../lib/vision-connection.ts';

class FakeSocket {
  readyState = 1;
  sent = [];
  closed = false;
  send(message) { this.sent.push(JSON.parse(message)); }
  close(code = 1000) {
    if (this.closed) return;
    this.closed = true;
    this.readyState = 3;
    this.onclose?.({ code });
  }
  message(message) { this.onmessage?.({ data: JSON.stringify(message) }); }
}
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
function setup(t, extra = {}) {
  const sockets = [], results = [], errors = [], connected = [];
  let resets = 0;
  const connection = new VisionConnection({
    url: 'ws://local.test/ws/vision', session: async () => 'session-token',
    resetSession: () => resets++,
    socket: () => { const socket = new FakeSocket(); sockets.push(socket); return socket; },
    onResult: result => results.push(result), onError: error => errors.push(error),
    onConnected: state => connected.push(state), reconnectMs: 1,
    ...extra,
  });
  t.after(() => connection.stop());
  return { connection, sockets, results, errors, connected, resets: () => resets };
}
async function ready(state) {
  state.connection.start();
  await tick();
  const socket = state.sockets.at(-1);
  socket.onopen?.({});
  socket.message({ type: 'session_ready' });
  return socket;
}

test('authenticates before frames and suppresses duplicate connections', async t => {
  const state = setup(t);
  state.connection.start();
  state.connection.start();
  await tick();
  const socket = state.sockets[0];
  assert.equal(state.connection.sendFrame('image'), false);
  socket.onopen({});
  assert.deepEqual(socket.sent[0], { type: 'session', session_id: 'session-token' });
  socket.message({ type: 'session_ready' });
  assert.equal(state.connection.sendFrame('image'), true);
  assert.equal(state.connection.sendFrame('another'), false);
  assert.equal(state.sockets.length, 1);
});

test('backend error message releases the next frame', async t => {
  const state = setup(t);
  const socket = await ready(state);
  state.connection.sendFrame('broken');
  socket.message({ type: 'error', message: 'Inference failed' });
  assert.equal(state.connection.sendFrame('next'), true);
  assert.deepEqual(state.errors, ['Inference failed']);
});

test('successful result releases the next frame', async t => {
  const state = setup(t);
  const socket = await ready(state);
  state.connection.sendFrame('frame');
  socket.message({ type: 'vision_result', scene: [] });
  assert.equal(state.connection.sendFrame('next'), true);
  assert.equal(state.results.length, 1);
});

test('malformed message releases processing', async t => {
  const state = setup(t);
  const socket = await ready(state);
  state.connection.sendFrame('frame');
  socket.onmessage({ data: 'not JSON' });
  assert.equal(state.connection.sendFrame('next'), true);
});

test('socket closure reconnects without keeping the old processing flag', async t => {
  const state = setup(t);
  const socket = await ready(state);
  state.connection.sendFrame('frame');
  socket.close();
  await new Promise(resolve => setTimeout(resolve, 15));
  const current = state.sockets.at(-1);
  assert.notEqual(current, socket);
  current.onopen({});
  current.message({ type: 'session_ready' });
  assert.equal(state.connection.sendFrame('next'), true);
  // A late event from the closed socket must not unlock the current frame.
  socket.message({ type: 'vision_result' });
  assert.equal(state.connection.sendFrame('duplicate'), false);
});

test('expired session resets the bearer before reconnect', async t => {
  const state = setup(t);
  const socket = await ready(state);
  socket.close(4401);
  assert.equal(state.resets(), 1);
});

test('watchdog reconnects after a missing frame response', async t => {
  const state = setup(t, { frameTimeoutMs: 5 });
  const socket = await ready(state);
  state.connection.sendFrame('frame');
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(socket.closed, true);
  assert.equal(state.sockets.length, 2);
  assert.match(state.errors[0], /timed out/);
});

test('stop cancels reconnect and releases resources', async t => {
  const state = setup(t);
  const socket = await ready(state);
  state.connection.sendFrame('frame');
  state.connection.stop();
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(socket.closed, true);
  assert.equal(state.sockets.length, 1);
  assert.equal(state.connection.sendFrame('frame'), false);
});

test('stop during asynchronous authentication prevents a late connection', async t => {
  let resolve;
  const state = setup(t, { session: () => new Promise(r => { resolve = r; }) });
  state.connection.start();
  state.connection.stop();
  resolve('session-token');
  await tick();
  assert.equal(state.sockets.length, 0);
});
