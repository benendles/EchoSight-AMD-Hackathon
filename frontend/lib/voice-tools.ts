// AssemblyAI requires tool.result after reply.done for the tool-call reply.
// HTTP analysis can finish on either side of that event; handle both orders.
// https://www.assemblyai.com/docs/voice-agents/voice-agent-api/events-reference
export class VoiceToolResults {
  private pending = new Map<string, { result?: unknown; ready: boolean; replyId?: string }>();
  private latestWasDone = false;
  private currentReplyId: string | undefined;
  private send: (message: { type: string; call_id: string; result: string }) => void;

  constructor(send: (message: { type: string; call_id: string; result: string }) => void) {
    this.send = send;
  }

  register(callId: string) {
    if (!this.pending.has(callId)) this.pending.set(callId, { ready: false, replyId: this.currentReplyId });
  }

  complete(callId: string, result: unknown) {
    const pending = this.pending.get(callId);
    if (!pending) return; // interrupted or previous session
    pending.result = result;
    this.flush();
  }

  receive(message: { type: string; status?: string; reply_id?: string }) {
    this.latestWasDone = false;
    if (message.type === "reply.started") this.currentReplyId = message.reply_id;
    if (message.type === "input.speech.started" || message.type === "session.error") {
      this.clear();
      return;
    }
    if (message.type !== "reply.done") return;
    if (message.status === "interrupted") { this.clear(); return; }
    this.latestWasDone = true;
    for (const [id, item] of this.pending) {
      // The live service also returns ordinary reply IDs for tool turns.
      // Correlate the actual reply.started ID; do not require an fc- prefix.
      if (!message.reply_id || message.reply_id === `fc-${id}` || message.reply_id === item.replyId) item.ready = true;
    }
    this.currentReplyId = undefined;
    this.flush();
  }

  private flush() {
    if (!this.latestWasDone) return;
    for (const [id, item] of this.pending) {
      if (!item.ready || item.result === undefined) continue;
      this.send({ type: "tool.result", call_id: id, result: JSON.stringify(item.result) });
      this.pending.delete(id);
    }
  }

  clear() {
    this.pending.clear();
    this.latestWasDone = false;
    this.currentReplyId = undefined;
  }
}
