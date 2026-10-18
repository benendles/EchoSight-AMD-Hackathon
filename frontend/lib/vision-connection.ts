type SocketLike = Pick<WebSocket, "readyState" | "send" | "close" | "onopen" | "onmessage" | "onerror" | "onclose">;
type Options = {
  url: string;
  session: () => Promise<string>;
  resetSession: () => void;
  onConnected: (connected: boolean) => void;
  onResult: (result: Record<string, unknown>) => void;
  onError: (message: string) => void;
  socket?: (url: string) => SocketLike;
  frameTimeoutMs?: number;
  reconnectMs?: number;
};

// Owns the complete socket lifecycle, including failures that are valid JSON.
// An obsolete socket can never reset the current socket's processing state.
export class VisionConnection {
  private options: Options;
  private socket: SocketLike | null = null;
  private active = false;
  private ready = false;
  private busy = false;
  private generation = 0;
  private retry = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private frameTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(options: Options) { this.options = options; }

  start() {
    if (this.active) return;
    this.active = true;
    void this.connect();
  }

  stop() {
    this.active = false;
    this.generation++;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.releaseFrame();
    const socket = this.socket;
    this.socket = null;
    this.ready = false;
    socket?.close();
    this.options.onConnected(false);
  }

  private releaseFrame() {
    this.busy = false;
    if (this.frameTimer) clearTimeout(this.frameTimer);
    this.frameTimer = null;
  }

  private scheduleReconnect() {
    if (!this.active || this.reconnectTimer) return;
    const delay = Math.min(10000, (this.options.reconnectMs ?? 500) * 2 ** this.retry++);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      void this.connect();
    }, delay);
  }

  private async connect() {
    if (!this.active || this.socket) return;
    const generation = ++this.generation;
    try {
      const session = await this.options.session();
      if (!this.active || generation !== this.generation) return;
      const socket = this.options.socket?.(this.options.url) ?? new WebSocket(this.options.url);
      this.socket = socket;
      const current = () => this.active && this.socket === socket && generation === this.generation;
      socket.onopen = () => {
        if (!current()) return;
        socket.send(JSON.stringify({ type: "session", session_id: session }));
      };
      this.frameTimer = setTimeout(() => {
        if (current() && !this.ready) socket.close();
      }, 10000);
      socket.onmessage = (event) => {
        if (!current()) return;
        try {
          const message = JSON.parse(event.data);
          if (!message || typeof message !== "object") throw new Error();
          if (message.type === "session_ready") {
            this.releaseFrame();
            this.ready = true;
            this.retry = 0;
            this.options.onConnected(true);
          } else if (message.type === "vision_result") {
            this.releaseFrame();
            this.options.onResult(message);
          } else if (message.type === "error") {
            this.releaseFrame();
            this.options.onError(typeof message.message === "string" ? message.message : "Vision analysis failed. Retrying the next frame.");
          }
        } catch {
          this.releaseFrame();
          this.options.onError("Could not read the vision result. Retrying the next frame.");
        }
      };
      socket.onerror = () => {
        if (!current()) return;
        this.releaseFrame();
        this.ready = false;
        this.options.onConnected(false);
        this.options.onError("Vision connection interrupted. Reconnecting...");
        socket.close();
      };
      socket.onclose = (event) => {
        if (!current()) return;
        this.releaseFrame();
        this.ready = false;
        this.socket = null;
        this.options.onConnected(false);
        if (event.code === 4401) this.options.resetSession();
        this.scheduleReconnect();
      };
    } catch {
      if (!this.active || generation !== this.generation) return;
      this.options.onError("Could not connect to local vision. Reconnecting...");
      this.scheduleReconnect();
    }
  }

  sendFrame(image: string): boolean {
    const socket = this.socket;
    if (!this.active || !this.ready || this.busy || !socket || socket.readyState !== 1) return false;
    this.busy = true;
    try {
      socket.send(JSON.stringify({ type: "frame", image }));
      this.frameTimer = setTimeout(() => {
        this.releaseFrame();
        this.ready = false;
        this.options.onError("Vision response timed out. Reconnecting...");
        socket.close();
      }, this.options.frameTimeoutMs ?? 45000);
      return true;
    } catch {
      this.releaseFrame();
      socket.close();
      return false;
    }
  }
}
