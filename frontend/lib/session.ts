// Kept only in this page's memory: tabs do not share camera context or tokens.
let token: string | null = null;
let pending: Promise<string> | null = null;

export function resetSession() {
  token = null;
  pending = null;
}

export function getSession(baseUrl: string): Promise<string> {
  if (token) return Promise.resolve(token);
  if (pending) return pending;
  pending = fetch(`${baseUrl}/api/session`, { method: "POST", cache: "no-store" })
    .then(async (response) => {
      if (!response.ok) throw new Error("Could not start an EchoSight session.");
      const data = await response.json();
      if (typeof data.session_id !== "string") throw new Error("Invalid session response.");
      token = data.session_id;
      return token!;
    })
    .finally(() => { pending = null; });
  return pending;
}

export async function apiFetch(baseUrl: string, path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  headers.set("X-EchoSight-Session", await getSession(baseUrl));
  const response = await fetch(`${baseUrl}${path}`, { ...init, headers, cache: "no-store" });
  if (response.status === 401) resetSession();
  return response;
}
