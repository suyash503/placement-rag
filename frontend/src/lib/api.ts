import type { Citation, RetrievalTrace, Stats } from "./types";

export interface StreamHandlers {
  onMeta: (citations: Citation[], trace: RetrievalTrace) => void;
  onToken: (value: string) => void;
  onDone: (latencyMs: number) => void;
  onError: (message: string) => void;
}

interface HistoryTurn {
  role: "user" | "assistant";
  content: string;
}

/**
 * EventSource is GET-only, so the stream is read off a POST response body and the
 * SSE frames are parsed by hand.
 */
export async function streamChat(
  question: string,
  history: HistoryTurn[],
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, mode: "hybrid_rerank", explain: true }),
      signal,
    });
  } catch (err) {
    handlers.onError(
      err instanceof DOMException && err.name === "AbortError"
        ? "Cancelled."
        : "Could not reach the API. Is the backend running on port 8000?",
    );
    return;
  }

  if (!response.ok || !response.body) {
    handlers.onError(`API returned ${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (raw: string) => {
    const dataLines = raw
      .split("\n")
      .filter((l) => l.startsWith("data:"))
      .map((l) => l.slice(5).trim());
    if (!dataLines.length) return;

    let payload: any;
    try {
      payload = JSON.parse(dataLines.join(""));
    } catch {
      return;
    }

    switch (payload.type) {
      case "meta":
        handlers.onMeta(payload.citations ?? [], payload.trace ?? {});
        break;
      case "token":
        handlers.onToken(payload.value ?? "");
        break;
      case "done":
        handlers.onDone(payload.latency_ms ?? 0);
        break;
      case "error":
        handlers.onError(payload.message ?? "Unknown error");
        break;
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      // sse-starlette separates lines with CRLF, so frames end in "\r\n\r\n".
      // Normalising first keeps one boundary check working for either style.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        dispatch(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
    }
    if (buffer.trim()) dispatch(buffer);
  } catch (err) {
    if (!(err instanceof DOMException && err.name === "AbortError")) {
      handlers.onError("Stream interrupted.");
    }
  }
}

export async function fetchStats(): Promise<Stats | null> {
  try {
    const res = await fetch("/api/stats");
    return res.ok ? ((await res.json()) as Stats) : null;
  } catch {
    return null;
  }
}
