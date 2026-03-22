/**
 * CogniTwin Capture SDK — Reference Client
 *
 * Buffers events locally, flushes to the CogniTwin ingest endpoint
 * with retry + back-pressure. Designed for both browser and Node
 * environments (uses the Fetch API).
 *
 * Usage:
 *   const sdk = new CogniTwinClient({ ingestUrl: "https://…/ingest", sessionId: "…" });
 *   sdk.capture(makeKeystrokeEvent(…));
 *   await sdk.flush();
 */

import type { CogniTwinEvent } from "./schema.js";

export interface CogniTwinClientOptions {
  /** Base URL of the CogniTwin ingest endpoint */
  ingestUrl: string;
  /** Session identifier (shared across all events from this client instance) */
  sessionId: string;
  /**
   * Maximum number of events to hold in the local buffer before auto-flush.
   * Default: 50
   */
  bufferSize?: number;
  /**
   * Auto-flush interval in milliseconds (0 = disabled).
   * Default: 5000 ms
   */
  flushIntervalMs?: number;
  /** Bearer token for authenticated ingest (optional) */
  authToken?: string;
  /** Called on unrecoverable send failure */
  onError?: (err: Error, events: CogniTwinEvent[]) => void;
}

const DEFAULT_BUFFER_SIZE = 50;
const DEFAULT_FLUSH_INTERVAL_MS = 5_000;
const MAX_RETRIES = 3;
const RETRY_BASE_MS = 500;

export class CogniTwinClient {
  private readonly opts: Required<
    Omit<CogniTwinClientOptions, "authToken" | "onError">
  > &
    Pick<CogniTwinClientOptions, "authToken" | "onError">;
  private buffer: CogniTwinEvent[] = [];
  private seq = 0;
  private flushTimer: ReturnType<typeof setInterval> | null = null;
  private flushing = false;

  constructor(opts: CogniTwinClientOptions) {
    this.opts = {
      ingestUrl: opts.ingestUrl,
      sessionId: opts.sessionId,
      bufferSize: opts.bufferSize ?? DEFAULT_BUFFER_SIZE,
      flushIntervalMs: opts.flushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS,
      authToken: opts.authToken,
      onError: opts.onError,
    };

    if (this.opts.flushIntervalMs > 0) {
      this.flushTimer = setInterval(
        () => void this.flush(),
        this.opts.flushIntervalMs
      );
    }
  }

  /** Enqueue a single event. Auto-assigns monotonic seq if seq === 0. */
  capture(event: CogniTwinEvent): void {
    // Stamp seq on the event if the caller left it at 0
    if (event.seq === 0) {
      (event as CogniTwinEvent & { seq: number }).seq = ++this.seq;
    } else {
      this.seq = Math.max(this.seq, event.seq);
    }
    this.buffer.push(event);
    if (this.buffer.length >= this.opts.bufferSize) {
      void this.flush();
    }
  }

  /** Flush all buffered events to the ingest endpoint. */
  async flush(): Promise<void> {
    if (this.flushing || this.buffer.length === 0) return;
    this.flushing = true;
    const batch = this.buffer.splice(0, this.buffer.length);
    try {
      await this.sendWithRetry(batch);
    } catch (err) {
      // Put events back at the front of the buffer for next attempt
      this.buffer.unshift(...batch);
      this.opts.onError?.(err as Error, batch);
    } finally {
      this.flushing = false;
    }
  }

  /** Stop the auto-flush timer and drain the buffer. */
  async dispose(): Promise<void> {
    if (this.flushTimer !== null) {
      clearInterval(this.flushTimer);
      this.flushTimer = null;
    }
    await this.flush();
  }

  // ── Private ──────────────────────────────────────────────────────────────

  private async sendWithRetry(events: CogniTwinEvent[]): Promise<void> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "X-CogniTwin-Schema": "0.1.0",
    };
    if (this.opts.authToken) {
      headers["Authorization"] = `Bearer ${this.opts.authToken}`;
    }

    let lastErr: Error | undefined;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      if (attempt > 0) {
        await sleep(RETRY_BASE_MS * 2 ** (attempt - 1));
      }
      try {
        const res = await fetch(this.opts.ingestUrl, {
          method: "POST",
          headers,
          body: JSON.stringify({ events }),
        });
        if (res.ok) return;
        // 4xx = do not retry (client error)
        if (res.status >= 400 && res.status < 500) {
          throw new Error(`Ingest rejected (${res.status}): ${await res.text()}`);
        }
        lastErr = new Error(`Ingest HTTP ${res.status}`);
      } catch (err) {
        if ((err as Error).message.startsWith("Ingest rejected")) throw err;
        lastErr = err as Error;
      }
    }
    throw lastErr ?? new Error("Unknown ingest failure");
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
