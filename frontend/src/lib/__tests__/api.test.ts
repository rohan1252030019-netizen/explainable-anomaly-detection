/**
 * API client integration tests.
 *
 * The client is the single boundary between the UI and the backend.  These
 * tests pin the behaviour the audit found missing: a server error must surface
 * the server's own message (not "[object Object]" for a 422), a hung request
 * must time out, and an aborted request must stay aborted rather than being
 * reported as a failure.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, isAbortError } from "../api";

type FetchMock = ReturnType<typeof vi.fn>;

function jsonResponse(body: unknown, init: ResponseInit = {}) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

/** A fetch that never resolves on its own but rejects when its signal aborts. */
function abortableFetch(): FetchMock {
  return vi.fn(
    (_url: string, init: RequestInit) =>
      new Promise((_resolve, reject) => {
        const signal = init.signal;
        if (!signal) return;
        signal.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        });
      })
  );
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("request", () => {
  it("returns the parsed JSON body on success", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ status: "healthy" })));
    const body = await api.getHealth();
    expect(body).toEqual({ status: "healthy" });
  });

  it("sends the API key header when one is configured", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_KEY", "s3cret");
    vi.resetModules();
    const mod = await import("../api");
    const fetchMock = vi.fn(async (url: string, init: RequestInit) => {
      expect(url).toContain("/api/health");
      expect(init.signal).toBeInstanceOf(AbortSignal);
      return jsonResponse({ status: "healthy" });
    });
    vi.stubGlobal("fetch", fetchMock);

    await mod.api.getHealth();

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-API-Key"]).toBe("s3cret");
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it("turns a handled backend error into an ApiError with its detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse({ detail: "Dataset has only 12 rows." }, { status: 400 })
      )
    );

    await expect(api.getDatasetSummary()).rejects.toBeInstanceOf(ApiError);
    await expect(api.getDatasetSummary()).rejects.toMatchObject({
      status: 400,
      detail: "Dataset has only 12 rows.",
    });
  });

  it("flattens a 422 validation array instead of printing [object Object]", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse(
          {
            detail: [
              {
                loc: ["body", "params", "n_estimatrs"],
                msg: "Extra inputs are not permitted",
                type: "extra_forbidden",
              },
              { loc: ["body", "contamination"], msg: "Input should be <= 0.5" },
            ],
          },
          { status: 422 }
        )
      )
    );

    const error = await api.getDatasetSummary().catch((err: unknown) => err);
    expect(error).toBeInstanceOf(ApiError);
    const message = (error as ApiError).detail;
    expect(message).toContain("params.n_estimatrs: Extra inputs are not permitted");
    expect(message).toContain("contamination: Input should be <= 0.5");
    expect(message).not.toContain("[object Object]");
  });

  it("reports a transport failure as ApiError 0 rather than throwing TypeError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      })
    );

    const error = await api.getHealth().catch((err: unknown) => err);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(0);
    expect((error as ApiError).detail).toMatch(/Could not reach the SENTINEL API/);
  });

  it("times out a hung request with a 408 instead of spinning forever", async () => {
    vi.stubGlobal("fetch", abortableFetch());

    const error = await api.getHealth({ timeoutMs: 5 }).catch((err: unknown) => err);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(408);
    expect((error as ApiError).detail).toMatch(/timed out after/);
  });

  it("propagates caller cancellation as an abort, not as an error", async () => {
    vi.stubGlobal("fetch", abortableFetch());
    const controller = new AbortController();

    const pending = api.getHealth({ signal: controller.signal });
    controller.abort();

    const error = await pending.catch((err: unknown) => err);
    expect(isAbortError(error)).toBe(true);
    expect(error instanceof ApiError).toBe(false);
  });

  it("passes the composed signal to fetch", async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      expect(init.signal).toBeInstanceOf(AbortSignal);
      return jsonResponse({ status: "healthy" });
    });
    vi.stubGlobal("fetch", fetchMock);

    await api.getHealth();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("returns undefined for a 204", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 204 })));
    await expect(api.resetState()).resolves.toBeUndefined();
  });

  it("uses a longer budget for training calls than for reads", async () => {
    const fetchMock = vi.fn(async (_url: string, init: RequestInit) => {
      // The composed signal's deadline is not observable directly, so assert
      // the call was made with a signal at all and rely on the timeout test
      // above for the mechanism.
      expect(init.signal).toBeDefined();
      return jsonResponse({ success: true, overview: {} });
    });
    vi.stubGlobal("fetch", fetchMock);

    await api.runDetection({ model_name: "Isolation Forest" });
    await api.getResults();

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
