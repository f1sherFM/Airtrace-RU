export type Coordinates = {
  lat: number;
  lon: number;
};

export type AirTraceClientOptions = {
  baseUrl?: string;
  timeoutMs?: number;
  retries?: number;
};

export type AirTraceErrorPayload = {
  detail?: string;
  error?: string;
  [key: string]: unknown;
};

export class AirTraceError extends Error {
  status: number;
  payload?: AirTraceErrorPayload;

  constructor(message: string, status: number, payload?: AirTraceErrorPayload) {
    super(message);
    this.name = "AirTraceError";
    this.status = status;
    this.payload = payload;
  }
}

type RequestInitWithTimeout = RequestInit & { timeoutMs?: number };

async function fetchWithTimeout(url: string, init: RequestInitWithTimeout): Promise<Response> {
  const controller = new AbortController();
  const timeoutMs = init.timeoutMs ?? 10_000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export class AirTraceClient {
  private baseUrl: string;
  private timeoutMs: number;
  private retries: number;

  constructor(options: AirTraceClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? "http://localhost:8000";
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.retries = options.retries ?? 2;
  }

  private async request<T>(path: string, params: Record<string, string>): Promise<T> {
    const url = new URL(`${this.baseUrl}${path}`);
    Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));

    let lastError: unknown;
    for (let attempt = 0; attempt <= this.retries; attempt++) {
      try {
        const response = await fetchWithTimeout(url.toString(), { method: "GET", timeoutMs: this.timeoutMs });
        if (!response.ok) {
          let payload: AirTraceErrorPayload | undefined;
          try {
            payload = (await response.json()) as AirTraceErrorPayload;
          } catch {
            payload = undefined;
          }
          throw new AirTraceError(`HTTP ${response.status} for ${path}`, response.status, payload);
        }
        return (await response.json()) as T;
      } catch (error) {
        lastError = error;
        if (attempt === this.retries) {
          throw error;
        }
      }
    }
    throw lastError;
  }

  getCurrent(coords: Coordinates): Promise<unknown> {
    return this.request("/v2/current", {
      lat: String(coords.lat),
      lon: String(coords.lon),
    });
  }

  getForecast(coords: Coordinates): Promise<unknown> {
    return this.request("/v2/forecast", {
      lat: String(coords.lat),
      lon: String(coords.lon),
    });
  }

  getHistoryByCity(city: string, range = "24h", page = 1, pageSize = 20): Promise<unknown> {
    return this.request("/v2/history", {
      city,
      range,
      page: String(page),
      page_size: String(pageSize),
    });
  }

  getHealth(): Promise<unknown> {
    return this.request("/v2/health", {});
  }
}
