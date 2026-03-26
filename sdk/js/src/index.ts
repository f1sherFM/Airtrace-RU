export type Coordinates = {
  lat: number;
  lon: number;
};

export type AirTraceClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  retries?: number;
};

export type AirTraceErrorPayload = {
  code?: string;
  message?: string;
  details?: unknown;
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
  private apiKey?: string;
  private timeoutMs: number;
  private retries: number;

  constructor(options: AirTraceClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? "http://localhost:8000";
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.retries = options.retries ?? 2;
  }

  private async request<T>(
    method: string,
    path: string,
    params: Record<string, string>,
    body?: unknown,
    idempotencyKey?: string,
  ): Promise<T> {
    const url = new URL(`${this.baseUrl}${path}`);
    Object.entries(params).forEach(([key, value]) => url.searchParams.set(key, value));

    let lastError: unknown;
    for (let attempt = 0; attempt <= this.retries; attempt++) {
      try {
        const headers: Record<string, string> = {};
        if (this.apiKey) headers["X-API-Key"] = this.apiKey;
        if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
        if (body !== undefined) headers["Content-Type"] = "application/json";
        const response = await fetchWithTimeout(url.toString(), {
          method,
          timeoutMs: this.timeoutMs,
          headers,
          body: body === undefined ? undefined : JSON.stringify(body),
        });
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

  getHealth(): Promise<unknown> {
    return this.request("GET", "/v2/health", {});
  }

  getCurrent(coords: Coordinates): Promise<unknown> {
    return this.request("GET", "/v2/current", {
      lat: String(coords.lat),
      lon: String(coords.lon),
    });
  }

  getForecast(coords: Coordinates, hours = 24): Promise<unknown> {
    return this.request("GET", "/v2/forecast", {
      lat: String(coords.lat),
      lon: String(coords.lon),
      hours: String(hours),
    });
  }

  getHistory(options: {
    range?: string;
    page?: number;
    pageSize?: number;
    sort?: "asc" | "desc";
    city?: string;
    lat?: number;
    lon?: number;
  } = {}): Promise<unknown> {
    const params: Record<string, string> = {
      range: options.range ?? "24h",
      page: String(options.page ?? 1),
      page_size: String(options.pageSize ?? 50),
      sort: options.sort ?? "desc",
    };
    if (options.city) params.city = options.city;
    if (options.lat !== undefined && options.lon !== undefined) {
      params.lat = String(options.lat);
      params.lon = String(options.lon);
    }
    return this.request("GET", "/v2/history", params);
  }

  getHistoryByCity(city: string, range = "24h", page = 1, pageSize = 50, sort: "asc" | "desc" = "desc"): Promise<unknown> {
    return this.getHistory({ city, range, page, pageSize, sort });
  }

  getTrends(options: { range?: string; city?: string; lat?: number; lon?: number } = {}): Promise<unknown> {
    const params: Record<string, string> = {
      range: options.range ?? "7d",
    };
    if (options.city) params.city = options.city;
    if (options.lat !== undefined && options.lon !== undefined) {
      params.lat = String(options.lat);
      params.lon = String(options.lon);
    }
    return this.request("GET", "/v2/trends", params);
  }

  getTrendsByCity(city: string, range = "7d"): Promise<unknown> {
    return this.getTrends({ city, range });
  }

  listAlerts(): Promise<unknown> {
    return this.request("GET", "/v2/alerts", {});
  }

  getAlert(subscriptionId: string): Promise<unknown> {
    return this.request("GET", `/v2/alerts/${subscriptionId}`, {});
  }

  createAlert(payload: Record<string, unknown>, idempotencyKey?: string): Promise<unknown> {
    return this.request("POST", "/v2/alerts", {}, payload, idempotencyKey);
  }

  updateAlert(subscriptionId: string, payload: Record<string, unknown>, idempotencyKey?: string): Promise<unknown> {
    return this.request("PATCH", `/v2/alerts/${subscriptionId}`, {}, payload, idempotencyKey);
  }

  deleteAlert(subscriptionId: string): Promise<unknown> {
    return this.request("DELETE", `/v2/alerts/${subscriptionId}`, {});
  }
}
