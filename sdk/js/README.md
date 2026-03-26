# AirTrace RU JS SDK

Generated JavaScript SDK for the stable AirTrace RU public API v2.

Source of truth: `openapi/airtrace-v2.openapi.json`

## Install (local)

```bash
cd sdk/js
npm install
npm run build
```

## Usage example

```ts
import { AirTraceClient } from "@airtrace-ru/sdk-js";

const client = new AirTraceClient({ baseUrl: "http://localhost:8000", apiKey: "dev-key" });

const health = await client.getHealth();
const current = await client.getCurrent({ lat: 55.7558, lon: 37.6176 });
const history = await client.getHistoryByCity("moscow", "24h", 1, 50, "desc");
const trends = await client.getTrendsByCity("moscow", "7d");
const alerts = await client.listAlerts();
console.log({ health, current, history, trends, alerts });
```

## Supported endpoints

- `/v2/health`
- `/v2/current`
- `/v2/forecast`
- `/v2/history`
- `/v2/trends`
- `/v2/alerts`
