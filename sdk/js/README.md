# AirTrace RU JS SDK (starter)

Minimal typed JS SDK starter for AirTrace RU API v2.

## Install (local starter)

```bash
cd sdk/js
npm install
npm run build
```

## Usage example

```ts
import { AirTraceClient } from "@airtrace-ru/sdk-js";

const client = new AirTraceClient({ baseUrl: "http://localhost:8000" });

const health = await client.getHealth();
const now = await client.getCurrent({ lat: 55.7558, lon: 37.6176 });
console.log({ health, now });
```

## Versioning

Current starter version: `0.1.0` (see `package.json`).
