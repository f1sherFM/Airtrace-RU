import { AirTraceClient } from "../sdk/js/dist/index.js";

async function main() {
  const client = new AirTraceClient({ baseUrl: "http://localhost:8000", timeoutMs: 10000, retries: 2 });
  const health = await client.getHealth();
  const current = await client.getCurrent({ lat: 55.7558, lon: 37.6176 });
  console.log(JSON.stringify({ health, current }, null, 2));
}

main().catch((error) => {
  console.error("SDK example failed:", error);
  process.exit(1);
});
