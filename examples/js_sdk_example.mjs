import { AirTraceClient } from "../sdk/js/dist/index.js";

async function main() {
  const client = new AirTraceClient({ baseUrl: "http://localhost:8000", apiKey: "dev-key", timeoutMs: 10000, retries: 2 });
  const health = await client.getHealth();
  const current = await client.getCurrent({ lat: 55.7558, lon: 37.6176 });
  const history = await client.getHistoryByCity("moscow", "24h", 1, 50, "desc");
  const trends = await client.getTrendsByCity("moscow", "7d");
  const alerts = await client.listAlerts();
  console.log(JSON.stringify({ health, current, history, trends, alerts }, null, 2));
}

main().catch((error) => {
  console.error("SDK example failed:", error);
  process.exit(1);
});
