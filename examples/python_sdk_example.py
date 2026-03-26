from airtrace_sdk import AirTraceClient


def main() -> None:
    with AirTraceClient(base_url="http://localhost:8000", retries=2, retry_delay=0.2) as client:
        health = client.get_health()
        current = client.get_current(lat=55.7558, lon=37.6176)
        history = client.get_history_by_city(city="moscow", sort="desc")
        trends = client.get_trends_by_city(city="moscow", range="7d")
        alerts = client.list_alerts()
        print({"health": health, "current": current, "history_total": history.get("total"), "trends": trends.get("trend"), "alerts": len(alerts)})


if __name__ == "__main__":
    main()
