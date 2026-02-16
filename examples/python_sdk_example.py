from airtrace_sdk import AirTraceClient


def main() -> None:
    with AirTraceClient(base_url="http://localhost:8000", retries=2, retry_delay=0.2) as client:
        health = client.get_health()
        current = client.get_current(lat=55.7558, lon=37.6176)
        print({"health": health, "current": current})


if __name__ == "__main__":
    main()
