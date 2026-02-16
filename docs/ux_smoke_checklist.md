# UX Smoke Checklist (Issue #14)

Goal: verify the city dashboard clearly separates `Now / Forecast / History` and works on desktop/mobile.

## Desktop Smoke

- Open city page (`/city/{city}`) on width >= 1280 px.
- Verify "Режимы данных" block is visible at top with 3 cards: `Now`, `Forecast`, `History`.
- Click each card and confirm page jumps to matching section anchors:
  - `#now-section`
  - `#forecast-section`
  - `#history-section`
- Confirm section semantics are explicit:
  - Now = current AQI
  - Forecast = predicted timeline
  - History = factual `/history` observations
- Trigger forecast period switch (24h/48h/72h/week) and verify active button changes.
- Trigger history period switch (24h/7d/30d) and verify active button changes.

## Mobile Smoke

- Open city page in device emulation width 390 px (or similar).
- Verify top "Режимы данных" cards stack vertically and remain readable.
- Verify major cards fit screen without horizontal overflow.
- Tap mode cards and confirm anchor navigation remains usable.
- Verify export/status controls remain reachable and readable.

## Data Integrity Smoke

- Confirm history entries show provenance (`data_source`, `freshness`, `confidence`) and anomaly badge when present.
- Confirm forecast blocks are labeled as forecast (not history).
- Confirm no section mixes semantics between forecast/history.
