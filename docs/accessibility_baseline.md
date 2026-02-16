# Accessibility Baseline (Issue #16)

This checklist documents the baseline A11y guarantees for the web UI.

## Implemented Baseline

- Semantic landmarks in base layout:
  - `role="banner"` on header
  - `role="main"` on main content container
  - `role="contentinfo"` on footer
- Skip link for keyboard users:
  - top-level link to `#main-content`
- Visible keyboard focus:
  - `:focus-visible` high-contrast outline in `base.html`
- Keyboard-friendly controls:
  - period/range controls are `<button>` with `aria-pressed`
- Dynamic region announcements:
  - notification live-region (`aria-live="polite"`)
  - loading state markers (`aria-busy`)

## Validation

- Template baseline tests:
  - `tests/test_web_accessibility_baseline.py`
